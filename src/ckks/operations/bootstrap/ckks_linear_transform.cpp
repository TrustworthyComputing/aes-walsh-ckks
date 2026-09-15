#include "ckks_linear_transform.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "ckks_addition.hpp"
#include "ckks_context.hpp"
#include "ckks_keyswitch.hpp"
#include "ckks_mul.hpp"
#include "ckks_rescale.hpp"
#include "ckks_rotation.hpp"
#include "modarith.hpp"

namespace {

enum class PlaintextAccumulationMode {
    Assign,
    AddMod,
    AddRaw,
};

size_t stage_diagonal_index(
    const CKKSLinearTransformStage& stage,
    size_t term_index) {
    return stage.diagonal_indices.empty()
        ? term_index
        : stage.diagonal_indices[term_index];
}

CKKSLinearTransformStage make_stage_view(
    const CKKSLinearTransformOwnedStage& owned_stage) {
    CKKSLinearTransformStage stage;
    stage.diagonals = std::span<const CKKSEncoding>(
        owned_stage.diagonals.data(),
        owned_stage.diagonals.size());
    stage.diagonal_indices = std::span<const size_t>(
        owned_stage.diagonal_indices.data(),
        owned_stage.diagonal_indices.size());
    stage.slot_count = owned_stage.slot_count;
    stage.baby_step_count = owned_stage.baby_step_count;
    stage.rescale_after = owned_stage.rescale_after;
    stage.output_encoding_state = owned_stage.output_encoding_state;
    return stage;
}

size_t linear_transform_stage_slot_count(
    const CKKSParams& params,
    const CKKSLinearTransformStage& stage) {
    const size_t full_slots = params.getSlots();
    const size_t slot_count = stage.slot_count == 0 ? full_slots : stage.slot_count;
    if (slot_count == 0 ||
        slot_count > full_slots ||
        full_slots % slot_count != 0 ||
        !std::has_single_bit(slot_count)) {
        throw std::invalid_argument(
            "ckks_linear_transform: invalid logical stage slot count");
    }
    return slot_count;
}

bool should_use_qp_linear_transform(
    const CKKSLinearTransformOwnedStage& owned_stage,
    bool plan_use_qp_evaluator) {
    if (owned_stage.diagonal_qp_plaintexts.empty()) {
        return false;
    }
    return plan_use_qp_evaluator;
}

void ensure_linear_transform_rotation_keys(
    const CKKSContext& context,
    const CKKSLinearTransformStage& stage,
    size_t slot_count) {
    const size_t baby_step_count = stage.baby_step_count;
    if (stage.diagonal_indices.empty()) {
        const size_t diagonal_count = stage.diagonals.size();
        const size_t baby_limit = std::min(diagonal_count, baby_step_count);
        for (size_t rotation = 1; rotation < baby_limit; ++rotation) {
            if (!context.hasRotationKey(static_cast<int64_t>(rotation))) {
                throw std::invalid_argument("ckks_linear_transform: missing baby-step rotation key");
            }
        }
        for (size_t rotation = baby_step_count; rotation < diagonal_count; rotation += baby_step_count) {
            if (!context.hasRotationKey(static_cast<int64_t>(rotation))) {
                throw std::invalid_argument("ckks_linear_transform: missing giant-step rotation key");
            }
        }
        return;
    }

    MyVector<char> baby_needed(baby_step_count, char{0});
    MyVector<char> giant_needed((slot_count + baby_step_count - 1) / baby_step_count, char{0});
    for (size_t term = 0; term < stage.diagonal_indices.size(); ++term) {
        const size_t diagonal_index = stage.diagonal_indices[term];
        const size_t baby_rotation = diagonal_index % baby_step_count;
        const size_t giant_index = diagonal_index / baby_step_count;
        baby_needed[baby_rotation] = char{1};
        giant_needed[giant_index] = char{1};
    }

    for (size_t rotation = 1; rotation < baby_needed.size(); ++rotation) {
        if (baby_needed[rotation] && !context.hasRotationKey(static_cast<int64_t>(rotation))) {
            throw std::invalid_argument("ckks_linear_transform: missing baby-step rotation key");
        }
    }
    for (size_t giant_index = 1; giant_index < giant_needed.size(); ++giant_index) {
        const size_t rotation = giant_index * baby_step_count;
        if (giant_needed[giant_index] && !context.hasRotationKey(static_cast<int64_t>(rotation))) {
            throw std::invalid_argument("ckks_linear_transform: missing giant-step rotation key");
        }
    }
}

void multiply_montgomery_plaintext_accumulate_ciphertext_pair_limb(
    const uint64_t* __restrict a_ptr,
    const uint64_t* __restrict b_ptr,
    const uint64_t* __restrict pt_ptr,
    uint64_t* __restrict acc_a_ptr,
    uint64_t* __restrict acc_b_ptr,
    size_t N,
    uint64_t q,
    uint64_t q_neg_inv,
    PlaintextAccumulationMode mode) {
    for (size_t i = 0; i < N; ++i) {
        const uint64_t term_a = ckks_montgomery_mul_normal_by_montgomery(
            a_ptr[i],
            pt_ptr[i],
            q,
            q_neg_inv);
        const uint64_t term_b = ckks_montgomery_mul_normal_by_montgomery(
            b_ptr[i],
            pt_ptr[i],
            q,
            q_neg_inv);
        if (mode == PlaintextAccumulationMode::Assign) {
            acc_a_ptr[i] = term_a;
            acc_b_ptr[i] = term_b;
        } else if (mode == PlaintextAccumulationMode::AddRaw) {
            acc_a_ptr[i] += term_a;
            acc_b_ptr[i] += term_b;
        } else {
            acc_a_ptr[i] = add_mod_q(acc_a_ptr[i], term_a, q);
            acc_b_ptr[i] = add_mod_q(acc_b_ptr[i], term_b, q);
        }
    }
}

void multiply_plaintext_accumulate_ciphertext_pair(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    const CKKSEncoding& plaintext,
    const MyVector<uint64_t>& montgomery_neg_inv,
    MyVector<uint64_t>& acc_a,
    MyVector<uint64_t>& acc_b,
    PlaintextAccumulationMode mode) {
    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const size_t N = params.getN();
    const size_t level = ct.getLevel();
    const size_t limb_count = level + 1;
    const size_t flat_size = N * limb_count;

    if (ct.getN() != N || plaintext.getN() != N) {
        throw std::invalid_argument("ckks_linear_transform: ciphertext/plaintext size mismatch");
    }
    if (ct.getNumPolys() != 2) {
        throw std::invalid_argument("ckks_linear_transform: expected a 2-polynomial ciphertext");
    }
    if (plaintext.getLevel() != level) {
        throw std::invalid_argument("ckks_linear_transform: ciphertext/plaintext level mismatch");
    }
    if (level > params.getMaxLevel() ||
        limb_count > moduli.size() ||
        limb_count > twiddle_ntt.size() ||
        montgomery_neg_inv.size() < limb_count) {
        throw std::invalid_argument("ckks_linear_transform: invalid plaintext multiplication metadata");
    }

    const auto& pt = plaintext.getPlaintext();
    const auto& a = ct.getA();
    const auto& b = ct.getB();
    if (pt.size() != flat_size || a.size() != flat_size || b.size() != flat_size) {
        throw std::invalid_argument("ckks_linear_transform: invalid plaintext multiplication buffer size");
    }
    if (mode == PlaintextAccumulationMode::Assign) {
        acc_a.resize(flat_size);
        acc_b.resize(flat_size);
    } else if (acc_a.size() != flat_size || acc_b.size() != flat_size) {
        throw std::invalid_argument("ckks_linear_transform: invalid accumulation buffer size");
    }

    for (size_t limb = 0; limb < limb_count; ++limb) {
        const uint64_t q = moduli[limb];
        const auto& tbl = twiddle_ntt[limb];
        const uint64_t* __restrict a_ptr = a.data() + limb * N;
        const uint64_t* __restrict b_ptr = b.data() + limb * N;
        const uint64_t* __restrict pt_ptr = pt.data() + limb * N;
        uint64_t* __restrict acc_a_ptr = acc_a.data() + limb * N;
        uint64_t* __restrict acc_b_ptr = acc_b.data() + limb * N;

        if (plaintext.isMontgomeryForm()) {
            const uint64_t q_neg_inv = montgomery_neg_inv[limb];
            multiply_montgomery_plaintext_accumulate_ciphertext_pair_limb(
                a_ptr,
                b_ptr,
                pt_ptr,
                acc_a_ptr,
                acc_b_ptr,
                N,
                q,
                q_neg_inv,
                mode);
        } else {
            const auto* barrett_const = &tbl.barrett_const;
            for (size_t i = 0; i < N; ++i) {
                const uint64_t term_a = mul_mod_u64(a_ptr[i], pt_ptr[i], q, barrett_const);
                const uint64_t term_b = mul_mod_u64(b_ptr[i], pt_ptr[i], q, barrett_const);
                if (mode == PlaintextAccumulationMode::Assign) {
                    acc_a_ptr[i] = term_a;
                    acc_b_ptr[i] = term_b;
                } else if (mode == PlaintextAccumulationMode::AddRaw) {
                    acc_a_ptr[i] += term_a;
                    acc_b_ptr[i] += term_b;
                } else {
                    acc_a_ptr[i] = add_mod_q(acc_a_ptr[i], term_a, q);
                    acc_b_ptr[i] = add_mod_q(acc_b_ptr[i], term_b, q);
                }
            }
        }
    }
}

size_t lazy_extra_term_limit_for_linear_transform(
    const MyVector<uint64_t>& moduli,
    size_t limb_count) {
    size_t limit = std::numeric_limits<size_t>::max();
    const auto max_u64 = static_cast<__uint128_t>(std::numeric_limits<uint64_t>::max());
    for (size_t limb = 0; limb < limb_count; ++limb) {
        const uint64_t q = moduli[limb];
        if (q <= 1) {
            return 0;
        }
        const auto max_terms = max_u64 / static_cast<__uint128_t>(q - 1);
        if (max_terms <= 1) {
            return 0;
        }
        const auto extra_terms = max_terms - 1;
        if (extra_terms < limit) {
            limit = static_cast<size_t>(extra_terms);
        }
    }
    return limit == std::numeric_limits<size_t>::max() ? 0 : limit;
}

void reduce_lazy_accumulation_pair_rns_inplace(
    MyVector<uint64_t>& acc_a,
    MyVector<uint64_t>& acc_b,
    size_t N,
    const MyVector<uint64_t>& moduli,
    size_t limb_count,
    size_t term_count) {
    if (term_count <= 1) {
        return;
    }
    if (acc_a.size() != N * limb_count || acc_b.size() != N * limb_count) {
        throw std::invalid_argument("ckks_linear_transform: invalid lazy accumulation buffer size");
    }

    size_t highest_power = 1;
    const size_t max_quotient = term_count - 1;
    while (highest_power <= max_quotient / 2) {
        highest_power <<= 1;
    }

    MyVector<uint64_t> multiples;
    const auto max_u64 = static_cast<__uint128_t>(std::numeric_limits<uint64_t>::max());
    for (size_t limb = 0; limb < limb_count; ++limb) {
        const uint64_t q = moduli[limb];
        multiples.clear();
        for (size_t step = highest_power; step > 0; step >>= 1) {
            const auto multiple = static_cast<__uint128_t>(q) * static_cast<__uint128_t>(step);
            if (multiple <= max_u64) {
                multiples.emplace_back(static_cast<uint64_t>(multiple));
            }
        }

        uint64_t* __restrict acc_a_ptr = acc_a.data() + limb * N;
        uint64_t* __restrict acc_b_ptr = acc_b.data() + limb * N;
        for (size_t i = 0; i < N; ++i) {
            uint64_t value_a = acc_a_ptr[i];
            uint64_t value_b = acc_b_ptr[i];
            for (const uint64_t multiple : multiples) {
                if (value_a >= multiple) {
                    value_a -= multiple;
                }
                if (value_b >= multiple) {
                    value_b -= multiple;
                }
            }
            acc_a_ptr[i] = value_a;
            acc_b_ptr[i] = value_b;
        }
    }
}

MyVector<uint64_t> active_q_moduli_for_level(const CKKSContext& context, size_t level) {
    const auto& moduli = context.getParams().getModuli();
    if (level >= moduli.size()) {
        throw std::invalid_argument("ckks_linear_transform: invalid active Q level");
    }
    return MyVector<uint64_t>(moduli.begin(), moduli.begin() + level + 1);
}

MyVector<uint64_t> active_qp_moduli_for_level(const CKKSContext& context, size_t level) {
    auto out = active_q_moduli_for_level(context, level);
    const auto& p_moduli = context.getParams().getPModuli();
    out.insert(out.end(), p_moduli.begin(), p_moduli.end());
    return out;
}

MyVector<Negacyclic_NTT_Twiddles> active_qp_twiddles_for_level(
    const CKKSContext& context,
    size_t level) {
    const auto& q_twiddles = context.getTwiddleNtt();
    const auto& p_twiddles = context.getPTwiddleNtt();
    if (level >= q_twiddles.size()) {
        throw std::invalid_argument("ckks_linear_transform: invalid active Q twiddle level");
    }

    MyVector<Negacyclic_NTT_Twiddles> out;
    out.reserve(level + 1 + p_twiddles.size());
    for (size_t i = 0; i <= level; ++i) {
        out.emplace_back(q_twiddles[i]);
    }
    for (const auto& twiddle : p_twiddles) {
        out.emplace_back(twiddle);
    }
    return out;
}

void add_qp_pair_inplace(
    MyVector<uint64_t>& dst_a,
    MyVector<uint64_t>& dst_b,
    const MyVector<uint64_t>& src_a,
    const MyVector<uint64_t>& src_b,
    size_t N,
    const MyVector<uint64_t>& moduli) {
    const size_t limb_count = moduli.size();
    if (dst_a.size() != N * limb_count || dst_b.size() != N * limb_count ||
        src_a.size() != N * limb_count || src_b.size() != N * limb_count) {
        throw std::invalid_argument("ckks_linear_transform: invalid QP add buffer size");
    }
    for (size_t limb = 0; limb < limb_count; ++limb) {
        const uint64_t q = moduli[limb];
        uint64_t* __restrict dst_a_ptr = dst_a.data() + limb * N;
        uint64_t* __restrict dst_b_ptr = dst_b.data() + limb * N;
        const uint64_t* __restrict src_a_ptr = src_a.data() + limb * N;
        const uint64_t* __restrict src_b_ptr = src_b.data() + limb * N;
        for (size_t i = 0; i < N; ++i) {
            dst_a_ptr[i] = add_mod_q(dst_a_ptr[i], src_a_ptr[i], q);
            dst_b_ptr[i] = add_mod_q(dst_b_ptr[i], src_b_ptr[i], q);
        }
    }
}

void automorphism_qp_pair_accumulate(
    const MyVector<uint64_t>& src_a,
    const MyVector<uint64_t>& src_b,
    const MyVector<uint64_t>& extra_b,
    const MyVector<size_t>& dst_to_src_map,
    MyVector<uint64_t>& dst_a,
    MyVector<uint64_t>& dst_b,
    size_t N,
    const MyVector<uint64_t>& moduli,
    bool assign) {
    const size_t limb_count = moduli.size();
    const size_t flat_size = N * limb_count;
    if (src_a.size() != flat_size || src_b.size() != flat_size ||
        extra_b.size() != flat_size || dst_to_src_map.size() != N) {
        throw std::invalid_argument("ckks_linear_transform: invalid QP automorphism accumulation input");
    }
    if (assign) {
        dst_a.resize(flat_size);
        dst_b.resize(flat_size);
    } else if (dst_a.size() != flat_size || dst_b.size() != flat_size) {
        throw std::invalid_argument("ckks_linear_transform: invalid QP automorphism accumulator");
    }

    for (size_t limb = 0; limb < limb_count; ++limb) {
        const uint64_t q = moduli[limb];
        const uint64_t* __restrict src_a_ptr = src_a.data() + limb * N;
        const uint64_t* __restrict src_b_ptr = src_b.data() + limb * N;
        const uint64_t* __restrict extra_b_ptr = extra_b.data() + limb * N;
        uint64_t* __restrict dst_a_ptr = dst_a.data() + limb * N;
        uint64_t* __restrict dst_b_ptr = dst_b.data() + limb * N;
        for (size_t coeff = 0; coeff < N; ++coeff) {
            const size_t src_idx = dst_to_src_map[coeff];
            const uint64_t add_a = src_a_ptr[src_idx];
            const uint64_t add_b = add_mod_q(src_b_ptr[src_idx], extra_b_ptr[src_idx], q);
            if (assign) {
                dst_a_ptr[coeff] = add_a;
                dst_b_ptr[coeff] = add_b;
            } else {
                dst_a_ptr[coeff] = add_mod_q(dst_a_ptr[coeff], add_a, q);
                dst_b_ptr[coeff] = add_mod_q(dst_b_ptr[coeff], add_b, q);
            }
        }
    }
}

void multiply_qp_plaintext_accumulate_pair(
    const MyVector<uint64_t>& ct_a_qp,
    const MyVector<uint64_t>& ct_b_qp,
    const MyVector<uint64_t>& plaintext_qp,
    const MyVector<uint64_t>& moduli_qp,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddles_qp,
    const MyVector<uint64_t>& montgomery_neg_inv_qp,
    bool plaintext_montgomery,
    MyVector<uint64_t>& acc_a_qp,
    MyVector<uint64_t>& acc_b_qp,
    size_t N,
    PlaintextAccumulationMode mode) {
    const size_t limb_count = moduli_qp.size();
    const size_t flat_size = N * limb_count;
    const bool assign = mode == PlaintextAccumulationMode::Assign;
    if (ct_a_qp.size() != flat_size || ct_b_qp.size() != flat_size ||
        plaintext_qp.size() != flat_size ||
        twiddles_qp.size() < limb_count ||
        (plaintext_montgomery && montgomery_neg_inv_qp.size() < limb_count)) {
        throw std::invalid_argument("ckks_linear_transform: invalid QP plaintext multiply input");
    }
    if (assign) {
        acc_a_qp.resize(flat_size);
        acc_b_qp.resize(flat_size);
    } else if (acc_a_qp.size() != flat_size || acc_b_qp.size() != flat_size) {
        throw std::invalid_argument("ckks_linear_transform: invalid QP plaintext accumulator");
    }

    for (size_t limb = 0; limb < limb_count; ++limb) {
        const uint64_t q = moduli_qp[limb];
        const auto& tbl = twiddles_qp[limb];
        const uint64_t* __restrict ct_a_ptr = ct_a_qp.data() + limb * N;
        const uint64_t* __restrict ct_b_ptr = ct_b_qp.data() + limb * N;
        const uint64_t* __restrict pt_ptr = plaintext_qp.data() + limb * N;
        uint64_t* __restrict acc_a_ptr = acc_a_qp.data() + limb * N;
        uint64_t* __restrict acc_b_ptr = acc_b_qp.data() + limb * N;
        if (plaintext_montgomery) {
            multiply_montgomery_plaintext_accumulate_ciphertext_pair_limb(
                ct_a_ptr,
                ct_b_ptr,
                pt_ptr,
                acc_a_ptr,
                acc_b_ptr,
                N,
                q,
                montgomery_neg_inv_qp[limb],
                mode);
            continue;
        }

        for (size_t i = 0; i < N; ++i) {
            const uint64_t term_a =
                mul_mod_u64(ct_a_ptr[i], pt_ptr[i], q, &tbl.barrett_const);
            const uint64_t term_b =
                mul_mod_u64(ct_b_ptr[i], pt_ptr[i], q, &tbl.barrett_const);
            if (assign) {
                acc_a_ptr[i] = term_a;
                acc_b_ptr[i] = term_b;
            } else if (mode == PlaintextAccumulationMode::AddRaw) {
                acc_a_ptr[i] += term_a;
                acc_b_ptr[i] += term_b;
            } else {
                acc_a_ptr[i] = add_mod_q(acc_a_ptr[i], term_a, q);
                acc_b_ptr[i] = add_mod_q(acc_b_ptr[i], term_b, q);
            }
        }
    }
}

void make_direct_input_qp(
    const CKKSContext& context,
    const MyVector<uint64_t>& a_q,
    const MyVector<uint64_t>& b_q,
    size_t level,
    MyVector<uint64_t>& a_qp,
    MyVector<uint64_t>& b_qp) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t q_limb_count = level + 1;
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const auto& q_twiddles = context.getTwiddleNtt();
    const size_t total_limb_count = q_limb_count + p_moduli.size();
    if (a_q.size() != N * q_limb_count || b_q.size() != N * q_limb_count) {
        throw std::invalid_argument("ckks_linear_transform: invalid direct Q input size");
    }

    a_qp.resize(N * total_limb_count);
    b_qp.resize(N * total_limb_count);
    for (size_t limb = 0; limb < q_limb_count; ++limb) {
        const uint64_t q = q_moduli[limb];
        const auto* barrett = &q_twiddles[limb].barrett_const;
        uint64_t direct_scalar = 1;
        if (!params.fold_hybrid_keyswitch) {
            for (uint64_t p : p_moduli) {
                direct_scalar = mul_mod_u64(direct_scalar, p % q, q, barrett);
            }
        }
        const uint64_t* __restrict src_a = a_q.data() + limb * N;
        const uint64_t* __restrict src_b = b_q.data() + limb * N;
        uint64_t* __restrict dst_a = a_qp.data() + limb * N;
        uint64_t* __restrict dst_b = b_qp.data() + limb * N;
        if (direct_scalar == 1) {
            std::copy_n(src_a, N, dst_a);
            std::copy_n(src_b, N, dst_b);
        } else {
            for (size_t i = 0; i < N; ++i) {
                dst_a[i] = mul_mod_u64(src_a[i], direct_scalar, q, barrett);
                dst_b[i] = mul_mod_u64(src_b[i], direct_scalar, q, barrett);
            }
        }
    }
    std::fill(a_qp.begin() + N * q_limb_count, a_qp.end(), uint64_t{0});
    std::fill(b_qp.begin() + N * q_limb_count, b_qp.end(), uint64_t{0});
}

void inverse_ntt_p_limbs_qp_inplace(
    const CKKSContext& context,
    MyVector<uint64_t>& qp_poly,
    size_t level) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t q_limb_count = level + 1;
    const auto& p_moduli = params.getPModuli();
    const auto& p_twiddles = context.getPTwiddleNtt();
    if (qp_poly.size() != N * (q_limb_count + p_moduli.size())) {
        throw std::invalid_argument("ckks_linear_transform: invalid QP P-limb INTT buffer");
    }
    for (size_t pi = 0; pi < p_moduli.size(); ++pi) {
        const auto& tbl = p_twiddles[pi];
        ntt_inverse_dif2_dispatch(
            qp_poly.data() + (q_limb_count + pi) * N, N, p_moduli[pi],
            tbl, true);
    }
}

void inverse_ntt_qp_inplace(
    const CKKSContext& context,
    MyVector<uint64_t>& qp_poly,
    size_t level,
    const MyVector<uint64_t>& moduli_qp) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t q_limb_count = level + 1;
    const auto& q_twiddles = context.getTwiddleNtt();
    const auto& p_twiddles = context.getPTwiddleNtt();
    const auto& p_moduli = params.getPModuli();
    if (qp_poly.size() != N * moduli_qp.size()) {
        throw std::invalid_argument("ckks_linear_transform: invalid QP INTT buffer");
    }
    for (size_t limb = 0; limb < q_limb_count; ++limb) {
        const auto& tbl = q_twiddles[limb];
        ntt_inverse_dif2_dispatch(
            qp_poly.data() + limb * N, N, moduli_qp[limb], tbl, true);
    }
    for (size_t pi = 0; pi < p_moduli.size(); ++pi) {
        const auto& tbl = p_twiddles[pi];
        ntt_inverse_dif2_dispatch(
            qp_poly.data() + (q_limb_count + pi) * N, N, p_moduli[pi],
            tbl, true);
    }
}

MyVector<uint64_t> moddown_qp_poly_to_q_eval(
    const CKKSContext& context,
    size_t level,
    MyVector<uint64_t> tmp,
    const MyVector<uint64_t>& moduli_qp) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t q_limb_count = level + 1;
    const auto& p_moduli = params.getPModuli();
    if (tmp.size() != N * moduli_qp.size()) {
        throw std::invalid_argument("ckks_linear_transform: invalid QP single ModDown input");
    }

    if (params.fold_hybrid_keyswitch) {
        inverse_ntt_p_limbs_qp_inplace(context, tmp, level);
        ckks_apply_folded_approx_crt_moddown_eval_q(
            p_moduli,
            active_q_moduli_for_level(context, level),
            context.getTwiddleNtt(),
            context.getFoldedPInvModQForLevel(level),
            context.getFoldedPInvModQShoupForLevel(level),
            tmp.data() + q_limb_count * N,
            tmp.data(),
            N);
        tmp.resize(N * q_limb_count);
        return tmp;
    }

    inverse_ntt_qp_inplace(context, tmp, level, moduli_qp);
    const auto& p_to_q_precomp = context.getPToQModUpPrecomp(level);
    const auto p_inv_mod_q = context.getPInvModQForLevel(level);
    const auto p_inv_mod_q_shoup = context.getPInvModQShoupForLevel(level);
    MyVector<const uint64_t*> src_p_ptrs;
    src_p_ptrs.reserve(p_moduli.size());
    MyVector<uint64_t*> dst_q_ptrs;
    dst_q_ptrs.reserve(q_limb_count);
    for (size_t pi = 0; pi < p_moduli.size(); ++pi) {
        src_p_ptrs.emplace_back(tmp.data() + (q_limb_count + pi) * N);
    }
    for (size_t qi = 0; qi < q_limb_count; ++qi) {
        dst_q_ptrs.emplace_back(tmp.data() + qi * N);
    }
    ckks_apply_approx_crt_moddown(
        p_to_q_precomp,
        src_p_ptrs,
        dst_q_ptrs,
        p_inv_mod_q,
        p_inv_mod_q_shoup,
        N);
    tmp.resize(N * q_limb_count);
    ntt_forward_rns_flat_inplace(
        tmp.data(),
        N,
        q_limb_count,
        active_q_moduli_for_level(context, level),
        context.getTwiddleNtt(),
        false);
    return tmp;
}

std::pair<MyVector<uint64_t>, MyVector<uint64_t>> moddown_qp_pair_to_q_eval(
    const CKKSContext& context,
    size_t level,
    MyVector<uint64_t> tmp_a,
    MyVector<uint64_t> tmp_b,
    const MyVector<uint64_t>& moduli_qp) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t q_limb_count = level + 1;
    const auto& p_moduli = params.getPModuli();
    if (tmp_a.size() != N * moduli_qp.size() ||
        tmp_b.size() != N * moduli_qp.size()) {
        throw std::invalid_argument("ckks_linear_transform: invalid QP pair ModDown input");
    }

    if (params.fold_hybrid_keyswitch) {
        inverse_ntt_p_limbs_qp_inplace(context, tmp_a, level);
        inverse_ntt_p_limbs_qp_inplace(context, tmp_b, level);
        ckks_apply_folded_approx_crt_moddown_pair_eval_q(
            p_moduli,
            active_q_moduli_for_level(context, level),
            context.getTwiddleNtt(),
            context.getFoldedPInvModQForLevel(level),
            context.getFoldedPInvModQShoupForLevel(level),
            tmp_a.data() + q_limb_count * N,
            tmp_a.data(),
            tmp_b.data() + q_limb_count * N,
            tmp_b.data(),
            N);
        tmp_a.resize(N * q_limb_count);
        tmp_b.resize(N * q_limb_count);
        return {std::move(tmp_a), std::move(tmp_b)};
    }

    inverse_ntt_qp_inplace(context, tmp_a, level, moduli_qp);
    inverse_ntt_qp_inplace(context, tmp_b, level, moduli_qp);
    const auto& p_to_q_precomp = context.getPToQModUpPrecomp(level);
    const auto p_inv_mod_q = context.getPInvModQForLevel(level);
    const auto p_inv_mod_q_shoup = context.getPInvModQShoupForLevel(level);
    MyVector<const uint64_t*> src_p_ptrs;
    src_p_ptrs.reserve(p_moduli.size());
    MyVector<uint64_t*> dst_q_ptrs;
    dst_q_ptrs.reserve(q_limb_count);
    for (size_t pi = 0; pi < p_moduli.size(); ++pi) {
        src_p_ptrs.emplace_back(tmp_a.data() + (q_limb_count + pi) * N);
    }
    for (size_t qi = 0; qi < q_limb_count; ++qi) {
        dst_q_ptrs.emplace_back(tmp_a.data() + qi * N);
    }
    ckks_apply_approx_crt_moddown(
        p_to_q_precomp,
        src_p_ptrs,
        dst_q_ptrs,
        p_inv_mod_q,
        p_inv_mod_q_shoup,
        N);
    src_p_ptrs.clear();
    dst_q_ptrs.clear();
    for (size_t pi = 0; pi < p_moduli.size(); ++pi) {
        src_p_ptrs.emplace_back(tmp_b.data() + (q_limb_count + pi) * N);
    }
    for (size_t qi = 0; qi < q_limb_count; ++qi) {
        dst_q_ptrs.emplace_back(tmp_b.data() + qi * N);
    }
    ckks_apply_approx_crt_moddown(
        p_to_q_precomp,
        src_p_ptrs,
        dst_q_ptrs,
        p_inv_mod_q,
        p_inv_mod_q_shoup,
        N);
    tmp_a.resize(N * q_limb_count);
    tmp_b.resize(N * q_limb_count);
    const auto q_moduli_active = active_q_moduli_for_level(context, level);
    ntt_forward_rns_flat_inplace(tmp_a.data(), N, q_limb_count, q_moduli_active, context.getTwiddleNtt(), false);
    ntt_forward_rns_flat_inplace(tmp_b.data(), N, q_limb_count, q_moduli_active, context.getTwiddleNtt(), false);
    return {std::move(tmp_a), std::move(tmp_b)};
}

CKKSLinearTransformCompiledStage compile_linear_transform_stage_metadata(
    size_t slots,
    size_t term_count,
    std::span<const size_t> diagonal_indices,
    size_t baby_step_count) {
    if (slots == 0 || term_count == 0 || baby_step_count == 0 || baby_step_count > slots) {
        throw std::invalid_argument("ckks_linear_transform: invalid compiled stage input");
    }

    CKKSLinearTransformCompiledStage compiled;
    compiled.baby_step_count = baby_step_count;
    compiled.giant_step_count = diagonal_indices.empty()
        ? ((term_count + baby_step_count - 1) / baby_step_count)
        : ((slots + baby_step_count - 1) / baby_step_count);
    compiled.terms_by_giant.resize(compiled.giant_step_count);
    compiled.baby_rotation_needed.assign(baby_step_count, char{0});
    compiled.giant_rotation_needed.assign(compiled.giant_step_count, char{0});
    for (size_t term = 0; term < term_count; ++term) {
        const size_t diagonal_index = diagonal_indices.empty() ? term : diagonal_indices[term];
        const size_t baby = diagonal_index % baby_step_count;
        const size_t giant = diagonal_index / baby_step_count;
        compiled.terms_by_giant[giant].emplace_back(term);
        compiled.baby_rotation_needed[baby] = char{1};
        compiled.giant_rotation_needed[giant] = char{1};
    }
    for (size_t baby = 1; baby < compiled.baby_rotation_needed.size(); ++baby) {
        if (compiled.baby_rotation_needed[baby]) {
            compiled.baby_rotations.emplace_back(baby);
        }
    }
    for (size_t giant = 1; giant < compiled.giant_rotation_needed.size(); ++giant) {
        if (compiled.giant_rotation_needed[giant]) {
            compiled.giant_rotations.emplace_back(giant * baby_step_count);
        }
    }
    return compiled;
}

void validate_stage_plaintexts(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    const CKKSLinearTransformStage& stage) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t slot_count = linear_transform_stage_slot_count(params, stage);
    const size_t level = ct.getLevel();
    const size_t limb_count = level + 1;
    const size_t flat_size = N * limb_count;

    if (stage.diagonals.empty()) {
        throw std::invalid_argument("ckks_linear_transform: empty diagonal set");
    }
    if (stage.baby_step_count == 0) {
        throw std::invalid_argument("ckks_linear_transform: baby-step count must be positive");
    }
    if (stage.diagonal_indices.size() != 0 &&
        stage.diagonal_indices.size() != stage.diagonals.size()) {
        throw std::invalid_argument("ckks_linear_transform: diagonal index count mismatch");
    }
    if (ct.getSecretOwner() != CKKSSecretOwner::Dense) {
        throw std::invalid_argument("ckks_linear_transform: expected a dense-secret ciphertext");
    }
    if (ct.getN() != N || ct.getNumPolys() != 2) {
        throw std::invalid_argument("ckks_linear_transform: invalid ciphertext shape");
    }
    if (level > params.getMaxLevel() || limb_count > params.getModuli().size()) {
        throw std::invalid_argument("ckks_linear_transform: invalid ciphertext level");
    }
    if (ct.getA().size() != flat_size || ct.getB().size() != flat_size) {
        throw std::invalid_argument("ckks_linear_transform: invalid ciphertext buffer size");
    }

    MyVector<char> seen_diagonals;
    if (!stage.diagonal_indices.empty()) {
        seen_diagonals.resize(slot_count, char{0});
    }
    for (size_t term = 0; term < stage.diagonals.size(); ++term) {
        if (!stage.diagonal_indices.empty()) {
            const size_t diagonal_index = stage.diagonal_indices[term];
            if (diagonal_index >= slot_count) {
                throw std::invalid_argument("ckks_linear_transform: diagonal index exceeds slot count");
            }
            if (seen_diagonals[diagonal_index]) {
                throw std::invalid_argument("ckks_linear_transform: duplicate diagonal index");
            }
            seen_diagonals[diagonal_index] = char{1};
        }
        const auto& diagonal = stage.diagonals[term];
        if (diagonal.getN() != N ||
            diagonal.getLevel() != level ||
            diagonal.getPlaintext().size() != flat_size) {
            throw std::invalid_argument("ckks_linear_transform: invalid diagonal plaintext metadata");
        }
    }
}

} // namespace

CKKSCiphertext ckks_apply_linear_transform_bsgs(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    const CKKSLinearTransformStage& stage) {
    validate_stage_plaintexts(context, ct, stage);

    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t slots = linear_transform_stage_slot_count(params, stage);
    const size_t level = ct.getLevel();
    const size_t limb_count = level + 1;
    const size_t term_count = stage.diagonals.size();
    const size_t baby_step_count = stage.baby_step_count;
    const auto& moduli = params.getModuli();
    const bool switching_key_montgomery = params.montgomery;

    ensure_linear_transform_rotation_keys(context, stage, slots);

    MyVector<uint64_t> montgomery_neg_inv(limb_count);
    for (size_t limb = 0; limb < limb_count; ++limb) {
        montgomery_neg_inv[limb] = ckks_montgomery_neg_inverse(moduli[limb]);
    }
    const size_t lazy_extra_term_limit =
        lazy_extra_term_limit_for_linear_transform(moduli, limb_count);

    const size_t giant_step_count =
        stage.diagonal_indices.empty()
            ? ((term_count + baby_step_count - 1) / baby_step_count)
            : ((slots + baby_step_count - 1) / baby_step_count);
    MyVector<MyVector<size_t>> terms_by_giant(giant_step_count);
    MyVector<char> baby_rotation_needed(baby_step_count, char{0});
    baby_rotation_needed[0] = char{1};
    for (size_t term = 0; term < term_count; ++term) {
        const size_t diagonal_index = stage_diagonal_index(stage, term);
        const size_t baby_rotation = diagonal_index % baby_step_count;
        const size_t giant_index = diagonal_index / baby_step_count;
        terms_by_giant[giant_index].emplace_back(term);
        baby_rotation_needed[baby_rotation] = char{1};
    }

    MyVector<CKKSCiphertext> generated_baby_rotations;
    generated_baby_rotations.reserve(term_count > 0 ? term_count - 1 : 0);
    MyVector<const CKKSCiphertext*> baby_rotations;
    baby_rotations.resize(baby_step_count, nullptr);
    baby_rotations[0] = &ct;
    if (std::any_of(
            baby_rotation_needed.begin() + 1,
            baby_rotation_needed.end(),
            [](char needed) { return needed != char{0}; })) {
        CKKSHoistWorkspace hoist_workspace;
        CKKSEvalHoistedCiphertextQP hoisted_ct;
        ckks_hoist_ciphertext_a_qp_eval(context, ct, hoist_workspace, hoisted_ct);
        for (size_t step = 1; step < baby_step_count; ++step) {
            if (!baby_rotation_needed[step]) {
                continue;
            }
            const int64_t rotation = static_cast<int64_t>(step);
            const auto& ntt_map = context.getRotationNttMap(rotation);
            MyVector<uint64_t> rotated_b = ct.getB();
            MyVector<uint64_t> rotated_a(N * limb_count, uint64_t{0});
            ckks_hybrid_key_switch_hoisted_inplace(
                context,
                hoisted_ct,
                context.getRotationKey(rotation),
                rotated_b,
                rotated_a,
                CKKSHybridKeySwitchCombine::Add,
                CKKSHybridKeySwitchCombine::Assign,
                switching_key_montgomery,
                ntt_map);
            generated_baby_rotations.emplace_back(
                N,
                std::move(rotated_a),
                std::move(rotated_b),
                ct.getScale(),
                level,
                ct.getSecretOwner(),
                ct.getMessageEncodingState());
            baby_rotations[step] = &generated_baby_rotations.back();
        }
    }

    std::optional<CKKSCiphertext> result;
    for (size_t giant = 0; giant < giant_step_count; ++giant) {
        const auto& giant_terms = terms_by_giant[giant];
        if (giant_terms.empty()) {
            continue;
        }

        MyVector<uint64_t> inner_a;
        MyVector<uint64_t> inner_b;
        bool have_inner_sum = false;
        double inner_scale = 0.0;
        size_t lazy_extra_terms = 0;
        const size_t giant_offset = giant * baby_step_count;

        for (const size_t term : giant_terms) {
            const size_t diagonal_index = stage_diagonal_index(stage, term);
            const size_t baby = diagonal_index - giant_offset;

            const auto* baby_ct_ptr = baby_rotations[baby];
            if (baby_ct_ptr == nullptr) {
                throw std::runtime_error("ckks_linear_transform: missing generated baby-step rotation");
            }
            const auto& baby_ct = *baby_ct_ptr;
            const PlaintextAccumulationMode mode =
                !have_inner_sum
                    ? PlaintextAccumulationMode::Assign
                    : (lazy_extra_term_limit > 0
                           ? PlaintextAccumulationMode::AddRaw
                           : PlaintextAccumulationMode::AddMod);
            multiply_plaintext_accumulate_ciphertext_pair(
                context,
                baby_ct,
                stage.diagonals[term],
                montgomery_neg_inv,
                inner_a,
                inner_b,
                mode);
            if (!have_inner_sum) {
                inner_scale = baby_ct.getScale() * stage.diagonals[term].getScale();
                have_inner_sum = true;
            } else if (lazy_extra_term_limit > 0) {
                ++lazy_extra_terms;
                if (lazy_extra_terms == lazy_extra_term_limit) {
                    reduce_lazy_accumulation_pair_rns_inplace(
                        inner_a,
                        inner_b,
                        N,
                        moduli,
                        limb_count,
                        lazy_extra_terms + 1);
                    lazy_extra_terms = 0;
                }
            }
        }

        if (!have_inner_sum) {
            continue;
        }
        if (lazy_extra_terms > 0) {
            reduce_lazy_accumulation_pair_rns_inplace(
                inner_a,
                inner_b,
                N,
                moduli,
                limb_count,
                lazy_extra_terms + 1);
        }

        CKKSCiphertext inner_sum(
            N,
            std::move(inner_a),
            std::move(inner_b),
            inner_scale,
            level,
            ct.getSecretOwner(),
            ct.getMessageEncodingState());
        if (giant_offset != 0) {
            ckks_rotate_inplace(context, inner_sum, static_cast<int64_t>(giant_offset));
        }

        if (!result.has_value()) {
            result.emplace(std::move(inner_sum));
        } else {
            ckks_add_inplace(context, *result, inner_sum);
        }
    }

    if (!result.has_value()) {
        throw std::runtime_error("ckks_linear_transform: failed to accumulate any diagonals");
    }

    if (stage.rescale_after) {
        ckks_rescale(context, *result);
    }
    result->setMessageEncodingState(stage.output_encoding_state);
    return std::move(*result);
}

CKKSCiphertext ckks_apply_linear_transform_owned_stage_q(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    const CKKSLinearTransformOwnedStage& owned_stage) {
    const CKKSLinearTransformStage stage = make_stage_view(owned_stage);
    validate_stage_plaintexts(context, ct, stage);

    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t slots = linear_transform_stage_slot_count(params, stage);
    const size_t level = ct.getLevel();
    const size_t limb_count = level + 1;
    const size_t term_count = owned_stage.diagonals.size();
    const auto& moduli = params.getModuli();
    const bool switching_key_montgomery = params.montgomery;

    const CKKSLinearTransformCompiledStage compiled =
        owned_stage.compiled.giant_step_count == 0
            ? compile_linear_transform_stage_metadata(
                  slots,
                  term_count,
                  stage.diagonal_indices,
                  owned_stage.baby_step_count)
            : owned_stage.compiled;
    if (compiled.baby_step_count != owned_stage.baby_step_count ||
        compiled.baby_rotation_needed.size() != owned_stage.baby_step_count ||
        compiled.terms_by_giant.empty() ||
        compiled.terms_by_giant.size() != compiled.giant_step_count) {
        throw std::invalid_argument("ckks_linear_transform: invalid compiled Q stage metadata");
    }

    ensure_linear_transform_rotation_keys(context, stage, slots);

    MyVector<uint64_t> montgomery_neg_inv(limb_count);
    for (size_t limb = 0; limb < limb_count; ++limb) {
        montgomery_neg_inv[limb] = ckks_montgomery_neg_inverse(moduli[limb]);
    }
    const size_t lazy_extra_term_limit =
        lazy_extra_term_limit_for_linear_transform(moduli, limb_count);

    MyVector<CKKSCiphertext> generated_baby_rotations;
    generated_baby_rotations.reserve(compiled.baby_rotations.size());
    MyVector<const CKKSCiphertext*> baby_rotations;
    baby_rotations.resize(compiled.baby_step_count, nullptr);
    baby_rotations[0] = &ct;
    if (!compiled.baby_rotations.empty()) {
        CKKSHoistWorkspace hoist_workspace;
        CKKSEvalHoistedCiphertextQP hoisted_ct;
        ckks_hoist_ciphertext_a_qp_eval(context, ct, hoist_workspace, hoisted_ct);
        for (const size_t baby_rotation : compiled.baby_rotations) {
            const int64_t rotation = static_cast<int64_t>(baby_rotation);
            const auto& ntt_map = context.getRotationNttMap(rotation);
            MyVector<uint64_t> rotated_b = ct.getB();
            MyVector<uint64_t> rotated_a(N * limb_count, uint64_t{0});
            ckks_hybrid_key_switch_hoisted_inplace(
                context,
                hoisted_ct,
                context.getRotationKey(rotation),
                rotated_b,
                rotated_a,
                CKKSHybridKeySwitchCombine::Add,
                CKKSHybridKeySwitchCombine::Assign,
                switching_key_montgomery,
                ntt_map);
            generated_baby_rotations.emplace_back(
                N,
                std::move(rotated_a),
                std::move(rotated_b),
                ct.getScale(),
                level,
                ct.getSecretOwner(),
                ct.getMessageEncodingState());
            baby_rotations[baby_rotation] = &generated_baby_rotations.back();
        }
    }

    std::optional<CKKSCiphertext> result;
    for (size_t giant = 0; giant < compiled.terms_by_giant.size(); ++giant) {
        const auto& giant_terms = compiled.terms_by_giant[giant];
        if (giant_terms.empty()) {
            continue;
        }

        MyVector<uint64_t> inner_a;
        MyVector<uint64_t> inner_b;
        bool have_inner_sum = false;
        double inner_scale = 0.0;
        size_t lazy_extra_terms = 0;
        const size_t giant_offset = giant * compiled.baby_step_count;

        for (const size_t term : giant_terms) {
            const size_t diagonal_index = stage_diagonal_index(stage, term);
            const size_t baby = diagonal_index - giant_offset;
            if (baby >= baby_rotations.size()) {
                throw std::runtime_error("ckks_linear_transform: invalid compiled baby-step index");
            }
            const auto* baby_ct_ptr = baby_rotations[baby];
            if (baby_ct_ptr == nullptr) {
                throw std::runtime_error("ckks_linear_transform: missing generated baby-step rotation");
            }
            const auto& baby_ct = *baby_ct_ptr;
            const PlaintextAccumulationMode mode =
                !have_inner_sum
                    ? PlaintextAccumulationMode::Assign
                    : (lazy_extra_term_limit > 0
                           ? PlaintextAccumulationMode::AddRaw
                           : PlaintextAccumulationMode::AddMod);
            multiply_plaintext_accumulate_ciphertext_pair(
                context,
                baby_ct,
                stage.diagonals[term],
                montgomery_neg_inv,
                inner_a,
                inner_b,
                mode);
            if (!have_inner_sum) {
                inner_scale = baby_ct.getScale() * stage.diagonals[term].getScale();
                have_inner_sum = true;
            } else if (lazy_extra_term_limit > 0) {
                ++lazy_extra_terms;
                if (lazy_extra_terms == lazy_extra_term_limit) {
                    reduce_lazy_accumulation_pair_rns_inplace(
                        inner_a,
                        inner_b,
                        N,
                        moduli,
                        limb_count,
                        lazy_extra_terms + 1);
                    lazy_extra_terms = 0;
                }
            }
        }

        if (!have_inner_sum) {
            continue;
        }
        if (lazy_extra_terms > 0) {
            reduce_lazy_accumulation_pair_rns_inplace(
                inner_a,
                inner_b,
                N,
                moduli,
                limb_count,
                lazy_extra_terms + 1);
        }

        CKKSCiphertext inner_sum(
            N,
            std::move(inner_a),
            std::move(inner_b),
            inner_scale,
            level,
            ct.getSecretOwner(),
            ct.getMessageEncodingState());
        if (giant_offset != 0) {
            ckks_rotate_inplace(context, inner_sum, static_cast<int64_t>(giant_offset));
        }

        if (!result.has_value()) {
            result.emplace(std::move(inner_sum));
        } else {
            ckks_add_inplace(context, *result, inner_sum);
        }
    }

    if (!result.has_value()) {
        throw std::runtime_error("ckks_linear_transform: failed to accumulate any diagonals");
    }

    if (owned_stage.rescale_after) {
        ckks_rescale(context, *result);
    }
    result->setMessageEncodingState(owned_stage.output_encoding_state);
    return std::move(*result);
}

CKKSCiphertext ckks_apply_linear_transform_owned_stage_qp(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    const CKKSLinearTransformOwnedStage& owned_stage) {
    const CKKSLinearTransformStage stage = make_stage_view(owned_stage);
    validate_stage_plaintexts(context, ct, stage);

    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t slots = linear_transform_stage_slot_count(params, stage);
    const size_t level = ct.getLevel();
    const size_t q_limb_count = level + 1;
    const size_t p_limb_count = params.getPModuli().size();
    const size_t total_limb_count = q_limb_count + p_limb_count;
    const size_t flat_qp_size = N * total_limb_count;
    const size_t term_count = owned_stage.diagonals.size();
    const bool switching_key_montgomery = params.montgomery;

    if (owned_stage.diagonal_qp_plaintexts.size() != term_count ||
        owned_stage.qp_limb_count != total_limb_count ||
        owned_stage.qp_p_limb_count != p_limb_count) {
        return ckks_apply_linear_transform_bsgs(context, ct, stage);
    }
    for (const auto& plaintext_qp : owned_stage.diagonal_qp_plaintexts) {
        if (plaintext_qp.size() != flat_qp_size) {
            throw std::invalid_argument("ckks_linear_transform: invalid QP plaintext buffer size");
        }
    }
    if (ct.getSecretOwner() != CKKSSecretOwner::Dense ||
        ct.getN() != N ||
        ct.getNumPolys() != 2 ||
        ct.getA().size() != N * q_limb_count ||
        ct.getB().size() != N * q_limb_count) {
        throw std::invalid_argument("ckks_linear_transform: invalid QP evaluator ciphertext input");
    }

    const CKKSLinearTransformCompiledStage compiled =
        owned_stage.compiled.giant_step_count == 0
            ? compile_linear_transform_stage_metadata(
                  slots,
                  term_count,
                  stage.diagonal_indices,
                  owned_stage.baby_step_count)
            : owned_stage.compiled;
    if (compiled.baby_step_count != owned_stage.baby_step_count ||
        compiled.baby_rotation_needed.size() != owned_stage.baby_step_count ||
        compiled.terms_by_giant.empty()) {
        throw std::invalid_argument("ckks_linear_transform: invalid compiled QP stage metadata");
    }

    ensure_linear_transform_rotation_keys(context, stage, slots);

    const auto moduli_qp = active_qp_moduli_for_level(context, level);
    const auto twiddles_qp = active_qp_twiddles_for_level(context, level);
    const size_t qp_limb_count = moduli_qp.size();
    MyVector<uint64_t> montgomery_neg_inv_qp;
    if (owned_stage.diagonal_qp_plaintexts_montgomery) {
        montgomery_neg_inv_qp.resize(qp_limb_count);
        for (size_t limb = 0; limb < qp_limb_count; ++limb) {
            montgomery_neg_inv_qp[limb] =
                ckks_montgomery_neg_inverse(moduli_qp[limb]);
        }
    }
    const size_t lazy_extra_term_limit =
        lazy_extra_term_limit_for_linear_transform(moduli_qp, qp_limb_count);

    MyVector<MyVector<uint64_t>> baby_a_qp(compiled.baby_step_count);
    MyVector<MyVector<uint64_t>> baby_b_qp(compiled.baby_step_count);
    if (compiled.baby_rotation_needed[0]) {
        make_direct_input_qp(
            context,
            ct.getA(),
            ct.getB(),
            level,
            baby_a_qp[0],
            baby_b_qp[0]);
    }
    if (!compiled.baby_rotations.empty()) {
        CKKSHoistWorkspace hoist_workspace;
        CKKSEvalHoistedCiphertextQP hoisted_ct;
        ckks_hoist_ciphertext_a_qp_eval(context, ct, hoist_workspace, hoisted_ct);
        for (const size_t baby_rotation : compiled.baby_rotations) {
            const int64_t rotation = static_cast<int64_t>(baby_rotation);
            const auto& ntt_map = context.getRotationNttMap(rotation);
            ckks_hybrid_key_switch_hoisted_qp_rotated(
                context,
                hoisted_ct,
                context.getRotationKey(rotation),
                ct.getB(),
                baby_b_qp[baby_rotation],
                baby_a_qp[baby_rotation],
                switching_key_montgomery,
                ntt_map);
        }
    }

    MyVector<uint64_t> out_a_qp;
    MyVector<uint64_t> out_b_qp;
    bool have_output = false;
    double output_scale = 0.0;
    MyVector<uint64_t> switch_b_qp;
    MyVector<uint64_t> switch_a_qp;

    for (size_t giant = 0; giant < compiled.terms_by_giant.size(); ++giant) {
        const auto& terms = compiled.terms_by_giant[giant];
        if (terms.empty()) {
            continue;
        }

        MyVector<uint64_t> inner_a_qp;
        MyVector<uint64_t> inner_b_qp;
        bool have_inner = false;
        size_t lazy_extra_terms = 0;
        const size_t giant_offset = giant * compiled.baby_step_count;
        for (const size_t term : terms) {
            const size_t diagonal_index = stage_diagonal_index(stage, term);
            const size_t baby = diagonal_index - giant_offset;
            if (baby >= baby_a_qp.size() ||
                baby_a_qp[baby].empty() ||
                baby_b_qp[baby].empty()) {
                throw std::runtime_error("ckks_linear_transform: missing QP baby-step rotation");
            }
            const PlaintextAccumulationMode mode =
                !have_inner
                    ? PlaintextAccumulationMode::Assign
                    : (lazy_extra_term_limit > 0
                           ? PlaintextAccumulationMode::AddRaw
                           : PlaintextAccumulationMode::AddMod);
            multiply_qp_plaintext_accumulate_pair(
                baby_a_qp[baby],
                baby_b_qp[baby],
                owned_stage.diagonal_qp_plaintexts[term],
                moduli_qp,
                twiddles_qp,
                montgomery_neg_inv_qp,
                owned_stage.diagonal_qp_plaintexts_montgomery,
                inner_a_qp,
                inner_b_qp,
                N,
                mode);
            if (!have_inner) {
                output_scale = ct.getScale() * owned_stage.diagonals[term].getScale();
                have_inner = true;
            } else if (lazy_extra_term_limit > 0) {
                ++lazy_extra_terms;
                if (lazy_extra_terms == lazy_extra_term_limit) {
                    reduce_lazy_accumulation_pair_rns_inplace(
                        inner_a_qp,
                        inner_b_qp,
                        N,
                        moduli_qp,
                        qp_limb_count,
                        lazy_extra_terms + 1);
                    lazy_extra_terms = 0;
                }
            }
        }
        if (!have_inner) {
            continue;
        }
        if (lazy_extra_terms > 0) {
            reduce_lazy_accumulation_pair_rns_inplace(
                inner_a_qp,
                inner_b_qp,
                N,
                moduli_qp,
                qp_limb_count,
                lazy_extra_terms + 1);
        }

        if (giant_offset == 0) {
            if (!have_output) {
                out_a_qp = std::move(inner_a_qp);
                out_b_qp = std::move(inner_b_qp);
                have_output = true;
            } else {
                add_qp_pair_inplace(out_a_qp, out_b_qp, inner_a_qp, inner_b_qp, N, moduli_qp);
            }
            continue;
        }

        const int64_t giant_rotation = static_cast<int64_t>(giant_offset);
        const auto& ntt_map = context.getRotationNttMap(giant_rotation);
        const auto key_term_q = moddown_qp_poly_to_q_eval(
            context,
            level,
            std::move(inner_a_qp),
            moduli_qp);
        ckks_hybrid_key_switch_qp_from_key_term(
            context,
            level,
            key_term_q,
            context.getRotationKey(giant_rotation),
            switch_b_qp,
            switch_a_qp,
            switching_key_montgomery);
        automorphism_qp_pair_accumulate(
            switch_a_qp,
            switch_b_qp,
            inner_b_qp,
            ntt_map,
            out_a_qp,
            out_b_qp,
            N,
            moduli_qp,
            !have_output);
        have_output = true;
    }

    if (!have_output) {
        throw std::runtime_error("ckks_linear_transform: failed to accumulate any QP diagonals");
    }

    auto [out_a, out_b] = moddown_qp_pair_to_q_eval(
        context,
        level,
        std::move(out_a_qp),
        std::move(out_b_qp),
        moduli_qp);
    CKKSCiphertext result(
        N,
        std::move(out_a),
        std::move(out_b),
        output_scale,
        level,
        ct.getSecretOwner(),
        ct.getMessageEncodingState());

    if (owned_stage.rescale_after) {
        ckks_rescale(context, result);
    }
    result.setMessageEncodingState(owned_stage.output_encoding_state);
    return result;
}

CKKSCiphertext ckks_apply_linear_transform_plan(
    const CKKSContext& context,
    CKKSCiphertext ct,
    const CKKSLinearTransformPlan& plan) {
    if (plan.stages.empty()) {
        throw std::invalid_argument("ckks_linear_transform: empty transform plan");
    }

    for (const auto& owned_stage : plan.stages) {
        if (should_use_qp_linear_transform(
                owned_stage,
                plan.use_qp_evaluator)) {
            ct = ckks_apply_linear_transform_owned_stage_qp(context, ct, owned_stage);
        } else {
            ct = ckks_apply_linear_transform_owned_stage_q(context, ct, owned_stage);
        }
    }
    return ct;
}
