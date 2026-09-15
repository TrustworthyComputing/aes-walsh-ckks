#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "ckks_context.hpp"
#include "ckks_keyswitch.hpp"

namespace {
struct HoistCiphertextConfig {
    size_t N = 0;
    size_t level = 0;
    size_t q_limb_count = 0;
    size_t p_limb_count = 0;
    size_t active_partition_count = 0;
    long double scale = 0.0L;
    MyVector<uint64_t> active_q_moduli;
};

size_t validate_hybrid_switch_key_part_shape(
    const CKKSCiphertext& key_part,
    size_t N,
    size_t active_q_limb_count,
    size_t p_limb_count) {
    const size_t key_q_limb_count = key_part.getLevel() + 1;
    const size_t key_limb_count = key_q_limb_count + p_limb_count;
    if (key_q_limb_count < active_q_limb_count) {
        throw std::runtime_error("ckks_hybrid_keyswitch: switch key level is below ciphertext level");
    }
    if (key_part.getA().size() != N * key_limb_count ||
        key_part.getB().size() != N * key_limb_count) {
        throw std::runtime_error("ckks_hybrid_keyswitch: invalid key limb count");
    }
    return key_q_limb_count;
}

void apply_ntt_automorphism_to_flat_rns_poly_inplace(
    MyVector<uint64_t>& poly,
    size_t N,
    size_t limb_count,
    std::span<const size_t> dst_to_src_map) {
    if (dst_to_src_map.empty()) {
        return;
    }
    if (poly.size() != N * limb_count || dst_to_src_map.size() != N) {
        throw std::invalid_argument("ckks_hybrid_keyswitch: invalid automorphism output size");
    }

    thread_local MyVector<uint64_t> tls_automorphism_out;
    MyVector<uint64_t>& out = tls_automorphism_out;
    out.resize(poly.size());
    for (size_t limb = 0; limb < limb_count; ++limb) {
        const uint64_t* __restrict src = poly.data() + limb * N;
        uint64_t* __restrict dst = out.data() + limb * N;
        for (size_t coeff = 0; coeff < N; ++coeff) {
            dst[coeff] = src[dst_to_src_map[coeff]];
        }
    }
    poly.swap(out);
}

HoistCiphertextConfig build_hoist_ciphertext_config(
    const CKKSContext& context,
    const CKKSCiphertext& ct) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    if (ct.getNumPolys() != 2 || ct.getN() != N) {
        throw std::invalid_argument("ckks_keyswitch: hoisting expects a 2-polynomial ciphertext");
    }

    const size_t level = ct.getLevel();
    const size_t q_limb_count = level + 1;
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const size_t dnum = params.getDnum();
    const size_t max_q_limb_count = params.getMaxLevel() + 1;
    const size_t alpha = (max_q_limb_count + dnum - 1) / dnum;
    const size_t active_partition_count =
        std::min(dnum, (q_limb_count + alpha - 1) / alpha);
    if (level > params.getMaxLevel() || q_limb_count > q_moduli.size()) {
        throw std::invalid_argument("ckks_keyswitch: invalid ciphertext level for hoisting");
    }

    HoistCiphertextConfig config;
    config.N = N;
    config.level = level;
    config.q_limb_count = q_limb_count;
    config.p_limb_count = p_moduli.size();
    config.active_partition_count = active_partition_count;
    config.scale = ct.getScale();
    config.active_q_moduli.assign(q_moduli.begin(), q_moduli.begin() + q_limb_count);
    return config;
}

void build_hoisted_a_partition_qp_buffers(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    const HoistCiphertextConfig& config,
    CKKSHoistWorkspace& workspace,
    MyVector<MyVector<uint64_t>>& a_partition_qp) {
    const auto& params = context.getParams();
    const size_t N = config.N;
    const size_t q_limb_count = config.q_limb_count;
    const auto& p_moduli = params.getPModuli();
    const auto& q_twiddle_ntt = context.getTwiddleNtt();
    const size_t dnum = params.getDnum();
    const size_t max_q_limb_count = params.getMaxLevel() + 1;
    const size_t alpha = (max_q_limb_count + dnum - 1) / dnum;
    const size_t total_limb_count = q_limb_count + p_moduli.size();

    workspace.a_q_coeffs = ct.getA();
    ntt_inverse_rns_flat_inplace(
        workspace.a_q_coeffs.data(),
        N,
        q_limb_count,
        config.active_q_moduli,
        q_twiddle_ntt,
        true);

    a_partition_qp.resize(config.active_partition_count);

    workspace.hybrid_part_modup_precomp.clear();
    workspace.hybrid_part_modup_precomp.reserve(config.active_partition_count);
    for (size_t part = 0; part < config.active_partition_count; ++part) {
        workspace.hybrid_part_modup_precomp.emplace_back(
            &context.getHybridPartitionModUpPrecomp(config.level, part));
    }

    workspace.src_ptrs.clear();
    workspace.src_ptrs.reserve(alpha);
    workspace.dst_ptrs.clear();
    workspace.dst_ptrs.reserve(total_limb_count);

    for (size_t part = 0; part < config.active_partition_count; ++part) {
        const size_t part_start = part * alpha;
        const size_t part_end = std::min(part_start + alpha, q_limb_count);
        if (part_start >= part_end) {
            continue;
        }
        const size_t src_count = part_end - part_start;

        auto& a_digit_qp = a_partition_qp[part];
        a_digit_qp.resize(N * total_limb_count);

        workspace.src_ptrs.clear();
        for (size_t s = 0; s < src_count; ++s) {
            workspace.src_ptrs.emplace_back(workspace.a_q_coeffs.data() + (part_start + s) * N);
        }
        workspace.dst_ptrs.clear();
        for (size_t i = 0; i < part_start; ++i) {
            workspace.dst_ptrs.emplace_back(a_digit_qp.data() + i * N);
        }
        for (size_t i = part_end; i < q_limb_count; ++i) {
            workspace.dst_ptrs.emplace_back(a_digit_qp.data() + i * N);
        }
        for (size_t i = 0; i < p_moduli.size(); ++i) {
            workspace.dst_ptrs.emplace_back(a_digit_qp.data() + (q_limb_count + i) * N);
        }
        if (!workspace.dst_ptrs.empty()) {
            ckks_apply_approx_crt_modup(
                *workspace.hybrid_part_modup_precomp[part],
                workspace.src_ptrs,
                workspace.dst_ptrs,
                N);
        }
    }
}

void copy_source_q_eval_limbs_to_hoisted_a_partitions(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    CKKSEvalHoistedCiphertextQP& eval) {
    const auto& params = context.getParams();
    const size_t N = eval.N;
    const size_t max_q_limb_count = params.getMaxLevel() + 1;
    const size_t dnum = params.getDnum();
    const size_t alpha = (max_q_limb_count + dnum - 1) / dnum;

    if (ct.getN() != N || ct.getA().size() != N * eval.q_limb_count) {
        throw std::invalid_argument("ckks_keyswitch: invalid eval source for hoisted A partition copy");
    }

    for (size_t part = 0; part < eval.a_partition_qp_eval.size(); ++part) {
        const size_t part_start = part * alpha;
        const size_t part_end = std::min(part_start + alpha, eval.q_limb_count);
        auto& a_qp = eval.a_partition_qp_eval[part];
        for (size_t limb = part_start; limb < part_end; ++limb) {
            std::copy_n(ct.getA().data() + limb * N, N, a_qp.data() + limb * N);
        }
    }
}

void forward_ntt_eval_hoisted_a_partitions_inplace(
    const CKKSContext& context,
    CKKSEvalHoistedCiphertextQP& eval,
    bool source_q_limbs_already_eval = false) {
    const size_t N = eval.N;
    const size_t total_limb_count = eval.q_limb_count + eval.p_limb_count;
    const auto& q_twiddle_ntt = context.getTwiddleNtt();
    const auto& p_twiddle_ntt = context.getPTwiddleNtt();
    const auto& params = context.getParams();
    const size_t max_q_limb_count = params.getMaxLevel() + 1;
    const size_t dnum = params.getDnum();
    const size_t alpha = (max_q_limb_count + dnum - 1) / dnum;
    for (size_t part = 0; part < eval.a_partition_qp_eval.size(); ++part) {
        auto& a_qp = eval.a_partition_qp_eval[part];
        if (a_qp.size() != N * total_limb_count) {
            throw std::invalid_argument("ckks_keyswitch: invalid hoisted A partition size for forward NTT");
        }
        const size_t part_start = part * alpha;
        const size_t part_end = std::min(part_start + alpha, eval.q_limb_count);
        for (size_t limb = 0; limb < eval.q_limb_count; ++limb) {
            if (source_q_limbs_already_eval && limb >= part_start && limb < part_end) {
                continue;
            }
            const auto& tbl = q_twiddle_ntt[limb];
            ntt_forward_dit2_dispatch(
                a_qp.data() + limb * N, N, eval.moduli[limb], tbl, true);
        }
        for (size_t pi = 0; pi < eval.p_limb_count; ++pi) {
            const size_t limb = eval.q_limb_count + pi;
            const auto& tbl = p_twiddle_ntt[pi];
            ntt_forward_dit2_dispatch(
                a_qp.data() + limb * N, N, eval.moduli[limb], tbl, true);
        }
    }
}
template <class ConsumeDstValue>
void apply_approx_crt_base_change(
    const ApproxCrtModUpPrecomp& precomp,
    const MyVector<const uint64_t*>& src_ptrs,
    size_t N,
    ConsumeDstValue&& consume_dst_value) {
    const size_t src_count = precomp.src_moduli.size();
    const size_t dst_count = precomp.dst_moduli.size();
    if (src_ptrs.size() != src_count) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid base-change source pointer count");
    }
    if (dst_count == 0) {
        return;
    }
    if (precomp.qhat_inv_mod_src_shoup.size() != src_count ||
        precomp.qhat_mod_dst_flat.size() != dst_count * src_count ||
        precomp.qhat_mod_dst_shoup_flat.size() != dst_count * src_count) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid base-change precompute metadata");
    }

    MyVector<uint64_t> alpha(src_count);
    for (size_t coeff = 0; coeff < N; ++coeff) {
        for (size_t i = 0; i < src_count; ++i) {
            const uint64_t qi = precomp.src_moduli[i];
            const uint64_t x = src_ptrs[i][coeff];
            alpha[i] = mul_mod_shoup(x, precomp.qhat_inv_mod_src[i], precomp.qhat_inv_mod_src_shoup[i], qi);
        }

        for (size_t t = 0; t < dst_count; ++t) {
            const uint64_t m = precomp.dst_moduli[t];
            uint64_t accum = 0;
            const size_t row_off = t * src_count;
            const uint64_t* __restrict qhat_mod_row = precomp.qhat_mod_dst_flat.data() + row_off;
            const uint64_t* __restrict qhat_mod_shoup_row =
                precomp.qhat_mod_dst_shoup_flat.data() + row_off;
            for (size_t i = 0; i < src_count; ++i) {
                const uint64_t term = mul_mod_shoup(alpha[i], qhat_mod_row[i], qhat_mod_shoup_row[i], m);
                accum = add_mod_q(accum, term, m);
            }
            consume_dst_value(coeff, t, accum, m);
        }
    }
}

template <class ConsumeDstValue>
// The HPS RNS basis-extension procedure is described in:
// S. Halevi, Y. Polyakov, and V. Shoup, "An Improved RNS Variant of the BFV
// Homomorphic Encryption Scheme," CT-RSA 2019, pp. 83-105.
// DOI: 10.1007/978-3-030-12612-4_5
// IACR ePrint: https://eprint.iacr.org/2018/117
void apply_hps_crt_base_change(
    const ApproxCrtModUpPrecomp& precomp,
    const MyVector<const uint64_t*>& src_ptrs,
    size_t N,
    ConsumeDstValue&& consume_dst_value) {
    const size_t src_count = precomp.src_moduli.size();
    const size_t dst_count = precomp.dst_moduli.size();
    if (src_ptrs.size() != src_count) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid HPS base-change source pointer count");
    }
    if (dst_count == 0) {
        return;
    }
    if (precomp.qhat_inv_mod_src_shoup.size() != src_count ||
        precomp.src_modulus_inverse.size() != src_count ||
        precomp.qhat_mod_dst_flat.size() != dst_count * src_count ||
        precomp.qhat_mod_dst_shoup_flat.size() != dst_count * src_count ||
        precomp.src_product_mod_dst.size() != dst_count ||
        precomp.src_product_mod_dst_shoup.size() != dst_count ||
        precomp.src_product_alpha_mod_dst_flat.size() != dst_count * (src_count + 1)) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid HPS base-change precompute metadata");
    }

    MyVector<uint64_t> scaled_residues(src_count);
    for (size_t coeff = 0; coeff < N; ++coeff) {
        double alpha_estimate = 0.0;
        for (size_t i = 0; i < src_count; ++i) {
            const uint64_t qi = precomp.src_moduli[i];
            const uint64_t x = reduce_mod_4q(src_ptrs[i][coeff], qi);
            const uint64_t scaled =
                mul_mod_shoup(x, precomp.qhat_inv_mod_src[i], precomp.qhat_inv_mod_src_shoup[i], qi);
            scaled_residues[i] = scaled;
            alpha_estimate += static_cast<double>(scaled) * precomp.src_modulus_inverse[i];
        }
        const uint64_t hps_alpha =
            std::min<uint64_t>(static_cast<uint64_t>(alpha_estimate + 0.5), src_count);

        for (size_t t = 0; t < dst_count; ++t) {
            const uint64_t m = precomp.dst_moduli[t];
            uint64_t accum = 0;
            const size_t row_off = t * src_count;
            const uint64_t* __restrict qhat_mod_row = precomp.qhat_mod_dst_flat.data() + row_off;
            const uint64_t* __restrict qhat_mod_shoup_row =
                precomp.qhat_mod_dst_shoup_flat.data() + row_off;
            for (size_t i = 0; i < src_count; ++i) {
                const uint64_t term =
                    mul_mod_shoup(scaled_residues[i], qhat_mod_row[i], qhat_mod_shoup_row[i], m);
                accum = add_mod_q(accum, term, m);
            }
            const uint64_t correction =
                precomp.src_product_alpha_mod_dst_flat[t * (src_count + 1) + hps_alpha];
            consume_dst_value(coeff, t, sub_mod_q(accum, correction, m), m);
        }
    }
}
} // namespace

void ckks_apply_ntt_automorphism_to_flat_rns_poly_inplace(
    MyVector<uint64_t>& poly,
    size_t N,
    size_t limb_count,
    std::span<const size_t> dst_to_src_map) {
    apply_ntt_automorphism_to_flat_rns_poly_inplace(
        poly, N, limb_count, dst_to_src_map);
}

void ckks_hoist_ciphertext_a_qp_eval(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    CKKSHoistWorkspace& workspace,
    CKKSEvalHoistedCiphertextQP& out) {
    const auto config = build_hoist_ciphertext_config(context, ct);
    const auto& p_moduli = context.getParams().getPModuli();
    out.N = config.N;
    out.level = config.level;
    out.q_limb_count = config.q_limb_count;
    out.p_limb_count = config.p_limb_count;
    out.scale = config.scale;
    out.moduli.resize(config.q_limb_count + p_moduli.size());
    std::copy(config.active_q_moduli.begin(), config.active_q_moduli.end(), out.moduli.begin());
    std::copy(p_moduli.begin(), p_moduli.end(), out.moduli.begin() + config.q_limb_count);
    out.a_partition_qp_eval.resize(config.active_partition_count);
    out.b_partition_qp_eval.clear();

    build_hoisted_a_partition_qp_buffers(
        context,
        ct,
        config,
        workspace,
        out.a_partition_qp_eval);
    copy_source_q_eval_limbs_to_hoisted_a_partitions(context, ct, out);
    forward_ntt_eval_hoisted_a_partitions_inplace(
        context,
        out,
        /*source_q_limbs_already_eval=*/true);
}

void ckks_apply_approx_crt_modup(
    const ApproxCrtModUpPrecomp& precomp,
    const MyVector<const uint64_t*>& src_ptrs,
    const MyVector<uint64_t*>& dst_ptrs,
    size_t N) {
    const size_t dst_count = precomp.dst_moduli.size();
    if (dst_ptrs.size() != dst_count) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid ModUp destination pointer count");
    }
    apply_hps_crt_base_change(
        precomp,
        src_ptrs,
        N,
        [&](size_t coeff, size_t t, uint64_t mapped, uint64_t /*modulus*/) {
            dst_ptrs[t][coeff] = mapped;
        });
}

void ckks_apply_approx_crt_moddown(
    const ApproxCrtModUpPrecomp& precomp,
    const MyVector<const uint64_t*>& src_ptrs,
    const MyVector<uint64_t*>& dst_q_ptrs,
    std::span<const uint64_t> p_inv_mod_q,
    std::span<const uint64_t> p_inv_mod_q_shoup,
    size_t N) {
    const size_t dst_count = precomp.dst_moduli.size();
    if (dst_q_ptrs.size() != dst_count) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid fused ModDown destination pointer count");
    }
    if (p_inv_mod_q.size() != dst_count) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid fused ModDown inverse count");
    }
    if (p_inv_mod_q_shoup.size() != dst_count) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid fused ModDown inverse shoup count");
    }
    if (dst_count == 0) {
        return;
    }
    apply_hps_crt_base_change(
        precomp,
        src_ptrs,
        N,
        [&](size_t coeff, size_t t, uint64_t modup, uint64_t q) {
            const uint64_t cur = reduce_mod_4q(dst_q_ptrs[t][coeff], q);
            const uint64_t diff = sub_mod_q(cur, modup, q);
            dst_q_ptrs[t][coeff] = mul_mod_shoup(diff, p_inv_mod_q[t], p_inv_mod_q_shoup[t], q);
        });
}

void ckks_pointwise_multiply_accumulate_montgomery_key(
    const uint64_t* __restrict digit,
    const uint64_t* __restrict key_montgomery,
    uint64_t* __restrict acc,
    size_t N,
    uint64_t q,
    uint64_t q_neg_inv,
    const Negacyclic_NTT_Twiddles&) {
    for (size_t i = 0; i < N; ++i) {
        const uint64_t term =
            ckks_montgomery_mul_normal_by_montgomery(
                digit[i],
                key_montgomery[i],
                q,
                q_neg_inv);
        acc[i] = add_mod_q(acc[i], term, q);
    }
}

namespace {

bool ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
    const uint64_t* __restrict digit,
    const uint64_t* __restrict key_montgomery,
    uint64_t* __restrict acc,
    size_t N,
    uint64_t q,
    uint64_t q_neg_inv,
    const Negacyclic_NTT_Twiddles&) {
    for (size_t i = 0; i < N; ++i) {
        const uint64_t term =
            ckks_montgomery_mul_normal_by_montgomery(
                digit[i],
                key_montgomery[i],
                q,
                q_neg_inv);
        acc[i] = add_mod_q(acc[i], term, q);
    }
    return false;
}

bool ckks_pointwise_multiply_accumulate_normal_key_pair(
    const uint64_t* __restrict digit,
    const uint64_t* __restrict key_a,
    const uint64_t* __restrict key_b,
    uint64_t* __restrict acc_a,
    uint64_t* __restrict acc_b,
    size_t N,
    uint64_t q,
    const Negacyclic_NTT_Twiddles& table,
    uint64_t input_mod_factor) {
    pointwise_multiply_accumulate_dispatch(
        digit, key_a, acc_a, N, q, table, input_mod_factor);
    pointwise_multiply_accumulate_dispatch(
        digit, key_b, acc_b, N, q, table, input_mod_factor);
    return false;
}

void reduce_lazy_accumulator_pair_scalar(
    MyVector<uint64_t>& acc_a,
    MyVector<uint64_t>& acc_b,
    size_t N,
    std::span<const uint64_t> q_moduli,
    std::span<const uint64_t> p_moduli) {
    size_t limb = 0;
    auto reduce_limb = [&](uint64_t modulus) {
        uint64_t* __restrict acc_a_ptr = acc_a.data() + limb * N;
        uint64_t* __restrict acc_b_ptr = acc_b.data() + limb * N;
        for (size_t coeff = 0; coeff < N; ++coeff) {
            acc_a_ptr[coeff] %= modulus;
            acc_b_ptr[coeff] %= modulus;
        }
        ++limb;
    };
    for (uint64_t q : q_moduli) {
        reduce_limb(q);
    }
    for (uint64_t p : p_moduli) {
        reduce_limb(p);
    }
}

void reduce_lazy_accumulator_pair_if_needed(
    bool needed,
    MyVector<uint64_t>& acc_a,
    MyVector<uint64_t>& acc_b,
    size_t N,
    std::span<const uint64_t> q_moduli,
    std::span<const uint64_t> p_moduli) {
    if (!needed) {
        return;
    }
    reduce_lazy_accumulator_pair_scalar(acc_a, acc_b, N, q_moduli, p_moduli);
}

}  // namespace

void ckks_apply_folded_approx_crt_moddown_pair_eval_q(
    const MyVector<uint64_t>& p_moduli,
    const MyVector<uint64_t>& q_moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& q_twiddle_ntt,
    std::span<const uint64_t> p_inv_mod_q,
    std::span<const uint64_t> p_inv_mod_q_shoup,
    const uint64_t* __restrict src_a_p_flat,
    uint64_t* __restrict dst_a_q_eval_flat,
    const uint64_t* __restrict src_b_p_flat,
    uint64_t* __restrict dst_b_q_eval_flat,
    size_t N) {
    const size_t p_limb_count = p_moduli.size();
    const size_t q_limb_count = q_moduli.size();
    if (src_a_p_flat == nullptr || dst_a_q_eval_flat == nullptr ||
        src_b_p_flat == nullptr || dst_b_q_eval_flat == nullptr) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid folded eval-Q ModDown pair pointer");
    }
    if (q_limb_count > q_twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid folded eval-Q ModDown twiddle count");
    }
    const size_t inverse_count = q_limb_count * p_limb_count;
    if (p_inv_mod_q.size() != inverse_count) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid folded eval-Q ModDown pair inverse count");
    }
    if (p_inv_mod_q_shoup.size() != inverse_count) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid folded eval-Q ModDown pair inverse shoup count");
    }
    MyVector<uint64_t> p_contribution_a(N);
    MyVector<uint64_t> p_contribution_b(N);
    MyVector<double> p_modulus_inverse(p_limb_count);
    for (size_t pi = 0; pi < p_limb_count; ++pi) {
        p_modulus_inverse[pi] = 1.0 / static_cast<double>(p_moduli[pi]);
    }
    for (size_t qi = 0; qi < q_limb_count; ++qi) {
        const uint64_t q = q_moduli[qi];
        const BarrettConst* q_barrett = &q_twiddle_ntt[qi].barrett_const;
        const size_t inv_row = qi * p_limb_count;
        const uint64_t* __restrict invs = p_inv_mod_q.data() + inv_row;
        const uint64_t* __restrict inv_shoups = p_inv_mod_q_shoup.data() + inv_row;
        for (size_t coeff = 0; coeff < N; ++coeff) {
            uint64_t coeff_contribution_a = 0;
            uint64_t coeff_contribution_b = 0;
            double alpha_estimate_a = 0.0;
            double alpha_estimate_b = 0.0;
            for (size_t pi = 0; pi < p_limb_count; ++pi) {
                const uint64_t p = p_moduli[pi];
                const uint64_t inv = invs[pi];
                const uint64_t inv_shoup = inv_shoups[pi];
                const uint64_t* __restrict src_a_p = src_a_p_flat + pi * N;
                const uint64_t* __restrict src_b_p = src_b_p_flat + pi * N;
                const uint64_t src_a = reduce_mod_4q(src_a_p[coeff], p);
                const uint64_t src_b = reduce_mod_4q(src_b_p[coeff], p);
                alpha_estimate_a += static_cast<double>(src_a) * p_modulus_inverse[pi];
                alpha_estimate_b += static_cast<double>(src_b) * p_modulus_inverse[pi];
                const uint64_t term_a = mul_mod_shoup(
                    barrett_reduce_u64(src_a, q, q_barrett),
                    inv,
                    inv_shoup,
                    q);
                const uint64_t term_b = mul_mod_shoup(
                    barrett_reduce_u64(src_b, q, q_barrett),
                    inv,
                    inv_shoup,
                    q);
                coeff_contribution_a = add_mod_q(coeff_contribution_a, term_a, q);
                coeff_contribution_b = add_mod_q(coeff_contribution_b, term_b, q);
            }
            const uint64_t hps_alpha_a =
                std::min<uint64_t>(static_cast<uint64_t>(alpha_estimate_a + 0.5), p_limb_count);
            const uint64_t hps_alpha_b =
                std::min<uint64_t>(static_cast<uint64_t>(alpha_estimate_b + 0.5), p_limb_count);
            p_contribution_a[coeff] = sub_mod_q(coeff_contribution_a, hps_alpha_a, q);
            p_contribution_b[coeff] = sub_mod_q(coeff_contribution_b, hps_alpha_b, q);
        }

        const auto& tbl = q_twiddle_ntt[qi];
        ntt_forward_dit2_dispatch(p_contribution_a.data(), N, q, tbl);
        ntt_forward_dit2_dispatch(p_contribution_b.data(), N, q, tbl);

        uint64_t* __restrict dst_a = dst_a_q_eval_flat + qi * N;
        uint64_t* __restrict dst_b = dst_b_q_eval_flat + qi * N;
        for (size_t coeff = 0; coeff < N; ++coeff) {
            dst_a[coeff] = sub_mod_q(dst_a[coeff], p_contribution_a[coeff], q);
            dst_b[coeff] = sub_mod_q(dst_b[coeff], p_contribution_b[coeff], q);
        }
    }
}

void ckks_apply_folded_approx_crt_moddown_eval_q(
    const MyVector<uint64_t>& p_moduli,
    const MyVector<uint64_t>& q_moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& q_twiddle_ntt,
    std::span<const uint64_t> p_inv_mod_q,
    std::span<const uint64_t> p_inv_mod_q_shoup,
    const uint64_t* __restrict src_p_flat,
    uint64_t* __restrict dst_q_eval_flat,
    size_t N) {
    const size_t p_limb_count = p_moduli.size();
    const size_t q_limb_count = q_moduli.size();
    if (src_p_flat == nullptr || dst_q_eval_flat == nullptr) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid folded eval-Q ModDown pointer");
    }
    if (q_limb_count > q_twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid folded eval-Q ModDown twiddle count");
    }
    const size_t inverse_count = q_limb_count * p_limb_count;
    if (p_inv_mod_q.size() != inverse_count) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid folded eval-Q ModDown inverse count");
    }
    if (p_inv_mod_q_shoup.size() != inverse_count) {
        throw std::invalid_argument("ckks_relin_hybrid: invalid folded eval-Q ModDown inverse shoup count");
    }
    thread_local MyVector<uint64_t> tls_p_contribution;
    MyVector<uint64_t>& p_contribution = tls_p_contribution;
    p_contribution.resize(N);
    MyVector<double> p_modulus_inverse(p_limb_count);
    for (size_t pi = 0; pi < p_limb_count; ++pi) {
        p_modulus_inverse[pi] = 1.0 / static_cast<double>(p_moduli[pi]);
    }
    for (size_t qi = 0; qi < q_limb_count; ++qi) {
        const uint64_t q = q_moduli[qi];
        const BarrettConst* q_barrett = &q_twiddle_ntt[qi].barrett_const;
        const size_t inv_row = qi * p_limb_count;
        const uint64_t* __restrict invs = p_inv_mod_q.data() + inv_row;
        const uint64_t* __restrict inv_shoups = p_inv_mod_q_shoup.data() + inv_row;
        for (size_t coeff = 0; coeff < N; ++coeff) {
            uint64_t coeff_contribution = 0;
            double alpha_estimate = 0.0;
            for (size_t pi = 0; pi < p_limb_count; ++pi) {
                const uint64_t p = p_moduli[pi];
                const uint64_t src = reduce_mod_4q(src_p_flat[pi * N + coeff], p);
                alpha_estimate += static_cast<double>(src) * p_modulus_inverse[pi];
                const uint64_t term = mul_mod_shoup(
                    barrett_reduce_u64(src, q, q_barrett),
                    invs[pi],
                    inv_shoups[pi],
                    q);
                coeff_contribution = add_mod_q(coeff_contribution, term, q);
            }
            const uint64_t hps_alpha =
                std::min<uint64_t>(static_cast<uint64_t>(alpha_estimate + 0.5), p_limb_count);
            p_contribution[coeff] = sub_mod_q(coeff_contribution, hps_alpha, q);
        }

        const auto& tbl = q_twiddle_ntt[qi];
        ntt_forward_dit2_dispatch(p_contribution.data(), N, q, tbl);

        uint64_t* __restrict dst_q = dst_q_eval_flat + qi * N;
        for (size_t coeff = 0; coeff < N; ++coeff) {
            dst_q[coeff] = sub_mod_q(dst_q[coeff], p_contribution[coeff], q);
        }
    }
}

void ckks_hybrid_key_switch_hoisted_qp_rotated(
    const CKKSContext& context,
    const CKKSEvalHoistedCiphertextQP& hoisted_key_term,
    std::span<const CKKSCiphertext> switching_key,
    const MyVector<uint64_t>& b_q_eval,
    MyVector<uint64_t>& b_qp_out,
    MyVector<uint64_t>& a_qp_out,
    bool switching_key_montgomery,
    std::span<const size_t> ntt_map) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t level = hoisted_key_term.level;
    const size_t limb_count = hoisted_key_term.q_limb_count;
    const size_t p_limb_count = hoisted_key_term.p_limb_count;
    const size_t total_active_limb_count = limb_count + p_limb_count;
    const size_t active_partition_count = hoisted_key_term.a_partition_qp_eval.size();
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const auto& q_twiddle_ntt = context.getTwiddleNtt();
    const auto& p_twiddle_ntt = context.getPTwiddleNtt();

    if (hoisted_key_term.N != N ||
        level > params.getMaxLevel() ||
        limb_count == 0 ||
        limb_count > q_moduli.size() ||
        p_limb_count != p_moduli.size() ||
        hoisted_key_term.moduli.size() != total_active_limb_count ||
        b_q_eval.size() != N * limb_count) {
        throw std::invalid_argument("ckks_hybrid_keyswitch: invalid hoisted QP rotation input");
    }
    if (ntt_map.size() != N) {
        throw std::invalid_argument("ckks_hybrid_keyswitch: invalid hoisted QP automorphism map");
    }
    if (switching_key.size() < active_partition_count) {
        throw std::runtime_error("ckks_hybrid_keyswitch: insufficient hoisted QP key partitions");
    }
    const size_t key_q_limb_count = validate_hybrid_switch_key_part_shape(
        switching_key.front(),
        N,
        limb_count,
        p_limb_count);

    MyVector<uint64_t> q_moduli_active(q_moduli.begin(), q_moduli.begin() + limb_count);
    b_qp_out.assign(N * total_active_limb_count, uint64_t{0});
    a_qp_out.assign(N * total_active_limb_count, uint64_t{0});

    for (size_t qi = 0; qi < limb_count; ++qi) {
        const uint64_t q = q_moduli_active[qi];
        const auto* barrett = &q_twiddle_ntt[qi].barrett_const;
        uint64_t direct_scalar = 1;
        if (!params.fold_hybrid_keyswitch) {
            for (const uint64_t p : p_moduli) {
                direct_scalar = mul_mod_u64(direct_scalar, p % q, q, barrett);
            }
        }
        const uint64_t* __restrict src = b_q_eval.data() + qi * N;
        uint64_t* __restrict dst = b_qp_out.data() + qi * N;
        if (direct_scalar == 1) {
            std::copy_n(src, N, dst);
        } else {
            for (size_t coeff = 0; coeff < N; ++coeff) {
                dst[coeff] = mul_mod_u64(src[coeff], direct_scalar, q, barrett);
            }
        }
    }

    MyVector<uint64_t> montgomery_neg_inv;
    if (switching_key_montgomery) {
        montgomery_neg_inv.resize(total_active_limb_count);
        for (size_t i = 0; i < limb_count; ++i) {
            montgomery_neg_inv[i] = ckks_montgomery_neg_inverse(q_moduli_active[i]);
        }
        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            montgomery_neg_inv[limb_count + pi] = ckks_montgomery_neg_inverse(p_moduli[pi]);
        }
    }

    bool used_lazy_accumulation = false;
    auto multiply_accumulate_limb = [&](
        const uint64_t* __restrict digit_ptr,
        const uint64_t* __restrict key_a_ptr,
        const uint64_t* __restrict key_b_ptr,
        uint64_t* __restrict acc_a_ptr,
        uint64_t* __restrict acc_b_ptr,
        size_t limb,
        uint64_t modulus,
        const Negacyclic_NTT_Twiddles& table) {
        if (switching_key_montgomery) {
            used_lazy_accumulation |=
                ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                digit_ptr,
                key_a_ptr,
                acc_a_ptr,
                N,
                modulus,
                montgomery_neg_inv[limb],
                table);
            used_lazy_accumulation |=
                ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                digit_ptr,
                key_b_ptr,
                acc_b_ptr,
                N,
                modulus,
                montgomery_neg_inv[limb],
                table);
            return;
        }

        used_lazy_accumulation |=
            ckks_pointwise_multiply_accumulate_normal_key_pair(
                digit_ptr, key_a_ptr, key_b_ptr, acc_a_ptr, acc_b_ptr, N,
                modulus, table, 4);
    };

    for (size_t part = 0; part < active_partition_count; ++part) {
        const auto& digit_qp = hoisted_key_term.a_partition_qp_eval[part];
        if (digit_qp.size() != N * total_active_limb_count) {
            throw std::invalid_argument("ckks_hybrid_keyswitch: invalid hoisted QP partition size");
        }

        const auto& key_part = switching_key[part];
        const size_t part_key_q_limb_count = validate_hybrid_switch_key_part_shape(
            key_part,
            N,
            limb_count,
            p_limb_count);
        if (part_key_q_limb_count != key_q_limb_count) {
            throw std::runtime_error("ckks_hybrid_keyswitch: inconsistent hoisted QP switch-key levels");
        }
        const auto& key_a = key_part.getA();
        const auto& key_b = key_part.getB();

        for (size_t i = 0; i < limb_count; ++i) {
            multiply_accumulate_limb(
                digit_qp.data() + i * N,
                key_a.data() + i * N,
                key_b.data() + i * N,
                a_qp_out.data() + i * N,
                b_qp_out.data() + i * N,
                i,
                q_moduli_active[i],
                q_twiddle_ntt[i]);
        }
        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            const size_t active_idx = limb_count + pi;
            const size_t key_idx = key_q_limb_count + pi;
            multiply_accumulate_limb(
                digit_qp.data() + active_idx * N,
                key_a.data() + key_idx * N,
                key_b.data() + key_idx * N,
                a_qp_out.data() + active_idx * N,
                b_qp_out.data() + active_idx * N,
                active_idx,
                p_moduli[pi],
                p_twiddle_ntt[pi]);
        }
    }

    reduce_lazy_accumulator_pair_if_needed(
        used_lazy_accumulation,
        a_qp_out,
        b_qp_out,
        N,
        std::span<const uint64_t>(q_moduli_active.data(), limb_count),
        std::span<const uint64_t>(p_moduli.data(), p_limb_count));

    apply_ntt_automorphism_to_flat_rns_poly_inplace(
        b_qp_out,
        N,
        total_active_limb_count,
        ntt_map);
    apply_ntt_automorphism_to_flat_rns_poly_inplace(
        a_qp_out,
        N,
        total_active_limb_count,
        ntt_map);
}

void ckks_hybrid_key_switch_qp_from_key_term(
    const CKKSContext& context,
    size_t level,
    const MyVector<uint64_t>& key_term_q_eval,
    std::span<const CKKSCiphertext> switching_key,
    MyVector<uint64_t>& b_qp_out,
    MyVector<uint64_t>& a_qp_out,
    bool switching_key_montgomery) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t limb_count = level + 1;
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const auto& q_twiddle_ntt = context.getTwiddleNtt();
    const auto& p_twiddle_ntt = context.getPTwiddleNtt();
    const size_t dnum = params.getDnum();
    const size_t p_limb_count = p_moduli.size();
    const size_t max_q_limb_count = params.getMaxLevel() + 1;
    const size_t total_active_limb_count = limb_count + p_limb_count;
    const size_t alpha = (max_q_limb_count + dnum - 1) / dnum;
    const size_t active_partition_count = std::min(dnum, (limb_count + alpha - 1) / alpha);

    if (level > params.getMaxLevel() || limb_count > q_moduli.size() ||
        key_term_q_eval.size() != N * limb_count) {
        throw std::invalid_argument("ckks_hybrid_keyswitch: invalid QP key-term input");
    }
    if (switching_key.size() < active_partition_count) {
        throw std::runtime_error("ckks_hybrid_keyswitch: insufficient QP key partitions");
    }
    const size_t key_q_limb_count = validate_hybrid_switch_key_part_shape(
        switching_key.front(),
        N,
        limb_count,
        p_limb_count);

    MyVector<uint64_t> q_moduli_active(q_moduli.begin(), q_moduli.begin() + limb_count);
    thread_local MyVector<uint64_t> tls_qp_key_term_coeff;
    thread_local MyVector<uint64_t> tls_qp_digit_qp;
    MyVector<uint64_t>& key_term_coeff = tls_qp_key_term_coeff;
    key_term_coeff.assign(key_term_q_eval.begin(), key_term_q_eval.end());
    ntt_inverse_rns_flat_inplace(
        key_term_coeff.data(),
        N,
        limb_count,
        q_moduli_active,
        q_twiddle_ntt,
        true);

    MyVector<const ApproxCrtModUpPrecomp*> hybrid_part_modup_precomp;
    hybrid_part_modup_precomp.reserve(active_partition_count);
    for (size_t part = 0; part < active_partition_count; ++part) {
        hybrid_part_modup_precomp.emplace_back(&context.getHybridPartitionModUpPrecomp(level, part));
    }

    b_qp_out.assign(N * total_active_limb_count, uint64_t{0});
    a_qp_out.assign(N * total_active_limb_count, uint64_t{0});
    MyVector<uint64_t>& digit_qp = tls_qp_digit_qp;
    digit_qp.resize(N * total_active_limb_count);
    MyVector<const uint64_t*> src_ptrs;
    src_ptrs.reserve(alpha);
    MyVector<uint64_t*> dst_ptrs;
    dst_ptrs.reserve(total_active_limb_count);

    MyVector<uint64_t> montgomery_neg_inv;
    if (switching_key_montgomery) {
        montgomery_neg_inv.resize(total_active_limb_count);
        for (size_t i = 0; i < limb_count; ++i) {
            montgomery_neg_inv[i] = ckks_montgomery_neg_inverse(q_moduli_active[i]);
        }
        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            montgomery_neg_inv[limb_count + pi] = ckks_montgomery_neg_inverse(p_moduli[pi]);
        }
    }

    bool used_lazy_accumulation = false;
    for (size_t part = 0; part < active_partition_count; ++part) {
        const size_t part_start = part * alpha;
        const size_t part_end = std::min(part_start + alpha, limb_count);
        if (part_start >= part_end) {
            continue;
        }

        const size_t src_count = part_end - part_start;
        src_ptrs.clear();
        for (size_t s = 0; s < src_count; ++s) {
            const size_t global_idx = part_start + s;
            src_ptrs.emplace_back(key_term_coeff.data() + global_idx * N);
        }

        dst_ptrs.clear();
        for (size_t i = 0; i < part_start; ++i) {
            dst_ptrs.emplace_back(digit_qp.data() + i * N);
        }
        for (size_t i = part_end; i < limb_count; ++i) {
            dst_ptrs.emplace_back(digit_qp.data() + i * N);
        }
        for (size_t i = 0; i < p_limb_count; ++i) {
            dst_ptrs.emplace_back(digit_qp.data() + (limb_count + i) * N);
        }

        if (!dst_ptrs.empty()) {
            const auto& part_precomp = *hybrid_part_modup_precomp[part];
            if (part_precomp.src_moduli.size() != src_count ||
                part_precomp.dst_moduli.size() != dst_ptrs.size()) {
                throw std::runtime_error("ckks_hybrid_keyswitch: invalid QP ModUp precompute shape");
            }
            ckks_apply_approx_crt_modup(part_precomp, src_ptrs, dst_ptrs, N);
        }

        const auto& key_part = switching_key[part];
        const size_t part_key_q_limb_count = validate_hybrid_switch_key_part_shape(
            key_part,
            N,
            limb_count,
            p_limb_count);
        if (part_key_q_limb_count != key_q_limb_count) {
            throw std::runtime_error("ckks_hybrid_keyswitch: inconsistent QP switch-key levels");
        }
        const auto& key_a = key_part.getA();
        const auto& key_b = key_part.getB();

        for (size_t i = 0; i < limb_count; ++i) {
            const uint64_t q = q_moduli_active[i];
            const auto& tbl = q_twiddle_ntt[i];
            const bool source_limb = (i >= part_start && i < part_end);
            const uint64_t* __restrict digit_ptr = key_term_q_eval.data() + i * N;
            if (!source_limb) {
                uint64_t* __restrict modup_digit_ptr = digit_qp.data() + i * N;
                ntt_forward_dit2_dispatch(modup_digit_ptr, N, q, tbl, true);
                digit_ptr = modup_digit_ptr;
            }

            if (switching_key_montgomery) {
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                    digit_ptr,
                    key_a.data() + i * N,
                    a_qp_out.data() + i * N,
                    N,
                    q,
                    montgomery_neg_inv[i],
                    tbl);
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                    digit_ptr,
                    key_b.data() + i * N,
                    b_qp_out.data() + i * N,
                    N,
                    q,
                    montgomery_neg_inv[i],
                    tbl);
            } else {
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_normal_key_pair(
                    digit_ptr,
                    key_a.data() + i * N,
                    key_b.data() + i * N,
                    a_qp_out.data() + i * N,
                    b_qp_out.data() + i * N,
                    N,
                    q,
                    tbl,
                    4);
            }
        }

        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            const size_t active_idx = limb_count + pi;
            const auto& tbl = p_twiddle_ntt[pi];
            const uint64_t p = p_moduli[pi];
            uint64_t* __restrict digit_ptr = digit_qp.data() + active_idx * N;
            ntt_forward_dit2_dispatch(digit_ptr, N, p, tbl, true);

            const size_t key_idx = key_q_limb_count + pi;
            if (switching_key_montgomery) {
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                    digit_ptr,
                    key_part.getA().data() + key_idx * N,
                    a_qp_out.data() + active_idx * N,
                    N,
                    p,
                    montgomery_neg_inv[active_idx],
                    tbl);
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                    digit_ptr,
                    key_part.getB().data() + key_idx * N,
                    b_qp_out.data() + active_idx * N,
                    N,
                    p,
                    montgomery_neg_inv[active_idx],
                    tbl);
            } else {
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_normal_key_pair(
                    digit_ptr,
                    key_part.getA().data() + key_idx * N,
                    key_part.getB().data() + key_idx * N,
                    a_qp_out.data() + active_idx * N,
                    b_qp_out.data() + active_idx * N,
                    N,
                    p,
                    tbl,
                    4);
            }
        }
    }

    reduce_lazy_accumulator_pair_if_needed(
        used_lazy_accumulation,
        a_qp_out,
        b_qp_out,
        N,
        std::span<const uint64_t>(q_moduli_active.data(), limb_count),
        std::span<const uint64_t>(p_moduli.data(), p_limb_count));
}

void ckks_hybrid_key_switch_hoisted_inplace(
    const CKKSContext& context,
    const CKKSEvalHoistedCiphertextQP& hoisted_key_term,
    std::span<const CKKSCiphertext> switching_key,
    MyVector<uint64_t>& b_out,
    MyVector<uint64_t>& a_out,
    CKKSHybridKeySwitchCombine b_mode,
    CKKSHybridKeySwitchCombine a_mode,
    bool switching_key_montgomery,
    std::span<const size_t> ntt_map) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t level = hoisted_key_term.level;
    const size_t limb_count = hoisted_key_term.q_limb_count;
    const size_t p_limb_count = hoisted_key_term.p_limb_count;
    const size_t total_active_limb_count = limb_count + p_limb_count;
    const size_t active_partition_count = hoisted_key_term.a_partition_qp_eval.size();
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const auto& q_twiddle_ntt = context.getTwiddleNtt();
    const auto& p_twiddle_ntt = context.getPTwiddleNtt();

    if (hoisted_key_term.N != N ||
        level > params.getMaxLevel() ||
        limb_count == 0 ||
        limb_count > q_moduli.size() ||
        p_limb_count != p_moduli.size() ||
        hoisted_key_term.moduli.size() != total_active_limb_count) {
        throw std::invalid_argument("ckks_hybrid_keyswitch: invalid hoisted key-switch input");
    }
    if (!ntt_map.empty() && ntt_map.size() != N) {
        throw std::invalid_argument("ckks_hybrid_keyswitch: invalid hoisted automorphism map");
    }
    if (switching_key.size() < active_partition_count) {
        throw std::runtime_error("ckks_hybrid_keyswitch: insufficient hoisted key partitions");
    }
    if (b_out.size() != N * limb_count || a_out.size() != N * limb_count) {
        throw std::invalid_argument("ckks_hybrid_keyswitch: invalid hoisted output polynomial size");
    }
    const size_t key_q_limb_count = validate_hybrid_switch_key_part_shape(
        switching_key.front(),
        N,
        limb_count,
        p_limb_count);

    MyVector<uint64_t> q_moduli_active(q_moduli.begin(), q_moduli.begin() + limb_count);
    std::span<const uint64_t> p_inv_mod_q;
    std::span<const uint64_t> p_inv_mod_q_shoup;
    const ApproxCrtModUpPrecomp* p_to_q_precomp = nullptr;
    if (params.fold_hybrid_keyswitch) {
        p_inv_mod_q = context.getFoldedPInvModQForLevel(level);
        p_inv_mod_q_shoup = context.getFoldedPInvModQShoupForLevel(level);
    } else {
        p_to_q_precomp = &context.getPToQModUpPrecomp(level);
        p_inv_mod_q = context.getPInvModQForLevel(level);
        p_inv_mod_q_shoup = context.getPInvModQShoupForLevel(level);
    }

    MyVector<uint64_t> montgomery_neg_inv;
    if (switching_key_montgomery) {
        montgomery_neg_inv.resize(total_active_limb_count);
        for (size_t i = 0; i < limb_count; ++i) {
            montgomery_neg_inv[i] = ckks_montgomery_neg_inverse(q_moduli_active[i]);
        }
        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            montgomery_neg_inv[limb_count + pi] = ckks_montgomery_neg_inverse(p_moduli[pi]);
        }
    }

    thread_local MyVector<uint64_t> tls_hoisted_acc_b_qp;
    thread_local MyVector<uint64_t> tls_hoisted_acc_a_qp;
    MyVector<uint64_t>& acc_b_qp = tls_hoisted_acc_b_qp;
    MyVector<uint64_t>& acc_a_qp = tls_hoisted_acc_a_qp;
    acc_b_qp.assign(N * total_active_limb_count, uint64_t{0});
    acc_a_qp.assign(N * total_active_limb_count, uint64_t{0});

    bool used_lazy_accumulation = false;
    auto multiply_accumulate_limb = [&](
        const uint64_t* __restrict digit_ptr,
        const uint64_t* __restrict key_a_ptr,
        const uint64_t* __restrict key_b_ptr,
        uint64_t* __restrict acc_a_ptr,
        uint64_t* __restrict acc_b_ptr,
        size_t limb,
        uint64_t modulus,
        const Negacyclic_NTT_Twiddles& table) {
        if (switching_key_montgomery) {
            used_lazy_accumulation |=
                ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                digit_ptr,
                key_a_ptr,
                acc_a_ptr,
                N,
                modulus,
                montgomery_neg_inv[limb],
                table);
            used_lazy_accumulation |=
                ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                digit_ptr,
                key_b_ptr,
                acc_b_ptr,
                N,
                modulus,
                montgomery_neg_inv[limb],
                table);
            return;
        }

        used_lazy_accumulation |=
            ckks_pointwise_multiply_accumulate_normal_key_pair(
                digit_ptr, key_a_ptr, key_b_ptr, acc_a_ptr, acc_b_ptr, N,
                modulus, table, 4);
    };

    for (size_t part = 0; part < active_partition_count; ++part) {
        const auto& digit_qp = hoisted_key_term.a_partition_qp_eval[part];
        if (digit_qp.size() != N * total_active_limb_count) {
            throw std::invalid_argument("ckks_hybrid_keyswitch: invalid hoisted partition size");
        }

        const auto& key_part = switching_key[part];
        const size_t part_key_q_limb_count = validate_hybrid_switch_key_part_shape(
            key_part,
            N,
            limb_count,
            p_limb_count);
        if (part_key_q_limb_count != key_q_limb_count) {
            throw std::runtime_error("ckks_hybrid_keyswitch: inconsistent switch-key levels");
        }
        const auto& key_a = key_part.getA();
        const auto& key_b = key_part.getB();

        for (size_t i = 0; i < limb_count; ++i) {
            const uint64_t q = q_moduli_active[i];
            multiply_accumulate_limb(
                digit_qp.data() + i * N,
                key_a.data() + i * N,
                key_b.data() + i * N,
                acc_a_qp.data() + i * N,
                acc_b_qp.data() + i * N,
                i,
                q,
                q_twiddle_ntt[i]);
        }
        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            const size_t active_idx = limb_count + pi;
            const size_t key_idx = key_q_limb_count + pi;
            const uint64_t p = p_moduli[pi];
            multiply_accumulate_limb(
                digit_qp.data() + active_idx * N,
                key_a.data() + key_idx * N,
                key_b.data() + key_idx * N,
                acc_a_qp.data() + active_idx * N,
                acc_b_qp.data() + active_idx * N,
                active_idx,
                p,
                p_twiddle_ntt[pi]);
        }
    }

    reduce_lazy_accumulator_pair_if_needed(
        used_lazy_accumulation,
        acc_a_qp,
        acc_b_qp,
        N,
        std::span<const uint64_t>(q_moduli_active.data(), limb_count),
        std::span<const uint64_t>(p_moduli.data(), p_limb_count));

    auto inverse_ntt_qp_inplace = [&](uint64_t* flat_poly) {
        for (size_t i = 0; i < limb_count; ++i) {
            const auto& tbl = q_twiddle_ntt[i];
            ntt_inverse_dif2_dispatch(
                flat_poly + i * N, N, q_moduli_active[i], tbl, true);
        }
        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            const auto& tbl = p_twiddle_ntt[pi];
            ntt_inverse_dif2_dispatch(
                flat_poly + (limb_count + pi) * N, N, p_moduli[pi], tbl,
                true);
        }
    };

    if (params.fold_hybrid_keyswitch) {
        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            const auto& tbl = p_twiddle_ntt[pi];
            ntt_inverse_dif2_dispatch(
                acc_a_qp.data() + (limb_count + pi) * N, N, p_moduli[pi],
                tbl, true);
            ntt_inverse_dif2_dispatch(
                acc_b_qp.data() + (limb_count + pi) * N, N, p_moduli[pi],
                tbl, true);
        }
        ckks_apply_folded_approx_crt_moddown_pair_eval_q(
            p_moduli,
            q_moduli_active,
            q_twiddle_ntt,
            p_inv_mod_q,
            p_inv_mod_q_shoup,
            acc_a_qp.data() + limb_count * N,
            acc_a_qp.data(),
            acc_b_qp.data() + limb_count * N,
            acc_b_qp.data(),
            N);
    } else {
        inverse_ntt_qp_inplace(acc_a_qp.data());
        inverse_ntt_qp_inplace(acc_b_qp.data());

        MyVector<const uint64_t*> src_p_ptrs;
        src_p_ptrs.reserve(p_limb_count);
        MyVector<uint64_t*> dst_q_ptrs;
        dst_q_ptrs.reserve(limb_count);

        for (size_t i = 0; i < p_limb_count; ++i) {
            src_p_ptrs.emplace_back(acc_a_qp.data() + (limb_count + i) * N);
        }
        for (size_t i = 0; i < limb_count; ++i) {
            dst_q_ptrs.emplace_back(acc_a_qp.data() + i * N);
        }
        ckks_apply_approx_crt_moddown(
            *p_to_q_precomp, src_p_ptrs, dst_q_ptrs, p_inv_mod_q, p_inv_mod_q_shoup, N);

        src_p_ptrs.clear();
        for (size_t i = 0; i < p_limb_count; ++i) {
            src_p_ptrs.emplace_back(acc_b_qp.data() + (limb_count + i) * N);
        }
        dst_q_ptrs.clear();
        for (size_t i = 0; i < limb_count; ++i) {
            dst_q_ptrs.emplace_back(acc_b_qp.data() + i * N);
        }
        ckks_apply_approx_crt_moddown(
            *p_to_q_precomp, src_p_ptrs, dst_q_ptrs, p_inv_mod_q, p_inv_mod_q_shoup, N);

        ntt_forward_rns_flat_inplace(acc_a_qp.data(), N, limb_count, q_moduli_active, q_twiddle_ntt, false);
        ntt_forward_rns_flat_inplace(acc_b_qp.data(), N, limb_count, q_moduli_active, q_twiddle_ntt, false);
    }

    for (size_t i = 0; i < limb_count; ++i) {
            const uint64_t q = q_moduli_active[i];
            uint64_t* __restrict dst_b_ptr = b_out.data() + i * N;
            uint64_t* __restrict dst_a_ptr = a_out.data() + i * N;
            const uint64_t* __restrict add_b_ptr = acc_b_qp.data() + i * N;
            const uint64_t* __restrict add_a_ptr = acc_a_qp.data() + i * N;
            for (size_t coeff = 0; coeff < N; ++coeff) {
                dst_b_ptr[coeff] =
                    (b_mode == CKKSHybridKeySwitchCombine::Add)
                        ? add_mod_q(dst_b_ptr[coeff], add_b_ptr[coeff], q)
                        : add_b_ptr[coeff];
                dst_a_ptr[coeff] =
                    (a_mode == CKKSHybridKeySwitchCombine::Add)
                        ? add_mod_q(dst_a_ptr[coeff], add_a_ptr[coeff], q)
                        : add_a_ptr[coeff];
            }
    }

    apply_ntt_automorphism_to_flat_rns_poly_inplace(b_out, N, limb_count, ntt_map);
    apply_ntt_automorphism_to_flat_rns_poly_inplace(a_out, N, limb_count, ntt_map);
}

void ckks_hybrid_key_switch_inplace(
    const CKKSContext& context,
    size_t level,
    MyVector<uint64_t>& key_term,
    std::span<const CKKSCiphertext> switching_key,
    MyVector<uint64_t>& b_out,
    MyVector<uint64_t>& a_out,
    CKKSHybridKeySwitchCombine b_mode,
    CKKSHybridKeySwitchCombine a_mode,
    bool switching_key_montgomery) {
    const auto& params = context.getParams();
    const auto N = params.getN();
    const size_t limb_count = level + 1;
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const auto& q_twiddle_ntt = context.getTwiddleNtt();
    const auto& p_twiddle_ntt = context.getPTwiddleNtt();
    const size_t dnum = params.getDnum();
    const size_t p_limb_count = p_moduli.size();
    const size_t max_q_limb_count = params.getMaxLevel() + 1;
    const size_t total_active_limb_count = limb_count + p_limb_count;
    const size_t alpha = (max_q_limb_count + dnum - 1) / dnum;
    const size_t active_partition_count = std::min(dnum, (limb_count + alpha - 1) / alpha);

    if (level > params.getMaxLevel() || limb_count > q_moduli.size()) {
        throw std::invalid_argument("ckks_hybrid_keyswitch: invalid ciphertext level");
    }
    if (dnum == 0 || dnum > max_q_limb_count) {
        throw std::invalid_argument("ckks_hybrid_keyswitch: invalid partition count");
    }
    if (key_term.size() != N * limb_count || b_out.size() != N * limb_count || a_out.size() != N * limb_count) {
        throw std::invalid_argument("ckks_hybrid_keyswitch: invalid polynomial buffer size");
    }
    if (switching_key.size() < active_partition_count) {
        throw std::runtime_error("ckks_hybrid_keyswitch: insufficient key partitions");
    }
    const size_t key_q_limb_count = validate_hybrid_switch_key_part_shape(
        switching_key.front(),
        N,
        limb_count,
        p_limb_count);

    MyVector<uint64_t> q_moduli_active(q_moduli.begin(), q_moduli.begin() + limb_count);
    const ApproxCrtModUpPrecomp* p_to_q_precomp = nullptr;
    std::span<const uint64_t> p_inv_mod_q;
    std::span<const uint64_t> p_inv_mod_q_shoup;
    if (params.fold_hybrid_keyswitch) {
        p_inv_mod_q = context.getFoldedPInvModQForLevel(level);
        p_inv_mod_q_shoup = context.getFoldedPInvModQShoupForLevel(level);
    } else {
        p_to_q_precomp = &context.getPToQModUpPrecomp(level);
        p_inv_mod_q = context.getPInvModQForLevel(level);
        p_inv_mod_q_shoup = context.getPInvModQShoupForLevel(level);
    }
    MyVector<uint64_t> montgomery_neg_inv;
    if (switching_key_montgomery) {
        montgomery_neg_inv.resize(total_active_limb_count);
        for (size_t i = 0; i < limb_count; ++i) {
            montgomery_neg_inv[i] = ckks_montgomery_neg_inverse(q_moduli_active[i]);
        }
        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            montgomery_neg_inv[limb_count + pi] = ckks_montgomery_neg_inverse(p_moduli[pi]);
        }
    }
    thread_local MyVector<uint64_t> tls_key_term_eval;
    thread_local MyVector<uint64_t> tls_acc_b_qp;
    thread_local MyVector<uint64_t> tls_acc_a_qp;
    thread_local MyVector<uint64_t> tls_digit_qp;

    MyVector<uint64_t>& key_term_eval = tls_key_term_eval;
    key_term_eval.assign(key_term.begin(), key_term.end());
    ntt_inverse_rns_flat_inplace(key_term.data(), N, limb_count, q_moduli_active, q_twiddle_ntt, true);

    MyVector<const ApproxCrtModUpPrecomp*> hybrid_part_modup_precomp;
    hybrid_part_modup_precomp.reserve(active_partition_count);
    for (size_t part = 0; part < active_partition_count; ++part) {
        hybrid_part_modup_precomp.emplace_back(&context.getHybridPartitionModUpPrecomp(level, part));
    }

    MyVector<uint64_t>& acc_b_qp = tls_acc_b_qp;
    MyVector<uint64_t>& acc_a_qp = tls_acc_a_qp;
    MyVector<uint64_t>& digit_qp = tls_digit_qp;
    acc_b_qp.assign(N * total_active_limb_count, uint64_t{0});
    acc_a_qp.assign(N * total_active_limb_count, uint64_t{0});
    digit_qp.resize(N * total_active_limb_count);
    MyVector<const uint64_t*> src_ptrs;
    src_ptrs.reserve(alpha);
    MyVector<uint64_t*> dst_ptrs;
    dst_ptrs.reserve(total_active_limb_count);

    bool used_lazy_accumulation = false;
    for (size_t part = 0; part < active_partition_count; ++part) {
        const size_t part_start = part * alpha;
        const size_t part_end = std::min(part_start + alpha, limb_count);
        if (part_start >= part_end) {
            continue;
        }

        const size_t src_count = part_end - part_start;
        src_ptrs.clear();
        for (size_t s = 0; s < src_count; ++s) {
            const size_t global_idx = part_start + s;
            src_ptrs.emplace_back(key_term.data() + global_idx * N);
        }

        dst_ptrs.clear();
        for (size_t i = 0; i < part_start; ++i) {
            dst_ptrs.emplace_back(digit_qp.data() + i * N);
        }
        for (size_t i = part_end; i < limb_count; ++i) {
            dst_ptrs.emplace_back(digit_qp.data() + i * N);
        }
        for (size_t i = 0; i < p_limb_count; ++i) {
            dst_ptrs.emplace_back(digit_qp.data() + (limb_count + i) * N);
        }

        if (!dst_ptrs.empty()) {
            const auto& part_precomp = *hybrid_part_modup_precomp[part];
            if (part_precomp.src_moduli.size() != src_count ||
                part_precomp.dst_moduli.size() != dst_ptrs.size()) {
                throw std::runtime_error("ckks_hybrid_keyswitch: invalid partition ModUp precompute shape");
            }
            ckks_apply_approx_crt_modup(part_precomp, src_ptrs, dst_ptrs, N);
        }

        const auto& key_part = switching_key[part];
        const size_t part_key_q_limb_count = validate_hybrid_switch_key_part_shape(
            key_part,
            N,
            limb_count,
            p_limb_count);
        if (part_key_q_limb_count != key_q_limb_count) {
            throw std::runtime_error("ckks_hybrid_keyswitch: inconsistent switch-key levels");
        }
        const auto& key_a = key_part.getA();
        const auto& key_b = key_part.getB();

        for (size_t i = 0; i < limb_count; ++i) {
            const uint64_t q = q_moduli_active[i];
            const auto& tbl = q_twiddle_ntt[i];
            const bool source_limb = (i >= part_start && i < part_end);
            const uint64_t* __restrict digit_ptr = key_term_eval.data() + i * N;
            if (!source_limb) {
                uint64_t* __restrict modup_digit_ptr = digit_qp.data() + i * N;
                ntt_forward_dit2_dispatch(modup_digit_ptr, N, q, tbl, true);
                digit_ptr = modup_digit_ptr;
            }

            const uint64_t* __restrict key_a_ptr = key_a.data() + i * N;
            const uint64_t* __restrict key_b_ptr = key_b.data() + i * N;
            if (switching_key_montgomery) {
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                    digit_ptr, key_a_ptr, acc_a_qp.data() + i * N, N, q,
                    montgomery_neg_inv[i], tbl);
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                    digit_ptr, key_b_ptr, acc_b_qp.data() + i * N, N, q,
                    montgomery_neg_inv[i], tbl);
            } else {
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_normal_key_pair(
                    digit_ptr,
                    key_a_ptr,
                    key_b_ptr,
                    acc_a_qp.data() + i * N,
                    acc_b_qp.data() + i * N,
                    N,
                    q,
                    tbl,
                    4);
            }
        }
        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            const size_t active_idx = limb_count + pi;
            const auto& tbl = p_twiddle_ntt[pi];
            const uint64_t p = p_moduli[pi];
            uint64_t* __restrict digit_ptr = digit_qp.data() + active_idx * N;
            ntt_forward_dit2_dispatch(digit_ptr, N, p, tbl, true);

            const size_t key_idx = key_q_limb_count + pi;
            const uint64_t* __restrict key_a_ptr = key_a.data() + key_idx * N;
            const uint64_t* __restrict key_b_ptr = key_b.data() + key_idx * N;
            if (switching_key_montgomery) {
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                    digit_ptr,
                    key_a_ptr,
                    acc_a_qp.data() + active_idx * N,
                    N,
                    p,
                    montgomery_neg_inv[active_idx],
                    tbl);
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_montgomery_key_lazy(
                    digit_ptr,
                    key_b_ptr,
                    acc_b_qp.data() + active_idx * N,
                    N,
                    p,
                    montgomery_neg_inv[active_idx],
                    tbl);
            } else {
                used_lazy_accumulation |=
                    ckks_pointwise_multiply_accumulate_normal_key_pair(
                    digit_ptr,
                    key_a_ptr,
                    key_b_ptr,
                    acc_a_qp.data() + active_idx * N,
                    acc_b_qp.data() + active_idx * N,
                    N,
                    p,
                    tbl,
                    4);
            }
        }
    }

    reduce_lazy_accumulator_pair_if_needed(
        used_lazy_accumulation,
        acc_a_qp,
        acc_b_qp,
        N,
        std::span<const uint64_t>(q_moduli_active.data(), limb_count),
        std::span<const uint64_t>(p_moduli.data(), p_limb_count));

    auto inverse_ntt_qp_inplace = [&](uint64_t* flat_poly) {
        for (size_t i = 0; i < limb_count; ++i) {
            const auto& tbl = q_twiddle_ntt[i];
            ntt_inverse_dif2_dispatch(
                flat_poly + i * N, N, q_moduli_active[i], tbl, true);
        }
        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            const auto& tbl = p_twiddle_ntt[pi];
            ntt_inverse_dif2_dispatch(
                flat_poly + (limb_count + pi) * N, N, p_moduli[pi], tbl,
                true);
        }
    };

    if (params.fold_hybrid_keyswitch) {
        for (size_t pi = 0; pi < p_limb_count; ++pi) {
            const auto& tbl = p_twiddle_ntt[pi];
            ntt_inverse_dif2_dispatch(
                acc_a_qp.data() + (limb_count + pi) * N, N, p_moduli[pi],
                tbl, true);
            ntt_inverse_dif2_dispatch(
                acc_b_qp.data() + (limb_count + pi) * N, N, p_moduli[pi],
                tbl, true);
        }
        ckks_apply_folded_approx_crt_moddown_pair_eval_q(
            p_moduli,
            q_moduli_active,
            q_twiddle_ntt,
            p_inv_mod_q,
            p_inv_mod_q_shoup,
            acc_a_qp.data() + limb_count * N,
            acc_a_qp.data(),
            acc_b_qp.data() + limb_count * N,
            acc_b_qp.data(),
            N);
    } else {
        inverse_ntt_qp_inplace(acc_a_qp.data());
        inverse_ntt_qp_inplace(acc_b_qp.data());

        MyVector<const uint64_t*> src_p_ptrs;
        src_p_ptrs.reserve(p_limb_count);
        MyVector<uint64_t*> dst_q_ptrs;
        dst_q_ptrs.reserve(limb_count);

        for (size_t i = 0; i < p_limb_count; ++i) {
            src_p_ptrs.emplace_back(acc_a_qp.data() + (limb_count + i) * N);
        }
        for (size_t i = 0; i < limb_count; ++i) {
            dst_q_ptrs.emplace_back(acc_a_qp.data() + i * N);
        }
        ckks_apply_approx_crt_moddown(
            *p_to_q_precomp, src_p_ptrs, dst_q_ptrs, p_inv_mod_q, p_inv_mod_q_shoup, N);

        src_p_ptrs.clear();
        for (size_t i = 0; i < p_limb_count; ++i) {
            src_p_ptrs.emplace_back(acc_b_qp.data() + (limb_count + i) * N);
        }
        dst_q_ptrs.clear();
        for (size_t i = 0; i < limb_count; ++i) {
            dst_q_ptrs.emplace_back(acc_b_qp.data() + i * N);
        }
        ckks_apply_approx_crt_moddown(
            *p_to_q_precomp, src_p_ptrs, dst_q_ptrs, p_inv_mod_q, p_inv_mod_q_shoup, N);

        ntt_forward_rns_flat_inplace(acc_a_qp.data(), N, limb_count, q_moduli_active, q_twiddle_ntt, false);
        ntt_forward_rns_flat_inplace(acc_b_qp.data(), N, limb_count, q_moduli_active, q_twiddle_ntt, false);
    }

    for (size_t i = 0; i < limb_count; ++i) {
            const uint64_t q = q_moduli_active[i];
            uint64_t* __restrict dst_b_ptr = b_out.data() + i * N;
            uint64_t* __restrict dst_a_ptr = a_out.data() + i * N;
            const uint64_t* __restrict add_b_ptr = acc_b_qp.data() + i * N;
            const uint64_t* __restrict add_a_ptr = acc_a_qp.data() + i * N;
            for (size_t coeff = 0; coeff < N; ++coeff) {
                dst_b_ptr[coeff] =
                    (b_mode == CKKSHybridKeySwitchCombine::Add)
                        ? add_mod_q(dst_b_ptr[coeff], add_b_ptr[coeff], q)
                        : add_b_ptr[coeff];
                dst_a_ptr[coeff] =
                    (a_mode == CKKSHybridKeySwitchCombine::Add)
                        ? add_mod_q(dst_a_ptr[coeff], add_a_ptr[coeff], q)
                        : add_a_ptr[coeff];
            }
    }
}

namespace {

uint64_t hps_base_change_single_limb(
    uint64_t value,
    uint64_t src_modulus,
    uint64_t dst_modulus,
    const BarrettConst* dst_barrett) {
    const uint64_t reduced = reduce_mod_4q(value, src_modulus);
    const uint64_t reduced_mod_dst =
        barrett_reduce_u64(reduced, dst_modulus, dst_barrett);
    const uint64_t src_mod_dst =
        barrett_reduce_u64(src_modulus, dst_modulus, dst_barrett);
    const uint64_t half_round_up = (src_modulus >> 1) + 1;
    return (reduced >= half_round_up)
        ? sub_mod_q(reduced_mod_dst, src_mod_dst, dst_modulus)
        : reduced_mod_dst;
}

void ckks_q0p0_key_switch_level0_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    std::span<const CKKSCiphertext> switching_key,
    bool switching_key_montgomery) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const auto& q_twiddle_ntt = context.getTwiddleNtt();
    const auto& p_twiddle_ntt = context.getPTwiddleNtt();
    constexpr size_t compact_q0p0_key_limb_count = 2;

    if (ct.getLevel() != 0) {
        throw std::invalid_argument("ckks_key_switch: q0p0 dense-to-sparse key requires a level-0 ciphertext");
    }
    if (q_moduli.empty() || p_moduli.empty()) {
        throw std::invalid_argument("ckks_key_switch: q0p0 key switch requires Q and P moduli");
    }
    if (ct.getA().size() != N || ct.getB().size() != N) {
        throw std::invalid_argument("ckks_key_switch: invalid level-0 ciphertext buffer size");
    }
    if (switching_key.empty()) {
        throw std::runtime_error("ckks_key_switch: q0p0 switch key is unavailable");
    }

    const auto& key_part = switching_key.front();
    if (key_part.getLevel() != 0) {
        throw std::runtime_error("ckks_key_switch: q0p0 switch key must be level 0");
    }
    const auto& key_a = key_part.getA();
    const auto& key_b = key_part.getB();
    if (key_a.size() != N * compact_q0p0_key_limb_count ||
        key_b.size() != N * compact_q0p0_key_limb_count) {
        throw std::runtime_error("ckks_key_switch: invalid q0p0 switch-key limb count");
    }

    const uint64_t q0 = q_moduli[0];
    const uint64_t p0 = p_moduli[0];
    const auto& q0_tbl = q_twiddle_ntt[0];
    const auto& p0_tbl = p_twiddle_ntt[0];
    constexpr size_t p0_key_limb = 1;

    MyVector<uint64_t> digit_q0_eval = ct.getA();
    MyVector<uint64_t> digit_q0_coeff = digit_q0_eval;
    ntt_inverse_dif2_dispatch(digit_q0_coeff.data(), N, q0, q0_tbl, true);

    MyVector<uint64_t> digit_p0_eval(N);
    for (size_t coeff = 0; coeff < N; ++coeff) {
            digit_p0_eval[coeff] = hps_base_change_single_limb(
                digit_q0_coeff[coeff],
                q0,
                p0,
                &p0_tbl.barrett_const);
    }
    ntt_forward_dit2_dispatch(digit_p0_eval.data(), N, p0, p0_tbl, true);

    MyVector<uint64_t> acc_a_q0(N, uint64_t{0});
    MyVector<uint64_t> acc_b_q0(N, uint64_t{0});
    MyVector<uint64_t> acc_a_p0(N, uint64_t{0});
    MyVector<uint64_t> acc_b_p0(N, uint64_t{0});

    if (switching_key_montgomery) {
        const uint64_t q0_neg_inv = ckks_montgomery_neg_inverse(q0);
        const uint64_t p0_neg_inv = ckks_montgomery_neg_inverse(p0);
        ckks_pointwise_multiply_accumulate_montgomery_key(
            digit_q0_eval.data(), key_a.data(), acc_a_q0.data(), N, q0,
            q0_neg_inv, q0_tbl);
        ckks_pointwise_multiply_accumulate_montgomery_key(
            digit_q0_eval.data(), key_b.data(), acc_b_q0.data(), N, q0,
            q0_neg_inv, q0_tbl);
        ckks_pointwise_multiply_accumulate_montgomery_key(
            digit_p0_eval.data(),
            key_a.data() + p0_key_limb * N,
            acc_a_p0.data(),
            N,
            p0,
            p0_neg_inv,
            p0_tbl);
        ckks_pointwise_multiply_accumulate_montgomery_key(
            digit_p0_eval.data(),
            key_b.data() + p0_key_limb * N,
            acc_b_p0.data(),
            N,
            p0,
            p0_neg_inv,
            p0_tbl);
    } else {
        ckks_pointwise_multiply_accumulate_normal_key_pair(
            digit_q0_eval.data(), key_a.data(), key_b.data(),
            acc_a_q0.data(), acc_b_q0.data(), N, q0, q0_tbl, 4);
        ckks_pointwise_multiply_accumulate_normal_key_pair(
            digit_p0_eval.data(),
            key_a.data() + p0_key_limb * N,
            key_b.data() + p0_key_limb * N,
            acc_a_p0.data(),
            acc_b_p0.data(),
            N,
            p0,
            p0_tbl,
            4);
    }

    ntt_inverse_dif2_dispatch(acc_a_p0.data(), N, p0, p0_tbl, true);
    ntt_inverse_dif2_dispatch(acc_b_p0.data(), N, p0, p0_tbl, true);

    const uint64_t p0_inv_mod_q0 = mod_inverse(p0 % q0, q0);
    const uint64_t p0_inv_shoup_mod_q0 = compute_shoup(p0_inv_mod_q0, q0);

    if (params.fold_hybrid_keyswitch) {
        const MyVector<uint64_t> p0_moduli{p0};
        const MyVector<uint64_t> q0_moduli{q0};
        const MyVector<uint64_t> p0_inv{p0_inv_mod_q0};
        const MyVector<uint64_t> p0_inv_shoup{p0_inv_shoup_mod_q0};
        ckks_apply_folded_approx_crt_moddown_pair_eval_q(
            p0_moduli,
            q0_moduli,
            q_twiddle_ntt,
            p0_inv,
            p0_inv_shoup,
            acc_a_p0.data(),
            acc_a_q0.data(),
            acc_b_p0.data(),
            acc_b_q0.data(),
            N);
    } else {
        ntt_inverse_dif2_dispatch(acc_a_q0.data(), N, q0, q0_tbl, true);
        ntt_inverse_dif2_dispatch(acc_b_q0.data(), N, q0, q0_tbl, true);

        for (size_t coeff = 0; coeff < N; ++coeff) {
            const uint64_t a_modup = hps_base_change_single_limb(
                acc_a_p0[coeff],
                p0,
                q0,
                &q0_tbl.barrett_const);
            const uint64_t b_modup = hps_base_change_single_limb(
                acc_b_p0[coeff],
                p0,
                q0,
                &q0_tbl.barrett_const);
            const uint64_t a_diff = sub_mod_q(reduce_mod_4q(acc_a_q0[coeff], q0), a_modup, q0);
            const uint64_t b_diff = sub_mod_q(reduce_mod_4q(acc_b_q0[coeff], q0), b_modup, q0);
            acc_a_q0[coeff] = mul_mod_shoup(a_diff, p0_inv_mod_q0, p0_inv_shoup_mod_q0, q0);
            acc_b_q0[coeff] = mul_mod_shoup(b_diff, p0_inv_mod_q0, p0_inv_shoup_mod_q0, q0);
        }
        ntt_forward_dit2_dispatch(acc_a_q0.data(), N, q0, q0_tbl);
        ntt_forward_dit2_dispatch(acc_b_q0.data(), N, q0, q0_tbl);
    }

    auto& b_out = ct.getBMutable();
    auto& a_out = ct.getAMutable();
    for (size_t coeff = 0; coeff < N; ++coeff) {
        b_out[coeff] = add_mod_q(b_out[coeff], acc_b_q0[coeff], q0);
        a_out[coeff] = acc_a_q0[coeff];
    }
}

} // namespace

void ckks_key_switch_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    std::span<const CKKSCiphertext> switching_key,
    bool switching_key_montgomery) {
    if (ct.getNumPolys() != 2) {
        throw std::invalid_argument("ckks_key_switch: expected a 2-polynomial ciphertext");
    }
    if (ct.getN() != context.getParams().getN()) {
        throw std::invalid_argument("ckks_key_switch: ciphertext/context size mismatch");
    }

    ckks_hybrid_key_switch_inplace(
        context,
        ct.getLevel(),
        ct.getAMutable(),
        switching_key,
        ct.getBMutable(),
        ct.getAMutable(),
        CKKSHybridKeySwitchCombine::Add,
        CKKSHybridKeySwitchCombine::Assign,
        switching_key_montgomery);
}

void ckks_key_switch_dense_to_sparse_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct) {
    if (ct.getSecretOwner() != CKKSSecretOwner::Dense) {
        throw std::invalid_argument("ckks_key_switch: dense-to-sparse expects a dense-secret ciphertext");
    }
    ckks_q0p0_key_switch_level0_inplace(
        context,
        ct,
        context.getDenseToSparseSwitchKey(),
        context.getParams().montgomery);
    ct.setSecretOwner(CKKSSecretOwner::BootstrapSparse);
}

void ckks_key_switch_sparse_to_dense_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct) {
    if (ct.getSecretOwner() != CKKSSecretOwner::BootstrapSparse) {
        throw std::invalid_argument("ckks_key_switch: sparse-to-dense expects a sparse-secret ciphertext");
    }
    ckks_key_switch_inplace(
        context,
        ct,
        context.getSparseToDenseSwitchKey(),
        context.getParams().montgomery);
    ct.setSecretOwner(CKKSSecretOwner::Dense);
}
