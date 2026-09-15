#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "ckks_bootstrap_dft.hpp"
#include "ckks_encoding.hpp"
#include "ckks_evalmod_scale_plan.hpp"
#include "ckks_context.hpp"
#include "ckks_params.hpp"
#include "conjugation_key_generation.hpp"
#include "fft.hpp"
#include "math_util.hpp"
#include "modarith.hpp"
#include "relinearization_key_generation.hpp"
#include "rotation_key_generation.hpp"
#include "secret_key_generation.hpp"

namespace {
using Complex = std::complex<double>;

constexpr size_t kAesWalshMinimumPackedSegments = 30;

size_t reverse_bits(size_t value, size_t width) {
    size_t reversed = 0;
    for (size_t i = 0; i < width; ++i) {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    return reversed;
}

size_t active_hybrid_partition_count(size_t q_limb_count, size_t max_q_limb_count, size_t dnum) {
    if (dnum == 0) {
        throw std::runtime_error("ckks_context: invalid hybrid partition count");
    }
    const size_t alpha = (max_q_limb_count + dnum - 1) / dnum;
    if (alpha == 0) {
        throw std::runtime_error("ckks_context: invalid hybrid partition width");
    }
    return std::min(dnum, (q_limb_count + alpha - 1) / alpha);
}

size_t modulus_bit_count(uint64_t modulus) {
    if (modulus == 0) {
        return 0;
    }
    return std::numeric_limits<uint64_t>::digits - std::countl_zero(modulus);
}

MyVector<std::size_t> perm_table_generation(std::size_t N) {
    const std::size_t half_N = N >> 1;
    const std::size_t two_N = N << 1;
    const std::size_t width = std::countr_zero(N);

    MyVector<std::size_t> perm_table(half_N);
    std::uint64_t curr = 1;
    for (std::size_t i = 0; i < half_N; i++) {
        perm_table[i] = reverse_bits((curr - 1) >> 1, width);
        curr = (curr * 5) % two_N;
    }
    return perm_table;
}

struct SecretKeyMaterial {
    MyVector<int8_t> secret_key_coeffs;
    MyVector<MyVector<uint64_t>> secret_key;
    MyVector<int8_t> low_secret_key_coeffs;
    MyVector<MyVector<uint64_t>> low_secret_key;
};

struct RotationKeyMaterial {
    MyVector<int64_t> rotation_key_values;
    MyVector<MyVector<CKKSCiphertext>> rotation_keys;
    MyVector<MyVector<size_t>> rotation_ntt_maps;
};

struct ConjugationKeyMaterial {
    MyVector<size_t> conjugation_ntt_map;
    MyVector<CKKSCiphertext> conjugation_key;
};

struct RelinKeyMaterial {
    size_t relin_key_level = 0;
    MyVector<CKKSCiphertext> relin_hybrid_digits;
    MyVector<CKKSCiphertext> dense_to_sparse_switch_key;
    MyVector<CKKSCiphertext> sparse_to_dense_switch_key;
};

struct ModUpPrecomputeMaterial {
    MyVector<MyVector<ApproxCrtModUpPrecomp>> hybrid_partition_modup_precomp_by_level;
    MyVector<ApproxCrtModUpPrecomp> p_to_q_modup_precomp_by_level;
    MyVector<uint64_t> p_inv_mod_q_prefix_by_level;
    MyVector<uint64_t> p_inv_mod_q_shoup_prefix_by_level;
    MyVector<MyVector<uint64_t>> folded_p_inv_mod_q_by_level;
    MyVector<MyVector<uint64_t>> folded_p_inv_mod_q_shoup_by_level;
};

MyVector<Negacyclic_NTT_Twiddles> build_ntt_twiddle_tables(
    size_t N,
    const MyVector<uint64_t>& moduli) {
    MyVector<Negacyclic_NTT_Twiddles> twiddles(moduli.size());
    for (size_t i = 0; i < moduli.size(); ++i) {
        twiddles[i] = generate_negacyclic_ntt_twiddles(N, moduli[i]);
    }
    return twiddles;
}

void validate_hybrid_keyswitch_auxiliary_basis(const CKKSParams& params) {
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const size_t max_level = params.getMaxLevel();
    const size_t q_limb_count = max_level + 1;
    const size_t dnum = params.getDnum();
    if (dnum == 0 || dnum > q_limb_count) {
        throw std::invalid_argument("ckks_context: invalid hybrid key-switch partition count");
    }
    if (p_moduli.empty()) {
        throw std::invalid_argument("ckks_context: hybrid key-switching requires at least one P modulus");
    }

    const size_t alpha = (q_limb_count + dnum - 1) / dnum;
    size_t p_bit_count = 0;
    for (uint64_t p : p_moduli) {
        p_bit_count += modulus_bit_count(p);
    }

    size_t max_digit_q_bit_count = 0;
    for (size_t part = 0; part < dnum; ++part) {
        const size_t part_start = part * alpha;
        const size_t part_end = std::min(part_start + alpha, q_limb_count);
        if (part_start >= part_end) {
            break;
        }
        size_t digit_q_bit_count = 0;
        for (size_t i = part_start; i < part_end; ++i) {
            digit_q_bit_count += modulus_bit_count(q_moduli[i]);
        }
        max_digit_q_bit_count = std::max(max_digit_q_bit_count, digit_q_bit_count);
    }

    if (p_bit_count < max_digit_q_bit_count) {
        throw std::invalid_argument(
            "ckks_context: hybrid key-switch P basis is smaller than the widest Q digit; "
            "dnum = " + std::to_string(dnum) +
            ", P bits = " + std::to_string(p_bit_count) +
            ", widest Q digit bits = " + std::to_string(max_digit_q_bit_count) +
            "; add P limbs or increase dnum");
    }
}

CKKSLinearTransformPlan build_evalmod_s2c_dft_plan(const CKKSContext& context) {
    const auto& params = context.getParams();
    const size_t s2c_depth = params.getBootstrapS2CDepth();
    if (s2c_depth == 0) {
        throw std::runtime_error("ckks_context: EvalMod S2C depth must be positive");
    }

    CKKSBootstrapDftMatrixLiteral literal;
    literal.type = CKKSBootstrapDftType::SlotsToCoeffs;
    literal.level = s2c_depth;
    literal.baby_step_count = params.getBootstrapLinearTransformBsgsBabyStepCounts()[0];
    literal.log_bsgs_ratio = params.getBootstrapS2CLogBsgsRatio();
    literal.stage_count = s2c_depth;
    literal.log_plaintext_scale_groups = params.getBootstrapS2CLogPlaintextScales();
    literal.merge_depths = params.getBootstrapS2CMergeDepths();
    literal.active_slots = params.getBootstrapActiveSlotCount();
    literal.transform_scale =
        (static_cast<double>(params.getModuli()[0]) * 0.5) /
        params.getScale();
    literal.plaintext_scale = static_cast<double>(params.getModuli()[s2c_depth]);
    literal.rescale_after_each_stage = true;
    return ckks_generate_bootstrap_dft_plan(context, literal);
}

double compute_evalmod_scaled_c2s_transform_scale(const CKKSParams& params) {
    return ckks_evalmod_scale_plan(params).binary_c2s_transform_scale;
}

PolynomialApproximation build_evalmod_polynomial(const CKKSParams& params) {
    return binboot_han_ki_discrete_multi_interval_approximate(
        static_cast<double>(params.getBootstrapEvalModK()),
        params.getBootstrapEvalModPolynomialDegree(),
        params.getBootstrapEvalModDoubleAngle(),
        params.getBootstrapEvalModDiscreteIntervalRadius(),
        params.getBootstrapEvalModDiscreteValidationPointsPerInterval(),
        params.getBootstrapEvalModCoefficientPruneThreshold());
}

CKKSLinearTransformPlan build_evalmod_c2s_dft_plan(
    const CKKSContext& context,
    double transform_scale) {
    const auto& params = context.getParams();
    const size_t c2s_depth = params.getBootstrapC2SDepth();
    const size_t level = params.getDepth();
    if (c2s_depth == 0) {
        throw std::runtime_error("ckks_context: EvalMod C2S depth must be positive");
    }
    if (c2s_depth > level) {
        throw std::runtime_error("ckks_context: EvalMod C2S depth exceeds bootstrap depth");
    }

    CKKSBootstrapDftMatrixLiteral literal;
    literal.type = CKKSBootstrapDftType::CoeffsToSlots;
    literal.level = level;
    literal.baby_step_count = params.getBootstrapLinearTransformBsgsBabyStepCounts()[1];
    literal.log_bsgs_ratio = params.getBootstrapC2SLogBsgsRatio();
    literal.stage_count = c2s_depth;
    literal.log_plaintext_scale_groups = params.getBootstrapC2SLogPlaintextScales();
    literal.merge_depths = params.getBootstrapC2SMergeDepths();
    literal.active_slots = params.getBootstrapActiveSlotCount();
    const size_t sparse_trace_factor = params.getSlots() / literal.active_slots;
    literal.transform_scale =
        transform_scale / static_cast<double>(sparse_trace_factor);
    literal.plaintext_scale = static_cast<double>(params.getModuli()[level]) / params.getScale();
    literal.rescale_after_each_stage = true;
    return ckks_generate_bootstrap_dft_plan(context, literal);
}

SecretKeyMaterial build_secret_key_material(
    const CKKSParams& params,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt) {
    const size_t N = params.getN();
    const auto& moduli = params.getModuli();

    SecretKeyMaterial material;
    const size_t secret_key_hamming_weight = params.getSecretKeyHammingWeight();
    material.secret_key_coeffs =
        secret_key_hamming_weight == 0
            ? generate_uniform_terenary_secret(N)
            : generate_hamming_weight_ternary_secret(
                  N,
                  secret_key_hamming_weight);
    material.secret_key = ntt_secret_key(material.secret_key_coeffs, twiddle_ntt, N, moduli);
    if (params.bts_keys) {
        material.low_secret_key_coeffs =
            generate_hamming_weight_ternary_secret(
                N,
                params.getBootstrapSparseSecretHammingWeight());
        material.low_secret_key =
            ntt_secret_key(material.low_secret_key_coeffs, twiddle_ntt, N, moduli);
    }
    return material;
}

template <typename AppendRotation>
void append_linear_transform_plan_rotation_values(
    const CKKSLinearTransformPlan& plan,
    AppendRotation append_rotation) {
    for (const auto& stage : plan.stages) {
        if (stage.baby_step_count == 0) {
            throw std::runtime_error("ckks_context: invalid linear-transform BSGS baby-step count");
        }
        for (size_t term = 0; term < stage.diagonals.size(); ++term) {
            const size_t diagonal_index = stage.diagonal_indices.empty()
                ? term
                : stage.diagonal_indices[term];
            const size_t baby_rotation = diagonal_index % stage.baby_step_count;
            const size_t giant_rotation = diagonal_index - baby_rotation;
            if (baby_rotation != 0) {
                append_rotation(static_cast<int64_t>(baby_rotation));
            }
            if (giant_rotation != 0) {
                append_rotation(static_cast<int64_t>(giant_rotation));
            }
        }
    }
}

template <typename AppendRotation>
void append_aes_walsh_rotation_values(
    const CKKSContext& context,
    AppendRotation append_rotation) {
    const auto& params = context.getParams();
    if (!params.usesAesWalsh()) {
        return;
    }

    const size_t slots = params.getSlots();
    const size_t block_count = params.getAesWalshBlockCount();
    const size_t packed_segment_count = params.getAesWalshPackedSegmentCount();
    if (block_count == 0) {
        throw std::invalid_argument(
            "ckks_context: AES Walsh block count must be positive");
    }
    if (packed_segment_count < kAesWalshMinimumPackedSegments) {
        throw std::invalid_argument(
            "ckks_context: AES Walsh packed segment count must be at least 30");
    }
    if (packed_segment_count > slots / block_count) {
        throw std::invalid_argument(
            "ckks_context: AES Walsh block count exceeds packed slot capacity");
    }

    for (size_t segment = 1; segment < packed_segment_count; ++segment) {
        const auto offset = static_cast<int64_t>(segment * block_count);
        append_rotation(offset);
        append_rotation(-offset);
    }
}

template <typename AppendRotation>
void append_sparse_bootstrap_trace_rotation_values(
    const CKKSContext& context,
    AppendRotation append_rotation) {
    const auto& params = context.getParams();
    if (!params.hasBootstrapParams()) {
        return;
    }
    const size_t active_slots = params.getBootstrapActiveSlotCount();
    const size_t full_slots = params.getSlots();
    for (size_t rotation = active_slots;
         rotation < full_slots;
         rotation <<= 1) {
        append_rotation(static_cast<int64_t>(rotation));
    }
}

size_t count_unique_plan_rotations(
    const CKKSContext& context,
    const CKKSLinearTransformPlan& plan) {
    const size_t N = context.getParams().getN();
    MyVector<int64_t> unique_rotations;

    auto append_unique_rotation = [&](int64_t rotation) {
        const int64_t normalized = normalize_rotation_value(N, rotation);
        if (normalized == 0) {
            return;
        }
        if (std::find(unique_rotations.begin(), unique_rotations.end(), normalized) !=
            unique_rotations.end()) {
            return;
        }
        unique_rotations.emplace_back(normalized);
    };

    append_linear_transform_plan_rotation_values(plan, append_unique_rotation);
    return unique_rotations.size();
}

MyVector<int64_t> build_unique_rotation_values(const CKKSContext& context) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    MyVector<int64_t> unique_rotations;
    unique_rotations.reserve(params.getRotateValues().size());

    auto append_unique_rotation = [&](int64_t rotation) {
        const int64_t normalized = normalize_rotation_value(N, rotation);
        if (normalized == 0) {
            return;
        }
        if (std::find(unique_rotations.begin(), unique_rotations.end(), normalized) !=
            unique_rotations.end()) {
            return;
        }
        unique_rotations.emplace_back(normalized);
    };

    for (int64_t rotation : params.getRotateValues()) {
        append_unique_rotation(rotation);
    }
    if (params.bts_keys || params.s2c_ptxt || params.c2s_ptxt) {
        if ((params.bts_keys || params.s2c_ptxt) &&
            params.getBootstrapS2CDepth() > 0) {
            append_linear_transform_plan_rotation_values(
                context.getEvalModS2CPlan(),
                append_unique_rotation);
        }
        if ((params.bts_keys || params.c2s_ptxt) &&
            params.getBootstrapC2SDepth() > 0) {
            append_linear_transform_plan_rotation_values(
                context.getEvalModScaledC2SPlan(),
                append_unique_rotation);
            append_sparse_bootstrap_trace_rotation_values(
                context,
                append_unique_rotation);
        }
    }
    append_aes_walsh_rotation_values(context, append_unique_rotation);

    return unique_rotations;
}

RotationKeyMaterial build_rotation_key_material(
    const CKKSContext& context,
    const MyVector<int8_t>& secret_key_coeffs) {
    const auto& params = context.getParams();
    const size_t N = params.getN();

    RotationKeyMaterial material;
    material.rotation_key_values = build_unique_rotation_values(context);
    material.rotation_ntt_maps.resize(material.rotation_key_values.size());
    material.rotation_keys.resize(material.rotation_key_values.size());

    #pragma omp parallel for schedule(dynamic) if (material.rotation_key_values.size() > 1)
    for (std::ptrdiff_t idx = 0;
         idx < static_cast<std::ptrdiff_t>(material.rotation_key_values.size());
         ++idx) {
        const int64_t rotation = material.rotation_key_values[static_cast<size_t>(idx)];
        material.rotation_ntt_maps[static_cast<size_t>(idx)] = build_ntt_automorphism_map(
            N,
            context.getTwiddleNtt()[0].log_N,
            rotation_automorphism_index(N, rotation));
        material.rotation_keys[static_cast<size_t>(idx)] =
            generate_rotation_key_hybrid(context, secret_key_coeffs, rotation);
    }

    return material;
}

ConjugationKeyMaterial build_conjugation_key_material(
    const CKKSContext& context,
    const MyVector<int8_t>& secret_key_coeffs) {
    const size_t N = context.getParams().getN();

    ConjugationKeyMaterial material;
    material.conjugation_ntt_map = build_ntt_automorphism_map(
        N,
        context.getTwiddleNtt()[0].log_N,
        conjugation_automorphism_index(N));
    material.conjugation_key = generate_conjugation_key_hybrid(context, secret_key_coeffs);
    return material;
}

RelinKeyMaterial build_relin_key_material(
    const CKKSContext& context,
    const MyVector<int8_t>& secret_key_coeffs,
    const MyVector<int8_t>& low_secret_key_coeffs) {
    const auto& params = context.getParams();
    const size_t max_level = params.getMaxLevel();

    RelinKeyMaterial material;
    material.relin_key_level = max_level;
    const size_t max_q_limb_count = max_level + 1;
    const size_t active_partition_count =
        active_hybrid_partition_count(max_q_limb_count, max_q_limb_count, params.getDnum());

    material.relin_hybrid_digits = encrypt_rlk_hybrid(context, secret_key_coeffs);
    if (material.relin_hybrid_digits.size() != active_partition_count) {
        throw std::runtime_error("ckks_context: unexpected hybrid s^2 relin key count");
    }

    if (params.bts_keys) {
        if (low_secret_key_coeffs.size() != secret_key_coeffs.size()) {
            throw std::runtime_error("ckks_context: sparse secret material is unavailable");
        }
        // The sparse target secret is only used at the lowest q0*p0 modulus.
        material.dense_to_sparse_switch_key =
            generate_secret_key_switch_key_q0p0_hybrid(context, secret_key_coeffs, low_secret_key_coeffs);
        if (material.dense_to_sparse_switch_key.size() != 1) {
            throw std::runtime_error("ckks_context: unexpected dense-to-sparse switch-key count");
        }
        material.sparse_to_dense_switch_key =
            generate_secret_key_switch_key_hybrid(context, low_secret_key_coeffs, secret_key_coeffs);
        if (material.sparse_to_dense_switch_key.size() != active_partition_count) {
            throw std::runtime_error("ckks_context: unexpected sparse-to-dense switch-key count");
        }
    }

    return material;
}

ModUpPrecomputeMaterial build_modup_precompute_material(
    const CKKSParams& params,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt) {
    const auto& moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const size_t max_level = params.getMaxLevel();
    const size_t limb_count = max_level + 1;
    const size_t dnum = params.getDnum();
    const size_t alpha = (limb_count + dnum - 1) / dnum;

    ModUpPrecomputeMaterial material;
    material.hybrid_partition_modup_precomp_by_level.resize(limb_count);
    for (size_t level = 0; level <= max_level; ++level) {
        const size_t level_limb_count = level + 1;
        const size_t active_partition_count =
            active_hybrid_partition_count(level_limb_count, limb_count, dnum);
        auto& level_precomp = material.hybrid_partition_modup_precomp_by_level[level];
        level_precomp.resize(active_partition_count);

        for (size_t part = 0; part < active_partition_count; ++part) {
            const size_t part_start = part * alpha;
            const size_t part_end = std::min(part_start + alpha, level_limb_count);
            if (part_start >= part_end) {
                throw std::runtime_error("ckks_context: empty hybrid partition during ModUp precompute");
            }

            MyVector<uint64_t> src_moduli;
            src_moduli.reserve(part_end - part_start);
            for (size_t i = part_start; i < part_end; ++i) {
                src_moduli.emplace_back(moduli[i]);
            }

            MyVector<uint64_t> dst_moduli;
            dst_moduli.reserve(level_limb_count - (part_end - part_start) + p_moduli.size());
            for (size_t i = 0; i < part_start; ++i) {
                dst_moduli.emplace_back(moduli[i]);
            }
            for (size_t i = part_end; i < level_limb_count; ++i) {
                dst_moduli.emplace_back(moduli[i]);
            }
            for (uint64_t p : p_moduli) {
                dst_moduli.emplace_back(p);
            }

            level_precomp[part] = build_approx_crt_modup_precomp(src_moduli, dst_moduli);
        }
    }

    if (!params.fold_hybrid_keyswitch) {
        material.p_to_q_modup_precomp_by_level.resize(limb_count);
        for (size_t level = 0; level <= max_level; ++level) {
            const size_t level_limb_count = level + 1;
            MyVector<uint64_t> q_moduli_active(moduli.begin(), moduli.begin() + level_limb_count);
            material.p_to_q_modup_precomp_by_level[level] =
                build_approx_crt_modup_precomp(p_moduli, q_moduli_active);
        }

        material.p_inv_mod_q_prefix_by_level.resize(limb_count, uint64_t{0});
        material.p_inv_mod_q_shoup_prefix_by_level.resize(limb_count, uint64_t{0});
        for (size_t i = 0; i < limb_count; ++i) {
            const uint64_t q = moduli[i];
            const auto* q_barrett_ptr = &twiddle_ntt[i].barrett_const;
            uint64_t p_mod_q = 1;
            for (uint64_t p : p_moduli) {
                p_mod_q = mul_mod_u64(p_mod_q, p % q, q, q_barrett_ptr);
            }
            const uint64_t p_inv_mod_q = mod_inverse(p_mod_q, q);
            material.p_inv_mod_q_prefix_by_level[i] = p_inv_mod_q;
            material.p_inv_mod_q_shoup_prefix_by_level[i] = compute_shoup(p_inv_mod_q, q);
        }
    }

    if (params.fold_hybrid_keyswitch) {
        material.folded_p_inv_mod_q_by_level.resize(limb_count);
        material.folded_p_inv_mod_q_shoup_by_level.resize(limb_count);
        for (size_t level = 0; level <= max_level; ++level) {
            const size_t level_limb_count = level + 1;
            auto& level_invs = material.folded_p_inv_mod_q_by_level[level];
            auto& level_shoups = material.folded_p_inv_mod_q_shoup_by_level[level];
            level_invs.resize(level_limb_count * p_moduli.size(), uint64_t{0});
            level_shoups.resize(level_limb_count * p_moduli.size(), uint64_t{0});

            for (size_t qi = 0; qi < level_limb_count; ++qi) {
                const uint64_t q = moduli[qi];
                for (size_t pi = 0; pi < p_moduli.size(); ++pi) {
                    const uint64_t inv = mod_inverse(p_moduli[pi] % q, q);
                    const size_t idx = qi * p_moduli.size() + pi;
                    level_invs[idx] = inv;
                    level_shoups[idx] = compute_shoup(inv, q);
                }
            }
        }
    }

    return material;
}

MyVector<CKKSRescaleConsts> build_rescale_constants(
    const CKKSParams& params,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt) {
    const size_t N = params.getN();
    const auto& moduli = params.getModuli();

    MyVector<CKKSRescaleConsts> constants(moduli.size());
    for (size_t d = 1; d < moduli.size(); ++d) {
        const uint64_t p = moduli[d];
        const uint64_t half_p = p >> 1;
        auto& rc = constants[d];
        rc.one_shoup_mod_pi.resize(d);
        rc.inv_p_mod_pi.resize(d);
        rc.inv_p_shoup_mod_pi.resize(d);
        rc.half_p_ntt_stride = N;
        rc.half_p_ntt_mod_pi_flat.resize(d * N);

        for (size_t i = 0; i < d; ++i) {
            const uint64_t q = moduli[i];
            const uint64_t one_shoup = compute_shoup(1, q);
            const uint64_t p_mod_q = reduce_u64_mod_q_shoup(p, q, one_shoup);
            const uint64_t inv_p = mod_inverse(p_mod_q, q);
            const uint64_t half_mod_q = reduce_u64_mod_q_shoup(half_p, q, one_shoup);

            uint64_t* half_ntt_ptr = rc.half_p_ntt_mod_pi_flat.data() + i * N;
            std::fill(half_ntt_ptr, half_ntt_ptr + N, half_mod_q);
            ntt_forward_dit2_dispatch(half_ntt_ptr, N, q, twiddle_ntt[i]);

            rc.one_shoup_mod_pi[i] = one_shoup;
            rc.inv_p_mod_pi[i] = inv_p;
            rc.inv_p_shoup_mod_pi[i] = compute_shoup(inv_p, q);
        }
    }
    return constants;
}

} // namespace

CKKSContext::CKKSContext(CKKSParams params)
    : ckks_params(std::move(params)) {
    if (ckks_params.bts_keys || ckks_params.s2c_ptxt || ckks_params.c2s_ptxt) {
        (void)ckks_params.getBootstrapParams();
    }
    validate_hybrid_keyswitch_auxiliary_basis(ckks_params);

    twiddle_ifft = twiddle_ifft_generation(ckks_params.getN());
    twiddle_fft.resize(twiddle_ifft.size());
    for (size_t i = 0; i < twiddle_ifft.size(); ++i) {
        twiddle_fft[i] = std::conj(twiddle_ifft[i]);
    }
    perm_table = perm_table_generation(ckks_params.getN());
    twiddle_ntt = build_ntt_twiddle_tables(
        ckks_params.getN(), ckks_params.getModuli());
    p_twiddle_ntt = build_ntt_twiddle_tables(
        ckks_params.getN(), ckks_params.getPModuli());

    const bool needs_evalmod_s2c_plan =
        (ckks_params.bts_keys || ckks_params.s2c_ptxt) &&
        ckks_params.getBootstrapS2CDepth() > 0;
    const bool needs_evalmod_c2s_plan =
        (ckks_params.bts_keys || ckks_params.c2s_ptxt) &&
        ckks_params.getBootstrapC2SDepth() > 0;
    const bool needs_evalmod_polynomial =
        ckks_params.bts_keys &&
        ckks_params.getBootstrapEvalModDepth() > 0;

    if (needs_evalmod_s2c_plan) {
        evalmod_s2c_plan = build_evalmod_s2c_dft_plan(*this);
    }
    if (needs_evalmod_c2s_plan) {
        evalmod_scaled_c2s_transform_scale =
            compute_evalmod_scaled_c2s_transform_scale(ckks_params);
        evalmod_scaled_c2s_plan = build_evalmod_c2s_dft_plan(
            *this,
            evalmod_scaled_c2s_transform_scale);
    }
    evalmod_bootstrap_dft_plans_generated =
        needs_evalmod_s2c_plan || needs_evalmod_c2s_plan;
    if (needs_evalmod_polynomial) {
        evalmod_polynomial = build_evalmod_polynomial(ckks_params);
        evalmod_polynomial_generated = true;
    }

    auto secret_keys = build_secret_key_material(ckks_params, twiddle_ntt);
    secret_key_coeffs = std::move(secret_keys.secret_key_coeffs);
    secret_key = std::move(secret_keys.secret_key);
    low_secret_key_coeffs = std::move(secret_keys.low_secret_key_coeffs);
    low_secret_key = std::move(secret_keys.low_secret_key);

    auto rotation_material = build_rotation_key_material(*this, secret_key_coeffs);
    rotation_key_values = std::move(rotation_material.rotation_key_values);
    rotation_keys = std::move(rotation_material.rotation_keys);
    rotation_ntt_maps = std::move(rotation_material.rotation_ntt_maps);
    rotation_key_index.clear();
    rotation_key_index.reserve(rotation_key_values.size());
    for (size_t i = 0; i < rotation_key_values.size(); ++i) {
        rotation_key_index.emplace(rotation_key_values[i], i);
    }
    rotation_key_count = rotation_key_values.size();
    if (needs_evalmod_s2c_plan) {
        const size_t s2c_rotation_count =
            count_unique_plan_rotations(*this, evalmod_s2c_plan);
        std::cout << "ckks_context rotation keys: total="
                  << rotation_key_count
                  << " s2c=" << s2c_rotation_count;
        if (needs_evalmod_c2s_plan) {
            const size_t c2s_rotation_count =
                count_unique_plan_rotations(*this, evalmod_scaled_c2s_plan);
            std::cout << " c2s=" << c2s_rotation_count;
        }
        std::cout << '\n';
    } else {
        std::cout << "ckks_context rotation keys: total="
                  << rotation_key_count << '\n';
    }
    auto conjugation_material = build_conjugation_key_material(*this, secret_key_coeffs);
    conjugation_ntt_map = std::move(conjugation_material.conjugation_ntt_map);
    conjugation_key = std::move(conjugation_material.conjugation_key);
    conjugation_key_generated = true;

    auto relin_material =
        build_relin_key_material(*this, secret_key_coeffs, low_secret_key_coeffs);
    relin_key_level = relin_material.relin_key_level;
    relin_hybrid_digits = std::move(relin_material.relin_hybrid_digits);
    dense_to_sparse_switch_key = std::move(relin_material.dense_to_sparse_switch_key);
    sparse_to_dense_switch_key = std::move(relin_material.sparse_to_dense_switch_key);

    auto modup_material = build_modup_precompute_material(ckks_params, twiddle_ntt);
    hybrid_partition_modup_precomp_by_level =
        std::move(modup_material.hybrid_partition_modup_precomp_by_level);
    p_to_q_modup_precomp_by_level = std::move(modup_material.p_to_q_modup_precomp_by_level);
    p_inv_mod_q_prefix_by_level = std::move(modup_material.p_inv_mod_q_prefix_by_level);
    p_inv_mod_q_shoup_prefix_by_level =
        std::move(modup_material.p_inv_mod_q_shoup_prefix_by_level);
    folded_p_inv_mod_q_by_level = std::move(modup_material.folded_p_inv_mod_q_by_level);
    folded_p_inv_mod_q_shoup_by_level =
        std::move(modup_material.folded_p_inv_mod_q_shoup_by_level);

    rescale_consts = build_rescale_constants(ckks_params, twiddle_ntt);

}

const CKKSParams &CKKSContext::getParams() const { return ckks_params; }

const MyVector<MyVector<uint64_t>> &CKKSContext::getSecretKey() const {
    return secret_key;
}

const MyVector<MyVector<uint64_t>>& CKKSContext::getLowSecretKey() const {
    if (!ckks_params.bts_keys || low_secret_key.empty()) {
        throw std::runtime_error("ckks_context: sparse bootstrap secret key is unavailable");
    }
    return low_secret_key;
}

std::span<const CKKSCiphertext> CKKSContext::getRelinHybrid(size_t level) const {
    if (level > ckks_params.getMaxLevel()) {
        throw std::invalid_argument("ckks_context: invalid hybrid relin level");
    }
    if (level > relin_key_level) {
        throw std::invalid_argument("ckks_context: hybrid relin key level is insufficient");
    }
    const size_t active_partition_count = active_hybrid_partition_count(
        ckks_params.getMaxLevel() + 1,
        ckks_params.getMaxLevel() + 1,
        ckks_params.getDnum());
    if (relin_hybrid_digits.size() < active_partition_count) {
        throw std::runtime_error("ckks_context: requested hybrid relin key is unavailable");
    }
    return std::span<const CKKSCiphertext>(
        relin_hybrid_digits.data(),
        active_partition_count);
}

bool CKKSContext::hasSparseSecretEncapsulationKeys() const {
    const size_t active_partition_count = active_hybrid_partition_count(
        ckks_params.getMaxLevel() + 1,
        ckks_params.getMaxLevel() + 1,
        ckks_params.getDnum());
    return ckks_params.bts_keys &&
           !dense_to_sparse_switch_key.empty() &&
           sparse_to_dense_switch_key.size() >= active_partition_count;
}

std::span<const CKKSCiphertext> CKKSContext::getDenseToSparseSwitchKey() const {
    if (dense_to_sparse_switch_key.empty()) {
        throw std::runtime_error("ckks_context: dense-to-sparse switch key is unavailable");
    }
    return std::span<const CKKSCiphertext>(dense_to_sparse_switch_key.data(), 1);
}

std::span<const CKKSCiphertext> CKKSContext::getSparseToDenseSwitchKey() const {
    const size_t active_partition_count = active_hybrid_partition_count(
        ckks_params.getMaxLevel() + 1,
        ckks_params.getMaxLevel() + 1,
        ckks_params.getDnum());
    if (sparse_to_dense_switch_key.size() < active_partition_count) {
        throw std::runtime_error("ckks_context: sparse-to-dense switch key is unavailable");
    }
    return std::span<const CKKSCiphertext>(
        sparse_to_dense_switch_key.data(),
        active_partition_count);
}

bool CKKSContext::hasRotationKey(int64_t rotation) const {
    const int64_t normalized = normalize_rotation_value(ckks_params.getN(), rotation);
    if (normalized == 0) {
        return true;
    }
    return rotation_key_index.find(normalized) != rotation_key_index.end();
}

bool CKKSContext::hasConjugationKey() const {
    return conjugation_key_generated;
}

std::span<const CKKSCiphertext> CKKSContext::getConjugationKey() const {
    if (!conjugation_key_generated) {
        throw std::runtime_error("ckks_context: conjugation key is unavailable");
    }
    return std::span<const CKKSCiphertext>(conjugation_key.data(), conjugation_key.size());
}

const MyVector<size_t>& CKKSContext::getConjugationNttMap() const {
    if (!conjugation_key_generated) {
        throw std::runtime_error("ckks_context: conjugation map is unavailable");
    }
    return conjugation_ntt_map;
}

size_t CKKSContext::getRotationKeyCount() const {
    return rotation_key_count;
}

std::span<const CKKSCiphertext> CKKSContext::getRotationKey(int64_t rotation) const {
    const int64_t normalized = normalize_rotation_value(ckks_params.getN(), rotation);
    if (normalized == 0) {
        return std::span<const CKKSCiphertext>();
    }

    const auto it = rotation_key_index.find(normalized);
    if (it != rotation_key_index.end()) {
        const size_t i = it->second;
        return std::span<const CKKSCiphertext>(
            rotation_keys[i].data(),
            rotation_keys[i].size());
    }

    throw std::invalid_argument("ckks_context: rotation key is unavailable");
}

const MyVector<size_t>& CKKSContext::getRotationNttMap(int64_t rotation) const {
    const int64_t normalized = normalize_rotation_value(ckks_params.getN(), rotation);
    if (normalized == 0) {
        throw std::invalid_argument("ckks_context: rotation map is undefined for zero rotation");
    }

    const auto it = rotation_key_index.find(normalized);
    if (it != rotation_key_index.end()) {
        return rotation_ntt_maps[it->second];
    }

    throw std::invalid_argument("ckks_context: rotation map is unavailable");
}

const ApproxCrtModUpPrecomp& CKKSContext::getHybridPartitionModUpPrecomp(size_t level, size_t partition) const {
    if (level > ckks_params.getMaxLevel()) {
        throw std::invalid_argument("ckks_context: invalid hybrid partition ModUp level");
    }
    if (level > relin_key_level) {
        throw std::invalid_argument("ckks_context: hybrid partition ModUp level is insufficient");
    }
    if (level >= hybrid_partition_modup_precomp_by_level.size()) {
        throw std::runtime_error("ckks_context: hybrid partition ModUp precompute is unavailable");
    }
    const auto& level_precomp = hybrid_partition_modup_precomp_by_level[level];
    if (partition >= level_precomp.size()) {
        throw std::invalid_argument("ckks_context: invalid hybrid partition index");
    }
    return level_precomp[partition];
}

const ApproxCrtModUpPrecomp& CKKSContext::getPToQModUpPrecomp(size_t level) const {
    if (level > ckks_params.getMaxLevel()) {
        throw std::invalid_argument("ckks_context: invalid P->Q ModUp level");
    }
    if (level > relin_key_level) {
        throw std::invalid_argument("ckks_context: P->Q ModUp level is insufficient");
    }
    if (level >= p_to_q_modup_precomp_by_level.size()) {
        throw std::runtime_error("ckks_context: P->Q ModUp precompute is unavailable");
    }
    return p_to_q_modup_precomp_by_level[level];
}

std::span<const uint64_t> CKKSContext::getPInvModQForLevel(size_t level) const {
    if (level > ckks_params.getMaxLevel()) {
        throw std::invalid_argument("ckks_context: invalid P inverse level");
    }
    if (level > relin_key_level) {
        throw std::invalid_argument("ckks_context: P inverse level is insufficient");
    }
    const size_t limb_count = level + 1;
    if (limb_count > p_inv_mod_q_prefix_by_level.size()) {
        throw std::runtime_error("ckks_context: P inverse prefix is unavailable");
    }
    return std::span<const uint64_t>(p_inv_mod_q_prefix_by_level.data(), limb_count);
}

std::span<const uint64_t> CKKSContext::getPInvModQShoupForLevel(size_t level) const {
    if (level > ckks_params.getMaxLevel()) {
        throw std::invalid_argument("ckks_context: invalid P inverse shoup level");
    }
    if (level > relin_key_level) {
        throw std::invalid_argument("ckks_context: P inverse shoup level is insufficient");
    }
    const size_t limb_count = level + 1;
    if (limb_count > p_inv_mod_q_shoup_prefix_by_level.size()) {
        throw std::runtime_error("ckks_context: P inverse shoup prefix is unavailable");
    }
    return std::span<const uint64_t>(p_inv_mod_q_shoup_prefix_by_level.data(), limb_count);
}

std::span<const uint64_t> CKKSContext::getFoldedPInvModQForLevel(size_t level) const {
    if (level > ckks_params.getMaxLevel()) {
        throw std::invalid_argument("ckks_context: invalid folded P inverse level");
    }
    if (level > relin_key_level) {
        throw std::invalid_argument("ckks_context: folded P inverse level is insufficient");
    }
    if (level >= folded_p_inv_mod_q_by_level.size()) {
        throw std::runtime_error("ckks_context: folded P inverse table is unavailable");
    }
    const size_t required_count = (level + 1) * ckks_params.getPModuli().size();
    const auto& level_invs = folded_p_inv_mod_q_by_level[level];
    if (level_invs.size() != required_count) {
        throw std::runtime_error("ckks_context: invalid folded P inverse table shape");
    }
    return std::span<const uint64_t>(level_invs.data(), level_invs.size());
}

std::span<const uint64_t> CKKSContext::getFoldedPInvModQShoupForLevel(size_t level) const {
    if (level > ckks_params.getMaxLevel()) {
        throw std::invalid_argument("ckks_context: invalid folded P inverse shoup level");
    }
    if (level > relin_key_level) {
        throw std::invalid_argument("ckks_context: folded P inverse shoup level is insufficient");
    }
    if (level >= folded_p_inv_mod_q_shoup_by_level.size()) {
        throw std::runtime_error("ckks_context: folded P inverse shoup table is unavailable");
    }
    const size_t required_count = (level + 1) * ckks_params.getPModuli().size();
    const auto& level_shoups = folded_p_inv_mod_q_shoup_by_level[level];
    if (level_shoups.size() != required_count) {
        throw std::runtime_error("ckks_context: invalid folded P inverse shoup table shape");
    }
    return std::span<const uint64_t>(level_shoups.data(), level_shoups.size());
}

bool CKKSContext::hasEvalModBootstrapDftPlans() const {
    return evalmod_bootstrap_dft_plans_generated;
}

const CKKSLinearTransformPlan& CKKSContext::getEvalModS2CPlan() const {
    if (!evalmod_bootstrap_dft_plans_generated || evalmod_s2c_plan.stages.empty()) {
        throw std::runtime_error("ckks_context: EvalMod S2C DFT plan cache is unavailable");
    }
    return evalmod_s2c_plan;
}

const CKKSLinearTransformPlan& CKKSContext::getEvalModScaledC2SPlan() const {
    if (!evalmod_bootstrap_dft_plans_generated || evalmod_scaled_c2s_plan.stages.empty()) {
        throw std::runtime_error("ckks_context: EvalMod scaled C2S DFT plan cache is unavailable");
    }
    return evalmod_scaled_c2s_plan;
}

double CKKSContext::getEvalModScaledC2STransformScale() const {
    if (!evalmod_bootstrap_dft_plans_generated || evalmod_scaled_c2s_plan.stages.empty()) {
        throw std::runtime_error("ckks_context: EvalMod scaled C2S transform scale is unavailable");
    }
    return evalmod_scaled_c2s_transform_scale;
}

const PolynomialApproximation& CKKSContext::getEvalModPolynomial() const {
    if (!evalmod_polynomial_generated) {
        throw std::runtime_error("ckks_context: EvalMod polynomial cache is unavailable");
    }
    return evalmod_polynomial;
}

const MyVector<Complex> &CKKSContext::getTwiddleIfft() const {
    return twiddle_ifft;
}

const MyVector<Complex> &CKKSContext::getTwiddleFft() const {
    return twiddle_fft;
}

const MyVector<Negacyclic_NTT_Twiddles> &CKKSContext::getTwiddleNtt() const {
    return twiddle_ntt;
}

const MyVector<Negacyclic_NTT_Twiddles> &CKKSContext::getPTwiddleNtt() const {
    return p_twiddle_ntt;
}

const MyVector<CKKSRescaleConsts> &CKKSContext::getRescaleConsts() const {
    return rescale_consts;
}

const MyVector<std::size_t> &CKKSContext::getPermTable() const {
    return perm_table;
}
