#include "ckks_params.hpp"
#include "ckks_aes_params.hpp"
#include "binboot_k_bound.hpp"
#include "random.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

CKKSParams::CKKSParams(CKKSParamsConfig config) {
    if (!ckks_params_detail::is_power_of_two(config.N)) {
        throw std::invalid_argument("CKKSParams: N must be a nonzero power of two");
    }
    if (config.slots != 0 && config.slots * 2 != config.N) {
        throw std::invalid_argument("CKKSParams: slots must equal N / 2");
    }
    if (config.q_bits.empty() && config.q_bit == 0) {
        throw std::invalid_argument("CKKSParams: q_bit must be positive when q_bits is empty");
    }
    if (!config.p_bits.empty()) {
        for (const uint64_t bit_count : config.p_bits) {
            if (bit_count == 0) {
                throw std::invalid_argument("CKKSParams: p_bits entries must be positive");
            }
        }
    } else if (config.pmoduli_count > 0 && config.p_bit == 0) {
        throw std::invalid_argument("CKKSParams: p_bit must be positive when P limbs are requested");
    }
    if (config.dnum == 0) {
        throw std::invalid_argument("CKKSParams: dnum must be positive");
    }
    if (!config.q_bits.empty() && !config.q_mult_bits.empty()) {
        throw std::invalid_argument(
            "CKKSParams: q_bits and q_mult_bits are mutually exclusive; "
            "q_bits overrides the full chain, q_mult_bits only overrides mult limbs");
    }
    if (!config.q_mult_bits.empty()) {
        if (config.q_mult_bits.size() != config.mult_depth) {
            throw std::invalid_argument(
                "CKKSParams: q_mult_bits size must equal mult_depth");
        }
        for (const uint64_t bit_count : config.q_mult_bits) {
            if (bit_count == 0) {
                throw std::invalid_argument(
                    "CKKSParams: q_mult_bits entries must be positive");
            }
        }
    }
    secure = config.secure;
    bts_keys = config.bts_keys;
    s2c_ptxt = config.s2c_ptxt.value_or(bts_keys);
    c2s_ptxt = config.c2s_ptxt.value_or(bts_keys);
    fold_hybrid_keyswitch = config.fold_hybrid_keyswitch;
    montgomery = config.montgomery;
    aes_walsh_params = config.aes_walsh_params;

    N = config.N;
    slots = config.slots == 0 ? N / 2 : config.slots;
    mult_depth = config.mult_depth;
    q0_bit = config.q0_bit;
    q_bit = config.q_bit;
    q_mult_bits = std::move(config.q_mult_bits);
    dnum = config.dnum;
    p_bits = config.p_bits.empty()
        ? std::vector<uint64_t>(config.pmoduli_count, static_cast<uint64_t>(config.p_bit))
        : std::move(config.p_bits);
    secret_key_hamming_weight = config.secret_key_hamming_weight;
    sigma = config.sigma;
    scale = config.scale > 0.0
        ? config.scale
        : std::exp2(static_cast<double>(q_bit));
    rotate_values = std::move(config.rotate_values);
    bootstrap_params = std::move(config.bootstrap_params);

    if (!config.q_bits.empty()) {
        q_bits = std::move(config.q_bits);
        if (q_bits.empty()) {
            throw std::invalid_argument("CKKSParams: q_bits must not be empty");
        }
        depth = q_bits.size() - 1;
        if (config.depth != 0 && config.depth != depth) {
            throw std::invalid_argument("CKKSParams: depth must match q_bits.size() - 1");
        }
        limbs_count = q_bits.size();
    } else {
        depth = config.depth == 0 ? calculateDepth() : config.depth;
        limbs_count = depth + 1;
        q_bits = defaultQBits();
    }

    if (q_bits.size() != limbs_count) {
        throw std::runtime_error("CKKSParams: Q bit schedule must match limb count");
    }
    if (depth != calculateDepth()) {
        throw std::runtime_error(
            "CKKSParams: depth must equal mult_depth plus requested bootstrap/S2C depths");
    }
    if (secure) {
        validate_homomorphic_standard_log_pq(
            N,
            q_bits,
            p_bits,
            secret_key_hamming_weight);
    }

    const auto [p_primes, q_primes] = find_p_and_q_primes(p_bits, q_bits, N);
    moduli = MyVector<uint64_t>(q_primes.begin(), q_primes.end());
    pmoduli = MyVector<uint64_t>(p_primes.begin(), p_primes.end());
}

size_t CKKSParams::getN() const{
    return N;
}

size_t CKKSParams::getSlots() const{
    if (slots == 0 || slots * 2 != N) {
        throw std::runtime_error("CKKSParams: slots must equal N / 2");
    }
    return slots;
}

double CKKSParams::getScale() const{
    return scale;
}

double CKKSParams::getSigma() const{
    return sigma;
}

size_t CKKSParams::getSecretKeyHammingWeight() const {
    return secret_key_hamming_weight;
}

const MyVector<uint64_t>& CKKSParams::getModuli() const{
    return moduli;
}

const std::vector<uint64_t>& CKKSParams::getQBits() const {
    return q_bits;
}

size_t CKKSParams::getQ0Bit() const {
    return q0_bit;
}

size_t CKKSParams::getDefaultQBit() const {
    return q_bit;
}

size_t CKKSParams::getMaxLevel() const{
    if (limbs_count == 0) {
        throw std::runtime_error("CKKSParams: limbs_count must be positive");
    }
    if (depth + 1 != limbs_count) {
        throw std::runtime_error("CKKSParams: depth must equal limbs_count - 1");
    }
    const size_t expected_depth = calculateDepth();
    if (depth != expected_depth) {
        throw std::runtime_error(
            "CKKSParams: depth must equal mult_depth plus requested bootstrap/S2C depths");
    }
    return depth;
}

size_t CKKSParams::getDnum() const {
    return dnum;
}

size_t CKKSParams::getDepth() const {
    return getMaxLevel();
}

const MyVector<uint64_t>& CKKSParams::getPModuli() const{
    return pmoduli;
}

const MyVector<int64_t>& CKKSParams::getRotateValues() const {
    return rotate_values;
}

bool CKKSParams::usesAesWalsh() const {
    return aes_walsh_params.enabled;
}

size_t CKKSParams::getAesWalshBlockCount() const {
    return aes_walsh_params.block_count;
}

size_t CKKSParams::getAesWalshPackedSegmentCount() const {
    if (aes_walsh_params.enabled &&
        aes_walsh_params.packed_segment_count != 0) {
        return aes_walsh_params.packed_segment_count;
    }
    const size_t block_count = getAesWalshBlockCount();
    if (block_count == 0) {
        return 0;
    }
    if (aes_walsh_params.packed_segment_count != 0) {
        return aes_walsh_params.packed_segment_count;
    }
    return getSlots() / block_count;
}

size_t CKKSParams::bootstrapDepth(const CKKSBootstrapParams& params) const {
    return ckks_params_detail::bootstrap_linear_transform_depth(params, 0) +
           ckks_params_detail::bootstrap_linear_transform_depth(params, 1) +
           params.evalmod_depth;
}

std::vector<uint64_t> CKKSParams::defaultQBits() const {
    std::vector<uint64_t> bits;
    bits.reserve(limbs_count);
    bits.emplace_back(static_cast<uint64_t>(q0_bit));

    if (bts_keys) {
        const auto& bootstrap = requireBootstrapParams();
        ckks_params_detail::append_dft_group_bits_bottom_to_top(
            bits,
            bootstrap.linear_transform_log_plaintext_scales[0],
            static_cast<uint64_t>(q_bit),
            ckks_params_detail::bootstrap_linear_transform_depth(bootstrap, 0));
        if (q_mult_bits.empty()) {
            bits.insert(bits.end(), mult_depth, static_cast<uint64_t>(q_bit));
        } else {
            bits.insert(bits.end(), q_mult_bits.begin(), q_mult_bits.end());
        }
        const uint64_t evalmod_bit_size =
            bootstrap.evalmod_bit_size == 0
                ? static_cast<uint64_t>(q_bit)
                : static_cast<uint64_t>(bootstrap.evalmod_bit_size);
        bits.insert(bits.end(), bootstrap.evalmod_depth, evalmod_bit_size);
        ckks_params_detail::append_dft_group_bits_bottom_to_top(
            bits,
            bootstrap.linear_transform_log_plaintext_scales[1],
            static_cast<uint64_t>(q_bit),
            ckks_params_detail::bootstrap_linear_transform_depth(bootstrap, 1));
    } else if (s2c_ptxt || c2s_ptxt) {
        const auto& bootstrap = requireBootstrapParams();
        if (s2c_ptxt) {
            ckks_params_detail::append_dft_group_bits_bottom_to_top(
                bits,
                bootstrap.linear_transform_log_plaintext_scales[0],
                static_cast<uint64_t>(q_bit),
                ckks_params_detail::bootstrap_linear_transform_depth(bootstrap, 0));
        }
        if (q_mult_bits.empty()) {
            bits.insert(bits.end(), mult_depth, static_cast<uint64_t>(q_bit));
        } else {
            bits.insert(bits.end(), q_mult_bits.begin(), q_mult_bits.end());
        }
        if (c2s_ptxt) {
            ckks_params_detail::append_dft_group_bits_bottom_to_top(
                bits,
                bootstrap.linear_transform_log_plaintext_scales[1],
                static_cast<uint64_t>(q_bit),
                ckks_params_detail::bootstrap_linear_transform_depth(bootstrap, 1));
        }
    } else {
        if (q_mult_bits.empty()) {
            bits.insert(bits.end(), mult_depth, static_cast<uint64_t>(q_bit));
        } else {
            bits.insert(bits.end(), q_mult_bits.begin(), q_mult_bits.end());
        }
    }
    return bits;
}

size_t CKKSParams::calculateDepth() const {
    size_t total_depth = mult_depth;
    if (bts_keys) {
        total_depth += bootstrapDepth(requireBootstrapParams());
    } else if (s2c_ptxt || c2s_ptxt) {
        const auto& bootstrap = requireBootstrapParams();
        if (s2c_ptxt) {
            total_depth += ckks_params_detail::bootstrap_linear_transform_depth(
                bootstrap,
                0);
        }
        if (c2s_ptxt) {
            total_depth += ckks_params_detail::bootstrap_linear_transform_depth(
                bootstrap,
                1);
        }
    }
    return total_depth;
}

const CKKSBootstrapParams& CKKSParams::requireBootstrapParams() const {
    if (!bts_keys && !s2c_ptxt && !c2s_ptxt) {
        throw std::runtime_error(
            "CKKSParams: bootstrap params are disabled because bts_keys, s2c_ptxt, and c2s_ptxt are false");
    }
    if (!bootstrap_params.has_value()) {
        throw std::runtime_error("CKKSParams: bootstrap parameters are required");
    }
    return *bootstrap_params;
}

bool CKKSParams::hasBootstrapParams() const {
    if (!bts_keys && !s2c_ptxt && !c2s_ptxt) {
        return false;
    }
    requireBootstrapParams();
    return bootstrap_params.has_value();
}

const CKKSBootstrapParams& CKKSParams::getBootstrapParams() const {
    return requireBootstrapParams();
}

size_t CKKSParams::getBootstrapActiveSlotCount() const {
    const size_t full_slots = getSlots();
    const size_t active_slots = requireBootstrapParams().active_slot_count;
    if (active_slots == 0) {
        return full_slots;
    }
    if (!ckks_params_detail::is_power_of_two(active_slots)) {
        throw std::runtime_error(
            "CKKSParams: bootstrap active slot count must be a power of two");
    }
    if (active_slots > full_slots || full_slots % active_slots != 0) {
        throw std::runtime_error(
            "CKKSParams: bootstrap active slot count must divide N/2 slots");
    }
    return active_slots;
}

bool CKKSParams::usesSparseBootstrapSlots() const {
    return getBootstrapActiveSlotCount() != getSlots();
}

size_t CKKSParams::getBootstrapSparseSecretHammingWeight() const {
    return requireBootstrapParams().sparse_secret_hamming_weight;
}

size_t CKKSParams::getBootstrapS2CDepth() const {
    return ckks_params_detail::bootstrap_linear_transform_depth(requireBootstrapParams(), 0);
}

size_t CKKSParams::getBootstrapC2SDepth() const {
    return ckks_params_detail::bootstrap_linear_transform_depth(requireBootstrapParams(), 1);
}

const CKKSBootstrapDftScaleSchedule& CKKSParams::getBootstrapS2CLogPlaintextScales() const {
    return requireBootstrapParams().linear_transform_log_plaintext_scales[0];
}

const CKKSBootstrapDftScaleSchedule& CKKSParams::getBootstrapC2SLogPlaintextScales() const {
    return requireBootstrapParams().linear_transform_log_plaintext_scales[1];
}

size_t CKKSParams::getBootstrapEvalModDepth() const {
    return requireBootstrapParams().evalmod_depth;
}

size_t CKKSParams::getBootstrapEvalModK() const {
    const auto& bootstrap = requireBootstrapParams();
    const size_t k = bootstrap.evalmod_k;
    if (k != 0) {
        return k;
    }
    const long double failure_log2 =
        bootstrap.evalmod_k_failure_probability_log2;
    if (!std::isfinite(failure_log2) || failure_log2 >= 0.0L) {
        throw std::runtime_error(
            "CKKSParams: automatic EvalMod K failure probability log2 must be finite and negative");
    }
    const auto k_bound = required_binboot_k_bound(
        N,
        getBootstrapSparseSecretHammingWeight(),
        std::exp2(failure_log2));
    return k_bound.evalmod_k;
}

size_t CKKSParams::getBootstrapEvalModPolynomialDegree() const {
    const size_t degree = requireBootstrapParams().evalmod_polynomial_degree;
    if (degree == 0) {
        throw std::runtime_error("CKKSParams: EvalMod polynomial degree must be positive");
    }
    return degree;
}

size_t CKKSParams::getBootstrapEvalModDoubleAngle() const {
    return requireBootstrapParams().evalmod_double_angle;
}

size_t CKKSParams::getBootstrapEvalModLogMessageRatio() const {
    return requireBootstrapParams().evalmod_log_message_ratio;
}

double CKKSParams::getBootstrapEvalModCoefficientPruneThreshold() const {
    const double threshold = requireBootstrapParams().evalmod_coefficient_prune_threshold;
    if (!std::isfinite(threshold) || threshold < 0.0) {
        throw std::runtime_error(
            "CKKSParams: EvalMod coefficient prune threshold must be finite and non-negative");
    }
    return threshold;
}

double CKKSParams::getBootstrapEvalModDiscreteIntervalRadius() const {
    const double radius = requireBootstrapParams().evalmod_discrete_interval_radius;
    if (!std::isfinite(radius) || radius < 0.0 || radius >= 0.25) {
        throw std::runtime_error(
            "CKKSParams: EvalMod discrete interval radius must be in [0, 0.25)");
    }
    if (radius > 0.0) {
        return radius;
    }
    return automatic_binboot_discrete_interval_radius(
        getBootstrapEvalModLogMessageRatio());
}

size_t CKKSParams::getBootstrapEvalModDiscreteValidationPointsPerInterval() const {
    const size_t points =
        requireBootstrapParams().evalmod_discrete_validation_points_per_interval;
    if (points == 0) {
        return kAutoEvalModDiscreteValidationPointsPerInterval;
    }
    if (points < 2) {
        throw std::runtime_error(
            "CKKSParams: EvalMod discrete validation points per interval must be at least 2");
    }
    return points;
}

const std::array<size_t, 2>& CKKSParams::getBootstrapLinearTransformBsgsBabyStepCounts() const {
    return requireBootstrapParams().linear_transform_bsgs_baby_step_counts;
}

int CKKSParams::getBootstrapS2CLogBsgsRatio() const {
    return requireBootstrapParams().linear_transform_log_bsgs_ratios[0];
}

int CKKSParams::getBootstrapC2SLogBsgsRatio() const {
    return requireBootstrapParams().linear_transform_log_bsgs_ratios[1];
}

const std::vector<size_t>& CKKSParams::getBootstrapS2CMergeDepths() const {
    return requireBootstrapParams().linear_transform_merge_depths[0];
}

const std::vector<size_t>& CKKSParams::getBootstrapC2SMergeDepths() const {
    return requireBootstrapParams().linear_transform_merge_depths[1];
}

bool CKKSParams::getBootstrapS2CUseQpLinearTransform() const {
    return requireBootstrapParams().linear_transform_use_qp[0];
}

bool CKKSParams::getBootstrapC2SUseQpLinearTransform() const {
    return requireBootstrapParams().linear_transform_use_qp[1];
}
