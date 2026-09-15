#ifndef CKKS_PARAMS_HPP
#define CKKS_PARAMS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "aligned_vector.hpp"
#include "random.hpp"

using CKKSBootstrapDftScaleSchedule = std::vector<std::vector<size_t>>;

struct CKKSHomomorphicStandardLogPQBound {
    size_t ring_dimension;
    size_t max_log_pq_128;
};

class CKKSParams;
struct CKKSParamsConfig;

inline constexpr size_t kAutoEvalModDiscreteValidationPointsPerInterval = 64;
inline constexpr double kAutoBinBootDiscreteIntervalRadius = 1.0 / 32.0;

inline double automatic_binboot_discrete_interval_radius(size_t log_message_ratio) {
    if (log_message_ratio == 0) {
        return kAutoBinBootDiscreteIntervalRadius;
    }
    if (log_message_ratio >=
        static_cast<size_t>(std::numeric_limits<double>::max_exponent)) {
        return std::numeric_limits<double>::min();
    }
    const double radius =
        std::ldexp(1.0, -static_cast<int>(log_message_ratio));
    return std::min(radius, std::nextafter(0.25, 0.0));
}

struct CKKSBootstrapParams {
    // Bootstrap metadata.
    size_t sparse_secret_hamming_weight = 0;
    // Logical slot count for sparse bootstrap DFTs. Zero means the full
    // CKKS slot capacity N/2.
    size_t active_slot_count = 0;
    std::array<size_t, 2> linear_transform_depths{0, 0};
    std::array<CKKSBootstrapDftScaleSchedule, 2> linear_transform_log_plaintext_scales;
    size_t evalmod_depth = 0;
    size_t evalmod_bit_size = 0;
    // Zero means derive from the sparse-secret overflow bound.
    size_t evalmod_k = 0;
    long double evalmod_k_failure_probability_log2 = -128.0L;
    size_t evalmod_polynomial_degree = 120;
    size_t evalmod_double_angle = 0;
    size_t evalmod_log_message_ratio = 0;
    double evalmod_coefficient_prune_threshold = 1e-10;
    double evalmod_discrete_interval_radius = 0.0;
    size_t evalmod_discrete_validation_points_per_interval = 0;
    std::array<size_t, 2> linear_transform_bsgs_baby_step_counts{0, 0};
    std::array<int, 2> linear_transform_log_bsgs_ratios{1, 1};
    std::array<std::vector<size_t>, 2> linear_transform_merge_depths;
    std::array<bool, 2> linear_transform_use_qp{false, false};
};

struct CKKSAesWalshParams {
    bool enabled = false;
    // Active AES blocks per bit ciphertext. This is also the packed segment
    // stride for the AES Walsh path.
    size_t block_count = 0;
    // Number of block_count-sized packed segments to provision rotation keys for.
    // 0 means "use the full slot capacity": slots / block_count. The Walsh S-box
    // needs at least 30; larger values are useful for denser output refresh.
    size_t packed_segment_count = 0;
};

struct CKKSParamsConfig {
    double scale = 0.0;
    double sigma = 3.19;
    size_t N = size_t{1} << 15;
    size_t slots = 0;
    size_t depth = 0;
    size_t mult_depth = 0;
    size_t q0_bit = 33;
    size_t q_bit = 32;
    // Optional per-multiplication/circuit-limb bit sizes. When non-empty, this
    // must contain exactly mult_depth entries and overrides q_bit only for
    // those mult_depth limbs. S2C/EvalMod/C2S still use their own schedules.
    std::vector<uint64_t> q_mult_bits;
    size_t dnum = 3;
    size_t p_bit = 61;
    size_t pmoduli_count = 1;
    // Optional per-P-limb bit sizes. When non-empty, this overrides p_bit and
    // pmoduli_count; the P limb count is p_bits.size().
    std::vector<uint64_t> p_bits;
    size_t secret_key_hamming_weight = 256;
    std::optional<CKKSBootstrapParams> bootstrap_params;
    std::vector<uint64_t> q_bits;
    MyVector<int64_t> rotate_values;
    bool bts_keys = false;
    // If unset, CKKSParams derives plaintext precomputation from bts_keys.
    std::optional<bool> s2c_ptxt;
    std::optional<bool> c2s_ptxt;
    bool secure = true;
    bool fold_hybrid_keyswitch = true;
    bool montgomery = true;
    CKKSAesWalshParams aes_walsh_params;
};

namespace ckks_params_detail {

inline bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

inline size_t exact_log2_power_of_two(size_t value, const char* label) {
    if (!is_power_of_two(value)) {
        throw std::invalid_argument(std::string("CKKSParams: ") + label + " must be a power of two");
    }
    size_t log = 0;
    while ((size_t{1} << log) != value) {
        ++log;
    }
    return log;
}

inline size_t ceil_div(size_t numerator, size_t denominator) {
    if (denominator == 0) {
        throw std::invalid_argument("CKKSParams: division by zero");
    }
    return (numerator + denominator - 1) / denominator;
}

inline CKKSBootstrapParams make_evalmod_bootstrap_params(
    size_t sparse_secret_hamming_weight,
    std::array<CKKSBootstrapDftScaleSchedule, 2> linear_transform_log_plaintext_scales,
    size_t evalmod_depth,
    size_t evalmod_bit_size = 0,
    std::array<size_t, 2> linear_transform_bsgs_baby_step_counts = {0, 0}) {
    CKKSBootstrapParams params;
    params.sparse_secret_hamming_weight = sparse_secret_hamming_weight;
    params.linear_transform_log_plaintext_scales = std::move(linear_transform_log_plaintext_scales);
    params.linear_transform_depths = {
        params.linear_transform_log_plaintext_scales[0].size(),
        params.linear_transform_log_plaintext_scales[1].size()};
    params.evalmod_depth = evalmod_depth;
    params.evalmod_bit_size = evalmod_bit_size;
    params.linear_transform_bsgs_baby_step_counts = linear_transform_bsgs_baby_step_counts;
    return params;
}

inline size_t dft_scale_schedule_depth(const CKKSBootstrapDftScaleSchedule& schedule) {
    return schedule.size();
}

inline size_t bootstrap_linear_transform_depth(
    const CKKSBootstrapParams& params,
    size_t transform_index) {
    if (transform_index >= 2) {
        throw std::invalid_argument("CKKSParams: invalid bootstrap linear-transform index");
    }
    const auto& schedule = params.linear_transform_log_plaintext_scales[transform_index];
    return schedule.empty()
        ? params.linear_transform_depths[transform_index]
        : dft_scale_schedule_depth(schedule);
}

inline uint64_t dft_scale_group_bit_sum(const std::vector<size_t>& group) {
    if (group.empty()) {
        throw std::runtime_error("CKKSParams: DFT scale schedule groups must not be empty");
    }
    return static_cast<uint64_t>(
        std::accumulate(group.begin(), group.end(), size_t{0}));
}

inline void append_dft_group_bits_bottom_to_top(
    std::vector<uint64_t>& q_bits,
    const CKKSBootstrapDftScaleSchedule& schedule,
    uint64_t fallback_q_bit,
    size_t fallback_depth) {
    if (schedule.empty()) {
        q_bits.insert(q_bits.end(), fallback_depth, fallback_q_bit);
        return;
    }
    for (size_t reverse_index = 0; reverse_index < schedule.size(); ++reverse_index) {
        const size_t group_index = schedule.size() - 1 - reverse_index;
        const uint64_t group_sum = dft_scale_group_bit_sum(schedule[group_index]);
        q_bits.emplace_back(group_sum);
    }
}

}  // namespace ckks_params_detail

inline constexpr CKKSHomomorphicStandardLogPQBound
    kCKKSHomomorphicStandardLogPQ128Bounds[] = {
        {size_t{1} << 13, 214},
        {size_t{1} << 14, 430},
        {size_t{1} << 15, 868},
        {size_t{1} << 16, 1747},
    };

// Fixed-Hamming-weight h=256 bounds from local Lattice Estimator runs with
// Xs=ND.SparseTernary(128, 128, n=N), Xe=ND.DiscreteGaussian(3.19), m=oo,
// and RC.MATZOV. These are intentionally separate from the HE-standard
// uniform-ternary table above.
inline constexpr CKKSHomomorphicStandardLogPQBound
    kCKKSFixedHammingWeight256LogPQ128Bounds[] = {
        {size_t{1} << 13, 210},
        {size_t{1} << 14, 422},
        {size_t{1} << 15, 853},
        {size_t{1} << 16, 1720},
    };

// Fixed-Hamming-weight h=192 bounds from local Lattice Estimator runs with
// Xs=ND.SparseTernary(96, 96, n=N), Xe=ND.DiscreteGaussian(3.19), m=oo,
// and RC.MATZOV. Only N=2^15 has been estimated; reject other dimensions
// until they are explicitly measured.
inline constexpr CKKSHomomorphicStandardLogPQBound
    kCKKSFixedHammingWeight192LogPQ128Bounds[] = {
        {size_t{1} << 15, 808},
    };

class CKKSParams{
public:
    explicit CKKSParams(CKKSParamsConfig config);
    static CKKSParams AESWalsh15();
    static CKKSParams XBootParamAes13Reference();
    static CKKSParams AESXBoot14Secure();
    static CKKSParams AESXBoot14SecureSparse1024();
    static CKKSParams AESXBoot15();
    static CKKSParams AESXBoot15Sparse1024();
    // If false, CKKSContext skips bootstrap evaluation key generation.
    // Bootstrap operations will throw if the required keys are not present.
    bool bts_keys = false;
    bool s2c_ptxt = false;
    bool c2s_ptxt = false;
    bool secure = true;
    // If true, CKKSContext folds hybrid key-switch ModDown constants into
    // generated switching keys and uses the matching folded evaluator.
    bool fold_hybrid_keyswitch = true;
    // If true, precomputed CKKS bootstrap plaintexts and key-switching keys are
    // stored in Montgomery form and consumed by the Montgomery pointwise paths.
    bool montgomery = true;
    // AES Walsh tuning bundle.
    CKKSAesWalshParams aes_walsh_params;
    size_t getN() const;
    size_t getSlots() const;
    double getScale() const;
    double getSigma() const;
    size_t getSecretKeyHammingWeight() const;
    const MyVector<uint64_t>& getModuli() const;
    const std::vector<uint64_t>& getQBits() const;
    size_t getQ0Bit() const;
    size_t getDefaultQBit() const;

    // Returns the highest level index (levels = limbs_count - 1).
    size_t getDnum() const;
    size_t getDepth() const;
    size_t getMaxLevel() const;
    const MyVector<uint64_t>& getPModuli() const;
    const MyVector<int64_t>& getRotateValues() const;
    bool usesAesWalsh() const;
    size_t getAesWalshBlockCount() const;
    size_t getAesWalshPackedSegmentCount() const;
    bool hasBootstrapParams() const;
    const CKKSBootstrapParams& getBootstrapParams() const;
    size_t getBootstrapActiveSlotCount() const;
    bool usesSparseBootstrapSlots() const;
    size_t getBootstrapSparseSecretHammingWeight() const;
    size_t getBootstrapS2CDepth() const;
    size_t getBootstrapC2SDepth() const;
    const CKKSBootstrapDftScaleSchedule& getBootstrapS2CLogPlaintextScales() const;
    const CKKSBootstrapDftScaleSchedule& getBootstrapC2SLogPlaintextScales() const;
    size_t getBootstrapEvalModDepth() const;
    size_t getBootstrapEvalModK() const;
    size_t getBootstrapEvalModPolynomialDegree() const;
    size_t getBootstrapEvalModDoubleAngle() const;
    size_t getBootstrapEvalModLogMessageRatio() const;
    double getBootstrapEvalModCoefficientPruneThreshold() const;
    double getBootstrapEvalModDiscreteIntervalRadius() const;
    size_t getBootstrapEvalModDiscreteValidationPointsPerInterval() const;
    const std::array<size_t, 2>& getBootstrapLinearTransformBsgsBabyStepCounts() const;
    int getBootstrapS2CLogBsgsRatio() const;
    int getBootstrapC2SLogBsgsRatio() const;
    const std::vector<size_t>& getBootstrapS2CMergeDepths() const;
    const std::vector<size_t>& getBootstrapC2SMergeDepths() const;
    bool getBootstrapS2CUseQpLinearTransform() const;
    bool getBootstrapC2SUseQpLinearTransform() const;
    
private:
    double scale;
    double sigma = 3.19;
    size_t N;
    size_t slots = 0;
    size_t limbs_count;
    size_t depth;
    size_t mult_depth;
    size_t q0_bit;
    size_t q_bit;
    std::vector<uint64_t> q_mult_bits;
    size_t dnum;
    size_t secret_key_hamming_weight = 0;
    std::optional<CKKSBootstrapParams> bootstrap_params;
    MyVector<uint64_t> pmoduli;
    MyVector<uint64_t> moduli;
    std::vector<uint64_t> q_bits;
    std::vector<uint64_t> p_bits;
    MyVector<int64_t> rotate_values;

    struct UncheckedTag {};
    explicit CKKSParams(UncheckedTag) {}
    size_t bootstrapDepth(const CKKSBootstrapParams& params) const;
    std::vector<uint64_t> defaultQBits() const;
    size_t calculateDepth() const;
    const CKKSBootstrapParams& requireBootstrapParams() const;

};


inline size_t homomorphic_standard_max_log_pq_128(size_t N) {
    for (const auto& entry : kCKKSHomomorphicStandardLogPQ128Bounds) {
        if (entry.ring_dimension == N) {
            return entry.max_log_pq_128;
        }
    }
    throw std::runtime_error(
        "CKKSParams: no Homomorphic Encryption Standard 128-bit log(PQ) bound for N=" +
        std::to_string(N));
}

inline size_t fixed_hamming_weight_256_max_log_pq_128(size_t N) {
    for (const auto& entry : kCKKSFixedHammingWeight256LogPQ128Bounds) {
        if (entry.ring_dimension == N) {
            return entry.max_log_pq_128;
        }
    }
    throw std::runtime_error(
        "CKKSParams: no fixed-hamming-weight h=256 128-bit log(PQ) bound for N=" +
        std::to_string(N));
}

inline size_t fixed_hamming_weight_192_max_log_pq_128(size_t N) {
    for (const auto& entry : kCKKSFixedHammingWeight192LogPQ128Bounds) {
        if (entry.ring_dimension == N) {
            return entry.max_log_pq_128;
        }
    }
    throw std::runtime_error(
        "CKKSParams: no fixed-hamming-weight h=192 128-bit log(PQ) bound for N=" +
        std::to_string(N));
}

inline size_t secure_max_log_pq_128(size_t N, size_t secret_key_hamming_weight) {
    if (secret_key_hamming_weight == 0) {
        return homomorphic_standard_max_log_pq_128(N);
    }
    if (secret_key_hamming_weight == 192) {
        return fixed_hamming_weight_192_max_log_pq_128(N);
    }
    if (secret_key_hamming_weight == 256) {
        return fixed_hamming_weight_256_max_log_pq_128(N);
    }
    throw std::runtime_error(
        "CKKSParams: secure fixed-hamming-weight secrets are only supported for h=192 and h=256; "
        "set secret_key_hamming_weight=0 to use standard uniform ternary validation");
}

inline const char* secure_secret_distribution_label(size_t secret_key_hamming_weight) {
    switch (secret_key_hamming_weight) {
        case 0:
            return "uniform ternary";
        case 192:
            return "fixed h=192 ternary";
        case 256:
            return "fixed h=256 ternary";
        default:
            return "fixed unsupported-H ternary";
    }
}

inline void validate_homomorphic_standard_log_pq(
    size_t N,
    const std::vector<uint64_t>& q_bits,
    const std::vector<uint64_t>& p_bits,
    size_t secret_key_hamming_weight) {
    size_t q_bit_sum = 0;
    for (const uint64_t bit_count : q_bits) {
        q_bit_sum += static_cast<size_t>(bit_count);
    }
    size_t p_bit_sum = 0;
    for (const uint64_t bit_count : p_bits) {
        if (bit_count == 0) {
            throw std::runtime_error("CKKSParams: P bit sizes must be positive");
        }
        p_bit_sum += static_cast<size_t>(bit_count);
    }
    const size_t log_pq = q_bit_sum + p_bit_sum;
    const size_t max_log_pq = secure_max_log_pq_128(N, secret_key_hamming_weight);
    std::cout << "CKKSParams "
              << ": total Q bits = " << q_bit_sum
              << ", total P bits = " << p_bit_sum
              << ", total PQ bits = " << log_pq
              << ", 128-bit max = " << max_log_pq
              << ", secret = "
              << secure_secret_distribution_label(secret_key_hamming_weight)
              << " for N=" << N << '\n';
    if (log_pq > max_log_pq) {
        throw std::runtime_error(
            std::string("CKKSParams ") +
            ": log(PQ) bit budget " + std::to_string(log_pq) +
            " exceeds 128-bit security bound " +
            std::to_string(max_log_pq) + " for N=" + std::to_string(N));
    }
}

inline void validate_homomorphic_standard_log_pq(
    size_t N,
    const std::vector<uint64_t>& q_bits,
    uint64_t p_bit,
    size_t pmoduli_count,
    size_t secret_key_hamming_weight) {
    validate_homomorphic_standard_log_pq(
        N,
        q_bits,
        std::vector<uint64_t>(pmoduli_count, p_bit),
        secret_key_hamming_weight);
}

inline void validate_homomorphic_standard_log_pq(
    const char*,
    size_t N,
    const std::vector<uint64_t>& q_bits,
    const std::vector<uint64_t>& p_bits) {
    validate_homomorphic_standard_log_pq(N, q_bits, p_bits, 0);
}

inline void validate_homomorphic_standard_log_pq(
    const char*,
    size_t N,
    const std::vector<uint64_t>& q_bits,
    uint64_t p_bit,
    size_t pmoduli_count) {
    validate_homomorphic_standard_log_pq(N, q_bits, p_bit, pmoduli_count, 0);
}

#endif
