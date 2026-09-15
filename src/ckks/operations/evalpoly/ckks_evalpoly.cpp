#include "ckks_evalpoly.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ckks_addition.hpp"
#include "ckks_mul.hpp"
#include "ckks_relin.hpp"
#include "ckks_rescale.hpp"
#include "ntt.hpp"

namespace {

struct ExactScale {
    uint64_t seed = 1;
    int seed_exp = 0;
    std::vector<int> modulus_den_exp;

    static ExactScale from_near_integer_double(
        double value,
        size_t modulus_count) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument(
                "ckks_evalchebyshev: invalid exact scale seed");
        }
        const long double rounded = std::round(static_cast<long double>(value));
        const long double tol =
            std::max<long double>(1.0L, std::abs(static_cast<long double>(value))) *
            0x1p-20L;
        if (std::abs(static_cast<long double>(value) - rounded) > tol) {
            throw std::invalid_argument(
                "ckks_evalchebyshev: non-integral initial scale is unsupported");
        }
        if (rounded > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
            throw std::invalid_argument(
                "ckks_evalchebyshev: initial scale exceeds uint64_t");
        }
        ExactScale scale;
        scale.seed = static_cast<uint64_t>(rounded);
        scale.seed_exp = 1;
        scale.modulus_den_exp.assign(modulus_count, 0);
        return scale;
    }

    void ensure_modulus_count(size_t modulus_count) {
        if (modulus_den_exp.size() < modulus_count) {
            modulus_den_exp.resize(modulus_count, 0);
        }
    }

    long double to_long_double(std::span<const uint64_t> moduli) const {
        long double value = 1.0L;
        for (int i = 0; i < seed_exp; ++i) {
            value *= static_cast<long double>(seed);
        }
        for (size_t i = 0; i < modulus_den_exp.size(); ++i) {
            for (int e = 0; e < modulus_den_exp[i]; ++e) {
                value /= static_cast<long double>(moduli[i]);
            }
        }
        return value;
    }

    double to_double(std::span<const uint64_t> moduli) const {
        return static_cast<double>(to_long_double(moduli));
    }
};

ExactScale multiply_exact_scales(const ExactScale& lhs, const ExactScale& rhs) {
    ExactScale out;
    out.seed = lhs.seed;
    out.seed_exp = lhs.seed_exp + rhs.seed_exp;
    out.modulus_den_exp = lhs.modulus_den_exp;
    if (out.modulus_den_exp.size() < rhs.modulus_den_exp.size()) {
        out.modulus_den_exp.resize(rhs.modulus_den_exp.size(), 0);
    }
    for (size_t i = 0; i < rhs.modulus_den_exp.size(); ++i) {
        out.modulus_den_exp[i] += rhs.modulus_den_exp[i];
    }
    return out;
}

void divide_exact_scale_by_modulus_index(ExactScale& scale, size_t modulus_index) {
    scale.ensure_modulus_count(modulus_index + 1);
    scale.modulus_den_exp[modulus_index] += 1;
}

uint64_t exact_scale_ratio_to_u64(
    const ExactScale& numerator,
    const ExactScale& denominator,
    std::span<const uint64_t> moduli) {
    long double ratio = 1.0L;

    if (numerator.seed != denominator.seed) {
        throw std::invalid_argument(
            "ckks_evalchebyshev: mixed exact-scale seeds are unsupported");
    }

    const int seed_exp_diff = numerator.seed_exp - denominator.seed_exp;
    for (int i = 0; i < seed_exp_diff; ++i) {
        ratio *= static_cast<long double>(numerator.seed);
    }
    for (int i = 0; i < -std::min(seed_exp_diff, 0); ++i) {
        ratio /= static_cast<long double>(numerator.seed);
    }
    for (size_t i = 0; i < moduli.size(); ++i) {
        const int num_exp =
            i < numerator.modulus_den_exp.size() ? numerator.modulus_den_exp[i] : 0;
        const int den_exp =
            i < denominator.modulus_den_exp.size() ? denominator.modulus_den_exp[i] : 0;
        const int diff = den_exp - num_exp;
        if (diff > 0) {
            for (int e = 0; e < diff; ++e) {
                ratio *= static_cast<long double>(moduli[i]);
            }
        } else if (diff < 0) {
            for (int e = 0; e < -diff; ++e) {
                ratio /= static_cast<long double>(moduli[i]);
            }
        }
    }

    const long double floored = std::floor(ratio);
    if (!std::isfinite(floored) ||
        floored < 0.0L ||
        floored > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
        throw std::runtime_error(
            "ckks_evalchebyshev: exact-scale ratio exceeds uint64_t");
    }
    return static_cast<uint64_t>(floored);
}

size_t trim_polynomial_degree(std::span<const double> coeffs) {
    if (coeffs.empty()) {
        throw std::invalid_argument("ckks_evalpoly: polynomial must have at least one coefficient");
    }
    for (double coeff : coeffs) {
        if (!std::isfinite(coeff)) {
            throw std::invalid_argument("ckks_evalpoly: polynomial coefficients must be finite");
        }
    }
    for (size_t i = coeffs.size(); i > 0; --i) {
        if (coeffs[i - 1] != 0.0) {
            return i - 1;
        }
    }
    return 0;
}



size_t bit_length(size_t value) {
    if (value == 0) {
        return 0;
    }
    return static_cast<size_t>(
        std::numeric_limits<size_t>::digits - std::countl_zero(value));
}

size_t ceil_log2_degree(size_t degree) {
    if (degree <= 1) {
        return 0;
    }
    return bit_length(degree - 1);
}

size_t optimal_power_of_two_split_log(size_t log_degree) {
    size_t log_split = log_degree >> 1;
    const size_t a =
        (size_t{1} << log_split) +
        (size_t{1} << (log_degree - log_split)) +
        log_degree - log_split - 3;
    const size_t b =
        (size_t{1} << (log_split + 1)) +
        (size_t{1} << (log_degree - log_split - 1)) +
        log_degree - log_split - 4;
    if (a > b) {
        ++log_split;
    }
    return log_split;
}

std::pair<size_t, size_t> split_chebyshev_degree(size_t degree) {
    if (degree == 0) {
        throw std::invalid_argument("ckks_evalchebyshev: cannot split degree zero");
    }
    if ((degree & (degree - 1)) == 0) {
        return {degree >> 1, degree >> 1};
    }
    const size_t k = bit_length(degree - 1) - 1;
    return {(size_t{1} << k) - 1, degree + 1 - (size_t{1} << k)};
}




uint64_t rounded_scaled_real_mod_q(
    double value,
    long double scale,
    uint64_t q) {
    const long double scaled =
        static_cast<long double>(value) *
        scale;
    if (!std::isfinite(scaled)) {
        throw std::invalid_argument("ckks_evalpoly: non-finite scaled constant");
    }
    const long double rounded = std::round(scaled);
    long double residue = std::fmod(
        rounded,
        static_cast<long double>(q));
    if (residue < 0.0L) {
        residue += static_cast<long double>(q);
    }
    uint64_t out = static_cast<uint64_t>(residue);
    if (out >= q) {
        out %= q;
    }
    return out;
}

size_t estimate_preconditioned_chebyshev_eval_depth(size_t degree) {
    return ceil_log2_degree(degree);
}

void validate_evalpoly_common_input(
    const CKKSContext& context,
    const CKKSCiphertext& x,
    double target_scale) {
    const auto& params = context.getParams();
    if (x.getN() != params.getN()) {
        throw std::invalid_argument("ckks_evalpoly: ciphertext/context size mismatch");
    }
    if (x.getNumPolys() != 2) {
        throw std::invalid_argument("ckks_evalpoly: expected a 2-polynomial ciphertext");
    }
    if (x.getSecretOwner() != CKKSSecretOwner::Dense) {
        throw std::invalid_argument("ckks_evalpoly: polynomial evaluation requires a dense-secret ciphertext");
    }
    if (x.getMessageEncodingState() != CKKSMessageEncodingState::Slots) {
        throw std::invalid_argument("ckks_evalpoly: polynomial evaluation expects slot-encoded input");
    }
    if (!std::isfinite(target_scale) || target_scale <= 0.0) {
        throw std::invalid_argument("ckks_evalpoly: invalid scale");
    }
}


CKKSCiphertext zero_ciphertext_like(
    const CKKSContext& context,
    size_t level,
    double scale,
    CKKSSecretOwner owner,
    CKKSMessageEncodingState encoding_state) {
    const size_t N = context.getParams().getN();
    const size_t limb_count = level + 1;
    return CKKSCiphertext(
        N,
        MyVector<uint64_t>(N * limb_count, uint64_t{0}),
        MyVector<uint64_t>(N * limb_count, uint64_t{0}),
        scale,
        level,
        owner,
        encoding_state);
}


void add_constant_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    double constant,
    long double scale) {
    if (constant == 0.0) {
        return;
    }
    auto& b = ct.getBMutable();
    const auto& moduli = context.getParams().getModuli();
    const size_t N = ct.getN();
    const size_t level = ct.getLevel();
    for (size_t limb = 0; limb <= level; ++limb) {
        const uint64_t q = moduli[limb];
        const uint64_t scalar_mod_q =
            rounded_scaled_real_mod_q(constant, scale, q);
        uint64_t* __restrict b_ptr = b.data() + limb * N;
        for (size_t i = 0; i < N; ++i) {
            b_ptr[i] = add_mod_q(b_ptr[i], scalar_mod_q, q);
        }
    }
}







void multiply_by_integer_scalar_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    long double scalar,
    long double output_scale) {
    if (!std::isfinite(scalar) || scalar < 0.0L) {
        throw std::invalid_argument("ckks_evalchebyshev: invalid scalar scale-up");
    }
    const long double scalar_rounded = std::round(scalar);
    if (scalar_rounded < 1.0L ||
        std::abs(scalar - scalar_rounded) >
            std::max<long double>(1.0L, std::abs(scalar)) * 0x1p-20L) {
        throw std::invalid_argument(
            "ckks_evalchebyshev: non-integral scalar scale-up");
    }
    if (scalar_rounded > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
        throw std::invalid_argument("ckks_evalchebyshev: scalar scale-up is too large");
    }
    const uint64_t scalar_u64 = static_cast<uint64_t>(scalar_rounded);
    if (scalar_u64 == 1) {
        ct.setScale(static_cast<double>(output_scale));
        return;
    }

    const auto& moduli = context.getParams().getModuli();
    const bool use_montgomery = context.getParams().montgomery;
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const size_t N = ct.getN();
    const size_t level = ct.getLevel();
    for (size_t poly_idx = 0; poly_idx < ct.getNumPolys(); ++poly_idx) {
        auto& poly = ct.getPolyMutable(poly_idx);
        for (size_t limb = 0; limb <= level; ++limb) {
            const uint64_t q = moduli[limb];
            const uint64_t scalar_mod_q = scalar_u64 % q;
            pointwise_multiply_scalar_inplace_dispatch(
                poly.data() + limb * N,
                scalar_mod_q,
                N,
                q,
                twiddle_ntt[limb],
                use_montgomery);
        }
    }
    ct.setScale(static_cast<double>(output_scale));
}

void fused_multiply_by_real_scalar_then_add_inplace(
    const CKKSContext& context,
    const CKKSCiphertext& src,
    double scalar,
    long double scalar_scale,
    CKKSCiphertext& dst) {
    if (src.getN() != dst.getN()) {
        throw std::invalid_argument(
            "ckks_evalchebyshev: fused scalar multiply-add size mismatch");
    }
    if (src.getLevel() != dst.getLevel()) {
        throw std::invalid_argument(
            "ckks_evalchebyshev: fused scalar multiply-add level mismatch");
    }
    if (src.getNumPolys() != dst.getNumPolys()) {
        throw std::invalid_argument(
            "ckks_evalchebyshev: fused scalar multiply-add degree mismatch");
    }

    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const size_t N = src.getN();
    const size_t level = src.getLevel();

    for (size_t limb = 0; limb <= level; ++limb) {
        const uint64_t q = moduli[limb];
        const uint64_t scalar_mod_q =
            rounded_scaled_real_mod_q(scalar, scalar_scale, q);
        if (scalar_mod_q == 0) {
            continue;
        }
        const auto* barrett_const = &twiddle_ntt[limb].barrett_const;
        for (size_t poly_idx = 0; poly_idx < src.getNumPolys(); ++poly_idx) {
            const uint64_t* __restrict src_ptr =
                src.getPoly(poly_idx).data() + limb * N;
            uint64_t* __restrict dst_ptr =
                dst.getPolyMutable(poly_idx).data() + limb * N;
            for (size_t i = 0; i < N; ++i) {
                dst_ptr[i] = add_mod_q(
                    dst_ptr[i],
                    mul_mod_u64(src_ptr[i], scalar_mod_q, q, barrett_const),
                    q);
            }
        }
    }
}

void scale_up_ciphertext_to_match_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    long double target_scale) {
    const double current_scale = ct.getScale();
    const double target_scale_double = static_cast<double>(target_scale);
    const double rel_tol = 0x1p-45;
    const double scale =
        std::max({1.0, std::abs(current_scale), std::abs(target_scale_double)});
    if (std::isfinite(current_scale) &&
        std::isfinite(target_scale_double) &&
        std::abs(current_scale - target_scale_double) <= rel_tol * scale) {
        ct.setScale(static_cast<double>(target_scale));
        return;
    }
    if (static_cast<long double>(ct.getScale()) > target_scale) {
        throw std::invalid_argument(
            "ckks_evalchebyshev: cannot scale down ciphertext during scale alignment");
    }
    const long double ratio =
        target_scale / static_cast<long double>(ct.getScale());
    multiply_by_integer_scalar_inplace(context, ct, ratio, target_scale);
}

size_t levels_consumed_per_rescaling(const CKKSContext& context) {
    return std::log2(context.getParams().getScale()) <= 64.0 ? 1 : 2;
}

long double rescale_modulus_product(
    const CKKSContext& context,
    size_t level) {
    const auto& moduli = context.getParams().getModuli();
    const size_t levels = levels_consumed_per_rescaling(context);
    if (level + 1 < levels) {
        throw std::invalid_argument(
            "ckks_evalchebyshev: insufficient level for rescale modulus product");
    }
    long double scale = 1.0L;
    for (size_t i = 0; i < levels; ++i) {
        scale *= static_cast<long double>(moduli[level - i]);
    }
    return scale;
}

bool scalar_is_integer(double value, double rel_tol = 0x1p-20) {
    if (!std::isfinite(value)) {
        return false;
    }
    const double rounded = std::round(value);
    const double scale = std::max(1.0, std::abs(value));
    return std::abs(value - rounded) <= rel_tol * scale;
}

bool scales_close(double lhs, double rhs, double rel_tol);

CKKSCiphertext align_ciphertext_for_add_with_polys(
    CKKSCiphertext ct,
    size_t level,
    double scale,
    size_t target_polys);

void multhenadd_real_scalar_scale_aware_inplace(
    const CKKSContext& context,
    const CKKSCiphertext& op0,
    double scalar,
    CKKSCiphertext& op_out) {
    if (scalar == 0.0) {
        return;
    }

    const size_t level = std::min(op0.getLevel(), op_out.getLevel());
    const double op_out_scale = op_out.getScale();
    const size_t target_polys = std::max(op0.getNumPolys(), op_out.getNumPolys());
    op_out = align_ciphertext_for_add_with_polys(
        std::move(op_out),
        level,
        op_out_scale,
        target_polys);
    auto op0_aligned = align_ciphertext_for_add_with_polys(
        op0.clone(),
        level,
        op0.getScale(),
        target_polys);

    long double scale_rlwe = 0.0L;
    if (scales_close(op0_aligned.getScale(), op_out.getScale(), 0x1p-45)) {
        if (scalar_is_integer(scalar)) {
            scale_rlwe = 1.0L;
        } else {
            scale_rlwe = rescale_modulus_product(context, level);
            scale_up_ciphertext_to_match_inplace(
                context,
                op_out,
                static_cast<long double>(op_out.getScale()) * scale_rlwe);
        }
    } else if (op0_aligned.getScale() < op_out.getScale()) {
        scale_rlwe =
            static_cast<long double>(op_out.getScale()) /
            static_cast<long double>(op0_aligned.getScale());
    } else {
        std::ostringstream oss;
        oss << "ckks_evalchebyshev: MulThenAdd requires op0.Scale <= opOut.Scale"
            << " (op0=" << op0_aligned.getScale()
            << ", opOut=" << op_out.getScale()
            << ", level=" << level
            << ")";
        throw std::invalid_argument(oss.str());
    }

    const long double term_scale =
        static_cast<long double>(op0_aligned.getScale()) * scale_rlwe;
    if (!scales_close(
            static_cast<double>(term_scale),
            op_out.getScale(),
            0x1p-45)) {
        throw std::runtime_error(
            "ckks_evalchebyshev: MulThenAdd scale mismatch");
    }
    fused_multiply_by_real_scalar_then_add_inplace(
        context,
        op0_aligned,
        scalar,
        scale_rlwe,
        op_out);
}

void pad_ciphertext_with_zero_polys(CKKSCiphertext& ct, size_t target_polys) {
    if (target_polys < ct.getNumPolys()) {
        throw std::invalid_argument("ckks_evalchebyshev: cannot shrink ciphertext degree by padding");
    }
    const size_t poly_size = ct.getN() * (ct.getLevel() + 1);
    while (ct.getNumPolys() < target_polys) {
        ct.addPoly(MyVector<uint64_t>(poly_size, 0));
    }
}

CKKSCiphertext align_ciphertext_for_add_with_polys(
    CKKSCiphertext ct,
    size_t level,
    double scale,
    size_t target_polys) {
    ckks_drop_to_level_inplace(ct, level);
    ct.setScale(scale);
    pad_ciphertext_with_zero_polys(ct, target_polys);
    return ct;
}

bool scales_close(double lhs, double rhs, double rel_tol = 0x1p-45) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return false;
    }
    const long double lhs_ld = static_cast<long double>(lhs);
    const long double rhs_ld = static_cast<long double>(rhs);
    const long double scale =
        std::max<long double>({1.0L, std::abs(lhs_ld), std::abs(rhs_ld)});
    return std::abs(lhs_ld - rhs_ld) <=
           static_cast<long double>(rel_tol) * scale;
}

// The scale-aware, level-optimal polynomial evaluation strategy below follows
// Section 3 of J.-P. Bossuat, C. Mouchet, J. Troncoso-Pastoriza, and
// J.-P. Hubaux, "Efficient Bootstrapping for Approximate Homomorphic
// Encryption with Non-Sparse Keys," EUROCRYPT 2021, pp. 587-617.
// DOI: 10.1007/978-3-030-77870-5_21
// IACR ePrint: https://eprint.iacr.org/2020/1203
struct ScaleSimOperand {
    size_t level = 0;
    long double scale = 0.0L;
};

size_t polynomial_evaluation_depth(size_t degree) {
    if (degree <= 1) {
        return 0;
    }
    return bit_length(degree) - 1;
}

class ScaleSimEvaluator {
public:
    explicit ScaleSimEvaluator(const CKKSContext& context)
        : context_(context) {}

    size_t polynomial_depth(size_t degree) const {
        return polynomial_evaluation_depth(degree);
    }

    void rescale(ScaleSimOperand& operand) const {
        operand.scale /=
            static_cast<long double>(context_.getParams().getModuli()[operand.level]);
        --operand.level;
    }

    ScaleSimOperand mul_new(
        const ScaleSimOperand& lhs,
        const ScaleSimOperand& rhs) const {
        return ScaleSimOperand{
            .level = std::min(lhs.level, rhs.level),
            .scale = lhs.scale * rhs.scale,
        };
    }

    std::pair<size_t, long double> update_level_and_scale_baby_step(
        bool lead,
        size_t level_old,
        long double scale_old) const {
        long double scale_new = scale_old;
        if (lead) {
            scale_new *= static_cast<long double>(
                context_.getParams().getModuli()[level_old]);
        }
        return {level_old, scale_new};
    }

    std::pair<size_t, long double> update_level_and_scale_giant_step(
        bool lead,
        size_t level_old,
        long double scale_old,
        long double xpow_scale) const {
        const auto& moduli = context_.getParams().getModuli();
        const uint64_t qi =
            lead ? moduli[level_old] : moduli[level_old + 1];
        return {
            level_old + 1,
            scale_old * static_cast<long double>(qi) / xpow_scale,
        };
    }

private:
    const CKKSContext& context_;
};

using ScaleSimPowerBasis = std::map<size_t, ScaleSimOperand>;

void simulate_power_basis_generation(
    size_t n,
    ScaleSimPowerBasis& power_basis,
    const ScaleSimEvaluator& evaluator) {
    if (n < 2 || power_basis.contains(n)) {
        return;
    }
    auto [a, b] = split_chebyshev_degree(n);
    simulate_power_basis_generation(a, power_basis, evaluator);
    simulate_power_basis_generation(b, power_basis, evaluator);
    power_basis[n] = evaluator.mul_new(power_basis.at(a), power_basis.at(b));
    evaluator.rescale(power_basis.at(n));
}

struct EvaluationPolynomial {
    std::vector<double> coeffs;
    size_t max_degree = 0;
    bool lead = true;
    size_t level = 0;
    long double scale = 0.0L;
    bool is_even = false;
    bool is_odd = false;

    size_t degree() const {
        return coeffs.empty() ? 0 : coeffs.size() - 1;
    }
};

std::pair<EvaluationPolynomial, EvaluationPolynomial> factorize_evaluation_polynomial(
    const EvaluationPolynomial& poly,
    size_t split_degree) {
    const size_t degree = poly.degree();
    if (split_degree < (degree >> 1)) {
        throw std::invalid_argument("ckks_evalchebyshev: invalid recursive PS split");
    }

    EvaluationPolynomial quotient;
    EvaluationPolynomial remainder;
    quotient.coeffs.assign(degree - split_degree + 1, 0.0);
    remainder.coeffs.assign(split_degree, 0.0);

    for (size_t i = 0; i < split_degree && i < poly.coeffs.size(); ++i) {
        remainder.coeffs[i] = poly.coeffs[i];
    }
    if (split_degree < poly.coeffs.size()) {
        quotient.coeffs[0] = poly.coeffs[split_degree];
    }
    for (size_t i = split_degree + 1, j = 1; i <= degree; ++i, ++j) {
        const double coeff = poly.coeffs[i];
        if (coeff == 0.0) {
            continue;
        }
        quotient.coeffs[i - split_degree] += 2.0 * coeff;
        remainder.coeffs[split_degree - j] -= coeff;
    }

    quotient.max_degree = poly.max_degree;
    quotient.lead = poly.lead;
    quotient.is_even = poly.is_even;
    quotient.is_odd = poly.is_odd;

    remainder.max_degree =
        poly.max_degree == degree
            ? (split_degree == 0 ? 0 : split_degree - 1)
            : poly.max_degree - (degree - split_degree + 1);
    remainder.lead = false;
    remainder.is_even = poly.is_even;
    remainder.is_odd = poly.is_odd;
    return {std::move(quotient), std::move(remainder)};
}

struct PatersonStockmeyerPlan {
    size_t degree = 0;
    size_t base = 0;
    size_t level = 0;
    long double scale = 0.0L;
    std::vector<EvaluationPolynomial> value;
};

std::pair<std::vector<EvaluationPolynomial>, ScaleSimOperand> build_paterson_stockmeyer_plan_recursive(
    const CKKSContext& context,
    size_t log_split,
    size_t target_level,
    const EvaluationPolynomial& poly,
    const ScaleSimPowerBasis& power_basis,
    long double output_scale,
    const ScaleSimEvaluator& evaluator) {
    const size_t degree = poly.degree();
    const size_t base = size_t{1} << log_split;

    if (degree < base) {
        if (poly.lead &&
            log_split > 1 &&
            poly.max_degree >
                ((size_t{1} << bit_length(poly.max_degree)) -
                 (size_t{1} << (log_split - 1)))) {
            const size_t degree_log = bit_length(degree);
            const size_t new_log_split =
                optimal_power_of_two_split_log(degree_log);
            return build_paterson_stockmeyer_plan_recursive(
                context,
                new_log_split,
                target_level,
                poly,
                power_basis,
                output_scale,
                evaluator);
        }

        auto leaf = poly;
        const auto [level, scale] =
            evaluator.update_level_and_scale_baby_step(
                leaf.lead,
                target_level,
                output_scale);
        leaf.level = level;
        leaf.scale = scale;
        return {
            std::vector<EvaluationPolynomial>{std::move(leaf)},
            ScaleSimOperand{.level = level, .scale = scale},
        };
    }

    size_t next_power = base;
    while (next_power < (degree >> 1) + 1) {
        next_power <<= 1;
    }

    const auto& xpow = power_basis.at(next_power);
    const auto [quotient, remainder] =
        factorize_evaluation_polynomial(poly, next_power);
    const auto [level_new, scale_new] =
        evaluator.update_level_and_scale_giant_step(
            poly.lead,
            target_level,
            output_scale,
            xpow.scale);

    auto [q_values, res] = build_paterson_stockmeyer_plan_recursive(
        context,
        log_split,
        level_new,
        quotient,
        power_basis,
        scale_new,
        evaluator);
    evaluator.rescale(res);
    res = evaluator.mul_new(res, xpow);

    auto [r_values, tmp] = build_paterson_stockmeyer_plan_recursive(
        context,
        log_split,
        target_level,
        remainder,
        power_basis,
        res.scale,
        evaluator);

    if (!scales_close(
            static_cast<double>(res.scale),
            static_cast<double>(tmp.scale))) {
        throw std::runtime_error(
            "ckks_evalchebyshev: simulated PS scale mismatch");
    }

    q_values.insert(q_values.end(), r_values.begin(), r_values.end());
    return {std::move(q_values), res};
}

PatersonStockmeyerPlan make_paterson_stockmeyer_plan(
    const CKKSContext& context,
    const EvaluationPolynomial& polynomial,
    size_t input_level,
    long double input_scale,
    long double output_scale) {
    const size_t degree = polynomial.degree();
    const size_t log_degree = bit_length(degree);
    const size_t log_split = optimal_power_of_two_split_log(log_degree);

    ScaleSimEvaluator evaluator(context);
    ScaleSimPowerBasis power_basis;
    power_basis[1] = ScaleSimOperand{
        .level = input_level,
        .scale = input_scale,
    };
    simulate_power_basis_generation(size_t{1} << log_degree, power_basis, evaluator);

    auto [value, _] = build_paterson_stockmeyer_plan_recursive(
        context,
        log_split,
        input_level - evaluator.polynomial_depth(degree),
        polynomial,
        power_basis,
        output_scale,
        evaluator);

    return PatersonStockmeyerPlan{
        .degree = degree,
        .base = size_t{1} << log_split,
        .level = input_level,
        .scale = output_scale,
        .value = std::move(value),
    };
}

class PowerBasis {
public:
    PowerBasis(
        const CKKSContext& context,
        const CKKSCiphertext& input)
        : context_(context) {
        values_.emplace(1, input.clone());
        exact_scales_.emplace(
            1,
            ExactScale::from_near_integer_double(
                input.getScale(),
                context_.getParams().getModuli().size()));
    }

    void gen_power(size_t n) {
        if (n < 2) {
            return;
        }
        if (values_.contains(n)) {
            return;
        }
        const bool needs_rescale = gen_power_recursive(n);
        if (needs_rescale) {
            const size_t level_before = values_.at(n).getLevel();
            ckks_rescale(context_, values_.at(n));
            divide_exact_scale_by_modulus_index(
                exact_scales_.at(n),
                level_before);
            values_.at(n).setScale(
                exact_scales_.at(n).to_double(context_.getParams().getModuli()));
        }
    }

    const CKKSCiphertext& at(size_t n) const {
        return values_.at(n);
    }

    CKKSCiphertext& at(size_t n) {
        return values_.at(n);
    }

private:
    bool gen_power_recursive(size_t n) {
        if (values_.contains(n)) {
            return false;
        }

        const auto [a, b] = split_chebyshev_degree(n);
        const bool rescale_a = gen_power_recursive(a);
        const bool rescale_b = gen_power_recursive(b);

        if (rescale_a) {
            const size_t level_before = values_.at(a).getLevel();
            ckks_rescale(context_, values_.at(a));
            divide_exact_scale_by_modulus_index(
                exact_scales_.at(a),
                level_before);
            values_.at(a).setScale(
                exact_scales_.at(a).to_double(context_.getParams().getModuli()));
        }
        if (rescale_b) {
            const size_t level_before = values_.at(b).getLevel();
            ckks_rescale(context_, values_.at(b));
            divide_exact_scale_by_modulus_index(
                exact_scales_.at(b),
                level_before);
            values_.at(b).setScale(
                exact_scales_.at(b).to_double(context_.getParams().getModuli()));
        }

        auto out = values_.at(a).clone();
        const size_t level = a == b
            ? out.getLevel()
            : std::min(out.getLevel(), values_.at(b).getLevel());
        ckks_drop_to_level_inplace(out, level);

        const ExactScale product_scale =
            multiply_exact_scales(exact_scales_.at(a), exact_scales_.at(b));
        if (a == b) {
            ckks_square_inplace(context_, out);
        } else {
            auto rhs = values_.at(b).clone();
            ckks_drop_to_level_inplace(rhs, level);
            ckks_mul_inplace(context_, out, rhs);
        }
        ckks_relin_hybrid_inplace(context_, out);
        out.setScale(product_scale.to_double(context_.getParams().getModuli()));

        multiply_by_integer_scalar_inplace(
            context_,
            out,
            2.0L,
            product_scale.to_long_double(context_.getParams().getModuli()));

        const size_t c = a > b ? a - b : b - a;
        if (c == 0) {
            add_constant_inplace(
                context_,
                out,
                -1.0,
                product_scale.to_long_double(context_.getParams().getModuli()));
        } else {
            gen_power(c);
            auto correction = values_.at(c).clone();
            const uint64_t exact_ratio =
                exact_scale_ratio_to_u64(
                    product_scale,
                    exact_scales_.at(c),
                    context_.getParams().getModuli());
            correction.setScale(
                exact_scales_.at(c).to_double(context_.getParams().getModuli()));
            const double correction_scale = correction.getScale();
            const size_t correction_level = out.getLevel();
            const size_t correction_polys = out.getNumPolys();
            correction = align_ciphertext_for_add_with_polys(
                std::move(correction),
                correction_level,
                correction_scale,
                correction_polys);
            multiply_by_integer_scalar_inplace(
                context_,
                correction,
                static_cast<long double>(exact_ratio),
                product_scale.to_long_double(context_.getParams().getModuli()));
            ckks_sub_inplace(context_, out, correction);
        }

        values_.insert_or_assign(n, std::move(out));
        exact_scales_.insert_or_assign(n, product_scale);
        return true;
    }

    const CKKSContext& context_;
    std::map<size_t, CKKSCiphertext> values_;
    std::map<size_t, ExactScale> exact_scales_;
};

bool polynomial_is_even(std::span<const double> coeffs) {
    for (size_t i = 1; i < coeffs.size(); i += 2) {
        if (coeffs[i] != 0.0) {
            return false;
        }
    }
    return true;
}

bool polynomial_is_odd(std::span<const double> coeffs) {
    for (size_t i = 0; i < coeffs.size(); i += 2) {
        if (coeffs[i] != 0.0) {
            return false;
        }
    }
    return true;
}

struct BabyStepEvaluation {
    size_t degree = 0;
    CKKSCiphertext value;
};

CKKSCiphertext evaluate_polynomial_from_power_basis(
    const CKKSContext& context,
    size_t target_level,
    const EvaluationPolynomial& polynomial,
    PowerBasis& power_basis) {
    const size_t degree = polynomial.degree();
    const auto& x1 = power_basis.at(1);
    auto result = zero_ciphertext_like(
        context,
        target_level,
        static_cast<double>(polynomial.scale),
        x1.getSecretOwner(),
        x1.getMessageEncodingState());

    if (polynomial.is_even && polynomial.coeffs[0] != 0.0) {
        add_constant_inplace(
            context,
            result,
            polynomial.coeffs[0],
            polynomial.scale);
    }

    if (degree == 0) {
        return result;
    }

    for (size_t exp = degree; exp > 0; --exp) {
        const double coeff = polynomial.coeffs[exp];
        if (coeff == 0.0) {
            continue;
        }
        if (!( (!polynomial.is_even && !polynomial.is_odd) ||
               ((exp & 1U) == 0U && polynomial.is_even) ||
               ((exp & 1U) == 1U && polynomial.is_odd))) {
            continue;
        }

        const size_t result_polys = result.getNumPolys();
        result = align_ciphertext_for_add_with_polys(
            std::move(result),
            target_level,
            static_cast<double>(polynomial.scale),
            result_polys);
        multhenadd_real_scalar_scale_aware_inplace(
            context,
            power_basis.at(exp),
            coeff,
            result);
    }

    return result;
}

BabyStepEvaluation evaluate_baby_step(
    const CKKSContext& context,
    size_t index,
    const PatersonStockmeyerPlan& polynomial,
    PowerBasis& power_basis) {
    const auto& poly = polynomial.value[index];
    return BabyStepEvaluation{
        .degree = poly.degree(),
        .value = evaluate_polynomial_from_power_basis(
            context,
            poly.level,
            poly,
            power_basis),
    };
}

void evaluate_monomial(
    const CKKSContext& context,
    CKKSCiphertext& even,
    CKKSCiphertext& odd,
    const CKKSCiphertext& xpow) {
    if (odd.getNumPolys() == 3) {
        ckks_relin_hybrid_inplace(context, odd);
    }
    ckks_rescale(context, odd);

    auto xpow_aligned = xpow.clone();
    ckks_drop_to_level_inplace(xpow_aligned, odd.getLevel());
    ckks_mul_inplace(context, odd, xpow_aligned);

    const double even_scale = even.getScale();
    const size_t even_level = odd.getLevel();
    const size_t even_polys = odd.getNumPolys();
    auto even_aligned = align_ciphertext_for_add_with_polys(
        std::move(even),
        even_level,
        even_scale,
        even_polys);
    if (!scales_close(even_aligned.getScale(), odd.getScale(), 0x1p-45)) {
        std::ostringstream oss;
        oss << "ckks_evalchebyshev: EvaluateMonomial scale mismatch"
            << " (even=" << even_aligned.getScale()
            << ", odd=" << odd.getScale()
            << ", level=" << odd.getLevel()
            << ")";
        throw std::runtime_error(oss.str());
    }
    const double common_scale = std::max(even_aligned.getScale(), odd.getScale());
    even_aligned.setScale(common_scale);
    odd.setScale(common_scale);
    ckks_add_inplace(context, odd, even_aligned);
}

CKKSCiphertext evaluate_paterson_stockmeyer_plan(
    const CKKSContext& context,
    const PatersonStockmeyerPlan& polynomial,
    PowerBasis& power_basis) {
    const size_t split = polynomial.value.size();
    std::vector<std::optional<BabyStepEvaluation>> baby_steps(split);
    for (size_t i = 0; i < split; ++i) {
        baby_steps[split - i - 1] =
            evaluate_baby_step(context, i, polynomial, power_basis);
    }

    while (baby_steps.size() != 1) {
        std::vector<int> giant_steps(baby_steps.size(), 0);
        for (size_t i = 0; i < baby_steps.size(); ++i) {
            if (i == baby_steps.size() - 1) {
                giant_steps[i] = 2;
            } else if (baby_steps[i]->degree == baby_steps[i + 1]->degree) {
                giant_steps[i] = 1;
                ++i;
            }
        }

        for (size_t i = 0; i < baby_steps.size(); ++i) {
            if (giant_steps[i] == 2) {
                baby_steps[i]->degree = baby_steps[i - 1]->degree;
            } else if (giant_steps[i] == 1) {
                auto& even = *baby_steps[i];
                auto& odd = *baby_steps[i + 1];
                const size_t deg = size_t{1} << bit_length(even.degree);
                evaluate_monomial(
                    context,
                    even.value,
                    odd.value,
                    power_basis.at(deg));
                odd.degree = 2 * deg - 1;
                baby_steps[i].reset();
                ++i;
            }
        }

        size_t write_index = 0;
        for (size_t i = 0; i < baby_steps.size(); ++i) {
            if (baby_steps[i].has_value()) {
                if (write_index != i) {
                    baby_steps[write_index] = std::move(baby_steps[i]);
                }
                ++write_index;
            }
        }
        baby_steps.resize(write_index);
    }

    auto result = std::move(baby_steps.front()->value);
    if (result.getNumPolys() == 3) {
        ckks_relin_hybrid_inplace(context, result);
    }
    ckks_rescale(context, result);
    return result;
}

CKKSCiphertext ckks_evalchebyshev_scale_aware_common(
    const CKKSContext& context,
    const CKKSCiphertext& mapped_x,
    std::span<const double> coeffs,
    size_t degree,
    double target_scale) {
    if (degree == 0) {
        auto out = zero_ciphertext_like(
            context,
            mapped_x.getLevel(),
            target_scale,
            mapped_x.getSecretOwner(),
            mapped_x.getMessageEncodingState());
        add_constant_inplace(context, out, coeffs[0], target_scale);
        return out;
    }

    const size_t polynomial_depth = polynomial_evaluation_depth(degree);
    if (mapped_x.getLevel() < polynomial_depth) {
        throw std::invalid_argument(
            "ckks_evalchebyshev: insufficient level for polynomial depth");
    }

    EvaluationPolynomial polynomial;
    polynomial.coeffs.assign(coeffs.begin(), coeffs.begin() + degree + 1);
    polynomial.max_degree = degree;
    polynomial.lead = true;
    polynomial.is_even = polynomial_is_even(coeffs);
    polynomial.is_odd = polynomial_is_odd(coeffs);

    const size_t log_degree = bit_length(degree);
    const size_t log_split = optimal_power_of_two_split_log(log_degree);

    PowerBasis power_basis(context, mapped_x);
    power_basis.gen_power(size_t{1} << (log_degree - 1));
    for (size_t i = (size_t{1} << log_split) - 1; i > 2; --i) {
        if ((!polynomial.is_even && !polynomial.is_odd) ||
            ((i & 1U) == 0U && polynomial.is_even) ||
            ((i & 1U) == 1U && polynomial.is_odd)) {
            power_basis.gen_power(i);
        }
    }

    auto ps = make_paterson_stockmeyer_plan(
        context,
        polynomial,
        mapped_x.getLevel(),
        static_cast<long double>(mapped_x.getScale()),
        static_cast<long double>(target_scale));

    auto out = evaluate_paterson_stockmeyer_plan(context, ps, power_basis);
    out.setScale(target_scale);
    return out;
}

}  // namespace

CKKSCiphertext ckks_evalchebyshev_preconditioned(
    const CKKSContext& context,
    const CKKSCiphertext& x,
    std::span<const double> chebyshev_coefficients,
    const CKKSEvalPolyOptions& options) {
    const size_t degree = trim_polynomial_degree(chebyshev_coefficients);
    const double target_scale =
        options.target_scale > 0.0 ? options.target_scale : x.getScale();
    validate_evalpoly_common_input(
        context,
        x,
        target_scale);

    const size_t required_depth =
        estimate_preconditioned_chebyshev_eval_depth(degree);
    if (x.getLevel() < required_depth) {
        throw std::invalid_argument("ckks_evalchebyshev_preconditioned: ciphertext level is too small for Chebyshev evaluation depth");
    }

    if (degree == 0) {
        auto out = zero_ciphertext_like(
            context,
            x.getLevel(),
            target_scale,
            x.getSecretOwner(),
            x.getMessageEncodingState());
        add_constant_inplace(context, out, chebyshev_coefficients[0], target_scale);
        return out;
    }
    return ckks_evalchebyshev_scale_aware_common(
        context,
        x,
        chebyshev_coefficients,
        degree,
        target_scale);
}
