#include "poly_approx.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <mpfr.h>
#include <stdexcept>
#include <utility>

namespace {

class Big {
public:
    static constexpr mpfr_prec_t kPrecisionBits = 256;

    Big() {
        mpfr_init2(value_, kPrecisionBits);
        mpfr_set_ui(value_, 0, MPFR_RNDN);
    }

    explicit Big(double value) {
        mpfr_init2(value_, kPrecisionBits);
        mpfr_set_d(value_, value, MPFR_RNDN);
    }

    explicit Big(int value) {
        mpfr_init2(value_, kPrecisionBits);
        mpfr_set_si(value_, value, MPFR_RNDN);
    }

    explicit Big(size_t value) {
        mpfr_init2(value_, kPrecisionBits);
        mpfr_set_ui(value_, static_cast<unsigned long>(value), MPFR_RNDN);
    }

    explicit Big(const char* value) {
        mpfr_init2(value_, kPrecisionBits);
        if (mpfr_set_str(value_, value, 10, MPFR_RNDN) != 0) {
            mpfr_clear(value_);
            throw std::invalid_argument("Big: invalid decimal string");
        }
    }

    Big(const Big& other) {
        mpfr_init2(value_, kPrecisionBits);
        mpfr_set(value_, other.value_, MPFR_RNDN);
    }

    Big(Big&& other) noexcept {
        mpfr_init2(value_, kPrecisionBits);
        mpfr_swap(value_, other.value_);
    }

    ~Big() {
        mpfr_clear(value_);
    }

    Big& operator=(const Big& other) {
        if (this != &other) {
            mpfr_set(value_, other.value_, MPFR_RNDN);
        }
        return *this;
    }

    Big& operator=(Big&& other) noexcept {
        if (this != &other) {
            mpfr_swap(value_, other.value_);
        }
        return *this;
    }

    double to_double() const {
        return mpfr_get_d(value_, MPFR_RNDN);
    }

    mpfr_ptr raw() {
        return value_;
    }

    mpfr_srcptr raw() const {
        return value_;
    }

private:
    mpfr_t value_;
};

Big operator+(const Big& lhs, const Big& rhs) {
    Big out;
    mpfr_add(out.raw(), lhs.raw(), rhs.raw(), MPFR_RNDN);
    return out;
}

Big operator-(const Big& lhs, const Big& rhs) {
    Big out;
    mpfr_sub(out.raw(), lhs.raw(), rhs.raw(), MPFR_RNDN);
    return out;
}

Big operator*(const Big& lhs, const Big& rhs) {
    Big out;
    mpfr_mul(out.raw(), lhs.raw(), rhs.raw(), MPFR_RNDN);
    return out;
}

Big operator/(const Big& lhs, const Big& rhs) {
    Big out;
    mpfr_div(out.raw(), lhs.raw(), rhs.raw(), MPFR_RNDN);
    return out;
}

Big operator-(const Big& value) {
    Big out;
    mpfr_neg(out.raw(), value.raw(), MPFR_RNDN);
    return out;
}

Big& operator+=(Big& lhs, const Big& rhs) {
    mpfr_add(lhs.raw(), lhs.raw(), rhs.raw(), MPFR_RNDN);
    return lhs;
}

Big& operator-=(Big& lhs, const Big& rhs) {
    mpfr_sub(lhs.raw(), lhs.raw(), rhs.raw(), MPFR_RNDN);
    return lhs;
}

Big& operator*=(Big& lhs, const Big& rhs) {
    mpfr_mul(lhs.raw(), lhs.raw(), rhs.raw(), MPFR_RNDN);
    return lhs;
}

Big& operator/=(Big& lhs, const Big& rhs) {
    mpfr_div(lhs.raw(), lhs.raw(), rhs.raw(), MPFR_RNDN);
    return lhs;
}

bool operator>(const Big& lhs, const Big& rhs) {
    return mpfr_cmp(lhs.raw(), rhs.raw()) > 0;
}

Big big_abs(const Big& value) {
    Big out;
    mpfr_abs(out.raw(), value.raw(), MPFR_RNDN);
    return out;
}

Big big_cos(const Big& value) {
    Big out;
    mpfr_cos(out.raw(), value.raw(), MPFR_RNDN);
    return out;
}

constexpr double pi() {
    return 3.141592653589793238462643383279502884;
}

double evaluate_chebyshev_basis_polynomial(
    const std::vector<double>& coeffs,
    double x,
    double interval_min,
    double interval_max);

Big big_pi() {
    Big out;
    mpfr_const_pi(out.raw(), MPFR_RNDN);
    return out;
}

Big big_pow2(size_t exponent) {
    Big out;
    mpfr_set_ui_2exp(out.raw(), 1, exponent, MPFR_RNDN);
    return out;
}

double map_interval_to_chebyshev(double x, double interval_min, double interval_max) {
    return (2.0 * x - interval_min - interval_max) / (interval_max - interval_min);
}

double evaluate_chebyshev_series(
    const std::vector<double>& coeffs,
    double t) {
    double b_k_plus_one = 0.0;
    double b_k_plus_two = 0.0;
    for (size_t k = coeffs.size(); k > 1; --k) {
        const double b_k =
            2.0 * t * b_k_plus_one - b_k_plus_two + coeffs[k - 1];
        b_k_plus_two = b_k_plus_one;
        b_k_plus_one = b_k;
    }
    return coeffs.empty()
        ? 0.0
        : coeffs[0] + t * b_k_plus_one - b_k_plus_two;
}
// The multi-interval cosine approximation follows:
// K. Han and D. Ki, "Better Bootstrapping for Approximate Homomorphic
// Encryption," CT-RSA 2020, pp. 364-390.
// DOI: 10.1007/978-3-030-40186-3_16
// IACR ePrint: https://eprint.iacr.org/2019/688
void validate_han_ki_input(
    int K,
    double interval_radius,
    size_t degree,
    size_t double_angle,
    double center_spacing,
    size_t validation_points_per_interval) {
    if (K <= 0) {
        throw std::invalid_argument("han_ki_discrete_cosine_approximate: K must be positive");
    }
    if (!std::isfinite(center_spacing) || center_spacing <= 0.0) {
        throw std::invalid_argument("han_ki_discrete_cosine_approximate: center spacing must be positive");
    }
    if (!std::isfinite(interval_radius) ||
        interval_radius <= 0.0 ||
        interval_radius > 0.5 * center_spacing) {
        throw std::invalid_argument("han_ki_discrete_cosine_approximate: interval radius must be in (0, center_spacing/2]");
    }
    if (degree == 0) {
        throw std::invalid_argument("han_ki_discrete_cosine_approximate: degree must be positive");
    }
    if (double_angle > 60) {
        throw std::invalid_argument("han_ki_discrete_cosine_approximate: double_angle is unreasonably large");
    }
    if (validation_points_per_interval < 2) {
        throw std::invalid_argument("han_ki_discrete_cosine_approximate: validation_points_per_interval must be at least 2");
    }
}

std::pair<std::vector<int>, int> han_ki_gen_degrees(
    int degree,
    int interval_K,
    double dev) {
    const int degree_bound = degree + 1;
    int total_degree = 2 * interval_K - 1;
    const double error = 1.0 / dev;

    std::vector<int> interval_degrees(interval_K, 1);
    std::vector<double> bounds(interval_K, 0.0);

    double temp = 0.0;
    for (int i = 1; i <= (2 * interval_K - 1); ++i) {
        temp -= std::log2(static_cast<double>(i));
    }
    temp += (2.0 * static_cast<double>(interval_K) - 1.0) *
            std::log2(2.0 * pi());
    temp += std::log2(error);

    for (int i = 0; i < interval_K; ++i) {
        bounds[i] = temp;
        for (int j = 1; j <= interval_K - 1 - i; ++j) {
            bounds[i] += std::log2(static_cast<double>(j) + error);
        }
        for (int j = 1; j <= interval_K - 1 + i; ++j) {
            bounds[i] += std::log2(static_cast<double>(j) + error);
        }
    }

    constexpr int max_iterations = 200;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        if (total_degree >= degree_bound) {
            break;
        }

        const int max_index = static_cast<int>(
            std::distance(
                bounds.begin(),
                std::max_element(bounds.begin(), bounds.end())));

        if (max_index != 0) {
            if (total_degree + 2 > degree_bound) {
                break;
            }

            for (int i = 0; i < interval_K; ++i) {
                bounds[i] -= std::log2(static_cast<double>(total_degree + 1));
                bounds[i] -= std::log2(static_cast<double>(total_degree + 2));
                bounds[i] += 2.0 * std::log2(2.0 * pi());

                if (i != max_index) {
                    bounds[i] += std::log2(std::abs(static_cast<double>(i - max_index)) + error);
                    bounds[i] += std::log2(static_cast<double>(i + max_index) + error);
                } else {
                    bounds[i] += std::log2(error) - 1.0;
                    bounds[i] += std::log2(2.0 * static_cast<double>(i) + error);
                }
            }

            total_degree += 2;
        } else {
            bounds[0] -= std::log2(static_cast<double>(total_degree + 1));
            bounds[0] += std::log2(error) - 1.0;
            bounds[0] += std::log2(2.0 * pi());
            for (int i = 1; i < interval_K; ++i) {
                bounds[i] -= std::log2(static_cast<double>(total_degree + 1));
                bounds[i] += std::log2(2.0 * pi());
                bounds[i] += std::log2(static_cast<double>(i) + error);
            }

            ++total_degree;
        }

        ++interval_degrees[max_index];
    }

    return {std::move(interval_degrees), total_degree};
}

template <typename TargetFn>
std::pair<std::vector<Big>, std::vector<Big>> han_ki_gen_nodes(
    const std::vector<int>& interval_degrees,
    double dev,
    int total_degree,
    int interval_K,
    double center_spacing,
    size_t double_angle,
    TargetFn&& target) {
    const Big interval_size = Big(1) / Big(dev);
    const Big spacing(center_spacing);

    std::vector<Big> nodes(total_degree, Big(0));
    int count = (interval_degrees[0] & 1) != 0 ? 1 : 0;

    for (int i = interval_K - 1; i > 0; --i) {
        const Big two_degree = Big(2 * interval_degrees[i]);
        for (int j = 0; j < interval_degrees[i]; ++j) {
            Big theta = big_pi() * Big(2 * j);
            theta /= two_degree;
            Big local = big_cos(theta) * interval_size;
            nodes[count] = (Big(i) + local) * spacing;
            ++count;
            nodes[count] = -nodes[count - 1];
            ++count;
        }
    }

    const Big two_degree_center = Big(2 * interval_degrees[0]);
    for (int j = 0; j < interval_degrees[0] / 2; ++j) {
        Big theta = big_pi() * Big(2 * j);
        theta /= two_degree_center;
        Big local = big_cos(theta) * interval_size;
        nodes[count] = local * spacing;
        ++count;
        nodes[count] = -nodes[count - 1];
        ++count;
    }

    std::vector<Big> values(total_degree);
    for (int i = 0; i < total_degree; ++i) {
        values[i] = target(nodes[i], double_angle);
    }

    return {std::move(nodes), std::move(values)};
}

std::vector<Big> han_ki_solve_system(
    int total_degree,
    int interval_K,
    size_t double_angle,
    double center_spacing,
    const std::vector<Big>& nodes_in,
    const std::vector<Big>& y_in) {
    std::vector<Big> nodes = nodes_in;
    std::vector<Big> y = y_in;

    Big tmp;
    for (int j = 1; j < total_degree; ++j) {
        for (int i = 0; i < total_degree - j; ++i) {
            y[i] = y[i + 1] - y[i];
            tmp = nodes[i + j] - nodes[i];
            y[i] /= tmp;
        }
    }

    const int system_size = total_degree + 1;
    const Big scale_factor = big_pow2(double_angle);
    const Big k_over_scale = Big(interval_K) * Big(center_spacing) / scale_factor;

    std::vector<Big> x(system_size);
    for (int i = 0; i < system_size; ++i) {
        Big theta = big_pi() * Big(i);
        theta /= Big(system_size - 1);
        x[i] = k_over_scale * big_cos(theta);
    }

    std::vector<Big> p(system_size);
    for (int i = 0; i < system_size; ++i) {
        p[i] = y[0];
        for (int j = 1; j < system_size - 1; ++j) {
            p[i] *= (x[i] - nodes[j]);
            p[i] += y[j];
        }
    }

    std::vector<std::vector<Big>> T(system_size, std::vector<Big>(system_size, Big(0)));
    for (int i = 0; i < system_size; ++i) {
        T[i][0] = Big(1);
        T[i][1] = x[i] / k_over_scale;
        for (int j = 2; j < system_size; ++j) {
            const Big x_normalized = x[i] / k_over_scale;
            T[i][j] = Big(2) * x_normalized * T[i][j - 1] - T[i][j - 2];
        }
    }

    for (int i = 0; i < system_size - 1; ++i) {
        int max_index = i;
        Big max_abs = big_abs(T[i][i]);
        for (int j = i + 1; j < system_size; ++j) {
            const Big candidate = big_abs(T[j][i]);
            if (candidate > max_abs) {
                max_abs = candidate;
                max_index = j;
            }
        }

        if (i != max_index) {
            std::swap(T[i], T[max_index]);
            std::swap(p[i], p[max_index]);
        }

        for (int j = i + 1; j < system_size; ++j) {
            T[i][j] /= T[i][i];
        }
        p[i] /= T[i][i];
        T[i][i] = Big(1);

        for (int j = i + 1; j < system_size; ++j) {
            tmp = T[j][i] * p[i];
            p[j] -= tmp;
            for (int l = i + 1; l < system_size; ++l) {
                tmp = T[j][i] * T[i][l];
                T[j][l] -= tmp;
            }
            T[j][i] = Big(0);
        }
    }

    std::vector<Big> coefficients(system_size);
    coefficients[system_size - 1] = p[system_size - 1];
    for (int i = system_size - 2; i >= 0; --i) {
        coefficients[i] = p[i];
        for (int j = i + 1; j < system_size; ++j) {
            coefficients[i] -= T[i][j] * coefficients[j];
        }
    }

    coefficients.resize(total_degree);
    return coefficients;
}

template <typename TargetFn, typename DoubleTargetFn>
PolynomialApproximation han_ki_discrete_cosine_approximate_impl(
    TargetFn&& big_target,
    DoubleTargetFn&& double_target,
    int K,
    double interval_radius,
    size_t degree,
    size_t double_angle,
    double center_spacing,
    size_t validation_points_per_interval) {
    validate_han_ki_input(
        K,
        interval_radius,
        degree,
        double_angle,
        center_spacing,
        validation_points_per_interval);

    const int interval_K = K;
    const int center_index_bound = K - 1;
    const double local_radius = interval_radius / center_spacing;
    const double dev = 1.0 / local_radius;
    auto [interval_degrees, total_degree] = han_ki_gen_degrees(
        static_cast<int>(degree),
        interval_K,
        dev);
    auto [nodes, values] = han_ki_gen_nodes(
        interval_degrees,
        dev,
        total_degree,
        interval_K,
        center_spacing,
        double_angle,
        std::forward<TargetFn>(big_target));
    auto big_coeffs = han_ki_solve_system(
        total_degree,
        interval_K,
        double_angle,
        center_spacing,
        nodes,
        values);

    std::vector<double> chebyshev_coeffs(big_coeffs.size(), 0.0);
    for (size_t i = 0; i < big_coeffs.size(); ++i) {
        chebyshev_coeffs[i] = big_coeffs[i].to_double();
    }

    const double interval_scale =
        std::pow(2.0, static_cast<double>(double_angle));
    const double interval_min =
        -center_spacing * static_cast<double>(interval_K) / interval_scale;
    const double interval_max =
        center_spacing * static_cast<double>(interval_K) / interval_scale;

    double max_error = 0.0;
    for (int center_index = -center_index_bound;
         center_index <= center_index_bound;
         ++center_index) {
        const double center = center_spacing * static_cast<double>(center_index);
        for (size_t j = 0; j < validation_points_per_interval; ++j) {
            const double u =
                static_cast<double>(j) /
                static_cast<double>(validation_points_per_interval - 1);
            const double x = center + interval_radius * (2.0 * u - 1.0);
            const double expected = double_target(x);
            const double actual = evaluate_chebyshev_basis_polynomial(
                chebyshev_coeffs,
                x,
                interval_min,
                interval_max);
            if (!std::isfinite(expected) || !std::isfinite(actual)) {
                throw std::runtime_error("han_ki_discrete_cosine_approximate: non-finite validation value");
            }
            max_error = std::max(max_error, std::abs(actual - expected));
        }
    }

    return PolynomialApproximation{
        .chebyshev_coeffs = std::move(chebyshev_coeffs),
        .max_error = max_error,
        .interval_min = interval_min,
        .interval_max = interval_max,
        .degree = big_coeffs.empty() ? 0 : big_coeffs.size() - 1,
    };
}

}  // namespace

namespace {

double evaluate_chebyshev_basis_polynomial(
    const std::vector<double>& coeffs,
    double x,
    double interval_min,
    double interval_max) {
    if (!std::isfinite(interval_min) ||
        !std::isfinite(interval_max) ||
        !(interval_min < interval_max)) {
        throw std::invalid_argument("evaluate_chebyshev_basis_polynomial: invalid interval");
    }
    const double t = map_interval_to_chebyshev(x, interval_min, interval_max);
    return evaluate_chebyshev_series(coeffs, t);
}

double binboot_evalmod_target(double x, double period) {
    if (!std::isfinite(period) || period <= 0.0) {
        throw std::invalid_argument("binboot_evalmod_target: period must be positive");
    }
    return 0.5 * (1.0 - std::cos(2.0 * pi() * x / period));
}

}  // namespace

PolynomialApproximation binboot_han_ki_discrete_multi_interval_approximate(
    double K,
    size_t degree,
    size_t double_angle,
    double interval_radius,
    size_t validation_points_per_interval,
    double coefficient_prune_threshold) {
    if (!std::isfinite(K) || K <= 0.0) {
        throw std::invalid_argument(
            "binboot_han_ki_discrete_multi_interval_approximate: K must be positive");
    }
    if (double_angle > 30) {
        throw std::invalid_argument(
            "binboot_han_ki_discrete_multi_interval_approximate: double_angle is too large");
    }
    if (!std::isfinite(interval_radius) ||
        interval_radius <= 0.0 ||
        interval_radius >= 0.25) {
        throw std::invalid_argument(
            "binboot_han_ki_discrete_multi_interval_approximate: interval_radius must be in (0, 0.25)");
    }
    if (validation_points_per_interval < 2) {
        throw std::invalid_argument(
            "binboot_han_ki_discrete_multi_interval_approximate: validation_points_per_interval must be at least 2");
    }
    if (!std::isfinite(coefficient_prune_threshold) ||
        coefficient_prune_threshold < 0.0) {
        throw std::invalid_argument(
            "binboot_han_ki_discrete_multi_interval_approximate: coefficient prune threshold must be finite and non-negative");
    }

    const int interval_K = static_cast<int>(std::llround(2.0 * K));
    if (std::abs(static_cast<double>(interval_K) - 2.0 * K) >
        64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, K)) {
        throw std::invalid_argument(
            "binboot_han_ki_discrete_multi_interval_approximate: K must be an integer or half-integer");
    }
    if (interval_K <= 0) {
        throw std::invalid_argument(
            "binboot_han_ki_discrete_multi_interval_approximate: invalid interval K");
    }

    const double center_spacing = 0.5 / K;
    const double normalized_interval_radius = interval_radius / K;
    auto approximation = han_ki_discrete_cosine_approximate_impl(
        [K](Big& z, size_t scnum) {
            const Big scaled_z = z / big_pow2(scnum);
            return Big(0.5) -
                   Big(0.5) * big_cos(Big(2) * big_pi() * Big(K) * scaled_z);
        },
        [K, double_angle](double u) {
            const double shrink = std::ldexp(1.0, static_cast<int>(double_angle));
            return binboot_evalmod_target(u, shrink / K);
        },
        interval_K,
        normalized_interval_radius,
        degree,
        double_angle,
        center_spacing,
        validation_points_per_interval);

    std::vector<double> chebyshev_coeffs = std::move(approximation.chebyshev_coeffs);
    for (size_t i = 1; i < chebyshev_coeffs.size(); i += 2) {
        chebyshev_coeffs[i] = 0.0;
    }

    size_t pruned_coeff_count = 0;
    for (double& coeff : chebyshev_coeffs) {
        if (coeff != 0.0 && std::abs(coeff) < coefficient_prune_threshold) {
            coeff = 0.0;
            ++pruned_coeff_count;
        }
    }
    std::cout << "binboot_han_ki_discrete_multi_interval_approximate: pruned "
              << pruned_coeff_count
              << " coefficients below "
              << coefficient_prune_threshold << '\n';

    const int half_center_bound = interval_K - 1;
    double max_error = 0.0;
    for (int center_index = -half_center_bound;
         center_index <= half_center_bound;
         ++center_index) {
        const double center_x = 0.5 * static_cast<double>(center_index);
        for (size_t j = 0; j < validation_points_per_interval; ++j) {
            const double alpha =
                static_cast<double>(j) /
                static_cast<double>(validation_points_per_interval - 1);
            const double x = center_x + interval_radius * (2.0 * alpha - 1.0);
            const double u = x / K;
            const double expected = binboot_evalmod_target(u, 1.0 / K);
            double actual = evaluate_chebyshev_basis_polynomial(
                chebyshev_coeffs,
                u,
                approximation.interval_min,
                approximation.interval_max);
            for (size_t angle = 0; angle < double_angle; ++angle) {
                actual = 4.0 * actual * (1.0 - actual);
            }
            if (!std::isfinite(expected) || !std::isfinite(actual)) {
                throw std::runtime_error(
                    "binboot_han_ki_discrete_multi_interval_approximate: non-finite validation value");
            }
            max_error = std::max(max_error, std::abs(actual - expected));
        }
    }
    const size_t output_degree = chebyshev_coeffs.empty()
        ? 0
        : chebyshev_coeffs.size() - 1;
    return PolynomialApproximation{
        .chebyshev_coeffs = std::move(chebyshev_coeffs),
        .max_error = max_error,
        .interval_min = approximation.interval_min,
        .interval_max = approximation.interval_max,
        .degree = output_degree,
    };
}
