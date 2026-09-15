#ifndef POLY_APPROX_HPP
#define POLY_APPROX_HPP

#include <cstddef>
#include <vector>

struct PolynomialApproximation {
    std::vector<double> chebyshev_coeffs;
    double max_error = 0.0;
    double interval_min = 0.0;
    double interval_max = 0.0;
    size_t degree = 0;
};

PolynomialApproximation binboot_han_ki_discrete_multi_interval_approximate(
    double K,
    size_t degree,
    size_t double_angle,
    double interval_radius,
    size_t validation_points_per_interval = 64,
    double coefficient_prune_threshold = 1e-10);

#endif // POLY_APPROX_HPP
