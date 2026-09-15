// SPDX-License-Identifier: BSD-2-Clause
//
// Portions adapted from OpenFHE's DiscreteGaussianGeneratorImpl.
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors.

#ifndef DISCRETE_GAUSSIAN_SAMPLER_HPP
#define DISCRETE_GAUSSIAN_SAMPLER_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

// Adapted from OpenFHE's DiscreteGaussianGeneratorImpl. OpenFHE uses a PRNG
// abstraction; this standalone version seeds a local std::mt19937_64 once and
// reuses it for each draw.
class DiscreteGaussianSampler {
public:
    static constexpr double karney_threshold = 300.0;

    explicit DiscreteGaussianSampler(double stddev = 1.0) {
        set_std(stddev);
    }

    double get_std() const {
        return stddev_;
    }

    void set_std(double stddev) {
        if (stddev <= 0.0 || std::isinf(stddev) || std::isnan(stddev)) {
            throw std::invalid_argument("DiscreteGaussianSampler: invalid standard deviation");
        }
        if (std::log2(stddev) > 59) {
            throw std::invalid_argument("DiscreteGaussianSampler: standard deviation cannot exceed 59 bits");
        }

        stddev_ = stddev;
        use_peikert_ = (stddev_ < karney_threshold);
        if (use_peikert_) {
            initialize_peikert();
        }
    }

    int64_t generate_int() const {
        if (!use_peikert_) {
            return generate_integer_karney(0.0, stddev_);
        }

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const double seed = dist(generator_) - 0.5;
        const double tmp = std::abs(seed) - a_ / 2.0;
        return (tmp <= 0.0) ? 0 : find_in_vector(vals_, tmp) * (seed > 0.0 ? 1 : -1);
    }

private:
    void initialize_peikert() {
        // OpenFHE uses M = sqrt(-2 * log(5e-32)), giving tail probability near 2^-100.
        constexpr double bound_multiplier = 12.00610553538285;
        const int64_t fin = static_cast<int64_t>(std::ceil(stddev_ * bound_multiplier));
        vals_.resize(static_cast<std::size_t>(fin));

        const double variance = 2.0 * stddev_ * stddev_;
        double cumulative = 0.0;
        for (int64_t x = 1; x <= fin; ++x) {
            cumulative += std::exp(-(static_cast<double>(x * x) / variance));
            vals_[static_cast<std::size_t>(x - 1)] = cumulative;
        }

        a_ = 1.0 / (2.0 * cumulative + 1.0);
        for (double& value : vals_) {
            value *= a_;
        }
    }

    static int64_t find_in_vector(const std::vector<double>& values, double search) {
        const auto lower = std::lower_bound(values.begin(), values.end(), search);
        if (lower == values.end()) {
            throw std::runtime_error("DiscreteGaussianSampler: inversion sample out of CDF range");
        }
        return static_cast<int64_t>(lower - values.begin() + 1);
    }

    int64_t generate_integer_karney(double mean, double stddev) const {
        std::uniform_int_distribution<int64_t> uniform_sign(0, 1);
        std::uniform_int_distribution<int64_t> uniform_j(0, static_cast<int64_t>(std::ceil(stddev)) - 1);

        while (true) {
            const int32_t k = algorithm_g();
            if (!algorithm_p(k * (k - 1))) {
                continue;
            }

            int64_t s = uniform_sign(generator_);
            if (s == 0) {
                s = -1;
            }

            const double di0 = stddev * k + s * mean;
            const int64_t i0 = static_cast<int64_t>(std::ceil(di0));
            const double x0 = (i0 - di0) / stddev;
            const int64_t j = uniform_j(generator_);
            const double x = x0 + static_cast<double>(j) / stddev;

            if (!(x < 1.0) || (x == 0.0 && s < 0 && k == 0)) {
                continue;
            }

            int32_t h = k + 1;
            while (h-- && algorithm_b(k, x)) {
            }
            if (h < 0) {
                return s * (i0 + j);
            }
        }
    }

    bool algorithm_p(int n) const {
        while (n-- && algorithm_h()) {
        }
        return n < 0;
    }

    int32_t algorithm_g() const {
        int32_t n = 0;
        while (algorithm_h()) {
            ++n;
        }
        return n;
    }

    bool algorithm_h() const {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float h_a = dist(generator_);

        if (h_a > 0.5f) {
            return true;
        }
        if (h_a < 0.5f) {
            for (;;) {
                const float h_b = dist(generator_);
                if (h_b > h_a) {
                    return false;
                }
                if (h_b < h_a) {
                    h_a = dist(generator_);
                } else {
                    return algorithm_h_double();
                }

                if (h_a > h_b) {
                    return true;
                }
                if (h_a == h_b) {
                    return algorithm_h_double();
                }
            }
        }
        return algorithm_h_double();
    }

    bool algorithm_h_double() const {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double h_a = dist(generator_);
        if (!(h_a < 0.5)) {
            return true;
        }
        for (;;) {
            const double h_b = dist(generator_);
            if (!(h_b < h_a)) {
                return false;
            }
            h_a = dist(generator_);
            if (!(h_a < h_b)) {
                return true;
            }
        }
    }

    bool algorithm_b(int32_t k, double x) const {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        float y = static_cast<float>(x);
        const int32_t m = 2 * k + 2;
        int32_t n = 0;
        for (;; ++n) {
            const float z = dist(generator_);
            if (z > y) {
                break;
            }
            if (z < y) {
                const float r = dist(generator_);
                const float threshold = static_cast<float>((2 * k + x) / m);
                if (r > threshold) {
                    break;
                }
                if (r < threshold) {
                    y = z;
                } else {
                    return algorithm_b_double(k, x);
                }
            } else {
                return algorithm_b_double(k, x);
            }
        }

        return (n % 2) == 0;
    }

    bool algorithm_b_double(int32_t k, double x) const {
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        double y = x;
        const int32_t m = 2 * k + 2;
        int32_t n = 0;
        for (;; ++n) {
            const double z = dist(generator_);
            if (!(z < y)) {
                break;
            }
            const double r = dist(generator_);
            if (!(r < (2 * k + x) / m)) {
                break;
            }
            y = z;
        }

        return (n % 2) == 0;
    }

    double stddev_ = 1.0;
    double a_ = 0.0;
    std::vector<double> vals_;
    bool use_peikert_ = false;
    mutable std::mt19937_64 generator_{[] {
        std::random_device rd;
        std::seed_seq seed{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        return std::mt19937_64(seed);
    }()};
};

#endif // DISCRETE_GAUSSIAN_SAMPLER_HPP
