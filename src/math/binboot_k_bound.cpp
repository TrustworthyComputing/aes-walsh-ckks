#include "binboot_k_bound.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

long double binomial_coefficient_ld(size_t n, size_t k) {
    if (k > n) {
        return 0.0L;
    }
    k = std::min(k, n - k);
    long double out = 1.0L;
    for (size_t i = 1; i <= k; ++i) {
        out *= static_cast<long double>(n - k + i);
        out /= static_cast<long double>(i);
    }
    return out;
}

long double irwin_hall_cdf(long double x, size_t h) {
    if (h == 0) {
        throw std::invalid_argument("binboot K bound: Hamming weight must be positive");
    }
    if (x <= 0.0L) {
        return 0.0L;
    }
    if (x >= static_cast<long double>(h)) {
        return 1.0L;
    }

    const size_t upper = static_cast<size_t>(std::floor(x));
    long double sum = 0.0L;
    for (size_t k = 0; k <= upper; ++k) {
        const long double term =
            binomial_coefficient_ld(h, k) *
            std::pow(x - static_cast<long double>(k), static_cast<long double>(h));
        sum += (k & 1U) == 0U ? term : -term;
    }
    return sum / std::tgammal(static_cast<long double>(h) + 1.0L);
}

long double shifted_irwin_hall_abs_tail_probability(size_t h, size_t K) {
    const long double half_h = static_cast<long double>(h) / 2.0L;
    if (static_cast<long double>(K) >= half_h) {
        return 0.0L;
    }
    return 2.0L * irwin_hall_cdf(half_h - static_cast<long double>(K), h);
}

}  // namespace

BinBootKBound required_binboot_k_bound(
    size_t coefficient_count,
    size_t hamming_weight,
    long double target_failure_probability) {
    if (coefficient_count == 0 || hamming_weight == 0) {
        throw std::invalid_argument("binboot K bound: invalid coefficient count or Hamming weight");
    }
    if (!std::isfinite(target_failure_probability) ||
        target_failure_probability <= 0.0L ||
        target_failure_probability >= 1.0L) {
        throw std::invalid_argument("binboot K bound: invalid failure probability");
    }

    const size_t max_k = (hamming_weight + 1) / 2;
    for (size_t K = 0; K <= max_k; ++K) {
        const long double per_coeff =
            shifted_irwin_hall_abs_tail_probability(hamming_weight, K);
        const long double total_failure =
            std::min(
                1.0L,
                static_cast<long double>(coefficient_count) * per_coeff);
        if (total_failure <= target_failure_probability) {
            return BinBootKBound{
                .mathematical_k = K,
                .evalmod_k = K + 1,
                .failure_probability = total_failure,
            };
        }
    }

    return BinBootKBound{
        .mathematical_k = max_k,
        .evalmod_k = max_k + 1,
        .failure_probability = 0.0L,
    };
}
