#ifndef BINBOOT_K_BOUND_HPP
#define BINBOOT_K_BOUND_HPP

#include <cstddef>

struct BinBootKBound {
    size_t mathematical_k = 0;
    size_t evalmod_k = 0;
    long double failure_probability = 0.0L;
};

BinBootKBound required_binboot_k_bound(
    size_t coefficient_count,
    size_t hamming_weight,
    long double target_failure_probability);

#endif // BINBOOT_K_BOUND_HPP
