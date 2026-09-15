#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <cstdint>
#include <utility>
#include <vector>

std::pair<std::vector<uint64_t>, std::vector<uint64_t>> find_p_and_q_primes(
    const std::vector<uint64_t>& p_bits_per_prime,
    const std::vector<uint64_t>& q_bits_per_prime,
    uint64_t N);

#endif
