#include "random.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>

namespace {

uint64_t modular_power(uint64_t base, uint64_t exponent, uint64_t modulus) {
    uint64_t result = 1;
    base %= modulus;
    while (exponent != 0) {
        if ((exponent & 1) != 0) {
            result = static_cast<uint64_t>(
                static_cast<__uint128_t>(result) * base % modulus);
        }
        base = static_cast<uint64_t>(
            static_cast<__uint128_t>(base) * base % modulus);
        exponent >>= 1;
    }
    return result;
}

bool is_composite(uint64_t n, uint64_t base, uint64_t odd_part, int power_of_two) {
    uint64_t value = modular_power(base, odd_part, n);
    if (value == 1 || value == n - 1) {
        return false;
    }
    for (int round = 1; round < power_of_two; ++round) {
        value = static_cast<uint64_t>(
            static_cast<__uint128_t>(value) * value % n);
        if (value == n - 1) {
            return false;
        }
    }
    return true;
}

bool is_prime(uint64_t n) {
    if (n < 2) {
        return false;
    }

    int power_of_two = 0;
    uint64_t odd_part = n - 1;
    while ((odd_part & 1) == 0) {
        odd_part >>= 1;
        ++power_of_two;
    }

    static constexpr std::array<uint64_t, 12> bases = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (uint64_t base : bases) {
        if (n == base) {
            return true;
        }
        if (is_composite(n, base, odd_part, power_of_two)) {
            return false;
        }
    }
    return true;
}

uint64_t prime_search_start(uint64_t bits, uint64_t N) {
    return (1ULL << (bits - 1)) / (2 * N);
}

uint64_t prime_search_end(uint64_t bits, uint64_t N) {
    return (1ULL << bits) / (2 * N);
}

uint64_t find_next_unused_prime(
    uint64_t bits,
    uint64_t N,
    std::map<uint64_t, uint64_t>& next_k_by_bits,
    std::set<uint64_t>& selected_primes) {
    const uint64_t start = prime_search_start(bits, N);
    uint64_t& next_k = next_k_by_bits[bits];
    if (next_k == 0) {
        next_k = prime_search_end(bits, N);
    }

    while (next_k-- > start) {
        const uint64_t candidate = next_k * 2 * N + 1;
        if (!selected_primes.insert(candidate).second) {
            continue;
        }
        if (is_prime(candidate)) {
            return candidate;
        }
        selected_primes.erase(candidate);
    }

    throw std::runtime_error("No unused prime found in the specified range.");
}

std::vector<uint64_t> find_primes(
    const std::vector<uint64_t>& bits_per_prime,
    uint64_t N) {
    std::vector<uint64_t> primes;
    primes.reserve(bits_per_prime.size());
    std::map<uint64_t, uint64_t> next_k_by_bits;
    std::set<uint64_t> selected_primes;
    for (uint64_t bits : bits_per_prime) {
        primes.emplace_back(
            find_next_unused_prime(bits, N, next_k_by_bits, selected_primes));
    }
    return primes;
}

}  // namespace

std::pair<std::vector<uint64_t>, std::vector<uint64_t>> find_p_and_q_primes(
    const std::vector<uint64_t>& p_bits_per_prime,
    const std::vector<uint64_t>& q_bits_per_prime,
    uint64_t N) {
    std::vector<uint64_t> all_bits;
    all_bits.reserve(p_bits_per_prime.size() + q_bits_per_prime.size());
    all_bits.insert(all_bits.end(), p_bits_per_prime.begin(), p_bits_per_prime.end());
    all_bits.insert(all_bits.end(), q_bits_per_prime.begin(), q_bits_per_prime.end());

    std::vector<uint64_t> all_primes = find_primes(all_bits, N);
    const auto q_begin =
        all_primes.begin() + static_cast<std::ptrdiff_t>(p_bits_per_prime.size());

    std::vector<uint64_t> p_primes(all_primes.begin(), q_begin);
    std::vector<uint64_t> q_primes(q_begin, all_primes.end());
    return {std::move(p_primes), std::move(q_primes)};
}
