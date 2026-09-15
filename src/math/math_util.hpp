#ifndef MATH_UTIL_HPP
#define MATH_UTIL_HPP

#include <complex>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cmath>

using Complex = std::complex<double>;

inline bool is_power_of_two(std::size_t num) {
    return std::has_single_bit(num);
}

constexpr unsigned ilog2_pow2(std::uint64_t x) {
        assert(x != 0 && std::has_single_bit(x));
        return static_cast<unsigned>(std::countr_zero(x));
}

inline size_t reverse_bits(size_t value, size_t width){
    size_t result = 0;
    for (size_t i = 0; i < width; i++){
        result = (result << 1) | (value & 1);
        value >>= 1;
    }
    return result;
}

inline Complex cis(double angle) {
    return {std::cos(angle), std::sin(angle)};
}

#endif // MATH_UTIL_HPP
