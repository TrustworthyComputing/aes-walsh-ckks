#include "modarith.hpp"

BarrettConst barrett_const_computation(uint64_t q) {
    __uint128_t ratio = (~static_cast<__uint128_t>(0)) / q;
    auto high = static_cast<uint64_t>(ratio >> 64);
    auto low = static_cast<uint64_t>(ratio);
    return BarrettConst{low, high};
}

uint64_t compute_shoup(uint64_t w, uint64_t q) {
    __uint128_t t = (static_cast<__uint128_t>(w) << kShoupBits) / q;
    return static_cast<uint64_t>(t);
}

uint64_t powmod(uint64_t a, uint64_t b, uint64_t m) {
    a %= m;
    uint64_t res = 1;
    while (b > 0) {
        if (b & 1) {
            res = naive_mul_mod_u64(res, a, m);
        }
        a = naive_mul_mod_u64(a, a, m);
        b >>= 1;
    }
    return res;
}


uint64_t mod_inverse(uint64_t a, uint64_t m) {
    int64_t m0 = m, t, q;
    int64_t x0 = 0, x1 = 1;
    if (m == 1) return 0;
    while (a > 1) {
        q = a / m;
        t = m; m = a % m, a = t;
        t = x0; x0 = x1 - q * x0; x1 = t;
    }
    if (x1 < 0) x1 += m0;
    return x1;
}
