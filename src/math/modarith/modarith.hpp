#ifndef MODARITH_HPP
#define MODARITH_HPP

#include <cstdint>

struct BarrettConst {
    uint64_t low;
    uint64_t high;
};

inline constexpr int kBarrettBits = 128;
inline constexpr int kShoupBits = 64;

BarrettConst barrett_const_computation(uint64_t q);

inline uint64_t barrett_reduce(uint64_t a, uint64_t b, uint64_t q, const BarrettConst* mu) {
    __uint128_t z = static_cast<__uint128_t>(a) * b;
    uint64_t z_lo = static_cast<uint64_t>(z);
    uint64_t z_hi = static_cast<uint64_t>(z >> 64);

    __uint128_t t = static_cast<__uint128_t>(z_hi) * mu->low +
                    (static_cast<__uint128_t>(z_lo) * mu->low >> 64);
    uint64_t carry = static_cast<uint64_t>(t >> 64);

    uint64_t q_hat = static_cast<uint64_t>(
        (static_cast<__uint128_t>(z_hi) * mu->high +
         (static_cast<__uint128_t>(z_lo) * mu->high >> 64) + carry));

    uint64_t res = z_lo - q_hat * q;
    if (res >= q) res -= q;
    return res;
}

inline uint64_t reduce_u64_mod_q_shoup(uint64_t x, uint64_t q, uint64_t one_shoup) {
    __uint128_t t = static_cast<__uint128_t>(x) * one_shoup;
    uint64_t u = static_cast<uint64_t>(t >> kShoupBits);
    uint64_t r = x - u * q;
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return r;
}

inline uint64_t reduce_mod_4q(uint64_t x, uint64_t q) {
    // NTT kernels in this repo use lazy reductions; x is expected to be < 4q.
    const uint64_t two_q = q << 1;
    if (x >= two_q) x -= two_q;
    if (x >= q) x -= q;
    return x;
}

// Specialized Barrett reduction for a single 64-bit value: computes a mod q.
// This avoids forming the full 128-bit product z = a * b used by the generic path.
inline uint64_t barrett_reduce_u64(uint64_t a, uint64_t q, const BarrettConst* mu) {
    const uint64_t q_hat = static_cast<uint64_t>((static_cast<__uint128_t>(a) * mu->high) >> 64);
    uint64_t res = a - q_hat * q;
    if (res >= q) res -= q;
    return res;
}

inline uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t q, const BarrettConst* barrett_const) {
    return barrett_reduce(a, b, q, barrett_const);
}
inline uint64_t naive_mul_mod_u64(uint64_t a, uint64_t b, uint64_t q) {
    __uint128_t z = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return static_cast<uint64_t>(z % q);
}

// Fast modular add for canonical residues a,b in [0,q).
inline uint64_t add_mod_q(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t res = a + b;
    if (res >= q) res -= q;
    return res;
}

// Fast modular subtract for canonical residues a,b in [0,q).
// Branchless: adds q back only if the subtraction underflowed.
inline uint64_t sub_mod_q(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t r = a - b;
    uint64_t mask = (uint64_t)-(a < b);
    return r + (mask & q);
}

inline uint64_t reduce_i64_mod_q(int64_t value, uint64_t q) {
    const int64_t q_i64 = static_cast<int64_t>(q);
    int64_t reduced = value % q_i64;
    if (reduced < 0) {
        reduced += q_i64;
    }
    return static_cast<uint64_t>(reduced);
}

uint64_t compute_shoup(uint64_t w, uint64_t q);
inline uint64_t mul_mod_shoup(uint64_t a, uint64_t w, uint64_t w_shoup, uint64_t q) {
    __uint128_t t = static_cast<__uint128_t>(a) * w_shoup;
    uint64_t u = static_cast<uint64_t>(t >> kShoupBits);
    uint64_t res = a * w - u * q;
    if (res >= q) res -= q;
    return res;
}

inline uint64_t ckks_montgomery_neg_inverse64(uint64_t q) {
    uint64_t inv = 1;
    for (int i = 0; i < 6; ++i) {
        inv *= uint64_t{2} - q * inv;
    }
    return uint64_t{0} - inv;
}

inline uint64_t ckks_montgomery_neg_inverse(uint64_t q) {
    return ckks_montgomery_neg_inverse64(q);
}

inline uint64_t ckks_montgomery_radix_mod_q(uint64_t q) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(1) << 64) % q);
}

inline uint64_t ckks_montgomery_mul_normal_by_montgomery(
    uint64_t normal,
    uint64_t montgomery,
    uint64_t q,
    uint64_t q_neg_inv) {
    const __uint128_t t = static_cast<__uint128_t>(normal) * montgomery;
    const uint64_t m = static_cast<uint64_t>(t) * q_neg_inv;
    const __uint128_t u128 = (t + static_cast<__uint128_t>(m) * q) >> 64;
    uint64_t u = static_cast<uint64_t>(u128);
    if (u >= q) {
        u -= q;
    }
    return u;
}

uint64_t powmod(uint64_t a, uint64_t b, uint64_t m);

uint64_t mod_inverse(uint64_t a, uint64_t m);

#endif // MODARITH_HPP
