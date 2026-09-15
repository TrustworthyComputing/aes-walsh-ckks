#include "ntt.hpp"
#include "../math_util.hpp"
#include "modarith.hpp"
#include "polynomial.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>

namespace {

uint64_t reduce_input_mod_factor(uint64_t x, uint64_t q, uint64_t factor) {
    if (factor == 1) {
        return x;
    }
    if (factor == 2) {
        return x >= q ? x - q : x;
    }
    if (factor == 4) {
        return reduce_mod_4q(x, q);
    }
    return x % q;
}

}  // namespace

inline uint64_t find_root_of_unity(size_t N, uint64_t q) {
    uint64_t exponent = (q - 1) / (2 * N);
    uint64_t root = 0;
    if ( (q-1) % (2*N) != 0 ){
        return 0;
    }
    for (uint64_t a = 2; a <= (q - 2); a++){
        uint64_t possible_root = powmod(a, exponent, q);
        if (powmod(possible_root, N, q) == (q - 1)){
            root = possible_root;
            break;
        }
    }
    return root;
}

inline uint64_t bit_reverse(uint64_t input, uint64_t bit_size){
    uint64_t result = 0;
    for (uint64_t i = 0; i < bit_size; i++){
        if (input & (1ULL << i)){
            result |= (1ULL << (bit_size - 1 - i));
        }
    }
    return result;
}

Negacyclic_NTT_Twiddles generate_negacyclic_ntt_twiddles(size_t N, uint64_t q){
    uint64_t root_of_unity = find_root_of_unity(N, q);
    uint64_t w = naive_mul_mod_u64(root_of_unity, root_of_unity, q);
    assert(powmod(w, N, q) == 1);

    MyVector<uint64_t> pre_ntt_twiddles(N);
    pre_ntt_twiddles[0] = 1;
    for (size_t i = 1; i < N; i++){
        pre_ntt_twiddles[i] = naive_mul_mod_u64(pre_ntt_twiddles[i-1], root_of_unity, q);
    }

    MyVector<uint64_t> post_intt_twiddles(N);
    uint64_t inverse_root_of_unity = powmod(root_of_unity, q-2, q);
    post_intt_twiddles[0] = 1;
    for (size_t i = 1; i < N; i++){
        post_intt_twiddles[i] = naive_mul_mod_u64(post_intt_twiddles[i - 1], inverse_root_of_unity, q);
    }

    uint64_t log_N = ilog2_pow2(N);

    MyVector<uint64_t> forward_ntt_twiddles(N, 1);
    MyVector<uint64_t> stage_powers(N >> 1);
    stage_powers[0] = 1;
    for (uint64_t i = 1; i < (N >> 1); ++i){
        stage_powers[i] = naive_mul_mod_u64(stage_powers[i - 1], w, q);
    }
    for (uint64_t m = 1; m < N; m <<= 1){
        uint64_t step = N / (m << 1);
            uint64_t stage_bits = ilog2_pow2(m);
        for (uint64_t i = 0; i < m; ++i){
            uint64_t reordered = bit_reverse(i, stage_bits);
            forward_ntt_twiddles[m + i] = stage_powers[reordered * step];
        }
    }

    // embeding pre ntt twiddles
    for (uint64_t m = 1, k = N >> 1; m < N; m <<= 1, k >>= 1) {
        uint64_t psi_k = pre_ntt_twiddles[k]; // = ψ^k
        for (uint64_t i = 0; i < m; ++i) {
            forward_ntt_twiddles[m + i] =
                naive_mul_mod_u64(forward_ntt_twiddles[m + i], psi_k, q);
        }
    }

    


    MyVector<uint64_t> inverse_ntt_twiddles(N, 1);
    auto w_inv = powmod(w, q-2, q);
    stage_powers[0] = 1;
    for (uint64_t i = 1; i < (N >> 1); ++i){
        stage_powers[i] = naive_mul_mod_u64(stage_powers[i - 1], w_inv, q);
    }
    for (uint64_t m = 1; m < N; m <<= 1){
        uint64_t step = N / (m << 1);
        uint64_t stage_bits = ilog2_pow2(m);
        for (uint64_t i = 0; i < m; ++i){
            uint64_t reordered = bit_reverse(i, stage_bits);
            inverse_ntt_twiddles[m + i] = stage_powers[reordered * step];
        }
    }
    
    // embed post-INTT twiddles (ψ^{-i}) into inverse NTT twiddles
    for (uint64_t m = N >> 1, k = 1; m > 0; m >>= 1, k <<= 1) {
        uint64_t psi_k = post_intt_twiddles[k]; // = ψ^{-k}
        for (uint64_t i = 0; i < m; ++i) {
            inverse_ntt_twiddles[m + i] =
                naive_mul_mod_u64(inverse_ntt_twiddles[m + i], psi_k, q);
        }
    }

    
    MyVector<uint64_t> forward_shoup(N);
    MyVector<uint64_t> inverse_shoup(N+1);

    for (size_t i = 0; i < N; ++i) {
        forward_shoup[i] = compute_shoup(forward_ntt_twiddles[i], q);
        inverse_shoup[i] = compute_shoup(inverse_ntt_twiddles[i], q);
    }
    inverse_shoup[N] = compute_shoup(powmod(N, q-2, q), q); // N_inv shoup

    auto N_inv = powmod(N, q-2, q);
    auto barrett_const = barrett_const_computation(q);

    return Negacyclic_NTT_Twiddles{
    std::move(forward_ntt_twiddles),
    std::move(inverse_ntt_twiddles),
    std::move(forward_shoup),
    std::move(inverse_shoup),
    log_N,
    N_inv,
    barrett_const
    };

}

// void pre_ntt(MyVector<uint64_t>& a, uint64_t q, MyVector<uint64_t>& pre_ntt_twiddles, std::pair<uint64_t, uint64_t>& barrett_const){
//     auto input_length = a.size();
//     for (int i = 0; i < input_length; i++){
//         a[i] = mul_mod_u64(a[i], pre_ntt_twiddles[i], q, barrett_const);
//         // a[i] = naive_mul_mod_u64(a[i], pre_ntt_twiddles[i], q);
//     }
// }

template <size_t K>
inline void ntt_forward_dit2_fixed_k(uint64_t* __restrict a,
                                     uint64_t N,
                                     uint64_t q,
                                     uint64_t two_q,
                                     const uint64_t* __restrict table,
                                     const uint64_t* __restrict shoup) {
    const size_t m = N / (K << 1);
    const uint64_t* __restrict psi_stage = table + m;
    const uint64_t* __restrict psi_shoup_stg = shoup + m;

    for (size_t i = 0; i < m; ++i) {
        const uint64_t w = psi_stage[i];
        const uint64_t w_shrp = psi_shoup_stg[i];

        uint64_t* __restrict x0 = a + (i << 1) * K;
        uint64_t* __restrict x1 = x0 + K;

        #pragma unroll
        for (size_t j = 0; j < K; ++j) {
            uint64_t t = x0[j];
            if (t >= two_q) t -= two_q;

            const uint64_t v = x1[j];
            const __uint128_t tmp = ((__uint128_t)v * w_shrp);
            const uint64_t u_hi = (uint64_t)(tmp >> kShoupBits);
            const uint64_t u = v * w - u_hi * q;

            x0[j] = t + u;
            x1[j] = t + two_q - u;
        }
    }
}

template <size_t K>
inline void ntt_inverse_dif2_fixed_k(uint64_t* __restrict a,
                                     uint64_t N,
                                     uint64_t q,
                                     const uint64_t* __restrict table,
                                     const uint64_t* __restrict shoup) {
    const size_t m = N / (K << 1);
    const uint64_t two_q = q << 1;

    for (size_t i = 0; i < m; ++i) {
        uint64_t* __restrict x0 = a + (i << 1) * K;
        uint64_t* __restrict x1 = x0 + K;
        const uint64_t psi = table[m + i];
        const uint64_t psi_shoup = shoup[m + i];

        #pragma unroll
        for (size_t j = 0; j < K; ++j) {
            const uint64_t t = x0[j];
            const uint64_t u = x1[j];
            uint64_t sum = t + u;
            if (sum >= two_q) sum -= two_q;
            x0[j] = sum;

            const uint64_t diff = t - u + two_q;
            const __uint128_t tmp = ((__uint128_t)diff * psi_shoup);
            const uint64_t u_hi = (uint64_t)(tmp >> kShoupBits);
            x1[j] = diff * psi - u_hi * q;
        }
    }
}

// static inline uint64_t mul_mod_shoup(uint64_t a,
//                                      uint64_t w,
//                                      uint64_t w_shoup,
//                                      uint64_t q) {
    
//     __uint128_t t = ( (__uint128_t)a * w_shoup );
//     uint64_t u = (uint64_t)(t >> 64);
//     uint64_t res = a * w - u * q; 
//     // if (res >= q) res -= q;
//     return res;
// }

void ntt_forward_dit2_lazy(uint64_t * __restrict a,
                           uint64_t N,
                           uint64_t q,
                           const uint64_t * __restrict table,
                           const uint64_t * __restrict shoup)
{
    const uint64_t two_q = q << 1;

    for (size_t m = 1, k = N >> 1; m < N; m <<= 1, k >>= 1) {
        switch (k) {
            case 8:
                ntt_forward_dit2_fixed_k<8>(a, N, q, two_q, table, shoup);
                continue;
            case 4:
                ntt_forward_dit2_fixed_k<4>(a, N, q, two_q, table, shoup);
                continue;
            case 2:
                ntt_forward_dit2_fixed_k<2>(a, N, q, two_q, table, shoup);
                continue;
            case 1:
                ntt_forward_dit2_fixed_k<1>(a, N, q, two_q, table, shoup);
                continue;
            default:
                break;
        }

        const uint64_t * __restrict psi_stage      = table + m;
        const uint64_t * __restrict psi_shoup_stg  = shoup + m;

        for (size_t i = 0; i < m; ++i) {
            uint64_t w      = psi_stage[i];
            uint64_t w_shrp = psi_shoup_stg[i];

            uint64_t * __restrict x0 = a + (i << 1) * k;
            uint64_t * __restrict x1 = x0 + k;

            for (size_t j = 0; j < k; ++j) {
                uint64_t t = x0[j];
                if (t >= two_q) t -= two_q;

                uint64_t v = x1[j];
                __uint128_t tmp = ( (__uint128_t)v * w_shrp );
                uint64_t u_hi   = (uint64_t)(tmp >> kShoupBits);
                uint64_t u      = v * w - u_hi * q;   // Harvey/Shoup

                x0[j] = t + u;
                x1[j] = t + two_q - u;
            }
        }
    }
}

void ntt_forward_dit2(uint64_t * __restrict a,
                      uint64_t N,
                      uint64_t q,
                      const uint64_t * __restrict table,
                      const uint64_t * __restrict shoup)
{
    ntt_forward_dit2_lazy(a, N, q, table, shoup);
    for (size_t i = 0; i < N; ++i) {
        a[i] = reduce_mod_4q(a[i], q);
    }
}



// void ntt_forward_dit2(uint64_t* __restrict a,
//                       uint64_t N,
//                       uint64_t q,
//                       uint64_t* __restrict table,
//                       uint64_t* __restrict shoup)
// {
//     for (uint64_t m = 1, k = N >> 1; m < N; m <<= 1, k >>= 1) {
//         uint64_t* __restrict psi_stage      = table + m;
//         uint64_t* __restrict psi_shoup_stg  = shoup + m;

//         for (uint64_t i = 0, jfirst = 0; i < m; ++i, jfirst += (k << 1)) {
//             uint64_t psi      = psi_stage[i];
//             uint64_t psi_shoup  = psi_shoup_stg[i];

//             for (uint64_t j = jfirst; j < jfirst + k; ++j) {
//                 auto t = a[j];
//                 if (t >= 2*q) t -= 2*q;
//                 auto a1 = a[j + k];
//                 __uint128_t temp = ( (__uint128_t)a1 * psi_shoup );
//                 uint64_t temp2 = (uint64_t)(temp >> 64);
//                 uint64_t u = a1 * psi - temp2 * q; 

//                 a[j] = t + u;
//                 a[j + k] = t - u + 2*q;
//             }
//         }
//     }
// }


inline void ntt_inverse_dif2_butterflies(uint64_t*__restrict a, uint64_t N, uint64_t q, const uint64_t*__restrict table, const uint64_t*__restrict shoup) {
    const uint64_t two_q = q << 1;

    for (size_t m = static_cast<size_t>(N) >> 1, k = 1; m > 0; m >>= 1, k <<= 1){
        switch (k) {
            case 1:
                ntt_inverse_dif2_fixed_k<1>(a, N, q, table, shoup);
                continue;
            case 2:
                ntt_inverse_dif2_fixed_k<2>(a, N, q, table, shoup);
                continue;
            case 4:
                ntt_inverse_dif2_fixed_k<4>(a, N, q, table, shoup);
                continue;
            case 8:
                ntt_inverse_dif2_fixed_k<8>(a, N, q, table, shoup);
                continue;
            default:
                break;
        }

        for (size_t i = 0; i < m; ++i){
            uint64_t* __restrict x0 = a + (i << 1) * k;
            uint64_t* __restrict x1 = x0 + k;
            const uint64_t psi = table[m + i];
            const uint64_t psi_shoup = shoup[m + i];
            for (size_t j = 0; j < k; ++j){
                const uint64_t t = x0[j];
                const uint64_t u = x1[j];
                uint64_t sum = t + u;
                if (sum >= two_q) sum -= two_q;
                x0[j] = sum;

                const uint64_t diff = t - u + two_q;
                const __uint128_t tmp = ((__uint128_t)diff * psi_shoup);
                const uint64_t hi = (uint64_t)(tmp >> kShoupBits);
                x1[j] = diff * psi - hi * q;
            }
        }
    }
}

void ntt_inverse_dif2_lazy(uint64_t*__restrict a, uint64_t N, uint64_t N_inv, uint64_t q, const uint64_t*__restrict table, const uint64_t*__restrict shoup) {
    ntt_inverse_dif2_butterflies(a, N, q, table, shoup);
    for (size_t i = 0; i < static_cast<size_t>(N); i++){
        a[i] = mul_mod_shoup(a[i], N_inv, shoup[N], q);
    }
    // need to canonicalize to [0, q)
}

void ntt_inverse_dif2(uint64_t*__restrict a, uint64_t N, uint64_t N_inv, uint64_t q, const uint64_t*__restrict table, const uint64_t*__restrict shoup) {
    ntt_inverse_dif2_butterflies(a, N, q, table, shoup);
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
        a[i] = reduce_mod_4q(mul_mod_shoup(a[i], N_inv, shoup[N], q), q);
    }
}

void ntt_forward_dit2_dispatch(uint64_t*__restrict a,
                               uint64_t N,
                               uint64_t q,
                               const Negacyclic_NTT_Twiddles& table,
                               bool lazy) {
    if (lazy) {
        ntt_forward_dit2_lazy(a, N, q, table.forward_ntt.data(),
                              table.forward_shoup.data());
    } else {
        ntt_forward_dit2(a, N, q, table.forward_ntt.data(),
                         table.forward_shoup.data());
    }
}

void ntt_inverse_dif2_dispatch(uint64_t*__restrict a,
                               uint64_t N,
                               uint64_t q,
                               const Negacyclic_NTT_Twiddles& table,
                               bool lazy) {
    if (lazy) {
        ntt_inverse_dif2_lazy(a, N, table.N_inv, q, table.inverse_ntt.data(),
                              table.inverse_shoup.data());
    } else {
        ntt_inverse_dif2(a, N, table.N_inv, q, table.inverse_ntt.data(),
                         table.inverse_shoup.data());
    }
}

void ntt_forward_rns_flat_inplace(uint64_t* __restrict rns,
                                        size_t N,
                                        size_t levels,
                                        const MyVector<uint64_t>& moduli,
                                        const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt,
                                        bool lazy) {
    assert(levels <= moduli.size());
    assert(levels <= twiddle_ntt.size());

    for (size_t i = 0; i < levels; ++i) {
        uint64_t* __restrict level_ptr = rns + i * N;
        const uint64_t q = moduli[i];
        const auto& tbl = twiddle_ntt[i];
        ntt_forward_dit2_dispatch(level_ptr, N, q, tbl, lazy);
    }
}

void ntt_inverse_rns_flat_inplace(uint64_t* __restrict rns,
                                        size_t N,
                                        size_t levels,
                                        const MyVector<uint64_t>& moduli,
                                        const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt, bool lazy) {
    assert(levels <= moduli.size());
    assert(levels <= twiddle_ntt.size());

    for (size_t i = 0; i < levels; ++i) {
        uint64_t* __restrict level_ptr = rns + i * N;
        const uint64_t q = moduli[i];
        const auto& tbl = twiddle_ntt[i];
        ntt_inverse_dif2_dispatch(level_ptr, N, q, tbl, lazy);
    }
}


void pointwise_multiply(const uint64_t* __restrict a, const uint64_t* __restrict b, uint64_t* __restrict c, uint64_t N, uint64_t q, const BarrettConst* barrett_const){
    for (size_t i = 0; i < N; ++i){
        c[i] = mul_mod_u64(a[i], b[i], q, barrett_const);
    }
}

void pointwise_multiply_inplace(uint64_t* __restrict a, const uint64_t* __restrict b, uint64_t N, uint64_t q, const BarrettConst* barrett_const){
    for (size_t i = 0; i < N; ++i){
        a[i] = mul_mod_u64(a[i], b[i], q, barrett_const);
    }
}

// C += A*B
void pointwise_multiply_accumulate(const uint64_t* __restrict a, const uint64_t* __restrict b, uint64_t* __restrict c, uint64_t N, uint64_t q, const BarrettConst* barrett_const){
    for (size_t i = 0; i < N; ++i){
        c[i] = add_mod_q(c[i], mul_mod_u64(a[i], b[i], q, barrett_const),q);
    }
}


void pointwise_multiply_scalar_inplace(uint64_t* __restrict a, uint64_t scalar, size_t N, uint64_t q, const BarrettConst* barrett_const){
    for (size_t i = 0; i < N; ++i){
        a[i] = mul_mod_u64(a[i], scalar, q, barrett_const);
    }
}

void pointwise_multiply_accumulate_dispatch(const uint64_t* __restrict a,
                                            const uint64_t* __restrict b,
                                            uint64_t* __restrict c,
                                            uint64_t N,
                                            uint64_t q,
                                            const Negacyclic_NTT_Twiddles& table,
                                            uint64_t input_mod_factor) {
    if (input_mod_factor == 1) {
        pointwise_multiply_accumulate(a, b, c, N, q, &table.barrett_const);
        return;
    }
    for (size_t i = 0; i < N; ++i) {
        const uint64_t a_i = reduce_input_mod_factor(a[i], q, input_mod_factor);
        const uint64_t b_i = reduce_input_mod_factor(b[i], q, input_mod_factor);
        c[i] = add_mod_q(c[i], mul_mod_u64(a_i, b_i, q, &table.barrett_const), q);
    }
}

void pointwise_multiply_dispatch(const uint64_t* __restrict a,
                                 const uint64_t* __restrict b,
                                 uint64_t* __restrict c,
                                 uint64_t N,
                                 uint64_t q,
                                 const Negacyclic_NTT_Twiddles& table,
                                 uint64_t input_mod_factor) {
    if (input_mod_factor == 1) {
        pointwise_multiply(a, b, c, N, q, &table.barrett_const);
        return;
    }
    for (size_t i = 0; i < N; ++i) {
        const uint64_t a_i = reduce_input_mod_factor(a[i], q, input_mod_factor);
        const uint64_t b_i = reduce_input_mod_factor(b[i], q, input_mod_factor);
        c[i] = mul_mod_u64(a_i, b_i, q, &table.barrett_const);
    }
}

void pointwise_multiply_inplace_dispatch(uint64_t* __restrict a,
                                         const uint64_t* __restrict b,
                                         uint64_t N,
                                         uint64_t q,
                                         const Negacyclic_NTT_Twiddles& table,
                                         uint64_t input_mod_factor) {
    if (input_mod_factor == 1) {
        pointwise_multiply_inplace(a, b, N, q, &table.barrett_const);
        return;
    }
    for (size_t i = 0; i < N; ++i) {
        const uint64_t a_i = reduce_input_mod_factor(a[i], q, input_mod_factor);
        const uint64_t b_i = reduce_input_mod_factor(b[i], q, input_mod_factor);
        a[i] = mul_mod_u64(a_i, b_i, q, &table.barrett_const);
    }
}

void pointwise_multiply_scalar_inplace_dispatch(
    uint64_t* __restrict a,
    uint64_t scalar,
    size_t N,
    uint64_t q,
    const Negacyclic_NTT_Twiddles& table) {
    pointwise_multiply_scalar_inplace(a, scalar, N, q, &table.barrett_const);
}

void pointwise_multiply_scalar_inplace_dispatch(
    uint64_t* __restrict a,
    uint64_t scalar,
    size_t N,
    uint64_t q,
    const Negacyclic_NTT_Twiddles& table,
    bool use_montgomery) {
    if (!use_montgomery) {
        pointwise_multiply_scalar_inplace_dispatch(a, scalar, N, q, table);
        return;
    }

    const uint64_t scalar_montgomery =
        static_cast<uint64_t>(
            (static_cast<__uint128_t>(scalar) * ckks_montgomery_radix_mod_q(q)) % q);
    const uint64_t q_neg_inv = ckks_montgomery_neg_inverse(q);

    thread_local MyVector<uint64_t> scalar_montgomery_poly;
    scalar_montgomery_poly.resize(N);
    std::fill(
        scalar_montgomery_poly.begin(),
        scalar_montgomery_poly.end(),
        scalar_montgomery);
    pointwise_multiply_montgomery_inplace_dispatch(
        a,
        scalar_montgomery_poly.data(),
        N,
        q,
        q_neg_inv);
}

void pointwise_multiply_montgomery_inplace_dispatch(
    uint64_t* __restrict normal,
    const uint64_t* __restrict montgomery,
    size_t N,
    uint64_t q,
    uint64_t q_neg_inv) {
    for (size_t i = 0; i < N; ++i) {
        normal[i] = ckks_montgomery_mul_normal_by_montgomery(
            normal[i], montgomery[i], q, q_neg_inv);
    }
}
