#ifndef NTT_HPP
#define NTT_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <utility>

#include "aligned_vector.hpp"

#include "modarith/modarith.hpp"

struct Negacyclic_NTT_Twiddles{
    MyVector<uint64_t> forward_ntt;
    MyVector<uint64_t> inverse_ntt;
    MyVector<uint64_t> forward_shoup;
    MyVector<uint64_t> inverse_shoup;
    uint64_t log_N;
    uint64_t N_inv;
    BarrettConst barrett_const;
};

uint64_t find_root_of_unity(size_t N, uint64_t q);
Negacyclic_NTT_Twiddles generate_negacyclic_ntt_twiddles(size_t N, uint64_t q);
void ntt_forward_dit2(uint64_t*__restrict a, uint64_t N, uint64_t q, const uint64_t*__restrict table, const uint64_t*__restrict shoup);
void ntt_inverse_dif2(uint64_t*__restrict a, uint64_t N, uint64_t N_inv, uint64_t q, const uint64_t*__restrict table, const uint64_t*__restrict shoup);
void ntt_forward_dit2_lazy(uint64_t*__restrict a, uint64_t N, uint64_t q, const uint64_t*__restrict table, const uint64_t*__restrict shoup);
void ntt_inverse_dif2_lazy(uint64_t*__restrict a, uint64_t N, uint64_t N_inv, uint64_t q, const uint64_t*__restrict table, const uint64_t*__restrict shoup);
void ntt_forward_dit2_dispatch(uint64_t*__restrict a,
                               uint64_t N,
                               uint64_t q,
                               const Negacyclic_NTT_Twiddles& table,
                               bool lazy = false);
void ntt_inverse_dif2_dispatch(uint64_t*__restrict a,
                               uint64_t N,
                               uint64_t q,
                               const Negacyclic_NTT_Twiddles& table,
                               bool lazy = false);

void pointwise_multiply(const uint64_t* __restrict a, const uint64_t* __restrict b, uint64_t* __restrict c, uint64_t N, uint64_t q, const BarrettConst* barrett_const);
void pointwise_multiply_inplace(uint64_t* __restrict a, const uint64_t* __restrict b, uint64_t N, uint64_t q, const BarrettConst* barrett_const);

void pointwise_multiply_accumulate(const uint64_t* __restrict a, const uint64_t* __restrict b, uint64_t* __restrict c, uint64_t N, uint64_t q, const BarrettConst* barrett_const);

void pointwise_multiply_scalar_inplace(uint64_t* __restrict a, uint64_t scalar, size_t N, uint64_t q, const BarrettConst* barrett_const);
void pointwise_multiply_accumulate_dispatch(const uint64_t* __restrict a,
                                            const uint64_t* __restrict b,
                                            uint64_t* __restrict c,
                                            uint64_t N,
                                            uint64_t q,
                                            const Negacyclic_NTT_Twiddles& table,
                                            uint64_t input_mod_factor = 1);
void pointwise_multiply_dispatch(const uint64_t* __restrict a,
                                 const uint64_t* __restrict b,
                                 uint64_t* __restrict c,
                                 uint64_t N,
                                 uint64_t q,
                                 const Negacyclic_NTT_Twiddles& table,
                                 uint64_t input_mod_factor = 1);
void pointwise_multiply_inplace_dispatch(uint64_t* __restrict a,
                                         const uint64_t* __restrict b,
                                         uint64_t N,
                                         uint64_t q,
                                         const Negacyclic_NTT_Twiddles& table,
                                         uint64_t input_mod_factor = 1);
void pointwise_multiply_scalar_inplace_dispatch(uint64_t* __restrict a,
                                                uint64_t scalar,
                                                size_t N,
                                                uint64_t q,
                                                const Negacyclic_NTT_Twiddles& table);
void pointwise_multiply_scalar_inplace_dispatch(uint64_t* __restrict a,
                                                uint64_t scalar,
                                                size_t N,
                                                uint64_t q,
                                                const Negacyclic_NTT_Twiddles& table,
                                                bool use_montgomery);
void pointwise_multiply_montgomery_inplace_dispatch(
    uint64_t* __restrict normal,
    const uint64_t* __restrict montgomery,
    size_t N,
    uint64_t q,
    uint64_t q_neg_inv);

void ntt_forward_rns_flat_inplace(uint64_t* __restrict rns, size_t N, size_t levels, const MyVector<uint64_t>& moduli, const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt, bool lazy = false);
void ntt_inverse_rns_flat_inplace(uint64_t* __restrict rns, size_t N, size_t levels, const MyVector<uint64_t>& moduli, const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt, bool lazy = false);

#endif // NTT_HPP
