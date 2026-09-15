#ifndef CKKS_SECRET_KEY_GENERATION_HPP
#define CKKS_SECRET_KEY_GENERATION_HPP

#include "ckks_context.hpp"

MyVector<int8_t> generate_uniform_terenary_secret(std::size_t N);
MyVector<int8_t> generate_hamming_weight_ternary_secret(
    std::size_t N,
    std::size_t hamming_weight);
MyVector<int8_t> generate_sparse_ternary_secret(
    std::size_t N,
    std::size_t hamming_weight,
    std::size_t window_width);
MyVector<MyVector<uint64_t>> ntt_secret_key(
    const MyVector<int8_t>& s,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt,
    size_t N,
    const MyVector<uint64_t>& moduli);
MyVector<CKKSCiphertext> generate_secret_key_switch_key_hybrid(
    const CKKSContext& context,
    const MyVector<int8_t>& source_secret_coeffs,
    const MyVector<int8_t>& target_secret_coeffs);
MyVector<CKKSCiphertext> generate_secret_key_switch_key_q0p0_hybrid(
    const CKKSContext& context,
    const MyVector<int8_t>& source_secret_coeffs,
    const MyVector<int8_t>& target_secret_coeffs);

#endif // CKKS_SECRET_KEY_GENERATION_HPP
