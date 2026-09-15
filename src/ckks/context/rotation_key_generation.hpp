#ifndef CKKS_ROTATION_KEY_GENERATION_HPP
#define CKKS_ROTATION_KEY_GENERATION_HPP

#include "ckks_context.hpp"

int64_t normalize_rotation_value(size_t N, int64_t rotation);
uint64_t rotation_automorphism_index(size_t N, int64_t rotation);
MyVector<size_t> build_ntt_automorphism_map(size_t N, size_t logN, uint64_t automorphism_index);
MyVector<CKKSCiphertext> generate_rotation_key_hybrid(
    const CKKSContext& context,
    const MyVector<int8_t>& secret_coeffs,
    int64_t rotation);

#endif // CKKS_ROTATION_KEY_GENERATION_HPP
