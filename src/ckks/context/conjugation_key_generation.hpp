#ifndef CKKS_CONJUGATION_KEY_GENERATION_HPP
#define CKKS_CONJUGATION_KEY_GENERATION_HPP

#include "ckks_context.hpp"

uint64_t conjugation_automorphism_index(size_t N);
MyVector<CKKSCiphertext> generate_conjugation_key_hybrid(
    const CKKSContext& context,
    const MyVector<int8_t>& secret_coeffs);

#endif // CKKS_CONJUGATION_KEY_GENERATION_HPP
