#ifndef CKKS_MUL_HPP
#define CKKS_MUL_HPP


#include <cstdint>

#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"
#include "ckks_rescale.hpp"

CKKSCiphertext ckks_mul(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    Complex value,
    double plaintext_scale);
void ckks_mul_inplace(const CKKSContext& context, CKKSCiphertext& ct1, const CKKSEncoding& ct2);
void ckks_mul_inplace(const CKKSContext& context, CKKSCiphertext& ct1, const CKKSCiphertext& ct2);
void ckks_mul_inplace(const CKKSContext& context, CKKSCiphertext& ct, int64_t scalar);
void ckks_mul_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    Complex value,
    double plaintext_scale);
void ckks_square_inplace(const CKKSContext& context, CKKSCiphertext& ct);


#endif // CKKS_MUL_HPP
