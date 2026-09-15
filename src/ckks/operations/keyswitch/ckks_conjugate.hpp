#ifndef CKKS_CONJUGATE_HPP
#define CKKS_CONJUGATE_HPP

#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

CKKSCiphertext ckks_conjugate(
    const CKKSContext& context,
    const CKKSCiphertext& ct);
void ckks_conjugate_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct);

#endif // CKKS_CONJUGATE_HPP
