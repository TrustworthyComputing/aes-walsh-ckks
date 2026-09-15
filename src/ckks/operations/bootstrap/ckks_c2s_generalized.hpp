#ifndef CKKS_C2S_GENERALIZED_HPP
#define CKKS_C2S_GENERALIZED_HPP

#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

CKKSCiphertext ckks_c2s_generalized(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    double transform_scale);

#endif // CKKS_C2S_GENERALIZED_HPP
