#ifndef CKKS_S2C_GENERALIZED_HPP
#define CKKS_S2C_GENERALIZED_HPP

#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

CKKSCiphertext ckks_s2c_generalized(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    size_t stage_count = 1);

#endif // CKKS_S2C_GENERALIZED_HPP
