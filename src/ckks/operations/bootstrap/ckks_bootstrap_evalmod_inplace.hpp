#ifndef CKKS_BOOTSTRAP_EVALMOD_INPLACE_HPP
#define CKKS_BOOTSTRAP_EVALMOD_INPLACE_HPP

#include <utility>

#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

void ckks_batch_bootstrap_evalmod_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct0,
    CKKSCiphertext& ct1);

std::pair<CKKSCiphertext, CKKSCiphertext>
ckks_batch_bootstrap_evalmod_from_complex_packed(
    const CKKSContext& context,
    CKKSCiphertext packed);

#endif // CKKS_BOOTSTRAP_EVALMOD_INPLACE_HPP
