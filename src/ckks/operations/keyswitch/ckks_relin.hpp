#ifndef CKKS_RELIN_HPP
#define CKKS_RELIN_HPP

#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

void ckks_relin_hybrid_inplace(const CKKSContext& context, CKKSCiphertext& ct);

#endif //CKKS_RELIN_HPP
