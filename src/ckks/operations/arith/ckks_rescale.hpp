#ifndef CKKS_RESCALE_HPP
#define CKKS_RESCALE_HPP

#include <cstddef>

#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

void ckks_rescale(const CKKSContext& context, CKKSCiphertext& ct);
void ckks_drop_to_level_inplace(CKKSCiphertext& ct, size_t target_level);

#endif  // CKKS_RESCALE_HPP
