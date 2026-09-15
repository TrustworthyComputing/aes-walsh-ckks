#ifndef CKKS_MODRAISE_HPP
#define CKKS_MODRAISE_HPP

#include <cstddef>

#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

void ckks_modraise_inplace(const CKKSContext& context, CKKSCiphertext& ct, size_t target_level);

#endif // CKKS_MODRAISE_HPP
