#ifndef CKKS_ROTATION_HPP
#define CKKS_ROTATION_HPP

#include <cstdint>

#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

CKKSCiphertext ckks_rotate(const CKKSContext& context, const CKKSCiphertext& ct, int64_t rotation);
void ckks_rotate_inplace(const CKKSContext& context, CKKSCiphertext& ct, int64_t rotation);

#endif // CKKS_ROTATION_HPP
