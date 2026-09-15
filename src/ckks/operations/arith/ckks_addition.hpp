#ifndef CKKS_ADDITION
#define CKKS_ADDITION

#include <cstdint>
#include <stdexcept>
#include "ckks_context.hpp"

void ckks_add_inplace(const CKKSContext& context, CKKSCiphertext& ct1, const CKKSEncoding& ct2);
void ckks_add_inplace(const CKKSContext& context, CKKSCiphertext& ct1, const CKKSCiphertext& ct2);
void ckks_add_integer_constant_inplace(const CKKSContext& context, CKKSCiphertext& ct, int64_t constant);


void ckks_sub_inplace(const CKKSContext& context, CKKSCiphertext& ct1, const CKKSCiphertext& ct2);



#endif  // CKKS_ADDITION
