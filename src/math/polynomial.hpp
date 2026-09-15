#ifndef POLYNOMIAL_HPP
#define POLYNOMIAL_HPP

#include <cstdint>
#include <cstddef>
#include "modarith.hpp"

void poly_add_modq_inplace(uint64_t* __restrict a,
                           const uint64_t* __restrict b,
                           size_t N,
                           uint64_t q);
void poly_sub_modq_inplace(uint64_t* __restrict a,
                           const uint64_t* __restrict b,
                           size_t N,
                           uint64_t q);

void poly_add_modq(const uint64_t* __restrict a,
                   const uint64_t* __restrict b,
                   uint64_t* __restrict  c,
                   size_t N,
                   uint64_t q);


#endif // POLYNOMIAL_HPP
