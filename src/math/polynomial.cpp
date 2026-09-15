#include "polynomial.hpp"
#include "modarith.hpp"
#include <cstddef>
#include <cstdint>

void poly_add_modq(const uint64_t* __restrict a,
                   const uint64_t* __restrict b,
                   uint64_t* __restrict c,
                   size_t N,
                   uint64_t q){
    for (size_t i = 0; i < N; i++){
        c[i] = add_mod_q(a[i], b[i], q);
    }
}

void poly_add_modq_inplace(uint64_t* __restrict a,
                           const uint64_t* __restrict b,
                           size_t N,
                           uint64_t q){
    for (size_t i = 0; i < N; i++){
        a[i] = add_mod_q(a[i], b[i], q);
    }
}

void poly_sub_modq_inplace(uint64_t* __restrict a,
                           const uint64_t* __restrict b,
                           size_t N,
                           uint64_t q){
    for (size_t i = 0; i < N; i++){
        a[i] = sub_mod_q(a[i], b[i], q);
    }
}
