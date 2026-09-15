#ifndef CKKS_AES_SBOX_HPP
#define CKKS_AES_SBOX_HPP

#include "aes/aes_tables.hpp"
#include "aligned_vector.hpp"

inline const MyVector<uint64_t> kAesSbox(
    kAesSboxTable.begin(),
    kAesSboxTable.end());

#endif  // CKKS_AES_SBOX_HPP
