#ifndef CKKS_AES_CTR_HPP
#define CKKS_AES_CTR_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "aligned_vector.hpp"
#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

struct CKKSAes128CtrOptions {
    bool parallelize_independent_work = true;
};

MyVector<CKKSCiphertext> ckks_aes128_encrypt_expanded_round_key_bits(
    const CKKSContext& context,
    const std::array<uint8_t, 176>& expanded_round_keys,
    size_t first_round_key_level);

MyVector<CKKSCiphertext> ckks_aes_ctr(
    CKKSContext& context,
    const MyVector<CKKSCiphertext>& encrypted_expanded_round_key_bits,
    std::span<const std::array<uint8_t, 16>> counter_blocks,
    std::span<const std::array<uint8_t, 16>> ciphertext_blocks,
    const CKKSAes128CtrOptions& options = {});

#endif  // CKKS_AES_CTR_HPP
