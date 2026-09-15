#ifndef CKKS_AES_CTR_WALSH_SBOX_HPP
#define CKKS_AES_CTR_WALSH_SBOX_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "aligned_vector.hpp"
#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

struct CKKSAesCtrWalshSboxOptions {
    bool parallelize_sbox_pairs = true;
};

// Round keys for the complex-block Walsh AES path. Each active CKKS complex
// slot carries two AES blocks: real lane block s and imaginary lane block
// active_complex_block_count + s.
MyVector<CKKSCiphertext> ckks_aes128_encrypt_expanded_round_key_bits_walsh_complex_blocks(
    const CKKSContext& context,
    const std::array<uint8_t, 176>& expanded_round_keys,
    size_t active_complex_block_count);

// AES-CTR path that treats the real and imaginary components of each active
// CKKS complex slot as independent AES blocks. The block spans must have size
// 2 * active_complex_block_count, with the second half mapped to the imaginary
// lane. The returned bit ciphertexts have already gone through the final
// binary refresh.
MyVector<CKKSCiphertext> ckks_aes_ctr_walsh_sbox_complex_blocks(
    CKKSContext& context,
    const MyVector<CKKSCiphertext>& encrypted_expanded_round_key_bits,
    std::span<const std::array<uint8_t, 16>> counter_blocks,
    std::span<const std::array<uint8_t, 16>> ciphertext_blocks,
    const CKKSAesCtrWalshSboxOptions& options = {});

#endif  // CKKS_AES_CTR_WALSH_SBOX_HPP
