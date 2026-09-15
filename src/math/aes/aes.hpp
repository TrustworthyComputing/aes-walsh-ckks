#ifndef WALSH_AES_MATH_AES_HPP
#define WALSH_AES_MATH_AES_HPP

#include <array>
#include <cstdint>

std::array<uint8_t, 176> aes128_expand_key(
    const std::array<uint8_t, 16>& key);

std::array<uint8_t, 16> aes128_encrypt_block(
    const std::array<uint8_t, 16>& key,
    const std::array<uint8_t, 16>& input_block);

std::array<uint8_t, 16> aes128_ctr_crypt_block(
    const std::array<uint8_t, 16>& key,
    const std::array<uint8_t, 16>& counter_block,
    const std::array<uint8_t, 16>& input_block);

#endif  // WALSH_AES_MATH_AES_HPP
