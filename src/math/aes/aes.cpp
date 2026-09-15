#include "aes/aes.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "aes/aes_tables.hpp"

namespace {

constexpr size_t kAesRoundCount = 10;
constexpr size_t kAesExpandedKeyBytes = 176;

constexpr std::array<uint8_t, 10> kAesRcon{
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};

uint8_t aes_xtime(uint8_t value) {
    const uint8_t shifted = static_cast<uint8_t>(value << 1U);
    return static_cast<uint8_t>((value & 0x80U) != 0U ? shifted ^ 0x1bU : shifted);
}

void aes_add_round_key(
    std::array<uint8_t, 16>& state,
    const std::array<uint8_t, 176>& expanded_key,
    size_t round) {
    const size_t offset = round * 16;
    for (size_t i = 0; i < 16; ++i) {
        state[i] ^= expanded_key[offset + i];
    }
}

void aes_sub_bytes(std::array<uint8_t, 16>& state) {
    for (auto& byte : state) {
        byte = static_cast<uint8_t>(kAesSboxTable[byte]);
    }
}

void aes_shift_rows(std::array<uint8_t, 16>& state) {
    auto old_state = state;
    for (size_t column = 0; column < 4; ++column) {
        for (size_t row = 0; row < 4; ++row) {
            state[4 * column + row] = old_state[4 * ((column + row) % 4) + row];
        }
    }
}

void aes_mix_columns(std::array<uint8_t, 16>& state) {
    for (size_t column = 0; column < 4; ++column) {
        uint8_t* col = state.data() + 4 * column;
        const uint8_t b0 = col[0];
        const uint8_t b1 = col[1];
        const uint8_t b2 = col[2];
        const uint8_t b3 = col[3];
        const uint8_t t = static_cast<uint8_t>(b0 ^ b1 ^ b2 ^ b3);
        const uint8_t u = b0;
        col[0] ^= t ^ aes_xtime(static_cast<uint8_t>(b0 ^ b1));
        col[1] ^= t ^ aes_xtime(static_cast<uint8_t>(b1 ^ b2));
        col[2] ^= t ^ aes_xtime(static_cast<uint8_t>(b2 ^ b3));
        col[3] ^= t ^ aes_xtime(static_cast<uint8_t>(b3 ^ u));
    }
}

}  // namespace

std::array<uint8_t, 176> aes128_expand_key(
    const std::array<uint8_t, 16>& key) {
    std::array<uint8_t, kAesExpandedKeyBytes> expanded{};
    std::copy(key.begin(), key.end(), expanded.begin());

    size_t bytes_generated = 16;
    size_t rcon_index = 0;
    std::array<uint8_t, 4> temp{};
    while (bytes_generated < expanded.size()) {
        for (size_t i = 0; i < 4; ++i) {
            temp[i] = expanded[bytes_generated - 4 + i];
        }

        if ((bytes_generated % 16) == 0) {
            const uint8_t first = temp[0];
            temp[0] = static_cast<uint8_t>(kAesSboxTable[temp[1]]) ^ kAesRcon[rcon_index++];
            temp[1] = static_cast<uint8_t>(kAesSboxTable[temp[2]]);
            temp[2] = static_cast<uint8_t>(kAesSboxTable[temp[3]]);
            temp[3] = static_cast<uint8_t>(kAesSboxTable[first]);
        }

        for (size_t i = 0; i < 4; ++i) {
            expanded[bytes_generated] =
                static_cast<uint8_t>(expanded[bytes_generated - 16] ^ temp[i]);
            ++bytes_generated;
        }
    }
    return expanded;
}

std::array<uint8_t, 16> aes128_encrypt_block(
    const std::array<uint8_t, 16>& key,
    const std::array<uint8_t, 16>& input_block) {
    const auto expanded_key = aes128_expand_key(key);
    auto state = input_block;

    aes_add_round_key(state, expanded_key, 0);
    for (size_t round = 1; round < kAesRoundCount; ++round) {
        aes_sub_bytes(state);
        aes_shift_rows(state);
        aes_mix_columns(state);
        aes_add_round_key(state, expanded_key, round);
    }
    aes_sub_bytes(state);
    aes_shift_rows(state);
    aes_add_round_key(state, expanded_key, kAesRoundCount);
    return state;
}

std::array<uint8_t, 16> aes128_ctr_crypt_block(
    const std::array<uint8_t, 16>& key,
    const std::array<uint8_t, 16>& counter_block,
    const std::array<uint8_t, 16>& input_block) {
    const auto keystream = aes128_encrypt_block(key, counter_block);
    std::array<uint8_t, 16> out{};
    for (size_t i = 0; i < 16; ++i) {
        out[i] = static_cast<uint8_t>(input_block[i] ^ keystream[i]);
    }
    return out;
}
