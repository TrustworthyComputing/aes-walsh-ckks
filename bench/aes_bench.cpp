#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <omp.h>

#include "aes/aes.hpp"
#include "aes/ckks_aes_ctr.hpp"
#include "aes/ckks_aes_ctr_walsh_sbox.hpp"
#include "ckks_aes_params.hpp"
#include "ckks_context.hpp"
#include "ckks_encoding.hpp"
#include "ckks_params.hpp"

namespace {

constexpr size_t kAesBlockBits = 128;
constexpr std::array<uint8_t, 16> kTestKey{
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

constexpr std::array<uint8_t, 16> kTestCounter{
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
    0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff};

constexpr std::array<uint8_t, 16> kTestPlaintext{
    0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
    0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a};

struct AesBlocks {
    std::vector<std::array<uint8_t, 16>> counters;
    std::vector<std::array<uint8_t, 16>> plaintexts;
    std::vector<std::array<uint8_t, 16>> ciphertexts;
};

struct Result {
    std::string method;
    std::string parameter;
    size_t blocks = 0;
    double seconds = 0.0;
    double max_error = 0.0;
    double average_error = 0.0;
    size_t mismatches = 0;
    size_t rotations = 0;
};

uint8_t block_bit(const std::array<uint8_t, 16>& block, size_t bit) {
    return static_cast<uint8_t>((block[bit / 8] >> (bit % 8)) & 1U);
}

std::array<uint8_t, 16> counter_for_block(size_t block_index) {
    auto counter = kTestCounter;
    size_t carry = block_index;
    for (size_t index = counter.size(); index-- > 0 && carry != 0;) {
        const size_t sum = static_cast<size_t>(counter[index]) + (carry & 0xffU);
        counter[index] = static_cast<uint8_t>(sum & 0xffU);
        carry = (carry >> 8) + (sum >> 8);
    }
    if (carry != 0) {
        throw std::overflow_error("AES-CTR counter overflow");
    }
    return counter;
}

std::array<uint8_t, 16> plaintext_for_block(size_t block_index) {
    if (block_index == 0) {
        return kTestPlaintext;
    }
    std::array<uint8_t, 16> block{};
    for (size_t byte = 0; byte < block.size(); ++byte) {
        block[byte] = static_cast<uint8_t>(
            (0x5aU + 17U * static_cast<unsigned>(block_index) +
             29U * static_cast<unsigned>(byte)) & 0xffU);
    }
    return block;
}

AesBlocks make_blocks(size_t count) {
    AesBlocks blocks{
        std::vector<std::array<uint8_t, 16>>(count),
        std::vector<std::array<uint8_t, 16>>(count),
        std::vector<std::array<uint8_t, 16>>(count)};
    for (size_t index = 0; index < count; ++index) {
        blocks.counters[index] = counter_for_block(index);
        blocks.plaintexts[index] = plaintext_for_block(index);
        blocks.ciphertexts[index] = aes128_ctr_crypt_block(
            kTestKey, blocks.counters[index], blocks.plaintexts[index]);
    }
    return blocks;
}

std::vector<std::array<uint8_t, 16>> periodic_expected_blocks(
    std::span<const std::array<uint8_t, 16>> blocks,
    size_t physical_slots,
    size_t logical_slots) {
    if (blocks.empty() || blocks.size() > logical_slots ||
        physical_slots % logical_slots != 0) {
        throw std::invalid_argument("invalid sparse XBOOT layout");
    }
    std::vector<std::array<uint8_t, 16>> expected(physical_slots);
    for (size_t slot = 0; slot < physical_slots; ++slot) {
        const size_t logical_slot = slot % logical_slots;
        expected[slot] = logical_slot < blocks.size()
            ? blocks[logical_slot]
            : std::array<uint8_t, 16>{};
    }
    return expected;
}

void validate_binary_outputs(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& bits,
    std::span<const std::array<uint8_t, 16>> expected,
    size_t mismatch_slots,
    double& max_error,
    double& average_error,
    size_t& mismatches) {
    if (bits.size() != kAesBlockBits || expected.size() != context.getParams().getSlots()) {
        throw std::invalid_argument("invalid XBOOT output dimensions");
    }
    std::vector<std::array<uint8_t, 16>> decoded(expected.size());
    long double error_sum = 0.0L;
    size_t error_count = 0;
    for (size_t bit = 0; bit < kAesBlockBits; ++bit) {
        const auto slots = ckks_decrypt(context, bits[bit]);
        for (size_t slot = 0; slot < slots.size(); ++slot) {
            if (slots[slot].real() >= 0.5) {
                decoded[slot][bit / 8] |= static_cast<uint8_t>(1U << (bit % 8));
            }
            const double real_error =
                std::abs(slots[slot].real() - block_bit(expected[slot], bit));
            max_error = std::max(max_error, real_error);
            max_error = std::max(max_error, std::abs(slots[slot].imag()));
            error_sum += real_error;
            ++error_count;
        }
    }
    for (size_t slot = 0; slot < mismatch_slots; ++slot) {
        mismatches += decoded[slot] != expected[slot];
    }
    average_error = static_cast<double>(error_sum / error_count);
}

void validate_walsh_outputs(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& bits,
    std::span<const std::array<uint8_t, 16>> expected,
    size_t complex_blocks,
    double& max_error,
    double& average_error,
    size_t& mismatches) {
    if (bits.size() != kAesBlockBits || expected.size() != 2 * complex_blocks) {
        throw std::invalid_argument("invalid Walsh output dimensions");
    }
    std::vector<std::array<uint8_t, 16>> decoded(expected.size());
    long double error_sum = 0.0L;
    size_t error_count = 0;
    for (size_t bit = 0; bit < kAesBlockBits; ++bit) {
        const auto slots = ckks_decrypt(context, bits[bit]);
        for (size_t slot = 0; slot < complex_blocks; ++slot) {
            const std::array<double, 2> values{
                slots[slot].real(), slots[slot].imag()};
            const std::array<size_t, 2> logical_slots{
                slot, complex_blocks + slot};
            for (size_t lane = 0; lane < 2; ++lane) {
                const size_t logical_slot = logical_slots[lane];
                if (values[lane] >= 0.5) {
                    decoded[logical_slot][bit / 8] |=
                        static_cast<uint8_t>(1U << (bit % 8));
                }
                const double error = std::abs(
                    values[lane] - block_bit(expected[logical_slot], bit));
                max_error = std::max(max_error, error);
                error_sum += error;
                ++error_count;
            }
        }
    }
    for (size_t slot = 0; slot < expected.size(); ++slot) {
        mismatches += decoded[slot] != expected[slot];
    }
    average_error = static_cast<double>(error_sum / error_count);
}

Result run_xboot(
    CKKSParams params,
    std::string method,
    std::string parameter,
    size_t active_blocks) {
    CKKSContext context(std::move(params));
    const size_t physical_slots = context.getParams().getSlots();
    const size_t logical_slots = context.getParams().getBootstrapActiveSlotCount();
    if (active_blocks == 0 || active_blocks > logical_slots) {
        throw std::invalid_argument("invalid active XBOOT block count");
    }

    const auto blocks = make_blocks(active_blocks);
    const auto expected = periodic_expected_blocks(
        blocks.plaintexts, physical_slots, logical_slots);
    const auto expanded_key = aes128_expand_key(kTestKey);
    const auto encrypted_round_keys = ckks_aes128_encrypt_expanded_round_key_bits(
        context, expanded_key, 4);

    CKKSAes128CtrOptions options;

    const auto start = std::chrono::steady_clock::now();
    const auto output = ckks_aes_ctr(
        context,
        encrypted_round_keys,
        blocks.counters,
        blocks.ciphertexts,
        options);
    const auto stop = std::chrono::steady_clock::now();

    Result result{
        std::move(method),
        std::move(parameter),
        active_blocks,
        std::chrono::duration<double>(stop - start).count(),
        0.0,
        0.0,
        0,
        context.getRotationKeyCount()};
    validate_binary_outputs(
        context,
        output,
        expected,
        active_blocks,
        result.max_error,
        result.average_error,
        result.mismatches);
    return result;
}

Result run_walsh15() {
    CKKSParams params = CKKSParams::AESWalsh15();
    const size_t complex_blocks = params.getAesWalshBlockCount();
    const size_t logical_blocks = 2 * complex_blocks;
    CKKSContext context(params);
    if (!context.hasSparseSecretEncapsulationKeys()) {
        throw std::runtime_error("Walsh sparse-secret keys are unavailable");
    }

    const auto blocks = make_blocks(logical_blocks);
    const auto expanded_key = aes128_expand_key(kTestKey);
    const auto encrypted_round_keys =
        ckks_aes128_encrypt_expanded_round_key_bits_walsh_complex_blocks(
            context, expanded_key, complex_blocks);

    CKKSAesCtrWalshSboxOptions options;
    options.parallelize_sbox_pairs = true;
    const auto start = std::chrono::steady_clock::now();
    const auto output = ckks_aes_ctr_walsh_sbox_complex_blocks(
        context,
        encrypted_round_keys,
        blocks.counters,
        blocks.ciphertexts,
        options);
    const auto stop = std::chrono::steady_clock::now();

    Result result{
        "Walsh",
        "AESWalsh15",
        logical_blocks,
        std::chrono::duration<double>(stop - start).count(),
        0.0,
        0.0,
        0,
        context.getRotationKeyCount()};
    validate_walsh_outputs(
        context,
        output,
        blocks.plaintexts,
        complex_blocks,
        result.max_error,
        result.average_error,
        result.mismatches);
    return result;
}

void print_result(const Result& result) {
    std::cout << std::fixed << std::setprecision(3)
              << "RESULT method=" << result.method
              << " parameter=" << result.parameter
              << " blocks=" << result.blocks
              << " seconds=" << result.seconds
              << std::scientific << std::setprecision(6)
              << " max_error=" << result.max_error
              << " average_error=" << result.average_error
              << std::fixed
              << " mismatches=" << result.mismatches
              << " rotations=" << result.rotations << '\n';
    if (result.mismatches != 0 || result.max_error >= 0.5) {
        throw std::runtime_error("AES benchmark correctness check failed");
    }
}

void print_table(const std::vector<Result>& results) {
    const auto walsh = std::find_if(
        results.begin(), results.end(), [](const Result& result) {
            return result.method == "Walsh";
        });
    std::cout << "\nReproduced CPU AES results\n";
    std::cout << std::left << std::setw(9) << "Method"
              << std::setw(18) << "Parameter"
              << std::right << std::setw(9) << "Blocks"
              << std::setw(13) << "Time (s)"
              << std::setw(12) << "Rel. time"
              << std::setw(15) << "Max error"
              << std::setw(15) << "Avg error"
              << std::setw(9) << "Rot." << '\n';
    for (const auto& result : results) {
        std::cout << std::left << std::setw(9) << result.method
                  << std::setw(18) << result.parameter
                  << std::right << std::setw(9) << result.blocks
                  << std::fixed << std::setprecision(3) << std::setw(13)
                  << result.seconds;
        if (walsh != results.end() && result.blocks == walsh->blocks) {
            std::ostringstream relative;
            relative << std::fixed << std::setprecision(2)
                     << result.seconds / walsh->seconds << 'x';
            std::cout << std::setw(12) << relative.str();
        } else {
            std::cout << std::setw(12) << "n/a";
        }
        std::cout
                  << std::scientific << std::setprecision(2) << std::setw(15)
                  << result.max_error << std::setw(15) << result.average_error
                  << std::fixed << std::setw(9) << result.rotations << '\n';
    }
}

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program << " COMMAND\n\n"
        << "Commands:\n"
        << "  walsh15                    AESWalsh15, 1024 blocks\n"
        << "  xboot15-full               AESXBoot15, 16384 blocks\n"
        << "  xboot15-sparse-1024        AESXBoot15, 1024 blocks\n"
        << "  xboot14-full               secure AESXBoot14, 8192 blocks\n"
        << "  xboot14-sparse-1024        secure AESXBoot14, 1024 blocks\n"
        << "  xboot13-reference-full     exact public Param-AES-13, 8192 blocks\n"
        << "  table2                     run the five main paper rows\n";
}

Result run_command(std::string_view command) {
    if (command == "walsh15") {
        return run_walsh15();
    }
    if (command == "xboot15-full") {
        auto params = CKKSParams::AESXBoot15();
        return run_xboot(
            std::move(params), "XBOOT", "AESXBoot15", size_t{1} << 14);
    }
    if (command == "xboot15-sparse-1024") {
        return run_xboot(
            CKKSParams::AESXBoot15Sparse1024(), "XBOOT*", "AESXBoot15", 1024);
    }
    if (command == "xboot14-full") {
        return run_xboot(
            CKKSParams::AESXBoot14Secure(),
            "XBOOT",
            "AESXBoot14",
            size_t{1} << 13);
    }
    if (command == "xboot14-sparse-1024") {
        return run_xboot(
            CKKSParams::AESXBoot14SecureSparse1024(),
            "XBOOT*",
            "AESXBoot14",
            1024);
    }
    if (command == "xboot13-reference-full") {
        return run_xboot(
            CKKSParams::XBootParamAes13Reference(),
            "XBOOT",
            "Param-AES-13-reference",
            size_t{1} << 13);
    }
    throw std::invalid_argument("unknown benchmark command: " + std::string(command));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        std::cout << "OpenMP threads: " << omp_get_max_threads() << '\n';
        const std::string_view command = argv[1];
        if (command == "table2") {
            const std::array<std::string_view, 5> commands{
                "xboot15-full",
                "xboot15-sparse-1024",
                "xboot14-full",
                "xboot14-sparse-1024",
                "walsh15"};
            std::vector<Result> results;
            results.reserve(commands.size());
            for (const auto item : commands) {
                std::cout << "\nRunning " << item << "\n";
                results.push_back(run_command(item));
                print_result(results.back());
            }
            print_table(results);
            return 0;
        }

        print_result(run_command(command));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
