#include "ckks_aes_ctr_walsh_sbox.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "ckks_addition.hpp"
#include "ckks_bootstrap_evalmod_inplace.hpp"
#include "ckks_ciphertext.hpp"
#include "ckks_encoding.hpp"
#include "ckks_mul.hpp"
#include "ckks_rescale.hpp"
#include "ckks_rotation.hpp"
#include "ckks_walsh_sbox.hpp"

namespace {

constexpr size_t kAesBlockBytes = 16;
constexpr size_t kAesBlockBits = 128;
constexpr size_t kAesRoundCount = 10;
constexpr size_t kAesExpandedKeyBytes = 176;
constexpr size_t kAesExpandedKeyBits = kAesExpandedKeyBytes * 8;
constexpr size_t kWalshMinimumPackedSegments = 30;

using CKKSByte = MyVector<CKKSCiphertext>;
using WalshComplexBlockSubBytesFn = MyVector<CKKSCiphertext> (*)(
    CKKSContext&,
    const MyVector<CKKSCiphertext>&,
    size_t,
    const CKKSAesCtrWalshSboxOptions&,
    std::array<CKKSWalshSboxScratch, kAesBlockBytes>&);

bool aes_scales_close(long double lhs, long double rhs, long double rel_tol = 0x1p-45L) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return false;
    }
    const long double scale =
        std::max<long double>({1.0L, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= rel_tol * scale;
}

size_t checked_mul(size_t lhs, size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::invalid_argument("ckks_aes_walsh: size overflow");
    }
    return lhs * rhs;
}

void pad_ciphertext_with_zero_polys(CKKSCiphertext& ct, size_t target_polys) {
    if (target_polys < ct.getNumPolys()) {
        throw std::invalid_argument("ckks_aes_walsh: cannot shrink ciphertext degree");
    }
    const size_t poly_size = checked_mul(ct.getN(), ct.getLevel() + 1);
    while (ct.getNumPolys() < target_polys) {
        ct.addPoly(MyVector<uint64_t>(poly_size, uint64_t{0}));
    }
}

bool ciphertext_is_zero(const CKKSCiphertext& ct) {
    for (size_t poly_idx = 0; poly_idx < ct.getNumPolys(); ++poly_idx) {
        const auto& poly = ct.getPoly(poly_idx);
        if (std::any_of(poly.begin(), poly.end(), [](uint64_t coeff) {
                return coeff != uint64_t{0};
            })) {
            return false;
        }
    }
    return true;
}

void multiply_by_integer_scalar_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    long double scalar,
    long double output_scale) {
    if (!std::isfinite(scalar) || scalar < 0.0L) {
        throw std::invalid_argument("ckks_aes_walsh: invalid scalar scale-up");
    }
    const long double scalar_rounded = std::round(scalar);
    if (scalar_rounded < 1.0L ||
        std::abs(scalar - scalar_rounded) >
            std::max<long double>(1.0L, std::abs(scalar)) * 1e-3L) {
        throw std::invalid_argument("ckks_aes_walsh: non-integral scalar scale-up");
    }
    if (scalar_rounded > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("ckks_aes_walsh: scalar scale-up is too large");
    }

    const auto scalar_i64 = static_cast<int64_t>(scalar_rounded);
    if (scalar_i64 != 1) {
        ckks_mul_inplace(context, ct, scalar_i64);
    }
    ct.setScale(output_scale);
}

void scale_up_ciphertext_to_match_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    long double target_scale) {
    if (aes_scales_close(ct.getScale(), target_scale)) {
        ct.setScale(target_scale);
        return;
    }
    if (ciphertext_is_zero(ct)) {
        ct.setScale(target_scale);
        return;
    }
    if (ct.getScale() > target_scale) {
        throw std::invalid_argument(
            "ckks_aes_walsh: cannot scale down ciphertext for addition");
    }
    multiply_by_integer_scalar_inplace(
        context,
        ct,
        target_scale / ct.getScale(),
        target_scale);
}

CKKSCiphertext align_for_add_exact(
    const CKKSContext& context,
    CKKSCiphertext ct,
    size_t level,
    long double scale,
    size_t target_polys) {
    ckks_drop_to_level_inplace(ct, level);
    pad_ciphertext_with_zero_polys(ct, target_polys);
    scale_up_ciphertext_to_match_inplace(context, ct, scale);
    return ct;
}

void add_term_inplace(const CKKSContext& context, CKKSCiphertext& acc, CKKSCiphertext term) {
    const size_t level = std::min(acc.getLevel(), term.getLevel());
    const long double scale = std::max(acc.getScale(), term.getScale());
    const size_t polys = std::max(acc.getNumPolys(), term.getNumPolys());
    acc = align_for_add_exact(context, std::move(acc), level, scale, polys);
    term = align_for_add_exact(context, std::move(term), level, scale, polys);
    ckks_add_inplace(context, acc, term);
}

uint8_t public_bit(const std::array<uint8_t, 16>& block, size_t bit_index) {
    return static_cast<uint8_t>(
        (block[bit_index / 8] >> (bit_index % 8)) & uint8_t{1});
}

size_t walsh_packing_block_count(const CKKSContext& context, size_t active_block_count) {
    const auto& params = context.getParams();
    const size_t pack_count = params.usesAesWalsh()
        ? params.getAesWalshBlockCount()
        : params.getSlots() / kWalshMinimumPackedSegments;
    const size_t packed_segment_count = params.usesAesWalsh()
        ? params.getAesWalshPackedSegmentCount()
        : kWalshMinimumPackedSegments;
    if (active_block_count == 0 ||
        pack_count == 0 ||
        packed_segment_count < kWalshMinimumPackedSegments ||
        active_block_count > pack_count ||
        packed_segment_count > params.getSlots() / pack_count) {
        throw std::invalid_argument(
            "ckks_aes_walsh: active blocks exceed padded Walsh packing capacity");
    }
    return pack_count;
}

size_t aes_walsh_state_entry_level(const CKKSContext& context) {
    return context.getParams().getBootstrapS2CDepth() + 1;
}

CKKSEncoding encode_active_complex_block_public_bit(
    const CKKSContext& context,
    std::span<const std::array<uint8_t, 16>> blocks,
    size_t bit_index,
    size_t active_complex_block_count,
    size_t level,
    long double scale) {
    if (blocks.size() != 2 * active_complex_block_count) {
        throw std::invalid_argument(
            "ckks_aes_walsh: complex-block spans must contain real and imaginary halves");
    }
    const auto& params = context.getParams();
    MyVector<Complex> slots(params.getSlots(), Complex(0.0, 0.0));
    for (size_t slot = 0; slot < active_complex_block_count; ++slot) {
        slots[slot] = Complex(
            static_cast<double>(public_bit(blocks[slot], bit_index)),
            static_cast<double>(
                public_bit(blocks[active_complex_block_count + slot], bit_index)));
    }
    return ckks_encode(context, slots, static_cast<double>(scale), level, /*eval=*/true);
}

CKKSEncoding encode_active_slot_mask(
    const CKKSContext& context,
    size_t block_count,
    size_t level,
    long double scale) {
    const auto& params = context.getParams();
    MyVector<Complex> slots(params.getSlots(), Complex(0.0, 0.0));
    for (size_t slot = 0; slot < block_count; ++slot) {
        slots[slot] = Complex(1.0, 0.0);
    }
    return ckks_encode(
        context,
        slots,
        static_cast<double>(scale),
        level,
        /*eval=*/true);
}

void mask_inactive_slots_inplace(
    const CKKSContext& context,
    MyVector<CKKSCiphertext>& state,
    size_t block_count) {
    if (state.empty()) {
        return;
    }
    const size_t level = state.front().getLevel();
    const size_t s2c_level = context.getParams().getBootstrapS2CDepth();
    if (level <= s2c_level) {
        throw std::invalid_argument(
            "ckks_aes_walsh: active slot mask requires a selection limb above S2C input level");
    }
    const long double target_scale = state.front().getScale();
    const auto mask =
        encode_active_slot_mask(
            context,
            block_count,
            level,
            static_cast<long double>(context.getParams().getModuli()[level]));
    std::exception_ptr first_exception;
    #pragma omp parallel for schedule(static) if (state.size() > 1)
    for (std::ptrdiff_t idx = 0;
         idx < static_cast<std::ptrdiff_t>(state.size());
         ++idx) {
        try {
            auto& ct = state[static_cast<size_t>(idx)];
            if (ct.getLevel() != mask.getLevel()) {
                throw std::invalid_argument(
                    "ckks_aes_walsh: active slot mask level mismatch");
            }
            if (!aes_scales_close(ct.getScale(), target_scale)) {
                throw std::invalid_argument(
                    "ckks_aes_walsh: active slot mask scale mismatch");
            }
            ckks_mul_inplace(context, ct, mask);
            ckks_rescale(context, ct);
            ct.setScale(target_scale);
        } catch (...) {
            #pragma omp critical
            {
                if (first_exception == nullptr) {
                    first_exception = std::current_exception();
                }
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
}

CKKSCiphertext zero_like(const CKKSContext& context, const CKKSCiphertext& ref) {
    auto out = ref.clone();
    ckks_mul_inplace(context, out, int64_t{0});
    return out;
}

void mask_final_refresh_slots_inplace(
    const CKKSContext& context,
    MyVector<CKKSCiphertext>& bits,
    size_t active_slot_count,
    size_t minimum_level_to_keep) {
    if (bits.empty()) {
        return;
    }
    const auto& params = context.getParams();
    if (active_slot_count > params.getSlots()) {
        throw std::invalid_argument(
            "ckks_aes_walsh: final refresh mask active slot count exceeds CKKS slots");
    }
    const size_t level = bits.front().getLevel();
    const bool consume_selection_limb = level > minimum_level_to_keep;
    const long double target_scale = bits.front().getScale();
    const long double mask_scale = consume_selection_limb
        ? static_cast<long double>(params.getModuli()[level])
        : 1.0L;
    const auto mask =
        encode_active_slot_mask(
            context,
            active_slot_count,
            level,
            mask_scale);

    std::exception_ptr first_exception;
    #pragma omp parallel for schedule(static) if (bits.size() > 1)
    for (std::ptrdiff_t idx = 0;
         idx < static_cast<std::ptrdiff_t>(bits.size());
         ++idx) {
        try {
            auto& bit = bits[static_cast<size_t>(idx)];
            if (bit.getLevel() != mask.getLevel()) {
                throw std::invalid_argument(
                    "ckks_aes_walsh: final refresh output mask level mismatch");
            }
            ckks_mul_inplace(context, bit, mask);
            if (consume_selection_limb) {
                ckks_rescale(context, bit);
                bit.setScale(target_scale);
            }
        } catch (...) {
            #pragma omp critical
            {
                if (first_exception == nullptr) {
                    first_exception = std::current_exception();
                }
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
}

void add_rotated_refresh_segment_inplace(
    const CKKSContext& context,
    CKKSCiphertext& packed,
    const CKKSCiphertext& source,
    size_t segment,
    size_t segment_stride) {
    auto term = source.clone();
    if (segment != 0) {
        ckks_rotate_inplace(
            context,
            term,
            static_cast<int64_t>(segment * segment_stride));
    }
    ckks_add_inplace(context, packed, term);
}

CKKSCiphertext pack_final_refresh_lane(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& bits,
    size_t first_bit,
    size_t bits_per_lane,
    size_t segment_stride) {
    auto packed = zero_like(context, bits.front());
    for (size_t segment = 0; segment < bits_per_lane; ++segment) {
        const size_t bit = first_bit + segment;
        if (bit >= bits.size()) {
            break;
        }
        add_rotated_refresh_segment_inplace(
            context,
            packed,
            bits[bit],
            segment,
            segment_stride);
    }
    return packed;
}

void unpack_final_refresh_lane_inplace(
    const CKKSContext& context,
    const CKKSCiphertext& packed,
    std::array<std::optional<CKKSCiphertext>, kAesBlockBits>& refreshed,
    size_t first_bit,
    size_t bits_per_lane,
    size_t segment_stride) {
    for (size_t segment = 0; segment < bits_per_lane; ++segment) {
        const size_t bit = first_bit + segment;
        if (bit >= refreshed.size()) {
            break;
        }
        auto out = packed.clone();
        if (segment != 0) {
            ckks_rotate_inplace(
                context,
                out,
                -static_cast<int64_t>(segment * segment_stride));
        }
        refreshed[bit] = std::move(out);
    }
}

CKKSCiphertext combine_real_imag_for_final_refresh(
    const CKKSContext& context,
    CKKSCiphertext real,
    const CKKSCiphertext& imag) {
    auto imag_i = ckks_mul(
        context,
        imag,
        Complex(0.0, 1.0),
        1.0);
    imag_i.setScale(imag.getScale());
    ckks_add_inplace(context, real, imag_i);
    return real;
}

void final_binary_refresh_complex_block_output_bits_inplace(
    const CKKSContext& context,
    MyVector<CKKSCiphertext>& bits,
    size_t active_complex_block_count,
    bool parallelize) {
    if (bits.size() != kAesBlockBits) {
        throw std::invalid_argument(
            "ckks_aes_walsh: expected 128 output bits for final refresh");
    }
    const auto& params = context.getParams();
    const size_t segment_stride = active_complex_block_count;
    const size_t bits_per_batch = segment_stride == 0
        ? size_t{0}
        : params.getSlots() / segment_stride;
    if (segment_stride == 0 ||
        bits_per_batch == 0 ||
        segment_stride * bits_per_batch > params.getSlots()) {
        throw std::invalid_argument(
            "ckks_aes_walsh: invalid final refresh packing layout");
    }
    mask_final_refresh_slots_inplace(
        context,
        bits,
        segment_stride,
        params.getBootstrapS2CDepth());

    const size_t batch_count =
        (kAesBlockBits + bits_per_batch - 1) / bits_per_batch;
    std::array<std::optional<CKKSCiphertext>, kAesBlockBits> refreshed;
    std::exception_ptr first_exception;

    #pragma omp parallel for schedule(dynamic) if (parallelize)
    for (std::ptrdiff_t batch_idx = 0;
         batch_idx < static_cast<std::ptrdiff_t>(batch_count);
         ++batch_idx) {
        try {
            const size_t first_bit =
                static_cast<size_t>(batch_idx) * bits_per_batch;
            auto packed = pack_final_refresh_lane(
                context,
                bits,
                first_bit,
                bits_per_batch,
                segment_stride);
            auto [real_refreshed, imag_refreshed] =
                ckks_batch_bootstrap_evalmod_from_complex_packed(
                    context,
                    std::move(packed));
            auto combined = combine_real_imag_for_final_refresh(
                context,
                std::move(real_refreshed),
                imag_refreshed);
            unpack_final_refresh_lane_inplace(
                context,
                combined,
                refreshed,
                first_bit,
                bits_per_batch,
                segment_stride);
        } catch (...) {
            #pragma omp critical
            {
                if (first_exception == nullptr) {
                    first_exception = std::current_exception();
                }
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }

    for (size_t bit = 0; bit < kAesBlockBits; ++bit) {
        if (!refreshed[bit].has_value()) {
            throw std::runtime_error(
                "ckks_aes_walsh: final refresh missing output bit");
        }
        bits[bit] = std::move(*refreshed[bit]);
    }
    mask_final_refresh_slots_inplace(
        context,
        bits,
        segment_stride,
        /*minimum_level_to_keep=*/0);
}

void add_public_complex_block_bits_lazy_inplace(
    const CKKSContext& context,
    MyVector<CKKSCiphertext>& bits,
    std::span<const std::array<uint8_t, 16>> public_blocks) {
    if (bits.size() != kAesBlockBits ||
        public_blocks.empty() ||
        public_blocks.size() % 2 != 0) {
        throw std::invalid_argument(
            "ckks_aes_walsh: complex-block public XOR input must be 128 bits and an even block count");
    }

    const size_t active_complex_block_count = public_blocks.size() / 2;
    std::exception_ptr first_exception;
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t bit_idx = 0;
         bit_idx < static_cast<std::ptrdiff_t>(kAesBlockBits);
         ++bit_idx) {
        try {
            const size_t bit = static_cast<size_t>(bit_idx);
            auto& ct = bits[bit];
            auto encoded_block = encode_active_complex_block_public_bit(
                context,
                public_blocks,
                bit,
                active_complex_block_count,
                ct.getLevel(),
                ct.getScale());
            ckks_add_inplace(context, ct, encoded_block);
        } catch (...) {
            #pragma omp critical
            {
                if (first_exception == nullptr) {
                    first_exception = std::current_exception();
                }
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
}

MyVector<CKKSCiphertext> make_initial_counter_complex_block_state(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& encrypted_expanded_round_key_bits,
    std::span<const std::array<uint8_t, 16>> counter_blocks) {
    if (counter_blocks.empty() || counter_blocks.size() % 2 != 0) {
        throw std::invalid_argument(
            "ckks_aes_walsh: complex-block counter span must have an even block count");
    }
    const size_t active_complex_block_count = counter_blocks.size() / 2;

    MyVector<CKKSCiphertext> state;
    state.reserve(kAesBlockBits);
    for (size_t bit = 0; bit < kAesBlockBits; ++bit) {
        state.emplace_back(encrypted_expanded_round_key_bits[bit].clone());
        auto encoded_counter =
            encode_active_complex_block_public_bit(
                context,
                counter_blocks,
                bit,
                active_complex_block_count,
                state.back().getLevel(),
                state.back().getScale());
        ckks_add_inplace(context, state.back(), encoded_counter);
    }
    return state;
}

CKKSByte clone_byte_bits(
    const MyVector<CKKSCiphertext>& state,
    size_t byte_index) {
    if (byte_index >= kAesBlockBytes) {
        throw std::invalid_argument("ckks_aes_walsh: byte index out of range");
    }
    CKKSByte out;
    out.reserve(8);
    for (size_t bit = 0; bit < 8; ++bit) {
        out.emplace_back(state[byte_index * 8 + bit].clone());
    }
    return out;
}

CKKSCiphertext add_bits_lazy(
    const CKKSContext& context,
    const CKKSCiphertext& lhs,
    const CKKSCiphertext& rhs) {
    auto out = lhs.clone();
    add_term_inplace(context, out, rhs.clone());
    return out;
}

CKKSByte add_bytes_lazy(
    const CKKSContext& context,
    const CKKSByte& lhs,
    const CKKSByte& rhs) {
    if (lhs.size() != 8 || rhs.size() != 8) {
        throw std::invalid_argument("ckks_aes_walsh: byte operands must have 8 bits");
    }
    CKKSByte out;
    out.reserve(8);
    for (size_t bit = 0; bit < 8; ++bit) {
        out.emplace_back(add_bits_lazy(context, lhs[bit], rhs[bit]));
    }
    return out;
}

void add_byte_lazy_inplace(
    const CKKSContext& context,
    CKKSByte& acc,
    const CKKSByte& term) {
    if (acc.size() != 8 || term.size() != 8) {
        throw std::invalid_argument("ckks_aes_walsh: byte operands must have 8 bits");
    }
    for (size_t bit = 0; bit < 8; ++bit) {
        add_term_inplace(context, acc[bit], term[bit].clone());
    }
}

CKKSByte xtime_byte_lazy(
    const CKKSContext& context,
    const CKKSByte& byte) {
    if (byte.size() != 8) {
        throw std::invalid_argument("ckks_aes_walsh: xtime input must have 8 bits");
    }
    CKKSByte out;
    out.reserve(8);
    out.emplace_back(byte[7].clone());
    out.emplace_back(add_bits_lazy(context, byte[0], byte[7]));
    out.emplace_back(byte[1].clone());
    out.emplace_back(add_bits_lazy(context, byte[2], byte[7]));
    out.emplace_back(add_bits_lazy(context, byte[3], byte[7]));
    out.emplace_back(byte[4].clone());
    out.emplace_back(byte[5].clone());
    out.emplace_back(byte[6].clone());
    return out;
}

CKKSByte fmix_column_byte(
    const CKKSContext& context,
    const CKKSByte& b0,
    const CKKSByte& b1,
    const CKKSByte& b2,
    const CKKSByte& b3) {
    auto doubled = xtime_byte_lazy(context, add_bytes_lazy(context, b0, b1));
    add_byte_lazy_inplace(context, doubled, b1);
    add_byte_lazy_inplace(context, doubled, b2);
    add_byte_lazy_inplace(context, doubled, b3);
    return doubled;
}

void flatten_bytes_to_bit_state(
    std::array<CKKSByte, kAesBlockBytes>& bytes,
    MyVector<CKKSCiphertext>& state) {
    state.clear();
    state.reserve(kAesBlockBits);
    for (size_t byte = 0; byte < kAesBlockBytes; ++byte) {
        if (bytes[byte].size() != 8) {
            throw std::runtime_error(
                "ckks_aes_walsh: internal bit byte has the wrong size");
        }
        for (size_t bit = 0; bit < 8; ++bit) {
            state.emplace_back(std::move(bytes[byte][bit]));
        }
    }
}

MyVector<CKKSCiphertext> sub_bytes_walsh_complex_blocks_mask_packed(
    CKKSContext& context,
    const MyVector<CKKSCiphertext>& state,
    size_t block_count,
    const CKKSAesCtrWalshSboxOptions& options,
    std::array<CKKSWalshSboxScratch, kAesBlockBytes>& scratch) {
    if (state.size() != kAesBlockBits) {
        throw std::invalid_argument(
            "ckks_aes_walsh: complex-block Walsh SubBytes input must have 128 bits");
    }

    std::array<std::optional<CKKSByte>, kAesBlockBytes> output_bytes;
    std::exception_ptr first_exception;

    #pragma omp parallel for schedule(dynamic) if (options.parallelize_sbox_pairs)
    for (std::ptrdiff_t byte_idx = 0;
         byte_idx < static_cast<std::ptrdiff_t>(kAesBlockBytes);
         ++byte_idx) {
        try {
            const size_t byte = static_cast<size_t>(byte_idx);
            MyVector<CKKSCiphertext> input;
            input.reserve(8);
            for (size_t bit = 0; bit < 8; ++bit) {
                input.emplace_back(state[byte * 8 + bit].clone());
            }

            auto output = ckks_walsh_sbox_complex_blocks_mask_packed(
                context,
                std::move(input),
                block_count,
                scratch[byte]);
            if (output.size() != 8) {
                throw std::runtime_error(
                    "ckks_aes_walsh: complex-block Walsh S-box returned wrong size");
            }
            output_bytes[byte] = std::move(output);
        } catch (...) {
            #pragma omp critical
            {
                if (first_exception == nullptr) {
                    first_exception = std::current_exception();
                }
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }

    MyVector<CKKSCiphertext> out;
    out.reserve(kAesBlockBits);
    for (size_t byte = 0; byte < kAesBlockBytes; ++byte) {
        if (!output_bytes[byte].has_value()) {
            throw std::runtime_error(
                "ckks_aes_walsh: complex-block Walsh S-box output missing");
        }
        auto& output = *output_bytes[byte];
        for (size_t bit = 0; bit < 8; ++bit) {
            out.emplace_back(std::move(output[bit]));
        }
    }
    return out;
}

void shift_rows_bits_inplace(MyVector<CKKSCiphertext>& state) {
    if (state.size() != kAesBlockBits) {
        throw std::invalid_argument(
            "ckks_aes_walsh: bit ShiftRows state must have 128 ciphertexts");
    }
    auto old_state = std::move(state);
    state.clear();
    state.reserve(kAesBlockBits);
    for (size_t column = 0; column < 4; ++column) {
        for (size_t row = 0; row < 4; ++row) {
            const size_t src_byte = 4 * ((column + row) % 4) + row;
            for (size_t bit = 0; bit < 8; ++bit) {
                state.emplace_back(std::move(old_state[src_byte * 8 + bit]));
            }
        }
    }
}

void fmix_columns_bits_inplace(
    const CKKSContext& context,
    MyVector<CKKSCiphertext>& state) {
    if (state.size() != kAesBlockBits) {
        throw std::invalid_argument(
            "ckks_aes_walsh: bit MixColumns state must have 128 ciphertexts");
    }

    std::array<CKKSByte, kAesBlockBytes> out_bytes;
    std::exception_ptr first_exception;
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t column_idx = 0; column_idx < 4; ++column_idx) {
        try {
            const size_t column = static_cast<size_t>(column_idx);
            const size_t b0_idx = 4 * column + 0;
            const size_t b1_idx = 4 * column + 1;
            const size_t b2_idx = 4 * column + 2;
            const size_t b3_idx = 4 * column + 3;

            const auto b0 = clone_byte_bits(state, b0_idx);
            const auto b1 = clone_byte_bits(state, b1_idx);
            const auto b2 = clone_byte_bits(state, b2_idx);
            const auto b3 = clone_byte_bits(state, b3_idx);

            out_bytes[b0_idx] = fmix_column_byte(context, b0, b1, b2, b3);
            out_bytes[b1_idx] = fmix_column_byte(context, b1, b2, b3, b0);
            out_bytes[b2_idx] = fmix_column_byte(context, b2, b3, b0, b1);
            out_bytes[b3_idx] = fmix_column_byte(context, b3, b0, b1, b2);
        } catch (...) {
            #pragma omp critical
            {
                if (first_exception == nullptr) {
                    first_exception = std::current_exception();
                }
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }

    flatten_bytes_to_bit_state(out_bytes, state);
}

void add_round_key_lazy_inplace(
    const CKKSContext& context,
    MyVector<CKKSCiphertext>& state,
    const MyVector<CKKSCiphertext>& encrypted_round_key_bits,
    size_t round_index) {
    if (state.size() != kAesBlockBits || round_index > kAesRoundCount) {
        throw std::invalid_argument("ckks_aes_walsh: invalid AddRoundKey bit-state input");
    }
    const size_t key_offset = round_index * kAesBlockBits;
    std::exception_ptr first_exception;
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t bit_idx = 0;
         bit_idx < static_cast<std::ptrdiff_t>(kAesBlockBits);
         ++bit_idx) {
        try {
            const size_t bit = static_cast<size_t>(bit_idx);
            add_term_inplace(
                context,
                state[bit],
                encrypted_round_key_bits[key_offset + bit].clone());
        } catch (...) {
            #pragma omp critical
            {
                if (first_exception == nullptr) {
                    first_exception = std::current_exception();
                }
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
}

void run_complex_block_full_round_inplace(
    CKKSContext& context,
    MyVector<CKKSCiphertext>& state,
    const MyVector<CKKSCiphertext>& encrypted_round_key_bits,
    size_t round,
    size_t packing_block_count,
    const CKKSAesCtrWalshSboxOptions& options,
    std::array<CKKSWalshSboxScratch, kAesBlockBytes>& sbox_scratch,
    WalshComplexBlockSubBytesFn sub_bytes) {
    mask_inactive_slots_inplace(context, state, packing_block_count);
    state = sub_bytes(
        context,
        state,
        packing_block_count,
        options,
        sbox_scratch);
    shift_rows_bits_inplace(state);
    fmix_columns_bits_inplace(context, state);
    add_round_key_lazy_inplace(
        context,
        state,
        encrypted_round_key_bits,
        round);
}

void run_complex_block_final_round_inplace(
    CKKSContext& context,
    MyVector<CKKSCiphertext>& state,
    const MyVector<CKKSCiphertext>& encrypted_round_key_bits,
    size_t packing_block_count,
    const CKKSAesCtrWalshSboxOptions& options,
    std::array<CKKSWalshSboxScratch, kAesBlockBytes>& sbox_scratch,
    WalshComplexBlockSubBytesFn sub_bytes) {
    mask_inactive_slots_inplace(context, state, packing_block_count);
    state = sub_bytes(
        context,
        state,
        packing_block_count,
        options,
        sbox_scratch);
    shift_rows_bits_inplace(state);
    add_round_key_lazy_inplace(
        context,
        state,
        encrypted_round_key_bits,
        kAesRoundCount);
}

MyVector<CKKSCiphertext> evaluate_aes128_rounds_walsh_complex_blocks_impl(
    CKKSContext& context,
    const MyVector<CKKSCiphertext>& encrypted_round_key_bits,
    MyVector<CKKSCiphertext> state,
    size_t packing_block_count,
    const CKKSAesCtrWalshSboxOptions& options,
    WalshComplexBlockSubBytesFn sub_bytes,
    const char* state_label) {
    if (state.size() != kAesBlockBits) {
        throw std::invalid_argument(
            std::string("ckks_aes_walsh: ") + state_label +
            " AES state must have 128 ciphertexts");
    }

    std::array<CKKSWalshSboxScratch, kAesBlockBytes> sbox_scratch;
    for (size_t round = 1; round < kAesRoundCount; ++round) {
        run_complex_block_full_round_inplace(
            context,
            state,
            encrypted_round_key_bits,
            round,
            packing_block_count,
            options,
            sbox_scratch,
            sub_bytes);
    }

    run_complex_block_final_round_inplace(
        context,
        state,
        encrypted_round_key_bits,
        packing_block_count,
        options,
        sbox_scratch,
        sub_bytes);
    return state;
}

void validate_complex_block_inputs(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& encrypted_expanded_round_key_bits,
    std::span<const std::array<uint8_t, 16>> counter_blocks,
    std::span<const std::array<uint8_t, 16>> ciphertext_blocks) {
    if (encrypted_expanded_round_key_bits.size() != kAesExpandedKeyBits) {
        throw std::invalid_argument("ckks_aes_walsh: expanded round key must have 1408 bits");
    }
    if (counter_blocks.empty() ||
        counter_blocks.size() != ciphertext_blocks.size() ||
        counter_blocks.size() % 2 != 0) {
        throw std::invalid_argument(
            "ckks_aes_walsh: complex-block counter/ciphertext spans must be non-empty, equal, and even");
    }
    const size_t active_complex_block_count = counter_blocks.size() / 2;
    static_cast<void>(walsh_packing_block_count(context, active_complex_block_count));
    const size_t key_level = aes_walsh_state_entry_level(context);
    if (key_level > context.getParams().getMaxLevel()) {
        throw std::invalid_argument(
            "ckks_aes_walsh: Walsh path requires a selection limb above S2C input level");
    }
    for (const auto& ct : encrypted_expanded_round_key_bits) {
        if (ct.getN() != context.getParams().getN() ||
            ct.getNumPolys() != 2 ||
            ct.getLevel() != key_level) {
            throw std::invalid_argument(
                "ckks_aes_walsh: complex-block path received a round key bit at the wrong level");
        }
    }
}

MyVector<CKKSCiphertext> run_aes_ctr_walsh_complex_block_path(
    CKKSContext& context,
    const MyVector<CKKSCiphertext>& encrypted_expanded_round_key_bits,
    std::span<const std::array<uint8_t, 16>> counter_blocks,
    std::span<const std::array<uint8_t, 16>> ciphertext_blocks,
    const CKKSAesCtrWalshSboxOptions& options,
    WalshComplexBlockSubBytesFn sub_bytes,
    const char* state_label) {
    validate_complex_block_inputs(
        context,
        encrypted_expanded_round_key_bits,
        counter_blocks,
        ciphertext_blocks);

    auto counter_state = make_initial_counter_complex_block_state(
        context,
        encrypted_expanded_round_key_bits,
        counter_blocks);
    const size_t active_complex_block_count = counter_blocks.size() / 2;
    const size_t packing_block_count =
        walsh_packing_block_count(context, active_complex_block_count);

    // CTR mode decrypts by XORing the ciphertext with AES(counter).
    auto plaintext_bits = evaluate_aes128_rounds_walsh_complex_blocks_impl(
        context,
        encrypted_expanded_round_key_bits,
        std::move(counter_state),
        packing_block_count,
        options,
        sub_bytes,
        state_label);
    add_public_complex_block_bits_lazy_inplace(
        context,
        plaintext_bits,
        ciphertext_blocks);
    final_binary_refresh_complex_block_output_bits_inplace(
        context,
        plaintext_bits,
        active_complex_block_count,
        options.parallelize_sbox_pairs);
    return plaintext_bits;
}

}  // namespace

MyVector<CKKSCiphertext> ckks_aes128_encrypt_expanded_round_key_bits_walsh_complex_blocks(
    const CKKSContext& context,
    const std::array<uint8_t, 176>& expanded_round_keys,
    size_t active_complex_block_count) {
    const auto& params = context.getParams();
    const size_t packing_block_count =
        walsh_packing_block_count(context, active_complex_block_count);
    static_cast<void>(packing_block_count);

    const size_t key_level = aes_walsh_state_entry_level(context);
    if (key_level > params.getMaxLevel()) {
        throw std::invalid_argument(
            "ckks_aes_walsh: selection key level exceeds parameter chain");
    }
    std::array<std::optional<CKKSCiphertext>, kAesExpandedKeyBits> encrypted_optional;

    #pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t bit_idx = 0;
         bit_idx < static_cast<std::ptrdiff_t>(kAesExpandedKeyBits);
         ++bit_idx) {
        const size_t absolute_bit = static_cast<size_t>(bit_idx);
        const size_t byte = absolute_bit / 8;
        const size_t bit = absolute_bit % 8;
        const double value =
            ((expanded_round_keys[byte] >> bit) & uint8_t{1}) != 0U ? 1.0 : 0.0;
        MyVector<Complex> slots(params.getSlots(), Complex(0.0, 0.0));
        for (size_t slot = 0; slot < active_complex_block_count; ++slot) {
            slots[slot] = Complex(value, value);
        }
        const double key_scale = absolute_bit < kAesBlockBits
            ? params.getScale()
            : (absolute_bit >= kAesRoundCount * kAesBlockBits
                   ? params.getScale()
                   : params.getScale() / 4.0);
        auto encoded_key = ckks_encode(
            context,
            slots,
            key_scale,
            key_level,
            /*eval=*/true);
        encrypted_optional[absolute_bit] =
            ckks_encrypt(context, encoded_key, /*eval=*/true);
    }

    MyVector<CKKSCiphertext> encrypted_bits;
    encrypted_bits.reserve(kAesExpandedKeyBits);
    for (size_t bit = 0; bit < kAesExpandedKeyBits; ++bit) {
        if (!encrypted_optional[bit].has_value()) {
            throw std::runtime_error(
                "ckks_aes_walsh: encrypted complex-block round-key bit missing");
        }
        encrypted_bits.emplace_back(std::move(*encrypted_optional[bit]));
    }
    return encrypted_bits;
}

MyVector<CKKSCiphertext> ckks_aes_ctr_walsh_sbox_complex_blocks(
    CKKSContext& context,
    const MyVector<CKKSCiphertext>& encrypted_expanded_round_key_bits,
    std::span<const std::array<uint8_t, 16>> counter_blocks,
    std::span<const std::array<uint8_t, 16>> ciphertext_blocks,
    const CKKSAesCtrWalshSboxOptions& options) {
    return run_aes_ctr_walsh_complex_block_path(
        context,
        encrypted_expanded_round_key_bits,
        counter_blocks,
        ciphertext_blocks,
        options,
        sub_bytes_walsh_complex_blocks_mask_packed,
        "complex-block");
}
