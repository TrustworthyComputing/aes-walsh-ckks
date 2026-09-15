#include "ckks_aes_ctr.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#include "aes_sbox.hpp"
#include "ckks_addition.hpp"
#include "ckks_bootstrap_evalmod_inplace.hpp"
#include "ckks_ciphertext.hpp"
#include "ckks_encoding.hpp"
#include "ckks_mul.hpp"
#include "ckks_relin.hpp"
#include "ckks_rescale.hpp"
#include "ntt.hpp"

namespace {

constexpr size_t kAesBlockBytes = 16;
constexpr size_t kAesBlockBits = 128;
constexpr size_t kAesRoundCount = 10;
constexpr size_t kAesExpandedKeyBytes = 176;
constexpr size_t kAesExpandedKeyBits = kAesExpandedKeyBytes * 8;

using CKKSByte = MyVector<CKKSCiphertext>;

struct XBootSboxMonomial {
    size_t mask = 0;
    CKKSCiphertext value;

    XBootSboxMonomial(size_t mask_in, CKKSCiphertext value_in)
        : mask(mask_in),
          value(std::move(value_in)) {}

    XBootSboxMonomial(XBootSboxMonomial&&) noexcept = default;
    XBootSboxMonomial& operator=(XBootSboxMonomial&&) noexcept = default;
    XBootSboxMonomial(const XBootSboxMonomial&) = delete;
    XBootSboxMonomial& operator=(const XBootSboxMonomial&) = delete;
};

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
        throw std::invalid_argument("ckks_aes: size overflow");
    }
    return lhs * rhs;
}

void pad_ciphertext_with_zero_polys(CKKSCiphertext& ct, size_t target_polys) {
    if (target_polys < ct.getNumPolys()) {
        throw std::invalid_argument("ckks_aes: cannot shrink ciphertext degree");
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

CKKSCiphertext zero_like(const CKKSContext& context, const CKKSCiphertext& ref) {
    auto out = ref.clone();
    ckks_mul_inplace(context, out, int64_t{0});
    return out;
}

void multiply_by_integer_scalar_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    long double scalar,
    long double output_scale) {
    if (!std::isfinite(scalar) || scalar < 0.0L) {
        throw std::invalid_argument("ckks_aes: invalid scalar scale-up");
    }
    const long double scalar_rounded = std::round(scalar);
    if (scalar_rounded < 1.0L ||
        std::abs(scalar - scalar_rounded) >
            std::max<long double>(1.0L, std::abs(scalar)) * 0x1p-20L) {
        throw std::invalid_argument("ckks_aes: non-integral scalar scale-up");
    }
    if (scalar_rounded > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("ckks_aes: scalar scale-up is too large");
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
        throw std::invalid_argument("ckks_aes: cannot scale down ciphertext for addition");
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

CKKSCiphertext xboot_mul_relin_rescale(
    const CKKSContext& context,
    const CKKSCiphertext& lhs,
    const CKKSCiphertext& rhs,
    long double target_scale) {
    if (lhs.getLevel() == 0 || rhs.getLevel() == 0) {
        throw std::invalid_argument("ckks_aes: insufficient level for XBOOT S-box multiplication");
    }
    const size_t level = std::min(lhs.getLevel(), rhs.getLevel());

    auto out = lhs.clone();
    ckks_drop_to_level_inplace(out, level);
    out.setScale(target_scale);

    auto rhs_aligned = rhs.clone();
    ckks_drop_to_level_inplace(rhs_aligned, level);
    rhs_aligned.setScale(target_scale);

    ckks_mul_inplace(context, out, rhs_aligned);
    ckks_relin_hybrid_inplace(context, out);
    ckks_rescale(context, out);
    out.setScale(target_scale);
    return out;
}

std::vector<XBootSboxMonomial> xboot_layered_monomials(
    const CKKSContext& context,
    const CKKSByte& input,
    size_t begin,
    size_t end,
    long double target_scale) {
    const size_t count = end - begin;
    if (count == 0 || (count & (count - 1)) != 0) {
        throw std::invalid_argument("ckks_aes: XBOOT S-box monomial range must be a power of two");
    }
    if (count == 1) {
        auto value = input[begin].clone();
        value.setScale(target_scale);
        std::vector<XBootSboxMonomial> out;
        out.emplace_back(size_t{1} << begin, std::move(value));
        return out;
    }
    if (count == 2) {
        auto lhs = input[begin].clone();
        lhs.setScale(target_scale);
        auto rhs = input[begin + 1].clone();
        rhs.setScale(target_scale);
        auto product = xboot_mul_relin_rescale(
            context,
            lhs,
            rhs,
            target_scale);
        std::vector<XBootSboxMonomial> out;
        out.reserve(3);
        out.emplace_back(size_t{1} << begin, std::move(lhs));
        out.emplace_back(size_t{1} << (begin + 1), std::move(rhs));
        out.emplace_back(
            (size_t{1} << begin) | (size_t{1} << (begin + 1)),
            std::move(product));
        return out;
    }

    const size_t mid = begin + count / 2;
    auto left = xboot_layered_monomials(
        context,
        input,
        begin,
        mid,
        target_scale);
    auto right = xboot_layered_monomials(
        context,
        input,
        mid,
        end,
        target_scale);

    std::vector<XBootSboxMonomial> out;
    out.reserve(left.size() * right.size() + left.size() + right.size());
    for (const auto& left_term : left) {
        for (const auto& right_term : right) {
            out.emplace_back(
                left_term.mask | right_term.mask,
                xboot_mul_relin_rescale(
                    context,
                    left_term.value,
                    right_term.value,
                    target_scale));
        }
    }
    for (auto& term : left) {
        out.emplace_back(std::move(term));
    }
    for (auto& term : right) {
        out.emplace_back(std::move(term));
    }
    return out;
}

const std::array<std::array<int64_t, 256>, 8>& xboot_sbox_coefficients() {
    static const std::array<std::array<int64_t, 256>, 8> coeffs = [] {
        std::array<std::array<int64_t, 256>, 8> out{};
        for (size_t bit = 0; bit < 8; ++bit) {
            for (size_t input = 0; input < 256; ++input) {
                out[bit][input] =
                    static_cast<int64_t>((kAesSboxTable[input] >> bit) & 1U);
            }
            for (size_t variable = 0; variable < 8; ++variable) {
                const size_t variable_mask = size_t{1} << variable;
                for (size_t mask = 0; mask < 256; ++mask) {
                    if ((mask & variable_mask) != 0) {
                        out[bit][mask] -= out[bit][mask ^ variable_mask];
                    }
                }
            }
        }
        return out;
    }();
    return coeffs;
}

CKKSByte ckks_aes_xboot_sbox_impl(
    const CKKSContext& context,
    const CKKSByte& input) {
    if (input.size() != 8) {
        throw std::invalid_argument("ckks_aes: XBOOT S-box input must have 8 bits");
    }
    const long double target_scale = input.front().getScale();
    auto layered = xboot_layered_monomials(
        context,
        input,
        0,
        8,
        target_scale);

    std::array<std::optional<CKKSCiphertext>, 256> monomials;
    for (auto& term : layered) {
        if (term.mask == 0 || term.mask >= monomials.size()) {
            throw std::runtime_error("ckks_aes: invalid XBOOT S-box monomial mask");
        }
        monomials[term.mask] = std::move(term.value);
    }
    for (size_t mask = 1; mask < monomials.size(); ++mask) {
        if (!monomials[mask].has_value()) {
            throw std::runtime_error("ckks_aes: missing XBOOT S-box monomial");
        }
    }

    const auto& coeffs = xboot_sbox_coefficients();
    CKKSByte output;
    output.reserve(8);
    for (size_t bit = 0; bit < 8; ++bit) {
        std::optional<CKKSCiphertext> acc;
        if (coeffs[bit][0] != 0) {
            acc = zero_like(context, input.front());
            ckks_add_integer_constant_inplace(context, *acc, coeffs[bit][0]);
        }
        for (size_t mask = 1; mask < 256; ++mask) {
            const int64_t coeff = coeffs[bit][mask];
            if (coeff == 0) {
                continue;
            }
            auto term = monomials[mask]->clone();
            ckks_mul_inplace(context, term, coeff);
            if (acc.has_value()) {
                add_term_inplace(context, *acc, std::move(term));
            } else {
                acc = std::move(term);
            }
        }
        if (!acc.has_value()) {
            acc = zero_like(context, input.front());
        }
        output.emplace_back(std::move(*acc));
    }
    return output;
}

CKKSEncoding encode_message_block(
    const CKKSContext& context,
    std::span<const std::array<uint8_t, 16>> public_blocks,
    size_t bit_index,
    size_t bit_level) {

    const auto& params = context.getParams();
    const size_t logical_slots = params.hasBootstrapParams()
        ? params.getBootstrapActiveSlotCount()
        : params.getSlots();
    if (public_blocks.empty() || public_blocks.size() > logical_slots) {
        throw std::invalid_argument(
            "ckks_aes: public block count exceeds logical slot count");
    }

    MyVector<Complex> block(params.getSlots());
    for (size_t slot = 0; slot < params.getSlots(); ++slot) {
        const size_t logical_slot = slot % logical_slots;
        const uint8_t bit = logical_slot < public_blocks.size()
            ? static_cast<uint8_t>(
                  (public_blocks[logical_slot][bit_index / 8] >> (bit_index % 8)) & 1U)
            : uint8_t{0};
        block[slot] = Complex(static_cast<double>(bit), 0.0);
    }
    auto encoded_block = ckks_encode(context, block, bit_level, true);
    return encoded_block;
}

CKKSByte clone_byte(const MyVector<CKKSCiphertext>& state, size_t byte_index) {
    if (byte_index >= kAesBlockBytes) {
        throw std::invalid_argument("ckks_aes: byte index out of range");
    }
    CKKSByte out;
    out.reserve(8);
    for (size_t bit = 0; bit < 8; ++bit) {
        out.emplace_back(state[byte_index * 8 + bit].clone());
    }
    return out;
}

CKKSByte add_bytes_lazy(
    const CKKSContext& context,
    const CKKSByte& lhs,
    const CKKSByte& rhs) {
    if (lhs.size() != 8 || rhs.size() != 8) {
        throw std::invalid_argument("ckks_aes: byte operands must have 8 bits");
    }
    CKKSByte out;
    out.reserve(8);
    for (size_t bit = 0; bit < 8; ++bit) {
        auto sum = lhs[bit].clone();
        ckks_add_inplace(context, sum, rhs[bit]);
        out.emplace_back(std::move(sum));
    }
    return out;
}

void add_byte_lazy_inplace(
    const CKKSContext& context,
    CKKSByte& acc,
    const CKKSByte& term) {
    if (acc.size() != 8 || term.size() != 8) {
        throw std::invalid_argument("ckks_aes: byte operands must have 8 bits");
    }
    for (size_t bit = 0; bit < 8; ++bit) {
        ckks_add_inplace(context, acc[bit], term[bit]);
    }
}

CKKSCiphertext add_bits_lazy(
    const CKKSContext& context,
    const CKKSCiphertext& lhs,
    const CKKSCiphertext& rhs) {
    auto out = lhs.clone();
    ckks_add_inplace(context, out, rhs);
    return out;
}

CKKSByte xtime_byte_lazy(const CKKSContext& context, const CKKSByte& byte) {
    if (byte.size() != 8) {
        throw std::invalid_argument("ckks_aes: xtime input must have 8 bits");
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

void flatten_bytes_to_state(
    std::array<CKKSByte, kAesBlockBytes>& bytes,
    MyVector<CKKSCiphertext>& state) {
    state.clear();
    state.reserve(kAesBlockBits);
    for (size_t byte = 0; byte < kAesBlockBytes; ++byte) {
        if (bytes[byte].size() != 8) {
            throw std::runtime_error("ckks_aes: internal byte has the wrong size");
        }
        for (size_t bit = 0; bit < 8; ++bit) {
            state.emplace_back(std::move(bytes[byte][bit]));
        }
    }
}

void add_round_key_lazy_inplace(
    const CKKSContext& context,
    MyVector<CKKSCiphertext>& state,
    const MyVector<CKKSCiphertext>& encrypted_round_key_bits,
    size_t round_index) {
    if (state.size() != kAesBlockBits || round_index > kAesRoundCount) {
        throw std::invalid_argument("ckks_aes: invalid AddRoundKey input");
    }
    const size_t key_offset = round_index * kAesBlockBits;
    for (size_t bit = 0; bit < kAesBlockBits; ++bit) {
        ckks_add_inplace(context, state[bit], encrypted_round_key_bits[key_offset + bit]);
    }
}

void sub_bytes_inplace(
    CKKSContext& context,
    MyVector<CKKSCiphertext>& state,
    const CKKSAes128CtrOptions& options) {
    if (state.size() != kAesBlockBits) {
        throw std::invalid_argument("ckks_aes: SubBytes state must have 128 bits");
    }
    std::array<std::optional<CKKSByte>, kAesBlockBytes> output_bytes;

    #pragma omp parallel for schedule(dynamic) if (options.parallelize_independent_work)
    for (std::ptrdiff_t byte_idx = 0;
         byte_idx < static_cast<std::ptrdiff_t>(kAesBlockBytes);
         ++byte_idx) {
        const size_t byte = static_cast<size_t>(byte_idx);
        MyVector<CKKSCiphertext> input;
        input.reserve(8);
        for (size_t bit = 0; bit < 8; ++bit) {
            input.emplace_back(state[byte * 8 + bit].clone());
        }
        auto output = ckks_aes_xboot_sbox_impl(context, input);
        if (output.size() != 8) {
            throw std::runtime_error("ckks_aes: AES S-box returned the wrong bit count");
        }
        output_bytes[byte] = std::move(output);
    }

    for (size_t byte = 0; byte < kAesBlockBytes; ++byte) {
        if (!output_bytes[byte].has_value()) {
            throw std::runtime_error("ckks_aes: AES S-box output missing");
        }
        auto& output = *output_bytes[byte];
        for (size_t bit = 0; bit < 8; ++bit) {
            state[byte * 8 + bit] = std::move(output[bit]);
        }
    }
}

void shift_rows_inplace(MyVector<CKKSCiphertext>& state) {
    if (state.size() != kAesBlockBits) {
        throw std::invalid_argument("ckks_aes: ShiftRows state must have 128 bits");
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

void fmix_columns_inplace(
    const CKKSContext& context,
    MyVector<CKKSCiphertext>& state) {
    if (state.size() != kAesBlockBits) {
        throw std::invalid_argument("ckks_aes: MixColumns state must have 128 bits");
    }

    std::array<CKKSByte, kAesBlockBytes> out_bytes;
    for (size_t column = 0; column < 4; ++column) {
        const size_t b0_idx = 4 * column + 0;
        const size_t b1_idx = 4 * column + 1;
        const size_t b2_idx = 4 * column + 2;
        const size_t b3_idx = 4 * column + 3;

        const auto b0 = clone_byte(state, b0_idx);
        const auto b1 = clone_byte(state, b1_idx);
        const auto b2 = clone_byte(state, b2_idx);
        const auto b3 = clone_byte(state, b3_idx);

        out_bytes[b0_idx] = fmix_column_byte(context, b0, b1, b2, b3);
        out_bytes[b1_idx] = fmix_column_byte(context, b1, b2, b3, b0);
        out_bytes[b2_idx] = fmix_column_byte(context, b2, b3, b0, b1);
        out_bytes[b3_idx] = fmix_column_byte(context, b3, b0, b1, b2);
    }

    flatten_bytes_to_state(out_bytes, state);
}

void xboot_state_inplace(
    const CKKSContext& context,
    MyVector<CKKSCiphertext>& state,
    const CKKSAes128CtrOptions& options) {
    if (state.size() != kAesBlockBits) {
        throw std::invalid_argument("ckks_aes: XBOOT state must have 128 bits");
    }
    constexpr size_t pair_count = kAesBlockBits / 2;
    #pragma omp parallel for schedule(dynamic) if (options.parallelize_independent_work)
    for (std::ptrdiff_t pair_idx = 0;
         pair_idx < static_cast<std::ptrdiff_t>(pair_count);
         ++pair_idx) {
        const size_t pair = static_cast<size_t>(pair_idx);
        ckks_batch_bootstrap_evalmod_inplace(
            context,
            state[2 * pair],
            state[2 * pair + 1]);
    }
}

MyVector<CKKSCiphertext> make_batched_initial_counter_state(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& encrypted_round_key_bits,
    std::span<const std::array<uint8_t, 16>> counter_blocks,
    bool parallelize_independent_work) {
    enum class PublicBitPlane : uint8_t {
        AllZero,
        AllOne,
        Varying,
    };

    const auto& params = context.getParams();
    const size_t logical_slots = params.hasBootstrapParams()
        ? params.getBootstrapActiveSlotCount()
        : params.getSlots();
    if (counter_blocks.empty() || counter_blocks.size() > logical_slots) {
        throw std::invalid_argument(
            "ckks_aes: public counter count exceeds logical slot count");
    }

    std::array<PublicBitPlane, kAesBlockBits> plane_kinds;
    for (size_t bit = 0; bit < kAesBlockBits; ++bit) {
        bool saw_zero = counter_blocks.size() < logical_slots;
        bool saw_one = false;
        for (const auto& block : counter_blocks) {
            const bool value =
                ((block[bit / 8] >> (bit % 8)) & uint8_t{1}) != 0;
            saw_zero = saw_zero || !value;
            saw_one = saw_one || value;
        }
        plane_kinds[bit] = !saw_one
            ? PublicBitPlane::AllZero
            : (!saw_zero ? PublicBitPlane::AllOne
                         : PublicBitPlane::Varying);
    }

    std::array<std::optional<CKKSCiphertext>, kAesBlockBits> output_bits;
    #pragma omp parallel for schedule(static) if (parallelize_independent_work)
    for (std::ptrdiff_t bit_idx = 0;
         bit_idx < static_cast<std::ptrdiff_t>(kAesBlockBits);
         ++bit_idx) {
        const size_t bit = static_cast<size_t>(bit_idx);
        auto out = encrypted_round_key_bits[bit].clone();

        if (out.getLevel() == 0) {
            throw std::invalid_argument(
                "ckks_aes: exact public XOR requires one level");
        }
        const size_t input_level = out.getLevel();
        const size_t output_level = input_level - 1;

        if (plane_kinds[bit] == PublicBitPlane::AllZero) {
            ckks_drop_to_level_inplace(out, output_level);
            output_bits[bit] = std::move(out);
            continue;
        }
        if (plane_kinds[bit] == PublicBitPlane::AllOne) {
            ckks_drop_to_level_inplace(out, output_level);
            ckks_mul_inplace(context, out, int64_t{-1});
            ckks_add_integer_constant_inplace(context, out, int64_t{1});
            output_bits[bit] = std::move(out);
            continue;
        }

        const double rescale_prime = static_cast<double>(
            params.getModuli()[input_level]);
        MyVector<Complex> signs(params.getSlots());
        MyVector<Complex> public_bits(params.getSlots());
        for (size_t slot = 0; slot < params.getSlots(); ++slot) {
            const size_t logical_slot = slot % logical_slots;
            const uint8_t value = logical_slot < counter_blocks.size()
                ? static_cast<uint8_t>(
                      (counter_blocks[logical_slot][bit / 8] >> (bit % 8)) & 1U)
                : uint8_t{0};
            public_bits[slot] = Complex(static_cast<double>(value), 0.0);
            signs[slot] = Complex(value == 0 ? 1.0 : -1.0, 0.0);
        }

        // x XOR b = (1 - 2b)x + b. Only a genuinely slot-varying b needs
        // the plaintext multiplication and its rescale.
        auto sign_selector = ckks_encode(
            context,
            signs,
            rescale_prime,
            input_level,
            true);
        ckks_mul_inplace(context, out, sign_selector);
        ckks_rescale(context, out);
        auto encoded_block = ckks_encode(
            context,
            public_bits,
            out.getLevel(),
            true);
        ckks_add_inplace(context, out, encoded_block);
        output_bits[bit] = std::move(out);
    }

    MyVector<CKKSCiphertext> state;
    state.reserve(kAesBlockBits);
    for (size_t bit = 0; bit < kAesBlockBits; ++bit) {
        if (!output_bits[bit].has_value()) {
            throw std::runtime_error(
                "ckks_aes: initial public-XOR output missing");
        }
        state.emplace_back(std::move(*output_bits[bit]));
    }
    return state;
}

}  // namespace

MyVector<CKKSCiphertext> ckks_aes128_encrypt_expanded_round_key_bits(
    const CKKSContext& context,
    const std::array<uint8_t, 176>& expanded_round_keys,
    size_t first_round_key_level) {
    const auto& params = context.getParams();
    const size_t base_round_key_level = params.getBootstrapS2CDepth();
    if (base_round_key_level + first_round_key_level > params.getMaxLevel()) {
        throw std::invalid_argument(
            "ckks_aes: invalid encryption level for first round key bits");
    }

    std::array<std::optional<CKKSCiphertext>, kAesExpandedKeyBits> encrypted_optional;

    #pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t bit_idx = 0;
         bit_idx < static_cast<std::ptrdiff_t>(kAesExpandedKeyBits);
         ++bit_idx) {
        const size_t absolute_bit = static_cast<size_t>(bit_idx);
        const size_t byte = absolute_bit / 8;
        const size_t bit = absolute_bit % 8;
        const size_t round = absolute_bit / kAesBlockBits;
        const size_t level =
            round == 0 ? base_round_key_level + first_round_key_level : base_round_key_level;
        const double value =
            ((expanded_round_keys[byte] >> bit) & uint8_t{1}) != 0U ? 1.0 : 0.0;
        MyVector<Complex> slots(params.getSlots(), Complex(value, 0.0));
        encrypted_optional[absolute_bit] = ckks_encrypt(context, slots, level);
    }

    MyVector<CKKSCiphertext> encrypted_bits;
    encrypted_bits.reserve(kAesExpandedKeyBits);
    for (size_t bit = 0; bit < kAesExpandedKeyBits; ++bit) {
        if (!encrypted_optional[bit].has_value()) {
            throw std::runtime_error("ckks_aes: encrypted round-key bit missing");
        }
        encrypted_bits.emplace_back(std::move(*encrypted_optional[bit]));
    }
    return encrypted_bits;
}

MyVector<CKKSCiphertext> evaluate_aes128_rounds_from_state(
    CKKSContext& context,
    const MyVector<CKKSCiphertext>& encrypted_expanded_round_key_bits,
    MyVector<CKKSCiphertext> state,
    const CKKSAes128CtrOptions& options) {
    if (state.size() != kAesBlockBits) {
        throw std::invalid_argument("ckks_aes: AES state must have 128 bits");
    }
    for (size_t round = 1; round < kAesRoundCount; ++round) {
        sub_bytes_inplace(context, state, options);
        shift_rows_inplace(state);
        fmix_columns_inplace(context, state);
        add_round_key_lazy_inplace(
            context,
            state,
            encrypted_expanded_round_key_bits,
            round);
        xboot_state_inplace(context, state, options);
    }

    sub_bytes_inplace(context, state, options);
    shift_rows_inplace(state);
    add_round_key_lazy_inplace(
        context,
        state,
        encrypted_expanded_round_key_bits,
        kAesRoundCount);
    return state;
}

MyVector<CKKSCiphertext> ckks_aes_ctr(
    CKKSContext& context,
    const MyVector<CKKSCiphertext>& encrypted_expanded_round_key_bits,
    std::span<const std::array<uint8_t, 16>> counter_blocks,
    std::span<const std::array<uint8_t, 16>> ciphertext_blocks,
    const CKKSAes128CtrOptions& options) {
    auto initial_state = make_batched_initial_counter_state(
        context,
        encrypted_expanded_round_key_bits,
        counter_blocks,
        options.parallelize_independent_work);
    auto plaintext_bits = evaluate_aes128_rounds_from_state(
        context,
        encrypted_expanded_round_key_bits,
        std::move(initial_state),
        options);
    for (size_t bit = 0; bit < kAesBlockBits; ++bit) {
        auto encoded_block = encode_message_block(context, ciphertext_blocks, bit, plaintext_bits[bit].getLevel());
        ckks_add_inplace(context, plaintext_bits[bit], encoded_block);
    }
    xboot_state_inplace(context, plaintext_bits, options);
    return plaintext_bits;
}
