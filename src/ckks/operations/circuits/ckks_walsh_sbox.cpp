#include "ckks_walsh_sbox.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "aes/aes_sbox.hpp"
#include "arith/ckks_addition.hpp"
#include "arith/ckks_mul.hpp"
#include "arith/ckks_rescale.hpp"
#include "bootstrap/ckks_bootstrap_evalmod_inplace.hpp"
#include "ckks_encoding.hpp"
#include "keyswitch/ckks_conjugate.hpp"
#include "keyswitch/ckks_keyswitch.hpp"
#include "keyswitch/ckks_relin.hpp"
#include "keyswitch/ckks_rotation.hpp"
#include "modarith.hpp"
#include "ntt.hpp"
#include "polynomial.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr size_t kByteBits = 8;
constexpr size_t kNibbleBits = 4;
constexpr size_t kNibbleValues = 16;
constexpr size_t kSboxValues = 256;
constexpr size_t kPackedWalshSignSegments = 2 * (kNibbleValues - 1);
constexpr size_t kMaxAbsWalshCoeff = 8;

using WalshTable = std::array<std::array<std::array<int16_t, kNibbleValues>, kNibbleValues>, kByteBits>;
using LowSignMultiples =
    std::array<std::array<std::optional<CKKSCiphertext>, kMaxAbsWalshCoeff + 1>, kNibbleValues>;
struct LowSignMultiplesScratch {
    LowSignMultiples multiples;
};

struct WalshConstantEncodingCache {
    std::optional<CKKSEncoding> encoding;
    Complex value = Complex(0.0, 0.0);
};

struct WalshEncodingScratch {
    WalshConstantEncodingCache i;
    WalshConstantEncodingCache half;
};

template <typename Fn>
void run_walsh_independent_job(Fn& fn, std::exception_ptr& first_exception) {
    try {
        fn();
    } catch (...) {
#ifdef _OPENMP
        #pragma omp critical(ckks_walsh_sbox_independent_job_exception)
#endif
        {
            if (first_exception == nullptr) {
                first_exception = std::current_exception();
            }
        }
    }
}

template <typename FirstFn, typename SecondFn>
void run_two_walsh_independent_jobs(
    FirstFn&& first_fn,
    SecondFn&& second_fn) {
#ifdef _OPENMP
    std::exception_ptr first_exception;
    if (omp_in_parallel() != 0 && omp_get_num_threads() > 1) {
        #pragma omp taskgroup
        {
            #pragma omp task shared(first_fn, first_exception)
            {
                run_walsh_independent_job(first_fn, first_exception);
            }
            #pragma omp task shared(second_fn, first_exception)
            {
                run_walsh_independent_job(second_fn, first_exception);
            }
        }
    } else {
        #pragma omp parallel sections num_threads(2) shared(first_fn, second_fn, first_exception)
        {
            #pragma omp section
            {
                run_walsh_independent_job(first_fn, first_exception);
            }
            #pragma omp section
            {
                run_walsh_independent_job(second_fn, first_exception);
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
#else
    first_fn();
    second_fn();
#endif
}

size_t checked_mul(size_t lhs, size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::invalid_argument("ckks_walsh_sbox: size overflow");
    }
    return lhs * rhs;
}

uint8_t parity8(uint8_t value) {
    value ^= static_cast<uint8_t>(value >> 4);
    value ^= static_cast<uint8_t>(value >> 2);
    value ^= static_cast<uint8_t>(value >> 1);
    return static_cast<uint8_t>(value & uint8_t{1});
}

WalshTable build_walsh_table() {
    WalshTable table{};
    for (size_t out_bit = 0; out_bit < kByteBits; ++out_bit) {
        for (size_t low_mask = 0; low_mask < kNibbleValues; ++low_mask) {
            for (size_t high_mask = 0; high_mask < kNibbleValues; ++high_mask) {
                const uint8_t mask =
                    static_cast<uint8_t>(low_mask | (high_mask << kNibbleBits));
                int coeff = 0;
                for (size_t x = 0; x < kSboxValues; ++x) {
                    const uint8_t sbox_value =
                        static_cast<uint8_t>(kAesSboxTable[x]);
                    const uint8_t output_bit =
                        static_cast<uint8_t>((sbox_value >> out_bit) & uint8_t{1});
                    const uint8_t dot =
                        parity8(static_cast<uint8_t>(mask & static_cast<uint8_t>(x)));
                    coeff += (output_bit ^ dot) == 0 ? 1 : -1;
                }
                if ((coeff % 4) != 0) {
                    throw std::runtime_error(
                        "ckks_walsh_sbox: AES Walsh coefficient is not divisible by four");
                }
                table[out_bit][low_mask][high_mask] =
                    static_cast<int16_t>(coeff / 4);
            }
        }
    }
    return table;
}

const WalshTable& walsh_table() {
    static const WalshTable table = build_walsh_table();
    return table;
}

bool scales_close(long double lhs, long double rhs, long double rel_tol = 0x1p-45L) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return false;
    }
    const long double scale =
        std::max<long double>({1.0L, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= rel_tol * scale;
}

const CKKSEncoding& cached_constant_encoding(
    const CKKSContext& context,
    WalshConstantEncodingCache& cache,
    Complex value,
    long double scale,
    size_t level) {
    const double encoded_scale = static_cast<double>(scale);
    const bool value_matches =
        cache.value.real() == value.real() && cache.value.imag() == value.imag();
    const bool metadata_matches =
        cache.encoding.has_value() &&
        cache.encoding->getN() == context.getParams().getN() &&
        cache.encoding->getLevel() == level &&
        cache.encoding->getScale() == static_cast<long double>(encoded_scale);
    if (!value_matches || !metadata_matches) {
        MyVector<Complex> slots(context.getParams().getSlots(), value);
        cache.encoding.emplace(ckks_encode(
            context,
            slots,
            encoded_scale,
            level,
            /*eval=*/true));
        cache.value = value;
    }
    return *cache.encoding;
}

void pad_ciphertext_with_zero_polys(CKKSCiphertext& ct, size_t target_polys) {
    if (target_polys < ct.getNumPolys()) {
        throw std::invalid_argument("ckks_walsh_sbox: cannot shrink ciphertext degree");
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
        throw std::invalid_argument("ckks_walsh_sbox: invalid scalar scale-up");
    }
    const long double scalar_rounded = std::round(scalar);
    if (scalar_rounded < 1.0L ||
        std::abs(scalar - scalar_rounded) >
            std::max<long double>(1.0L, std::abs(scalar)) * 0x1p-20L) {
        throw std::invalid_argument("ckks_walsh_sbox: non-integral scalar scale-up");
    }
    if (scalar_rounded > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("ckks_walsh_sbox: scalar scale-up is too large");
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
    if (scales_close(ct.getScale(), target_scale)) {
        ct.setScale(target_scale);
        return;
    }
    if (ciphertext_is_zero(ct)) {
        ct.setScale(target_scale);
        return;
    }
    if (ct.getScale() > target_scale) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: cannot scale down ciphertext for addition");
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

CKKSCiphertext mul_raw(
    const CKKSContext& context,
    CKKSCiphertext lhs,
    const CKKSCiphertext& rhs,
    long double target_scale) {
    if (lhs.getLevel() == 0 || rhs.getLevel() == 0) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: insufficient level for 4+4 Walsh product");
    }
    const size_t level = std::min(lhs.getLevel(), rhs.getLevel());

    ckks_drop_to_level_inplace(lhs, level);
    lhs.setScale(target_scale);

    if (rhs.getLevel() == level && scales_close(rhs.getScale(), target_scale)) {
        ckks_mul_inplace(context, lhs, rhs);
        return lhs;
    }

    auto rhs_aligned = rhs.clone();
    ckks_drop_to_level_inplace(rhs_aligned, level);
    rhs_aligned.setScale(target_scale);
    ckks_mul_inplace(context, lhs, rhs_aligned);
    return lhs;
}

void relin_inplace(const CKKSContext& context, CKKSCiphertext& ct) {
    if (ct.getNumPolys() != 3) {
        throw std::invalid_argument("ckks_walsh_sbox: expected raw product before relin");
    }
    ckks_relin_hybrid_inplace(context, ct);
}

void validate_input(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& input) {
    if (input.size() != kByteBits) {
        throw std::invalid_argument("ckks_walsh_sbox: input must contain 8 bit ciphertexts");
    }
    const auto& params = context.getParams();
    const auto& ref = input.front();
    if (ref.getN() != params.getN()) {
        throw std::invalid_argument("ckks_walsh_sbox: ciphertext/context size mismatch");
    }
    if (ref.getLevel() != params.getBootstrapS2CDepth()) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: input level must equal bootstrap S2C entry depth");
    }
    if (ref.getNumPolys() != 2 ||
        ref.getSecretOwner() != CKKSSecretOwner::Dense ||
        ref.getMessageEncodingState() != CKKSMessageEncodingState::Slots) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: expected dense slot-encoded 2-polynomial inputs");
    }
    for (const auto& bit : input) {
        if (bit.getN() != ref.getN() ||
            bit.getLevel() != ref.getLevel() ||
            bit.getNumPolys() != ref.getNumPolys() ||
            bit.getSecretOwner() != ref.getSecretOwner() ||
            bit.getMessageEncodingState() != ref.getMessageEncodingState() ||
            !scales_close(bit.getScale(), ref.getScale())) {
            throw std::invalid_argument(
                "ckks_walsh_sbox: inconsistent input ciphertexts");
        }
    }
}

CKKSCiphertext parity_sum(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& input,
    size_t bit_offset,
    size_t mask) {
    std::optional<CKKSCiphertext> out;
    for (size_t bit = 0; bit < kNibbleBits; ++bit) {
        if (((mask >> bit) & size_t{1}) == 0) {
            continue;
        }
        const auto& term = input[bit_offset + bit];
        if (!out.has_value()) {
            out.emplace(term.clone());
        } else {
            ckks_add_inplace(context, *out, term);
        }
    }
    if (!out.has_value()) {
        throw std::invalid_argument("ckks_walsh_sbox: parity mask must be nonzero");
    }
    return std::move(*out);
}

void bit_to_sign_inplace(const CKKSContext& context, CKKSCiphertext& bit) {
    ckks_mul_inplace(context, bit, int64_t{-2});
    ckks_add_integer_constant_inplace(context, bit, int64_t{1});
}


void validate_mask_packed_latency_input(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& input,
    size_t block_count) {
    validate_input(context, input);
    const size_t slots = context.getParams().getSlots();
    if (block_count == 0) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: mask-packed block count must be nonzero");
    }
    if (block_count * kPackedWalshSignSegments > slots) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: mask-packed block count exceeds slot capacity");
    }
    for (size_t segment = 1; segment < kPackedWalshSignSegments; ++segment) {
        const auto offset = static_cast<int64_t>(segment * block_count);
        if (!context.hasRotationKey(offset) || !context.hasRotationKey(-offset)) {
            throw std::invalid_argument(
                "ckks_walsh_sbox: missing mask-packed segment rotation key");
        }
    }
}


size_t packed_walsh_segment(bool high_nibble, size_t mask) {
    if (mask == 0 || mask >= kNibbleValues) {
        throw std::invalid_argument("ckks_walsh_sbox: invalid packed Walsh mask");
    }
    return (high_nibble ? (kNibbleValues - 1) : 0) + (mask - 1);
}

void rotate_segment_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    size_t segment,
    size_t block_count,
    bool inverse) {
    if (segment == 0) {
        return;
    }
    const auto offset = static_cast<int64_t>(segment * block_count);
    ckks_rotate_inplace(context, ct, inverse ? -offset : offset);
}

CKKSCiphertext rotate_from_hoisted_inplace_source(
    const CKKSContext& context,
    const CKKSCiphertext& source,
    const CKKSEvalHoistedCiphertextQP& hoisted,
    int64_t rotation) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t level = source.getLevel();
    const size_t limb_count = level + 1;
    const int64_t slots_i64 = static_cast<int64_t>(params.getSlots());
    int64_t normalized = rotation % slots_i64;
    if (normalized < 0) {
        normalized += slots_i64;
    }
    if (normalized == 0) {
        return source.clone();
    }
    if (source.getN() != N ||
        source.getNumPolys() != 2 ||
        source.getSecretOwner() != CKKSSecretOwner::Dense ||
        level > params.getMaxLevel()) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: invalid hoisted rotation source");
    }
    if (!context.hasRotationKey(normalized)) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: missing hoisted unpack rotation key");
    }

    MyVector<uint64_t> rotated_b = source.getB();
    MyVector<uint64_t> rotated_a(checked_mul(N, limb_count), uint64_t{0});
    ckks_hybrid_key_switch_hoisted_inplace(
        context,
        hoisted,
        context.getRotationKey(normalized),
        rotated_b,
        rotated_a,
        CKKSHybridKeySwitchCombine::Add,
        CKKSHybridKeySwitchCombine::Assign,
        params.montgomery,
        context.getRotationNttMap(normalized));

    return CKKSCiphertext(
        N,
        std::move(rotated_a),
        std::move(rotated_b),
        source.getScale(),
        level,
        source.getSecretOwner(),
        source.getMessageEncodingState());
}

CKKSCiphertext unpack_segment_from_hoist(
    const CKKSContext& context,
    const CKKSCiphertext& packed,
    const CKKSEvalHoistedCiphertextQP& hoisted,
    size_t segment,
    size_t block_count) {
    if (segment == 0) {
        return packed.clone();
    }
    const auto offset = static_cast<int64_t>(segment * block_count);
    return rotate_from_hoisted_inplace_source(context, packed, hoisted, -offset);
}

void add_rotated_segment_term_inplace(
    const CKKSContext& context,
    std::optional<CKKSCiphertext>& packed,
    CKKSCiphertext term,
    size_t segment,
    size_t block_count) {
    rotate_segment_inplace(
        context,
        term,
        segment,
        block_count,
        /*inverse=*/false);
    if (!packed.has_value()) {
        packed.emplace(std::move(term));
    } else {
        ckks_add_inplace(context, *packed, term);
    }
}

CKKSCiphertext build_mask_packed_parity_input(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& input,
    size_t block_count) {
    std::optional<CKKSCiphertext> packed;
    for (size_t mask = 1; mask < kNibbleValues; ++mask) {
        add_rotated_segment_term_inplace(
            context,
            packed,
            parity_sum(context, input, /*bit_offset=*/0, mask),
            packed_walsh_segment(/*high_nibble=*/false, mask),
            block_count);
        add_rotated_segment_term_inplace(
            context,
            packed,
            parity_sum(context, input, /*bit_offset=*/kNibbleBits, mask),
            packed_walsh_segment(/*high_nibble=*/true, mask),
            block_count);
    }

    if (!packed.has_value()) {
        throw std::runtime_error(
            "ckks_walsh_sbox: failed to build packed parity input");
    }
    return std::move(*packed);
}


std::pair<MyVector<CKKSCiphertext>, MyVector<CKKSCiphertext>>
unpack_mask_packed_signs(
    const CKKSContext& context,
    const CKKSCiphertext& packed,
    size_t block_count) {
    MyVector<CKKSCiphertext> low_signs;
    MyVector<CKKSCiphertext> high_signs;
    low_signs.reserve(kNibbleValues);
    high_signs.reserve(kNibbleValues);

    auto low_constant = zero_like(context, packed);
    ckks_add_integer_constant_inplace(context, low_constant, int64_t{1});
    low_signs.emplace_back(std::move(low_constant));

    auto high_constant = zero_like(context, packed);
    ckks_add_integer_constant_inplace(context, high_constant, int64_t{1});
    high_signs.emplace_back(std::move(high_constant));

    CKKSHoistWorkspace hoist_workspace;
    CKKSEvalHoistedCiphertextQP hoisted;
    ckks_hoist_ciphertext_a_qp_eval(context, packed, hoist_workspace, hoisted);

    for (size_t mask = 1; mask < kNibbleValues; ++mask) {
        auto low = unpack_segment_from_hoist(
            context,
            packed,
            hoisted,
            packed_walsh_segment(false, mask),
            block_count);
        low_signs.emplace_back(std::move(low));

        auto high = unpack_segment_from_hoist(
            context,
            packed,
            hoisted,
            packed_walsh_segment(true, mask),
            block_count);
        high_signs.emplace_back(std::move(high));
    }

    return {std::move(low_signs), std::move(high_signs)};
}

CKKSCiphertext zero_ciphertext_like_metadata(const CKKSCiphertext& ref) {
    const size_t N = ref.getN();
    const size_t poly_size = checked_mul(N, ref.getLevel() + 1);
    MyVector<MyVector<uint64_t>> polys;
    polys.reserve(ref.getNumPolys());
    for (size_t poly_idx = 0; poly_idx < ref.getNumPolys(); ++poly_idx) {
        polys.emplace_back(poly_size, uint64_t{0});
    }
    return CKKSCiphertext(
        N,
        std::move(polys),
        ref.getScale(),
        ref.getLevel(),
        ref.getSecretOwner(),
        ref.getMessageEncodingState());
}

CKKSCiphertext ciphertext_like_metadata_uninitialized(const CKKSCiphertext& ref) {
    const size_t N = ref.getN();
    const size_t poly_size = checked_mul(N, ref.getLevel() + 1);
    MyVector<MyVector<uint64_t>> polys;
    polys.reserve(ref.getNumPolys());
    for (size_t poly_idx = 0; poly_idx < ref.getNumPolys(); ++poly_idx) {
        polys.emplace_back(poly_size);
    }
    return CKKSCiphertext(
        N,
        std::move(polys),
        ref.getScale(),
        ref.getLevel(),
        ref.getSecretOwner(),
        ref.getMessageEncodingState());
}

void add_or_sub_ciphertext_same_metadata_inplace(
    const CKKSContext& context,
    CKKSCiphertext& out,
    const CKKSCiphertext& term,
    bool subtract) {
    if (out.getN() != term.getN() ||
        out.getLevel() != term.getLevel() ||
        out.getNumPolys() != term.getNumPolys() ||
        out.getSecretOwner() != term.getSecretOwner() ||
        out.getMessageEncodingState() != term.getMessageEncodingState() ||
        !scales_close(out.getScale(), term.getScale())) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: same-metadata addition mismatch");
    }

    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const size_t N = out.getN();
    const size_t level = out.getLevel();
    const size_t limb_count = level + 1;
    const size_t poly_size = checked_mul(N, limb_count);
    if (level > params.getMaxLevel() ||
        limb_count > moduli.size() ||
        limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: same-metadata addition invalid level");
    }

    for (size_t poly_idx = 0; poly_idx < out.getNumPolys(); ++poly_idx) {
        auto& dst_poly = out.getPolyMutable(poly_idx);
        const auto& src_poly = term.getPoly(poly_idx);
        if (dst_poly.size() != poly_size || src_poly.size() != poly_size) {
            throw std::invalid_argument(
                "ckks_walsh_sbox: same-metadata addition invalid buffer");
        }
        for (size_t limb = 0; limb < limb_count; ++limb) {
            const uint64_t q = moduli[limb];
            uint64_t* __restrict dst = dst_poly.data() + limb * N;
            const uint64_t* __restrict src = src_poly.data() + limb * N;
            if (subtract) {
                poly_sub_modq_inplace(dst, src, N, q);
            } else {
                poly_add_modq_inplace(dst, src, N, q);
            }
        }
    }
}

bool ciphertext_storage_matches(
    const CKKSCiphertext& dst,
    const CKKSCiphertext& src) {
    if (dst.getN() != src.getN() ||
        dst.getLevel() != src.getLevel() ||
        dst.getNumPolys() != src.getNumPolys() ||
        dst.getSecretOwner() != src.getSecretOwner() ||
        dst.getMessageEncodingState() != src.getMessageEncodingState() ||
        !scales_close(dst.getScale(), src.getScale())) {
        return false;
    }
    for (size_t poly_idx = 0; poly_idx < src.getNumPolys(); ++poly_idx) {
        if (dst.getPoly(poly_idx).size() != src.getPoly(poly_idx).size()) {
            return false;
        }
    }
    return true;
}

void copy_ciphertext_polys_inplace(
    CKKSCiphertext& dst,
    const CKKSCiphertext& src) {
    if (!ciphertext_storage_matches(dst, src)) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: reusable ciphertext storage mismatch");
    }
    for (size_t poly_idx = 0; poly_idx < src.getNumPolys(); ++poly_idx) {
        const auto& src_poly = src.getPoly(poly_idx);
        auto& dst_poly = dst.getPolyMutable(poly_idx);
        std::copy(src_poly.begin(), src_poly.end(), dst_poly.begin());
    }
}

CKKSCiphertext& refill_ciphertext_slot(
    std::optional<CKKSCiphertext>& slot,
    const CKKSCiphertext& src) {
    if (!slot.has_value() || !ciphertext_storage_matches(*slot, src)) {
        slot.emplace(src.clone());
    } else {
        copy_ciphertext_polys_inplace(*slot, src);
    }
    return *slot;
}

CKKSCiphertext& refill_ciphertext_sum_slot(
    const CKKSContext& context,
    std::optional<CKKSCiphertext>& slot,
    const CKKSCiphertext& lhs,
    const CKKSCiphertext& rhs) {
    if (!ciphertext_storage_matches(lhs, rhs)) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: reusable ciphertext sum metadata mismatch");
    }
    if (!slot.has_value() || !ciphertext_storage_matches(*slot, lhs)) {
        slot.emplace(ciphertext_like_metadata_uninitialized(lhs));
    }

    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const size_t N = lhs.getN();
    const size_t level = lhs.getLevel();
    const size_t limb_count = level + 1;
    const size_t poly_size = checked_mul(N, limb_count);
    if (level > params.getMaxLevel() ||
        limb_count > moduli.size() ||
        limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: reusable ciphertext sum invalid level");
    }

    auto& out = *slot;
    for (size_t poly_idx = 0; poly_idx < lhs.getNumPolys(); ++poly_idx) {
        const auto& lhs_poly = lhs.getPoly(poly_idx);
        const auto& rhs_poly = rhs.getPoly(poly_idx);
        auto& out_poly = out.getPolyMutable(poly_idx);
        if (lhs_poly.size() != poly_size ||
            rhs_poly.size() != poly_size ||
            out_poly.size() != poly_size) {
            throw std::invalid_argument(
                "ckks_walsh_sbox: reusable ciphertext sum invalid buffer");
        }
        for (size_t limb = 0; limb < limb_count; ++limb) {
            const uint64_t q = moduli[limb];
            poly_add_modq(
                lhs_poly.data() + limb * N,
                rhs_poly.data() + limb * N,
                out_poly.data() + limb * N,
                N,
                q);
        }
    }
    return out;
}

const LowSignMultiples& build_low_sign_multiples(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& low_signs,
    LowSignMultiplesScratch& scratch) {
    auto& multiples = scratch.multiples;
    for (size_t mask = 1; mask < kNibbleValues; ++mask) {
        refill_ciphertext_slot(multiples[mask][1], low_signs[mask]);
        for (size_t abs_coeff = 2; abs_coeff <= kMaxAbsWalshCoeff; ++abs_coeff) {
            refill_ciphertext_sum_slot(
                context,
                multiples[mask][abs_coeff],
                *multiples[mask][abs_coeff - 1],
                low_signs[mask]);
        }
    }
    return multiples;
}

void add_scaled_poly_inplace(
    uint64_t* __restrict dst,
    const uint64_t* __restrict src,
    size_t N,
    uint64_t scalar_mod_q,
    uint64_t q,
    const Negacyclic_NTT_Twiddles& table,
    bool use_montgomery) {
    if (scalar_mod_q == 0) {
        return;
    }
    if (scalar_mod_q == 1) {
        poly_add_modq_inplace(dst, src, N, q);
        return;
    }
    if (scalar_mod_q + 1 == q) {
        poly_sub_modq_inplace(dst, src, N, q);
        return;
    }

    thread_local MyVector<uint64_t> scaled;
    scaled.resize(N);
    std::copy(src, src + N, scaled.data());
    pointwise_multiply_scalar_inplace_dispatch(
        scaled.data(),
        scalar_mod_q,
        N,
        q,
        table,
        use_montgomery);
    poly_add_modq_inplace(dst, scaled.data(), N, q);
}

void add_scaled_ciphertext_inplace(
    const CKKSContext& context,
    CKKSCiphertext& out,
    const CKKSCiphertext& term,
    int16_t coeff) {
    if (coeff == 0) {
        return;
    }
    if (out.getN() != term.getN() ||
        out.getLevel() != term.getLevel() ||
        out.getNumPolys() != term.getNumPolys() ||
        out.getSecretOwner() != term.getSecretOwner() ||
        out.getMessageEncodingState() != term.getMessageEncodingState() ||
        !scales_close(out.getScale(), term.getScale())) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: fused linear combination metadata mismatch");
    }

    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const size_t N = out.getN();
    const size_t level = out.getLevel();
    const size_t limb_count = level + 1;
    const size_t poly_size = checked_mul(N, limb_count);
    if (level > params.getMaxLevel() ||
        limb_count > moduli.size() ||
        limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: fused linear combination invalid level");
    }

    for (size_t limb = 0; limb < limb_count; ++limb) {
        const uint64_t q = moduli[limb];
        const uint64_t scalar_mod_q = reduce_i64_mod_q(coeff, q);
        const auto& table = twiddle_ntt[limb];
        for (size_t poly_idx = 0; poly_idx < out.getNumPolys(); ++poly_idx) {
            auto& dst_poly = out.getPolyMutable(poly_idx);
            const auto& src_poly = term.getPoly(poly_idx);
            if (dst_poly.size() != poly_size || src_poly.size() != poly_size) {
                throw std::invalid_argument(
                    "ckks_walsh_sbox: fused linear combination invalid buffer");
            }
            add_scaled_poly_inplace(
                dst_poly.data() + limb * N,
                src_poly.data() + limb * N,
                N,
                scalar_mod_q,
                q,
                table,
                params.montgomery);
        }
    }
}

CKKSCiphertext linear_combination_with_constant(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& signs,
    const std::array<int16_t, kNibbleValues>& coeffs,
    const LowSignMultiples* multiples = nullptr) {
    auto out = zero_ciphertext_like_metadata(signs.front());
    if (coeffs[0] != 0) {
        ckks_add_integer_constant_inplace(context, out, coeffs[0]);
    }
    for (size_t mask = 1; mask < kNibbleValues; ++mask) {
        const int16_t coeff = coeffs[mask];
        if (coeff == 0) {
            continue;
        }
        if (multiples != nullptr) {
            const int abs_coeff = coeff < 0 ? -coeff : coeff;
            if (abs_coeff <= 0 ||
                static_cast<size_t>(abs_coeff) > kMaxAbsWalshCoeff ||
                !(*multiples)[mask][static_cast<size_t>(abs_coeff)].has_value()) {
                throw std::runtime_error(
                    "ckks_walsh_sbox: missing low-sign multiple");
            }
            add_or_sub_ciphertext_same_metadata_inplace(
                context,
                out,
                *(*multiples)[mask][static_cast<size_t>(abs_coeff)],
                coeff < 0);
        } else {
            add_scaled_ciphertext_inplace(context, out, signs[mask], coeff);
        }
    }
    return out;
}

CKKSCiphertext multiply_by_i(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    double sign) {
    auto out = ckks_mul(
        context,
        ct,
        Complex(0.0, sign),
        1.0);
    out.setScale(ct.getScale());
    return out;
}

void multiply_by_cached_i_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    const CKKSEncoding* encoded_i) {
    const long double scale = ct.getScale();
    if (encoded_i == nullptr) {
        ckks_mul_inplace(context, ct, Complex(0.0, 1.0), 1.0);
    } else {
        ckks_mul_inplace(context, ct, *encoded_i);
    }
    ct.setScale(scale);
}

CKKSCiphertext combine_real_imag(
    const CKKSContext& context,
    CKKSCiphertext real,
    CKKSCiphertext imag,
    const CKKSEncoding* encoded_i = nullptr) {
    multiply_by_cached_i_inplace(context, imag, encoded_i);
    if (ciphertext_storage_matches(real, imag)) {
        add_or_sub_ciphertext_same_metadata_inplace(
            context,
            real,
            imag,
            /*subtract=*/false);
    } else {
        add_term_inplace(context, real, std::move(imag));
    }
    return real;
}

CKKSCiphertext linear_combination_pair_with_constant(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& signs,
    const std::array<int16_t, kNibbleValues>& real_coeffs,
    const std::array<int16_t, kNibbleValues>& imag_coeffs,
    const CKKSEncoding* encoded_i = nullptr,
    const LowSignMultiples* multiples = nullptr) {
    auto real = linear_combination_with_constant(
        context,
        signs,
        real_coeffs,
        multiples);
    auto imag = linear_combination_with_constant(
        context,
        signs,
        imag_coeffs,
        multiples);
    return combine_real_imag(
        context,
        std::move(real),
        std::move(imag),
        encoded_i);
}

CKKSCiphertext walsh_sum_for_output_pair(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& low_signs,
    const MyVector<CKKSCiphertext>& high_signs,
    size_t real_out_bit,
    size_t imag_out_bit,
    const CKKSEncoding* encoded_i = nullptr,
    const LowSignMultiples* multiples = nullptr) {
    const auto& table = walsh_table();
    const long double target_scale = low_signs.front().getScale();

    std::array<int16_t, kNibbleValues> real_low_only_coeffs{};
    std::array<int16_t, kNibbleValues> imag_low_only_coeffs{};
    for (size_t low_mask = 0; low_mask < kNibbleValues; ++low_mask) {
        real_low_only_coeffs[low_mask] = table[real_out_bit][low_mask][0];
        imag_low_only_coeffs[low_mask] = table[imag_out_bit][low_mask][0];
    }
    auto acc = linear_combination_pair_with_constant(
        context,
        low_signs,
        real_low_only_coeffs,
        imag_low_only_coeffs,
        encoded_i,
        multiples);

    std::optional<CKKSCiphertext> raw_products;
    for (size_t high_mask = 1; high_mask < kNibbleValues; ++high_mask) {
        std::array<int16_t, kNibbleValues> real_low_coeffs{};
        std::array<int16_t, kNibbleValues> imag_low_coeffs{};
        for (size_t low_mask = 0; low_mask < kNibbleValues; ++low_mask) {
            real_low_coeffs[low_mask] =
                table[real_out_bit][low_mask][high_mask];
            imag_low_coeffs[low_mask] =
                table[imag_out_bit][low_mask][high_mask];
        }

        auto low_sum = linear_combination_pair_with_constant(
            context,
            low_signs,
            real_low_coeffs,
            imag_low_coeffs,
            encoded_i,
            multiples);
        auto product = mul_raw(
            context,
            std::move(low_sum),
            high_signs[high_mask],
            target_scale);
        if (!raw_products.has_value()) {
            raw_products.emplace(std::move(product));
        } else {
            add_term_inplace(context, *raw_products, std::move(product));
        }
    }

    if (raw_products.has_value()) {
        relin_inplace(context, *raw_products);
        add_term_inplace(context, acc, std::move(*raw_products));
    }
    return acc;
}

CKKSCiphertext walsh_sum_to_bit_pair(
    const CKKSContext& context,
    CKKSCiphertext walsh_sum,
    WalshConstantEncodingCache* half_cache = nullptr) {
    // The table stores D = W / 4, so z = (1/2 + i/2) - D/128.
    // W is kept at level 1 with scale S^2 during accumulation so the
    // unnormalized Walsh sum never has to fit at q0.
    if (walsh_sum.getLevel() == 0) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: Walsh sum must have one level for final normalization");
    }
    ckks_rescale(context, walsh_sum);
    const long double output_scale = walsh_sum.getScale() * 128.0L;

    walsh_sum.setScale(output_scale);
    ckks_mul_inplace(context, walsh_sum, int64_t{-1});

    if (half_cache != nullptr) {
        const auto& half = cached_constant_encoding(
            context,
            *half_cache,
            Complex(0.5, 0.5),
            output_scale,
            walsh_sum.getLevel());
        ckks_add_inplace(context, walsh_sum, half);
    } else {
        MyVector<Complex> half_slots(
            context.getParams().getSlots(),
            Complex(0.5, 0.5));
        auto half = ckks_encode(
            context,
            half_slots,
            static_cast<double>(output_scale),
            walsh_sum.getLevel(),
            /*eval=*/true);
        ckks_add_inplace(context, walsh_sum, half);
    }
    return walsh_sum;
}

std::pair<CKKSCiphertext, CKKSCiphertext> split_real_imag_pair(
    const CKKSContext& context,
    const CKKSCiphertext& pair) {
    auto conjugated = ckks_conjugate(context, pair);

    auto real = pair.clone();
    ckks_add_inplace(context, real, conjugated);
    real.setScale(pair.getScale() * 2.0L);

    auto imag = pair.clone();
    ckks_sub_inplace(context, imag, conjugated);
    imag = multiply_by_i(context, imag, -1.0);
    imag.setScale(pair.getScale() * 2.0L);

    return {std::move(real), std::move(imag)};
}

MyVector<CKKSCiphertext> recombine_mask_packed_signs_to_paired_outputs(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& low_signs,
    const MyVector<CKKSCiphertext>& high_signs,
    const char* label,
    LowSignMultiplesScratch* low_multiples_scratch = nullptr,
    WalshEncodingScratch* encoding_scratch = nullptr) {
    if (low_signs.size() != kNibbleValues ||
        high_signs.size() != kNibbleValues ||
        low_signs[1].getLevel() != high_signs[1].getLevel() ||
        !scales_close(low_signs[1].getScale(), high_signs[1].getScale())) {
        throw std::runtime_error(
            std::string("ckks_walsh_sbox: ") + label +
            " low/high sign metadata mismatch");
    }
    if (low_signs[1].getLevel() == 0) {
        throw std::runtime_error(
            std::string("ckks_walsh_sbox: ") + label +
            " sign output level is too low for 4+4 products");
    }

    WalshEncodingScratch local_encoding_scratch;
    if (encoding_scratch == nullptr) {
        encoding_scratch = &local_encoding_scratch;
    }
    const auto& encoded_i = cached_constant_encoding(
        context,
        encoding_scratch->i,
        Complex(0.0, 1.0),
        1.0,
        low_signs[1].getLevel());

    LowSignMultiplesScratch local_low_multiples_scratch;
    if (low_multiples_scratch == nullptr) {
        low_multiples_scratch = &local_low_multiples_scratch;
    }
    const auto& low_multiples =
        build_low_sign_multiples(context, low_signs, *low_multiples_scratch);

    MyVector<CKKSCiphertext> outputs;
    outputs.reserve(kByteBits / 2);
    for (size_t pair_idx = 0; pair_idx < kByteBits / 2; ++pair_idx) {
        auto walsh_sum = walsh_sum_for_output_pair(
            context,
            low_signs,
            high_signs,
            pair_idx,
            pair_idx + kByteBits / 2,
            &encoded_i,
            &low_multiples);
        outputs.emplace_back(walsh_sum_to_bit_pair(
            context,
            std::move(walsh_sum),
            &encoding_scratch->half));
    }
    return outputs;
}

MyVector<CKKSCiphertext> combine_paired_lane_outputs_as_complex_bits(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& real_lane_paired,
    const MyVector<CKKSCiphertext>& imag_lane_paired) {
    if (real_lane_paired.size() != kByteBits / 2 ||
        imag_lane_paired.size() != kByteBits / 2) {
        throw std::invalid_argument(
            "ckks_walsh_sbox: complex-block output recombine expects paired lane outputs");
    }

    std::array<std::optional<CKKSCiphertext>, kByteBits> bit_outputs;
    for (size_t pair = 0; pair < kByteBits / 2; ++pair) {
        auto [real_low, real_high] =
            split_real_imag_pair(context, real_lane_paired[pair]);
        auto [imag_low, imag_high] =
            split_real_imag_pair(context, imag_lane_paired[pair]);
        bit_outputs[pair] = combine_real_imag(
            context,
            std::move(real_low),
            std::move(imag_low));
        bit_outputs[pair + kByteBits / 2] = combine_real_imag(
            context,
            std::move(real_high),
            std::move(imag_high));
    }

    MyVector<CKKSCiphertext> out;
    out.reserve(kByteBits);
    for (size_t bit = 0; bit < kByteBits; ++bit) {
        if (!bit_outputs[bit].has_value()) {
            throw std::runtime_error(
                "ckks_walsh_sbox: missing complex-block output bit");
        }
        out.emplace_back(std::move(*bit_outputs[bit]));
    }
    return out;
}

using WalshPackedParityLanes =
    std::pair<CKKSCiphertext, CKKSCiphertext>;
using WalshUnpackedNibbleSigns =
    std::pair<MyVector<CKKSCiphertext>, MyVector<CKKSCiphertext>>;

struct WalshUnpackedSignLanes {
    WalshUnpackedNibbleSigns lane0;
    WalshUnpackedNibbleSigns lane1;
};

struct WalshPairedOutputLanes {
    MyVector<CKKSCiphertext> lane0;
    MyVector<CKKSCiphertext> lane1;
};

WalshPackedParityLanes bootstrap_complex_packed_parity_lanes(
    CKKSContext& context,
    CKKSCiphertext packed) {
    return ckks_batch_bootstrap_evalmod_from_complex_packed(
        context,
        std::move(packed));
}


void convert_packed_parities_to_signs_inplace(
    const CKKSContext& context,
    WalshPackedParityLanes& lanes) {
    bit_to_sign_inplace(context, lanes.first);
    bit_to_sign_inplace(context, lanes.second);
}

WalshUnpackedSignLanes unpack_packed_sign_lanes(
    const CKKSContext& context,
    const WalshPackedParityLanes& lanes,
    size_t block_count) {
    WalshUnpackedSignLanes unpacked;
    run_two_walsh_independent_jobs(
        [&] {
            unpacked.lane0 =
                unpack_mask_packed_signs(context, lanes.first, block_count);
        },
        [&] {
            unpacked.lane1 =
                unpack_mask_packed_signs(context, lanes.second, block_count);
        });
    return unpacked;
}

WalshPairedOutputLanes recombine_unpacked_sign_lanes(
    const CKKSContext& context,
    const MyVector<CKKSCiphertext>& lane0_low_signs,
    const MyVector<CKKSCiphertext>& lane0_high_signs,
    const MyVector<CKKSCiphertext>& lane1_low_signs,
    const MyVector<CKKSCiphertext>& lane1_high_signs,
    const char* lane0_label,
    const char* lane1_label,
    std::array<LowSignMultiplesScratch, 2>& low_multiples,
    std::array<WalshEncodingScratch, 2>& encodings) {
    WalshPairedOutputLanes outputs;
    run_two_walsh_independent_jobs(
        [&] {
            outputs.lane0 = recombine_mask_packed_signs_to_paired_outputs(
                context,
                lane0_low_signs,
                lane0_high_signs,
                lane0_label,
                &low_multiples[0],
                &encodings[0]);
        },
        [&] {
            outputs.lane1 = recombine_mask_packed_signs_to_paired_outputs(
                context,
                lane1_low_signs,
                lane1_high_signs,
                lane1_label,
                &low_multiples[1],
                &encodings[1]);
        });
    return outputs;
}

}  // namespace

struct CKKSWalshSboxScratch::Impl {
    std::array<LowSignMultiplesScratch, 2> low_multiples;
    std::array<WalshEncodingScratch, 2> encodings;
};

CKKSWalshSboxScratch::CKKSWalshSboxScratch()
    : impl_(std::make_unique<Impl>()) {}

CKKSWalshSboxScratch::~CKKSWalshSboxScratch() = default;

MyVector<CKKSCiphertext> ckks_walsh_sbox_complex_blocks_mask_packed(
    CKKSContext& context,
    MyVector<CKKSCiphertext> input,
    size_t block_count,
    CKKSWalshSboxScratch& scratch) {
    if (scratch.impl_ == nullptr) {
        scratch.impl_ = std::make_unique<CKKSWalshSboxScratch::Impl>();
    }
    validate_mask_packed_latency_input(context, input, block_count);
    auto packed = build_mask_packed_parity_input(
        context,
        input,
        block_count);

    auto packed_lanes =
        bootstrap_complex_packed_parity_lanes(context, std::move(packed));
    convert_packed_parities_to_signs_inplace(context, packed_lanes);

    auto unpacked =
        unpack_packed_sign_lanes(context, packed_lanes, block_count);
    auto paired_lanes = recombine_unpacked_sign_lanes(
        context,
        unpacked.lane0.first,
        unpacked.lane0.second,
        unpacked.lane1.first,
        unpacked.lane1.second,
        "complex-block mask-packed real lane",
        "complex-block mask-packed imag lane",
        scratch.impl_->low_multiples,
        scratch.impl_->encodings);

    auto out = combine_paired_lane_outputs_as_complex_bits(
        context,
        paired_lanes.lane0,
        paired_lanes.lane1);

    return out;
}
