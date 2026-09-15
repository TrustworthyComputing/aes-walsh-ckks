#include "ckks_bootstrap_dft.hpp"

#include <algorithm>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>

#include "ckks_context.hpp"
#include "ckks_encoding.hpp"
#include "fft.hpp"
#include "modarith.hpp"
#include "ntt.hpp"

namespace {
using Complex = std::complex<double>;

inline uint64_t double_to_uint64_local(int64_t val, uint64_t q) {
    const int64_t q_int = static_cast<int64_t>(q);
    const int64_t r = val % q_int;
    return static_cast<uint64_t>(r + (q_int & (r >> 63)));
}

void make_hermitian_evals_rotated(
    std::span<const Complex> src,
    MyVector<Complex>& dest,
    const MyVector<size_t>& perm_table,
    size_t N,
    size_t rotate_right) {
    const size_t slots = N >> 1;
    if (src.size() != slots || dest.size() != N) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid rotated Hermitian embedding shape");
    }

    std::fill(dest.begin(), dest.end(), Complex(0.0, 0.0));
    const size_t mask = slots - 1;
    rotate_right &= mask;
    for (size_t i = 0; i < slots; ++i) {
        const Complex value = src[(i + slots - rotate_right) & mask];
        const size_t dest_idx = perm_table[i];
        dest[dest_idx] = value;
        dest[N - 1 - dest_idx] = std::conj(value);
    }
}

MyVector<uint64_t> encode_slots_custom_basis_rotated(
    size_t N,
    std::span<const Complex> values,
    size_t rotate_right,
    double scale,
    const MyVector<uint64_t>& moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt,
    const MyVector<Complex>& twiddle_fft,
    const MyVector<size_t>& perm_table,
    bool eval = true,
    bool montgomery_form = false) {
    const size_t slots = N >> 1;
    if (values.size() != slots || slots == 0 || !std::has_single_bit(slots)) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid rotated encoder slot count");
    }
    if (!(scale > 0.0)) {
        throw std::invalid_argument("ckks_bootstrap_dft: rotated encoder scale must be positive");
    }
    if (moduli.empty() || moduli.size() > twiddle_ntt.size() || twiddle_fft.size() != N) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid rotated encoder basis");
    }

    MyVector<Complex> evals(N);
    make_hermitian_evals_rotated(values, evals, perm_table, N, rotate_right);
    fft_dif(evals, twiddle_fft, N);

    MyVector<uint64_t> plaintext(moduli.size() * N);
    const double scaled_inv_N = scale / static_cast<double>(N);
    for (size_t q_index = 0; q_index < moduli.size(); ++q_index) {
        const uint64_t q = moduli[q_index];
        uint64_t* __restrict limb_ptr = plaintext.data() + q_index * N;
        for (size_t i = 0; i < N; ++i) {
            const int64_t val = std::llrint(evals[i].real() * scaled_inv_N);
            limb_ptr[i] = double_to_uint64_local(val, q);
        }
    }

    if (eval) {
        ntt_forward_rns_flat_inplace(plaintext.data(), N, moduli.size(), moduli, twiddle_ntt, false);
    }
    if (montgomery_form) {
        ckks_convert_plaintext_to_montgomery_inplace(plaintext, N, moduli, twiddle_ntt);
    }
    return plaintext;
}

MyVector<Negacyclic_NTT_Twiddles> make_qp_twiddles(
    const CKKSContext& context,
    size_t q_limb_count) {
    const auto& q_twiddles = context.getTwiddleNtt();
    const auto& p_twiddles = context.getPTwiddleNtt();
    if (q_limb_count > q_twiddles.size()) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid QP twiddle Q limb count");
    }

    MyVector<Negacyclic_NTT_Twiddles> out;
    out.reserve(q_limb_count + p_twiddles.size());
    for (size_t i = 0; i < q_limb_count; ++i) {
        out.emplace_back(q_twiddles[i]);
    }
    for (const auto& twiddle : p_twiddles) {
        out.emplace_back(twiddle);
    }
    return out;
}

MyVector<uint64_t> make_qp_moduli(
    const CKKSContext& context,
    size_t q_limb_count) {
    const auto& params = context.getParams();
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    if (q_limb_count > q_moduli.size()) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid QP modulus Q limb count");
    }

    MyVector<uint64_t> out;
    out.reserve(q_limb_count + p_moduli.size());
    out.insert(out.end(), q_moduli.begin(), q_moduli.begin() + q_limb_count);
    out.insert(out.end(), p_moduli.begin(), p_moduli.end());
    return out;
}

size_t dft_literal_active_slots(
    const CKKSContext& context,
    const CKKSBootstrapDftMatrixLiteral& literal) {
    const size_t full_slots = context.getParams().getSlots();
    const size_t active_slots =
        literal.active_slots == 0 ? full_slots : literal.active_slots;
    if (active_slots == 0 ||
        active_slots > full_slots ||
        full_slots % active_slots != 0 ||
        !std::has_single_bit(active_slots)) {
        throw std::invalid_argument(
            "ckks_bootstrap_dft: invalid active DFT slot count");
    }
    return active_slots;
}

MyVector<Complex> expand_periodic_diagonal_to_physical_slots(
    std::span<const Complex> values,
    size_t active_slots,
    size_t full_slots) {
    if (values.size() == full_slots) {
        return MyVector<Complex>(values.begin(), values.end());
    }
    if (values.size() != active_slots || full_slots % active_slots != 0) {
        throw std::invalid_argument(
            "ckks_bootstrap_dft: invalid sparse diagonal value count");
    }

    MyVector<Complex> expanded(full_slots);
    for (size_t slot = 0; slot < full_slots; ++slot) {
        expanded[slot] = values[slot % active_slots];
    }
    return expanded;
}

struct DftFactorScheduleEntry {
    size_t group_index = 0;
    size_t log_plaintext_scale = 0;
    bool ends_group = true;
};

size_t dft_factor_count(const CKKSBootstrapDftMatrixLiteral& literal) {
    if (literal.log_plaintext_scale_groups.empty()) {
        return literal.stage_count;
    }
    size_t count = 0;
    for (const auto& group : literal.log_plaintext_scale_groups) {
        count += group.size();
    }
    return count;
}

size_t dft_rescale_group_count(const CKKSBootstrapDftMatrixLiteral& literal) {
    return literal.log_plaintext_scale_groups.empty()
        ? literal.stage_count
        : literal.log_plaintext_scale_groups.size();
}

size_t dft_group_log_plaintext_scale_sum(
    const CKKSBootstrapDftScaleSchedule& schedule,
    size_t group_index) {
    if (group_index >= schedule.size() || schedule[group_index].empty()) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid DFT scale schedule group");
    }
    size_t sum = 0;
    for (size_t value : schedule[group_index]) {
        if (value == 0) {
            throw std::invalid_argument("ckks_bootstrap_dft: DFT scale schedule entries must be positive");
        }
        sum += value;
    }
    return sum;
}

size_t dft_group_log_plaintext_scale_total(
    const CKKSBootstrapDftMatrixLiteral& literal,
    size_t group_index) {
    return dft_group_log_plaintext_scale_sum(
        literal.log_plaintext_scale_groups,
        group_index);
}

MyVector<DftFactorScheduleEntry> flatten_dft_factor_schedule(
    const CKKSBootstrapDftMatrixLiteral& literal) {
    MyVector<DftFactorScheduleEntry> entries;
    if (literal.log_plaintext_scale_groups.empty()) {
        entries.reserve(literal.stage_count);
        for (size_t i = 0; i < literal.stage_count; ++i) {
            entries.push_back({i, 0, true});
        }
        return entries;
    }

    for (size_t group = 0; group < literal.log_plaintext_scale_groups.size(); ++group) {
        const auto& factors = literal.log_plaintext_scale_groups[group];
        if (factors.empty()) {
            throw std::invalid_argument("ckks_bootstrap_dft: DFT scale schedule groups must not be empty");
        }
        for (size_t factor = 0; factor < factors.size(); ++factor) {
            entries.push_back({group, factors[factor], factor + 1 == factors.size()});
        }
    }
    return entries;
}

double scheduled_plaintext_scale(
    const CKKSContext& context,
    const CKKSBootstrapDftMatrixLiteral& literal,
    const DftFactorScheduleEntry& entry,
    size_t stage_level,
    double max_abs_value) {
    if (!(max_abs_value > 0.0)) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid zero DFT diagonal");
    }
    const double q_drop = static_cast<double>(context.getParams().getModuli()[stage_level]);
    if (literal.log_plaintext_scale_groups.empty()) {
        return q_drop / (2.01 * max_abs_value);
    }

    const size_t group_sum =
        dft_group_log_plaintext_scale_total(literal, entry.group_index);
    return std::pow(q_drop, static_cast<double>(entry.log_plaintext_scale) /
                              static_cast<double>(group_sum));
}

using SparseDftDiagonalMap = std::map<size_t, MyVector<Complex>>;

void apply_input_selector_to_diagonal_map(
    SparseDftDiagonalMap& diagonal_map,
    size_t slots,
    size_t input_selector_start,
    size_t input_selector_size) {
    if (input_selector_size == 0) {
        return;
    }
    if (input_selector_start >= slots || input_selector_size > slots) {
        throw std::invalid_argument(
            "ckks_bootstrap_dft: S2C input selector exceeds slot count");
    }
    for (auto& [diagonal_index, values] : diagonal_map) {
        if (diagonal_index >= slots || values.size() != slots) {
            throw std::invalid_argument(
                "ckks_bootstrap_dft: invalid S2C selector diagonal");
        }
        for (size_t row = 0; row < slots; ++row) {
            const size_t input_column = (row + diagonal_index) % slots;
            const size_t relative =
                (input_column + slots - input_selector_start) % slots;
            if (relative >= input_selector_size) {
                values[row] = Complex(0.0, 0.0);
            }
        }
    }
}

struct FactorizedDftPlainVectors {
    MyVector<MyVector<Complex>> a;
    MyVector<MyVector<Complex>> b;
    MyVector<MyVector<Complex>> c;
};

struct FactorizedDftStageSource {
    size_t log_slots = 0;
    size_t slots = 0;
    CKKSBootstrapDftType type = CKKSBootstrapDftType::SlotsToCoeffs;
    bool bitreversed = false;
    MyVector<size_t> merge;
    FactorizedDftPlainVectors plain_vectors;
};

MyVector<Complex> factorized_dft_roots(size_t root_count) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    MyVector<Complex> roots(root_count + 1);
    for (size_t i = 0; i <= root_count; ++i) {
        const double angle = 2.0 * pi * static_cast<double>(i) / static_cast<double>(root_count);
        roots[i] = Complex(std::cos(angle), std::sin(angle));
    }
    return roots;
}

MyVector<size_t> dft_pow5_table(size_t slots) {
    const size_t modulus_mask = (slots << 2) - 1;
    MyVector<size_t> pow5((slots << 1) + 1);
    pow5[0] = 1;
    for (size_t i = 1; i < pow5.size(); ++i) {
        pow5[i] = (pow5[i - 1] * 5) & modulus_mask;
    }
    return pow5;
}

size_t bit_reverse_value(size_t value, size_t width) {
    size_t reversed = 0;
    for (size_t i = 0; i < width; ++i) {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    return reversed;
}

void bit_reverse_dft_plain_vector(MyVector<Complex>& values, size_t log_slots) {
    const size_t slots = size_t{1} << log_slots;
    if (values.size() != slots && values.size() != 2 * slots) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid bit-reversed DFT vector size");
    }

    const size_t block_count = values.size() / slots;
    for (size_t block = 0; block < block_count; ++block) {
        const size_t offset = block * slots;
        for (size_t i = 0; i < slots; ++i) {
            const size_t j = bit_reverse_value(i, log_slots);
            if (i < j) {
                std::swap(values[offset + i], values[offset + j]);
            }
        }
    }
}

void bit_reverse_dft_plain_vectors(
    FactorizedDftPlainVectors& vectors,
    size_t log_slots) {
    for (size_t i = 0; i < vectors.a.size(); ++i) {
        bit_reverse_dft_plain_vector(vectors.a[i], log_slots);
        bit_reverse_dft_plain_vector(vectors.b[i], log_slots);
        bit_reverse_dft_plain_vector(vectors.c[i], log_slots);
    }
}

FactorizedDftPlainVectors fft_plain_vectors(
    size_t log_slots,
    size_t dslots,
    const MyVector<Complex>& roots,
    const MyVector<size_t>& pow5) {
    const size_t slots = size_t{1} << log_slots;
    const size_t copy_count = (2 * slots == dslots) ? 2 : 1;

    FactorizedDftPlainVectors vectors;
    vectors.a.resize(log_slots);
    vectors.b.resize(log_slots);
    vectors.c.resize(log_slots);

    size_t index = 0;
    for (size_t m = 2; m <= slots; m <<= 1) {
        MyVector<Complex> a(dslots, Complex(0.0, 0.0));
        MyVector<Complex> b(dslots, Complex(0.0, 0.0));
        MyVector<Complex> c(dslots, Complex(0.0, 0.0));

        const size_t half_m = m >> 1;
        for (size_t i = 0; i < slots; i += m) {
            const size_t gap = slots / m;
            const size_t mask = (m << 2) - 1;
            for (size_t j = 0; j < half_m; ++j) {
                const size_t root_index = (pow5[j] & mask) * gap;
                const size_t idx1 = i + j;
                const size_t idx2 = idx1 + half_m;

                for (size_t u = 0; u < copy_count; ++u) {
                    const size_t offset = u * slots;
                    a[idx1 + offset] = Complex(1.0, 0.0);
                    a[idx2 + offset] = -roots[root_index];
                    b[idx1 + offset] = roots[root_index];
                    c[idx2 + offset] = Complex(1.0, 0.0);
                }
            }
        }

        vectors.a[index] = std::move(a);
        vectors.b[index] = std::move(b);
        vectors.c[index] = std::move(c);
        ++index;
    }

    return vectors;
}

FactorizedDftPlainVectors ifft_plain_vectors(
    size_t log_slots,
    size_t dslots,
    const MyVector<Complex>& roots,
    const MyVector<size_t>& pow5) {
    const size_t slots = size_t{1} << log_slots;
    const size_t copy_count = (2 * slots == dslots) ? 2 : 1;

    FactorizedDftPlainVectors vectors;
    vectors.a.resize(log_slots);
    vectors.b.resize(log_slots);
    vectors.c.resize(log_slots);

    size_t index = 0;
    for (size_t m = slots; m >= 2; m >>= 1) {
        MyVector<Complex> a(dslots, Complex(0.0, 0.0));
        MyVector<Complex> b(dslots, Complex(0.0, 0.0));
        MyVector<Complex> c(dslots, Complex(0.0, 0.0));

        const size_t half_m = m >> 1;
        for (size_t i = 0; i < slots; i += m) {
            const size_t gap = slots / m;
            const size_t mask = (m << 2) - 1;
            for (size_t j = 0; j < half_m; ++j) {
                const size_t root_index = ((m << 2) - (pow5[j] & mask)) * gap;
                const size_t idx1 = i + j;
                const size_t idx2 = idx1 + half_m;

                for (size_t u = 0; u < copy_count; ++u) {
                    const size_t offset = u * slots;
                    a[idx1 + offset] = Complex(1.0, 0.0);
                    a[idx2 + offset] = -roots[root_index];
                    b[idx1 + offset] = Complex(1.0, 0.0);
                    c[idx2 + offset] = roots[root_index];
                }
            }
        }

        vectors.a[index] = std::move(a);
        vectors.b[index] = std::move(b);
        vectors.c[index] = std::move(c);
        ++index;
        if (m == 2) {
            break;
        }
    }

    return vectors;
}

void add_to_dft_diagonal_map(
    SparseDftDiagonalMap& diagonal_map,
    size_t diagonal_index,
    const MyVector<Complex>& values) {
    auto [it, inserted] = diagonal_map.emplace(diagonal_index, values);
    if (!inserted) {
        auto& dest = it->second;
        if (dest.size() != values.size()) {
            throw std::invalid_argument("ckks_bootstrap_dft: inconsistent sparse diagonal vector size");
        }
        for (size_t i = 0; i < dest.size(); ++i) {
            dest[i] += values[i];
        }
    }
}

MyVector<Complex> rotate_and_multiply_dft_diagonal(
    const MyVector<Complex>& values,
    int64_t rotation,
    const MyVector<Complex>& multiplier) {
    if (values.empty() || values.size() != multiplier.size() || !std::has_single_bit(values.size())) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid sparse DFT diagonal multiplication");
    }

    const size_t mask = values.size() - 1;
    const size_t offset =
        static_cast<size_t>(
            (rotation % static_cast<int64_t>(values.size()) + static_cast<int64_t>(values.size())) %
            static_cast<int64_t>(values.size()));
    MyVector<Complex> out(values.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = multiplier[i] * values[(i + offset) & mask];
    }
    return out;
}

size_t dft_stage_rotation(
    size_t log_slots,
    size_t modulo,
    size_t level,
    CKKSBootstrapDftType type,
    bool bitreversed) {
    const bool homomorphic_encode = type == CKKSBootstrapDftType::CoeffsToSlots;
    const size_t rotation =
        (homomorphic_encode && !bitreversed) || (!homomorphic_encode && bitreversed)
            ? (size_t{1} << (level - 1))
            : (size_t{1} << (log_slots - level));
    return rotation & (modulo - 1);
}

SparseDftDiagonalMap generate_fft_diagonal_map(
    size_t log_slots,
    size_t fft_level,
    const MyVector<Complex>& a,
    const MyVector<Complex>& b,
    const MyVector<Complex>& c,
    CKKSBootstrapDftType type,
    bool bitreversed) {
    const size_t slots = size_t{1} << log_slots;
    const size_t rotation = dft_stage_rotation(log_slots, slots, fft_level, type, bitreversed);

    SparseDftDiagonalMap diagonal_map;
    add_to_dft_diagonal_map(diagonal_map, 0, a);
    add_to_dft_diagonal_map(diagonal_map, rotation, b);
    add_to_dft_diagonal_map(diagonal_map, (slots - rotation) & (slots - 1), c);
    return diagonal_map;
}

SparseDftDiagonalMap merge_fft_diagonal_map_with_next_level(
    const SparseDftDiagonalMap& diagonal_map,
    size_t log_slots,
    size_t modulo,
    size_t next_level,
    const MyVector<Complex>& a,
    const MyVector<Complex>& b,
    const MyVector<Complex>& c,
    CKKSBootstrapDftType type,
    bool bitreversed) {
    const size_t rotation = dft_stage_rotation(log_slots, modulo, next_level, type, bitreversed);

    SparseDftDiagonalMap merged_map;
    for (const auto& [diagonal_index, values] : diagonal_map) {
        add_to_dft_diagonal_map(
            merged_map,
            diagonal_index,
            rotate_and_multiply_dft_diagonal(values, 0, a));
        add_to_dft_diagonal_map(
            merged_map,
            (diagonal_index + rotation) & (modulo - 1),
            rotate_and_multiply_dft_diagonal(values, static_cast<int64_t>(rotation), b));
        add_to_dft_diagonal_map(
            merged_map,
            (diagonal_index + modulo - rotation) & (modulo - 1),
            rotate_and_multiply_dft_diagonal(values, -static_cast<int64_t>(rotation), c));
    }
    return merged_map;
}

MyVector<size_t> default_dft_merge_depths(
    size_t log_slots,
    size_t factor_count,
    CKKSBootstrapDftType type) {
    if (factor_count == 0 || factor_count > log_slots) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid DFT factor count");
    }

    MyVector<size_t> merge(factor_count);
    size_t level = log_slots;
    for (size_t i = 0; i < factor_count; ++i) {
        const size_t remaining_factors = factor_count - i;
        const size_t depth = (level + remaining_factors - 1) / remaining_factors;
        if (type == CKKSBootstrapDftType::CoeffsToSlots) {
            merge[i] = depth;
        } else {
            merge[factor_count - i - 1] = depth;
        }
        level -= depth;
    }
    return merge;
}

MyVector<size_t> validate_default_dft_merge_depths(
    size_t log_slots,
    size_t factor_count,
    const std::vector<size_t>& merge_depths) {
    if (merge_depths.empty()) {
        throw std::invalid_argument("ckks_bootstrap_dft: empty DFT merge-depth override");
    }
    if (merge_depths.size() != factor_count) {
        throw std::invalid_argument("ckks_bootstrap_dft: DFT merge-depth override size mismatch");
    }

    size_t total_depth = 0;
    MyVector<size_t> out;
    out.reserve(merge_depths.size());
    for (size_t depth : merge_depths) {
        if (depth == 0) {
            throw std::invalid_argument("ckks_bootstrap_dft: DFT merge depths must be positive");
        }
        total_depth += depth;
        out.emplace_back(depth);
    }
    if (total_depth != log_slots) {
        throw std::invalid_argument("ckks_bootstrap_dft: DFT merge-depth override must sum to log_slots");
    }
    return out;
}

FactorizedDftStageSource build_factorized_dft_stage_source(
    size_t log_slots,
    size_t factor_count,
    CKKSBootstrapDftType type,
    const std::vector<size_t>& merge_depths) {
    const size_t slots = size_t{1} << log_slots;
    const auto roots = factorized_dft_roots(slots << 2);
    const auto pow5 = dft_pow5_table(slots);

    FactorizedDftStageSource source;
    source.log_slots = log_slots;
    source.slots = slots;
    source.type = type;
    source.bitreversed = false;
    source.plain_vectors =
        type == CKKSBootstrapDftType::CoeffsToSlots
            ? ifft_plain_vectors(log_slots, slots, roots, pow5)
            : fft_plain_vectors(log_slots, slots, roots, pow5);
    if (source.bitreversed) {
        bit_reverse_dft_plain_vectors(source.plain_vectors, log_slots);
    }
    source.merge = merge_depths.empty()
        ? default_dft_merge_depths(log_slots, factor_count, type)
        : validate_default_dft_merge_depths(log_slots, factor_count, merge_depths);
    return source;
}

SparseDftDiagonalMap build_factorized_dft_stage_diagonal_map(
    const FactorizedDftStageSource& source,
    size_t stage_index,
    double stage_matrix_scale) {
    size_t fft_level = source.log_slots;
    for (size_t i = 0; i < stage_index; ++i) {
        fft_level -= source.merge[i];
    }

    SparseDftDiagonalMap diagonal_map = generate_fft_diagonal_map(
        source.log_slots,
        fft_level,
        source.plain_vectors.a[source.log_slots - fft_level],
        source.plain_vectors.b[source.log_slots - fft_level],
        source.plain_vectors.c[source.log_slots - fft_level],
        source.type,
        source.bitreversed);

    size_t next_level = fft_level - 1;
    for (size_t j = 0; j < source.merge[stage_index] - 1; ++j) {
        diagonal_map = merge_fft_diagonal_map_with_next_level(
            diagonal_map,
            source.log_slots,
            source.slots,
            next_level,
            source.plain_vectors.a[source.log_slots - next_level],
            source.plain_vectors.b[source.log_slots - next_level],
            source.plain_vectors.c[source.log_slots - next_level],
            source.type,
            source.bitreversed);
        --next_level;
    }

    for (auto& [diagonal_index, values] : diagonal_map) {
        if (diagonal_index >= source.slots || values.size() != source.slots) {
            throw std::invalid_argument("ckks_bootstrap_dft: invalid sparse DFT diagonal");
        }
        for (auto& value : values) {
            value *= stage_matrix_scale;
        }
    }
    return diagonal_map;
}

double max_abs_diagonal_value(const SparseDftDiagonalMap& diagonal_map) {
    double max_abs = 0.0;
    for (const auto& [diagonal_index, diagonal] : diagonal_map) {
        (void)diagonal_index;
        for (const auto& value : diagonal) {
            max_abs = std::max(max_abs, std::abs(value));
        }
    }
    return max_abs;
}

bool is_zero_diagonal(const MyVector<Complex>& diagonal) {
    for (const auto& value : diagonal) {
        if (value.real() != 0.0 || value.imag() != 0.0) {
            return false;
        }
    }
    return true;
}

size_t auto_bsgs_baby_step_count_from_diagonal_indices(
    size_t slots,
    const MyVector<size_t>& diagonal_indices,
    int log_bsgs_ratio) {
    if (slots == 0 || diagonal_indices.empty()) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid empty diagonal index set");
    }
    if (diagonal_indices.size() <= 1) {
        return 1;
    }

    size_t best = 1;
    size_t best_cost = std::numeric_limits<size_t>::max();
    size_t best_giant_count = std::numeric_limits<size_t>::max();
    const size_t giant_weight = size_t{1} << std::max(0, log_bsgs_ratio);
    for (size_t n1 = 1; n1 < slots; n1 <<= 1) {
        MyVector<char> rot_n1(slots, char{0});
        MyVector<char> rot_n2(slots, char{0});
        size_t count_n1 = 0;
        size_t count_n2 = 0;
        for (size_t diagonal_index : diagonal_indices) {
            if (diagonal_index >= slots) {
                throw std::invalid_argument("ckks_bootstrap_dft: diagonal index exceeds slot count");
            }
            const size_t rot = diagonal_index & (slots - 1);
            const size_t idx_n1 = ((rot / n1) * n1) & (slots - 1);
            const size_t idx_n2 = rot & (n1 - 1);
            if (!rot_n1[idx_n1]) {
                rot_n1[idx_n1] = char{1};
                ++count_n1;
            }
            if (!rot_n2[idx_n2]) {
                rot_n2[idx_n2] = char{1};
                ++count_n2;
            }
        }

        const size_t nonzero_n1 = count_n1 > 0 && rot_n1[0] ? count_n1 - 1 : count_n1;
        const size_t nonzero_n2 = count_n2 > 0 && rot_n2[0] ? count_n2 - 1 : count_n2;
        const size_t cost = giant_weight * nonzero_n1 + nonzero_n2;
        if (cost < best_cost ||
            (cost == best_cost && nonzero_n1 < best_giant_count)) {
            best_cost = cost;
            best_giant_count = nonzero_n1;
            best = n1;
        }
    }
    return best;
}

CKKSLinearTransformCompiledStage compile_linear_transform_stage_metadata(
    size_t slots,
    size_t term_count,
    const MyVector<size_t>& diagonal_indices,
    size_t baby_step_count) {
    if (slots == 0 || term_count == 0 || baby_step_count == 0 || baby_step_count > slots) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid linear-transform compile input");
    }

    CKKSLinearTransformCompiledStage compiled;
    compiled.baby_step_count = baby_step_count;
    compiled.giant_step_count = (slots + baby_step_count - 1) / baby_step_count;
    compiled.terms_by_giant.resize(compiled.giant_step_count);
    compiled.baby_rotation_needed.assign(baby_step_count, char{0});
    compiled.giant_rotation_needed.assign(compiled.giant_step_count, char{0});

    for (size_t term = 0; term < term_count; ++term) {
        const size_t diagonal_index = diagonal_indices.empty() ? term : diagonal_indices[term];
        if (diagonal_index >= slots) {
            throw std::invalid_argument("ckks_bootstrap_dft: compiled diagonal index exceeds slot count");
        }
        const size_t baby = diagonal_index % baby_step_count;
        const size_t giant = diagonal_index / baby_step_count;
        compiled.terms_by_giant[giant].emplace_back(term);
        compiled.baby_rotation_needed[baby] = char{1};
        compiled.giant_rotation_needed[giant] = char{1};
    }

    for (size_t baby = 1; baby < compiled.baby_rotation_needed.size(); ++baby) {
        if (compiled.baby_rotation_needed[baby]) {
            compiled.baby_rotations.emplace_back(baby);
        }
    }
    for (size_t giant = 1; giant < compiled.giant_rotation_needed.size(); ++giant) {
        if (compiled.giant_rotation_needed[giant]) {
            compiled.giant_rotations.emplace_back(giant * baby_step_count);
        }
    }
    return compiled;
}

CKKSLinearTransformOwnedStage encode_sparse_dft_stage(
    const CKKSContext& context,
    const CKKSBootstrapDftMatrixLiteral& literal,
    const SparseDftDiagonalMap& diagonal_map,
    bool final_stage = true) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t full_slots = params.getSlots();
    const size_t active_slots = dft_literal_active_slots(context, literal);
    const size_t level = literal.level;
    const size_t limb_count = level + 1;
    const bool montgomery_form = params.montgomery;

    if (diagonal_map.empty()) {
        throw std::invalid_argument("ckks_bootstrap_dft: empty sparse diagonal map");
    }
    if (level > params.getMaxLevel() || limb_count > params.getModuli().size()) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid sparse matrix level");
    }
    if (!(literal.plaintext_scale > 0.0)) {
        throw std::invalid_argument("ckks_bootstrap_dft: sparse plaintext scale must be positive");
    }

    MyVector<uint64_t> active_moduli(
        params.getModuli().begin(),
        params.getModuli().begin() + limb_count);
    const auto qp_moduli = make_qp_moduli(context, limb_count);
    const auto qp_twiddles = make_qp_twiddles(context, limb_count);

    MyVector<size_t> nonzero_diagonal_indices;
    MyVector<const MyVector<Complex>*> nonzero_diagonal_values;
    nonzero_diagonal_indices.reserve(diagonal_map.size());
    nonzero_diagonal_values.reserve(diagonal_map.size());
    for (const auto& [diagonal_index, values] : diagonal_map) {
        if (diagonal_index >= active_slots ||
            (values.size() != active_slots && values.size() != full_slots)) {
            throw std::invalid_argument("ckks_bootstrap_dft: invalid sparse DFT diagonal");
        }
        if (!is_zero_diagonal(values)) {
            nonzero_diagonal_indices.emplace_back(diagonal_index);
            nonzero_diagonal_values.emplace_back(&values);
        }
    }
    if (nonzero_diagonal_indices.empty()) {
        throw std::runtime_error("ckks_bootstrap_dft: generated sparse stage has no nonzero diagonals");
    }

    const size_t baby_step_count = literal.baby_step_count == 0
        ? auto_bsgs_baby_step_count_from_diagonal_indices(
              active_slots,
              nonzero_diagonal_indices,
              literal.log_bsgs_ratio)
        : literal.baby_step_count;
    if (baby_step_count == 0 || baby_step_count > active_slots) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid sparse baby-step count");
    }

    CKKSLinearTransformOwnedStage stage;
    stage.diagonals.reserve(nonzero_diagonal_indices.size());
    stage.diagonal_qp_plaintexts.reserve(nonzero_diagonal_indices.size());
    stage.diagonal_indices = nonzero_diagonal_indices;
    stage.slot_count = active_slots;
    stage.baby_step_count = baby_step_count;
    stage.qp_limb_count = qp_moduli.size();
    stage.qp_p_limb_count = params.getPModuli().size();
    stage.diagonal_qp_plaintexts_montgomery = montgomery_form;
    stage.compiled = compile_linear_transform_stage_metadata(
        active_slots,
        nonzero_diagonal_indices.size(),
        nonzero_diagonal_indices,
        baby_step_count);
    stage.rescale_after = literal.rescale_after_each_stage;
    stage.output_encoding_state =
        (final_stage && literal.type == CKKSBootstrapDftType::SlotsToCoeffs)
            ? CKKSMessageEncodingState::Coefficients
            : CKKSMessageEncodingState::Slots;

    const size_t term_count = nonzero_diagonal_indices.size();
    MyVector<MyVector<uint64_t>> plaintext_buffers(term_count);
    MyVector<MyVector<uint64_t>> plaintext_qp_buffers(term_count);

    #pragma omp parallel for schedule(dynamic) if (term_count > 1)
    for (std::ptrdiff_t term_idx = 0;
         term_idx < static_cast<std::ptrdiff_t>(term_count);
         ++term_idx) {
        const size_t term = static_cast<size_t>(term_idx);
        const size_t diagonal_index = nonzero_diagonal_indices[term];
        const size_t giant_offset = (diagonal_index / baby_step_count) * baby_step_count;
        const auto diagonal = expand_periodic_diagonal_to_physical_slots(
            *nonzero_diagonal_values[term],
            active_slots,
            full_slots);

        plaintext_buffers[term] = encode_slots_custom_basis_rotated(
            N,
            diagonal,
            giant_offset,
            literal.plaintext_scale,
            active_moduli,
            context.getTwiddleNtt(),
            context.getTwiddleFft(),
            context.getPermTable(),
            /*eval=*/true,
            montgomery_form);
        plaintext_qp_buffers[term] = encode_slots_custom_basis_rotated(
            N,
            diagonal,
            giant_offset,
            literal.plaintext_scale,
            qp_moduli,
            qp_twiddles,
            context.getTwiddleFft(),
            context.getPermTable(),
            /*eval=*/true,
            montgomery_form);
    }

    for (size_t term = 0; term < term_count; ++term) {
        stage.diagonals.emplace_back(
            N,
            std::move(plaintext_buffers[term]),
            literal.plaintext_scale,
            level,
            montgomery_form);
        stage.diagonal_qp_plaintexts.emplace_back(std::move(plaintext_qp_buffers[term]));
    }

    return stage;
}

void validate_dft_literal(
    const CKKSContext& context,
    const CKKSBootstrapDftMatrixLiteral& literal) {
    const auto& params = context.getParams();
    const size_t slots = dft_literal_active_slots(context, literal);
    const size_t factor_count = dft_factor_count(literal);
    const size_t rescale_group_count = dft_rescale_group_count(literal);
    if (factor_count == 0) {
        throw std::invalid_argument("ckks_bootstrap_dft: stage count must be positive");
    }
    if (literal.level > params.getMaxLevel()) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid matrix level");
    }
    if (!(literal.plaintext_scale > 0.0)) {
        throw std::invalid_argument("ckks_bootstrap_dft: plaintext scale must be positive");
    }
    if (!(literal.transform_scale > 0.0)) {
        throw std::invalid_argument("ckks_bootstrap_dft: transform scale must be positive");
    }
    if (literal.input_selector_size != 0 &&
        literal.type != CKKSBootstrapDftType::SlotsToCoeffs) {
        throw std::invalid_argument(
            "ckks_bootstrap_dft: input selector is only valid for S2C");
    }
    if (literal.input_selector_size != 0 &&
        (literal.input_selector_start >= slots ||
         literal.input_selector_size > slots)) {
        throw std::invalid_argument(
            "ckks_bootstrap_dft: input selector exceeds slot count");
    }
    const size_t total_layers = std::countr_zero(slots);
    if (!std::has_single_bit(params.getN()) ||
        factor_count > total_layers) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid factorized DFT stage count");
    }
    if (literal.rescale_after_each_stage && rescale_group_count > literal.level) {
        throw std::invalid_argument("ckks_bootstrap_dft: factorized DFT consumes more levels than available");
    }
}

} // namespace

CKKSLinearTransformPlan ckks_generate_bootstrap_dft_plan(
    const CKKSContext& context,
    const CKKSBootstrapDftMatrixLiteral& literal) {
    validate_dft_literal(context, literal);

    const size_t factor_count = dft_factor_count(literal);
    const auto factor_schedule = flatten_dft_factor_schedule(literal);

    CKKSLinearTransformPlan plan;
    plan.use_qp_evaluator =
        literal.type == CKKSBootstrapDftType::SlotsToCoeffs
            ? context.getParams().getBootstrapS2CUseQpLinearTransform()
            : context.getParams().getBootstrapC2SUseQpLinearTransform();
    plan.stages.reserve(factor_count);

    if (literal.type == CKKSBootstrapDftType::SlotsToCoeffs) {
        size_t stage_level = literal.level;
        const size_t slots = dft_literal_active_slots(context, literal);
        const size_t log_slots = std::countr_zero(slots);
        const auto dft_source =
            build_factorized_dft_stage_source(
                log_slots,
                factor_count,
                literal.type,
                literal.merge_depths);
        const double stage_matrix_scale =
            std::pow(literal.transform_scale, 1.0 / static_cast<double>(factor_count));
        for (size_t stage = 0; stage < factor_count; ++stage) {
            auto diagonal_map = build_factorized_dft_stage_diagonal_map(
                dft_source,
                stage,
                stage_matrix_scale);
            if (stage == 0 && literal.input_selector_size != 0) {
                apply_input_selector_to_diagonal_map(
                    diagonal_map,
                    slots,
                    literal.input_selector_start,
                    literal.input_selector_size);
            }
            const double max_abs_value = max_abs_diagonal_value(diagonal_map);
            if (!(max_abs_value > 0.0)) {
                throw std::runtime_error("ckks_bootstrap_dft: generated empty S2C stage");
            }

            CKKSBootstrapDftMatrixLiteral stage_literal = literal;
            stage_literal.level = stage_level;
            stage_literal.plaintext_scale = scheduled_plaintext_scale(
                context,
                literal,
                factor_schedule[stage],
                stage_level,
                max_abs_value);
            stage_literal.rescale_after_each_stage = factor_schedule[stage].ends_group;
            plan.stages.emplace_back(
                encode_sparse_dft_stage(
                    context,
                    stage_literal,
                    diagonal_map,
                    /*final_stage=*/stage + 1 == factor_count));
            if (factor_schedule[stage].ends_group && stage + 1 < factor_count) {
                if (stage_level == 0) {
                    throw std::runtime_error("ckks_bootstrap_dft: invalid exhausted S2C stage level");
                }
                --stage_level;
            }
        }
        return plan;
    }

    const size_t slots = dft_literal_active_slots(context, literal);
    const size_t log_slots = std::countr_zero(slots);
    if (factor_count > log_slots) {
        throw std::invalid_argument("ckks_bootstrap_dft: invalid native C2S stage count");
    }
    const auto dft_source =
        build_factorized_dft_stage_source(
            log_slots,
            factor_count,
            literal.type,
            literal.merge_depths);
    size_t stage_level = literal.level;
    const double stage_matrix_scale =
        std::pow(
            literal.transform_scale / static_cast<double>(slots),
            1.0 / static_cast<double>(factor_count));
    for (size_t stage = 0; stage < factor_count; ++stage) {
        auto diagonal_map = build_factorized_dft_stage_diagonal_map(
            dft_source,
            stage,
            stage_matrix_scale);
        const double max_abs_value = max_abs_diagonal_value(diagonal_map);
        if (!(max_abs_value > 0.0)) {
            throw std::runtime_error("ckks_bootstrap_dft: generated empty C2S stage");
        }

        CKKSBootstrapDftMatrixLiteral stage_literal = literal;
        stage_literal.level = stage_level;
        stage_literal.plaintext_scale = scheduled_plaintext_scale(
            context,
            literal,
            factor_schedule[stage],
            stage_level,
            max_abs_value);
        stage_literal.rescale_after_each_stage = factor_schedule[stage].ends_group;
        plan.stages.emplace_back(
            encode_sparse_dft_stage(
                context,
                stage_literal,
                diagonal_map,
                /*final_stage=*/stage + 1 == factor_count));
        if (factor_schedule[stage].ends_group && stage + 1 < factor_count) {
            if (stage_level == 0) {
                throw std::runtime_error("ckks_bootstrap_dft: invalid exhausted C2S stage level");
            }
            --stage_level;
        }
    }
    return plan;
}
