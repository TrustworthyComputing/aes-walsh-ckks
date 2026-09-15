#include "secret_key_generation.hpp"

#include <algorithm>
#include <cerrno>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sys/random.h>
#include <utility>

#include "ckks_encoding.hpp"
#include "hybrid_key_folding.hpp"
#include "hybrid_key_montgomery.hpp"
#include "modarith.hpp"
#include "ntt.hpp"

namespace {

struct QupBasis {
    MyVector<uint64_t> moduli;
    MyVector<Negacyclic_NTT_Twiddles> twiddles;
};

std::pair<std::size_t, std::size_t> bootstrap_secret_interval_bounds(
    std::size_t N,
    std::size_t hamming_weight,
    std::size_t window_width,
    std::size_t nonzero_rank) {
    if (hamming_weight == 0) {
        throw std::invalid_argument("ckks_context: sparse secret Hamming weight must be positive");
    }
    if (window_width == 0 || window_width > N) {
        throw std::invalid_argument("ckks_context: invalid sparse secret interval width");
    }
    const std::size_t center = (nonzero_rank * N) / hamming_weight;
    const std::size_t max_window_start = N - window_width;
    const std::size_t half_window = window_width / 2;
    const std::size_t unclamped_start = (center > half_window) ? (center - half_window) : 0;
    const std::size_t window_start = std::min(unclamped_start, max_window_start);
    return {window_start, window_start + window_width};
}

QupBasis build_qup_basis(const CKKSContext& context, size_t q_limb_count) {
    const auto& params = context.getParams();
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const auto& q_twiddle_ntt = context.getTwiddleNtt();
    const auto& p_twiddle_ntt = context.getPTwiddleNtt();

    QupBasis basis;
    basis.moduli.reserve(q_limb_count + p_moduli.size());
    basis.moduli.insert(basis.moduli.end(), q_moduli.begin(), q_moduli.begin() + q_limb_count);
    basis.moduli.insert(basis.moduli.end(), p_moduli.begin(), p_moduli.end());

    basis.twiddles.reserve(basis.moduli.size());
    for (size_t i = 0; i < q_limb_count; ++i) {
        basis.twiddles.emplace_back(q_twiddle_ntt[i]);
    }
    for (size_t i = 0; i < p_moduli.size(); ++i) {
        basis.twiddles.emplace_back(p_twiddle_ntt[i]);
    }
    return basis;
}

} // namespace

MyVector<int8_t> generate_uniform_terenary_secret(std::size_t N) {
    auto getrandom_fill = [](uint8_t* out, std::size_t n) {
        std::size_t off = 0;
        while (off < n) {
            const ssize_t r = ::getrandom(out + off, n - off, 0);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("getrandom() failed");
            }
            off += static_cast<std::size_t>(r);
        }
    };

    MyVector<int8_t> sk(N);
    MyVector<uint8_t> buf(4096);
    std::size_t i = 0;
    while (i < N) {
        getrandom_fill(buf.data(), buf.size());
        for (uint8_t b : buf) {
            for (int k = 0; k < 4 && i < N; ++k) {
                const uint8_t r = (b >> (2 * k)) & 0x03u;
                if (r == 3u) {
                    continue;
                }
                sk[i++] = (r == 0u) ? int8_t(-1) : (r == 1u) ? int8_t(0) : int8_t(1);
            }
            if (i == N) {
                break;
            }
        }
    }
    return sk;
}

MyVector<int8_t> generate_hamming_weight_ternary_secret(
    std::size_t N,
    std::size_t hamming_weight) {
    if (hamming_weight > N) {
        throw std::invalid_argument("ckks_context: secret Hamming weight exceeds ring dimension");
    }

    auto getrandom_u64 = []() {
        uint64_t value = 0;
        std::size_t off = 0;
        while (off < sizeof(value)) {
            const ssize_t r =
                ::getrandom(reinterpret_cast<uint8_t*>(&value) + off, sizeof(value) - off, 0);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("getrandom() failed");
            }
            off += static_cast<std::size_t>(r);
        }
        return value;
    };

    MyVector<std::size_t> indices(N);
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    std::mt19937_64 rng(getrandom_u64());
    std::shuffle(indices.begin(), indices.end(), rng);

    MyVector<int8_t> sk(N, int8_t{0});
    for (std::size_t i = 0; i < hamming_weight; ++i) {
        sk[indices[i]] = (rng() & 1ULL) == 0ULL ? int8_t{-1} : int8_t{1};
    }
    return sk;
}

MyVector<int8_t> generate_sparse_ternary_secret(
    std::size_t N,
    std::size_t hamming_weight,
    std::size_t window_width) {
    if (hamming_weight > N) {
        throw std::invalid_argument("ckks_context: sparse secret Hamming weight exceeds ring dimension");
    }

    auto getrandom_u64 = []() {
        uint64_t value = 0;
        std::size_t off = 0;
        while (off < sizeof(value)) {
            const ssize_t r =
                ::getrandom(reinterpret_cast<uint8_t*>(&value) + off, sizeof(value) - off, 0);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("getrandom() failed");
            }
            off += static_cast<std::size_t>(r);
        }
        return value;
    };

    MyVector<int8_t> sk_low(N, int8_t{0});
    MyVector<std::size_t> interval_starts(hamming_weight, 0);
    MyVector<std::size_t> interval_ends(hamming_weight, 0);
    for (std::size_t k = 0; k < hamming_weight; ++k) {
        const auto [interval_start, interval_end] =
            bootstrap_secret_interval_bounds(N, hamming_weight, window_width, k);
        interval_starts[k] = interval_start;
        interval_ends[k] = interval_end;
    }

    MyVector<std::size_t> latest_feasible(hamming_weight, 0);
    latest_feasible[hamming_weight - 1] = interval_ends[hamming_weight - 1] - 1;
    for (std::size_t k = hamming_weight - 1; k-- > 0;) {
        if (latest_feasible[k + 1] == 0) {
            throw std::invalid_argument(
                "ckks_context: sparse secret interval configuration has no feasible assignment");
        }
        latest_feasible[k] = std::min(interval_ends[k] - 1, latest_feasible[k + 1] - 1);
        if (latest_feasible[k] < interval_starts[k]) {
            throw std::invalid_argument(
                "ckks_context: sparse secret interval configuration has no feasible assignment");
        }
    }

    std::size_t previous_index = static_cast<std::size_t>(-1);

    for (std::size_t k = 0; k < hamming_weight; ++k) {
        const std::size_t start = std::max(interval_starts[k], previous_index + 1);
        const std::size_t end_inclusive = latest_feasible[k];
        if (start > end_inclusive) {
            throw std::runtime_error("ckks_context: failed to place sparse secret index in interval");
        }

        const std::size_t span = end_inclusive - start + 1;
        const std::size_t index = start + static_cast<std::size_t>(getrandom_u64() % span);
        sk_low[index] = ((getrandom_u64() & 1ULL) == 0ULL) ? int8_t{-1} : int8_t{1};
        previous_index = index;
    }
    return sk_low;
}

MyVector<MyVector<uint64_t>> ntt_secret_key(
    const MyVector<int8_t>& s,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt,
    size_t N,
    const MyVector<uint64_t>& moduli) {
    const size_t limb_count = moduli.size();
    MyVector<MyVector<uint64_t>> secret_key;
    secret_key.reserve(limb_count);

    for (size_t i = 0; i < limb_count; ++i) {
        const uint64_t q = moduli[i];
        secret_key.emplace_back(N);
        auto& s_q = secret_key.back();
        for (size_t j = 0; j < N; ++j) {
            const int8_t sj = s[j];
            s_q[j] = (sj < 0) ? (q - static_cast<uint64_t>(-sj)) : static_cast<uint64_t>(sj);
        }
        ntt_forward_dit2_dispatch(s_q.data(), N, q, twiddle_ntt[i]);
    }
    return secret_key;
}

MyVector<CKKSCiphertext> generate_secret_key_switch_key_hybrid(
    const CKKSContext& context,
    const MyVector<int8_t>& source_secret_coeffs,
    const MyVector<int8_t>& target_secret_coeffs) {
    const auto& params = context.getParams();
    const auto level = params.getMaxLevel();
    const size_t q_limb_count = level + 1;
    const size_t N = params.getN();
    const size_t dnum = params.getDnum();
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const double sigma = params.getSigma();

    if (target_secret_coeffs.size() != N || source_secret_coeffs.size() != N) {
        throw std::invalid_argument("ckks_context: secret key coefficients size mismatch");
    }

    const auto basis = build_qup_basis(context, q_limb_count);
    const auto source_sk_qup = ntt_secret_key(source_secret_coeffs, basis.twiddles, N, basis.moduli);
    const auto target_sk_qup = ntt_secret_key(target_secret_coeffs, basis.twiddles, N, basis.moduli);

    MyVector<uint64_t> p_mod_m(basis.moduli.size(), uint64_t{1});
    for (size_t i = 0; i < basis.moduli.size(); ++i) {
        const uint64_t m = basis.moduli[i];
        const auto* barrett = &basis.twiddles[i].barrett_const;
        uint64_t accum = 1;
        for (uint64_t p : p_moduli) {
            accum = mul_mod_u64(accum, p % m, m, barrett);
        }
        p_mod_m[i] = accum;
    }

    const size_t alpha = (q_limb_count + dnum - 1) / dnum;
    MyVector<CKKSCiphertext> out;
    out.reserve(dnum);
    for (size_t part = 0; part < dnum; ++part) {
        const size_t start = part * alpha;
        const size_t end = std::min(start + alpha, q_limb_count);
        if (start >= end) {
            continue;
        }

        MyVector<uint64_t> ptxt(N * basis.moduli.size(), uint64_t{0});
        for (size_t i = start; i < end; ++i) {
            const uint64_t q = q_moduli[i];
            const auto& tbl = basis.twiddles[i];
            uint64_t* dst = ptxt.data() + i * N;
            std::copy(source_sk_qup[i].begin(), source_sk_qup[i].end(), dst);
            pointwise_multiply_scalar_inplace_dispatch(dst, p_mod_m[i], N, q,
                                                       tbl);
        }

        auto key = ckks_encrypt_custom_eval(
            N,
            std::move(ptxt),
            target_sk_qup,
            basis.moduli,
            basis.twiddles,
            sigma,
            /*scale=*/1.0,
            level);
        if (params.fold_hybrid_keyswitch) {
            ckks_fold_hybrid_switch_key_constants(key, q_moduli, p_moduli, basis.twiddles);
        }
        if (params.montgomery) {
            ckks_convert_switch_key_to_montgomery_inplace(
                key,
                basis.moduli,
                basis.twiddles);
        }
        out.emplace_back(std::move(key));
    }
    return out;
}

MyVector<CKKSCiphertext> generate_secret_key_switch_key_q0p0_hybrid(
    const CKKSContext& context,
    const MyVector<int8_t>& source_secret_coeffs,
    const MyVector<int8_t>& target_secret_coeffs) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const double sigma = params.getSigma();

    if (target_secret_coeffs.size() != N || source_secret_coeffs.size() != N) {
        throw std::invalid_argument("ckks_context: secret key coefficients size mismatch");
    }
    if (q_moduli.empty() || p_moduli.empty()) {
        throw std::invalid_argument("ckks_context: q0p0 switch key requires Q and P moduli");
    }

    QupBasis basis;
    basis.moduli.reserve(2);
    basis.moduli.emplace_back(q_moduli[0]);
    basis.moduli.emplace_back(p_moduli[0]);
    basis.twiddles.reserve(2);
    basis.twiddles.emplace_back(context.getTwiddleNtt()[0]);
    basis.twiddles.emplace_back(context.getPTwiddleNtt()[0]);

    const MyVector<uint64_t> q0_moduli{q_moduli[0]};
    const MyVector<uint64_t> p0_moduli{p_moduli[0]};
    const auto source_sk_q0p0 =
        ntt_secret_key(source_secret_coeffs, basis.twiddles, N, basis.moduli);
    const auto target_sk_q0p0 =
        ntt_secret_key(target_secret_coeffs, basis.twiddles, N, basis.moduli);

    MyVector<uint64_t> ptxt(N * basis.moduli.size(), uint64_t{0});
    const uint64_t q0 = q_moduli[0];
    const uint64_t p0 = p_moduli[0];
    std::copy(source_sk_q0p0[0].begin(), source_sk_q0p0[0].end(), ptxt.begin());
    pointwise_multiply_scalar_inplace_dispatch(ptxt.data(), p0 % q0, N, q0,
                                               basis.twiddles[0]);

    auto key = ckks_encrypt_custom_eval(
        N,
        std::move(ptxt),
        target_sk_q0p0,
        basis.moduli,
        basis.twiddles,
        sigma,
        /*scale=*/1.0,
        /*level=*/0);
    if (params.fold_hybrid_keyswitch) {
        ckks_fold_hybrid_switch_key_constants(key, q0_moduli, p0_moduli, basis.twiddles);
    }
    if (params.montgomery) {
        ckks_convert_switch_key_to_montgomery_inplace(
            key,
            basis.moduli,
            basis.twiddles);
    }

    MyVector<CKKSCiphertext> out;
    out.reserve(1);
    out.emplace_back(std::move(key));
    return out;
}
