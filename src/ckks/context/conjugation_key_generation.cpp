#include "conjugation_key_generation.hpp"

#include <algorithm>
#include <stdexcept>

#include "ckks_encoding.hpp"
#include "hybrid_key_folding.hpp"
#include "hybrid_key_montgomery.hpp"
#include "modarith.hpp"
#include "ntt.hpp"
#include "polynomial.hpp"
#include "secret_key_generation.hpp"

namespace {

struct AutomorphismMap {
    MyVector<size_t> dst_index;
    MyVector<uint8_t> negate;
};

AutomorphismMap build_automorphism_map(size_t N, uint64_t automorphism_index) {
    const size_t ring_dim = N << 1;
    AutomorphismMap map;
    map.dst_index.resize(N);
    map.negate.resize(N, uint8_t{0});

    for (size_t coeff = 0; coeff < N; ++coeff) {
        const size_t mapped = (coeff * automorphism_index) % ring_dim;
        if (mapped >= N) {
            map.dst_index[coeff] = mapped - N;
            map.negate[coeff] = 1;
        } else {
            map.dst_index[coeff] = mapped;
        }
    }
    return map;
}

MyVector<int8_t> apply_automorphism_to_secret(
    const MyVector<int8_t>& secret_coeffs,
    const AutomorphismMap& map) {
    if (secret_coeffs.size() != map.dst_index.size()) {
        throw std::invalid_argument("ckks_context: invalid automorphism map for conjugation key");
    }

    MyVector<int8_t> rotated(secret_coeffs.size(), int8_t{0});
    for (size_t i = 0; i < secret_coeffs.size(); ++i) {
        int8_t value = secret_coeffs[i];
        if (map.negate[i] != 0) {
            value = static_cast<int8_t>(-value);
        }
        rotated[map.dst_index[i]] = value;
    }
    return rotated;
}

} // namespace

uint64_t conjugation_automorphism_index(size_t N) {
    return static_cast<uint64_t>((N << 1) - 1);
}

MyVector<CKKSCiphertext> generate_conjugation_key_hybrid(
    const CKKSContext& context,
    const MyVector<int8_t>& secret_coeffs) {
    const auto& params = context.getParams();
    const auto level = params.getMaxLevel();
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const auto& q_twiddle_ntt = context.getTwiddleNtt();
    const auto& p_twiddle_ntt = context.getPTwiddleNtt();
    const size_t N = params.getN();
    const size_t q_limb_count = level + 1;
    const size_t dnum = params.getDnum();
    const double sigma = params.getSigma();

    const uint64_t automorphism_index = conjugation_automorphism_index(N);
    const auto coeff_map = build_automorphism_map(N, automorphism_index);
    const auto conjugated_secret_coeffs = apply_automorphism_to_secret(secret_coeffs, coeff_map);

    MyVector<uint64_t> qup_moduli;
    qup_moduli.reserve(q_limb_count + p_moduli.size());
    qup_moduli.insert(qup_moduli.end(), q_moduli.begin(), q_moduli.begin() + q_limb_count);
    qup_moduli.insert(qup_moduli.end(), p_moduli.begin(), p_moduli.end());

    MyVector<Negacyclic_NTT_Twiddles> qup_twiddle_ntt;
    qup_twiddle_ntt.reserve(qup_moduli.size());
    for (size_t i = 0; i < q_limb_count; ++i) {
        qup_twiddle_ntt.emplace_back(q_twiddle_ntt[i]);
    }
    for (size_t i = 0; i < p_moduli.size(); ++i) {
        qup_twiddle_ntt.emplace_back(p_twiddle_ntt[i]);
    }

    const auto sk_qup = ntt_secret_key(secret_coeffs, qup_twiddle_ntt, N, qup_moduli);
    const auto conjugated_sk_qup =
        ntt_secret_key(conjugated_secret_coeffs, qup_twiddle_ntt, N, qup_moduli);

    MyVector<uint64_t> p_mod_m(qup_moduli.size(), uint64_t{1});
    for (size_t i = 0; i < qup_moduli.size(); ++i) {
        const uint64_t m = qup_moduli[i];
        const auto* barrett = &qup_twiddle_ntt[i].barrett_const;
        uint64_t accum = 1;
        for (uint64_t p : p_moduli) {
            accum = mul_mod_u64(accum, p % m, m, barrett);
        }
        p_mod_m[i] = accum;
    }

    const size_t alpha = (q_limb_count + dnum - 1) / dnum;
    if (alpha == 0) {
        throw std::runtime_error("ckks_context: invalid hybrid conjugation-key partition width");
    }

    MyVector<CKKSCiphertext> switching_key;
    switching_key.reserve(dnum);
    for (size_t part = 0; part < dnum; ++part) {
        const size_t part_start = part * alpha;
        const size_t part_end = std::min(part_start + alpha, q_limb_count);
        if (part_start >= part_end) {
            continue;
        }

        MyVector<uint64_t> b_poly(N * qup_moduli.size(), uint64_t{0});
        for (size_t i = part_start; i < part_end; ++i) {
            const uint64_t q = q_moduli[i];
            const auto& tbl = qup_twiddle_ntt[i];
            uint64_t* dst_ptr = b_poly.data() + i * N;
            std::copy(conjugated_sk_qup[i].begin(), conjugated_sk_qup[i].end(), dst_ptr);
            pointwise_multiply_scalar_inplace_dispatch(dst_ptr, p_mod_m[i], N,
                                                       q, tbl);
        }

        auto a_poly = ckks_uniform_mask_polys(N, qup_moduli, qup_moduli.size());
        auto e_poly = ckks_gaussian_noise_polys(N, qup_moduli, qup_moduli.size(), sigma);
        ntt_forward_rns_flat_inplace(
            e_poly.data(), N, qup_moduli.size(), qup_moduli, qup_twiddle_ntt, false);

        MyVector<uint64_t> as_poly(N);
        for (size_t i = 0; i < qup_moduli.size(); ++i) {
            const uint64_t q = qup_moduli[i];
            const auto& tbl = qup_twiddle_ntt[i];
            uint64_t* a_ptr = a_poly.data() + i * N;
            const uint64_t* s_ptr = sk_qup[i].data();
            uint64_t* e_ptr = e_poly.data() + i * N;
            uint64_t* b_ptr = b_poly.data() + i * N;

            pointwise_multiply_dispatch(a_ptr, s_ptr, as_poly.data(), N, q,
                                        tbl);
            poly_add_modq_inplace(b_ptr, e_ptr, N, q);
            poly_sub_modq_inplace(b_ptr, as_poly.data(), N, q);
        }

        CKKSCiphertext key(N, std::move(a_poly), std::move(b_poly), 1.0, level);
        if (params.fold_hybrid_keyswitch) {
            ckks_fold_hybrid_switch_key_constants(key, q_moduli, p_moduli, qup_twiddle_ntt);
        }
        if (params.montgomery) {
            ckks_convert_switch_key_to_montgomery_inplace(
                key,
                qup_moduli,
                qup_twiddle_ntt);
        }
        switching_key.emplace_back(std::move(key));
    }
    return switching_key;
}
