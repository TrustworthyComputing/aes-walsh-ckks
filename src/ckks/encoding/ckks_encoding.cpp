#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <bit>
#include <cmath>
// #include <stdexcept> // Required for standard exceptions
#include <stdexcept>
#include <utility>
#include <chrono>
#include <iostream>
#include <numbers>

#include "ckks_encoding.hpp"
#include "ckks_ciphertext.hpp"
#include "fft.hpp"
#include "ntt.hpp"
#include "modarith.hpp"
#include "polynomial.hpp"
#include "discrete_gaussian_sampler.hpp"


typedef unsigned __int128 uint128_t;
using Complex = std::complex<double>;

MyVector<Complex> crt_interpolate(const MyVector<uint64_t>& ptxt, size_t N, MyVector<uint64_t>& moduli);

namespace {
    void make_hermitian_evals(
        std::span<const Complex> src,
        MyVector<Complex>& dest,
        const MyVector<size_t>& perm_table,
        size_t N) {
        const Complex* __restrict in_ptr = src.data();
        Complex* __restrict out_ptr = dest.data();

        const size_t half_N = N >> 1;
        for (size_t i = 0; i < half_N; i++) {
            const Complex val = in_ptr[i];
            const size_t dest_idx = perm_table[i];
            out_ptr[dest_idx] = val;
            out_ptr[N - 1 - dest_idx] = std::conj(val);
        }
    }

    MyVector<Complex> extract_slots_from_evals(
        const MyVector<Complex>& evals,
        const MyVector<size_t>& perm_table,
        size_t half_N) {
        MyVector<Complex> slots(half_N);
        for (size_t i = 0; i < half_N; ++i) {
            slots[i] = evals[perm_table[i]];
        }
        return slots;
    }

    size_t bit_reverse_value(size_t value, size_t width) {
        size_t reversed = 0;
        for (size_t i = 0; i < width; ++i) {
            reversed = (reversed << 1) | (value & 1);
            value >>= 1;
        }
        return reversed;
    }


    inline uint64_t double_to_uint64(int64_t val, uint64_t q) {
        int64_t q_int = static_cast<int64_t>(q);
        int64_t r = val % q_int;
        // r >> 63 gives -1 (all 1s) if negative, 0 if positive
        // Masking with q adds q only when r is negative
        return static_cast<uint64_t>(r + (q_int & (r >> 63)));
    }

    void convert_plaintext_limb_from_montgomery(
        uint64_t* __restrict limb_ptr,
        size_t N,
        uint64_t q,
        uint64_t q_neg_inv) {
        for (size_t i = 0; i < N; ++i) {
            limb_ptr[i] = ckks_montgomery_mul_normal_by_montgomery(
                uint64_t{1},
                limb_ptr[i],
                q,
                q_neg_inv);
        }
    }

    MyVector<Complex> ckks_decrypt_with_secret_eval(
        const CKKSContext& context,
        const CKKSCiphertext& ct,
        const MyVector<MyVector<uint64_t>>& secret_key_eval,
        CKKSDecryptDomain domain) {
        const auto& a = ct.getA();
        const auto& b = ct.getB();
        const auto polys = ct.getNumPolys();
        const long double scale = ct.getScale();
        const auto& params = context.getParams();
        const auto& moduli = params.getModuli();
        const auto& twiddle_ntt = context.getTwiddleNtt();
        const auto N = params.getN();
        const auto level = ct.getLevel();
        const size_t limb_count = level + 1;

        if (polys != 2 && polys != 3) {
            throw std::invalid_argument("ckks_decrypt: expected 2- or 3-component ciphertext");
        }
        if (level > params.getMaxLevel() ||
            limb_count > moduli.size() ||
            limb_count > secret_key_eval.size() ||
            limb_count > twiddle_ntt.size()) {
            throw std::invalid_argument("ckks_decrypt: invalid level");
        }

        MyVector<uint64_t> temp(N * limb_count);
        MyVector<uint64_t> s2;
        MyVector<uint64_t> term2;
        const MyVector<uint64_t>* c2 = nullptr;
        if (polys == 3) {
            c2 = &ct.getPoly(2);
            s2.resize(N * limb_count);
            term2.resize(N * limb_count);
        }
        for (size_t i = 0; i < limb_count; ++i) {
            const uint64_t q = moduli[i];

            const uint64_t* __restrict a_ptr = a.data() + i * N;
            const uint64_t* __restrict s_ptr = secret_key_eval[i].data();
            const uint64_t* __restrict b_ptr = b.data() + i * N;
            uint64_t* __restrict temp_ptr = temp.data() + i * N;
            const auto& tbl = twiddle_ntt[i];

            pointwise_multiply_dispatch(a_ptr, s_ptr, temp_ptr, N, q, tbl);
            poly_add_modq_inplace(temp_ptr, b_ptr, N, q);

            if (polys == 3) {
                uint64_t* __restrict s2_ptr = s2.data() + i * N;
                uint64_t* __restrict term2_ptr = term2.data() + i * N;
                const uint64_t* __restrict c2_ptr = c2->data() + i * N;

                pointwise_multiply_dispatch(s_ptr, s_ptr, s2_ptr, N, q, tbl);
                pointwise_multiply_dispatch(c2_ptr, s2_ptr, term2_ptr, N, q,
                                            tbl);
                poly_add_modq_inplace(temp_ptr, term2_ptr, N, q);
            }
        }

        ntt_inverse_rns_flat_inplace(temp.data(), N, limb_count, moduli, twiddle_ntt);
        MyVector<uint64_t> moduli_used(moduli.begin(), moduli.begin() + limb_count);
        auto m = crt_interpolate(temp, N, moduli_used);
        for (size_t i = 0; i < N; ++i) {
            m[i] = Complex(
                static_cast<double>(static_cast<long double>(m[i].real()) / scale),
                0.0);
        }

        const auto half_N = N >> 1;
        if (domain == CKKSDecryptDomain::Coefficients) {
            const size_t log_slots = std::countr_zero(half_N);
            MyVector<Complex> decoded(half_N);
            for (size_t i = 0; i < half_N; ++i) {
                // Keep the public coefficient-vector order natural while the
                // homomorphic DFT uses the bit-reversed coefficient layout.
                const size_t coeff_index = bit_reverse_value(i, log_slots);
                decoded[i] = Complex(m[coeff_index].real(), m[half_N + coeff_index].real());
            }
            return decoded;
        }
        const auto& twiddle_ifft = context.getTwiddleIfft();
        const auto& perm_table = context.getPermTable();
        fft_dit(m, twiddle_ifft, N);
        return extract_slots_from_evals(m, perm_table, half_N);
    }

} // anon namespace

void ckks_convert_plaintext_to_montgomery_inplace(
    MyVector<uint64_t>& plaintext,
    size_t N,
    const MyVector<uint64_t>& moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt) {
    if (plaintext.size() != N * moduli.size() || twiddle_ntt.size() < moduli.size()) {
        throw std::invalid_argument("ckks_encoding: invalid Montgomery plaintext conversion basis");
    }
    for (size_t limb = 0; limb < moduli.size(); ++limb) {
        const uint64_t q = moduli[limb];
        pointwise_multiply_scalar_inplace_dispatch(
            plaintext.data() + limb * N,
            ckks_montgomery_radix_mod_q(q),
            N,
            q,
            twiddle_ntt[limb]);
    }
}

void ckks_convert_plaintext_from_montgomery_inplace(
    MyVector<uint64_t>& plaintext,
    size_t N,
    const MyVector<uint64_t>& moduli) {
    if (plaintext.size() != N * moduli.size()) {
        throw std::invalid_argument("ckks_encoding: invalid Montgomery plaintext conversion basis");
    }
    for (size_t limb = 0; limb < moduli.size(); ++limb) {
        const uint64_t q = moduli[limb];
        const uint64_t q_neg_inv = ckks_montgomery_neg_inverse(q);
        uint64_t* __restrict limb_ptr = plaintext.data() + limb * N;
        convert_plaintext_limb_from_montgomery(limb_ptr, N, q, q_neg_inv);
    }
}

MyVector<uint64_t> ckks_uniform_mask_polys(size_t N, const MyVector<uint64_t>& moduli, size_t limb_count) {
    MyVector<uint64_t> a_polys(N * limb_count);
    static thread_local std::mt19937_64 gen(std::random_device{}());

    for (size_t i = 0; i < limb_count; i++) {
        std::uniform_int_distribution<uint64_t> dist(0, moduli[i] - 1);
        uint64_t* level_ptr = &a_polys[i * N];
        for (size_t j = 0; j < N; j++) {
            level_ptr[j] = dist(gen);
        }
    }
    return a_polys;
}

MyVector<uint64_t> ckks_gaussian_noise_polys(size_t N,
                                             const MyVector<uint64_t>& moduli,
                                             size_t limb_count,
                                             double sigma) {
    static thread_local DiscreteGaussianSampler sampler;
    if (sampler.get_std() != sigma) {
        sampler.set_std(sigma);
    }

    MyVector<int64_t> noise_coeffs(N);
    for (size_t i = 0; i < N; i++) {
        noise_coeffs[i] = sampler.generate_int();
    }

    MyVector<uint64_t> e_polys(N * limb_count);
    for (size_t i = 0; i < limb_count; i++) {
        const uint64_t qi = moduli[i];
        uint64_t* level_ptr = &e_polys[i * N];
        for (size_t j = 0; j < N; j++) {
            level_ptr[j] = reduce_i64_mod_q(noise_coeffs[j], qi);
        }
    }

    return e_polys;
}



// aproximate function
MyVector<Complex> crt_interpolate(const MyVector<uint64_t>& ptxt, size_t N, MyVector<uint64_t>& moduli){
    
    if (ptxt.size() == 0){
        std::runtime_error("ptxt is empty crt interpolation failed");
    }
    MyVector<Complex> result(N);
    if (ptxt.size() == 1){
        for(size_t i = 0; i < N; ++i) {
            result[i] = Complex(ptxt[0 * N + i], 0.0);
        }    
        return result;  
    }

    const uint128_t max_u128 = static_cast<uint128_t>(~uint128_t(0));
    size_t limbs_to_use = 0;
    uint128_t P = 1;
    for (size_t i = 0; i < moduli.size(); ++i) {
        if (moduli[i] == 0) break;
        if (P > max_u128 / moduli[i]) break; // would overflow
        P *= moduli[i];
        ++limbs_to_use;
    }



	uint128_t P_half = P / 2;

        // Precompute CRT constants for the subset P
    MyVector<uint128_t> P_hat(limbs_to_use);
    MyVector<uint128_t> P_hat_inv(limbs_to_use); // (P/q_i)^-1 mod q_i

    for (size_t i = 0; i < limbs_to_use; ++i) {
        P_hat[i] = P / moduli[i];
        // reduce P_hat mod q_i to compute inverse
        uint64_t p_hat_mod_qi = (uint64_t)(P_hat[i] % moduli[i]);
        uint64_t inv = mod_inverse(p_hat_mod_qi, moduli[i]);
        P_hat_inv[i] = inv; // This fits in uint64
    }

        // --- CRT Interpolation (Subset) ---
        

        for (size_t j = 0; j < N; ++j) {
            uint128_t accum_real = 0;
            for (size_t k = 0; k < limbs_to_use; ++k) {
                uint128_t alpha_real = ((uint128_t)ptxt[k * N + j] * P_hat_inv[k]) % moduli[k];
                uint128_t term_real = alpha_real * P_hat[k]; 
                // term is strictly < P, because alpha < q_i and P_hat = P/q_i.

                const uint128_t remaining = P - accum_real;
                if (term_real >= remaining) {
                    accum_real = term_real - remaining;
                } else {
                    accum_real += term_real;
                }
            }

            // --- Centered Lift (Mod P to Signed Int) ---
            // If accum > P/2, it represents a negative number
            double val_real;
            if (accum_real > P_half) {
                // Negative value: val = accum - P
                // We cast to double carefully
                uint128_t diff = P - accum_real;
                val_real = -1.0 * (double)diff; 
            } else {
                val_real = (double)accum_real;
            }

            // Divide by scale immediately

            result[j] = Complex(val_real, 0);
        }
        
        return result;
}


CKKSEncoding ckks_encode(
    const CKKSContext& context,
    const MyVector<Complex>& values,
    double scale,
    size_t level,
    bool eval) {
    const auto& params = context.getParams();
    const auto N = params.getN();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const size_t limb_count = level + 1;
    if (!(scale > 0.0)) {
        throw std::invalid_argument("ckks_encode: scale must be positive");
    }
    if (level > params.getMaxLevel() || limb_count > moduli.size() || limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_encode: invalid level");
    }

    MyVector<uint64_t> active_moduli(moduli.begin(), moduli.begin() + limb_count);
    auto plaintext = encode_slots_custom_basis(
        N,
        values,
        scale,
        active_moduli,
        twiddle_ntt,
        context.getTwiddleFft(),
        context.getPermTable(),
        eval,
        /*montgomery_form=*/false);
    return CKKSEncoding(N, std::move(plaintext), scale, level, /*montgomery_form=*/false);
}

CKKSEncoding ckks_encode(const CKKSContext& context, const MyVector<Complex>& values, size_t level, bool eval) {
    return ckks_encode(context, values, context.getParams().getScale(), level, eval);
}

MyVector<uint64_t> encode_slots_custom_basis(
    size_t N,
    std::span<const Complex> values,
    double scale,
    const MyVector<uint64_t>& moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt,
    const MyVector<Complex>& twiddle_fft,
    const MyVector<size_t>& perm_table,
    bool eval,
    bool montgomery_form) {
    const size_t half_N = N >> 1;
    if (values.size() != half_N) {
        throw std::invalid_argument("encode_slots_custom_basis: expected exactly N/2 slot values");
    }
    if (!(scale > 0.0)) {
        throw std::invalid_argument("encode_slots_custom_basis: scale must be positive");
    }
    if (moduli.empty() || moduli.size() > twiddle_ntt.size()) {
        throw std::invalid_argument("encode_slots_custom_basis: invalid modulus basis");
    }
    if (twiddle_fft.size() != N) {
        throw std::invalid_argument("encode_slots_custom_basis: FFT twiddle table size mismatch");
    }

    MyVector<Complex> evals(N);
    make_hermitian_evals(values, evals, perm_table, N);
    fft_dif(evals, twiddle_fft, N);

    MyVector<uint64_t> plaintext(moduli.size() * N);
    const double scaled_inv_N = scale / static_cast<double>(N);
    for (size_t q_index = 0; q_index < moduli.size(); ++q_index) {
        const uint64_t q = moduli[q_index];
        uint64_t* __restrict limb_ptr = plaintext.data() + q_index * N;
        for (size_t i = 0; i < N; ++i) {
            const int64_t val = std::llrint(evals[i].real() * scaled_inv_N);
            limb_ptr[i] = double_to_uint64(val, q);
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

CKKSEncoding ckks_encode_coefficients(
    const CKKSContext& context,
    std::span<const Complex> values,
    double scale,
    size_t level,
    bool eval) {
    const auto& params = context.getParams();
    const auto N = params.getN();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const size_t limb_count = level + 1;
    const size_t slots = N >> 1;

    if (values.size() != slots) {
        throw std::invalid_argument("ckks_encode_coefficients: expected exactly N/2 slot values");
    }
    if (!(scale > 0.0)) {
        throw std::invalid_argument("ckks_encode_coefficients: scale must be positive");
    }
    if (level > params.getMaxLevel() || limb_count > moduli.size() || limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_encode_coefficients: invalid level");
    }

    MyVector<uint64_t> plaintext(limb_count * N, uint64_t{0});
    const size_t log_slots = std::countr_zero(slots);
    for (size_t q_index = 0; q_index < limb_count; ++q_index) {
        const uint64_t q = moduli[q_index];
        uint64_t* __restrict limb_ptr = plaintext.data() + q_index * N;
        for (size_t i = 0; i < slots; ++i) {
            // Match the the homomorphic DFT coefficient layout
            // without exposing bit-reversed order to callers.
            const size_t coeff_index = bit_reverse_value(i, log_slots);
            limb_ptr[coeff_index] = double_to_uint64(
                std::llrint(values[i].real() * scale),
                q);
            limb_ptr[slots + coeff_index] = double_to_uint64(
                std::llrint(values[i].imag() * scale),
                q);
        }
    }

    if (eval) {
        ntt_forward_rns_flat_inplace(plaintext.data(), N, limb_count, moduli, twiddle_ntt, false);
    }

    return CKKSEncoding(N, std::move(plaintext), scale, level, /*montgomery_form=*/false);
}


CKKSCiphertext ckks_encrypt(
    const CKKSContext& context,
    const MyVector<Complex>& input,
    size_t level,
    bool eval,
    CKKSMessageEncodingState message_encoding_state) {
    CKKSEncoding encd =
        message_encoding_state == CKKSMessageEncodingState::Coefficients
            ? ckks_encode_coefficients(context, input, context.getParams().getScale(), level, eval)
            : ckks_encode(context, input, level, eval);
    return ckks_encrypt(context, encd, eval, message_encoding_state);
}

CKKSCiphertext ckks_encrypt_custom_eval(
    size_t N,
    MyVector<uint64_t> plaintext_eval,
    const MyVector<MyVector<uint64_t>>& secret_key_eval,
    const MyVector<uint64_t>& moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt,
    double sigma,
    long double scale,
    size_t level) {
    const size_t limb_count = moduli.size();
    if (limb_count == 0 || limb_count > secret_key_eval.size() || limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_encrypt_custom_eval: invalid custom basis");
    }
    if (plaintext_eval.size() != N * limb_count) {
        throw std::invalid_argument("ckks_encrypt_custom_eval: invalid plaintext buffer size");
    }
    if (level + 1 > limb_count) {
        throw std::invalid_argument("ckks_encrypt_custom_eval: invalid ciphertext level for supplied basis");
    }

    auto a_polys = ckks_uniform_mask_polys(N, moduli, limb_count);
    auto e_polys = ckks_gaussian_noise_polys(N, moduli, limb_count, sigma);
    ntt_forward_rns_flat_inplace(e_polys.data(), N, limb_count, moduli, twiddle_ntt, false);

    MyVector<uint64_t> b_poly(N * limb_count);
#pragma omp parallel for schedule(static) if (limb_count > 1)
    for (std::ptrdiff_t limb = 0; limb < static_cast<std::ptrdiff_t>(limb_count); ++limb) {
        const size_t i = static_cast<size_t>(limb);
        const uint64_t q = moduli[i];
        uint64_t* __restrict a_ptr = a_polys.data() + i * N;
        const uint64_t* __restrict s_ptr = secret_key_eval[i].data();
        uint64_t* __restrict e_ptr = e_polys.data() + i * N;
        uint64_t* __restrict m_ptr = plaintext_eval.data() + i * N;
        uint64_t* __restrict b_ptr = b_poly.data() + i * N;

        pointwise_multiply_dispatch(a_ptr, s_ptr, b_ptr, N, q,
                                    twiddle_ntt[i]);
        poly_add_modq_inplace(m_ptr, e_ptr, N, q);
        poly_sub_modq_inplace(m_ptr, b_ptr, N, q);
    }

    return CKKSCiphertext(N, std::move(a_polys), std::move(plaintext_eval), scale, level);
}

CKKSCiphertext ckks_encrypt(
    const CKKSContext& context,
    const CKKSEncoding& encd,
    bool eval,
    CKKSMessageEncodingState message_encoding_state) {
    auto ptxt = encd.getPlaintext();
    const auto& sk = context.getSecretKey();
    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const auto N = params.getN();
    const auto sigma = params.getSigma();
    const size_t level = encd.getLevel();
    const size_t limb_count = level + 1;

    if (encd.getN() != N) {
        throw std::invalid_argument("ckks_encrypt(ptxt): plaintext/context size mismatch");
    }
    if (level > params.getMaxLevel() ||
        limb_count > moduli.size() ||
        limb_count > sk.size() ||
        limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_encrypt(ptxt): invalid level");
    }
    if (ptxt.size() != N * limb_count) {
        throw std::invalid_argument("ckks_encrypt(ptxt): invalid plaintext buffer size");
    }

    MyVector<uint64_t> active_moduli(moduli.begin(), moduli.begin() + limb_count);
    if (encd.isMontgomeryForm()) {
        ckks_convert_plaintext_from_montgomery_inplace(ptxt, N, active_moduli);
    }
    if (!eval) {
        ntt_forward_rns_flat_inplace(ptxt.data(), N, limb_count, moduli, twiddle_ntt, false);
    }

    auto ct = ckks_encrypt_custom_eval(
        N,
        std::move(ptxt),
        sk,
        active_moduli,
        twiddle_ntt,
        sigma,
        encd.getScale(),
        level);
    ct.setMessageEncodingState(message_encoding_state);
    return ct;
}

MyVector<Complex> ckks_decrypt(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    CKKSDecryptDomain domain,
    bool use_low_secret) {
    const bool decrypt_with_sparse_secret =
        use_low_secret || ct.getSecretOwner() == CKKSSecretOwner::BootstrapSparse;
    const auto& sk = decrypt_with_sparse_secret ? context.getLowSecretKey() : context.getSecretKey();
    return ckks_decrypt_with_secret_eval(context, ct, sk, domain);
}
