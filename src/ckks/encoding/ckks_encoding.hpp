#ifndef CKKS_ENCODING_HPP
#define CKKS_ENCODING_HPP

#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

#include <sys/types.h>
#include <complex>
#include <cstddef>
#include <span>
#include "aligned_vector.hpp"

using Complex = std::complex<double>;

MyVector<uint64_t> ckks_uniform_mask_polys(size_t N, const MyVector<uint64_t>& moduli, size_t limb_count);
MyVector<uint64_t> ckks_gaussian_noise_polys(size_t N,
                                             const MyVector<uint64_t>& moduli,
                                             size_t limb_count,
                                             double sigma);

CKKSEncoding ckks_encode(const CKKSContext& context, const MyVector<Complex>& values, size_t level, bool eval = false);
CKKSEncoding ckks_encode(
    const CKKSContext& context,
    const MyVector<Complex>& values,
    double scale,
    size_t level,
    bool eval = false);
void ckks_convert_plaintext_to_montgomery_inplace(
    MyVector<uint64_t>& plaintext,
    size_t N,
    const MyVector<uint64_t>& moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt);
void ckks_convert_plaintext_from_montgomery_inplace(
    MyVector<uint64_t>& plaintext,
    size_t N,
    const MyVector<uint64_t>& moduli);
MyVector<uint64_t> encode_slots_custom_basis(
    size_t N,
    std::span<const Complex> values,
    double scale,
    const MyVector<uint64_t>& moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt,
    const MyVector<Complex>& twiddle_fft,
    const MyVector<size_t>& perm_table,
    bool eval = true,
    bool montgomery_form = false);
CKKSEncoding ckks_encode_coefficients(
    const CKKSContext& context,
    std::span<const Complex> values,
    double scale,
    size_t level,
    bool eval = false);
CKKSCiphertext ckks_encrypt(
    const CKKSContext& context,
    const MyVector<Complex>& values,
    size_t level,
    bool eval = false,
    CKKSMessageEncodingState message_encoding_state = CKKSMessageEncodingState::Slots);
CKKSCiphertext ckks_encrypt(
    const CKKSContext& context,
    const CKKSEncoding& encd,
    bool eval = false,
    CKKSMessageEncodingState message_encoding_state = CKKSMessageEncodingState::Slots);
CKKSCiphertext ckks_encrypt_custom_eval(
    size_t N,
    MyVector<uint64_t> plaintext_eval,
    const MyVector<MyVector<uint64_t>>& secret_key_eval,
    const MyVector<uint64_t>& moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddle_ntt,
    double sigma,
    long double scale,
    size_t level);
MyVector<Complex> ckks_decrypt(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    CKKSDecryptDomain domain = CKKSDecryptDomain::Slots,
    bool use_low_secret = false);

#endif // CKKS_ENCODING_HPP
