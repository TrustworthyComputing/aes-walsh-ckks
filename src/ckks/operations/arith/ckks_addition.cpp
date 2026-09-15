#include <cstddef>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>


#include "ckks_addition.hpp"
#include "ckks_ciphertext.hpp"
#include "ckks_encoding.hpp"
#include "modarith.hpp"
#include "polynomial.hpp"

namespace {

uint64_t rounded_scaled_integer_mod_q(int64_t value, long double scale, uint64_t q) {
    const long double scaled = static_cast<long double>(value) * scale;
    if (!std::isfinite(scaled)) {
        throw std::invalid_argument("ckks_add_integer_constant_inplace: non-finite scaled constant");
    }
    long double residue = std::fmod(std::round(scaled), static_cast<long double>(q));
    if (residue < 0.0L) {
        residue += static_cast<long double>(q);
    }
    auto out = static_cast<uint64_t>(residue);
    if (out >= q) {
        out %= q;
    }
    return out;
}

}  // namespace

void ckks_add_inplace(const CKKSContext& context, CKKSCiphertext& ct1, const CKKSEncoding& ct2){
    auto& b = ct1.getBMutable();
    const auto N = ct1.getN();
    const auto level = ct1.getLevel();
    const auto limb_count = level + 1;
    const auto& moduli = context.getParams().getModuli();
    MyVector<uint64_t> pt_normal;
    const MyVector<uint64_t>* pt = &ct2.getPlaintext();

    assert(ct1.getN() == ct2.getN());
    assert(ct1.getLevel() == ct2.getLevel());

    if (ct2.isMontgomeryForm()) {
        pt_normal = ct2.getPlaintext();
        MyVector<uint64_t> active_moduli(moduli.begin(), moduli.begin() + limb_count);
        ckks_convert_plaintext_from_montgomery_inplace(pt_normal, N, active_moduli);
        pt = &pt_normal;
    }

    for (size_t l = 0; l < limb_count; l++){
        auto b_ptr = b.data() + l * N;
        auto pt_ptr = pt->data() + l * N;
        poly_add_modq_inplace(b_ptr, pt_ptr, N, moduli[l]);
    }
}

void ckks_add_inplace(const CKKSContext& context, CKKSCiphertext& ct1, const CKKSCiphertext& ct2){
    const auto N = ct1.getN();
    const auto level = ct1.getLevel();
    const auto limb_count = level + 1;
    const auto& moduli = context.getParams().getModuli();
    const size_t polys1 = ct1.getNumPolys();

    assert(ct1.getN() == ct2.getN());
    assert(ct1.getLevel() == ct2.getLevel());
    assert(polys1 == ct2.getNumPolys());
    if (ct1.getSecretOwner() != ct2.getSecretOwner() ||
        ct1.getMessageEncodingState() != ct2.getMessageEncodingState()) {
        throw std::invalid_argument("ckks_add_inplace(ct,ct): ciphertext metadata mismatch");
    }

    for (size_t poly_idx = 0; poly_idx < polys1; ++poly_idx) {
        auto& p1 = ct1.getPolyMutable(poly_idx);
        const auto& p2 = ct2.getPoly(poly_idx);
        for (size_t l = 0; l < limb_count; l++){
            auto p1_ptr = p1.data() + l * N;
            auto p2_ptr = p2.data() + l * N;
            poly_add_modq_inplace(p1_ptr, p2_ptr, N, moduli[l]);
        }
    }
}

void ckks_add_integer_constant_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    int64_t constant) {
    if (constant == 0) {
        return;
    }

    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const size_t N = ct.getN();
    const size_t level = ct.getLevel();
    if (N != params.getN() || level > params.getMaxLevel()) {
        throw std::invalid_argument("ckks_add_integer_constant_inplace: invalid ciphertext");
    }

    auto& b = ct.getBMutable();
    for (size_t limb = 0; limb <= level; ++limb) {
        const uint64_t q = moduli[limb];
        const uint64_t scalar_mod_q =
            rounded_scaled_integer_mod_q(constant, ct.getScale(), q);
        uint64_t* __restrict b_ptr = b.data() + limb * N;
        for (size_t i = 0; i < N; ++i) {
            b_ptr[i] = add_mod_q(b_ptr[i], scalar_mod_q, q);
        }
    }
}

void ckks_sub_inplace(const CKKSContext& context, CKKSCiphertext& ct1, const CKKSCiphertext& ct2){
    const auto N = ct1.getN();
    const auto level = ct1.getLevel();
    const auto limb_count = level + 1;
    const auto& moduli = context.getParams().getModuli();
    const size_t polys1 = ct1.getNumPolys();

    assert(ct1.getN() == ct2.getN());
    assert(ct1.getLevel() == ct2.getLevel());
    assert(polys1 == ct2.getNumPolys());
    if (ct1.getSecretOwner() != ct2.getSecretOwner() ||
        ct1.getMessageEncodingState() != ct2.getMessageEncodingState()) {
        throw std::invalid_argument("ckks_sub_inplace(ct,ct): ciphertext metadata mismatch");
    }

    for (size_t poly_idx = 0; poly_idx < polys1; ++poly_idx) {
        auto& p1 = ct1.getPolyMutable(poly_idx);
        const auto& p2 = ct2.getPoly(poly_idx);
        for (size_t l = 0; l < limb_count; l++){
            auto p1_ptr = p1.data() + l * N;
            auto p2_ptr = p2.data() + l * N;
            poly_sub_modq_inplace(p1_ptr, p2_ptr, N, moduli[l]);
        }
    }
}
