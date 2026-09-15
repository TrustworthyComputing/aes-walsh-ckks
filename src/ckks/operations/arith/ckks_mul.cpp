#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "ckks_mul.hpp"
#include "ckks_context.hpp"
#include "ckks_encoding.hpp"
#include "modarith.hpp"
#include "ntt.hpp"
#include "polynomial.hpp"

namespace {
void pointwise_multiply_montgomery_plaintext_inplace(
    uint64_t* __restrict poly_normal,
    const uint64_t* __restrict plaintext_montgomery,
    size_t N,
    uint64_t q,
    uint64_t q_neg_inv) {
    pointwise_multiply_montgomery_inplace_dispatch(
        poly_normal, plaintext_montgomery, N, q, q_neg_inv);
}
}  // namespace



CKKSCiphertext ckks_mul(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    Complex value,
    double plaintext_scale) {
    auto out = ct.clone();
    ckks_mul_inplace(context, out, value, plaintext_scale);
    return out;
}

void ckks_square_inplace(const CKKSContext& context, CKKSCiphertext& ct) {
    const auto N = ct.getN();
    const auto level = ct.getLevel();
    const auto limb_count = level + 1;
    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();

    if (N != params.getN()) {
        throw std::invalid_argument("ckks_square_inplace: ciphertext/context size mismatch");
    }
    if (ct.getNumPolys() != 2) {
        throw std::invalid_argument("ckks_square_inplace: expected 2-component input");
    }
    if (level > params.getMaxLevel() ||
        limb_count > moduli.size() ||
        limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_square_inplace: invalid level");
    }

    const auto orig_a = ct.getA();
    const auto orig_b = ct.getB();

    ct.addPoly(MyVector<uint64_t>(N * limb_count));
    auto& out_a = ct.getAMutable();
    auto& out_b = ct.getBMutable();
    auto& out_c = ct.getPolyMutable(2);

    for (size_t limb = 0; limb < limb_count; ++limb) {
        const uint64_t q = moduli[limb];
        const auto& tbl = twiddle_ntt[limb];

        const uint64_t* __restrict a_ptr = orig_a.data() + limb * N;
        const uint64_t* __restrict b_ptr = orig_b.data() + limb * N;
        uint64_t* __restrict out_a_ptr = out_a.data() + limb * N;
        uint64_t* __restrict out_b_ptr = out_b.data() + limb * N;
        uint64_t* __restrict out_c_ptr = out_c.data() + limb * N;

        pointwise_multiply_dispatch(a_ptr, a_ptr, out_c_ptr, N, q, tbl);
        pointwise_multiply_dispatch(b_ptr, b_ptr, out_b_ptr, N, q, tbl);
        pointwise_multiply_dispatch(a_ptr, b_ptr, out_a_ptr, N, q, tbl);
        pointwise_multiply_scalar_inplace_dispatch(
            out_a_ptr,
            2,
            N,
            q,
            tbl,
            params.montgomery);
    }

    ct.setScale(ct.getScale() * ct.getScale());
}

void ckks_mul_inplace(const CKKSContext& context, CKKSCiphertext& ct1, const CKKSCiphertext& ct2){
    const auto N = ct1.getN();
    const auto level = ct1.getLevel();
    const auto limb_count = level + 1;
    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();

    if (N != ct2.getN() || level != ct2.getLevel()) {
        throw std::invalid_argument("ckks_mul_inplace(ct,ct): ciphertext size mismatch");
    }
    if (N != params.getN()) {
        throw std::invalid_argument("ckks_mul_inplace(ct,ct): ciphertext/context size mismatch");
    }
    if (ct1.getNumPolys() != 2 || ct2.getNumPolys() != 2) {
        throw std::invalid_argument("ckks_mul_inplace(ct,ct): expected 2-component inputs");
    }
    if (ct1.getSecretOwner() != ct2.getSecretOwner() ||
        ct1.getMessageEncodingState() != ct2.getMessageEncodingState()) {
        throw std::invalid_argument("ckks_mul_inplace(ct,ct): ciphertext metadata mismatch");
    }
    if (level > params.getMaxLevel() || limb_count > moduli.size() || limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_mul_inplace(ct,ct): invalid level");
    }

    // Allocate the quadratic component directly inside ct1 so we don't need a
    // separate temporary vector for c2.
    ct1.addPoly(MyVector<uint64_t>(N * limb_count));
    auto& c2 = ct1.getPolyMutable(2);

    // Overwrite (a1,b1) with (c1,c0) and write c2 in the same pass.
    auto& a1 = ct1.getAMutable();
    auto& b1 = ct1.getBMutable();
    const auto& a2 = ct2.getA();
    const auto& b2 = ct2.getB();
    MyVector<uint64_t> tmp(N);

    for (size_t i = 0; i < limb_count; ++i) {
        const uint64_t q = moduli[i];
        const auto& tbl = twiddle_ntt[i];

        uint64_t* __restrict a1_ptr = a1.data() + i * N;
        uint64_t* __restrict b1_ptr = b1.data() + i * N;
        const uint64_t* __restrict a2_ptr = a2.data() + i * N;
        const uint64_t* __restrict b2_ptr = b2.data() + i * N;
        uint64_t* __restrict c2_ptr = c2.data() + i * N;

        // Convention: poly[0] = constant term (b), poly[1] = linear term, poly[2] = quadratic term.
        // After this block: b1 <- c0, a1 <- c1.

        // c2 = a1 * a2
        pointwise_multiply_dispatch(a1_ptr, a2_ptr, c2_ptr, N, q, tbl);

        // tmp = b1 * a2
        pointwise_multiply_dispatch(b1_ptr, a2_ptr, tmp.data(), N, q, tbl);

        // b1 = b1 * b2  (c0)
        pointwise_multiply_inplace_dispatch(b1_ptr, b2_ptr, N, q, tbl);

        // a1 = a1 * b2  (partial c1)
        pointwise_multiply_inplace_dispatch(a1_ptr, b2_ptr, N, q, tbl);

        // a1 += tmp  (finish c1)
        poly_add_modq_inplace(a1_ptr, tmp.data(), N, q);
    }

    ct1.setScale(ct1.getScale() * ct2.getScale());
}


void ckks_mul_inplace(const CKKSContext& context, CKKSCiphertext& ct1, const CKKSEncoding& ct2){
    const auto& ptxt2 = ct2.getPlaintext();
    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const auto N = ct1.getN();
    const auto level = ct1.getLevel();
    const auto limb_count = level + 1;

    if (N != ct2.getN() || level != ct2.getLevel()) {
        throw std::invalid_argument("ckks_mul_inplace: ciphertext/plaintext size mismatch");
    }
    if (N != params.getN()) {
        throw std::invalid_argument("ckks_mul_inplace: ciphertext/context size mismatch");
    }
    if (level > params.getMaxLevel() || limb_count > moduli.size() || limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_mul_inplace: invalid level");
    }
    if (ptxt2.size() != N * limb_count) {
        throw std::invalid_argument("ckks_mul_inplace: invalid plaintext buffer size");
    }

    const auto polys = ct1.getNumPolys();
    if (polys == 0) {
        throw std::invalid_argument("ckks_mul_inplace: empty ciphertext");
    }

    for (size_t poly_idx = 0; poly_idx < polys; ++poly_idx) {
        auto& poly = ct1.getPolyMutable(poly_idx);
        if (poly.size() != N * limb_count) {
            throw std::invalid_argument("ckks_mul_inplace: invalid ciphertext buffer size");
        }

        for (size_t i = 0; i < limb_count; ++i) {
            const uint64_t q = moduli[i];
            const auto& tbl = twiddle_ntt[i];

            uint64_t* __restrict poly_ptr = poly.data() + i * N;
            const uint64_t* __restrict ptxt2_ptr = ptxt2.data() + i * N;
            if (ct2.isMontgomeryForm()) {
                pointwise_multiply_montgomery_plaintext_inplace(
                    poly_ptr,
                    ptxt2_ptr,
                    N,
                    q,
                    ckks_montgomery_neg_inverse(q));
            } else {
                pointwise_multiply_inplace_dispatch(poly_ptr, ptxt2_ptr, N, q,
                                                    tbl);
            }
        }
    }

    ct1.setScale(ct1.getScale() * ct2.getScale());
}

void ckks_mul_inplace(const CKKSContext& context, CKKSCiphertext& ct, int64_t scalar) {
    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const auto N = ct.getN();
    const auto level = ct.getLevel();
    const auto limb_count = level + 1;
    const auto polys = ct.getNumPolys();

    if (N != params.getN()) {
        throw std::invalid_argument("ckks_mul_inplace(ct,int): ciphertext/context size mismatch");
    }
    if (level > params.getMaxLevel() ||
        limb_count > moduli.size() ||
        limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_mul_inplace(ct,int): invalid level");
    }
    if (polys == 0) {
        throw std::invalid_argument("ckks_mul_inplace(ct,int): empty ciphertext");
    }

    if (scalar == 1) {
        return;
    }

    for (size_t poly_idx = 0; poly_idx < polys; ++poly_idx) {
        auto& poly = ct.getPolyMutable(poly_idx);
        if (poly.size() != N * limb_count) {
            throw std::invalid_argument("ckks_mul_inplace(ct,int): invalid ciphertext buffer size");
        }
    }

    if (scalar == 0) {
        for (size_t poly_idx = 0; poly_idx < polys; ++poly_idx) {
            auto& poly = ct.getPolyMutable(poly_idx);
            std::fill(poly.begin(), poly.end(), uint64_t{0});
        }
        return;
    }

    for (size_t limb = 0; limb < limb_count; ++limb) {
        const uint64_t q = moduli[limb];
        const uint64_t scalar_mod_q = reduce_i64_mod_q(scalar, q);
        const auto& tbl = twiddle_ntt[limb];

        for (size_t poly_idx = 0; poly_idx < polys; ++poly_idx) {
            auto& poly = ct.getPolyMutable(poly_idx);
            uint64_t* __restrict poly_ptr = poly.data() + limb * N;
            pointwise_multiply_scalar_inplace_dispatch(
                poly_ptr,
                scalar_mod_q,
                N,
                q,
                tbl,
                params.montgomery);
        }
    }
}

void ckks_mul_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    Complex value,
    double plaintext_scale) {
    MyVector<Complex> slots(context.getParams().getSlots(), value);
    auto plaintext = ckks_encode(
        context,
        slots,
        plaintext_scale,
        ct.getLevel(),
        /*eval=*/true);
    ckks_mul_inplace(context, ct, plaintext);
}
