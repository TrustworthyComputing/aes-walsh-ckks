#ifndef CKKS_HYBRID_KEY_FOLDING_HPP
#define CKKS_HYBRID_KEY_FOLDING_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "ckks_ciphertext.hpp"
#include "modarith.hpp"
#include "ntt.hpp"

inline uint64_t ckks_product_mod(const MyVector<uint64_t>& values,
                                 uint64_t modulus,
                                 const BarrettConst* barrett) {
    uint64_t out = 1;
    for (uint64_t value : values) {
        out = mul_mod_u64(out, value % modulus, modulus, barrett);
    }
    return out;
}

inline uint64_t ckks_product_except_mod(const MyVector<uint64_t>& values,
                                        size_t skip,
                                        uint64_t modulus,
                                        const BarrettConst* barrett) {
    uint64_t out = 1;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i == skip) {
            continue;
        }
        out = mul_mod_u64(out, values[i] % modulus, modulus, barrett);
    }
    return out;
}

inline void ckks_scale_ciphertext_limb(CKKSCiphertext& ct,
                                       size_t limb,
                                       uint64_t scalar,
                                       uint64_t modulus,
                                       const Negacyclic_NTT_Twiddles& table) {
    for (auto& poly : ct.getPolysMutable()) {
        if ((limb + 1) * ct.getN() > poly.size()) {
            throw std::invalid_argument("ckks_context: invalid folded hybrid key limb");
        }
        pointwise_multiply_scalar_inplace_dispatch(
            poly.data() + limb * ct.getN(), scalar, ct.getN(), modulus, table);
    }
}

inline void ckks_fold_hybrid_switch_key_constants(
    CKKSCiphertext& key,
    const MyVector<uint64_t>& q_moduli,
    const MyVector<uint64_t>& p_moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& qup_twiddles) {
    const size_t q_limb_count = q_moduli.size();
    if (qup_twiddles.size() != q_limb_count + p_moduli.size()) {
        throw std::invalid_argument("ckks_context: invalid folded hybrid key basis");
    }

    for (size_t i = 0; i < q_limb_count; ++i) {
        const uint64_t q = q_moduli[i];
        const auto* barrett = &qup_twiddles[i].barrett_const;
        const uint64_t p_mod_q = ckks_product_mod(p_moduli, q, barrett);
        const uint64_t inv_p_mod_q = mod_inverse(p_mod_q, q);
        ckks_scale_ciphertext_limb(key, i, inv_p_mod_q, q, qup_twiddles[i]);
    }

    for (size_t pi = 0; pi < p_moduli.size(); ++pi) {
        const size_t limb = q_limb_count + pi;
        const uint64_t p = p_moduli[pi];
        const auto* barrett = &qup_twiddles[limb].barrett_const;
        const uint64_t phat_mod_p = ckks_product_except_mod(p_moduli, pi, p, barrett);
        const uint64_t inv_phat_mod_p = mod_inverse(phat_mod_p, p);
        ckks_scale_ciphertext_limb(key, limb, inv_phat_mod_p, p,
                                   qup_twiddles[limb]);
    }
}

#endif  // CKKS_HYBRID_KEY_FOLDING_HPP
