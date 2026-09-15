#ifndef CKKS_HYBRID_KEY_MONTGOMERY_HPP
#define CKKS_HYBRID_KEY_MONTGOMERY_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "ckks_ciphertext.hpp"
#include "modarith.hpp"
#include "ntt.hpp"

inline void ckks_convert_switch_key_to_montgomery_inplace(
    CKKSCiphertext& key,
    const MyVector<uint64_t>& moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& twiddles) {
    if (moduli.size() > twiddles.size()) {
        throw std::invalid_argument("ckks_context: invalid Montgomery key conversion basis");
    }
    const size_t N = key.getN();
    const size_t limb_count = moduli.size();
    for (auto& poly : key.getPolysMutable()) {
        if (poly.size() != N * limb_count) {
            throw std::invalid_argument("ckks_context: invalid Montgomery key polynomial size");
        }
        for (size_t limb = 0; limb < limb_count; ++limb) {
            const uint64_t q = moduli[limb];
            const uint64_t r_mod_q = ckks_montgomery_radix_mod_q(q);
            pointwise_multiply_scalar_inplace_dispatch(
                poly.data() + limb * N,
                r_mod_q,
                N,
                q,
                twiddles[limb]);
        }
    }
}

#endif // CKKS_HYBRID_KEY_MONTGOMERY_HPP
