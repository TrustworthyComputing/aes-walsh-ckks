#include "ckks_conjugate.hpp"

#include <cstddef>
#include <stdexcept>

#include "ckks_keyswitch.hpp"

CKKSCiphertext ckks_conjugate(
    const CKKSContext& context,
    const CKKSCiphertext& ct) {
    auto conjugated = ct.clone();
    ckks_conjugate_inplace(context, conjugated);
    return conjugated;
}

void ckks_conjugate_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct) {
    const auto& params = context.getParams();
    const size_t N = params.getN();

    if (!context.hasConjugationKey()) {
        throw std::invalid_argument("ckks_conjugate: conjugation key is unavailable");
    }
    if (ct.getNumPolys() != 2) {
        throw std::invalid_argument("ckks_conjugate: expected a 2-polynomial ciphertext");
    }
    if (ct.getN() != N) {
        throw std::invalid_argument("ckks_conjugate: ciphertext/context size mismatch");
    }
    if (ct.getSecretOwner() != CKKSSecretOwner::Dense) {
        throw std::invalid_argument("ckks_conjugate: sparse-secret conjugation keys are not available");
    }

    const size_t level = ct.getLevel();
    const size_t limb_count = level + 1;
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    if (level > params.getMaxLevel() || limb_count > moduli.size() || limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_conjugate: invalid ciphertext level");
    }

    const auto& ntt_map = context.getConjugationNttMap();
    const auto switching_key = context.getConjugationKey();

    for (size_t poly_idx = 0; poly_idx < ct.getNumPolys(); ++poly_idx) {
        ckks_apply_ntt_automorphism_to_flat_rns_poly_inplace(
            ct.getPolyMutable(poly_idx), N, limb_count, ntt_map);
    }

    ckks_hybrid_key_switch_inplace(
        context,
        level,
        ct.getAMutable(),
        switching_key,
        ct.getBMutable(),
        ct.getAMutable(),
        CKKSHybridKeySwitchCombine::Add,
        CKKSHybridKeySwitchCombine::Assign,
        params.montgomery);
}
