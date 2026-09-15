#include <cstddef>
#include <stdexcept>

#include "ckks_context.hpp"
#include "ckks_keyswitch.hpp"
#include "ckks_relin.hpp"

void ckks_relin_hybrid_inplace(const CKKSContext& context, CKKSCiphertext& ct){
    const auto& params = context.getParams();
    const auto N = params.getN();
    const auto level = ct.getLevel();
    const size_t limb_count = level + 1;
    const auto& q_moduli = params.getModuli();
    const auto num_polys = ct.getNumPolys();
    const auto rlk_key = context.getRelinHybrid(level);

    if (level > params.getMaxLevel() || limb_count > q_moduli.size()) {
        throw std::invalid_argument("ckks_relin: invalid ciphertext level");
    }
    if (num_polys != 3) {
        throw std::invalid_argument("ckks_relin: expected 3-polynomial ciphertext");
    }
    if (ct.getSecretOwner() != CKKSSecretOwner::Dense) {
        throw std::invalid_argument("ckks_relin: sparse-secret relinearization keys are not available");
    }
    const size_t active_stride = N * limb_count;
    if (ct.getPoly(0).size() != active_stride ||
        ct.getPoly(1).size() != active_stride ||
        ct.getPoly(2).size() != active_stride) {
        throw std::invalid_argument("ckks_relin: ciphertext polynomial size does not match level");
    }
    ckks_hybrid_key_switch_inplace(
        context,
        level,
        ct.getPolyMutable(2),
        rlk_key,
        ct.getPolyMutable(0),
        ct.getPolyMutable(1),
        CKKSHybridKeySwitchCombine::Add,
        CKKSHybridKeySwitchCombine::Add,
        params.montgomery);

    ct.getPolysMutable().resize(2);
}
