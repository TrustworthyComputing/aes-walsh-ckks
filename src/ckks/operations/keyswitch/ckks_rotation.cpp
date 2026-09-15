#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "ckks_keyswitch.hpp"
#include "ckks_rotation.hpp"
#include "modarith.hpp"
#include "ntt.hpp"

namespace {

void apply_ntt_automorphism_to_rns_poly(
    MyVector<uint64_t>& poly,
    size_t N,
    size_t limb_count,
    const MyVector<size_t>& dst_to_src_map) {
    if (poly.size() != N * limb_count || dst_to_src_map.size() != N) {
        throw std::invalid_argument("ckks_rotate: invalid ciphertext polynomial size");
    }

    MyVector<uint64_t> out(poly.size());
    for (size_t limb = 0; limb < limb_count; ++limb) {
        const uint64_t* __restrict src = poly.data() + limb * N;
        uint64_t* __restrict dst = out.data() + limb * N;
        for (size_t i = 0; i < N; ++i) {
            dst[i] = src[dst_to_src_map[i]];
        }
    }
    poly.swap(out);
}

} // namespace

CKKSCiphertext ckks_rotate(const CKKSContext& context, const CKKSCiphertext& ct, int64_t rotation) {
    auto rotated = ct.clone();
    ckks_rotate_inplace(context, rotated, rotation);
    return rotated;
}

void ckks_rotate_inplace(const CKKSContext& context, CKKSCiphertext& ct, int64_t rotation) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t slots = params.getSlots();
    if (slots == 0) {
        throw std::invalid_argument("ckks_rotate: invalid parameter set");
    }

    const int64_t slots_i64 = static_cast<int64_t>(slots);
    int64_t normalized = rotation % slots_i64;
    if (normalized < 0) {
        normalized += slots_i64;
    }
    if (normalized == 0) {
        return;
    }
    if (!context.hasRotationKey(normalized)) {
        throw std::invalid_argument("ckks_rotate: rotation key is unavailable for the requested rotation");
    }
    if (ct.getNumPolys() != 2) {
        throw std::invalid_argument("ckks_rotate: expected a 2-polynomial ciphertext");
    }
    if (ct.getN() != N) {
        throw std::invalid_argument("ckks_rotate: ciphertext/context size mismatch");
    }
    if (ct.getSecretOwner() != CKKSSecretOwner::Dense) {
        throw std::invalid_argument("ckks_rotate: sparse-secret rotation keys are not available");
    }

    const size_t level = ct.getLevel();
    const size_t limb_count = level + 1;
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    if (level > params.getMaxLevel() || limb_count > moduli.size() || limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_rotate: invalid ciphertext level");
    }

    // Rotation keys switch from s to pi^-1(s); applying pi after the switch returns to s.
    const auto switching_key = context.getRotationKey(normalized);

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

    const auto& ntt_map = context.getRotationNttMap(normalized);
    for (size_t poly_idx = 0; poly_idx < ct.getNumPolys(); ++poly_idx) {
        apply_ntt_automorphism_to_rns_poly(ct.getPolyMutable(poly_idx), N, limb_count, ntt_map);
    }
}
