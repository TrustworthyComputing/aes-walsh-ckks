#include "ckks_rescale.hpp"

#include <cstdint>
#include <stdexcept>

#include "modarith.hpp"
#include "ntt.hpp"

namespace {

void compute_rescale_remainder_scalar(uint64_t* __restrict rem,
                                      const uint64_t* __restrict drop,
                                      size_t N,
                                      uint64_t p,
                                      uint64_t half_p,
                                      uint64_t q,
                                      uint64_t one_shoup) {
    for (size_t j = 0; j < N; ++j) {
        uint64_t shift = drop[j] + half_p;
        if (shift >= p) shift -= p;
        rem[j] = reduce_u64_mod_q_shoup(shift, q, one_shoup);
    }
}

void finish_rescale_limb_scalar(uint64_t* __restrict poly,
                                const uint64_t* __restrict half_ntt,
                                const uint64_t* __restrict rem,
                                size_t N,
                                uint64_t q,
                                uint64_t inv_p,
                                uint64_t inv_p_shoup) {
    for (size_t j = 0; j < N; ++j) {
        const uint64_t num = add_mod_q(poly[j], half_ntt[j], q);
        const uint64_t diff = sub_mod_q(num, rem[j], q);
        poly[j] = mul_mod_shoup(diff, inv_p, inv_p_shoup, q);
    }
}

}  // namespace

void ckks_drop_to_level_inplace(CKKSCiphertext& ct, size_t target_level) {
    const size_t level = ct.getLevel();
    if (target_level > level) {
        throw std::invalid_argument("ckks_drop_to_level_inplace: cannot raise ciphertext level");
    }
    if (target_level == level) {
        return;
    }

    const size_t N = ct.getN();
    const size_t current_size = (level + 1) * N;
    const size_t target_size = (target_level + 1) * N;
    const size_t polys = ct.getNumPolys();
    if (polys == 0) {
        throw std::invalid_argument("ckks_drop_to_level_inplace: empty ciphertext");
    }
    for (size_t poly_idx = 0; poly_idx < polys; ++poly_idx) {
        auto& poly = ct.getPolyMutable(poly_idx);
        if (poly.size() != current_size) {
            throw std::invalid_argument(
                "ckks_drop_to_level_inplace: invalid ciphertext buffer size");
        }
        poly.resize(target_size);
    }
    ct.setLevel(target_level);
}

void ckks_rescale(const CKKSContext& context, CKKSCiphertext& ct) {
    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const auto& rescale_consts = context.getRescaleConsts();
    auto N = params.getN();
    auto level = ct.getLevel();
    const auto limb_count = level + 1;

    if (level == 0 ||
        level > params.getMaxLevel() ||
        limb_count > moduli.size() ||
        limb_count > twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_rescale: invalid level");
    }

    const size_t new_level = level - 1;
    const size_t drop_limb_idx = level;
    const size_t new_limb_count = level;
    const uint64_t p = moduli[drop_limb_idx];
    const uint64_t half_p = p >> 1;
    const auto polys = ct.getNumPolys();
    if (polys == 0) {
        throw std::invalid_argument("ckks_rescale: empty ciphertext");
    }
    for (size_t poly_idx = 0; poly_idx < polys; ++poly_idx) {
        auto& poly = ct.getPolyMutable(poly_idx);
        if (poly.size() != limb_count * N) {
            throw std::invalid_argument("ckks_rescale: invalid ciphertext buffer size");
        }
    }

    const auto& rc = rescale_consts[drop_limb_idx];
    if (rc.one_shoup_mod_pi.size() != new_limb_count ||
        rc.inv_p_mod_pi.size() != new_limb_count ||
        rc.inv_p_shoup_mod_pi.size() != new_limb_count ||
        rc.half_p_ntt_stride != N ||
        rc.half_p_ntt_mod_pi_flat.size() != new_limb_count * N) {
        throw std::runtime_error("ckks_rescale: invalid precomputed constants");
    }

    MyVector<uint64_t> rem(N);

    const auto& last_tbl = twiddle_ntt[drop_limb_idx];
    MyVector<uint64_t*> drop_ptrs(polys);
    for (size_t poly_idx = 0; poly_idx < polys; ++poly_idx) {
        auto& poly = ct.getPolyMutable(poly_idx);
        uint64_t* __restrict drop_ptr = poly.data() + drop_limb_idx * N;
        drop_ptrs[poly_idx] = drop_ptr;
        ntt_inverse_dif2_dispatch(drop_ptr, N, p, last_tbl);
    }

    for (size_t i = 0; i < new_limb_count; ++i) {
        const uint64_t q = moduli[i];
        const uint64_t one_shoup = rc.one_shoup_mod_pi[i];
        const uint64_t* __restrict half_ntt = rc.half_p_ntt_mod_pi_flat.data() + i * N;

        const uint64_t inv_p = rc.inv_p_mod_pi[i];
        const uint64_t inv_p_shoup = rc.inv_p_shoup_mod_pi[i];
        const auto& tbl = twiddle_ntt[i];

        for (size_t poly_idx = 0; poly_idx < polys; ++poly_idx) {
            compute_rescale_remainder_scalar(
                rem.data(), drop_ptrs[poly_idx], N, p, half_p, q,
                one_shoup);
            ntt_forward_dit2_dispatch(rem.data(), N, q, tbl);
            uint64_t* __restrict poly_ptr =
                ct.getPolyMutable(poly_idx).data() + i * N;
            finish_rescale_limb_scalar(poly_ptr, half_ntt, rem.data(), N,
                                       q, inv_p, inv_p_shoup);
        }
    }

    const long double new_scale =
        ct.getScale() / static_cast<long double>(p);
    for (size_t poly_idx = 0; poly_idx < polys; ++poly_idx) {
        ct.getPolyMutable(poly_idx).resize(new_limb_count * N);
    }
    ct.setScale(new_scale);
    ct.setLevel(new_level);
}
