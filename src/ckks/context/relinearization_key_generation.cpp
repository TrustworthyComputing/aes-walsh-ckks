#include "relinearization_key_generation.hpp"

#include <algorithm>
#include <stdexcept>

#include "ckks_encoding.hpp"
#include "hybrid_key_folding.hpp"
#include "hybrid_key_montgomery.hpp"
#include "modarith.hpp"
#include "ntt.hpp"
#include "polynomial.hpp"
#include "secret_key_generation.hpp"

namespace {

struct QupBasis {
    MyVector<uint64_t> moduli;
    MyVector<Negacyclic_NTT_Twiddles> twiddles;
};

QupBasis build_qup_basis(const CKKSContext& context, size_t q_limb_count) {
    const auto& params = context.getParams();
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();
    const auto& q_twiddle_ntt = context.getTwiddleNtt();
    const auto& p_twiddle_ntt = context.getPTwiddleNtt();

    QupBasis basis;
    basis.moduli.reserve(q_limb_count + p_moduli.size());
    basis.moduli.insert(basis.moduli.end(), q_moduli.begin(), q_moduli.begin() + q_limb_count);
    basis.moduli.insert(basis.moduli.end(), p_moduli.begin(), p_moduli.end());

    basis.twiddles.reserve(basis.moduli.size());
    for (size_t i = 0; i < q_limb_count; ++i) {
        basis.twiddles.emplace_back(q_twiddle_ntt[i]);
    }
    for (size_t i = 0; i < p_moduli.size(); ++i) {
        basis.twiddles.emplace_back(p_twiddle_ntt[i]);
    }
    return basis;
}

} // namespace

ApproxCrtModUpPrecomp build_approx_crt_modup_precomp(
    const MyVector<uint64_t>& src_moduli,
    const MyVector<uint64_t>& dst_moduli) {
    if (src_moduli.empty()) {
        throw std::invalid_argument("ckks_context: empty source basis in ModUp precompute");
    }

    ApproxCrtModUpPrecomp precomp;
    precomp.src_moduli = src_moduli;
    precomp.dst_moduli = dst_moduli;
    precomp.src_barrett.resize(src_moduli.size());
    for (size_t i = 0; i < src_moduli.size(); ++i) {
        precomp.src_barrett[i] = barrett_const_computation(src_moduli[i]);
    }

    precomp.dst_barrett.resize(dst_moduli.size());
    for (size_t i = 0; i < dst_moduli.size(); ++i) {
        precomp.dst_barrett[i] = barrett_const_computation(dst_moduli[i]);
    }

    const size_t src_count = src_moduli.size();
    const size_t dst_count = dst_moduli.size();
    precomp.qhat_inv_mod_src.resize(src_count, uint64_t{0});
    precomp.qhat_inv_mod_src_shoup.resize(src_count, uint64_t{0});
    precomp.src_modulus_inverse.resize(src_count, 0.0);
    precomp.qhat_mod_dst_flat.assign(dst_count * src_count, uint64_t{0});
    precomp.qhat_mod_dst_shoup_flat.assign(dst_count * src_count, uint64_t{0});
    precomp.src_product_mod_dst.resize(dst_count, uint64_t{0});
    precomp.src_product_mod_dst_shoup.resize(dst_count, uint64_t{0});
    precomp.src_product_alpha_mod_dst_flat.assign(
        dst_count * (src_count + 1),
        uint64_t{0});

    for (size_t i = 0; i < src_count; ++i) {
        const uint64_t qi = src_moduli[i];
        const auto* qi_barrett = &precomp.src_barrett[i];
        precomp.src_modulus_inverse[i] = 1.0 / static_cast<double>(qi);

        uint64_t qhat_mod_qi = 1;
        for (size_t j = 0; j < src_count; ++j) {
            if (j == i) {
                continue;
            }
            qhat_mod_qi = mul_mod_u64(qhat_mod_qi, src_moduli[j] % qi, qi, qi_barrett);
        }
        precomp.qhat_inv_mod_src[i] = mod_inverse(qhat_mod_qi, qi);
        precomp.qhat_inv_mod_src_shoup[i] = compute_shoup(precomp.qhat_inv_mod_src[i], qi);

        for (size_t t = 0; t < dst_count; ++t) {
            const uint64_t m = dst_moduli[t];
            const auto* m_barrett = &precomp.dst_barrett[t];
            uint64_t qhat_mod_m = 1;
            for (size_t j = 0; j < src_count; ++j) {
                if (j == i) {
                    continue;
                }
                qhat_mod_m = mul_mod_u64(qhat_mod_m, src_moduli[j] % m, m, m_barrett);
            }
            const size_t flat_idx = t * src_count + i;
            precomp.qhat_mod_dst_flat[flat_idx] = qhat_mod_m;
            precomp.qhat_mod_dst_shoup_flat[flat_idx] = compute_shoup(qhat_mod_m, m);
        }
    }

    for (size_t t = 0; t < dst_count; ++t) {
        const uint64_t m = dst_moduli[t];
        const auto* m_barrett = &precomp.dst_barrett[t];
        uint64_t src_product_mod_m = 1;
        for (const uint64_t qi : src_moduli) {
            src_product_mod_m = mul_mod_u64(src_product_mod_m, qi % m, m, m_barrett);
        }
        precomp.src_product_mod_dst[t] = src_product_mod_m;
        precomp.src_product_mod_dst_shoup[t] = compute_shoup(src_product_mod_m, m);

        uint64_t correction = 0;
        const size_t table_off = t * (src_count + 1);
        precomp.src_product_alpha_mod_dst_flat[table_off] = 0;
        for (size_t alpha = 1; alpha <= src_count; ++alpha) {
            correction = add_mod_q(correction, src_product_mod_m, m);
            precomp.src_product_alpha_mod_dst_flat[table_off + alpha] = correction;
        }
    }

    return precomp;
}

MyVector<CKKSCiphertext> encrypt_rlk_hybrid(
    const CKKSContext& context,
    const MyVector<int8_t>& sk_coeff) {
    const auto& params = context.getParams();
    const auto level = params.getMaxLevel();
    const size_t q_limb_count = level + 1;
    const size_t N = params.getN();
    const size_t dnum = params.getDnum();
    const auto& q_moduli = params.getModuli();
    const auto& p_moduli = params.getPModuli();

    const auto basis = build_qup_basis(context, q_limb_count);
    const auto sk_qup = ntt_secret_key(sk_coeff, basis.twiddles, N, basis.moduli);

    MyVector<uint64_t> p_mod_m(basis.moduli.size(), uint64_t{1});
    for (size_t i = 0; i < basis.moduli.size(); ++i) {
        const uint64_t m = basis.moduli[i];
        const auto* barrett = &basis.twiddles[i].barrett_const;
        uint64_t accum = 1;
        for (uint64_t p : p_moduli) {
            accum = mul_mod_u64(accum, p % m, m, barrett);
        }
        p_mod_m[i] = accum;
    }

    const size_t alpha = (q_limb_count + dnum - 1) / dnum;
    MyVector<CKKSCiphertext> out;
    out.reserve(dnum);
    for (size_t part = 0; part < dnum; ++part) {
        const size_t start = part * alpha;
        const size_t end = std::min(start + alpha, q_limb_count);
        if (start >= end) {
            continue;
        }

        MyVector<uint64_t> ptxt(N * basis.moduli.size(), uint64_t{0});
        for (size_t i = start; i < end; ++i) {
            const uint64_t q = q_moduli[i];
            const auto& tbl = basis.twiddles[i];
            const uint64_t* s_ptr = sk_qup[i].data();
            uint64_t* out_ptr = ptxt.data() + i * N;
            std::copy_n(s_ptr, N, out_ptr);
            pointwise_multiply_inplace_dispatch(out_ptr, s_ptr, N, q, tbl);
            pointwise_multiply_scalar_inplace_dispatch(out_ptr, p_mod_m[i], N,
                                                       q, tbl);
        }

        auto key = ckks_encrypt_custom_eval(
            N,
            std::move(ptxt),
            sk_qup,
            basis.moduli,
            basis.twiddles,
            params.getSigma(),
            /*scale=*/1.0,
            level);
        if (params.fold_hybrid_keyswitch) {
            ckks_fold_hybrid_switch_key_constants(key, q_moduli, p_moduli, basis.twiddles);
        }
        if (params.montgomery) {
            ckks_convert_switch_key_to_montgomery_inplace(
                key,
                basis.moduli,
                basis.twiddles);
        }
        out.emplace_back(std::move(key));
    }
    return out;
}
