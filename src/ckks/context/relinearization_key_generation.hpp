#ifndef CKKS_RELINEARIZATION_KEY_GENERATION_HPP
#define CKKS_RELINEARIZATION_KEY_GENERATION_HPP

#include "ckks_context.hpp"

ApproxCrtModUpPrecomp build_approx_crt_modup_precomp(
    const MyVector<uint64_t>& src_moduli,
    const MyVector<uint64_t>& dst_moduli);
MyVector<CKKSCiphertext> encrypt_rlk_hybrid(const CKKSContext& context, const MyVector<int8_t>& sk_coeff);

#endif // CKKS_RELINEARIZATION_KEY_GENERATION_HPP
