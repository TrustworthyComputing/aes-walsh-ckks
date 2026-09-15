#ifndef CKKS_KEYSWITCH_HPP
#define CKKS_KEYSWITCH_HPP

#include <cstddef>
#include <cstdint>
#include <span>

#include "aligned_vector.hpp"
#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

struct ApproxCrtModUpPrecomp;

void ckks_apply_approx_crt_modup(
    const ApproxCrtModUpPrecomp& precomp,
    const MyVector<const uint64_t*>& src_ptrs,
    const MyVector<uint64_t*>& dst_ptrs,
    size_t N);

void ckks_apply_approx_crt_moddown(
    const ApproxCrtModUpPrecomp& precomp,
    const MyVector<const uint64_t*>& src_ptrs,
    const MyVector<uint64_t*>& dst_q_ptrs,
    std::span<const uint64_t> p_inv_mod_q,
    std::span<const uint64_t> p_inv_mod_q_shoup,
    size_t N);

void ckks_apply_folded_approx_crt_moddown_pair_eval_q(
    const MyVector<uint64_t>& p_moduli,
    const MyVector<uint64_t>& q_moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& q_twiddle_ntt,
    std::span<const uint64_t> p_inv_mod_q,
    std::span<const uint64_t> p_inv_mod_q_shoup,
    const uint64_t* src_a_p_flat,
    uint64_t* dst_a_q_eval_flat,
    const uint64_t* src_b_p_flat,
    uint64_t* dst_b_q_eval_flat,
    size_t N);

void ckks_apply_folded_approx_crt_moddown_eval_q(
    const MyVector<uint64_t>& p_moduli,
    const MyVector<uint64_t>& q_moduli,
    const MyVector<Negacyclic_NTT_Twiddles>& q_twiddle_ntt,
    std::span<const uint64_t> p_inv_mod_q,
    std::span<const uint64_t> p_inv_mod_q_shoup,
    const uint64_t* src_p_flat,
    uint64_t* dst_q_eval_flat,
    size_t N);

void ckks_pointwise_multiply_accumulate_montgomery_key(
    const uint64_t* digit,
    const uint64_t* key_montgomery,
    uint64_t* acc,
    size_t N,
    uint64_t q,
    uint64_t q_neg_inv,
    const Negacyclic_NTT_Twiddles& table);

void ckks_apply_ntt_automorphism_to_flat_rns_poly_inplace(
    MyVector<uint64_t>& poly,
    size_t N,
    size_t limb_count,
    std::span<const size_t> dst_to_src_map);

enum class CKKSHybridKeySwitchCombine {
    Add,
    Assign,
};

struct CKKSEvalHoistedCiphertextQP {
    size_t N = 0;
    size_t level = 0;
    size_t q_limb_count = 0;
    size_t p_limb_count = 0;
    long double scale = 0.0L;
    MyVector<uint64_t> moduli;
    MyVector<MyVector<uint64_t>> a_partition_qp_eval;
    MyVector<MyVector<uint64_t>> b_partition_qp_eval;
};

struct CKKSHoistWorkspace {
    MyVector<uint64_t> a_q_coeffs;
    MyVector<const ApproxCrtModUpPrecomp*> hybrid_part_modup_precomp;
    MyVector<const uint64_t*> src_ptrs;
    MyVector<uint64_t*> dst_ptrs;
};

void ckks_hoist_ciphertext_a_qp_eval(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    CKKSHoistWorkspace& workspace,
    CKKSEvalHoistedCiphertextQP& out);

void ckks_hybrid_key_switch_inplace(
    const CKKSContext& context,
    size_t level,
    MyVector<uint64_t>& key_term,
    std::span<const CKKSCiphertext> switching_key,
    MyVector<uint64_t>& b_out,
    MyVector<uint64_t>& a_out,
    CKKSHybridKeySwitchCombine b_mode,
    CKKSHybridKeySwitchCombine a_mode,
    bool switching_key_montgomery = false);

void ckks_hybrid_key_switch_hoisted_inplace(
    const CKKSContext& context,
    const CKKSEvalHoistedCiphertextQP& hoisted_key_term,
    std::span<const CKKSCiphertext> switching_key,
    MyVector<uint64_t>& b_out,
    MyVector<uint64_t>& a_out,
    CKKSHybridKeySwitchCombine b_mode,
    CKKSHybridKeySwitchCombine a_mode,
    bool switching_key_montgomery,
    std::span<const size_t> ntt_map = {});

// Lazy automorphism: key-switches the unrotated component,
// adds the unrotated b component in QP, then applies ntt_map to the QP pair.
void ckks_hybrid_key_switch_hoisted_qp_rotated(
    const CKKSContext& context,
    const CKKSEvalHoistedCiphertextQP& hoisted_key_term,
    std::span<const CKKSCiphertext> switching_key,
    const MyVector<uint64_t>& b_q_eval,
    MyVector<uint64_t>& b_qp_out,
    MyVector<uint64_t>& a_qp_out,
    bool switching_key_montgomery,
    std::span<const size_t> ntt_map);

void ckks_hybrid_key_switch_qp_from_key_term(
    const CKKSContext& context,
    size_t level,
    const MyVector<uint64_t>& key_term_q_eval,
    std::span<const CKKSCiphertext> switching_key,
    MyVector<uint64_t>& b_qp_out,
    MyVector<uint64_t>& a_qp_out,
    bool switching_key_montgomery);

void ckks_key_switch_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    std::span<const CKKSCiphertext> switching_key,
    bool switching_key_montgomery = false);

void ckks_key_switch_dense_to_sparse_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct);

void ckks_key_switch_sparse_to_dense_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct);

#endif // CKKS_KEYSWITCH_HPP
