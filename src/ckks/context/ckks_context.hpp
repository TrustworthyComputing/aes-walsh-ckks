#ifndef CKKS_CONTEXT_HPP
#define CKKS_CONTEXT_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <utility>
#include "aligned_vector.hpp"

#include "ckks_ciphertext.hpp"
#include "ckks_linear_transform_plan.hpp"
#include "ckks_params.hpp"
#include "ntt.hpp"
#include "poly_approx.hpp"

using Complex = std::complex<double>;

enum class CKKSDecryptDomain {
    Slots,
    Coefficients,
};

struct CKKSRescaleConsts {
    MyVector<uint64_t> one_shoup_mod_pi;
    MyVector<uint64_t> inv_p_mod_pi;
    MyVector<uint64_t> inv_p_shoup_mod_pi;
    MyVector<uint64_t> half_p_ntt_mod_pi_flat;
    size_t half_p_ntt_stride = 0;
};

struct ApproxCrtModUpPrecomp {
    MyVector<uint64_t> src_moduli;
    MyVector<uint64_t> dst_moduli;
    MyVector<BarrettConst> src_barrett;
    MyVector<BarrettConst> dst_barrett;
    MyVector<uint64_t> qhat_inv_mod_src;
    MyVector<uint64_t> qhat_inv_mod_src_shoup;
    MyVector<double> src_modulus_inverse;
    // Row-major layout for better cache locality in base-change inner loops.
    // Access element [dst][src] as table[dst * src_count + src].
    MyVector<uint64_t> qhat_mod_dst_flat;
    MyVector<uint64_t> qhat_mod_dst_shoup_flat;
    MyVector<uint64_t> src_product_mod_dst;
    MyVector<uint64_t> src_product_mod_dst_shoup;
    // Row-major correction table [dst][alpha] = alpha * prod(src_moduli) mod dst.
    MyVector<uint64_t> src_product_alpha_mod_dst_flat;
};

struct CKKSCoefficientAutomorphismMap {
    MyVector<size_t> dst_index;
    MyVector<uint8_t> negate;
};

class CKKSContext {
    
public:
    CKKSContext(CKKSParams ckks_params);
    const CKKSParams &getParams() const;
    const MyVector<Complex> &getTwiddleFft() const;
    const MyVector<Complex> &getTwiddleIfft() const;
    const MyVector<std::size_t> &getPermTable() const;
    const MyVector<Negacyclic_NTT_Twiddles> &getTwiddleNtt() const;
    const MyVector<Negacyclic_NTT_Twiddles> &getPTwiddleNtt() const;
    const MyVector<CKKSRescaleConsts> &getRescaleConsts() const;
    const MyVector<MyVector<uint64_t>> &getSecretKey() const;
    const MyVector<MyVector<uint64_t>>& getLowSecretKey() const;
    std::span<const CKKSCiphertext> getRelinHybrid(size_t level) const;
    bool hasSparseSecretEncapsulationKeys() const;
    std::span<const CKKSCiphertext> getDenseToSparseSwitchKey() const;
    std::span<const CKKSCiphertext> getSparseToDenseSwitchKey() const;
    bool hasConjugationKey() const;
    std::span<const CKKSCiphertext> getConjugationKey() const;
    const MyVector<size_t>& getConjugationNttMap() const;
    size_t getRotationKeyCount() const;
    bool hasRotationKey(int64_t rotation) const;
    std::span<const CKKSCiphertext> getRotationKey(int64_t rotation) const;
    const MyVector<size_t>& getRotationNttMap(int64_t rotation) const;
    const ApproxCrtModUpPrecomp& getHybridPartitionModUpPrecomp(size_t level, size_t partition) const;
    const ApproxCrtModUpPrecomp& getPToQModUpPrecomp(size_t level) const;
    std::span<const uint64_t> getPInvModQForLevel(size_t level) const;
    std::span<const uint64_t> getPInvModQShoupForLevel(size_t level) const;
    std::span<const uint64_t> getFoldedPInvModQForLevel(size_t level) const;
    std::span<const uint64_t> getFoldedPInvModQShoupForLevel(size_t level) const;
    bool hasEvalModBootstrapDftPlans() const;
    const CKKSLinearTransformPlan& getEvalModS2CPlan() const;
    const CKKSLinearTransformPlan& getEvalModScaledC2SPlan() const;
    double getEvalModScaledC2STransformScale() const;
    const PolynomialApproximation& getEvalModPolynomial() const;

private:
    CKKSParams ckks_params;
    MyVector<int8_t> secret_key_coeffs;
    MyVector<MyVector<uint64_t>> secret_key;
    MyVector<int8_t> low_secret_key_coeffs;
    MyVector<MyVector<uint64_t>> low_secret_key;
    MyVector<CKKSCiphertext> relin_hybrid_digits;
    MyVector<CKKSCiphertext> dense_to_sparse_switch_key;
    MyVector<CKKSCiphertext> sparse_to_dense_switch_key;
    MyVector<CKKSCiphertext> conjugation_key;
    MyVector<size_t> conjugation_ntt_map;
    bool conjugation_key_generated = false;
    MyVector<int64_t> rotation_key_values;
    MyVector<MyVector<CKKSCiphertext>> rotation_keys;
    MyVector<MyVector<size_t>> rotation_ntt_maps;
    std::unordered_map<int64_t, size_t> rotation_key_index;
    size_t rotation_key_count = 0;
    MyVector<MyVector<ApproxCrtModUpPrecomp>> hybrid_partition_modup_precomp_by_level;
    MyVector<ApproxCrtModUpPrecomp> p_to_q_modup_precomp_by_level;
    MyVector<uint64_t> p_inv_mod_q_prefix_by_level;
    MyVector<uint64_t> p_inv_mod_q_shoup_prefix_by_level;
    MyVector<MyVector<uint64_t>> folded_p_inv_mod_q_by_level;
    MyVector<MyVector<uint64_t>> folded_p_inv_mod_q_shoup_by_level;
    size_t relin_key_level = 0;
    MyVector<Complex> twiddle_fft;
    MyVector<Complex> twiddle_ifft;
    MyVector<Negacyclic_NTT_Twiddles> twiddle_ntt;
    MyVector<Negacyclic_NTT_Twiddles> p_twiddle_ntt;
    MyVector<CKKSRescaleConsts> rescale_consts;
    MyVector<std::size_t> perm_table;
    CKKSLinearTransformPlan evalmod_s2c_plan;
    CKKSLinearTransformPlan evalmod_scaled_c2s_plan;
    double evalmod_scaled_c2s_transform_scale = 1.0;
    PolynomialApproximation evalmod_polynomial;
    bool evalmod_polynomial_generated = false;
    bool evalmod_bootstrap_dft_plans_generated = false;
};

MyVector<CKKSCiphertext> generate_secret_key_switch_key_hybrid(
    const CKKSContext& context,
    const MyVector<int8_t>& source_secret_coeffs,
    const MyVector<int8_t>& target_secret_coeffs);

#endif // CKKS_CONTEXT_HPP
