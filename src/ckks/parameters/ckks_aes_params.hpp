#ifndef CKKS_AES_PARAMS_HPP
#define CKKS_AES_PARAMS_HPP

#include <cmath>
#include <cstddef>

#include "binboot_k_bound.hpp"
#include "ckks_params.hpp"

namespace ckks_aes_params_detail {

inline CKKSBootstrapParams XBootParamAes13ReferenceBootstrapParams() {
    // XBOOT Table 2 Param-AES-13: ring logN=14, batch log=13.
    // XBOOT Mod1: mod1.CosDiscrete, degree 60, K=7,
    // DoubleAngle=0, LogMessageRatio=1.
    auto params = ckks_params_detail::make_evalmod_bootstrap_params(
        16,
        {CKKSBootstrapDftScaleSchedule{{28}},
         CKKSBootstrapDftScaleSchedule{{28}, {28}}},
        6,
        32,
        {0, 0});
    params.evalmod_k = 7;
    params.evalmod_polynomial_degree = 60;
    params.evalmod_double_angle = 0;
    params.evalmod_log_message_ratio = 1;
    params.evalmod_coefficient_prune_threshold = 0.0;
    params.evalmod_discrete_interval_radius = std::nextafter(0.25, 0.0);
    params.evalmod_discrete_validation_points_per_interval = 64;
    params.linear_transform_log_bsgs_ratios = {1, 2};
    params.linear_transform_use_qp = {true, true};
    return params;
}

inline CKKSBootstrapParams AESWalsh15BootstrapParams() {
    CKKSBootstrapDftScaleSchedule s2c_schedule{{18, 20, 23}};

    auto params = ckks_params_detail::make_evalmod_bootstrap_params(
        32,
        {s2c_schedule,
         CKKSBootstrapDftScaleSchedule{{25, 25}, {25, 25}}},
        7,
        38,
        {0, 0});
    const auto k_bound = required_binboot_k_bound(
        size_t{1} << 15,
        params.sparse_secret_hamming_weight,
        std::exp2(-16.0L));
    params.evalmod_k = k_bound.evalmod_k;
    params.evalmod_polynomial_degree = 100;
    params.evalmod_double_angle = 0;
    params.evalmod_coefficient_prune_threshold = 1e-10;
    params.evalmod_discrete_interval_radius = std::nextafter(0.25, 0.0);
    params.evalmod_discrete_validation_points_per_interval = 128;
    params.linear_transform_log_bsgs_ratios = {1, 1};
    params.linear_transform_use_qp = {true, true};
    return params;
}

inline CKKSParamsConfig AESWalsh15Config() {
    CKKSParamsConfig config;
    config.secure = true;
    config.bts_keys = true;

    config.N = size_t{1} << 15;
    config.mult_depth = 2;
    config.q0_bit = 36;
    // config.q_bit = 42;
    config.q_mult_bits = {42, 42};
    config.scale = std::exp2(35);
    config.dnum = 3;
    config.pmoduli_count = 4;
    config.p_bits = {61, 61, 61, 61};
    config.secret_key_hamming_weight = 256;

    config.aes_walsh_params.enabled = true;
    config.aes_walsh_params.block_count = 512;
    config.aes_walsh_params.packed_segment_count = 32;
    config.bootstrap_params = AESWalsh15BootstrapParams();
    return config;
}

inline CKKSParamsConfig XBootParamAes13ReferenceConfig() {
    CKKSParamsConfig config;
    // Exact public XBOOT Param-AES-13: XBOOT reports 128-bit security, but its
    // log(PQ)=433 exceeds this library's conservative fixed-h=256 bound of 422.
    config.secure = false;
    config.bts_keys = true;

    config.N = size_t{1} << 14;
    config.mult_depth = 3;
    config.q0_bit = 32;
    config.q_bit = 32;
    config.q_bits = {
        32,       // Base
        28,       // SlotsToCoeffs
        31, 31, 31,  // AES round circuit
        32, 32, 32, 32, 32, 32,  // EvalMod
        28, 28,   // CoeffsToSlots
    };
    config.scale = std::exp2(31);
    config.dnum = 13;
    config.pmoduli_count = 1;
    config.p_bits = {32};
    config.secret_key_hamming_weight = 256;

    config.bootstrap_params = XBootParamAes13ReferenceBootstrapParams();
    return config;
}

inline CKKSBootstrapParams AESXBoot14SecureBootstrapParams() {
    // Secure N=2^14 variant of XBOOT Param-AES-13. Each Q limb is one
    // bit smaller, giving log(QP)=420 under the declared bit budget.
    auto params = ckks_params_detail::make_evalmod_bootstrap_params(
        16,
        {CKKSBootstrapDftScaleSchedule{{27}},
         CKKSBootstrapDftScaleSchedule{{27}, {27}}},
        6,
        31,
        {0, 0});
    params.evalmod_k = 7;
    params.evalmod_polynomial_degree = 60;
    params.evalmod_double_angle = 0;
    params.evalmod_log_message_ratio = 1;
    params.evalmod_coefficient_prune_threshold = 0.0;
    params.evalmod_discrete_interval_radius = std::nextafter(0.25, 0.0);
    params.evalmod_discrete_validation_points_per_interval = 64;
    params.linear_transform_log_bsgs_ratios = {1, 2};
    params.linear_transform_use_qp = {true, true};
    return params;
}

inline CKKSParamsConfig AESXBoot14SecureConfig() {
    auto config = XBootParamAes13ReferenceConfig();
    config.secure = true;
    config.q0_bit = 31;
    config.q_bit = 31;
    config.q_bits = {
        31,       // Base
        27,       // SlotsToCoeffs
        30, 30, 30,  // AES round circuit
        31, 31, 31, 31, 31, 31,  // EvalMod
        27, 27,   // CoeffsToSlots
    };
    config.scale = std::exp2(30);
    config.bootstrap_params = AESXBoot14SecureBootstrapParams();
    return config;
}

inline CKKSParamsConfig AESXBoot14SecureSparse1024Config() {
    auto config = AESXBoot14SecureConfig();
    config.bootstrap_params->active_slot_count = 1024;
    return config;
}

inline CKKSBootstrapParams AESXBoot15BootstrapParams() {
    // XBOOT Table 2 Param-AES-14-style modulus schedule at ring logN=15.
    // XBOOT Mod1: mod1.CosDiscrete, degree 120, K=12,
    // DoubleAngle=0, LogMessageRatio=1.
    auto params = ckks_params_detail::make_evalmod_bootstrap_params(
        32,
        {CKKSBootstrapDftScaleSchedule{{36}, {36}},
         CKKSBootstrapDftScaleSchedule{{38}, {38}, {38}, {38}}},
        7,
        38,
        {0, 0});
    params.evalmod_k = 12;
    params.evalmod_polynomial_degree = 120;
    params.evalmod_double_angle = 0;
    params.evalmod_log_message_ratio = 1;
    params.evalmod_coefficient_prune_threshold = 0.0;
    params.evalmod_discrete_interval_radius = std::nextafter(0.25, 0.0);
    params.evalmod_discrete_validation_points_per_interval = 64;
    params.linear_transform_log_bsgs_ratios = {2, 1};
    params.linear_transform_use_qp = {true, true};
    return params;
}

inline CKKSParamsConfig AESXBoot15Config() {
    CKKSParamsConfig config;
    config.secure = true;
    config.bts_keys = true;

    config.N = size_t{1} << 15;
    config.mult_depth = 3;
    config.q0_bit = 38;
    config.q_bit = 38;
    config.q_bits = {
        38,       // Base
        36, 36,   // SlotsToCoeffs
        37, 37, 37,  // AES round circuit
        38, 38, 38, 38, 38, 38, 38,  // EvalMod
        38, 38, 38, 38,  // CoeffsToSlots
    };
    config.scale = std::exp2(37);
    config.dnum = 5;
    config.pmoduli_count = 5;
    config.p_bits = {38, 38, 38, 38, 38};
    config.secret_key_hamming_weight = 256;

    config.bootstrap_params = AESXBoot15BootstrapParams();
    return config;
}

inline CKKSParamsConfig AESXBoot15Sparse1024Config() {
    auto config = AESXBoot15Config();
    config.bootstrap_params->active_slot_count = 1024;
    return config;
}

}  // namespace ckks_aes_params_detail

inline CKKSParams CKKSParams::AESWalsh15() {
    return CKKSParams(ckks_aes_params_detail::AESWalsh15Config());
}

inline CKKSParams CKKSParams::XBootParamAes13Reference() {
    return CKKSParams(ckks_aes_params_detail::XBootParamAes13ReferenceConfig());
}

inline CKKSParams CKKSParams::AESXBoot14Secure() {
    return CKKSParams(ckks_aes_params_detail::AESXBoot14SecureConfig());
}

inline CKKSParams CKKSParams::AESXBoot14SecureSparse1024() {
    return CKKSParams(ckks_aes_params_detail::AESXBoot14SecureSparse1024Config());
}

inline CKKSParams CKKSParams::AESXBoot15() {
    return CKKSParams(ckks_aes_params_detail::AESXBoot15Config());
}

inline CKKSParams CKKSParams::AESXBoot15Sparse1024() {
    return CKKSParams(ckks_aes_params_detail::AESXBoot15Sparse1024Config());
}

#endif  // CKKS_AES_PARAMS_HPP
