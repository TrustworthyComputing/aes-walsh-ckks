#ifndef CKKS_EVALMOD_SCALE_PLAN_HPP
#define CKKS_EVALMOD_SCALE_PLAN_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "ckks_params.hpp"

struct CKKSEvalModScalePlan {
    double binary_c2s_transform_scale = 1.0;
};

inline size_t ckks_evalmod_log_scale_bits(const CKKSParams& params) {
    const auto& bootstrap = params.getBootstrapParams();
    return bootstrap.evalmod_bit_size == 0
        ? params.getDefaultQBit()
        : bootstrap.evalmod_bit_size;
}

inline CKKSEvalModScalePlan ckks_evalmod_scale_plan(
    const CKKSParams& params) {
    const auto& moduli = params.getModuli();
    if (moduli.empty()) {
        throw std::invalid_argument("ckks EvalMod scale plan: empty modulus chain");
    }

    CKKSEvalModScalePlan plan;
    const double q0 = static_cast<double>(moduli.front());
    if (!std::isfinite(q0) || q0 <= 0.0 || params.getScale() <= 0.0) {
        throw std::invalid_argument("ckks EvalMod scale plan: invalid q0");
    }
    const double K = static_cast<double>(params.getBootstrapEvalModK());
    if (!std::isfinite(K) || K <= 0.0) {
        throw std::invalid_argument("ckks EvalMod scale plan: invalid K");
    }

    const int scale_shift =
        static_cast<int>(ckks_evalmod_log_scale_bits(params)) -
        static_cast<int>(params.getQ0Bit());
    plan.binary_c2s_transform_scale =
        std::ldexp(params.getScale(), scale_shift) /
        q0 /
        K;
    return plan;
}

#endif  // CKKS_EVALMOD_SCALE_PLAN_HPP
