#include "ckks_c2s_generalized.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ckks_linear_transform.hpp"

namespace {

bool scales_close(double lhs, double rhs, double rel_tol = 1e-12) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return false;
    }
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= rel_tol * scale;
}

[[noreturn]] void throw_cached_c2s_error(const std::string& reason) {
    std::cerr << reason << '\n';
    throw std::runtime_error(reason);
}

}  // namespace

CKKSCiphertext ckks_c2s_generalized(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    double transform_scale) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t level = ct.getLevel();
    const size_t c2s_depth = params.getBootstrapC2SDepth();

    if (ct.getN() != N) {
        throw std::invalid_argument("ckks_c2s_generalized: ciphertext/context size mismatch");
    }
    if (ct.getNumPolys() != 2) {
        throw std::invalid_argument("ckks_c2s_generalized: expected a 2-polynomial ciphertext");
    }
    if (c2s_depth == 0) {
        throw std::invalid_argument("ckks_c2s_generalized: C2S depth must be positive");
    }
    if (level == 0 || level > params.getMaxLevel()) {
        throw std::invalid_argument("ckks_c2s_generalized: C2S requires a ciphertext with at least one available level");
    }
    if (c2s_depth > level) {
        throw std::invalid_argument("ckks_c2s_generalized: C2S depth exceeds input ciphertext level");
    }
    if (ct.getSecretOwner() != CKKSSecretOwner::Dense) {
        throw std::invalid_argument("ckks_c2s_generalized: expected a dense-secret input ciphertext");
    }
    if (ct.getMessageEncodingState() != CKKSMessageEncodingState::Coefficients) {
        throw std::invalid_argument("ckks_c2s_generalized: expected coefficient-encoded input ciphertext");
    }
    if (!(transform_scale > 0.0)) {
        throw std::invalid_argument("ckks_c2s_generalized: transform scale must be positive");
    }

    if (!context.hasEvalModBootstrapDftPlans()) {
        throw_cached_c2s_error(
            "ckks_c2s_generalized: cached EvalMod C2S DFT plan is unavailable");
    }
    if (level != params.getDepth()) {
        throw_cached_c2s_error(
            "ckks_c2s_generalized: cached EvalMod C2S DFT plan requires input level " +
            std::to_string(params.getDepth()) + ", got level " + std::to_string(level));
    }
    const double cached_transform_scale = context.getEvalModScaledC2STransformScale();
    if (!scales_close(transform_scale, cached_transform_scale)) {
        throw_cached_c2s_error(
            "ckks_c2s_generalized: cached EvalMod C2S DFT plan requires transform scale " +
            std::to_string(cached_transform_scale) + ", got " +
            std::to_string(transform_scale));
    }

    return ckks_apply_linear_transform_plan(
        context,
        ct.clone(),
        context.getEvalModScaledC2SPlan());
}
