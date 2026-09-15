#include "ckks_s2c_generalized.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#include "ckks_linear_transform.hpp"

namespace {

[[noreturn]] void throw_cached_s2c_error(const std::string& reason) {
    std::cerr << reason << '\n';
    throw std::runtime_error(reason);
}

}  // namespace

CKKSCiphertext ckks_s2c_generalized(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    size_t stage_count) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const size_t level = ct.getLevel();
    const size_t s2c_depth = params.getBootstrapS2CDepth();

    if (ct.getN() != N) {
        throw std::invalid_argument("ckks_s2c_generalized: ciphertext/context size mismatch");
    }
    if (ct.getNumPolys() != 2) {
        throw std::invalid_argument("ckks_s2c_generalized: expected a 2-polynomial ciphertext");
    }
    if (level == 0 || level > params.getMaxLevel()) {
        throw std::invalid_argument("ckks_s2c_generalized: S2C requires a ciphertext with at least one available level");
    }
    if (ct.getSecretOwner() != CKKSSecretOwner::Dense) {
        throw std::invalid_argument("ckks_s2c_generalized: expected a dense-secret input ciphertext");
    }
    if (ct.getMessageEncodingState() != CKKSMessageEncodingState::Slots) {
        throw std::invalid_argument("ckks_s2c_generalized: expected slot-encoded input ciphertext");
    }
    if (stage_count == 0 || stage_count != s2c_depth || s2c_depth > level) {
        throw std::invalid_argument("ckks_s2c_generalized: invalid S2C stage count for input level");
    }

    if (!context.hasEvalModBootstrapDftPlans()) {
        throw_cached_s2c_error(
            "ckks_s2c_generalized: cached EvalMod S2C DFT plan is unavailable");
    }
    if (level != s2c_depth) {
        throw_cached_s2c_error(
            "ckks_s2c_generalized: cached EvalMod S2C DFT plan requires input level " +
            std::to_string(s2c_depth) + ", got level " + std::to_string(level));
    }

    return ckks_apply_linear_transform_plan(
        context,
        ct.clone(),
        context.getEvalModS2CPlan());
}
