#include "ckks_bootstrap_evalmod_inplace.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "ckks_addition.hpp"
#include "ckks_bootstrap_dft.hpp"
#include "ckks_c2s_generalized.hpp"
#include "ckks_conjugate.hpp"
#include "ckks_encoding.hpp"
#include "ckks_evalmod_scale_plan.hpp"
#include "ckks_evalpoly.hpp"
#include "ckks_keyswitch.hpp"
#include "ckks_linear_transform.hpp"
#include "ckks_modraise.hpp"
#include "ckks_mul.hpp"
#include "ckks_relin.hpp"
#include "ckks_rescale.hpp"
#include "ckks_rotation.hpp"
#include "ckks_s2c_generalized.hpp"
#include "ntt.hpp"

namespace {

using Complex = std::complex<double>;


void validate_evalmod_input(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    const char* label) {
    const auto& params = context.getParams();
    const size_t s2c_depth = params.getBootstrapS2CDepth();
    if (s2c_depth == 0) {
        throw std::invalid_argument(std::string(label) + ": S2C depth must be positive");
    }
    if (ct.getN() != params.getN()) {
        throw std::invalid_argument(std::string(label) + ": ciphertext/context size mismatch");
    }
    if (ct.getNumPolys() != 2) {
        throw std::invalid_argument(std::string(label) + ": expected a 2-polynomial ciphertext");
    }
    if (ct.getLevel() != s2c_depth) {
        throw std::invalid_argument(std::string(label) + ": expected ciphertext level to equal S2C depth");
    }
    if (ct.getSecretOwner() != CKKSSecretOwner::Dense) {
        throw std::invalid_argument(std::string(label) + ": expected a dense-secret ciphertext");
    }
    if (ct.getMessageEncodingState() != CKKSMessageEncodingState::Slots) {
        throw std::invalid_argument(std::string(label) + ": expected slot-encoded input");
    }
}

uint64_t evalmod_input_scale_ratio(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    const char* label) {
    const long double target_scale = context.getParams().getScale();
    const long double input_scale = ct.getScale();
    if (!std::isfinite(input_scale) || input_scale <= 0.0L) {
        throw std::invalid_argument(std::string(label) + ": invalid ciphertext scale");
    }
    if (input_scale < target_scale) {
        throw std::invalid_argument(
            std::string(label) + ": EvalMod input scale is below the S2C target scale");
    }
    const long double ratio = input_scale / target_scale;
    const long double rounded = std::round(ratio);
    if (rounded < 1.0L ||
        rounded > static_cast<long double>(std::numeric_limits<uint64_t>::max()) ||
        std::abs(ratio - rounded) >
            std::max<long double>(1.0L, std::abs(ratio)) * 1e-3L) {
        throw std::invalid_argument(
            std::string(label) + ": EvalMod entry scale ratio is not an exact integer");
    }
    return static_cast<uint64_t>(rounded);
}

double evalmod_s2c_transform_scale(
    const CKKSParams& params,
    uint64_t scale_ratio) {
    if (scale_ratio == 0) {
        throw std::invalid_argument("ckks EvalMod: invalid S2C scale ratio");
    }
    return (static_cast<double>(params.getModuli()[0]) * 0.5) /
           (params.getScale() * static_cast<double>(scale_ratio));
}

std::string scaled_s2c_cache_key(const CKKSContext& context, uint64_t scale_ratio) {
    const auto& params = context.getParams();
    std::string key =
        std::to_string(params.getN()) + ":" +
        std::to_string(params.getQ0Bit()) + ":" +
        std::to_string(params.getScale()) + ":" +
        std::to_string(params.getDepth()) + ":" +
        std::to_string(params.getBootstrapActiveSlotCount()) + ":" +
        std::to_string(scale_ratio);
    for (const auto q_bit : params.getQBits()) {
        key += ":" + std::to_string(q_bit);
    }
    return key;
}

std::shared_ptr<const CKKSLinearTransformPlan> get_scaled_s2c_plan(
    const CKKSContext& context,
    uint64_t scale_ratio) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<const CKKSLinearTransformPlan>> cache;

    const auto key = scaled_s2c_cache_key(context, scale_ratio);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    const auto& params = context.getParams();
    const size_t s2c_depth = params.getBootstrapS2CDepth();
    if (s2c_depth == 0) {
        throw std::runtime_error("ckks EvalMod: scaled S2C depth must be positive");
    }

    CKKSBootstrapDftMatrixLiteral literal;
    literal.type = CKKSBootstrapDftType::SlotsToCoeffs;
    literal.level = s2c_depth;
    literal.baby_step_count =
        params.getBootstrapLinearTransformBsgsBabyStepCounts()[0];
    literal.log_bsgs_ratio = params.getBootstrapS2CLogBsgsRatio();
    literal.stage_count = s2c_depth;
    literal.log_plaintext_scale_groups = params.getBootstrapS2CLogPlaintextScales();
    literal.merge_depths = params.getBootstrapS2CMergeDepths();
    literal.active_slots = params.getBootstrapActiveSlotCount();
    literal.transform_scale = evalmod_s2c_transform_scale(params, scale_ratio);
    literal.plaintext_scale = static_cast<double>(params.getModuli()[s2c_depth]);
    literal.rescale_after_each_stage = true;

    auto plan = std::make_shared<CKKSLinearTransformPlan>(
        ckks_generate_bootstrap_dft_plan(context, literal));
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto [it, inserted] = cache.emplace(key, plan);
    return inserted ? plan : it->second;
}

CKKSCiphertext evalmod_s2c_for_input_scale(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    const char* label) {
    const auto& params = context.getParams();
    const uint64_t scale_ratio = evalmod_input_scale_ratio(context, ct, label);
    const bool needs_scaled_plan = scale_ratio != 1;
    CKKSCiphertext out =
        !needs_scaled_plan
            ? ckks_s2c_generalized(
                  context,
                  ct,
                  params.getBootstrapS2CDepth())
            : ckks_apply_linear_transform_plan(
                  context,
                  ct.clone(),
                  *get_scaled_s2c_plan(context, scale_ratio));
    // S2C collapses lazy slot overflow into target-normalized q0 representatives.
    // Reset scale metadata after that collapse; modularly dividing the noisy
    // ciphertext before S2C can turn odd encryption noise into a Q/2 error.
    out.setScale(context.getParams().getScale());
    return out;
}

void sparse_bootstrap_trace_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct) {
    const auto& params = context.getParams();
    if (!params.usesSparseBootstrapSlots()) {
        return;
    }
    if (ct.getSecretOwner() != CKKSSecretOwner::Dense) {
        throw std::invalid_argument(
            "ckks EvalMod sparse trace: expected a dense-secret ciphertext");
    }
    const size_t active_slots = params.getBootstrapActiveSlotCount();
    const size_t full_slots = params.getSlots();
    for (size_t rotation = active_slots;
         rotation < full_slots;
         rotation <<= 1) {
        auto rotated = ckks_rotate(
            context,
            ct,
            static_cast<int64_t>(rotation));
        ckks_add_inplace(context, ct, rotated);
    }
}

uint64_t rounded_scaled_real_mod_q(
    double value,
    long double scale,
    uint64_t q) {
    const long double scaled =
        static_cast<long double>(value) * scale;
    if (!std::isfinite(scaled)) {
        throw std::invalid_argument(
            "evalmod_binary_inplace: non-finite double-angle scalar");
    }
    const long double rounded = std::round(scaled);
    long double residue = std::fmod(
        rounded,
        static_cast<long double>(q));
    if (residue < 0.0L) {
        residue += static_cast<long double>(q);
    }
    uint64_t out = static_cast<uint64_t>(residue);
    if (out >= q) {
        out %= q;
    }
    return out;
}

void multiply_by_real_scalar_to_scale_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    double scalar,
    long double output_scale) {
    if (!std::isfinite(scalar) ||
        !std::isfinite(output_scale) ||
        output_scale <= 0.0L ||
        ct.getScale() <= 0.0) {
        throw std::invalid_argument(
            "evalmod_binary_inplace: invalid double-angle scalar scaling");
    }

    const auto& params = context.getParams();
    const auto& moduli = params.getModuli();
    const auto& twiddle_ntt = context.getTwiddleNtt();
    const size_t N = ct.getN();
    const size_t level = ct.getLevel();
    if (N != params.getN() || level > params.getMaxLevel()) {
        throw std::invalid_argument(
            "evalmod_binary_inplace: invalid double-angle ciphertext");
    }

    const long double scalar_scale =
        output_scale / static_cast<long double>(ct.getScale());
    for (size_t poly_idx = 0; poly_idx < ct.getNumPolys(); ++poly_idx) {
        auto& poly = ct.getPolyMutable(poly_idx);
        if (poly.size() != (level + 1) * N) {
            throw std::invalid_argument(
                "evalmod_binary_inplace: invalid double-angle ciphertext buffer");
        }
        for (size_t limb = 0; limb <= level; ++limb) {
            const uint64_t q = moduli[limb];
            const uint64_t scalar_mod_q =
                rounded_scaled_real_mod_q(scalar, scalar_scale, q);
            pointwise_multiply_scalar_inplace_dispatch(
                poly.data() + limb * N,
                scalar_mod_q,
                N,
                q,
                twiddle_ntt[limb],
                params.montgomery);
        }
    }
    ct.setScale(static_cast<double>(output_scale));
}

void apply_binboot_double_angle_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct,
    size_t double_angle) {
    const long double target_scale = ct.getScale();
    for (size_t i = 0; i < double_angle; ++i) {
        if (ct.getLevel() == 0) {
            throw std::invalid_argument(
                "evalmod_binary_inplace: insufficient level for BinBoot double-angle");
        }

        auto squared = ct.clone();
        ckks_square_inplace(context, squared);
        ckks_relin_hybrid_inplace(context, squared);
        ckks_rescale(context, squared);
        multiply_by_real_scalar_to_scale_inplace(
            context,
            squared,
            -4.0,
            target_scale);
        ckks_drop_to_level_inplace(ct, squared.getLevel());
        multiply_by_real_scalar_to_scale_inplace(
            context,
            ct,
            4.0,
            target_scale);
        ckks_add_inplace(context, ct, squared);
        ct.setScale(target_scale);
    }
}

void evalmod_binary_inplace(const CKKSContext& context, CKKSCiphertext& ct) {
    const auto& params = context.getParams();
    const auto& binboot_poly = context.getEvalModPolynomial();
    CKKSEvalPolyOptions eval_options;
    eval_options.target_scale = params.getScale();

    const size_t evalmod_log_scale = ckks_evalmod_log_scale_bits(params);
    ct.setScale(std::exp2(static_cast<double>(evalmod_log_scale)));

    ct = ckks_evalchebyshev_preconditioned(
        context,
        ct,
        binboot_poly.chebyshev_coeffs,
        eval_options);
    ct.setScale(params.getScale());
    apply_binboot_double_angle_inplace(
        context,
        ct,
        params.getBootstrapEvalModDoubleAngle());
    ct.setScale(params.getScale());
}

void validate_s2c_coeff_packed_input(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    const char* label) {
    const auto& params = context.getParams();
    if (ct.getN() != params.getN()) {
        throw std::invalid_argument(std::string(label) + ": ciphertext/context size mismatch");
    }
    if (ct.getNumPolys() != 2) {
        throw std::invalid_argument(std::string(label) + ": expected a 2-polynomial ciphertext");
    }
    if (ct.getLevel() != 0) {
        throw std::invalid_argument(std::string(label) + ": expected a level-0 ciphertext");
    }
    if (ct.getSecretOwner() != CKKSSecretOwner::Dense) {
        throw std::invalid_argument(std::string(label) + ": expected a dense-secret ciphertext");
    }
    if (ct.getMessageEncodingState() != CKKSMessageEncodingState::Coefficients) {
        throw std::invalid_argument(
            std::string(label) + ": expected coefficient-encoded ciphertext");
    }
    if (!(ct.getScale() > 0.0)) {
        throw std::invalid_argument(std::string(label) + ": invalid ciphertext scale");
    }
}

std::pair<CKKSCiphertext, CKKSCiphertext>
finish_batch_evalmod_from_s2c_coeff_packed(
    const CKKSContext& context,
    CKKSCiphertext packed,
    const char* label) {
    validate_s2c_coeff_packed_input(context, packed, label);

    const auto& params = context.getParams();
    ckks_key_switch_dense_to_sparse_inplace(context, packed);
    ckks_modraise_inplace(context, packed, params.getDepth());
    ckks_key_switch_sparse_to_dense_inplace(context, packed);
    sparse_bootstrap_trace_inplace(context, packed);
    packed = ckks_c2s_generalized(
        context,
        packed,
        context.getEvalModScaledC2STransformScale());

    auto conjugated = ckks_conjugate(context, packed);
    auto out0 = packed.clone();
    ckks_add_inplace(context, out0, conjugated);
    out0.setScale(packed.getScale());

    auto out1 = packed.clone();
    ckks_sub_inplace(context, out1, conjugated);
    out1 = ckks_mul(
        context,
        out1,
        Complex(0.0, -1.0),
        1.0);
    out1.setScale(packed.getScale());

    evalmod_binary_inplace(context, out0);
    evalmod_binary_inplace(context, out1);

    return {std::move(out0), std::move(out1)};
}

}  // namespace

void ckks_batch_bootstrap_evalmod_inplace(
    const CKKSContext& context,
    CKKSCiphertext& ct0,
    CKKSCiphertext& ct1) {
    validate_evalmod_input(context, ct0, "ckks_batch_bootstrap_evalmod");
    validate_evalmod_input(context, ct1, "ckks_batch_bootstrap_evalmod");

    auto packed = ct0.clone();
    auto imag = ckks_mul(
        context,
        ct1,
        Complex(0.0, 1.0),
        1.0);
    ckks_add_inplace(context, packed, imag);
    packed.setScale(ct0.getScale());

    auto [out0, out1] = ckks_batch_bootstrap_evalmod_from_complex_packed(
        context,
        std::move(packed));
    ct0 = std::move(out0);
    ct1 = std::move(out1);
}

std::pair<CKKSCiphertext, CKKSCiphertext>
ckks_batch_bootstrap_evalmod_from_complex_packed(
    const CKKSContext& context,
    CKKSCiphertext packed) {
    validate_evalmod_input(
        context,
        packed,
        "ckks_batch_bootstrap_evalmod_from_complex_packed");

    packed = evalmod_s2c_for_input_scale(
        context,
        packed,
        "ckks_batch_bootstrap_evalmod_from_complex_packed");

    return finish_batch_evalmod_from_s2c_coeff_packed(
        context,
        std::move(packed),
        "ckks_batch_bootstrap_evalmod_from_complex_packed");
}
