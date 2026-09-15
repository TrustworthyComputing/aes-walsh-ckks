#include "ckks_modraise.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "modarith.hpp"
#include "ntt.hpp"

namespace {

int64_t center_lift_mod_q(uint64_t value, uint64_t q) {
    if (q == 0) {
        throw std::invalid_argument("ckks_modraise: modulus must be non-zero");
    }
    const uint64_t half = q >> 1;
    if (value > half) {
        return -static_cast<int64_t>(q - value);
    }
    return static_cast<int64_t>(value);
}

uint64_t signed_to_mod_q(int64_t value, uint64_t q) {
    if (value >= 0) {
        return static_cast<uint64_t>(value) % q;
    }
    const uint64_t magnitude_mod_q = static_cast<uint64_t>(-value) % q;
    return magnitude_mod_q == 0 ? uint64_t{0} : q - magnitude_mod_q;
}

MyVector<uint64_t> modraise_eval_poly_from_q0(
    const CKKSContext& context,
    const MyVector<uint64_t>& poly_q0_eval,
    size_t target_level) {
    const auto& params = context.getParams();
    const size_t N = params.getN();
    const auto& q_moduli = params.getModuli();
    const auto& q_twiddle_ntt = context.getTwiddleNtt();
    const size_t target_limb_count = target_level + 1;

    if (poly_q0_eval.size() != N) {
        throw std::invalid_argument("ckks_modraise: expected a single q0 limb");
    }
    if (target_limb_count > q_moduli.size() ||
        target_limb_count > q_twiddle_ntt.size()) {
        throw std::invalid_argument("ckks_modraise: invalid target level");
    }

    MyVector<uint64_t> coeff_q0(poly_q0_eval);
    ntt_inverse_rns_flat_inplace(
        coeff_q0.data(),
        N,
        1,
        q_moduli,
        q_twiddle_ntt,
        /*lazy=*/false);

    MyVector<uint64_t> raised(target_limb_count * N);
    for (size_t coeff = 0; coeff < N; ++coeff) {
        const int64_t centered = center_lift_mod_q(coeff_q0[coeff], q_moduli[0]);
        for (size_t limb = 0; limb < target_limb_count; ++limb) {
            raised[limb * N + coeff] = signed_to_mod_q(centered, q_moduli[limb]);
        }
    }

    MyVector<uint64_t> active_q_moduli(q_moduli.begin(), q_moduli.begin() + target_limb_count);
    ntt_forward_rns_flat_inplace(
        raised.data(),
        N,
        target_limb_count,
        active_q_moduli,
        q_twiddle_ntt,
        /*lazy=*/false);
    return raised;
}

void validate_modraise_input(const CKKSContext& context, const CKKSCiphertext& ct, size_t target_level) {
    const auto& params = context.getParams();
    if (ct.getN() != params.getN()) {
        throw std::invalid_argument("ckks_modraise: ciphertext/context size mismatch");
    }
    if (ct.getNumPolys() != 2) {
        throw std::invalid_argument("ckks_modraise: expected a 2-polynomial ciphertext");
    }
    if (ct.getLevel() != 0) {
        throw std::invalid_argument("ckks_modraise: expected a level-0 ciphertext");
    }
    if (ct.getSecretOwner() != CKKSSecretOwner::BootstrapSparse) {
        throw std::invalid_argument("ckks_modraise: expected a sparse-secret ciphertext");
    }
    if (target_level == 0 || target_level > params.getMaxLevel()) {
        throw std::invalid_argument("ckks_modraise: target level must be in [1, max_level]");
    }
    if (ct.getA().size() != params.getN() || ct.getB().size() != params.getN()) {
        throw std::invalid_argument("ckks_modraise: invalid level-0 ciphertext buffer size");
    }
}

}  // namespace

void ckks_modraise_inplace(const CKKSContext& context, CKKSCiphertext& ct, size_t target_level) {
    validate_modraise_input(context, ct, target_level);

    auto raised_b = modraise_eval_poly_from_q0(context, ct.getB(), target_level);
    auto raised_a = modraise_eval_poly_from_q0(context, ct.getA(), target_level);

    ct.getBMutable() = std::move(raised_b);
    ct.getAMutable() = std::move(raised_a);
    ct.setLevel(target_level);
}
