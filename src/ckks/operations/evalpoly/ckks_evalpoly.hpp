#ifndef CKKS_EVALPOLY_HPP
#define CKKS_EVALPOLY_HPP

#include <cstddef>
#include <span>

#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

struct CKKSEvalPolyOptions {
    // Scale assigned after each rescale. When zero, the input ciphertext scale is used.
    double target_scale = 0.0;

};

CKKSCiphertext ckks_evalchebyshev_preconditioned(
    const CKKSContext& context,
    const CKKSCiphertext& x,
    std::span<const double> chebyshev_coefficients,
    const CKKSEvalPolyOptions& options = {});

#endif // CKKS_EVALPOLY_HPP
