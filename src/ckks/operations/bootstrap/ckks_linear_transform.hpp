#ifndef CKKS_LINEAR_TRANSFORM_HPP
#define CKKS_LINEAR_TRANSFORM_HPP

#include <cstddef>

#include "ckks_ciphertext.hpp"
#include "ckks_linear_transform_plan.hpp"

class CKKSContext;

CKKSCiphertext ckks_apply_linear_transform_bsgs(
    const CKKSContext& context,
    const CKKSCiphertext& ct,
    const CKKSLinearTransformStage& stage);

CKKSCiphertext ckks_apply_linear_transform_plan(
    const CKKSContext& context,
    CKKSCiphertext ct,
    const CKKSLinearTransformPlan& plan);

#endif // CKKS_LINEAR_TRANSFORM_HPP
