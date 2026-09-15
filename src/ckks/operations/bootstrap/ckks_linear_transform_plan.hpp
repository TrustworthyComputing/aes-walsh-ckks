#ifndef CKKS_LINEAR_TRANSFORM_PLAN_HPP
#define CKKS_LINEAR_TRANSFORM_PLAN_HPP

#include <cstddef>
#include <span>

#include "ckks_ciphertext.hpp"

struct CKKSLinearTransformStage {
    std::span<const CKKSEncoding> diagonals;
    std::span<const size_t> diagonal_indices;
    // Logical cyclic dimension for the diagonal map. Zero means N/2 slots.
    size_t slot_count = 0;
    size_t baby_step_count = 0;
    bool rescale_after = true;
    CKKSMessageEncodingState output_encoding_state = CKKSMessageEncodingState::Slots;
};

struct CKKSLinearTransformCompiledStage {
    size_t baby_step_count = 0;
    size_t giant_step_count = 0;
    MyVector<MyVector<size_t>> terms_by_giant;
    MyVector<char> baby_rotation_needed;
    MyVector<char> giant_rotation_needed;
    MyVector<size_t> baby_rotations;
    MyVector<size_t> giant_rotations;
};

struct CKKSLinearTransformOwnedStage {
    MyVector<CKKSEncoding> diagonals;
    MyVector<MyVector<uint64_t>> diagonal_qp_plaintexts;
    MyVector<size_t> diagonal_indices;
    CKKSLinearTransformCompiledStage compiled;
    // Logical cyclic dimension for BSGS metadata. Zero means N/2 slots.
    size_t slot_count = 0;
    size_t baby_step_count = 0;
    size_t qp_limb_count = 0;
    size_t qp_p_limb_count = 0;
    bool diagonal_qp_plaintexts_montgomery = false;
    bool rescale_after = true;
    CKKSMessageEncodingState output_encoding_state = CKKSMessageEncodingState::Slots;
};

struct CKKSLinearTransformPlan {
    MyVector<CKKSLinearTransformOwnedStage> stages;
    bool use_qp_evaluator = false;
};

#endif // CKKS_LINEAR_TRANSFORM_PLAN_HPP
