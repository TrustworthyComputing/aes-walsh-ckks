#ifndef CKKS_BOOTSTRAP_DFT_HPP
#define CKKS_BOOTSTRAP_DFT_HPP

#include <cstddef>
#include <vector>

#include "ckks_linear_transform_plan.hpp"
#include "ckks_params.hpp"

class CKKSContext;

enum class CKKSBootstrapDftType {
    SlotsToCoeffs,
    CoeffsToSlots,
};

struct CKKSBootstrapDftMatrixLiteral {
    CKKSBootstrapDftType type = CKKSBootstrapDftType::SlotsToCoeffs;
    size_t level = 0;
    size_t baby_step_count = 0;
    size_t stage_count = 1;
    CKKSBootstrapDftScaleSchedule log_plaintext_scale_groups;
    std::vector<size_t> merge_depths;
    int log_bsgs_ratio = 1;
    // Logical DFT slot count. Zero means use the full physical CKKS slot
    // count. When smaller, DFT diagonals are generated for this count and
    // expanded periodically across the full slot space.
    size_t active_slots = 0;
    double transform_scale = 1.0;
    double plaintext_scale = 1.0;
    // Optional input-slot selector fused into the first S2C stage.
    // A zero size means no selector. Nonzero is only valid for SlotsToCoeffs.
    size_t input_selector_start = 0;
    size_t input_selector_size = 0;
    bool rescale_after_each_stage = true;
};

CKKSLinearTransformPlan ckks_generate_bootstrap_dft_plan(
    const CKKSContext& context,
    const CKKSBootstrapDftMatrixLiteral& literal);

#endif // CKKS_BOOTSTRAP_DFT_HPP
