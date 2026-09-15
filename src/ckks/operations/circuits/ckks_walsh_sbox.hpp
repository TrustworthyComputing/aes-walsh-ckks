#ifndef CKKS_WALSH_SBOX_HPP
#define CKKS_WALSH_SBOX_HPP

#include <cstddef>
#include <memory>

#include "aligned_vector.hpp"
#include "ckks_ciphertext.hpp"
#include "ckks_context.hpp"

class CKKSWalshSboxScratch {
public:
    CKKSWalshSboxScratch();
    ~CKKSWalshSboxScratch();

    CKKSWalshSboxScratch(const CKKSWalshSboxScratch&) = delete;
    CKKSWalshSboxScratch& operator=(const CKKSWalshSboxScratch&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend MyVector<CKKSCiphertext>
    ckks_walsh_sbox_complex_blocks_mask_packed(
        CKKSContext& context,
        MyVector<CKKSCiphertext> input,
        size_t block_count,
        CKKSWalshSboxScratch& scratch);

};

// Native complex-block variant for the Walsh AES layout:
// - input bit ciphertexts carry block s in the real lane and block
//   block_count+s in the imaginary lane.
// - Walsh parity packing is applied directly to those complex-block inputs.
// - The batched EvalMod separates real/imag lanes, and outputs are recombined
//   back to the same complex-block bit layout.
MyVector<CKKSCiphertext> ckks_walsh_sbox_complex_blocks_mask_packed(
    CKKSContext& context,
    MyVector<CKKSCiphertext> input,
    size_t block_count,
    CKKSWalshSboxScratch& scratch);

#endif  // CKKS_WALSH_SBOX_HPP
