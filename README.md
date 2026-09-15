# Walsh LUT Evaluation on Lazy Bits for CKKS AES Transciphering

This is the implementation repository for the paper
["Walsh LUT Evaluation on Lazy Bits for CKKS AES Transciphering"](https://eprint.iacr.org/2026/1385).
The benchmark driver covers AESWalsh15 and the full and sparse AESXBoot15 and
secure AESXBoot14 baselines in Tables 1 and 2 of the paper.

## Requirements

- A C++20 compiler
- CMake 3.18 or newer
- OpenMP
- GMP and MPFR development libraries
- 64 GB RAM

On Ubuntu, the required packages can be installed with:

```text
sudo apt install build-essential cmake libgmp-dev libmpfr-dev
```

The paper measurements used Ubuntu, an AMD Ryzen Threadripper 7970X with 32
physical cores and 64 hardware threads, 160 GB of RAM, and 64 OpenMP threads.

## Build

From the repository root:

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
```

The resulting executable is `build/bench/aes_bench`.

## Run the paper experiments

Run each main Table 2 configuration independently:

```text
OMP_NUM_THREADS=64 ./build/bench/aes_bench walsh15
OMP_NUM_THREADS=64 ./build/bench/aes_bench xboot15-full
OMP_NUM_THREADS=64 ./build/bench/aes_bench xboot15-sparse-1024
OMP_NUM_THREADS=64 ./build/bench/aes_bench xboot14-full
OMP_NUM_THREADS=64 ./build/bench/aes_bench xboot14-sparse-1024
```

Run all five rows and print one combined table:

```text
OMP_NUM_THREADS=64 ./build/bench/aes_bench table2
```

The exact public XBOOT Param-AES-13 configuration used for the appendix
cross-check is also available. It reproduces the in-house C++ row of the
appendix table and is separate from the secure AESXBoot14 configuration used
in the main comparison.

```text
OMP_NUM_THREADS=64 ./build/bench/aes_bench xboot13-reference-full
```

Each command generates a CKKS context and evaluation keys, encrypts the AES
round key, times the homomorphic AES-CTR evaluation, decrypts every output bit,
and prints a `RESULT` line containing latency, maximum error, average error,
decoded mismatches, and rotation-key count. Context generation and round-key
encryption are intentionally outside the reported evaluation time, matching
the paper methodology.
