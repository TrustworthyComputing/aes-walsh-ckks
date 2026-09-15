#ifndef FFT_HPP
#define FFT_HPP

#include <cstddef>
#include <complex>
#include <cstdint>

#include "aligned_vector.hpp"


MyVector<std::complex<double>> twiddle_ifft_generation(std::size_t N);
void fft_dif(MyVector<std::complex<double>>& input,
             const MyVector<std::complex<double>>& twiddles,
             std::size_t N);
void fft_dit(MyVector<std::complex<double>>& input,
             const MyVector<std::complex<double>>& twiddles,
             std::size_t N);

#endif
