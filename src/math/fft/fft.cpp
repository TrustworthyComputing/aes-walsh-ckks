#include "fft.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <utility>


namespace {
    using Complex = std::complex<double>;
    size_t reverse_bits(size_t value, size_t width){
        size_t result = 0;
        for (size_t i = 0; i < width; i++){
            result = (result << 1) | (value & 1);
            value >>= 1;
        }
        return result;
    }

    // x is power of two
    constexpr unsigned ilog2_pow2(std::uint64_t x) {
        assert(x != 0 && std::has_single_bit(x));
        return static_cast<unsigned>(std::countr_zero(x));
    }

    inline Complex cis(double ang) {
        return {std::cos(ang), std::sin(ang)};
    }

}

MyVector<std::complex<double>> twiddle_ifft_generation(std::size_t N) {
    MyVector<Complex> tw(N);
    tw[0] = {0, 0};             // unused

    const Complex w_inv = cis(2.0 * std::numbers::pi / static_cast<double>(N));
    MyVector<Complex> stage_powers(N >> 1);
    stage_powers[0] = {1.0, 0.0};
    for (size_t i = 1; i < (N >> 1); ++i) {
        stage_powers[i] = stage_powers[i - 1] * w_inv;
    }

    for (size_t m = 1; m < N; m <<= 1) {
        const size_t step = N / (m << 1);
        const unsigned stage_bits = ilog2_pow2(m);
        for (size_t i = 0; i < m; ++i) {
            const size_t reordered = reverse_bits(i, stage_bits);
            tw[m + i] = stage_powers[reordered * step];
        }
    }

    // embedding the twists
    const Complex psi = cis(std::numbers::pi / static_cast<double>(N));
    MyVector<Complex> psi_pows(N);
    psi_pows[0] = Complex(1.0, 0.0);
    for (size_t j = 1; j < N; ++j) {
        psi_pows[j] = psi_pows[j - 1] * psi;
    }

    for (size_t m = N >> 1, k = 1; m > 0; m >>= 1, k <<= 1) {
        auto psi_k = psi_pows[k]; // = ψ^{-k}
        for (size_t i = 0; i < m; ++i) {
            tw[m + i] =
                tw[m + i] * psi_k;
        }
    }
    return tw;
}

void fft_dif(MyVector<std::complex<double>>& input,
             const MyVector<std::complex<double>>& twiddles,
             std::size_t N)
{
    const Complex* __restrict twiddle_ptr = twiddles.data();
    Complex* __restrict in_ptr = input.data();

    for (size_t m = N >> 1, k = 1; m > 0; m >>= 1, k <<= 1){
        for (size_t i = 0; i < m; ++i){
            const Complex w = twiddle_ptr[m + i];
            Complex* __restrict x0 = in_ptr + ((i << 1) * k);
            Complex* __restrict x1 = x0 + k;
            for (size_t j = 0; j < k; ++j){
                const Complex u = x0[j];
                const Complex v = x1[j];
                x0[j] = u + v;
                x1[j] = (u - v) * w;
            }
        }
    }
}

void fft_dit(MyVector<std::complex<double>>& input,
             const MyVector<std::complex<double>>& twiddles,
             std::size_t N)
{
    const Complex* __restrict twiddle_ptr = twiddles.data();
    Complex* __restrict in_ptr = input.data();

    for (size_t m = 1, k = N >> 1; m < N; m <<= 1, k >>= 1){
        for (size_t i = 0; i < m; ++i){
            const Complex w = twiddle_ptr[m + i];
            Complex* __restrict x0 = in_ptr + ((i << 1) * k);
            Complex* __restrict x1 = x0 + k;
            for (size_t j = 0; j < k; ++j){
                const Complex t = x0[j];
                const Complex u = x1[j] * w;
                x0[j] = t + u;
                x1[j] = t - u;
            }
        }
    }
}
