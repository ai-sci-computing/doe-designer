/// @file fft.hpp
/// @brief Orthonormal 2-D FFT on square arrays (FFTW back end, float and double).
#pragma once

#include "doe/array.hpp"

#include <cstddef>

namespace doe {

/// Number of threads FFTW uses for plans created from now on. 0 selects the
/// hardware concurrency. Threaded transforms give the same result as
/// single-threaded ones to rounding (the plan may differ, so bit-exactness
/// across thread counts is not guaranteed; T14 compares runs with the same
/// setting). Default: all cores.
void set_fft_threads(int n);
/// Threads used for new plans.
int fft_threads();

/// Square 2-D discrete Fourier transform with **orthonormal** scaling.
///
/// Both directions are scaled by @f$1/n@f$ (the square root of the number of
/// samples), i.e. NumPy's `norm="ortho"`. With that scaling the transform is
/// unitary, so a propagator built as @f$ \mathcal F^{-1} H \mathcal F @f$ with
/// @f$|H| \le 1@f$ is a partial isometry and its adjoint is obtained by
/// conjugating @f$H@f$ (Wirtinger/CR calculus background in
/// @cite kreutzdelgado2009). Sign convention is FFTW's, which equals NumPy's:
/// forward uses @f$ e^{-2\pi i k n / N} @f$.
///
/// Plans are created once per size with `FFTW_ESTIMATE` (deterministic plan
/// choice, hence bit-reproducible results across runs; `DOE_FFTW_MEASURE`
/// opts into the faster but non-deterministic planner) and `FFTW_UNALIGNED`
/// so they execute on arbitrary `std::vector` storage. Plan creation is
/// serialized behind a mutex because FFTW's planner is not thread-safe;
/// execution is.
template <class T>
class Fft2 {
public:
    /// Create plans for n x n transforms.
    explicit Fft2(std::size_t n);
    ~Fft2();
    Fft2(const Fft2&) = delete;
    Fft2& operator=(const Fft2&) = delete;
    /// Move: takes over the plans.
    Fft2(Fft2&& other) noexcept;
    /// Move assignment: destroys own plans, takes over the other's.
    Fft2& operator=(Fft2&& other) noexcept;

    /// Side length of the transforms this object was planned for.
    std::size_t size() const { return n_; }

    /// out = F in (orthonormal). `out` is resized if needed; `in` and `out` must differ.
    void forward(const Field<T>& in, Field<T>& out) const;
    /// out = F^{-1} in (orthonormal). `out` is resized if needed; `in` and `out` must differ.
    void inverse(const Field<T>& in, Field<T>& out) const;

    /// Allocating variant of forward().
    Field<T> forward(const Field<T>& in) const;
    /// Allocating variant of inverse().
    Field<T> inverse(const Field<T>& in) const;

private:
    std::size_t n_ = 0;
    void* fwd_ = nullptr;  ///< opaque fftw(f)_plan
    void* inv_ = nullptr;  ///< opaque fftw(f)_plan
    void execute(void* plan, const Field<T>& in, Field<T>& out) const;
};

/// @cond INTERNAL
extern template class Fft2<float>;
extern template class Fft2<double>;
/// @endcond

}  // namespace doe
