/// @file grid.hpp
/// @brief Sampling geometry of the DOE plane (and every parallel plane).
#pragma once

#include <cstddef>

namespace doe {

/// Smallest integer @f$ m \ge n @f$ whose prime factors all lie in {2, 3, 5, 7}.
///
/// The computational window is zero-padded to twice the active aperture
/// (Matsushima & Shimobaba 2009 @cite matsushima2009 §2.2: "the area of the
/// sampling window of the input field needs to be doubled along both the x-
/// and y-axes ... and the additional sampling points must be padded with
/// zeros"). Rounding that size up to the next power of two is the reflex and
/// it is expensive: for an active aperture of 600 the padded size would jump
/// from 1200 = 2^4 3 5^2 to 2048, 2.9x the pixels. FFTW "works most
/// efficiently for arrays whose size can be factored into small primes
/// (2, 3, 5, and 7)" @cite fftw_manual, so a 7-smooth size costs nothing.
///
/// Implementation: enumerate all 7-smooth numbers up to 2n and take the
/// smallest one >= n. The candidate set is never empty because the power of
/// two in [n, 2n) is 7-smooth.
///
/// @param n requested minimum size; values < 1 return 1.
/// @return the smallest 7-smooth integer >= n.
int next_smooth(int n);

/// Square sampling grid of the padded computational window.
///
/// Conventions (identical to the NumPy reference so arrays compare 1:1):
/// - sample coordinate  @f$ x_i = (i - \lfloor n/2 \rfloor)\,p @f$,
/// - spatial frequency  @f$ f_i @f$ in FFT order (`numpy.fft.fftfreq`):
///   @f$ i/(np) @f$ for @f$ i < \lceil n/2 \rceil @f$, else @f$ (i-n)/(np) @f$,
/// - the first array index runs along x.
struct Grid {
    int n;              ///< samples per side of the *padded* window
    double pitch;       ///< sample pitch in meters
    double wavelength;  ///< vacuum wavelength in meters

    /// Grid whose window holds `active` samples plus zero padding, rounded up
    /// to the next 7-smooth size: n = next_smooth(ceil(pad * active)).
    static Grid padded(int active, double pitch, double wavelength, double pad = 2.0);

    /// Grid whose window is set by the propagation distance (the picture-clear
    /// rule). The FFT propagator is periodic in the window @f$S@f$: light that
    /// leaves the plate edge at the steepest angle @f$\theta_{\max}@f$ reaches
    /// @f$D/2 + d\tan\theta_{\max}@f$ and re-enters from the opposite edge at
    /// @f$x - S@f$. It misses the picture @f$|x| \le D/2@f$ as long as
    /// @f$S \ge D + d\tan\theta_{\max}@f$, so the window is
    /// n = next_smooth(max(ceil(pad_min * active), ceil((D + d tan θ) / p))).
    /// With the defaults (512 px, 8 µm, 532 nm, 50 mm) this gives 720 px; at
    /// 256 px 480; at 100 mm 945. Wrapped light lands in the don't-care
    /// padding only, where nothing is evaluated.
    static Grid padded_for(int active, double pitch, double wavelength, double distance, double pad_min = 1.0);

    /// Physical side length of the window, n * pitch.
    double extent() const { return n * pitch; }
    /// Vacuum wavenumber @f$ k = 2\pi/\lambda @f$.
    double k() const;
    /// Coordinate of sample i along either axis.
    double x(std::size_t i) const { return (static_cast<double>(i) - static_cast<double>(n / 2)) * pitch; }
    /// Spatial frequency of FFT bin i along either axis (unshifted order).
    double f(std::size_t i) const;
    /// Largest propagation angle the grid can represent: the Nyquist
    /// frequency @f$ 1/(2p) @f$ maps to @f$ \sin\theta = \lambda/(2p) @f$
    /// (Goodman, eq. 3-66 @cite goodman2017), clamped to 90 degrees.
    double max_diffraction_angle() const;
};

}  // namespace doe
