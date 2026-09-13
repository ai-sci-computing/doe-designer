/// @file propagate.hpp
/// @brief Band-limited angular spectrum propagation between parallel planes.
///
/// The propagator @f$ A @f$ is the angular spectrum transfer function of
/// Goodman @cite goodman2017 §3.10 applied with orthonormal FFTs:
/// @f[
///   A u = \mathcal F^{-1}\big[\mathcal F u \cdot H\big], \qquad
///   H(f_x, f_y) = \exp\!\Big[i\,2\pi \frac{z}{\lambda}
///        \sqrt{1 - (\lambda f_x)^2 - (\lambda f_y)^2}\Big]
/// @f]
/// (eqs. 3-77, 3-78), with the evanescent region
/// @f$(\lambda f_x)^2 + (\lambda f_y)^2 > 1@f$ (eqs. 3-71, 3-72) set to zero.
/// Because the FFTs are unitary and @f$|H| \le 1@f$, @f$A@f$ is a partial
/// isometry and its adjoint is the same operator with @f$\bar H@f$
/// (CR calculus, @cite kreutzdelgado2009). That is what makes the Wirtinger
/// gradient in the design module cost one forward and one adjoint transform.
///
/// The band limit of Matsushima & Shimobaba @cite matsushima2009 removes the
/// part of @f$H@f$ that the frequency grid cannot sample; see
/// transfer_function().
#pragma once

#include "doe/array.hpp"
#include "doe/fft.hpp"
#include "doe/grid.hpp"

#include <complex>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace doe {

/// Transfer function of one propagation distance, always in double precision.
struct TransferFunction {
    Field<double> H;                    ///< @f$H(f_x,f_y)@f$ in FFT order, zero where cut
    Array2<unsigned char> evanescent;   ///< 1 where @f$(\lambda f_x)^2+(\lambda f_y)^2 \ge 1@f$
    double f_limit;                     ///< band-limit frequency (infinity when not applied)
};

/// Angular spectrum transfer function for one distance (single source of truth).
///
/// Both the single-plane propagator and the volume sweep call this; two copies
/// would invite the bug where the sweep freezes the band limit of the design
/// distance and silently disagrees with the propagator at intermediate planes.
///
/// **Band limit** (Matsushima & Shimobaba 2009 @cite matsushima2009). The
/// local frequency of the phase of @f$H@f$ along @f$u@f$ is
/// @f$ f_u = -z u / \sqrt{\lambda^{-2} - u^2} @f$ (eq. 11). Sampling the
/// transfer function at intervals @f$\Delta u@f$ requires
/// @f$ 1/(2\Delta u) \ge |f_u| @f$ (eq. 12), which bounds the usable
/// frequencies by
/// @f[
///   u_{\text{limit}} = \frac{1}{\lambda\sqrt{(2\Delta u\, z)^2 + 1}} \qquad \text{(eq. 13)},
/// @f]
/// and @f$H@f$ is multiplied by @f$\mathrm{rect}(u / 2u_{\text{limit}})@f$
/// in each direction (eq. 14). The paper is explicit that
/// @f$\Delta u = 1/(2 S_x)@f$ with @f$S_x@f$ the *unpadded* source extent,
/// "because the source sampling area is doubled" (§2.2); with @f$S@f$ the
/// extent of this (padded) grid that is @f$\Delta u = 1/S@f$, so
/// @f$ f_{\text{lim}} = 1 / (\lambda \sqrt{(2 z / S)^2 + 1}) @f$.
/// (The Python reference used @f$2z/(S/2)@f$ here, a passband up to 2x
/// narrower at long range.)
///
/// **Precision.** @f$k_z z@f$ is a large phase (@f$5.9\cdot10^5@f$ rad per
/// 5 cm at 532 nm); it is always evaluated in double and only cast to the
/// working precision afterwards, otherwise float32 loses the angle before
/// `exp` ever runs.
///
/// @param grid       sampling grid of the padded window
/// @param distance   propagation distance in meters (negative propagates backwards)
/// @param band_limit apply the Matsushima–Shimobaba cut (skipped for distance 0)
TransferFunction transfer_function(const Grid& grid, double distance, bool band_limit = true);

/// The distance-independent part of the transfer function on a grid: the axial
/// wavenumber @f$k_z = \frac{2\pi}{\lambda}\sqrt{1 - (\lambda f_x)^2 - (\lambda f_y)^2}@f$
/// per bin and the evanescent mask (Goodman eq. 3-72). transfer_function() is
/// this map turned into @f$H = e^{i k_z z}@f$ with the band limit for one
/// distance, so a sweep over many planes builds the map once (the square root
/// per bin) and pays only the complex exponential per plane.
struct KzMap {
    Array2<double> kz;                 ///< @f$k_z@f$ per FFT bin, 0 where evanescent
    Array2<unsigned char> evanescent;  ///< 1 where @f$(\lambda f)^2 \ge 1@f$
};
/// Compute the KzMap of a grid.
KzMap kz_map(const Grid& grid);
/// transfer_function() from a precomputed map (identical result, cheaper per distance).
TransferFunction transfer_function(const KzMap& kz, const Grid& grid, double distance, bool band_limit = true);

/// Band-limited angular spectrum propagator over a fixed distance.
/// @tparam T working precision of the fields (float or double).
template <class T>
class AngularSpectrum {
public:
    /// Build the transfer function for `distance` on `grid` (see transfer_function()).
    AngularSpectrum(const Grid& grid, double distance, bool band_limit = true);

    /// Sampling grid the operator was built for.
    const Grid& grid() const { return grid_; }
    /// Propagation distance in meters.
    double distance() const { return distance_; }
    /// Whether the Matsushima–Shimobaba band limit is applied.
    bool band_limit() const { return band_limit_; }
    /// Transfer function in the working precision (built in double, then cast).
    const Field<T>& transfer() const { return H_; }
    /// Fraction of frequency bins that are evanescent.
    double evanescent_fraction() const { return evanescent_fraction_; }
    /// Fraction of frequency bins with non-zero transfer function.
    double passband_fraction() const { return passband_fraction_; }

    /// @f$ v = A u = \mathcal F^{-1}[\mathcal F u \cdot H] @f$.
    Field<T> forward(const Field<T>& u) const;
    /// @f$ A^H v = \mathcal F^{-1}[\mathcal F v \cdot \bar H] @f$.
    Field<T> adjoint(const Field<T>& v) const;
    /// Propagate one field to many planes, sharing a single forward FFT:
    /// n_z + 1 transforms instead of 2 n_z. Each plane rebuilds its own
    /// transfer function through transfer_function() (same band limit flag).
    std::vector<Field<T>> volume(const Field<T>& u, std::span<const double> distances) const;
    /// Streaming variant of volume(): the same planes, handed one at a time to
    /// `on_plane(index, field)` so a long sweep never holds all planes in memory.
    void sweep(const Field<T>& u, std::span<const double> distances,
               const std::function<void(std::size_t, const Field<T>&)>& on_plane) const;

    /// Same as forward().
    Field<T> operator()(const Field<T>& u) const { return forward(u); }

private:
    Grid grid_;
    double distance_;
    bool band_limit_;
    Field<T> H_, Hc_;
    double evanescent_fraction_ = 0.0, passband_fraction_ = 1.0;
    Fft2<T> fft_;
};

/// @cond INTERNAL
extern template class AngularSpectrum<float>;
extern template class AngularSpectrum<double>;
/// @endcond

/// Diagnostics that catch the usual silent failures before optimization.
struct SamplingReport {
    double max_angle_rad;      ///< largest angle the grid represents, @f$\arcsin(\lambda/2p)@f$
    double spot_size_m;        ///< diffraction-limited spot at the target, @f$\lambda d / D@f$
    double spot_size_px;       ///< the same in target pixels
    double signal_spread_m;    ///< full first-order spread, @f$2 d \tan\theta_{\max}@f$
    double window_extent_m;    ///< padded window width
    double fresnel_number;     ///< @f$D^2 / (\lambda d)@f$ (spec convention; textbooks use @f$a^2@f$)
    bool wrap_misses_picture;  ///< D + d tan θ <= S: light wrapping around the periodic window lands outside the picture (Grid::padded_for)
    bool spot_resolved;        ///< spot <= one pixel: finer target detail is unreachable otherwise
};

/// @param grid     padded grid
/// @param distance DOE-to-target distance in meters
/// @param aperture physical width of the illuminated (active) DOE region in meters
SamplingReport sampling_report(const Grid& grid, double distance, double aperture);

}  // namespace doe
