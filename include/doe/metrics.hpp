/// @file metrics.hpp
/// @brief Figures of merit of a reconstruction on the signal window.
#pragma once

#include "doe/array.hpp"

namespace doe {

/// Standard DOE figures of merit. Efficiency and RMSE are
/// reported separately on purpose: they are in tension, and a combined score
/// hides which trade-off a design makes. A scale-invariant objective always
/// needs an independent metric that measures the scale itself (efficiency);
/// otherwise convergence is checked against exactly the quantity that was
/// normalized away.
struct Metrics {
    /// Energy in the signal window over total energy (diffraction efficiency,
    /// Wyrowski & Bryngdahl 1988 @cite wyrowski1988).
    double efficiency;
    /// RMS amplitude error on the window after least-squares scaling, divided
    /// by the peak of the scaled target.
    double amplitude_rmse;
    /// Speckle contrast @f$\sigma_I/\langle I\rangle@f$ over the pixels that
    /// should be bright (@f$b > 0.5\max b@f$): 1 for fully developed speckle
    /// (Goodman @cite goodman2007 ch. 2-3; Aagedal et al. 1996 @cite aagedal1996).
    /// NaN when there are no bright pixels.
    double speckle_contrast;
    /// Pearson correlation of the intensities @f$|v|^2@f$ and @f$(s b)^2@f$
    /// on the window; NaN when either is constant (undefined).
    double ncc;
    /// @f$10\log_{10}(\text{peak}^2/\text{MSE})@f$ of the intensities on the
    /// window, peak = maximum scaled target intensity; +inf for a perfect match.
    double psnr_db;
    /// Optical vortices (non-zero plaquette residues, vortex_charge_map()) whose
    /// plaquette lies inside the window.
    std::size_t vortex_count;
    /// Vortices whose plaquette lies inside the bright features (@f$b > 0.5\max b@f$
    /// at all four corners).
    std::size_t vortex_count_bright;
    /// vortex_count_bright per bright plaquette area (pitch = 1 here; multiply by
    /// @f$1/p^2@f$ for a physical density). Speckle in the signal is what this
    /// measures; vortices in the dark surround do not disturb the image.
    double vortex_density_bright;
};

/// Compute the metrics of the target-plane field `v` against the target
/// amplitude `b` on the window `mask`. All sums accumulate in double.
template <class T>
Metrics metrics(const Field<T>& v, const Array2<T>& b, const Array2<T>& mask);

/// @cond INTERNAL
extern template Metrics metrics<float>(const Field<float>&, const Array2<float>&, const Array2<float>&);
extern template Metrics metrics<double>(const Field<double>&, const Array2<double>&, const Array2<double>&);
/// @endcond

}  // namespace doe
