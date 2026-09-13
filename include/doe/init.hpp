/// @file init.hpp
/// @brief Starting phases for the optimization: transport-of-intensity or random.
#pragma once

#include "doe/array.hpp"
#include "doe/grid.hpp"
#include "doe/propagate.hpp"

namespace doe {

/// Solve @f$\nabla^2\varphi = \text{rhs}@f$ on the periodic grid spectrally.
///
/// @f$\hat\varphi(f) = \widehat{\text{rhs}}(f) / (-(2\pi)^2 |f|^2)@f$. The
/// right-hand side is projected to zero mean first (otherwise the periodic
/// problem has no solution) and the DC component of @f$\varphi@f$ is set to
/// zero. Exact for band-limited data (the manufactured-solution test T6
/// reaches rounding level).
Array2<double> poisson_periodic(const Grid& grid, const Array2<double>& rhs);

/// Transport-of-intensity initialization (Teague 1983 @cite teague1983).
///
/// Teague's eq. (4), for @f$u = \sqrt{I}\,e^{i\varphi}@f$ and the
/// @f$e^{+ikz}@f$ carrier of his eq. (1), which is also the convention of
/// our transfer function:
/// @f[ \frac{2\pi}{\lambda}\frac{\partial I}{\partial z} = -\nabla\cdot(I\nabla\varphi). @f]
/// With uniform intensity @f$I_0@f$ on the DOE and the finite difference
/// @f$\partial I/\partial z \approx (I_{\text{target}} - I_0)/d@f$ this is the
/// Poisson problem (see also Paganin & Nugent 1998
/// @cite paganin1998, Zuo et al. 2020 @cite zuo2020)
/// @f[ \nabla^2\varphi = \frac{k}{d}\Big(1 - \frac{I_{\text{target}}}{I_0}\Big). @f]
/// The target is rescaled to the source energy (only its shape matters) and
/// the source is floored at @f$10^{-6}\max I_0@f$ so the ratio stays finite
/// in the padding.
///
/// As a DOE on its own the result is poor; as a starting point it puts the
/// optimizer in a basin with far fewer optical vortices than a random phase.
/// Vortices are topologically protected and cannot be removed later, and they
/// are where most speckle comes from (Senthilkumaran et al. 2005
/// @cite senthilkumaran2005).
template <class T>
Array2<T> init_tie(const Grid& grid, double distance, const Array2<T>& i_target, const Array2<T>& i_source);

/// Back-propagation initialization: the phase of @f$A^H b@f$, i.e. the target
/// amplitude sent back to the DOE plane with a flat phase and projected onto
/// unit modulus. This is the classical first half-cycle of Gerchberg–Saxton
/// @cite gerchberg1972 / Fienup @cite fienup1980 started from the target
/// (Wyrowski 1990 @cite wyrowski1990 eq. 13 uses the target with a random
/// phase instead). It is the right start for full-field targets such as
/// photographs, where TIE's uniform-background assumption fails: measured on a
/// photograph it puts the optimizer in a basin with 10x fewer vortices inside
/// the image and gains 8 dB over TIE (README, natural images).
template <class T>
Array2<T> init_backprop(const Grid& grid, const AngularSpectrum<T>& prop, const Array2<double>& b);

/// Uniform random phase in @f$(-\pi, \pi]@f$ from a seeded 64-bit Mersenne
/// twister (deterministic for a given seed and grid size).
template <class T>
Array2<T> init_random(const Grid& grid, unsigned long long seed);

/// @cond INTERNAL
extern template Array2<float> init_tie<float>(const Grid&, double, const Array2<float>&, const Array2<float>&);
extern template Array2<double> init_tie<double>(const Grid&, double, const Array2<double>&, const Array2<double>&);
extern template Array2<float> init_backprop<float>(const Grid&, const AngularSpectrum<float>&, const Array2<double>&);
extern template Array2<double> init_backprop<double>(const Grid&, const AngularSpectrum<double>&, const Array2<double>&);
extern template Array2<float> init_random<float>(const Grid&, unsigned long long);
extern template Array2<double> init_random<double>(const Grid&, unsigned long long);
/// @endcond

}  // namespace doe
