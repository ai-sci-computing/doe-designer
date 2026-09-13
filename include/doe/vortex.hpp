/// @file vortex.hpp
/// @brief Optical vortex (phase singularity) counting on a sampled complex field.
#pragma once

#include "doe/array.hpp"
#include "doe/grid.hpp"

#include <cstddef>

namespace doe {

/// Topological charge of every plaquette of the field.
///
/// For each 2 x 2 cell (i, j), (i+1, j), (i+1, j+1), (i, j+1) the four phase
/// differences along the loop are wrapped to @f$(-\pi, \pi]@f$ and summed;
/// divided by @f$2\pi@f$ the result is an integer, the discrete curvature of
/// the phase. Goldstein, Zebker & Werner 1988
/// @cite goldstein1988 introduced this as the "residue": "the sum of the phase
/// differences clockwise around each set of four adjacent points: It is either
/// zero, plus one cycle, or minus one cycle". The points are the branch points
/// of the phase function of Fried & Vaughn 1992 @cite fried1992 (eq. 1: the
/// phase is @f$\mathrm{Arg}\,U + 2\pi n@f$; they sit where @f$|U| = 0@f$ and
/// carry the sign of the circulation), i.e. the optical vortices of Berry &
/// Dennis 2000 @cite berry2000 with charge
/// @f$\mathrm{sgn}(\xi_x\eta_y - \xi_y\eta_x)@f$ for @f$U = \xi + i\eta@f$
/// (their eq. 2.4). Loop orientation: (i, j) -> (i+1, j) -> (i+1, j+1) -> (i, j+1),
/// so a vortex with the phase of @f$(x - x_0) + i(y - y_0)@f$ has charge +1.
/// Plaquettes with a zero-amplitude corner are skipped (no phase there).
/// @return an (n-1) x (n-1) array of integer charges.
template <class T>
Array2<int> vortex_charge_map(const Field<T>& u);

/// Number of plaquettes with non-zero charge (counts a charge-2 vortex once per
/// plaquette it occupies).
std::size_t vortex_count(const Array2<int>& charges);

/// Vortices per unit area, @f$N / A@f$ with @f$A@f$ the area covered by the
/// plaquettes, @f$((n-1) p)^2@f$. For an isotropic random (speckle) field
/// with planar spectrum second moment @f$K_2@f$ the expectation is
/// @f$K_2 / (4\pi)@f$ (Berry & Dennis 2000 @cite berry2000 eq. 4.6), which
/// makes the density a direct speckle indicator.
template <class T>
double vortex_density(const Field<T>& u, const Grid& grid);

/// @cond INTERNAL
extern template Array2<int> vortex_charge_map<float>(const Field<float>&);
extern template Array2<int> vortex_charge_map<double>(const Field<double>&);
extern template double vortex_density<float>(const Field<float>&, const Grid&);
extern template double vortex_density<double>(const Field<double>&, const Grid&);
/// @endcond

}  // namespace doe
