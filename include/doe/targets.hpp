/// @file targets.hpp
/// @brief Synthetic grayscale targets generated in code (no external files needed).
#pragma once

#include "doe/array.hpp"

#include <string>

namespace doe::targets {

/// The spec's test target: a cross of two bars plus a ring, binary, on an
/// n x n canvas. Geometry scales with n: bar half-width 6/256 n, bar
/// half-length 60/256 n, ring radii 80/256 n .. 92/256 n. Coordinates are
/// measured from the pixel-center origin (n-1)/2 so the image is symmetric
/// under i -> n-1-i and under i <-> j.
Array2<double> cross_plus_ring(std::size_t n);

/// nx x ny equal-intensity disks of radius `spot_radius_px` on a regular
/// lattice with pitch `spacing * n`, centered: a beam-splitter / fan-out target
/// (Dammann & Gortler 1971 @cite dammann1971; Krackhardt & Streibl 1989 @cite krackhardt1989).
Array2<double> spot_array(std::size_t n, int nx, int ny, double spot_radius_px, double spacing);

/// Centered disk of radius `radius_fraction * n`.
Array2<double> disk(std::size_t n, double radius_fraction);

/// Binary stripe grating along x with the given period in pixels (50 % duty cycle).
Array2<double> grating(std::size_t n, int period_px);

/// Text rendered from a built-in 5x7 font (A-Z, 0-9, space, + - . :), scaled
/// to fill 80 % of the width and centered. Throws for glyphs not in the font.
Array2<double> binary_logo(std::size_t n, const std::string& text);

/// Gaussian blur with standard deviation `sigma_px` (separable, kernel
/// truncated at 4 sigma and renormalized, so the total is conserved away from
/// the border). Softens the hard edges that a diffraction-limited spot cannot
/// reproduce anyway. `sigma_px <= 0` returns the input unchanged.
Array2<double> soft_edges(const Array2<double>& target, double sigma_px);

}  // namespace doe::targets
