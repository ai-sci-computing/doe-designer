/// @file image.hpp
/// @brief PNG I/O, grayscale targets, resampling, color maps and field rendering.
///
/// Layout conventions: an ImageU8 stores pixel (x, y) at
/// `data[(y * width + x) * channels + c]` (row-major by image rows). An
/// Array2 has its first index along x, so an Array2 of `rows x cols` renders
/// to an image of `width = rows`, `height = cols`.
#pragma once

#include "doe/array.hpp"

#include <complex>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace doe {

/// 8-bit image with 1 (gray), 3 (RGB) or 4 (RGBA) channels.
struct ImageU8 {
    std::size_t width = 0;      ///< pixels along x
    std::size_t height = 0;     ///< pixels along y
    std::size_t channels = 1;   ///< 1, 3 or 4
    std::vector<std::uint8_t> data;  ///< width * height * channels bytes
};

/// Read a PNG (any bit depth / color type; expanded to 8-bit gray, RGB or RGBA).
ImageU8 read_png(const std::filesystem::path& path);
/// Write an 8-bit PNG with 1, 3 or 4 channels.
void write_png(const std::filesystem::path& path, const ImageU8& image);

/// Grayscale target in [0, 1] with the (i = x, j = y) layout. RGB(A) images use
/// the Rec. 601 luma 0.299 R + 0.587 G + 0.114 B; alpha is ignored.
Array2<double> to_gray(const ImageU8& image);

/// Bilinear resampling to `rows x cols`, sample-aligned at both ends (the first
/// and last samples of each axis map onto each other), exact for linear data.
Array2<double> resample(const Array2<double>& src, std::size_t rows, std::size_t cols);

/// Center `src` in an `n x n` window of zeros (throws if it does not fit).
Array2<double> embed(const Array2<double>& src, std::size_t n);

/// Extract the centered `rows x cols` region (inverse of embed(); throws if larger than `a`).
Array2<double> crop_center(const Array2<double>& a, std::size_t rows, std::size_t cols);
/// Square crop.
inline Array2<double> crop_center(const Array2<double>& a, std::size_t n) { return crop_center(a, n, n); }
/// Centered square crop of a complex field.
Field<double> crop_center_field(const Field<double>& a, std::size_t n);

/// 8-bit RGB triple.
struct RGB {
    std::uint8_t r;  ///< red
    std::uint8_t g;  ///< green
    std::uint8_t b;  ///< blue
};

/// matplotlib's cyclic `twilight` color map at t in [0, 1] (clamped); t = 0 and t = 1 coincide.
RGB colormap_twilight(double t);
/// matplotlib's sequential `viridis` color map at t in [0, 1] (clamped).
RGB colormap_viridis(double t);
/// Complex value as a color: hue = phase (0 = red, 2 pi / 3 = green, -2 pi / 3 = blue),
/// saturation 1, value = |z| / amp_max (clamped). Zeros are black; a vortex is a point
/// around which the hue wheel closes once.
RGB hsv_phasor(std::complex<double> z, double amp_max);

/// Phase map with the twilight map: (phi + pi) / 2 pi -> [0, 1].
ImageU8 render_phase(const Array2<double>& phi);
/// Intensity with viridis, normalized to its maximum; with `log_scale` the
/// value is 10 log10(I / I_max) mapped from -floor_db..0 onto 0..1.
ImageU8 render_intensity(const Array2<double>& intensity, bool log_scale, double floor_db);
/// Complex field as HSV (hue = phase, value = amplitude / max amplitude).
ImageU8 render_hsv(const Field<double>& v);

}  // namespace doe
