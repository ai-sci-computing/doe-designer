#include "doe/propagate.hpp"

#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace doe {

KzMap kz_map(const Grid& grid) {
    const std::size_t n = static_cast<std::size_t>(grid.n);
    const double lam = grid.wavelength;
    KzMap m{Array2<double>(n, n, 0.0), Array2<unsigned char>(n, n, 0)};
    par::for_each(n, [&](std::size_t i) {
        const double fx = grid.f(i);
        for (std::size_t j = 0; j < n; ++j) {
            const double fy = grid.f(j);
            // Goodman eq. (3-78): exponent 2 pi (z / lambda) sqrt(1 - (lambda fx)^2 - (lambda fy)^2)
            const double arg = 1.0 - (lam * fx) * (lam * fx) - (lam * fy) * (lam * fy);
            if (arg <= 0.0) {  // evanescent (eq. 3-72): carries no energy away, dropped
                m.evanescent(i, j) = 1;
                continue;
            }
            m.kz(i, j) = 2.0 * std::numbers::pi / lam * std::sqrt(arg);
        }
    });
    return m;
}

TransferFunction transfer_function(const KzMap& kz, const Grid& grid, double distance, bool band_limit) {
    const std::size_t n = static_cast<std::size_t>(grid.n);
    const double lam = grid.wavelength;
    TransferFunction tf{Field<double>(n, n), kz.evanescent, std::numeric_limits<double>::infinity()};

    // Matsushima & Shimobaba 2009 eq. (13) with du = 1 / S (S = padded extent):
    // f_lim = 1 / (lambda sqrt((2 z / S)^2 + 1)). Skipped for zero distance
    // (H == 1, nothing to alias).
    if (band_limit && distance != 0.0) {
        const double S = grid.extent();
        tf.f_limit = 1.0 / (lam * std::sqrt(std::pow(2.0 * std::abs(distance) / S, 2) + 1.0));
    }

    par::for_each(n, [&](std::size_t i) {
        const double fx = grid.f(i);
        for (std::size_t j = 0; j < n; ++j) {
            const double fy = grid.f(j);
            if (kz.evanescent(i, j)) {
                tf.H(i, j) = 0.0;
                continue;
            }
            if (std::abs(fx) > tf.f_limit || std::abs(fy) > tf.f_limit) {  // eq. (14) rect window
                tf.H(i, j) = 0.0;
                continue;
            }
            tf.H(i, j) = std::polar(1.0, kz.kz(i, j) * distance);
        }
    });
    return tf;
}

TransferFunction transfer_function(const Grid& grid, double distance, bool band_limit) {
    return transfer_function(kz_map(grid), grid, distance, band_limit);
}

template <class T>
AngularSpectrum<T>::AngularSpectrum(const Grid& grid, double distance, bool band_limit)
    : grid_(grid), distance_(distance), band_limit_(band_limit),
      H_(static_cast<std::size_t>(grid.n), static_cast<std::size_t>(grid.n)),
      Hc_(static_cast<std::size_t>(grid.n), static_cast<std::size_t>(grid.n)),
      fft_(static_cast<std::size_t>(grid.n)) {
    const TransferFunction tf = transfer_function(grid, distance, band_limit);
    std::size_t n_evan = 0, n_pass = 0;
    for (std::size_t k = 0; k < tf.H.data.size(); ++k) {
        H_.data[k] = std::complex<T>(tf.H.data[k]);
        Hc_.data[k] = std::conj(H_.data[k]);
        n_evan += tf.evanescent.data[k];
        n_pass += (std::abs(tf.H.data[k]) > 0.0);
    }
    evanescent_fraction_ = static_cast<double>(n_evan) / static_cast<double>(tf.H.data.size());
    passband_fraction_ = static_cast<double>(n_pass) / static_cast<double>(tf.H.data.size());
}

template <class T>
Field<T> AngularSpectrum<T>::forward(const Field<T>& u) const {
    Field<T> U = fft_.forward(u);
    par::for_each(U.data.size(), [&](std::size_t k) { U.data[k] *= H_.data[k]; });
    return fft_.inverse(U);
}

template <class T>
Field<T> AngularSpectrum<T>::adjoint(const Field<T>& v) const {
    Field<T> V = fft_.forward(v);
    par::for_each(V.data.size(), [&](std::size_t k) { V.data[k] *= Hc_.data[k]; });
    return fft_.inverse(V);
}

template <class T>
std::vector<Field<T>> AngularSpectrum<T>::volume(const Field<T>& u, std::span<const double> distances) const {
    const Field<T> U = fft_.forward(u);
    std::vector<Field<T>> planes;
    planes.reserve(distances.size());
    Field<T> W(U.rows, U.cols);
    const KzMap kz = kz_map(grid_);  // the square roots once, the exponentials per plane
    for (double d : distances) {
        const TransferFunction tf = transfer_function(kz, grid_, d, band_limit_);
        par::for_each(U.data.size(), [&](std::size_t k) { W.data[k] = U.data[k] * std::complex<T>(tf.H.data[k]); });
        planes.push_back(fft_.inverse(W));
    }
    return planes;
}

template <class T>
void AngularSpectrum<T>::sweep(const Field<T>& u, std::span<const double> distances,
                               const std::function<void(std::size_t, const Field<T>&)>& on_plane) const {
    const Field<T> U = fft_.forward(u);
    Field<T> W(U.rows, U.cols), plane(U.rows, U.cols);
    const KzMap kz = kz_map(grid_);
    for (std::size_t p = 0; p < distances.size(); ++p) {
        const TransferFunction tf = transfer_function(kz, grid_, distances[p], band_limit_);
        par::for_each(U.data.size(), [&](std::size_t k) { W.data[k] = U.data[k] * std::complex<T>(tf.H.data[k]); });
        fft_.inverse(W, plane);
        on_plane(p, plane);
    }
}

template class AngularSpectrum<float>;
template class AngularSpectrum<double>;

SamplingReport sampling_report(const Grid& grid, double distance, double aperture) {
    const double lam = grid.wavelength;
    const double theta = grid.max_diffraction_angle();
    SamplingReport r{};
    r.max_angle_rad = theta;
    r.spot_size_m = lam * distance / std::max(aperture, 1e-12);
    r.spot_size_px = r.spot_size_m / grid.pitch;
    r.signal_spread_m = 2.0 * distance * std::tan(theta);
    r.window_extent_m = grid.extent();
    r.fresnel_number = aperture * aperture / (lam * std::max(distance, 1e-12));
    // Light leaving the plate edge reaches D/2 + d tan(theta); the periodic transform wraps
    // it to x - S, which misses the picture |x| <= D/2 while D + d tan(theta) <= S.
    r.wrap_misses_picture = aperture + 0.5 * r.signal_spread_m <= r.window_extent_m * (1 + 1e-12);
    r.spot_resolved = r.spot_size_px <= 1.0;
    return r;
}

}  // namespace doe
