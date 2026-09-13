// Figures of merit on the signal window: diffraction
// efficiency, amplitude RMSE after optimal scaling, speckle contrast
// sigma/mean over the bright pixels (Goodman, Speckle Phenomena; Aagedal 1996),
// normalized cross-correlation of intensities, PSNR.
// Efficiency and RMSE are reported separately: they are in tension, and a
// combined score would hide which trade-off a design makes.
// Oracle values: design.metrics() on the field of scratchpad/oracle_energy.py.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/energy.hpp"
#include "doe/grid.hpp"
#include "doe/metrics.hpp"
#include "doe/propagate.hpp"

#include <cmath>
#include <numbers>
#include <random>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
constexpr double pi = std::numbers::pi;

namespace {
// window = central half; b = 1 on the window (uniform target), 0 outside
struct Setup {
    std::size_t n = 32;
    doe::Array2<double> b{32, 32, 0.0}, mask{32, 32, 0.0};
    Setup() {
        for (std::size_t i = 8; i < 24; ++i)
            for (std::size_t j = 8; j < 24; ++j) b(i, j) = mask(i, j) = 1.0;
    }
};
}  // namespace

TEST_CASE("metrics: a perfect reconstruction of a uniform target: efficiency 1, RMSE 0, PSNR inf, speckle 0", "[metrics]") {
    Setup S;
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> uni(-pi, pi);
    doe::Field<double> v(S.n, S.n);
    for (std::size_t k = 0; k < v.data.size(); ++k) v.data[k] = std::polar(2.0 * S.b.data[k], uni(rng));  // scaled copy
    auto m = doe::metrics(v, S.b, S.mask);
    CHECK_THAT(m.efficiency, WithinAbs(1.0, 1e-15));
    CHECK_THAT(m.amplitude_rmse, WithinAbs(0.0, 1e-12));  // |polar(2b, theta)| = 2b to rounding
    CHECK(std::isnan(m.ncc));  // Pearson correlation against a constant target is undefined
    CHECK(m.psnr_db > 250.0);   // +inf for an exact match, ~ 300 dB at rounding level
    CHECK_THAT(m.speckle_contrast, WithinAbs(0.0, 1e-15));
}

TEST_CASE("metrics: a perfect reconstruction of a structured target: NCC 1, RMSE 0", "[metrics]") {
    Setup S;
    auto b = S.b;
    for (std::size_t i = 8; i < 24; ++i)
        for (std::size_t j = 8; j < 24; ++j) b(i, j) = 1.0 + 0.5 * std::cos(2 * pi * double(i) / 8);
    doe::Field<double> v(S.n, S.n);
    for (std::size_t k = 0; k < v.data.size(); ++k) v.data[k] = std::polar(0.3 * b.data[k], 0.7);
    auto m = doe::metrics(v, b, S.mask);
    CHECK_THAT(m.ncc, WithinAbs(1.0, 1e-12));
    CHECK_THAT(m.amplitude_rmse, WithinAbs(0.0, 1e-12));
    CHECK(m.psnr_db > 250.0);
}

TEST_CASE("metrics: efficiency is the window fraction of the energy", "[metrics]") {
    Setup S;
    doe::Field<double> v(S.n, S.n);
    for (std::size_t k = 0; k < v.data.size(); ++k) v.data[k] = S.b.data[k];
    v(0, 0) = std::sqrt(256.0);  // as much energy outside as inside (16x16 window of ones)
    auto m = doe::metrics(v, S.b, S.mask);
    CHECK_THAT(m.efficiency, WithinRel(0.5, 1e-15));
    CHECK_THAT(m.amplitude_rmse, WithinAbs(0.0, 1e-15));  // outside pixels do not enter the fit
}

TEST_CASE("metrics: speckle contrast is sigma/mean of the intensity on the bright pixels", "[metrics]") {
    Setup S;
    doe::Field<double> v(S.n, S.n);
    // intensities alternate 1 and 3 on the window: mean 2, population std 1 -> contrast 0.5
    std::size_t k = 0;
    for (std::size_t i = 8; i < 24; ++i)
        for (std::size_t j = 8; j < 24; ++j, ++k) v(i, j) = std::sqrt((k % 2) ? 3.0 : 1.0);
    auto m = doe::metrics(v, S.b, S.mask);
    CHECK_THAT(m.speckle_contrast, WithinRel(0.5, 1e-14));
    // bright pixels are those with b > 0.5 max(b): dim target pixels do not count
    auto b2 = S.b;
    for (std::size_t i = 8; i < 24; ++i)
        for (std::size_t j = 8; j < 16; ++j) b2(i, j) = 0.4;
    doe::Field<double> v2(S.n, S.n);
    for (std::size_t i = 8; i < 24; ++i)
        for (std::size_t j = 8; j < 24; ++j) v2(i, j) = (j < 16) ? 5.0 : 1.0;  // dim half is wildly off, bright half uniform
    CHECK_THAT(doe::metrics(v2, b2, S.mask).speckle_contrast, WithinAbs(0.0, 1e-15));
}

TEST_CASE("metrics: PSNR and RMSE against a known deviation", "[metrics]") {
    Setup S;
    doe::Field<double> v(S.n, S.n);
    for (std::size_t k = 0; k < v.data.size(); ++k) v.data[k] = S.b.data[k];
    v(8, 8) = 1.5;  // one pixel 50 % too bright in amplitude
    auto m = doe::metrics(v, S.b, S.mask);
    // optimal scale s = sum b|v| / sum b^2 = (255 + 1.5) / 256
    const double s = 256.5 / 256.0;
    // amplitude rmse normalized by the peak of s*b
    double err = 0;
    for (std::size_t i = 8; i < 24; ++i)
        for (std::size_t j = 8; j < 24; ++j) err += std::pow(std::abs(v(i, j)) - s, 2);
    CHECK_THAT(m.amplitude_rmse, WithinRel(std::sqrt(err / 256) / s, 1e-12));
    // PSNR on intensities: peak = s^2, mse over the window
    double mse = 0;
    for (std::size_t i = 8; i < 24; ++i)
        for (std::size_t j = 8; j < 24; ++j) mse += std::pow(std::norm(v(i, j)) - s * s, 2);
    mse /= 256;
    CHECK_THAT(m.psnr_db, WithinRel(10 * std::log10(s * s * s * s / mse), 1e-12));
}

TEST_CASE("oracle: metrics match design.py on the oracle field", "[metrics][oracle]") {
    const std::size_t n = 32;
    doe::Grid g{32, 8e-6, 532e-9};
    doe::AngularSpectrum<double> prop(g, 0.02, false);
    doe::Array2<double> phi(n, n), illum(n, n, 1.0), b(n, n), mask(n, n, 0.0);
    const double p = g.pitch;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double x = g.x(i), y = g.x(j);
            phi(i, j) = std::sin(3 * x / p / n * 2 * pi) + 0.5 * std::cos(5 * y / p / n * 2 * pi) + 0.2 * (x / p) * (y / p) / n;
            b(i, j) = 0.5 + 0.5 * std::cos(2 * pi * x / (8 * p)) * std::cos(2 * pi * y / (8 * p));
            if (i < 4 || i >= n - 4 || j < 4 || j >= n - 4) illum(i, j) = 0.0;
            if (i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4) mask(i, j) = 1.0;
        }
    auto r = doe::energy_and_grad<double>(phi, illum, prop, b, mask, {1.0, 0.3});
    auto m = doe::metrics(r.field, b, mask);
    CHECK_THAT(m.efficiency, WithinRel(0.4177829949550023, 1e-12));
    CHECK_THAT(m.amplitude_rmse, WithinRel(0.3808923489557326, 1e-12));
    // 1e-11: the reference divides by (mean + 1e-12); this implementation divides by mean.
    CHECK_THAT(m.speckle_contrast, WithinRel(0.9297602808702286, 1e-11));
    CHECK_THAT(m.ncc, WithinRel(0.006992360649502711, 1e-9));
    CHECK_THAT(m.psnr_db, WithinRel(6.692611635752258, 1e-12));
}

TEST_CASE("metrics: vortex statistics on the window and inside the bright features", "[metrics][vortex]") {
    // Two vortices inside the bright square (8..23), one in the dark part of a
    // larger window, none outside the window.
    const std::size_t n = 32;
    doe::Array2<double> b(n, n, 0.0), mask(n, n, 0.0);
    for (std::size_t i = 4; i < 28; ++i)
        for (std::size_t j = 4; j < 28; ++j) mask(i, j) = 1.0;
    for (std::size_t i = 8; i < 24; ++i)
        for (std::size_t j = 8; j < 24; ++j) b(i, j) = 1.0;
    doe::Field<double> v(n, n, std::complex<double>(1.0, 0.0));
    auto add_vortex = [&](double x0, double y0, int m) {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                std::complex<double> z(double(i) - x0, double(j) - y0);
                v(i, j) *= std::pow(z / std::abs(z), m);
            }
    };
    add_vortex(12.4, 15.6, +1);
    add_vortex(19.3, 10.7, -1);
    add_vortex(5.5, 25.3, +1);   // in the window, dark
    add_vortex(1.4, 1.6, +1);    // outside the window
    auto m = doe::metrics(v, b, mask);
    CHECK(m.vortex_count == 3);
    CHECK(m.vortex_count_bright == 2);
    // density = count / (bright plaquette area): 16 x 16 bright pixels = 15 x 15 plaquettes, unit pitch here
    CHECK_THAT(m.vortex_density_bright, WithinRel(2.0 / 225.0, 1e-12));
}

TEST_CASE("metrics: single precision field, double accumulation", "[metrics][precision]") {
    Setup S;
    doe::Field<float> v(S.n, S.n);
    doe::Array2<float> b(S.n, S.n), mask(S.n, S.n);
    for (std::size_t k = 0; k < v.data.size(); ++k) {
        v.data[k] = std::complex<float>(float(S.b.data[k]) * 0.7f, 0.f);
        b.data[k] = float(S.b.data[k]);
        mask.data[k] = float(S.mask.data[k]);
    }
    auto m = doe::metrics(v, b, mask);
    CHECK_THAT(m.efficiency, WithinAbs(1.0, 1e-7));
    CHECK_THAT(m.amplitude_rmse, WithinAbs(0.0, 1e-7));
}
