// Grid: sampling geometry of the (padded) computational window.
//
// Conventions pinned here match the Python reference (optics.Grid):
//   coordinate of sample i      x_i = (i - n/2) * pitch          (n/2 = integer division)
//   spatial frequency of bin i  f_i = fftfreq(n, pitch)[i]        (unshifted, FFT order)
//   extent                      n * pitch
//   max diffraction angle       asin(min(1, lambda / (2 pitch)))  (Nyquist frequency)
//   padded(active, pad)         n = next_smooth(ceil(pad * active))
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/grid.hpp"
#include "doe/propagate.hpp"

#include <cmath>
#include <numbers>

using Catch::Matchers::WithinRel;

TEST_CASE("Grid: extent, wavenumber, coordinates, frequencies", "[grid]") {
    doe::Grid g{8, 8e-6, 532e-9};
    CHECK_THAT(g.extent(), WithinRel(64e-6, 1e-15));
    CHECK_THAT(g.k(), WithinRel(2.0 * std::numbers::pi / 532e-9, 1e-15));

    // x = (i - n/2) * pitch: -4..3 samples
    CHECK_THAT(g.x(0), WithinRel(-4 * 8e-6, 1e-15));
    CHECK_THAT(g.x(4), WithinRel(0.0, 1e-15));
    CHECK(g.x(4) == 0.0);
    CHECK_THAT(g.x(7), WithinRel(3 * 8e-6, 1e-15));

    // numpy.fft.fftfreq(8, d=pitch) = [0, 1, 2, 3, -4, -3, -2, -1] / (8 * pitch)
    const double df = 1.0 / (8 * 8e-6);
    CHECK(g.f(0) == 0.0);
    CHECK_THAT(g.f(1), WithinRel(df, 1e-15));
    CHECK_THAT(g.f(3), WithinRel(3 * df, 1e-15));
    CHECK_THAT(g.f(4), WithinRel(-4 * df, 1e-15));
    CHECK_THAT(g.f(7), WithinRel(-df, 1e-15));
}

TEST_CASE("Grid: odd size follows numpy conventions too", "[grid]") {
    doe::Grid g{7, 1e-6, 500e-9};
    // fftfreq(7) = [0, 1, 2, 3, -3, -2, -1] / (7 d)
    const double df = 1.0 / 7e-6;
    CHECK_THAT(g.f(3), WithinRel(3 * df, 1e-15));
    CHECK_THAT(g.f(4), WithinRel(-3 * df, 1e-15));
    // x: n//2 = 3 -> -3..3
    CHECK_THAT(g.x(0), WithinRel(-3e-6, 1e-15));
    CHECK(g.x(3) == 0.0);
}

TEST_CASE("Grid: maximum diffraction angle from the Nyquist frequency", "[grid]") {
    doe::Grid g{64, 8e-6, 532e-9};
    CHECK_THAT(g.max_diffraction_angle(), WithinRel(std::asin(532e-9 / 16e-6), 1e-15));
    doe::Grid fine{64, 0.2e-6, 532e-9};  // lambda / (2 pitch) > 1: clamp to 90 degrees
    CHECK_THAT(fine.max_diffraction_angle(), WithinRel(std::numbers::pi / 2, 1e-15));
}

TEST_CASE("Grid::padded rounds 2x the active aperture up to a 7-smooth size", "[grid]") {
    auto g = doe::Grid::padded(600, 8e-6, 532e-9);
    CHECK(g.n == 1200);
    CHECK(g.pitch == 8e-6);
    CHECK(g.wavelength == 532e-9);
    CHECK(doe::Grid::padded(513, 8e-6, 532e-9).n == 1029);  // ceil(1026) -> 3 * 7^3
    CHECK(doe::Grid::padded(512, 8e-6, 532e-9).n == 1024);
    CHECK(doe::Grid::padded(100, 8e-6, 532e-9, 1.5).n == 150);  // 2 3 5^2
}

TEST_CASE("Grid::padded_for: the window holds plate + d tan(theta), so wrapped light misses the picture; 7-smooth", "[grid][sampling]") {
    const double p = 8e-6, lam = 532e-9;
    // S >= D + d tan(theta): light from the plate edge wraps to x - S, which stays outside |x| <= D/2.
    CHECK(doe::Grid::padded_for(512, p, lam, 0.05).n == 720);   // 4.10 mm + 1.66 mm = 5.76 mm -> 720 px (was 1024 with the fixed factor 2)
    CHECK(doe::Grid::padded_for(256, p, lam, 0.05).n == 480);   // 2.05 mm + 1.66 mm -> 464 -> 480
    CHECK(doe::Grid::padded_for(512, p, lam, 0.10).n == 945);   // 4.10 mm + 3.33 mm -> 928 -> 945
    CHECK(doe::Grid::padded_for(32, p, lam, 0.02).n == 120);    // 0.26 mm + 0.67 mm -> 116 -> 120
    for (auto [N, d] : {std::pair{512, 0.05}, std::pair{256, 0.05}, std::pair{512, 0.1}, std::pair{32, 0.02}, std::pair{128, 0.3}}) {
        auto g = doe::Grid::padded_for(N, p, lam, d);
        CHECK(g.n >= N);
        CHECK(doe::sampling_report(g, d, N * p).wrap_misses_picture);
    }
    // a minimum padding factor still applies when it asks for more
    CHECK(doe::Grid::padded_for(512, p, lam, 0.05, 2.0).n == 1024);
    CHECK(doe::Grid::padded_for(512, p, lam, 0.05, 1.0).n == 720);
}
