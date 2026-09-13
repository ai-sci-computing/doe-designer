// Initialization of the phase: transport-of-intensity (TIE) and random.
//
// Teague 1983 eq. (4) (convention u = sqrt(I) exp(i phi), carrier exp(+ikz) as
// in Teague eq. (1) and in our transfer function):
//     (2 pi / lambda) dI/dz = - div( I grad phi ).
// For uniform intensity I0 over the DOE and dI/dz ~ (I_target - I0) / d this
// is the Poisson equation
//     laplacian(phi) = (k / d) (1 - I_target / I0)
// solved spectrally on the periodic grid: phi_hat = rhs_hat / (-(2 pi f)^2),
// after projecting rhs to zero mean (solvability under periodic BC), DC of phi = 0.
//
// T6: manufactured solution, rel. dev. < 1e-8.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/grid.hpp"
#include "doe/init.hpp"
#include "doe/propagate.hpp"

#include <cmath>
#include <numbers>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
constexpr double pi = std::numbers::pi;

TEST_CASE("T6 poisson_periodic: manufactured solution sin(2 pi x/L) cos(4 pi y/L)", "[init][T6]") {
    const std::size_t n = 48;
    doe::Grid g{48, 8e-6, 532e-9};
    const double L = g.extent();
    doe::Array2<double> phi_exact(n, n), rhs(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double x = g.x(i), y = g.x(j);
            phi_exact(i, j) = std::sin(2 * pi * x / L) * std::cos(4 * pi * y / L);
            // laplacian = -( (2 pi/L)^2 + (4 pi/L)^2 ) phi
            rhs(i, j) = -(std::pow(2 * pi / L, 2) + std::pow(4 * pi / L, 2)) * phi_exact(i, j);
        }
    auto phi = doe::poisson_periodic(g, rhs);
    double num = 0, den = 0;
    for (std::size_t k = 0; k < phi.data.size(); ++k) {
        num += std::pow(phi.data[k] - phi_exact.data[k], 2);
        den += std::pow(phi_exact.data[k], 2);
    }
    CHECK(std::sqrt(num / den) < 1e-8);  // spectral: rounding level in practice
}

TEST_CASE("poisson_periodic: a non-zero-mean rhs is projected, the solution has zero mean", "[init][T6]") {
    const std::size_t n = 32;
    doe::Grid g{32, 8e-6, 532e-9};
    doe::Array2<double> rhs(n, n, 3.0);  // constant offset: no periodic solution unless projected
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) rhs(i, j) += std::cos(2 * pi * double(i) / n) * std::cos(4 * pi * double(j) / n);
    auto phi = doe::poisson_periodic(g, rhs);
    double mean = 0;
    for (double x : phi.data) mean += x;
    CHECK_THAT(mean / double(phi.data.size()), WithinAbs(0.0, 1e-12));
    // and the solution reproduces the projected rhs: laplacian(phi) == rhs - mean(rhs)
    double rhs_mean = 0;
    for (double x : rhs.data) rhs_mean += x;
    rhs_mean /= double(rhs.data.size());
    const double h = g.pitch;
    double err = 0, ref = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            // the spectral solution is exact for the spectral Laplacian; the 5-point
            // stencil is second order, (k h)^2 / 12 ~ 1.3 % for the 4 pi mode
            const double lap = (phi((i + 1) % n, j) + phi((i + n - 1) % n, j) + phi(i, (j + 1) % n) + phi(i, (j + n - 1) % n) - 4 * phi(i, j)) / (h * h);
            err += std::pow(lap - (rhs(i, j) - rhs_mean), 2);
            ref += std::pow(rhs(i, j) - rhs_mean, 2);
        }
    CHECK(std::sqrt(err / ref) < 2e-2);
}

TEST_CASE("init_tie: the Poisson right-hand side follows Teague eq. (4) with the +ikz convention", "[init][TIE]") {
    // A target brighter than the source at the center must yield a phase with
    // negative curvature there (a converging lens is phi = -k r^2 / 2f for the
    // exp(+ikz) carrier), and forward propagation by d must then concentrate
    // energy toward the center.
    const int active = 64;
    auto g = doe::Grid::padded(active, 8e-6, 532e-9);
    const std::size_t n = static_cast<std::size_t>(g.n);
    const double d = 0.02;
    doe::Array2<double> i_source(n, n, 0.0), i_target(n, n, 0.0);
    const std::size_t lo = (n - active) / 2;
    for (std::size_t i = lo; i < lo + active; ++i)
        for (std::size_t j = lo; j < lo + active; ++j) {
            i_source(i, j) = 1.0;
            const double r2 = std::pow(g.x(i), 2) + std::pow(g.x(j), 2);
            i_target(i, j) = std::exp(-r2 / std::pow(8 * g.pitch, 2));  // bright center
        }
    auto phi = doe::init_tie(g, d, i_target, i_source);
    REQUIRE(phi.rows == n);
    // negative curvature at the center: phi(center) > phi(neighbors)
    const std::size_t c = n / 2;
    CHECK(phi(c, c) > phi(c + 3, c));
    CHECK(phi(c, c) > phi(c, c + 3));
    CHECK(phi(c, c) > phi(c - 3, c));
    // forward propagation focuses: central-region energy grows
    doe::Field<double> u(n, n);
    for (std::size_t k = 0; k < u.data.size(); ++k) u.data[k] = std::polar(std::sqrt(i_source.data[k]), phi.data[k]);
    doe::AngularSpectrum<double> A(g, d, true);
    auto v = A.forward(u);
    auto central = [&](const doe::Field<double>& f) {
        double e = 0;
        for (std::size_t i = c - 8; i < c + 8; ++i)
            for (std::size_t j = c - 8; j < c + 8; ++j) e += std::norm(f(i, j));
        return e;
    };
    CHECK(central(v) > 1.5 * central(u));
    // with the opposite sign of d (backward) it would defocus instead
    doe::AngularSpectrum<double> B(g, -d, true);
    CHECK(central(B.forward(u)) < central(u));
}

TEST_CASE("init_tie: energy normalization and the source floor", "[init][TIE]") {
    // The target is rescaled to the source energy (only the shape matters) and
    // the source is floored at 1e-6 of its maximum so the ratio stays finite.
    const std::size_t n = 32;
    doe::Grid g{32, 8e-6, 532e-9};
    doe::Array2<double> src(n, n, 0.0), tgt(n, n, 0.0);
    for (std::size_t i = 8; i < 24; ++i)
        for (std::size_t j = 8; j < 24; ++j) src(i, j) = 1.0;
    tgt(16, 16) = 1.0;
    auto phi_a = doe::init_tie(g, 0.02, tgt, src);
    for (auto& x : tgt.data) x *= 1000.0;  // same shape, other scale
    auto phi_b = doe::init_tie(g, 0.02, tgt, src);
    for (std::size_t k = 0; k < phi_a.data.size(); ++k) CHECK(std::abs(phi_a.data[k] - phi_b.data[k]) < 1e-9);
    for (double x : phi_a.data) CHECK(std::isfinite(x));
}

TEST_CASE("init_backprop: phase of the back-propagated flat-phase target (Gerchberg-Saxton / Fienup start)", "[init]") {
    // If the target amplitude is |A u*| of a phase-only field u* whose target
    // field has flat phase, then arg(A^H b) inside the aperture is arg(u*).
    auto g = doe::Grid::padded(32, 8e-6, 532e-9);
    const std::size_t n = static_cast<std::size_t>(g.n);
    doe::AngularSpectrum<double> A(g, 0.02, false);
    // construct a target with flat phase: v = b real >= 0; u = A^H v has some amplitude/phase
    doe::Array2<double> b(n, n, 0.0), illum(n, n, 0.0);
    const std::size_t lo = (n - 32) / 2;
    for (std::size_t i = lo; i < lo + 32; ++i)
        for (std::size_t j = lo; j < lo + 32; ++j) {
            illum(i, j) = 1.0;
            b(i, j) = 0.5 + 0.5 * std::cos(2 * pi * double(i) / 16) * std::cos(2 * pi * double(j) / 16);
        }
    auto phi = doe::init_backprop<double>(g, A, b);
    REQUIRE(phi.rows == n);
    doe::Field<double> v(n, n);
    for (std::size_t k = 0; k < v.data.size(); ++k) v.data[k] = b.data[k];
    auto u = A.adjoint(v);
    for (std::size_t k = 0; k < u.data.size(); ++k)
        if (std::abs(u.data[k]) > 1e-9) CHECK_THAT(phi.data[k], WithinAbs(std::arg(u.data[k]), 1e-12));
    for (double x : phi.data) {
        CHECK(x <= pi);
        CHECK(x > -pi);
    }
    auto phif = doe::init_backprop<float>(g, doe::AngularSpectrum<float>(g, 0.02, false), b);
    CHECK(phif.rows == n);
}

TEST_CASE("init_random: uniform in (-pi, pi], reproducible by seed", "[init]") {
    doe::Grid g{64, 8e-6, 532e-9};
    auto a = doe::init_random<double>(g, 42);
    auto b = doe::init_random<double>(g, 42);
    auto c = doe::init_random<double>(g, 43);
    REQUIRE(a.rows == 64);
    double lo = 10, hi = -10, mean = 0;
    for (double x : a.data) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
        mean += x;
    }
    CHECK(lo >= -pi);
    CHECK(hi <= pi);
    CHECK(lo < -0.9 * pi);
    CHECK(hi > 0.9 * pi);
    CHECK(std::abs(mean / double(a.data.size())) < 0.1);
    CHECK(a.data == b.data);
    CHECK(a.data != c.data);
    auto f = doe::init_random<float>(g, 42);
    CHECK(f.rows == 64);
}
