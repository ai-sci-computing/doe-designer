// Solvers: Adam on the phase (Kingma & Ba 2015, Algorithm 1) and
// Gerchberg-Saxton (Gerchberg & Saxton 1972; Fienup 1982 error reduction)
// with a free amplitude outside the signal window (Fienup 1980).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/energy.hpp"
#include "doe/grid.hpp"
#include "doe/init.hpp"
#include "doe/metrics.hpp"
#include "doe/propagate.hpp"
#include "doe/solvers.hpp"

#include <cmath>
#include <numbers>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
constexpr double pi = std::numbers::pi;

namespace {

// A small design problem: uniform square illumination of 48 px on a 96 px
// padded grid, a disk target of radius 10 px in the central 48 px window.
struct Problem {
    doe::Grid grid = doe::Grid::padded(48, 8e-6, 532e-9);
    doe::AngularSpectrum<double> prop;
    doe::Array2<double> illum, b, mask;
    explicit Problem(bool band_limit = true) : prop(grid, 0.02, band_limit) {
        const std::size_t n = static_cast<std::size_t>(grid.n), lo = (n - 48) / 2;
        illum = b = mask = doe::Array2<double>(n, n, 0.0);
        for (std::size_t i = lo; i < lo + 48; ++i)
            for (std::size_t j = lo; j < lo + 48; ++j) {
                illum(i, j) = 1.0;
                mask(i, j) = 1.0;
                const double r2 = std::pow(g().x(i), 2) + std::pow(g().x(j), 2);
                b(i, j) = r2 <= std::pow(10 * g().pitch, 2) ? 1.0 : 0.0;
            }
    }
    const doe::Grid& g() const { return grid; }
};

}  // namespace

TEST_CASE("Adam: constant gradient moves each coordinate by lr per step (bias correction exact)", "[solvers][adam]") {
    // With g constant, m_hat = g and v_hat = g^2 at every step, so the update
    // is lr * g / (|g| + eps): a unit step of size lr in the descent direction.
    doe::Adam<double> adam(2, 2, {0.05, 0.9, 0.999, 1e-8});
    doe::Array2<double> x(2, 2, 1.0), g(2, 2);
    g(0, 0) = 3.0; g(0, 1) = -0.2; g(1, 0) = 1e-3; g(1, 1) = 0.0;
    for (int t = 0; t < 3; ++t) adam.step(x, g);
    CHECK(adam.iteration() == 3);
    CHECK_THAT(x(0, 0), WithinAbs(1.0 - 3 * 0.05, 1e-8));
    CHECK_THAT(x(0, 1), WithinAbs(1.0 + 3 * 0.05, 1e-6));
    CHECK_THAT(x(1, 0), WithinAbs(1.0 - 3 * 0.05, 1e-4));  // eps = 1e-8 vs |g| = 1e-3: 1e-5 relative
    CHECK(x(1, 1) == 1.0);                                   // zero gradient, zero move
}

TEST_CASE("Adam: matches Algorithm 1 written out by hand for a varying gradient", "[solvers][adam]") {
    const double lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    doe::Adam<double> adam(1, 1, {lr, b1, b2, eps});
    doe::Array2<double> x(1, 1, 0.5), g(1, 1);
    double m = 0, v = 0, xr = 0.5;
    for (int t = 1; t <= 10; ++t) {
        const double gt = std::sin(0.7 * t) + 0.3 * xr;  // some gradient sequence
        g(0, 0) = gt;
        adam.step(x, g);
        m = b1 * m + (1 - b1) * gt;
        v = b2 * v + (1 - b2) * gt * gt;
        const double mhat = m / (1 - std::pow(b1, t)), vhat = v / (1 - std::pow(b2, t));
        xr -= lr * mhat / (std::sqrt(vhat) + eps);
        CHECK_THAT(x(0, 0), WithinAbs(xr, 1e-15));
    }
}

TEST_CASE("Adam: minimizes a quadratic", "[solvers][adam]") {
    doe::Adam<double> adam(3, 3, {0.05, 0.9, 0.999, 1e-8});
    doe::Array2<double> x(3, 3, 2.0), g(3, 3), c(3, 3);
    for (std::size_t k = 0; k < 9; ++k) c.data[k] = 0.1 * double(k) - 0.4;
    for (int t = 0; t < 600; ++t) {
        for (std::size_t k = 0; k < 9; ++k) g.data[k] = 2 * (x.data[k] - c.data[k]);
        adam.step(x, g);
    }
    for (std::size_t k = 0; k < 9; ++k) CHECK_THAT(x.data[k], WithinAbs(c.data[k], 1e-2));
}

TEST_CASE("Adam in single precision runs and agrees with double for a few steps", "[solvers][adam][precision]") {
    doe::Adam<float> af(2, 2, {0.05, 0.9, 0.999, 1e-8});
    doe::Adam<double> ad(2, 2, {0.05, 0.9, 0.999, 1e-8});
    doe::Array2<float> xf(2, 2, 1.f), gf(2, 2, 0.3f);
    doe::Array2<double> xd(2, 2, 1.0), gd(2, 2, 0.3);
    for (int t = 0; t < 5; ++t) {
        af.step(xf, gf);
        ad.step(xd, gd);
    }
    for (std::size_t k = 0; k < 4; ++k) CHECK_THAT(double(xf.data[k]), WithinAbs(xd.data[k], 1e-6));
}

TEST_CASE("Gerchberg-Saxton: a self-consistent target is a fixed point", "[solvers][gs]") {
    // b = |A(illum e^{i phi*})| on the window: starting at phi* the iteration
    // must not move (inside the aperture, where the phase is defined). Needs
    // A^H A = I, i.e. the band limit off: with it, A^H A only projects onto
    // the passband and a hard-edged aperture is not exactly reproduced.
    Problem P(false);
    const std::size_t n = static_cast<std::size_t>(P.g().n);
    doe::Array2<double> phi_star(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            phi_star(i, j) = -P.g().k() * (std::pow(P.g().x(i), 2) + std::pow(P.g().x(j), 2)) / (2 * 0.05) + 3.0 * P.g().x(i) / P.g().pitch / n;
    doe::Field<double> u(n, n);
    for (std::size_t k = 0; k < n * n; ++k) u.data[k] = std::polar(P.illum.data[k], phi_star.data[k]);
    auto v = P.prop.forward(u);
    doe::Array2<double> b(n, n);
    for (std::size_t k = 0; k < n * n; ++k) b.data[k] = P.mask.data[k] * std::abs(v.data[k]);
    auto r = doe::gerchberg_saxton<double>(phi_star, P.illum, P.prop, b, P.mask, {5, 0});
    REQUIRE(r.history.size() == 5);
    REQUIRE(r.residual_history.size() == 5);
    for (double e : r.residual_history) CHECK(e < 1e-20);
    for (std::size_t k = 0; k < n * n; ++k)
        if (P.illum.data[k] > 0) CHECK(std::abs(std::arg(std::polar(1.0, r.phi.data[k] - phi_star.data[k]))) < 1e-9);
}

TEST_CASE("Gerchberg-Saxton: Wyrowski's two stages; the residual never increases within a stage (Fienup 1982)", "[solvers][gs]") {
    Problem P(false);  // unitary propagator: the error-reduction theorem applies exactly
    auto phi0 = doe::init_random<double>(P.g(), 5);
    auto r = doe::gerchberg_saxton<double>(phi0, P.illum, P.prop, P.b, P.mask, {40, 20});
    REQUIRE(r.residual_history.size() == 40);
    for (std::size_t t = 1; t < 20; ++t) CHECK(r.residual_history[t] <= r.residual_history[t - 1] * (1 + 1e-12));
    for (std::size_t t = 21; t < 40; ++t) CHECK(r.residual_history[t] <= r.residual_history[t - 1] * (1 + 1e-12));
    CHECK(r.phi.rows == static_cast<std::size_t>(P.g().n));
    CHECK(r.field.rows == static_cast<std::size_t>(P.g().n));

    // Stage X alone drives the energy into the window (measured 0.93-0.98);
    // stage X' then trades efficiency for fidelity (Wyrowski 1990 §2.B).
    auto x_only = doe::gerchberg_saxton<double>(phi0, P.illum, P.prop, P.b, P.mask, {20, 20});
    auto m1 = doe::metrics(x_only.field, P.b, P.mask);
    auto m2 = doe::metrics(r.field, P.b, P.mask);
    CHECK(m1.efficiency > 0.9);
    CHECK(m2.efficiency > 0.3);
    CHECK(m2.efficiency < m1.efficiency);
    CHECK(m2.ncc > m1.ncc);
    CHECK(m2.ncc > 0.8);

    // Amplitude freedom from the very first cycle lets the window shrink
    // (the degeneracy the two-stage schedule exists to avoid).
    auto bad = doe::gerchberg_saxton<double>(phi0, P.illum, P.prop, P.b, P.mask, {40, 0});
    CHECK(doe::metrics(bad.field, P.b, P.mask).efficiency < 0.5 * m2.efficiency);
}

TEST_CASE("Gerchberg-Saxton: phases are wrapped to (-pi, pi]", "[solvers][gs]") {
    Problem P;
    auto r = doe::gerchberg_saxton<double>(doe::init_random<double>(P.g(), 6), P.illum, P.prop, P.b, P.mask, {3, 2});
    for (double x : r.phi.data) {
        CHECK(x <= pi);
        CHECK(x > -pi);
    }
}

TEST_CASE("optimize (Adam on the energy): energy falls, efficiency does not, phases stay on the torus", "[solvers][adam][T13]") {
    Problem P;
    auto phi0 = doe::init_random<double>(P.g(), 7);
    const double eff0 = doe::metrics(doe::energy_and_grad<double>(phi0, P.illum, P.prop, P.b, P.mask).field, P.b, P.mask).efficiency;
    doe::OptimizeParams opt;
    opt.iters = 120;
    opt.lr = 0.05;
    opt.weights = {1.0, 0.3};
    auto r = doe::optimize<double>(phi0, P.illum, P.prop, P.b, P.mask, opt);
    REQUIRE(r.history.size() == 120);
    CHECK(r.history.back() < 0.7 * r.history.front());
    for (double x : r.phi.data) {
        CHECK(x <= pi);
        CHECK(x > -pi);
    }
    auto m = doe::metrics(r.field, P.b, P.mask);
    CHECK(m.efficiency >= eff0);  // the degeneracy guard (full T13 in the pipeline tests)
    CHECK(m.ncc > 0.5);
}

TEST_CASE("optimize: progress callback sees every iteration and can stop early", "[solvers][adam]") {
    Problem P;
    doe::OptimizeParams opt;
    opt.iters = 50;
    int seen = 0;
    auto r = doe::optimize<double>(doe::init_random<double>(P.g(), 8), P.illum, P.prop, P.b, P.mask, opt,
                                   [&](int it, double e) { ++seen; return it < 9 && e > 0; });  // false = stop
    CHECK(seen == 10);
    CHECK(r.history.size() == 10);
}

TEST_CASE("optimize in single precision produces finite phases and a falling energy", "[solvers][adam][precision]") {
    Problem P;
    const std::size_t n = static_cast<std::size_t>(P.g().n);
    doe::AngularSpectrum<float> propf(P.g(), 0.02, true);
    doe::Array2<float> illum(n, n), b(n, n), mask(n, n);
    for (std::size_t k = 0; k < n * n; ++k) {
        illum.data[k] = float(P.illum.data[k]);
        b.data[k] = float(P.b.data[k]);
        mask.data[k] = float(P.mask.data[k]);
    }
    doe::OptimizeParams opt;
    opt.iters = 40;
    auto r = doe::optimize<float>(doe::init_random<float>(P.g(), 9), illum, propf, b, mask, opt);
    for (float x : r.phi.data) CHECK(std::isfinite(x));
    CHECK(r.history.back() < r.history.front());
}

TEST_CASE("solvers log the shape and efficiency terms separately; they sum to the logged energy", "[solvers][history]") {
    Problem P;
    auto phi0 = doe::init_random<double>(P.g(), 11);
    SECTION("optimize (Adam)") {
        doe::OptimizeParams opt;
        opt.iters = 12;
        opt.weights = {1.0, 0.3};
        auto r = doe::optimize<double>(phi0, P.illum, P.prop, P.b, P.mask, opt);
        REQUIRE(r.history.size() == 12);
        REQUIRE(r.shape_history.size() == 12);
        REQUIRE(r.efficiency_history.size() == 12);
        for (std::size_t t = 0; t < 12; ++t) {
            CHECK(r.shape_history[t] > 0.0);
            CHECK(r.efficiency_history[t] >= 0.0);
            CHECK_THAT(r.shape_history[t] + r.efficiency_history[t], WithinAbs(r.history[t], 1e-12));
        }
        // the shape term is the one the solver drives down; the efficiency term stays small
        CHECK(r.shape_history.back() < r.shape_history.front());
        CHECK(r.efficiency_history.back() < 0.3);
    }
    SECTION("Gerchberg-Saxton") {
        doe::GsParams gs;
        gs.iters = 6;
        gs.phase_only_iters = 3;
        auto r = doe::gerchberg_saxton<double>(phi0, P.illum, P.prop, P.b, P.mask, gs);
        REQUIRE(r.history.size() == 6);
        REQUIRE(r.shape_history.size() == 6);
        REQUIRE(r.efficiency_history.size() == 6);
        for (std::size_t t = 0; t < 6; ++t)
            CHECK_THAT(r.shape_history[t] + r.efficiency_history[t], WithinAbs(r.history[t], 1e-12));
    }
}
