// Energy and Wirtinger gradient of the DOE design problem.
//
//   E(phi) = w_s [1 - p^2 / (c s2)]  +  w_e [1 - s2 / E_in]
//   p = sum_W b|v|,  c = sum_W b^2,  s2 = sum_W |v|^2,  E_in = sum |illum|^2,  v = A(illum e^{i phi})
//
// Sources: the shape term is Fienup 1997 eq. (20) (normalized mean-square
// error minimized over a real multiplicative constant, no translation search),
// the efficiency term is the diffraction efficiency of Wyrowski 1990 /
// Wyrowski & Bryngdahl 1988; the gradient uses Wirtinger calculus
// (Kreutz-Delgado 2009; Chakravarthula 2019 eqs. 10-14): dE/dv* = dE/d|v| * v/(2|v|),
// G = A^H dE/dv*, dE/dphi = 2 Im(conj(u) G).
//
// T4: ratio finite-difference / analytic = 1 +- 1e-6 in double, for each term
// separately (a combined test hides constant-factor errors).
// Oracle: design.py on a fixed input (scratchpad/oracle_energy.py).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/energy.hpp"
#include "doe/fft.hpp"
#include "doe/grid.hpp"
#include "doe/propagate.hpp"

#include <cmath>
#include <numbers>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
constexpr double pi = std::numbers::pi;

namespace {

struct Problem {
    doe::Grid grid;
    doe::AngularSpectrum<double> prop;
    doe::Array2<double> phi, illum, b, mask;
};

// The oracle problem of scratchpad/oracle_energy.py (n = 32, d = 2 cm, band limit off).
Problem oracle_problem() {
    const std::size_t n = 32;
    doe::Grid g{32, 8e-6, 532e-9};
    Problem P{g, doe::AngularSpectrum<double>(g, 0.02, false), {n, n}, {n, n, 1.0}, {n, n}, {n, n, 0.0}};
    const double p = g.pitch;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double x = g.x(i), y = g.x(j);
            P.phi(i, j) = std::sin(3 * x / p / n * 2 * pi) + 0.5 * std::cos(5 * y / p / n * 2 * pi) + 0.2 * (x / p) * (y / p) / n;
            P.b(i, j) = 0.5 + 0.5 * std::cos(2 * pi * x / (8 * p)) * std::cos(2 * pi * y / (8 * p));
            if (i < 4 || i >= n - 4 || j < 4 || j >= n - 4) P.illum(i, j) = 0.0;
            if (i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4) P.mask(i, j) = 1.0;
        }
    return P;
}

Problem random_problem(unsigned seed) {
    const std::size_t n = 32;
    doe::Grid g{32, 8e-6, 532e-9};
    Problem P{g, doe::AngularSpectrum<double>(g, 0.02, true), {n, n}, {n, n, 1.0}, {n, n}, {n, n, 0.0}};
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            P.phi(i, j) = (2 * uni(rng) - 1) * pi;
            P.b(i, j) = uni(rng);
            if (i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4) P.mask(i, j) = 1.0;
        }
    return P;
}

double energy_of(const Problem& P, const doe::Array2<double>& phi, doe::EnergyWeights w) {
    return doe::energy_and_grad<double>(phi, P.illum, P.prop, P.b, P.mask, w).energy;
}

// Central finite difference of E along coordinate (i, j).
double fd(const Problem& P, doe::EnergyWeights w, std::size_t i, std::size_t j, double h) {
    auto pp = P.phi, pm = P.phi;
    pp(i, j) += h;
    pm(i, j) -= h;
    return (energy_of(P, pp, w) - energy_of(P, pm, w)) / (2 * h);
}

// Median of numeric/analytic ratios over 40 random coordinates (check
// ratios, not error norms).
double median_ratio(const Problem& P, doe::EnergyWeights w, double h, unsigned seed) {
    auto r = doe::energy_and_grad<double>(P.phi, P.illum, P.prop, P.b, P.mask, w);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, 31);
    std::vector<double> ratios;
    while (ratios.size() < 40) {
        const std::size_t i = pick(rng), j = pick(rng);
        if (std::abs(r.grad(i, j)) < 1e-9) continue;
        ratios.push_back(fd(P, w, i, j, h) / r.grad(i, j));
    }
    std::sort(ratios.begin(), ratios.end());
    return ratios[ratios.size() / 2];
}

}  // namespace

TEST_CASE("optimal_scale: least-squares factor matching |v| to b on the window", "[energy]") {
    doe::Array2<double> b(4, 4), v(4, 4), mask(4, 4, 0.0);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            b(i, j) = 1.0 + double(i) + 0.5 * double(j);
            v(i, j) = 2.5 * b(i, j);
            mask(i, j) = (i < 2) ? 1.0 : 0.0;
        }
    CHECK_THAT(doe::optimal_scale(v, b, mask), WithinRel(2.5, 1e-15));
    // Pixels outside the window do not count.
    v(3, 3) = 1e6;
    CHECK_THAT(doe::optimal_scale(v, b, mask), WithinRel(2.5, 1e-15));
    // Empty window: scale defaults to 1.
    doe::Array2<double> none(4, 4, 0.0);
    CHECK(doe::optimal_scale(v, b, none) == 1.0);
}

TEST_CASE("energy: terms, ranges and the returned field", "[energy]") {
    auto P = oracle_problem();
    auto r = doe::energy_and_grad<double>(P.phi, P.illum, P.prop, P.b, P.mask, {1.0, 0.3});
    CHECK(r.field.rows == 32);
    // shape term is 1 - cos^2 between |v| and b on the window: in [0, 1]
    CHECK(r.shape >= 0.0);
    CHECK(r.shape <= 1.0);
    // efficiency term is mu (1 - s2 / E_in) with s2 <= E_in (partial isometry): in [0, mu]
    CHECK(r.efficiency >= 0.0);
    CHECK(r.efficiency <= 0.3);
    CHECK_THAT(r.energy, WithinRel(r.shape + r.efficiency, 1e-15));
    CHECK(r.grad.rows == 32);
    // the field equals the propagated illum e^{i phi}
    doe::Field<double> u(32, 32);
    for (std::size_t k = 0; k < u.data.size(); ++k) u.data[k] = P.illum.data[k] * std::polar(1.0, P.phi.data[k]);
    auto v = P.prop.forward(u);
    for (std::size_t k = 0; k < u.data.size(); ++k) CHECK(std::abs(v.data[k] - r.field.data[k]) < 1e-14);
}

TEST_CASE("energy: the shape term is invariant under |v| -> a|v| (Fienup 1997 eq. 20)", "[energy]") {
    // Scaling the illumination scales v; the shape term must not move, the
    // efficiency term must (it is what measures the scale).
    auto P = oracle_problem();
    auto r1 = doe::energy_and_grad<double>(P.phi, P.illum, P.prop, P.b, P.mask, {1.0, 0.0});
    auto illum2 = P.illum;
    for (auto& x : illum2.data) x *= 3.0;
    auto r2 = doe::energy_and_grad<double>(P.phi, illum2, P.prop, P.b, P.mask, {1.0, 0.0});
    CHECK_THAT(r2.shape, WithinRel(r1.shape, 1e-12));
}

TEST_CASE("oracle: energy, scale and gradient samples match design.py", "[energy][oracle]") {
    auto P = oracle_problem();
    CHECK(P.phi(16, 16) == 0.5);
    CHECK(P.b(16, 16) == 1.0);
    auto r0 = doe::energy_and_grad<double>(P.phi, P.illum, P.prop, P.b, P.mask, {1.0, 0.0});
    CHECK_THAT(r0.energy, WithinRel(0.31705779190246364, 1e-12));
    doe::Array2<double> vabs(32, 32);
    for (std::size_t k = 0; k < vabs.data.size(); ++k) vabs.data[k] = std::abs(r0.field.data[k]);
    CHECK_THAT(doe::optimal_scale(vabs, P.b, P.mask), WithinRel(1.4332884621656703, 1e-12));
    CHECK_THAT(r0.grad(16, 16), WithinRel(-0.00023902830430183174, 1e-9));
    CHECK_THAT(r0.grad(10, 20), WithinRel(0.003502575813781137, 1e-9));
    CHECK_THAT(r0.grad(5, 5), WithinRel(-0.001036648854564691, 1e-9));
    CHECK(r0.grad(28, 3) == 0.0);  // outside the illuminated aperture: u = 0
    auto r3 = doe::energy_and_grad<double>(P.phi, P.illum, P.prop, P.b, P.mask, {1.0, 0.3});
    CHECK_THAT(r3.energy, WithinRel(0.491722893415963, 1e-12));
    CHECK_THAT(r3.grad(16, 16), WithinRel(-0.0009029853688901319, 1e-9));
    CHECK_THAT(r3.grad(10, 20), WithinRel(0.003413665291682304, 1e-9));
    CHECK_THAT(r3.grad(5, 5), WithinRel(-0.000794014823297614, 1e-9));
}

TEST_CASE("T4 gradient of the shape term alone: FD / analytic = 1 +- 1e-6", "[energy][T4]") {
    auto P = random_problem(7);
    CHECK_THAT(median_ratio(P, {1.0, 0.0}, 1e-6, 1), WithinAbs(1.0, 1e-6));
}

TEST_CASE("T4 gradient of the efficiency term alone: FD / analytic = 1 +- 1e-6", "[energy][T4]") {
    auto P = random_problem(8);
    CHECK_THAT(median_ratio(P, {0.0, 0.3}, 1e-6, 2), WithinAbs(1.0, 1e-6));
}

TEST_CASE("T4 gradient of the sum: FD / analytic = 1 +- 1e-6", "[energy][T4]") {
    auto P = random_problem(9);
    CHECK_THAT(median_ratio(P, {1.0, 0.3}, 1e-6, 3), WithinAbs(1.0, 1e-6));
}

TEST_CASE("T4 h-sweep: the finite-difference error falls like h^2, then rounding takes over", "[energy][T4]") {
    // A flat error curve would mean a bug, not discretization.
    auto P = random_problem(10);
    const doe::EnergyWeights w{1.0, 0.3};
    auto r = doe::energy_and_grad<double>(P.phi, P.illum, P.prop, P.b, P.mask, w);
    const std::size_t i = 12, j = 19;
    REQUIRE(std::abs(r.grad(i, j)) > 1e-6);
    std::vector<double> hs{1e-1, 1e-2, 1e-3, 1e-4}, errs;
    for (double h : hs) errs.push_back(std::abs(fd(P, w, i, j, h) - r.grad(i, j)) / std::abs(r.grad(i, j)));
    // second-order: each decade of h gains about two decades of accuracy
    CHECK(errs[1] < errs[0] * 3e-2);
    CHECK(errs[2] < errs[1] * 3e-2);
    CHECK(errs[3] < 1e-6);
    // rounding floor: at h = 1e-9 the error is dominated by cancellation and larger than at 1e-4
    CHECK(std::abs(fd(P, w, i, j, 1e-9) - r.grad(i, j)) / std::abs(r.grad(i, j)) > errs[3]);
}

TEST_CASE("energy and gradient are identical to 1e-13 with 1 and many compute threads", "[energy][threads]") {
    auto P = random_problem(12);
    doe::set_fft_threads(1);
    doe::AngularSpectrum<double> p1(P.grid, 0.02, true);
    auto r1 = doe::energy_and_grad<double>(P.phi, P.illum, p1, P.b, P.mask, {1.0, 0.3});
    doe::set_fft_threads(4);
    doe::AngularSpectrum<double> p4(P.grid, 0.02, true);
    auto r4 = doe::energy_and_grad<double>(P.phi, P.illum, p4, P.b, P.mask, {1.0, 0.3});
    doe::set_fft_threads(0);
    CHECK_THAT(r4.energy, WithinRel(r1.energy, 1e-13));
    for (std::size_t k = 0; k < r1.grad.data.size(); ++k) CHECK(std::abs(r4.grad.data[k] - r1.grad.data[k]) < 1e-13);
}

TEST_CASE("energy in single precision agrees with double to 1e-4", "[energy][precision]") {
    auto P = random_problem(11);
    const std::size_t n = 32;
    doe::AngularSpectrum<float> propf(P.grid, 0.02, true);
    doe::Array2<float> phi(n, n), illum(n, n), b(n, n), mask(n, n);
    for (std::size_t k = 0; k < n * n; ++k) {
        phi.data[k] = float(P.phi.data[k]);
        illum.data[k] = float(P.illum.data[k]);
        b.data[k] = float(P.b.data[k]);
        mask.data[k] = float(P.mask.data[k]);
    }
    auto rd = doe::energy_and_grad<double>(P.phi, P.illum, P.prop, P.b, P.mask, {1.0, 0.3});
    auto rf = doe::energy_and_grad<float>(phi, illum, propf, b, mask, {1.0, 0.3});
    CHECK_THAT(rf.energy, WithinRel(rd.energy, 1e-4));
    double gmax = 0, gerr = 0;
    for (std::size_t k = 0; k < n * n; ++k) {
        gmax = std::max(gmax, std::abs(rd.grad.data[k]));
        gerr = std::max(gerr, std::abs(double(rf.grad.data[k]) - rd.grad.data[k]));
    }
    CHECK(gerr < 1e-4 * gmax);
}
