// Optical vortex (phase singularity) counter.
//
// Goldstein, Zebker & Werner 1988: the sum of the wrapped phase differences
// around four adjacent points "is either zero, plus one cycle, or minus one
// cycle"; a non-zero result is a "residue". Fried & Vaughn 1992 call the same
// objects branch points of the phase function, located where the scalar
// field's intensity is zero, with the sign given by the circulation. Berry &
// Dennis 2000 eq. (2.4): the topological charge is sgn(xi_x eta_y - xi_y eta_x)
// for the field xi + i eta; their eq. (2.7) gives the mean density of these
// points in an isotropic random (speckle) field, checked statistically below.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/fft.hpp"
#include "doe/grid.hpp"
#include "doe/vortex.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <random>

using Catch::Matchers::WithinRel;
constexpr double pi = std::numbers::pi;

namespace {
// field with vortices of the given charges at the given (x, y) positions (pixel units)
doe::Field<double> vortices(std::size_t n, const std::vector<std::tuple<double, double, int>>& v) {
    doe::Field<double> u(n, n, std::complex<double>(1.0, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (const auto& [x0, y0, m] : v) {
                const std::complex<double> z(double(i) - x0, double(j) - y0);
                u(i, j) *= std::pow(z / (std::abs(z) + 1e-300), m);
            }
    return u;
}
long sum_region(const doe::Array2<int>& q, std::size_t i0, std::size_t i1, std::size_t j0, std::size_t j1) {
    long s = 0;
    for (std::size_t i = i0; i < i1; ++i)
        for (std::size_t j = j0; j < j1; ++j) s += q(i, j);
    return s;
}
}  // namespace

TEST_CASE("vortex_charge_map: a single vortex is one plaquette of charge +-1, the rest zero", "[vortex]") {
    const std::size_t n = 32;
    auto u = vortices(n, {{10.3, 20.6, +1}});
    auto q = doe::vortex_charge_map(u);
    REQUIRE(q.rows == n - 1);
    REQUIRE(q.cols == n - 1);
    CHECK(sum_region(q, 0, n - 1, 0, n - 1) == 1);
    CHECK(q(10, 20) == 1);  // the plaquette (10,20)-(11,21) contains (10.3, 20.6)
    auto qm = doe::vortex_charge_map(vortices(n, {{10.3, 20.6, -1}}));
    CHECK(qm(10, 20) == -1);
    CHECK(sum_region(qm, 0, n - 1, 0, n - 1) == -1);
    CHECK(doe::vortex_count(q) == 1);
    CHECK(doe::vortex_count(qm) == 1);
}

TEST_CASE("vortex_charge_map: charges add, opposite pairs cancel in the total, the count sees both", "[vortex]") {
    const std::size_t n = 48;
    auto q = doe::vortex_charge_map(vortices(n, {{8.4, 8.2, +1}, {30.7, 12.1, +1}, {20.2, 35.9, -1}}));
    CHECK(sum_region(q, 0, n - 1, 0, n - 1) == 1);
    CHECK(doe::vortex_count(q) == 3);
    CHECK(q(8, 8) == 1);
    CHECK(q(30, 12) == 1);
    CHECK(q(20, 35) == -1);
}

TEST_CASE("vortex_charge_map: a charge-2 vortex carries winding 2 around it", "[vortex]") {
    // Core off the plaquette center: with the core exactly at a center every
    // phase step around it is exactly pi, the ambiguous limit of the sampling
    // criterion (GZW: "one-half cycle/point"), and the residues are undefined.
    const std::size_t n = 32;
    auto q = doe::vortex_charge_map(vortices(n, {{15.3, 15.6, 2}}));
    CHECK(sum_region(q, 10, 21, 10, 21) == 2);  // winding of a loop around the singularity
    CHECK(sum_region(q, 0, n - 1, 0, n - 1) == 2);
}

TEST_CASE("vortex_charge_map: a smooth phase has no residues; zero amplitude is not a vortex", "[vortex]") {
    const std::size_t n = 32;
    doe::Field<double> u(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) u(i, j) = std::polar(1.0 + 0.1 * double(i), 0.3 * double(i) + 0.05 * double(i) * double(j) / n);
    auto q = doe::vortex_charge_map(u);
    CHECK(doe::vortex_count(q) == 0);
    doe::Field<double> zero(n, n, std::complex<double>(0.0, 0.0));
    CHECK(doe::vortex_count(doe::vortex_charge_map(zero)) == 0);
}

TEST_CASE("vortex_density: count per unit area; float fields are accepted", "[vortex]") {
    const std::size_t n = 40;
    auto u = vortices(n, {{8.4, 8.2, +1}, {30.7, 12.1, -1}});
    doe::Grid g{40, 8e-6, 532e-9};
    CHECK_THAT(doe::vortex_density(u, g), WithinRel(2.0 / std::pow(39 * 8e-6, 2), 1e-12));
    doe::Field<float> uf(n, n);
    for (std::size_t k = 0; k < uf.data.size(); ++k) uf.data[k] = std::complex<float>(u.data[k]);
    CHECK(doe::vortex_count(doe::vortex_charge_map(uf)) == 2);
}

TEST_CASE("speckle: the vortex density of an isotropic random wave follows Berry & Dennis 2000 (2.7)", "[vortex][analytic]") {
    // Random field with a thin ring spectrum |k| = k0: mean dislocation point
    // density k0^2 / (4 pi). Count over a 512^2 sample and compare (statistical,
    // 10 % tolerance).
    const std::size_t n = 512;
    doe::Grid g{512, 1.0, 1.0};  // unit pitch: k in radians per pixel
    doe::Fft2<double> fft(n);
    doe::Field<double> U(n, n, std::complex<double>(0.0, 0.0));
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> uni(0.0, 2 * pi);
    const double k0 = 2 * pi * 0.08;  // cycles: 0.08 per pixel -> well resolved
    std::size_t modes = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double k = 2 * pi * std::hypot(g.f(i), g.f(j));
            if (std::abs(k - k0) < 2 * pi * 0.004) {
                U(i, j) = std::polar(1.0, uni(rng));
                ++modes;
            }
        }
    REQUIRE(modes > 200);
    auto u = fft.inverse(U);
    const double density = doe::vortex_density(u, g);
    const double expected = k0 * k0 / (4 * pi);
    INFO("measured " << density << " per px^2, Berry-Dennis " << expected);
    CHECK_THAT(density, WithinRel(expected, 0.10));
}
