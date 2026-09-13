// Band-limited angular spectrum propagator.
//
// Sources (see docs/references.bib):
//   Goodman 2017 §3.10: transfer function H = exp[i 2 pi (z/lambda) sqrt(1 - (lambda fx)^2 - (lambda fy)^2)]
//     (eqs. 3-77, 3-78), evanescent components (lambda fx)^2 + (lambda fy)^2 > 1 (eqs. 3-71, 3-72).
//   Matsushima & Shimobaba 2009: band limit u_limit = 1 / (lambda sqrt((2 du z)^2 + 1)), du = 1/S,
//     S = padded window extent (eqs. 11-13), rect window (eq. 14), zero padding (§2.2).
//   Siegman 1986 ch. 17: Gaussian beam w(z) = w0 sqrt(1 + (z/zR)^2), zR = pi w0^2 / lambda.
//   Montgomery 1967 / Rayleigh 1881: exact self-imaging distance z_T = 2 pi / (k - k_z(f0)).
//   Wyant, Fresnel Diffraction notes eq. (5) & Born-Wolf §8.3: on-axis field of a circular aperture,
//     exact scalar (Rayleigh-Sommerfeld) form U/U0 = exp(ikz) - (z/R) exp(ikR), R = sqrt(z^2 + a^2).
//   Oracle values: optics.py (Python reference) on a fixed input, band limit off.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/grid.hpp"
#include "doe/propagate.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace std::complex_literals;
constexpr double pi = std::numbers::pi;

namespace {

template <class T>
doe::Field<T> random_field(std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    doe::Field<T> u(n, n);
    for (auto& z : u.data) z = std::complex<T>(T(uni(rng)), T(uni(rng)));
    return u;
}

template <class T>
std::complex<double> inner(const doe::Field<T>& a, const doe::Field<T>& b) {
    std::complex<double> s = 0;
    for (std::size_t k = 0; k < a.data.size(); ++k)
        s += std::conj(std::complex<double>(a.data[k])) * std::complex<double>(b.data[k]);
    return s;
}

template <class T>
double energy(const doe::Field<T>& a) { return inner(a, a).real(); }

// The oracle input of scratchpad/oracle_propagate.py:
//   u = exp(-(x^2+y^2) / (2 (12 p)^2)) * exp(i (2 pi x / (16 p) + 0.3 (y/p)^2 / 16))
doe::Field<double> oracle_input(const doe::Grid& g) {
    const std::size_t n = static_cast<std::size_t>(g.n);
    const double p = g.pitch;
    doe::Field<double> u(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double x = g.x(i), y = g.x(j);
            const double amp = std::exp(-(x * x + y * y) / (2.0 * (12 * p) * (12 * p)));
            const double ph = 2 * pi * x / (16 * p) + 0.3 * (y / p) * (y / p) / 16;
            u(i, j) = std::polar(amp, ph);
        }
    return u;
}

// Intensity-weighted centroid and second moment along x (first index).
struct Moments { double mean, sigma; };
template <class T>
Moments moments_x(const doe::Field<T>& v, const doe::Grid& g) {
    double w = 0, m1 = 0, m2 = 0;
    for (std::size_t i = 0; i < v.rows; ++i)
        for (std::size_t j = 0; j < v.cols; ++j) {
            const double I = std::norm(std::complex<double>(v(i, j)));
            w += I;
            m1 += I * g.x(i);
            m2 += I * g.x(i) * g.x(i);
        }
    const double mean = m1 / w;
    return {mean, std::sqrt(m2 / w - mean * mean)};
}

}  // namespace

// ---------------------------------------------------------------------------
// Transfer function
// ---------------------------------------------------------------------------

TEST_CASE("transfer_function: Goodman eq. 3-78 bin by bin, evanescent bins zeroed", "[propagate][H]") {
    doe::Grid g{16, 0.3e-6, 532e-9};  // lambda/(2p) = 0.89: the corner bins are evanescent
    const double d = 1e-4;
    auto tf = doe::transfer_function(g, d, /*band_limit=*/false);
    REQUIRE(tf.H.rows == 16);
    REQUIRE(tf.H.cols == 16);
    std::size_t n_evan = 0;
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 16; ++j) {
            const double arg = 1.0 - std::pow(g.wavelength * g.f(i), 2) - std::pow(g.wavelength * g.f(j), 2);
            if (arg <= 0.0) {
                ++n_evan;
                CHECK(tf.evanescent(i, j) == 1);
                CHECK(tf.H(i, j) == 0.0);
            } else {
                CHECK(tf.evanescent(i, j) == 0);
                const auto expected = std::polar(1.0, 2 * pi / g.wavelength * std::sqrt(arg) * d);
                CHECK(std::abs(tf.H(i, j) - expected) < 1e-12);
            }
        }
    CHECK(n_evan > 0);
    CHECK(std::isinf(tf.f_limit));
}

TEST_CASE("transfer_function: band limit is Matsushima & Shimobaba eq. 13 with du = 1/S", "[propagate][H]") {
    doe::Grid g{64, 8e-6, 532e-9};
    const double d = 0.02;
    auto tf = doe::transfer_function(g, d, true);
    const double S = g.extent();
    const double f_lim = 1.0 / (g.wavelength * std::sqrt(std::pow(2.0 * d / S, 2) + 1.0));
    CHECK_THAT(tf.f_limit, WithinRel(f_lim, 1e-15));

    // Rect window of full width 2 f_lim in each frequency direction (eq. 14).
    std::size_t pass = 0;
    for (std::size_t i = 0; i < 64; ++i)
        for (std::size_t j = 0; j < 64; ++j) {
            const bool inside = std::abs(g.f(i)) <= f_lim && std::abs(g.f(j)) <= f_lim;
            if (inside) {
                ++pass;
                CHECK(std::abs(std::abs(tf.H(i, j)) - 1.0) < 1e-14);
            } else {
                CHECK(tf.H(i, j) == 0.0);
            }
        }
    CHECK(pass > 0);
    CHECK(pass < 64 * 64);

    // d = 0 disables the band limit (H == 1 everywhere); the sign of d does not matter.
    CHECK(std::isinf(doe::transfer_function(g, 0.0, true).f_limit));
    CHECK_THAT(doe::transfer_function(g, -d, true).f_limit, WithinRel(f_lim, 1e-15));
}

TEST_CASE("transfer_function is built in double even for the float propagator", "[propagate][H][precision]") {
    // At d = 0.5 m the phase k_z d ~ 5.9e6 rad; float32 loses ~0.15 rad of it
    // Building H in double and casting afterwards keeps the
    // float propagator's H within float epsilon of the double one.
    doe::Grid g{64, 8e-6, 532e-9};
    const double d = 0.5;
    doe::AngularSpectrum<float> pf(g, d, false);
    doe::AngularSpectrum<double> pd(g, d, false);
    double max_err = 0;
    for (std::size_t k = 0; k < pd.transfer().data.size(); ++k)
        max_err = std::max(max_err, std::abs(std::complex<double>(pf.transfer().data[k]) - pd.transfer().data[k]));
    CHECK(max_err < 2e-7);
}

// ---------------------------------------------------------------------------
// Operator identities: T1, T2, T12, T10, T7
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("T1/T12 adjointness <A u, v> = <u, A^H v>", "[propagate][T1][T12]", double, float) {
    using T = TestType;
    const double tol = std::is_same_v<T, float> ? 1e-5 : 1e-10;
    doe::Grid g{48, 8e-6, 532e-9};
    for (bool bl : {false, true}) {
        doe::AngularSpectrum<T> A(g, 0.03, bl);
        auto u = random_field<T>(48, 1);
        auto v = random_field<T>(48, 2);
        const auto lhs = inner(A.forward(u), v);
        const auto rhs = inner(u, A.adjoint(v));
        CHECK(std::abs(lhs - rhs) / std::abs(lhs) < tol);
    }
}

TEST_CASE("T2 energy conservation with band limit off and no evanescent waves", "[propagate][T2]") {
    doe::Grid g{48, 8e-6, 532e-9};  // Nyquist 6.25e4 /m << 1/lambda: nothing evanescent
    doe::AngularSpectrum<double> A(g, 0.05, false);
    CHECK(A.evanescent_fraction() == 0.0);
    CHECK(A.passband_fraction() == 1.0);
    auto u = random_field<double>(48, 3);
    CHECK_THAT(std::sqrt(energy(A.forward(u))), WithinRel(std::sqrt(energy(u)), 1e-10));
    CHECK_THAT(std::sqrt(energy(A.adjoint(u))), WithinRel(std::sqrt(energy(u)), 1e-10));
    // A^H A = identity when nothing is cut.
    auto back = A.adjoint(A.forward(u));
    for (std::size_t k = 0; k < u.data.size(); ++k) CHECK(std::abs(back.data[k] - u.data[k]) < 1e-12);
}

TEST_CASE("T10 volume() equals forward() at every plane to 1e-14", "[propagate][T10]") {
    doe::Grid g{40, 8e-6, 532e-9};
    doe::AngularSpectrum<double> A(g, 0.05, true);
    auto u = random_field<double>(40, 4);
    std::vector<double> zs{0.0, 0.01, 0.025, 0.05, 0.07};
    auto planes = A.volume(u, zs);
    REQUIRE(planes.size() == zs.size());
    for (std::size_t p = 0; p < zs.size(); ++p) {
        doe::AngularSpectrum<double> Az(g, zs[p], true);
        auto ref = Az.forward(u);
        double num = 0, den = 0;
        for (std::size_t k = 0; k < ref.data.size(); ++k) {
            num += std::norm(planes[p].data[k] - ref.data[k]);
            den += std::norm(ref.data[k]);
        }
        CHECK(std::sqrt(num / den) < 1e-14);
    }
}

TEST_CASE("T7 no wraparound: energy at the padding border < 1e-3 of the total", "[propagate][T7]") {
    // A hard-edged square aperture of 128 px, propagated 5 cm on the 2x padded
    // window. Two measures:
    //  (a) the spec's: energy in the outermost pixel ring of the window (the
    //      padding border) relative to the total, < 1e-3;
    //  (b) the actual circular-convolution error: the field inside the active
    //      region compared with the same propagation on a 4x padded window.
    //      The hard edge radiates Fresnel tails (~ 1/(2 pi^2 v^2) beyond the
    //      shadow edge, Born & Wolf §8.7) whose energy past the window edge is
    //      of order 5e-3 of the total; with no padding at all it is > 10x worse.
    const int active = 128;
    const double d = 0.05;
    auto propagate_padded = [&](double pad, bool band_limit) {
        auto g = doe::Grid::padded(active, 8e-6, 532e-9, pad);
        const std::size_t n = static_cast<std::size_t>(g.n);
        doe::Field<double> u(n, n);
        const std::size_t lo = (n - active) / 2, hi = lo + active;
        for (std::size_t i = lo; i < hi; ++i)
            for (std::size_t j = lo; j < hi; ++j) u(i, j) = 1.0;
        doe::AngularSpectrum<double> A(g, d, band_limit);
        return std::pair{g, A.forward(u)};
    };
    auto [g2, v2] = propagate_padded(2.0, true);
    REQUIRE(g2.n == 256);
    const std::size_t n2 = 256;

    // (a) outermost ring, 2 pixels wide
    double total = 0, edge = 0;
    for (std::size_t i = 0; i < n2; ++i)
        for (std::size_t j = 0; j < n2; ++j) {
            const double I = std::norm(v2(i, j));
            total += I;
            if (i < 2 || i >= n2 - 2 || j < 2 || j >= n2 - 2) edge += I;
        }
    CHECK(edge / total < 1e-3);

    // (b) active-region error against 4x padding (where neither the band limit
    //     nor the wrap bite), and the unpadded case for contrast. The band limit
    //     stays on: it is part of the production configuration and removes the
    //     components that would alias, which is the same light that wraps.
    auto [g4, v4] = propagate_padded(4.0, true);
    auto [g2b, v2b] = propagate_padded(2.0, true);
    auto [g1, v1] = propagate_padded(1.0, true);
    auto active_error = [&](const doe::Field<double>& v, const doe::Grid& g) {
        const std::size_t n = static_cast<std::size_t>(g.n), n4 = static_cast<std::size_t>(g4.n);
        const std::size_t lo = (n - active) / 2, lo4 = (n4 - active) / 2;
        double num = 0, den = 0;
        for (int i = 0; i < active; ++i)
            for (int j = 0; j < active; ++j) {
                const auto ref = v4(lo4 + i, lo4 + j);
                num += std::norm(v(lo + i, lo + j) - ref);
                den += std::norm(ref);
            }
        return std::sqrt(num / den);
    };
    // Squared relative field error = wrapped energy / signal energy.
    const double e2 = std::pow(active_error(v2b, g2b), 2);
    const double e1 = std::pow(active_error(v1, g1), 2);
    INFO("relative error energy in the active region: 2x padding " << e2 << ", no padding " << e1);
    CHECK(e2 < 1e-3);        // measured 1.35e-4 for the hard-edged square at 5 cm
    CHECK(e1 > 100 * e2);
}

TEST_CASE("band limit: the paper's du = 1/S beats both 'off' and the reference's stricter cut", "[propagate][T7]") {
    // The band-limit angle satisfies tan(theta) = S / (2 z): it removes exactly
    // the plane-wave components that would travel more than half the window
    // sideways, i.e. the light that wraps. The Python reference used du = 2/S,
    // which discards half the legitimately propagating spectrum; measured on a
    // hard-edged square at 20 cm (wrapped energy / signal energy vs an 8x
    // padded reference): off 1.06e-2, paper 7.9e-4, reference 1.07e-2.
    const int active = 128;
    const double d = 0.2;
    auto propagate = [&](double pad, double f_limit) {
        auto g = doe::Grid::padded(active, 8e-6, 532e-9, pad);
        const std::size_t n = static_cast<std::size_t>(g.n), lo = (n - active) / 2;
        doe::Field<double> u(n, n);
        for (std::size_t i = lo; i < lo + active; ++i)
            for (std::size_t j = lo; j < lo + active; ++j) u(i, j) = 1.0;
        auto tf = doe::transfer_function(g, d, false);
        doe::Fft2<double> fft(n);
        auto U = fft.forward(u);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                U(i, j) *= (std::abs(g.f(i)) > f_limit || std::abs(g.f(j)) > f_limit) ? 0.0 : tf.H(i, j);
        return std::pair{g, fft.inverse(U)};
    };
    auto error_energy = [&](const doe::Field<double>& v, const doe::Grid& g, const doe::Field<double>& ref,
                            const doe::Grid& gr) {
        const std::size_t lo = (static_cast<std::size_t>(g.n) - active) / 2;
        const std::size_t lr = (static_cast<std::size_t>(gr.n) - active) / 2;
        double num = 0, den = 0;
        for (int i = 0; i < active; ++i)
            for (int j = 0; j < active; ++j) {
                num += std::norm(v(lo + i, lo + j) - ref(lr + i, lr + j));
                den += std::norm(ref(lr + i, lr + j));
            }
        return num / den;
    };
    const double inf = std::numeric_limits<double>::infinity();
    auto [g8, ref] = propagate(8.0, inf);  // 8x window: nothing wraps, nothing needs cutting
    auto g2 = doe::Grid::padded(active, 8e-6, 532e-9, 2.0);
    const double S = g2.extent();
    const double f_paper = 1.0 / (g2.wavelength * std::sqrt(std::pow(2 * d / S, 2) + 1));
    const double f_strict = 1.0 / (g2.wavelength * std::sqrt(std::pow(2 * d / (S / 2), 2) + 1));
    auto [ga, v_off] = propagate(2.0, inf);
    auto [gb, v_paper] = propagate(2.0, f_paper);
    auto [gc, v_strict] = propagate(2.0, f_strict);
    const double e_off = error_energy(v_off, ga, ref, g8);
    const double e_paper = error_energy(v_paper, gb, ref, g8);
    const double e_strict = error_energy(v_strict, gc, ref, g8);
    INFO("wrapped energy: off " << e_off << ", paper " << e_paper << ", strict " << e_strict);
    CHECK(e_paper < 1e-3);
    CHECK(e_off > 5 * e_paper);
    CHECK(e_strict > 5 * e_paper);
    // and the production propagator applies exactly the paper's cut
    doe::AngularSpectrum<double> A(g2, d, true);
    CHECK_THAT(doe::transfer_function(g2, d, true).f_limit, WithinRel(f_paper, 1e-15));
}

// ---------------------------------------------------------------------------
// Parity with the Python reference (band limit off: identical semantics)
// ---------------------------------------------------------------------------

TEST_CASE("oracle: forward/adjoint match optics.py on a fixed input", "[propagate][oracle]") {
    doe::Grid g{64, 8e-6, 532e-9};
    auto u = oracle_input(g);
    CHECK(std::abs(u(32, 32) - 1.0) < 1e-15);
    CHECK(std::abs(u(20, 40) - std::complex<double>(-0.45266508680234896, 0.17598693751503813)) < 1e-14);

    doe::AngularSpectrum<double> A(g, 0.02, false);
    CHECK_THAT(A.passband_fraction(), WithinAbs(1.0, 0.0));
    auto tf = doe::transfer_function(g, 0.02, false);
    CHECK(std::abs(tf.H(3, 5) - std::complex<double>(-0.2787113253624401, 0.9603749252842413)) < 1e-12);
    CHECK(std::abs(tf.H(0, 0) - std::complex<double>(0.995539707503161, -0.09434347240016427)) < 1e-12);

    auto v = A.forward(u);
    CHECK_THAT(energy(v), WithinRel(452.2398169233331, 1e-12));
    CHECK(std::abs(v(32, 32) - std::complex<double>(-0.2747625678732164, -0.4069645258809752)) < 1e-12);
    CHECK(std::abs(v(20, 40) - std::complex<double>(0.11438191186653568, 0.01810656768916843)) < 1e-12);
    CHECK(std::abs(v(45, 10) - std::complex<double>(0.29167218593647054, 0.5149766770568627)) < 1e-12);
    CHECK(std::abs(v(0, 0) - std::complex<double>(0.015434318220935536, 0.11868018888170859)) < 1e-12);
    auto a = A.adjoint(v);
    CHECK(std::abs(a(32, 32) - std::complex<double>(1.0, 5.551115123125783e-17)) < 1e-12);
}

// ---------------------------------------------------------------------------
// Analytic solutions
// ---------------------------------------------------------------------------

TEST_CASE("analytic: tilted beam centroid moves by d tan(asin(lambda f0))", "[propagate][analytic]") {
    doe::Grid g{128, 8e-6, 532e-9};
    const std::size_t n = 128;
    const double p = g.pitch;
    const double f0 = 4.0 / (n * p);                       // exactly on an FFT bin
    const double theta = std::asin(g.wavelength * f0);    // exact plane-wave angle
    const double shift_target = 5 * p;                    // five pixels
    const double d = shift_target / std::tan(theta);
    const double w0 = 16 * p;                             // Gaussian envelope, zR ~ 97 mm >> d
    doe::Field<double> u(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double x = g.x(i), y = g.x(j);
            u(i, j) = std::polar(std::exp(-(x * x + y * y) / (w0 * w0)), 2 * pi * f0 * x);
        }
    doe::AngularSpectrum<double> A(g, d, false);
    const auto before = moments_x(u, g);
    const auto after = moments_x(A.forward(u), g);
    CHECK_THAT(after.mean - before.mean, WithinRel(shift_target, 1e-4));
}

TEST_CASE("analytic: Talbot self-imaging at the exact non-paraxial distance", "[propagate][analytic]") {
    // u = 1 + m cos(2 pi f0 x) has three plane-wave components. They are back
    // in phase when (k - k_z(f0)) z = 2 pi (Montgomery 1967), so the field at
    // z_T = 2 pi / (k - k_z) equals exp(i k z_T) u exactly. At z_T / 2 the
    // +-1 orders pick up a sign: the image is shifted by half a period.
    doe::Grid g{64, 8e-6, 532e-9};
    const std::size_t n = 64;
    const double f0 = 5.0 / (n * g.pitch);
    const double k = g.k();
    const double kz = std::sqrt(k * k - std::pow(2 * pi * f0, 2));
    const double zT = 2 * pi / (k - kz);
    const double zT_paraxial = 2.0 / (g.wavelength * f0 * f0);  // Rayleigh 1881: 2 p^2 / lambda
    CHECK_THAT(zT, WithinRel(zT_paraxial, 1e-4));  // close, but not equal
    CHECK(zT != zT_paraxial);

    const double m = 0.6;
    doe::Field<double> u(n, n), u_half(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            u(i, j) = 1.0 + m * std::cos(2 * pi * f0 * g.x(i));
            u_half(i, j) = 1.0 - m * std::cos(2 * pi * f0 * g.x(i));
        }
    doe::AngularSpectrum<double> A(g, zT, false);
    auto v = A.forward(u);
    const auto phase = std::polar(1.0, k * zT);
    for (std::size_t q = 0; q < v.data.size(); ++q) CHECK(std::abs(v.data[q] - phase * u.data[q]) < 1e-9);

    doe::AngularSpectrum<double> A2(g, zT / 2, false);
    auto v2 = A2.forward(u);
    const auto phase2 = std::polar(1.0, k * zT / 2);
    for (std::size_t q = 0; q < v2.data.size(); ++q) CHECK(std::abs(v2.data[q] - phase2 * u_half.data[q]) < 1e-9);
}

TEST_CASE("T3 analytic: Gaussian beam waist w(z) = w0 sqrt(1 + (z/zR)^2)", "[propagate][T3][analytic]") {
    doe::Grid g{256, 8e-6, 532e-9};
    const std::size_t n = 256;
    const double w0 = 80e-6;  // 10 pixels
    const double zR = pi * w0 * w0 / g.wavelength;
    doe::Field<double> u(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double x = g.x(i), y = g.x(j);
            u(i, j) = std::exp(-(x * x + y * y) / (w0 * w0));
        }
    // Intensity ~ exp(-2 r^2 / w^2) has sigma = w / 2 along each axis.
    CHECK_THAT(2 * moments_x(u, g).sigma, WithinRel(w0, 1e-6));
    for (double z : {0.5 * zR, zR, 2.0 * zR}) {
        doe::AngularSpectrum<double> A(g, z, false);
        const double w = w0 * std::sqrt(1.0 + (z / zR) * (z / zR));
        CHECK_THAT(2 * moments_x(A.forward(u), g).sigma, WithinRel(w, 1e-3));  // spec: 1 %
    }
}

TEST_CASE("analytic: on-axis intensity of a circular aperture (exact scalar solution)", "[propagate][analytic]") {
    // Plane wave through a disk of radius a. On axis the Rayleigh-Sommerfeld
    // integral is elementary: U/U0 = exp(ikz) - (z/R) exp(ikR), R = sqrt(z^2 + a^2)
    // (Born & Wolf §8.3; in the Fresnel limit 4 sin^2(k a^2 / 4z), Wyant eq. 5).
    doe::Grid g{512, 8e-6, 532e-9};
    const std::size_t n = 512;
    const double a = 40 * g.pitch;
    doe::Field<double> u(n, n);
    // Area-weighted (4x4 supersampled) disk edge to keep pixelisation error small.
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            int inside = 0;
            for (int si = 0; si < 4; ++si)
                for (int sj = 0; sj < 4; ++sj) {
                    const double x = g.x(i) + (si - 1.5) * g.pitch / 4;
                    const double y = g.x(j) + (sj - 1.5) * g.pitch / 4;
                    if (x * x + y * y <= a * a) ++inside;
                }
            u(i, j) = inside / 16.0;
        }
    const double k = g.k();
    for (double NF : {1.0, 1.5, 3.0}) {  // Fresnel number a^2 / (lambda z)
        const double z = a * a / (g.wavelength * NF);
        const double R = std::sqrt(z * z + a * a);
        const std::complex<double> exact = std::polar(1.0, k * z) - (z / R) * std::polar(1.0, k * R);
        doe::AngularSpectrum<double> A(g, z, true);
        const auto v = A.forward(u);
        CHECK_THAT(std::norm(v(n / 2, n / 2)), WithinRel(std::norm(exact), 2e-2));
    }
}

// ---------------------------------------------------------------------------
// Sampling report
// ---------------------------------------------------------------------------

TEST_CASE("sampling_report reproduces the documented sampling formulas", "[propagate][sampling]") {
    auto g = doe::Grid::padded(512, 8e-6, 532e-9);
    const double d = 0.05, D = 512 * 8e-6;
    auto r = doe::sampling_report(g, d, D);
    const double theta = std::asin(532e-9 / 16e-6);
    CHECK_THAT(r.max_angle_rad, WithinRel(theta, 1e-15));
    CHECK_THAT(r.spot_size_m, WithinRel(532e-9 * d / D, 1e-15));
    CHECK_THAT(r.spot_size_px, WithinRel(532e-9 * d / D / 8e-6, 1e-15));
    CHECK_THAT(r.signal_spread_m, WithinRel(2 * d * std::tan(theta), 1e-15));
    CHECK_THAT(r.window_extent_m, WithinRel(g.extent(), 1e-15));
    CHECK_THAT(r.fresnel_number, WithinRel(D * D / (532e-9 * d), 1e-15));
    // Light leaving the plate edge reaches D/2 + d tan(theta); on the periodic window it
    // wraps to x - S, which misses the picture |x| <= D/2 as long as D + d tan(theta) <= S.
    CHECK(r.wrap_misses_picture == (D + 0.5 * r.signal_spread_m <= g.extent()));
    CHECK(r.wrap_misses_picture);
    CHECK(r.spot_resolved == (r.spot_size_px <= 1.0));

    // 8.19 mm window, 4.10 mm plate: the picture is clear up to d tan(theta) = 4.10 mm, i.e. 123 mm.
    CHECK(doe::sampling_report(g, 0.12, D).wrap_misses_picture);
    CHECK_FALSE(doe::sampling_report(g, 0.13, D).wrap_misses_picture);

    // A long distance violates both checks.
    auto far = doe::sampling_report(g, 5.0, D);
    CHECK_FALSE(far.wrap_misses_picture);
    CHECK_FALSE(far.spot_resolved);
}
