// FFT wrapper: orthonormal 2-D transforms in float and double.
//
// The propagator is built on 'ortho'-normalized FFTs so that, with a
// unit-modulus transfer function, it is a partial isometry and its adjoint is
// the propagator with the conjugate transfer function. These
// tests pin exactly that normalization: sum|F u|^2 = sum|u|^2 (Parseval),
// delta <-> constant 1/n, and a single-frequency phasor mapping to one bin
// of height n.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/array.hpp"
#include "doe/fft.hpp"

#include <cmath>
#include <complex>
#include <numbers>

namespace {
template <class T>
T tol() { return std::is_same_v<T, float> ? T(1e-4) : T(1e-12); }

template <class T>
doe::Field<T> ramp_field(std::size_t n) {
    doe::Field<T> u(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            u(i, j) = std::complex<T>(T(0.1) * T(i) - T(0.05) * T(j), std::sin(T(i * j) / T(7)));
    return u;
}
template <class T>
double sum_abs2(const doe::Field<T>& f) {
    double s = 0;
    for (const auto& z : f.data) s += std::norm(std::complex<double>(z));
    return s;
}
}  // namespace

TEMPLATE_TEST_CASE("fft2: inverse(forward(u)) == u", "[fft]", float, double) {
    using T = TestType;
    const std::size_t n = 24;  // 2^3 * 3
    doe::Fft2<T> fft(n);
    auto u = ramp_field<T>(n);
    auto v = fft.inverse(fft.forward(u));
    for (std::size_t k = 0; k < u.data.size(); ++k)
        CHECK(std::abs(v.data[k] - u.data[k]) < tol<T>() * T(10));
}

TEMPLATE_TEST_CASE("fft2: orthonormal (Parseval)", "[fft]", float, double) {
    using T = TestType;
    const std::size_t n = 30;
    doe::Fft2<T> fft(n);
    auto u = ramp_field<T>(n);
    auto U = fft.forward(u);
    CHECK_THAT(sum_abs2(U), Catch::Matchers::WithinRel(sum_abs2(u), double(tol<T>())));
    auto w = fft.inverse(u);
    CHECK_THAT(sum_abs2(w), Catch::Matchers::WithinRel(sum_abs2(u), double(tol<T>())));
}

TEST_CASE("fft2: delta maps to the constant 1/n, constant maps to n at DC", "[fft]") {
    const std::size_t n = 16;
    doe::Fft2<double> fft(n);
    doe::Field<double> delta(n, n);
    delta(0, 0) = 1.0;
    auto D = fft.forward(delta);
    for (const auto& z : D.data) CHECK(std::abs(z - std::complex<double>(1.0 / n, 0.0)) < 1e-14);

    doe::Field<double> one(n, n, std::complex<double>(1.0, 0.0));
    auto O = fft.forward(one);
    CHECK(std::abs(O(0, 0) - std::complex<double>(double(n), 0.0)) < 1e-12);
    CHECK(std::abs(O(1, 0)) < 1e-12);
    CHECK(std::abs(O(0, 3)) < 1e-12);
}

TEST_CASE("fft2: a single-frequency phasor lands in one bin with the numpy sign convention", "[fft]") {
    // u(i,j) = exp(+2 pi i * 3 i / n): numpy's forward transform uses exp(-2 pi i k n / N),
    // so the energy sits in bin (3, 0) with amplitude n (ortho: n^2 / n).
    const std::size_t n = 20;
    doe::Fft2<double> fft(n);
    doe::Field<double> u(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            u(i, j) = std::polar(1.0, 2.0 * std::numbers::pi * 3.0 * double(i) / double(n));
    auto U = fft.forward(u);
    CHECK(std::abs(U(3, 0) - std::complex<double>(double(n), 0.0)) < 1e-12);
    CHECK(std::abs(U(n - 3, 0)) < 1e-12);
    double off = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (!(i == 3 && j == 0)) off += std::norm(U(i, j));
    CHECK(off < 1e-20);
}

TEST_CASE("fft2: float and double transforms agree", "[fft]") {
    const std::size_t n = 28;  // 2^2 * 7
    auto ud = ramp_field<double>(n);
    doe::Field<float> uf(n, n);
    for (std::size_t k = 0; k < ud.data.size(); ++k) uf.data[k] = std::complex<float>(ud.data[k]);
    auto Ud = doe::Fft2<double>(n).forward(ud);
    auto Uf = doe::Fft2<float>(n).forward(uf);
    for (std::size_t k = 0; k < ud.data.size(); ++k)
        CHECK(std::abs(std::complex<double>(Uf.data[k]) - Ud.data[k]) < 1e-4 * (1.0 + std::abs(Ud.data[k])));
}

TEST_CASE("fft threads: configurable, and transforms are identical to 1e-12 with 1 or many threads", "[fft][threads]") {
    const std::size_t n = 60;
    auto u = ramp_field<double>(n);
    doe::set_fft_threads(1);
    CHECK(doe::fft_threads() == 1);
    auto U1 = doe::Fft2<double>(n).forward(u);
    doe::set_fft_threads(4);
    CHECK(doe::fft_threads() == 4);
    auto U4 = doe::Fft2<double>(n).forward(u);
    for (std::size_t k = 0; k < U1.data.size(); ++k) CHECK(std::abs(U1.data[k] - U4.data[k]) < 1e-12);
    doe::set_fft_threads(0);  // 0 = hardware concurrency
    CHECK(doe::fft_threads() >= 1);
}

TEST_CASE("Array2: shape, indexing, fill", "[array]") {
    doe::Array2<double> a(3, 5, 2.5);
    CHECK(a.rows == 3);
    CHECK(a.cols == 5);
    CHECK(a.size() == 15);
    a(2, 4) = -1.0;
    CHECK(a.data[2 * 5 + 4] == -1.0);
    CHECK(a(0, 0) == 2.5);
}

TEST_CASE("Fft2: move assignment over a live plan leaves a working plan of the new size; the source holds the displaced plan", "[fft]") {
    doe::Fft2<double> a(8), b(16);
    a = std::move(b);
    CHECK(a.size() == 16);
    CHECK(b.size() == 8);  // swapped in, destroyed with b
    doe::Field<double> u(16, 16);
    for (std::size_t k = 0; k < u.data.size(); ++k) u.data[k] = std::complex<double>(std::sin(0.3 * k), std::cos(0.7 * k));
    const auto back = a.inverse(a.forward(u));
    double worst = 0;
    for (std::size_t k = 0; k < u.data.size(); ++k) worst = std::max(worst, std::abs(back.data[k] - u.data[k]));
    CHECK(worst < 1e-12);
}

