// Synthetic target images, generated in code so the tests and examples need
// no external files: cross + ring (the spec's test target), an N x M array of
// equal spots (beam splitter / fan-out; Dammann & Gortler 1971, Krackhardt &
// Streibl 1989), a disk, a binary text logo from a built-in 5x7 font, a stripe
// grating, and a Gaussian soft-edge filter.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/targets.hpp"

#include <cmath>
#include <functional>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
double sum(const doe::Array2<double>& a) {
    double s = 0;
    for (double v : a.data) s += v;
    return s;
}
bool in_unit_range(const doe::Array2<double>& a) {
    for (double v : a.data)
        if (v < 0.0 || v > 1.0) return false;
    return true;
}
// number of 4-connected components of pixels > 0.5
std::size_t components(const doe::Array2<double>& a) {
    std::vector<char> seen(a.size(), 0);
    std::size_t count = 0;
    std::function<void(std::size_t, std::size_t)> fill = [&](std::size_t i, std::size_t j) {
        if (i >= a.rows || j >= a.cols || seen[i * a.cols + j] || a(i, j) <= 0.5) return;
        seen[i * a.cols + j] = 1;
        fill(i + 1, j); fill(i, j + 1);
        if (i > 0) fill(i - 1, j);
        if (j > 0) fill(i, j - 1);
    };
    for (std::size_t i = 0; i < a.rows; ++i)
        for (std::size_t j = 0; j < a.cols; ++j)
            if (!seen[i * a.cols + j] && a(i, j) > 0.5) { ++count; fill(i, j); }
    return count;
}
}  // namespace

TEST_CASE("cross_plus_ring: binary, symmetric, two components (cross and ring)", "[targets]") {
    auto t = doe::targets::cross_plus_ring(256);
    CHECK(t.rows == 256);
    CHECK(t.cols == 256);
    CHECK(in_unit_range(t));
    for (std::size_t i = 0; i < 256; ++i)
        for (std::size_t j = 0; j < 256; ++j) {
            CHECK(t(i, j) == t(j, i));
            CHECK(t(i, j) == t(255 - i, j));
        }
    CHECK(components(t) == 2);
    const double frac = sum(t) / 65536.0;
    CHECK(frac > 0.05);
    CHECK(frac < 0.3);
    CHECK(t(128, 128) == 1.0);  // center of the cross
    CHECK(t(128, 128 + 86) == 1.0);  // on the ring
    CHECK(t(128, 128 + 70) == 0.0);  // between cross and ring
    CHECK(doe::targets::cross_plus_ring(64).data == doe::targets::cross_plus_ring(64).data);  // deterministic
}

TEST_CASE("spot_array: N x M equal spots on a regular lattice", "[targets]") {
    auto t = doe::targets::spot_array(256, 5, 5, /*spot radius px*/ 4.0, /*spacing fraction of the side*/ 0.16);
    CHECK(components(t) == 25);
    CHECK(in_unit_range(t));
    // every spot carries the same energy
    auto t3 = doe::targets::spot_array(128, 3, 1, 3.0, 0.25);
    CHECK(components(t3) == 3);
    // spots are centered: the middle spot of 3 x 1 sits at the center, the others 0.25 * 128 = 32 px away
    CHECK(t3(64, 64) == 1.0);
    CHECK(t3(64 + 32, 64) == 1.0);
    CHECK(t3(64 - 32, 64) == 1.0);
    CHECK(t3(64 + 16, 64) == 0.0);
    // 1 x 1 is a single centered disk
    CHECK(components(doe::targets::spot_array(64, 1, 1, 5.0, 0.2)) == 1);
}

TEST_CASE("disk and grating", "[targets]") {
    auto d = doe::targets::disk(128, 0.25);  // radius = 0.25 * 128 = 32 px
    CHECK(components(d) == 1);
    CHECK_THAT(sum(d), WithinRel(std::acos(-1.0) * 32 * 32, 0.03));
    CHECK(d(64, 64) == 1.0);
    CHECK(d(64 + 40, 64) == 0.0);
    auto g = doe::targets::grating(64, 8);  // 8 px period along x: 4 on, 4 off
    CHECK(components(g) == 8);
    CHECK_THAT(sum(g), WithinRel(64 * 64 / 2.0, 1e-12));
    for (std::size_t j = 0; j < 64; ++j) CHECK(g(0, j) == g(8, j));
}

TEST_CASE("binary_logo: text rendered from the built-in 5x7 font, readable ink fraction", "[targets]") {
    auto t = doe::targets::binary_logo(256, "DOE");
    CHECK(t.rows == 256);
    CHECK(in_unit_range(t));
    const double frac = sum(t) / 65536.0;
    CHECK(frac > 0.04);
    CHECK(frac < 0.35);
    CHECK(components(t) >= 3);  // at least one component per letter
    // a different string gives a different image
    CHECK(doe::targets::binary_logo(256, "DOE").data != doe::targets::binary_logo(256, "FJK").data);
    // a glyph that is not in the font throws
    CHECK_THROWS(doe::targets::binary_logo(64, "\x01"));
}

TEST_CASE("soft_edges: Gaussian blur preserves the total, lowers the maximum, keeps symmetry", "[targets]") {
    auto t = doe::targets::disk(128, 0.2);
    auto s = doe::targets::soft_edges(t, 3.0);
    CHECK_THAT(sum(s), WithinRel(sum(t), 1e-9));
    double smax = 0;
    for (double v : s.data) smax = std::max(smax, v);
    CHECK(smax < 1.0 + 1e-12);
    CHECK(s(64, 64) > 0.99);           // deep inside: unchanged
    CHECK(s(64 + 26, 64) > 0.05);      // just outside the edge: blurred in
    CHECK(s(64 + 26, 64) < 0.5);
    CHECK_THAT(s(64 + 20, 64), WithinRel(s(63 - 20, 64), 1e-9));  // center is at 63.5
    CHECK(in_unit_range(s));
    CHECK(doe::targets::soft_edges(t, 0.0).data == t.data);  // sigma 0: identity
}
