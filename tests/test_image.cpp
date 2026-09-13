// Image I/O and color mapping: PNG (libpng) round trips, grayscale conversion,
// bilinear resampling, embedding into the padded window, the twilight (cyclic,
// for phase) and viridis (sequential, for intensity) LUTs sampled from
// matplotlib, and the HSV rendering of complex fields (hue = phase, value =
// amplitude).
//
// Conventions: Array2(i, j) has i along x and j along y; an image of an Array2
// with rows x cols has width = rows and height = cols, pixel (x = i, y = j).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/image.hpp"

#include <complex>
#include <filesystem>
#include <numbers>

using Catch::Matchers::WithinAbs;
constexpr double pi = std::numbers::pi;

namespace {
std::filesystem::path tmp(const char* name) { return std::filesystem::temp_directory_path() / name; }
}  // namespace

TEST_CASE("PNG: 8-bit gray and RGB round trips", "[image][png]") {
    doe::ImageU8 g{5, 3, 1, {}};
    g.data.resize(15);
    for (std::size_t k = 0; k < 15; ++k) g.data[k] = static_cast<std::uint8_t>(k * 17);
    doe::write_png(tmp("doe_test_gray.png"), g);
    auto g2 = doe::read_png(tmp("doe_test_gray.png"));
    CHECK(g2.width == 5);
    CHECK(g2.height == 3);
    CHECK(g2.channels == 1);
    CHECK(g2.data == g.data);

    doe::ImageU8 c{4, 2, 3, {}};
    c.data.resize(24);
    for (std::size_t k = 0; k < 24; ++k) c.data[k] = static_cast<std::uint8_t>(255 - k * 10);
    doe::write_png(tmp("doe_test_rgb.png"), c);
    auto c2 = doe::read_png(tmp("doe_test_rgb.png"));
    CHECK(c2.channels == 3);
    CHECK(c2.data == c.data);
    CHECK_THROWS(doe::read_png(tmp("doe_does_not_exist.png")));
}

TEST_CASE("to_gray: [0,1] luminance in the (i = x, j = y) layout", "[image]") {
    doe::ImageU8 c{2, 3, 3, {}};  // width 2, height 3
    c.data = {255, 0, 0,   0, 255, 0,      // y = 0: red, green
              0, 0, 255,   255, 255, 255,  // y = 1: blue, white
              0, 0, 0,     128, 128, 128}; // y = 2: black, gray
    auto a = doe::to_gray(c);
    CHECK(a.rows == 2);  // x
    CHECK(a.cols == 3);  // y
    CHECK_THAT(a(0, 0), WithinAbs(0.299, 1e-6));  // Rec. 601 luma of red
    CHECK_THAT(a(1, 0), WithinAbs(0.587, 1e-6));
    CHECK_THAT(a(0, 1), WithinAbs(0.114, 1e-6));
    CHECK_THAT(a(1, 1), WithinAbs(1.0, 1e-12));
    CHECK_THAT(a(0, 2), WithinAbs(0.0, 1e-12));
    CHECK_THAT(a(1, 2), WithinAbs(128.0 / 255.0, 1e-12));
    doe::ImageU8 g{1, 1, 1, {51}};
    CHECK_THAT(doe::to_gray(g)(0, 0), WithinAbs(0.2, 1e-12));
}

TEST_CASE("resample: bilinear, keeps constants and endpoints", "[image]") {
    doe::Array2<double> c(4, 6, 0.75);
    auto up = doe::resample(c, 9, 13);
    CHECK(up.rows == 9);
    CHECK(up.cols == 13);
    for (double v : up.data) CHECK_THAT(v, WithinAbs(0.75, 1e-12));

    // linear ramp along x stays linear (bilinear is exact for linear data)
    doe::Array2<double> ramp(5, 2);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 2; ++j) ramp(i, j) = double(i);
    auto r = doe::resample(ramp, 9, 2);
    for (std::size_t i = 0; i < 9; ++i) CHECK_THAT(r(i, 0), WithinAbs(double(i) * 4.0 / 8.0, 1e-12));
    // downsampling a ramp keeps the endpoints
    auto d = doe::resample(ramp, 3, 2);
    CHECK_THAT(d(0, 0), WithinAbs(0.0, 1e-12));
    CHECK_THAT(d(2, 0), WithinAbs(4.0, 1e-12));
    CHECK_THAT(d(1, 0), WithinAbs(2.0, 1e-12));
    CHECK(doe::resample(ramp, 5, 2).data == ramp.data);  // identity at the same size
}

TEST_CASE("embed: centered in an n x n zero window", "[image]") {
    doe::Array2<double> src(4, 2, 1.0);
    auto e = doe::embed(src, 8);
    CHECK(e.rows == 8);
    CHECK(e.cols == 8);
    double sum = 0;
    for (double v : e.data) sum += v;
    CHECK(sum == 8.0);
    CHECK(e(2, 3) == 1.0);
    CHECK(e(5, 4) == 1.0);
    CHECK(e(1, 3) == 0.0);
    CHECK(e(6, 3) == 0.0);
    CHECK(e(2, 2) == 0.0);
    CHECK(e(2, 5) == 0.0);
    CHECK_THROWS(doe::embed(src, 3));
}

TEST_CASE("crop_center: inverse of embed", "[image]") {
    doe::Array2<double> src(4, 2, 0.0);
    for (std::size_t k = 0; k < 8; ++k) src.data[k] = double(k) + 1.0;
    auto e = doe::embed(src, 8);
    auto c = doe::crop_center(e, 4, 2);
    CHECK(c.rows == 4);
    CHECK(c.cols == 2);
    CHECK(c.data == src.data);
    auto sq = doe::crop_center(e, 4);
    CHECK(sq.rows == 4);
    CHECK(sq.cols == 4);
    CHECK(sq(0, 1) == 1.0);  // src(0,0) sits at column offset (4-2)/2 = 1
    CHECK_THROWS(doe::crop_center(e, 9));
}

TEST_CASE("colormaps: LUT endpoints from matplotlib, twilight is cyclic, inputs are clamped", "[image][colormap]") {
    auto t0 = doe::colormap_twilight(0.0), t1 = doe::colormap_twilight(1.0), tm = doe::colormap_twilight(0.5);
    CHECK((t0.r == 226 && t0.g == 217 && t0.b == 226));
    CHECK((t1.r == t0.r && t1.g == t0.g && t1.b == t0.b));
    CHECK((tm.r == 48 && tm.g == 20 && tm.b == 55));
    auto v0 = doe::colormap_viridis(0.0), v1 = doe::colormap_viridis(1.0);
    CHECK((v0.r == 68 && v0.g == 1 && v0.b == 84));
    CHECK((v1.r == 253 && v1.g == 231 && v1.b == 37));
    auto lo = doe::colormap_viridis(-3.0), hi = doe::colormap_viridis(7.0);
    CHECK((lo.r == v0.r && hi.r == v1.r));
}

TEST_CASE("hsv_phasor: hue from the phase, value from the amplitude", "[image][colormap]") {
    auto red = doe::hsv_phasor(std::polar(1.0, 0.0), 1.0);
    CHECK((red.r == 255 && red.g == 0 && red.b == 0));
    auto green = doe::hsv_phasor(std::polar(1.0, 2 * pi / 3), 1.0);
    CHECK((green.g == 255 && green.r == 0 && green.b == 0));
    auto blue = doe::hsv_phasor(std::polar(1.0, -2 * pi / 3), 1.0);
    CHECK((blue.b == 255 && blue.r == 0 && blue.g == 0));
    auto dark = doe::hsv_phasor(std::polar(0.0, 1.0), 1.0);
    CHECK((dark.r == 0 && dark.g == 0 && dark.b == 0));
    auto half = doe::hsv_phasor(std::polar(0.5, 0.0), 1.0);
    CHECK(half.r == 128);
    CHECK(half.g == 0);
}

TEST_CASE("render_phase / render_intensity / render_hsv produce images of the right size and mapping", "[image]") {
    auto px = [](const doe::ImageU8& im, std::size_t x, std::size_t y) { return im.data[(y * im.width + x) * 3]; };  // red channel
    doe::Array2<double> phi(3, 2, 0.0);
    phi(0, 0) = -pi + 1e-9;
    phi(1, 0) = 0.0;
    phi(2, 0) = pi;
    auto img = doe::render_phase(phi);
    CHECK(img.width == 3);
    CHECK(img.height == 2);
    CHECK(img.channels == 3);
    // phase -pi and +pi map to the same (cyclic) color; phase 0 to the middle of twilight
    CHECK(px(img, 0, 0) == 226);
    CHECK(px(img, 2, 0) == 226);
    CHECK(px(img, 1, 0) == 48);

    doe::Array2<double> I(2, 2, 0.0);
    I(0, 0) = 1.0;
    I(1, 0) = 0.5;
    I(0, 1) = 1e-9;
    I(1, 1) = 0.0;
    auto lin = doe::render_intensity(I, false, 40.0);
    CHECK(px(lin, 0, 0) == 253);  // max -> viridis(1)
    CHECK(px(lin, 1, 1) == 68);   // zero -> viridis(0)
    auto lg = doe::render_intensity(I, true, 40.0);
    CHECK(px(lg, 0, 0) == 253);
    CHECK(px(lg, 0, 1) == 68);    // 1e-9 is 90 dB down: clamped to the floor
    CHECK(px(lg, 1, 0) != 253);   // 0.5 is 3 dB down: distinct from the max
    CHECK(px(lg, 1, 0) != 68);

    doe::Field<double> v(2, 1);
    v(0, 0) = std::polar(1.0, 0.0);
    v(1, 0) = std::polar(0.0, 0.0);
    auto h = doe::render_hsv(v);
    CHECK((h.data[0] == 255 && h.data[1] == 0 && h.data[2] == 0));
    CHECK((h.data[3] == 0 && h.data[4] == 0 && h.data[5] == 0));
}
