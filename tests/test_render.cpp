// Rendering layer: streaming volume sweep between the DOE and
// target planes, xz / yz cross-sections, per-plane vortex statistics, the
// .doev volume file for the viewer, and the four figures.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/image.hpp"
#include "doe/pipeline.hpp"
#include "doe/propagate.hpp"
#include "doe/render.hpp"
#include "doe/targets.hpp"
#include "doe/vortex.hpp"

#include <cmath>
#include <filesystem>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
namespace fs = std::filesystem;

namespace {
struct Small {
    doe::DesignConfig cfg;
    doe::DesignResult result;
    Small() {
        cfg.active = 48;
        cfg.iters = 5;
        cfg.gs = {4, 2};
        cfg.run_gs = false;
        result = doe::design(doe::targets::cross_plus_ring(48), cfg);
    }
};
}  // namespace

TEST_CASE("AngularSpectrum::sweep streams the same planes as volume()", "[render][T10]") {
    doe::Grid g{32, 8e-6, 532e-9};
    doe::AngularSpectrum<double> A(g, 0.02, true);
    doe::Field<double> u(32, 32);
    for (std::size_t k = 0; k < u.data.size(); ++k) u.data[k] = std::polar(1.0, 0.01 * double(k));
    std::vector<double> zs{0.0, 0.005, 0.02, 0.03};
    auto planes = A.volume(u, zs);
    std::size_t seen = 0;
    A.sweep(u, zs, [&](std::size_t p, const doe::Field<double>& v) {
        CHECK(p == seen);
        for (std::size_t k = 0; k < v.data.size(); ++k) CHECK(v.data[k] == planes[p].data[k]);
        ++seen;
    });
    CHECK(seen == 4);
}

TEST_CASE("sweep_volume: planes, cuts, vortex statistics", "[render]") {
    Small S;
    const auto& in = S.result.inputs;
    doe::Field<double> u(in.grid.n, in.grid.n);
    for (std::size_t k = 0; k < u.data.size(); ++k) u.data[k] = std::polar(in.illum.data[k], S.result.runs.back().phi.data[k]);
    const int nz = 7;
    const std::size_t view = 40;
    auto vol = doe::sweep_volume(u, in.grid, 0.0, S.cfg.distance, nz, view, true);
    REQUIRE(vol.z.size() == nz);
    CHECK(vol.z.front() == 0.0);
    CHECK_THAT(vol.z.back(), WithinRel(S.cfg.distance, 1e-15));
    CHECK(vol.view == view);
    REQUIRE(vol.amplitude.size() == nz);
    REQUIRE(vol.phase.size() == nz);
    REQUIRE(vol.charges.size() == nz);
    CHECK(vol.amplitude[0].rows == view);
    CHECK(vol.charges[0].rows == view - 1);
    CHECK(vol.xz.rows == nz);
    CHECK(vol.xz.cols == view);
    CHECK(vol.yz.rows == nz);
    CHECK(vol.vortex_count.size() == nz);
    CHECK(vol.vortex_density.size() == nz);
    CHECK_THAT(vol.pitch, WithinRel(in.grid.pitch, 1e-15));

    // plane 0 is the (cropped) input field; the last plane is the reconstruction
    const auto u_c = doe::crop_center_field(u, view);
    for (std::size_t k = 0; k < u_c.data.size(); ++k) {
        CHECK_THAT(double(vol.amplitude[0].data[k]), WithinAbs(std::abs(u_c.data[k]), 1e-6));
        if (std::abs(u_c.data[k]) > 0) CHECK_THAT(double(vol.phase[0].data[k]), WithinAbs(std::arg(u_c.data[k]), 1e-6));
    }
    doe::AngularSpectrum<double> A(in.grid, S.cfg.distance, true);
    const auto v_c = doe::crop_center_field(A.forward(u), view);
    for (std::size_t k = 0; k < v_c.data.size(); ++k) CHECK_THAT(double(vol.amplitude.back().data[k]), WithinAbs(std::abs(v_c.data[k]), 1e-6));

    // cuts are the center row / column intensities of each plane
    const std::size_t c = view / 2;
    for (int p = 0; p < nz; ++p)
        for (std::size_t i = 0; i < view; ++i) {
            CHECK_THAT(vol.xz(p, i), WithinRel(std::pow(double(vol.amplitude[p](i, c)), 2) + 1e-30, 1e-5));
            CHECK_THAT(vol.yz(p, i), WithinRel(std::pow(double(vol.amplitude[p](c, i)), 2) + 1e-30, 1e-5));
        }
    // vortex statistics agree with the counter applied to the stored plane
    for (int p = 0; p < nz; ++p) {
        doe::Field<double> plane(view, view);
        for (std::size_t k = 0; k < plane.data.size(); ++k) plane.data[k] = std::polar(double(vol.amplitude[p].data[k]), double(vol.phase[p].data[k]));
        auto q = doe::vortex_charge_map(plane);
        CHECK(vol.vortex_count[p] == doe::vortex_count(q));
        for (std::size_t k = 0; k < q.data.size(); ++k) CHECK(int(vol.charges[p].data[k]) == q.data[k]);
    }
}

TEST_CASE("doev: volume file round trip", "[render][doev]") {
    Small S;
    const auto& in = S.result.inputs;
    doe::Field<double> u(in.grid.n, in.grid.n);
    for (std::size_t k = 0; k < u.data.size(); ++k) u.data[k] = std::polar(in.illum.data[k], S.result.runs.back().phi.data[k]);
    auto vol = doe::sweep_volume(u, in.grid, 0.0, S.cfg.distance, 5, 24, true);
    vol.doe_phase = doe::crop_center(S.result.runs.back().phi, 24);
    vol.target = doe::crop_center(in.i_target, 24);
    vol.wavelength = in.grid.wavelength;
    const fs::path p = fs::temp_directory_path() / "doe_test.doev";
    doe::write_doev(p, vol);
    auto back = doe::read_doev(p);
    CHECK(back.view == 24);
    CHECK(back.z == vol.z);
    CHECK(back.pitch == vol.pitch);
    CHECK(back.wavelength == vol.wavelength);
    for (std::size_t q = 0; q < 5; ++q) {
        CHECK(back.amplitude[q].data == vol.amplitude[q].data);
        CHECK(back.phase[q].data == vol.phase[q].data);
        CHECK(back.charges[q].data == vol.charges[q].data);
    }
    CHECK(back.xz.data == vol.xz.data);
    CHECK(back.yz.data == vol.yz.data);
    CHECK(back.vortex_count == vol.vortex_count);
    CHECK(back.doe_phase.data == vol.doe_phase.data);
    CHECK(back.target.data == vol.target.data);
    CHECK_THROWS(doe::read_doev(fs::temp_directory_path() / "doe_missing.doev"));
}

TEST_CASE("doev v2: full-resolution DOE and illumination travel with the file so a viewer can re-sweep", "[render][doev]") {
    Small S;
    const auto& in = S.result.inputs;
    doe::Field<double> u(in.grid.n, in.grid.n);
    for (std::size_t k = 0; k < u.data.size(); ++k) u.data[k] = std::polar(in.illum.data[k], S.result.runs.back().phi.data[k]);
    auto vol = doe::sweep_volume(u, in.grid, 0.0, S.cfg.distance, 3, 16, true);
    doe::attach_source(vol, S.result);  // full phase, illumination, grid, distance, band limit
    CHECK(vol.has_source());
    const fs::path p = fs::temp_directory_path() / "doe_test_v2.doev";
    doe::write_doev(p, vol);
    auto back = doe::read_doev(p);
    REQUIRE(back.has_source());
    CHECK(back.source.n == in.grid.n);
    CHECK(back.source.pitch == in.grid.pitch);
    CHECK(back.source.wavelength == in.grid.wavelength);
    CHECK(back.source.distance == S.cfg.distance);
    CHECK(back.source.band_limit == S.cfg.band_limit);
    CHECK(back.source.phase.data == vol.source.phase.data);
    CHECK(back.source.illum.data == vol.source.illum.data);
    CHECK(back.source.target.data == vol.source.target.data);
    // and a sweep from the source reproduces the stored planes
    auto again = doe::sweep_from_source(back, 3, 16, S.cfg.distance);
    for (std::size_t q = 0; q < 3; ++q)
        for (std::size_t k = 0; k < again.amplitude[q].data.size(); ++k)
            CHECK_THAT(double(again.amplitude[q].data[k]), WithinAbs(double(vol.amplitude[q].data[k]), 1e-5));
    // files written by write_figures carry the source
    const fs::path dir = fs::temp_directory_path() / "doe_figures_v2";
    fs::remove_all(dir);
    doe::write_figures(S.result, dir, 3, 16);
    CHECK(doe::read_doev(dir / "volume.doev").has_source());
}

TEST_CASE("write_figures: the four spec figures, the cuts and the vortex plot are written", "[render][figures]") {
    Small S;
    const fs::path dir = fs::temp_directory_path() / "doe_figures_test";
    fs::remove_all(dir);
    auto files = doe::write_figures(S.result, dir, /*nz=*/6, /*view=*/32);
    for (const char* name : {"fig1_doe_phase.png", "fig2_target_reconstruction.png", "fig3_xz_cross_section.png",
                             "fig4_hsv_planes.png", "vortex_density.svg", "volume.doev", "xz_cut.png", "yz_cut.png"})
        CHECK(fs::exists(dir / name));
    CHECK(files.size() >= 8);
    auto f1 = doe::read_png(dir / "fig1_doe_phase.png");
    CHECK(f1.width == 48);  // active aperture
    auto f2 = doe::read_png(dir / "fig2_target_reconstruction.png");
    CHECK(f2.width == 3 * 32 + 2 * 4);  // target | reconstruction | difference, 4 px gaps, cropped to the view
    CHECK(f2.height == 32);
    auto f3 = doe::read_png(dir / "fig3_xz_cross_section.png");
    CHECK(f3.width == 32);   // x across
    CHECK(f3.height >= 6);   // z down, upsampled to a readable height
    auto f4 = doe::read_png(dir / "fig4_hsv_planes.png");
    CHECK(f4.height == 32);
    CHECK(f4.width >= 2 * 32);  // a row of planes
    auto vol = doe::read_doev(dir / "volume.doev");
    CHECK(vol.z.size() == 6);
    CHECK(vol.view == 32);
}
