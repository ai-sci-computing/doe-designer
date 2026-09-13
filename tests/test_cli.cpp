// Command-line front end: argument parsing into a DesignConfig and the output
// writer (DOE phase as .npy + PNG, reconstruction, target, report JSON,
// markdown table, convergence SVG, per-run phases). Kept in the library so it
// can be tested without spawning the executable.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/cli.hpp"
#include "doe/image.hpp"
#include "doe/io.hpp"
#include "doe/pipeline.hpp"
#include "doe/targets.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

using Catch::Matchers::WithinRel;
namespace fs = std::filesystem;

namespace {
doe::CliOptions parse(std::vector<const char*> argv) {
    argv.insert(argv.begin(), "doe_design");
    return doe::parse_cli(static_cast<int>(argv.size()), const_cast<char**>(argv.data()));
}
}  // namespace

TEST_CASE("parse_cli: defaults and every option", "[cli]") {
    auto d = parse({"--target", "t.png"});
    CHECK(d.target == "t.png");
    CHECK(d.out == "results");
    CHECK(d.config.distance == 0.05);
    CHECK(d.config.active == 512);
    CHECK(d.config.levels == 0);
    CHECK(d.config.quant_ramp == doe::QuantRamp::linear);
    CHECK_FALSE(d.help);

    auto o = parse({"--target", "logo.png", "--distance", "0.1", "--levels", "4", "--out", "res/", "--iters", "250",
                    "--mu", "0.5", "--init", "random", "--pitch", "6e-6", "--wavelength", "633e-9", "--active", "300",
                    "--single", "--seed", "7", "--no-band-limit", "--quant", "choi", "--quant-start", "0.6", "--quant-ramp", "table", "--gs-cycles", "20,60",
                    "--illum", "gaussian", "--lr", "0.02", "--no-gs", "--no-letterbox", "--soft-edge", "1.5", "--nz", "48",
                    "--view-size", "200", "--choi-gain", "200,800", "--choi-tau", "3", "--threads", "3"});
    CHECK(o.target == "logo.png");
    CHECK(o.out == "res/");
    CHECK(o.config.distance == 0.1);
    CHECK(o.config.levels == 4);
    CHECK(o.config.iters == 250);
    CHECK(o.config.mu == 0.5);
    CHECK(o.config.init == doe::InitMethod::random);
    CHECK(o.config.pitch == 6e-6);
    CHECK(o.config.wavelength == 633e-9);
    CHECK(o.config.active == 300);
    CHECK(o.config.single_precision);
    CHECK(o.config.seed == 7);
    CHECK_FALSE(o.config.band_limit);
    CHECK_FALSE(o.config.letterbox);
    CHECK(o.config.quant_method == doe::QuantMethod::choi);
    CHECK(o.config.quant_start == 0.6);
    CHECK(o.config.quant_ramp == doe::QuantRamp::table);
    CHECK(o.config.gs.phase_only_iters == 20);
    CHECK(o.config.gs.iters == 80);  // 20 zeroed-outside cycles + 60 free-amplitude cycles
    CHECK(o.config.illum == doe::IllumShape::gaussian);
    CHECK(o.config.lr == 0.02);
    CHECK_FALSE(o.config.run_gs);
    CHECK(o.config.soft_edge_sigma == 1.5);
    CHECK(o.nz == 48);
    CHECK(o.view_size == 200);
    CHECK(o.config.choi.choi_gain0 == 200.0);
    CHECK(o.config.choi.choi_gain1 == 800.0);
    CHECK(o.config.choi.choi_tau0 == 3.0);
    CHECK(o.threads == 3);
    CHECK(d.threads == 0);  // default: all cores

    CHECK(parse({"--help"}).help);
    CHECK(parse({"-h"}).help);
    auto m = parse({"--make-targets", "images/targets"});
    CHECK(m.make_targets == "images/targets");
}

TEST_CASE("parse_cli: errors are reported, not ignored", "[cli]") {
    CHECK_THROWS(parse({"--target"}));                     // missing value
    CHECK_THROWS(parse({"--target", "t.png", "--bogus"}));  // unknown option
    CHECK_THROWS(parse({"--target", "t.png", "--levels", "x"}));
    CHECK_THROWS(parse({"--target", "t.png", "--init", "magic"}));
    CHECK(parse({"--target", "t.png", "--init", "backprop"}).config.init == doe::InitMethod::backprop);
    CHECK_THROWS(parse({"--target", "t.png", "--quant", "magic"}));
    CHECK_THROWS(parse({"--target", "t.png", "--quant-ramp", "magic"}));
    CHECK_THROWS(parse({"--target", "t.png", "--gs-cycles", "20"}));
    CHECK_THROWS(parse({"--target", "t.png", "--illum", "magic"}));
    CHECK_THROWS(parse({}));  // neither --target nor --make-targets nor --help
}

TEST_CASE("usage text lists every option", "[cli]") {
    const std::string u = doe::cli_usage();
    for (const char* opt : {"--target", "--distance", "--levels", "--out", "--iters", "--mu", "--init", "--pitch", "--wavelength",
                            "--active", "--single", "--seed", "--no-band-limit", "--quant", "--quant-start", "--quant-ramp", "--gs-cycles", "--illum", "--lr",
                            "--no-gs", "--no-letterbox", "--soft-edge", "--nz", "--view-size", "--make-targets", "--choi-gain", "--choi-tau", "--threads"})
        CHECK(u.find(opt) != std::string::npos);
}

TEST_CASE("write_targets: the synthetic targets are written as PNGs", "[cli]") {
    const fs::path dir = fs::temp_directory_path() / "doe_targets_test";
    fs::remove_all(dir);
    auto written = doe::write_targets(dir, 128);
    CHECK(written.size() >= 4);
    for (const auto& p : written) CHECK(fs::exists(p));
    CHECK(fs::exists(dir / "cross_ring.png"));
    CHECK(fs::exists(dir / "spot_array_5x5.png"));
    CHECK(fs::exists(dir / "logo.png"));
    CHECK(fs::exists(dir / "disk.png"));
    auto img = doe::read_png(dir / "cross_ring.png");
    CHECK(img.width == 128);
    CHECK(img.channels == 1);
}

TEST_CASE("write_outputs: a tiny design run produces the documented files", "[cli]") {
    const fs::path dir = fs::temp_directory_path() / "doe_outputs_test";
    fs::remove_all(dir);
    doe::DesignConfig cfg;
    cfg.active = 32;
    cfg.iters = 3;
    cfg.levels = 4;
    cfg.gs = {4, 2};
    auto r = doe::design(doe::targets::disk(32, 0.3), cfg);
    auto files = doe::write_outputs(r, dir, /*nz=*/5, /*view=*/24);
    for (const char* name : {"phase.npy", "phase.png", "reconstruction.png", "target.png", "report.json", "table.md",
                             "convergence.svg", "convergence_terms.svg", "phase_gs.npy", "phase_adam.npy", "phase_adam_q4.npy", "history.csv",
                             "fig1_doe_phase.png", "fig2_target_reconstruction.png", "fig3_xz_cross_section.png",
                             "fig4_hsv_planes.png", "vortex_density.svg", "volume.doev", "vortex_map.png",
                             "reconstruction_gs.png", "reconstruction_adam.png", "reconstruction_adam_q4.png",
                             "phase_gs.png", "phase_adam.png", "phase_adam_q4.png"})
        CHECK(fs::exists(dir / name));
    CHECK(files.size() >= 25);
    // history.csv carries the total energy and its two terms per run (separate convergence curves)
    {
        std::ifstream h(dir / "history.csv");
        std::string header;
        std::getline(h, header);
        CHECK(header == "iteration,gs,gs_shape,gs_efficiency,adam,adam_shape,adam_efficiency,adam_q4,adam_q4_shape,adam_q4_efficiency");
    }
    auto vm = doe::read_png(dir / "vortex_map.png");
    CHECK(vm.width == 24);   // the view crop
    CHECK(vm.channels == 3);
    auto rp = doe::read_png(dir / "reconstruction_gs.png");
    CHECK(rp.width == static_cast<std::size_t>(r.inputs.grid.n));   // full window (240 px: picture-clear rule), like reconstruction.png
    CHECK(r.inputs.grid.n == 240);
    // the report carries the vortex statistics
    std::ifstream f(dir / "report.json");
    std::stringstream ss;
    ss << f.rdbuf();
    CHECK(ss.str().find("\"vortex_count\"") != std::string::npos);
    CHECK(ss.str().find("\"vortex_density_bright_per_mm2\"") != std::string::npos);
    // phase.npy is the final (quantized) design cropped to the active aperture:
    // the DOE is the aperture, the padding carries no illumination
    auto phi = doe::read_npy(dir / "phase.npy");
    CHECK(phi.rows == 32);
    CHECK(phi.cols == 32);
    CHECK(phi.data == doe::crop_center(r.runs.back().phi, 32).data);
    auto png = doe::read_png(dir / "phase.png");
    CHECK(png.width == 32);
    CHECK(png.channels == 3);
    // the reconstruction and target images keep the full window (the padding is the don't-care region)
    CHECK(doe::read_png(dir / "reconstruction.png").width == static_cast<std::size_t>(r.inputs.grid.n));
}
