// End-to-end pipeline: T8, T9, T13, T14 at the documented budgets,
// on the binary test target (cross + ring) at 256 px active aperture.
// These take tens of seconds and are tagged [slow].
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinRel;

#include "doe/pipeline.hpp"
#include "doe/targets.hpp"

#include <cmath>
#include <numbers>

using Catch::Matchers::WithinAbs;
constexpr double pi = std::numbers::pi;

namespace {
doe::DesignConfig base_config() {
    doe::DesignConfig c;
    c.active = 256;
    c.distance = 0.05;
    c.iters = 400;
    c.mu = 0.3;
    c.init = doe::InitMethod::tie;
    c.seed = 1;
    return c;
}
}  // namespace

TEST_CASE("prepare: target resampled to the active aperture, embedded, b = sqrt(I), mask = active window", "[pipeline]") {
    auto target = doe::targets::disk(100, 0.3);  // any size in
    auto cfg = base_config();
    cfg.active = 64;
    auto in = doe::prepare(target, cfg);
    // the window follows the picture-clear rule: 0.51 mm plate + 1.66 mm reach -> 272 -> 280 px (7-smooth)
    CHECK(in.grid.n == 280);
    CHECK(in.b.rows == static_cast<std::size_t>(in.grid.n));
    const std::size_t c = in.grid.n / 2;   // center of the padded grid
    double mask_sum = 0, illum_e = 0, b_max = 0;
    for (std::size_t k = 0; k < in.mask.data.size(); ++k) {
        mask_sum += in.mask.data[k];
        illum_e += in.illum.data[k] * in.illum.data[k];
        b_max = std::max(b_max, in.b.data[k]);
    }
    CHECK(mask_sum == 64 * 64);
    CHECK(illum_e == 64 * 64);  // square, uniform
    CHECK_THAT(b_max, WithinAbs(1.0, 1e-12));
    CHECK(in.b(c, c) == 1.0);
    CHECK(in.b(c + 25, c) == 0.0);  // outside the (resampled) disk (radius 19 px), inside the window (ends at c + 31)
    CHECK(in.mask(c + 25, c) == 1.0);
    CHECK(in.mask(10, 10) == 0.0);    // padding
    CHECK(in.illum(10, 10) == 0.0);
    // b^2 is the (resampled) target intensity used for TIE
    for (std::size_t k = 0; k < in.b.data.size(); ++k) CHECK_THAT(in.i_target.data[k], WithinAbs(in.b.data[k] * in.b.data[k], 1e-12));

    cfg.illum = doe::IllumShape::disk;
    auto d = doe::prepare(target, cfg);
    double de = 0;
    for (double v : d.illum.data) de += v * v;
    CHECK_THAT(de, WithinAbs(pi * 32 * 32, 0.03 * pi * 32 * 32));
    cfg.illum = doe::IllumShape::gaussian;
    auto g = doe::prepare(target, cfg);
    CHECK(g.illum(c, c) > 0.99);
    CHECK(g.illum(c + 31, c) < 0.5);
    CHECK(g.illum(c + 33, c) == 0.0);  // truncated at the aperture
}

TEST_CASE("prepare: a non-square target keeps its aspect ratio; the longer side fills the aperture and the window is that rectangle", "[pipeline]") {
    // 100 x 60 (i along x: 100 wide, 60 high) with a bright band in the middle rows
    doe::Array2<double> target(100, 60, 0.0);
    for (std::size_t i = 40; i < 60; ++i)
        for (std::size_t j = 0; j < 60; ++j) target(i, j) = 1.0;
    auto cfg = base_config();
    cfg.active = 64;
    cfg.letterbox = false;   // pure rectangular window; the letterbox default is tested below
    auto in = doe::prepare(target, cfg);
    REQUIRE(in.grid.n == 280);
    // window: 64 x 38 (60 * 64 / 100 = 38.4 -> 38), centered in the padded grid
    double mask_sum = 0, illum_e = 0;
    for (std::size_t k = 0; k < in.mask.data.size(); ++k) {
        mask_sum += in.mask.data[k];
        illum_e += in.illum.data[k] * in.illum.data[k];
    }
    CHECK(mask_sum == 64 * 38);
    CHECK(illum_e == 64 * 64);  // the DOE aperture stays square
    const std::size_t n = in.grid.n, c = n / 2, i0 = (n - 64) / 2, j0 = (n - 38) / 2;
    CHECK(in.mask(i0, j0) == 1.0);
    CHECK(in.mask(i0 + 63, j0 + 37) == 1.0);
    CHECK(in.mask(i0, j0 - 1) == 0.0);        // beside the rectangle: free region
    CHECK(in.mask(i0 + 63, j0 + 38) == 0.0);
    CHECK(in.illum(i0, j0 - 1) == 1.0);       // ... but still illuminated
    // the band lands in the middle of the window, with the target's proportions
    CHECK(in.b(c, c) == 1.0);
    CHECK(in.b(i0 + 5, c) == 0.0);
    CHECK(in.b(i0 + 63 - 5, c) == 0.0);
    // a portrait target is handled the same way (the other side is the long one)
    doe::Array2<double> tall(60, 100, 1.0);
    auto tin = doe::prepare(tall, cfg);
    // the metrics mask is the image rectangle in every mode
    double msum = 0;
    for (double v : in.metrics_mask.data) msum += v;
    CHECK(msum == 64 * 38);
    double tsum = 0;
    for (double v : tin.mask.data) tsum += v;
    CHECK(tsum == 38 * 64);
    CHECK(tin.mask((tin.grid.n - 38) / 2 - 1, tin.grid.n / 2) == 0.0);
    CHECK(tin.mask((tin.grid.n - 38) / 2, tin.grid.n / 2) == 1.0);
}

TEST_CASE("prepare: letterbox (default) keeps the square window with the strips forced dark; metrics on the image rectangle", "[pipeline]") {
    doe::Array2<double> target(100, 60, 0.0);   // landscape, bright band in the middle rows
    for (std::size_t i = 40; i < 60; ++i)
        for (std::size_t j = 0; j < 60; ++j) target(i, j) = 1.0;
    auto cfg = base_config();
    cfg.active = 64;
    CHECK(cfg.letterbox);   // the default
    auto in = doe::prepare(target, cfg);
    double mask_sum = 0, metrics_sum = 0, strip_b = 0;
    const std::size_t n = in.grid.n, i0 = (n - 64) / 2, j0 = (n - 38) / 2;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            mask_sum += in.mask(i, j);
            metrics_sum += in.metrics_mask(i, j);
            const bool in_square = i >= i0 && i < i0 + 64 && j >= i0 && j < i0 + 64;
            const bool in_rect = i >= i0 && i < i0 + 64 && j >= j0 && j < j0 + 38;
            if (in_square && !in_rect) strip_b += in.b(i, j);
        }
    CHECK(mask_sum == 64 * 64);        // energy window: the whole aperture, strips included ...
    CHECK(metrics_sum == 64 * 38);     // ... metrics: the image rectangle
    CHECK(strip_b == 0.0);             // ... and the strips are dark targets, not "don't care"
    CHECK(in.mask(i0, j0 - 1) == 1.0);
    CHECK(in.metrics_mask(i0, j0 - 1) == 0.0);
    // a square target: both masks are the aperture, letterbox or not
    auto sq = doe::prepare(doe::targets::disk(50, 0.3), cfg);
    double a = 0, b = 0;
    for (std::size_t k = 0; k < sq.mask.data.size(); ++k) { a += sq.mask.data[k]; b += sq.metrics_mask.data[k]; }
    CHECK(a == 64 * 64);
    CHECK(b == 64 * 64);
    // design() reports the metrics on the rectangle: efficiency = light in the rectangle over total
    cfg.iters = 3;
    cfg.run_gs = false;
    auto r = doe::design(target, cfg);
    double total = 0, rect = 0;
    for (std::size_t k = 0; k < r.runs.back().field.data.size(); ++k) {
        const double I = std::norm(r.runs.back().field.data[k]);
        total += I;
        rect += in.metrics_mask.data[k] * I;
    }
    CHECK_THAT(r.runs.back().metrics.efficiency, WithinAbs(rect / total, 1e-12));
}

TEST_CASE("prepare: the padded window is set by the picture-clear rule, 480 px for 256 px at 50 mm, and the sampling check passes", "[pipeline]") {
    auto cfg = base_config();
    auto in = doe::prepare(doe::targets::cross_plus_ring(64), cfg);
    CHECK(in.grid.n == 480);
    CHECK(doe::sampling_report(in.grid, cfg.distance, cfg.active * cfg.pitch).wrap_misses_picture);
    cfg.active = 512;
    CHECK(doe::prepare(doe::targets::cross_plus_ring(64), cfg).grid.n == 720);
}

TEST_CASE("sampling_for: the pre-flight report the CLI prints equals the one the pipeline stores, near and far", "[pipeline][sampling]") {
    for (double d : {0.05, 0.2}) {
        auto cfg = base_config();
        cfg.active = 32;
        cfg.iters = 1;
        cfg.run_gs = false;
        cfg.distance = d;
        const auto pre = doe::sampling_for(cfg);
        auto r = doe::design(doe::targets::disk(32, 0.3), cfg);
        CHECK(pre.window_extent_m == r.sampling.window_extent_m);
        CHECK(pre.wrap_misses_picture == r.sampling.wrap_misses_picture);
        CHECK(pre.spot_size_px == r.sampling.spot_size_px);
        CHECK(pre.signal_spread_m == r.sampling.signal_spread_m);
        CHECK(pre.window_extent_m == r.inputs.grid.extent());
    }
}

TEST_CASE("design: the Gerchberg-Saxton history is logged with the configured efficiency weight", "[pipeline][gs]") {
    auto cfg = base_config();
    cfg.active = 32;
    cfg.iters = 1;
    cfg.gs = doe::GsParams{6, 3};
    auto r3 = doe::design(doe::targets::disk(32, 0.3), cfg);
    cfg.mu = 0.1;
    auto r1 = doe::design(doe::targets::disk(32, 0.3), cfg);
    REQUIRE(r3.runs[0].name == "gs");
    REQUIRE(r3.runs[0].efficiency_history.size() == 6);
    for (std::size_t k = 0; k < 6; ++k) {
        CHECK_THAT(r3.runs[0].efficiency_history[k], WithinRel(3.0 * r1.runs[0].efficiency_history[k], 1e-9));
        CHECK(r3.runs[0].shape_history[k] == r1.runs[0].shape_history[k]);
    }
}

TEST_CASE("sampling report and warnings are part of the design result", "[pipeline]") {
    auto cfg = base_config();
    cfg.iters = 2;
    cfg.run_gs = false;
    auto r = doe::design(doe::targets::cross_plus_ring(64), cfg);
    CHECK(r.sampling.spot_size_px > 0);
    CHECK(r.sampling.wrap_misses_picture);
    CHECK(r.runs.size() == 1);
    CHECK(r.runs[0].name == "adam");
    const std::string json = r.report().dump();
    CHECK(json.find("\"sampling\"") != std::string::npos);
    CHECK(json.find("\"efficiency\"") != std::string::npos);
    CHECK(json.find("\"amplitude_rmse\"") != std::string::npos);
    CHECK(json.find("\"config\"") != std::string::npos);
}

TEST_CASE("T8 end-to-end on the binary test image, q = 0, 400 iterations: NCC > 0.9, efficiency > 0.6", "[pipeline][T8][slow]") {
    auto cfg = base_config();
    cfg.run_gs = true;
    auto r = doe::design(doe::targets::cross_plus_ring(256), cfg);
    REQUIRE(r.runs.size() == 2);
    const auto& gs = r.runs[0];
    const auto& adam = r.runs[1];
    CHECK(gs.name == "gs");
    CHECK(adam.name == "adam");
    CHECK(adam.metrics.ncc > 0.9);
    CHECK(adam.metrics.efficiency > 0.6);
    CHECK(adam.history.size() == 400);
    CHECK(gs.history.size() == 40);
    CHECK(adam.seconds > 0.0);
    // the final phase is on the torus and the field is the propagation of it
    for (double x : adam.phi.data) {
        CHECK(x <= pi);
        CHECK(x > -pi);
    }
}

TEST_CASE("T13 degeneracy guard: efficiency after 300 iterations does not drop below the random start and exceeds 0.5", "[pipeline][T13][slow]") {
    auto cfg = base_config();
    cfg.iters = 300;
    cfg.init = doe::InitMethod::random;
    cfg.run_gs = false;
    auto r = doe::design(doe::targets::cross_plus_ring(256), cfg);
    REQUIRE(r.runs.size() == 1);
    CHECK(r.runs[0].metrics.efficiency > 0.5);
    CHECK(r.runs[0].metrics.efficiency >= r.initial_metrics.efficiency);
}

TEST_CASE("T9 q = 4 with continuation: all phases on a level +- 1e-9, PSNR loss vs continuous < 3 dB", "[pipeline][T9][slow]") {
    // Both documented methods: Wyrowski's stepwise projection (default, 1.3 dB
    // measured) and Choi's Gumbel-Softmax surrogate with the hyper-parameters
    // of their Supplement S2.3 (2.0 dB measured).
    auto cfg = base_config();
    cfg.levels = 4;
    cfg.run_gs = false;
    for (auto method : {doe::QuantMethod::wyrowski, doe::QuantMethod::choi}) {
        cfg.quant_method = method;
        auto r = doe::design(doe::targets::cross_plus_ring(256), cfg);
        REQUIRE(r.runs.size() == 2);  // adam (continuous), adam_q4
        const auto& cont = r.runs[0];
        const auto& quant = r.runs[1];
        CHECK(quant.name == "adam_q4");
        for (double x : quant.phi.data) {
            double best = 10;
            for (int z = 0; z < 4; ++z) best = std::min(best, std::abs(std::remainder(x - (-pi + z * pi / 2), 2 * pi)));
            CHECK(best < 1e-9);
        }
        const double loss = cont.metrics.psnr_db - quant.metrics.psnr_db;
        INFO("method " << (method == doe::QuantMethod::wyrowski ? "wyrowski" : "choi") << ": PSNR continuous " << cont.metrics.psnr_db << " dB, quantized " << quant.metrics.psnr_db << " dB");
        CHECK(loss < 3.0);
    }
}

TEST_CASE("T14 single vs double precision, same seed, 300 iterations: NCC, efficiency, RMSE equal to 4 decimals", "[pipeline][T14][slow]") {
    auto cfg = base_config();
    cfg.iters = 300;
    cfg.run_gs = false;
    auto rd = doe::design(doe::targets::cross_plus_ring(256), cfg);
    cfg.single_precision = true;
    auto rf = doe::design(doe::targets::cross_plus_ring(256), cfg);
    const auto& md = rd.runs[0].metrics;
    const auto& mf = rf.runs[0].metrics;
    INFO("double: " << md.ncc << " " << md.efficiency << " " << md.amplitude_rmse << "   float: " << mf.ncc << " " << mf.efficiency << " " << mf.amplitude_rmse);
    CHECK_THAT(mf.ncc, WithinAbs(md.ncc, 5e-5));
    CHECK_THAT(mf.efficiency, WithinAbs(md.efficiency, 5e-5));
    CHECK_THAT(mf.amplitude_rmse, WithinAbs(md.amplitude_rmse, 5e-5));
}
