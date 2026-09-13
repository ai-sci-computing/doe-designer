// Quantization inside the solvers (T9 at unit-test scale; the full T9 with the
// pipeline budget lives in test_pipeline).
//
// Wyrowski 1990 §3.C: the stepwise quantizer Q_Z^(p) replaces the unit-modulus
// operator in the DOE plane; "in each step, except the last one, Q iteration
// cycles are performed. For p = P the identity Q_Z = Q_Z^(P) is obtained; i.e.,
// in the last step the direct quantization operator is reached, and only one
// iteration cycle is performed", J = Q (P - 1) + 1 cycles in total.
// Choi et al. 2022 eq. (5): forward pass with the hard quantizer, backward
// pass with the Gumbel-Softmax surrogate gradient, temperature annealed.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinRel;

#include "doe/grid.hpp"
#include "doe/init.hpp"
#include "doe/metrics.hpp"
#include "doe/phase.hpp"
#include "doe/propagate.hpp"
#include "doe/quantize.hpp"
#include "doe/solvers.hpp"

#include <cmath>
#include <numbers>

constexpr double pi = std::numbers::pi;

namespace {

struct Problem {
    doe::Grid grid = doe::Grid::padded(48, 8e-6, 532e-9);
    doe::AngularSpectrum<double> prop{grid, 0.02, true};
    doe::Array2<double> illum, b, mask;
    Problem() {
        const std::size_t n = static_cast<std::size_t>(grid.n), lo = (n - 48) / 2;
        illum = b = mask = doe::Array2<double>(n, n, 0.0);
        for (std::size_t i = lo; i < lo + 48; ++i)
            for (std::size_t j = lo; j < lo + 48; ++j) {
                illum(i, j) = mask(i, j) = 1.0;
                // a cross: two bars, a binary target with structure
                const bool bar = std::abs(grid.x(i)) < 4 * grid.pitch || std::abs(grid.x(j)) < 4 * grid.pitch;
                b(i, j) = (bar && std::abs(grid.x(i)) < 16 * grid.pitch && std::abs(grid.x(j)) < 16 * grid.pitch) ? 1.0 : 0.0;
            }
    }
};

bool all_on_levels(const doe::Array2<double>& phi, int q, double tol = 1e-9) {
    for (double x : phi.data) {
        double best = 10;
        for (int z = 0; z < q; ++z) best = std::min(best, std::abs(doe::wrap_to_pi(x - (-pi + z * 2 * pi / q))));
        if (best > tol) return false;
    }
    return true;
}

}  // namespace

TEST_CASE("optimize + Wyrowski stepwise quantization: on levels, T9 loss < 3 dB vs continuous", "[solvers][quantize][T9]") {
    Problem P;
    auto phi0 = doe::init_random<double>(P.grid, 3);
    doe::OptimizeParams cont;
    cont.iters = 200;
    auto rc = doe::optimize<double>(phi0, P.illum, P.prop, P.b, P.mask, cont);
    auto mc = doe::metrics(rc.field, P.b, P.mask);

    doe::OptimizeParams quant = cont;
    quant.quant.levels = 4;
    quant.quant.method = doe::QuantMethod::wyrowski;
    quant.quant.start = 0.4;
    auto rq = doe::optimize<double>(phi0, P.illum, P.prop, P.b, P.mask, quant);
    REQUIRE(rq.history.size() == 200);
    CHECK(all_on_levels(rq.phi, 4));
    auto mq = doe::metrics(rq.field, P.b, P.mask);
    CHECK(mc.psnr_db - mq.psnr_db < 3.0);
    CHECK(mq.ncc > 0.7);
    for (double e : rq.history) CHECK(std::isfinite(e));
}

TEST_CASE("optimize + linear capture ramp (Skeren et al. 2002): on levels, T9 loss < 3 dB, a different schedule than the table", "[solvers][quantize][T9]") {
    Problem P;
    auto phi0 = doe::init_random<double>(P.grid, 3);
    doe::OptimizeParams cont;
    cont.iters = 200;
    auto rc = doe::optimize<double>(phi0, P.illum, P.prop, P.b, P.mask, cont);
    auto mc = doe::metrics(rc.field, P.b, P.mask);

    doe::OptimizeParams lin = cont;
    lin.quant.levels = 4;
    lin.quant.ramp = doe::QuantRamp::linear;
    auto rl = doe::optimize<double>(phi0, P.illum, P.prop, P.b, P.mask, lin);
    CHECK(all_on_levels(rl.phi, 4));
    auto ml = doe::metrics(rl.field, P.b, P.mask);
    CHECK(mc.psnr_db - ml.psnr_db < 3.0);
    CHECK(ml.ncc > 0.7);

    doe::OptimizeParams tab = lin;
    tab.quant.ramp = doe::QuantRamp::table;
    auto rt = doe::optimize<double>(phi0, P.illum, P.prop, P.b, P.mask, tab);
    CHECK(all_on_levels(rt.phi, 4));
    double diff = 0;
    for (std::size_t k = 0; k < rl.phi.data.size(); ++k) diff = std::max(diff, std::abs(doe::wrap_to_pi(rl.phi.data[k] - rt.phi.data[k])));
    CHECK(diff > 1e-6);  // the two ramps are different schedules
    // the default ramp is the linear one
    doe::OptimizeParams def = cont;
    def.quant.levels = 4;
    auto rd = doe::optimize<double>(phi0, P.illum, P.prop, P.b, P.mask, def);
    for (std::size_t k = 0; k < rl.phi.data.size(); ++k) CHECK(rd.phi.data[k] == rl.phi.data[k]);
}

TEST_CASE("optimize + Choi surrogate gradient: on levels, T9 loss < 3 dB vs continuous", "[solvers][quantize][T9][choi]") {
    Problem P;
    auto phi0 = doe::init_random<double>(P.grid, 4);
    doe::OptimizeParams cont;
    cont.iters = 200;
    auto rc = doe::optimize<double>(phi0, P.illum, P.prop, P.b, P.mask, cont);
    auto mc = doe::metrics(rc.field, P.b, P.mask);

    doe::OptimizeParams quant = cont;
    quant.quant.levels = 4;
    quant.quant.method = doe::QuantMethod::choi;
    quant.quant.start = 0.4;
    auto rq = doe::optimize<double>(phi0, P.illum, P.prop, P.b, P.mask, quant);
    REQUIRE(rq.history.size() == 200);
    CHECK(all_on_levels(rq.phi, 4));
    auto mq = doe::metrics(rq.field, P.b, P.mask);
    CHECK(mc.psnr_db - mq.psnr_db < 3.0);
    CHECK(mq.ncc > 0.7);
    for (double e : rq.history) CHECK(std::isfinite(e));
    // the term histories are logged through the surrogate stage as well
    REQUIRE(rq.shape_history.size() == 200);
    REQUIRE(rq.efficiency_history.size() == 200);
    for (std::size_t t = 0; t < 200; ++t)
        CHECK(std::abs(rq.shape_history[t] + rq.efficiency_history[t] - rq.history[t]) < 1e-12);
}

TEST_CASE("optimize: before the quantization stage the iterates are continuous, at the end exactly quantized", "[solvers][quantize]") {
    Problem P;
    doe::OptimizeParams quant;
    quant.iters = 50;
    quant.quant.levels = 2;
    quant.quant.method = doe::QuantMethod::wyrowski;
    quant.quant.start = 0.5;
    bool continuous_seen = false;
    auto r = doe::optimize<double>(doe::init_random<double>(P.grid, 5), P.illum, P.prop, P.b, P.mask, quant,
                                   [&](int it, double) {
                                       if (it == 10) continuous_seen = true;
                                       return true;
                                   });
    CHECK(continuous_seen);
    CHECK(all_on_levels(r.phi, 2));
    // binary: exactly two distinct values
    double lo = 10, hi = -10;
    for (double x : r.phi.data) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
    }
    CHECK(std::abs(hi - lo - pi) < 1e-9);
}

TEST_CASE("Gerchberg-Saxton + Wyrowski Q_Z^(p): J = Q(P-1)+1 quantization cycles, on levels, T9", "[solvers][quantize][gs][T9]") {
    Problem P;
    auto phi0 = doe::init_random<double>(P.grid, 6);
    doe::GsParams analog{40, 20};
    auto rc = doe::gerchberg_saxton<double>(phi0, P.illum, P.prop, P.b, P.mask, analog);
    auto mc = doe::metrics(rc.field, P.b, P.mask);

    doe::GsParams quant = analog;
    quant.quant.levels = 4;
    quant.quant.steps = 10;        // P
    quant.quant.cycles_per_step = 5;  // Q
    auto rq = doe::gerchberg_saxton<double>(phi0, P.illum, P.prop, P.b, P.mask, quant);
    CHECK(rq.history.size() == 40 + 5 * 9 + 1);
    CHECK(all_on_levels(rq.phi, 4));
    auto mq = doe::metrics(rq.field, P.b, P.mask);
    CHECK(mc.psnr_db - mq.psnr_db < 3.0);
    CHECK(mq.ncc > 0.6);
}

TEST_CASE("Gerchberg-Saxton + linear capture ramp: the same J = Q(P-1)+1 cycles, on levels, T9", "[solvers][quantize][gs][T9]") {
    Problem P;
    auto phi0 = doe::init_random<double>(P.grid, 6);
    doe::GsParams analog{40, 20};
    auto rc = doe::gerchberg_saxton<double>(phi0, P.illum, P.prop, P.b, P.mask, analog);
    auto mc = doe::metrics(rc.field, P.b, P.mask);
    doe::GsParams quant = analog;
    quant.quant.levels = 4;
    quant.quant.ramp = doe::QuantRamp::linear;
    auto rq = doe::gerchberg_saxton<double>(phi0, P.illum, P.prop, P.b, P.mask, quant);
    CHECK(rq.history.size() == 40 + 5 * 9 + 1);
    CHECK(all_on_levels(rq.phi, 4));
    auto mq = doe::metrics(rq.field, P.b, P.mask);
    CHECK(mc.psnr_db - mq.psnr_db < 3.0);
    CHECK(mq.ncc > 0.6);
}

TEST_CASE("gerchberg_saxton: the logged energy terms use GsParams::weights, the iterates do not depend on them", "[solvers][gs]") {
    Problem P;
    auto phi0 = doe::init_random<double>(P.grid, 8);
    doe::GsParams a{10, 5};
    a.weights = doe::EnergyWeights{1.0, 0.3};
    doe::GsParams b = a;
    b.weights.efficiency = 0.1;
    auto ra = doe::gerchberg_saxton<double>(phi0, P.illum, P.prop, P.b, P.mask, a);
    auto rb = doe::gerchberg_saxton<double>(phi0, P.illum, P.prop, P.b, P.mask, b);
    CHECK(ra.phi.data == rb.phi.data);
    REQUIRE(ra.efficiency_history.size() == 10);
    for (std::size_t k = 0; k < 10; ++k) {
        CHECK_THAT(ra.efficiency_history[k], WithinRel(3.0 * rb.efficiency_history[k], 1e-9));
        CHECK(ra.shape_history[k] == rb.shape_history[k]);
    }
}

TEST_CASE("quantized solvers in single precision stay on levels", "[solvers][quantize][precision]") {
    Problem P;
    const std::size_t n = static_cast<std::size_t>(P.grid.n);
    doe::AngularSpectrum<float> propf(P.grid, 0.02, true);
    doe::Array2<float> illum(n, n), b(n, n), mask(n, n);
    for (std::size_t k = 0; k < n * n; ++k) {
        illum.data[k] = float(P.illum.data[k]);
        b.data[k] = float(P.b.data[k]);
        mask.data[k] = float(P.mask.data[k]);
    }
    doe::OptimizeParams quant;
    quant.iters = 30;
    quant.quant.levels = 4;
    auto r = doe::optimize<float>(doe::init_random<float>(P.grid, 7), illum, propf, b, mask, quant);
    doe::Array2<double> phi(n, n);
    for (std::size_t k = 0; k < n * n; ++k) phi.data[k] = r.phi.data[k];
    CHECK(all_on_levels(phi, 4, 1e-6));
}
