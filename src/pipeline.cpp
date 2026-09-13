#include "doe/pipeline.hpp"

#include "doe/energy.hpp"
#include "doe/image.hpp"
#include "doe/init.hpp"
#include "doe/targets.hpp"

#include <chrono>
#include <cmath>
#include <format>
#include <stdexcept>

namespace doe {

Grid grid_for(const DesignConfig& cfg) {
    return Grid::padded_for(cfg.active, cfg.pitch, cfg.wavelength, cfg.distance, cfg.pad);
}

SamplingReport sampling_for(const DesignConfig& cfg) {
    return sampling_report(grid_for(cfg), cfg.distance, cfg.active * cfg.pitch);
}

DesignInputs prepare(const Array2<double>& target, const DesignConfig& cfg) {
    if (cfg.active < 2) throw std::invalid_argument("prepare: active aperture must be >= 2 px");
    DesignInputs in;
    in.grid = grid_for(cfg);
    const std::size_t n = static_cast<std::size_t>(in.grid.n), a = static_cast<std::size_t>(cfg.active);
    // Keep the target's aspect ratio: the longer side fills the aperture, the
    // window is the resampled rectangle. The DOE stays square.
    const double scale = double(a) / double(std::max(target.rows, target.cols));
    const std::size_t ar = target.rows >= target.cols ? a : std::max<std::size_t>(1, static_cast<std::size_t>(std::lround(target.rows * scale)));
    const std::size_t ac = target.cols >= target.rows ? a : std::max<std::size_t>(1, static_cast<std::size_t>(std::lround(target.cols * scale)));
    Array2<double> t = resample(target, ar, ac);
    if (cfg.soft_edge_sigma > 0.0) t = targets::soft_edges(t, cfg.soft_edge_sigma);
    double tmax = 0.0;
    for (double v : t.data) tmax = std::max(tmax, v);
    if (tmax > 0.0)
        for (double& v : t.data) v = std::max(v, 0.0) / tmax;
    in.i_target = embed(t, n);
    in.b = in.i_target;
    for (double& v : in.b.data) v = std::sqrt(v);

    // Metrics always on the image rectangle; the energy window is the whole aperture in
    // letterbox mode (the strips are dark targets) or the rectangle otherwise.
    in.metrics_mask = embed(Array2<double>(ar, ac, 1.0), n);
    in.mask = cfg.letterbox ? embed(Array2<double>(a, a, 1.0), n) : in.metrics_mask;
    Array2<double> il(a, a, 0.0);
    const double c = (double(a) - 1.0) / 2.0, half = double(a) / 2.0;
    for (std::size_t i = 0; i < a; ++i)
        for (std::size_t j = 0; j < a; ++j) {
            const double x = double(i) - c, y = double(j) - c, r = std::hypot(x, y);
            switch (cfg.illum) {
                case IllumShape::square: il(i, j) = 1.0; break;
                case IllumShape::disk: il(i, j) = r <= half ? 1.0 : 0.0; break;
                case IllumShape::gaussian: {
                    const double w = cfg.gaussian_waist * half;
                    il(i, j) = r <= half ? std::exp(-(r * r) / (w * w)) : 0.0;
                    break;
                }
            }
        }
    in.illum = embed(il, n);
    return in;
}

namespace {

template <class T>
Array2<T> cast(const Array2<double>& a) {
    Array2<T> out(a.rows, a.cols);
    for (std::size_t k = 0; k < a.data.size(); ++k) out.data[k] = T(a.data[k]);
    return out;
}
Array2<double> to_double(const Array2<float>& a) {
    Array2<double> out(a.rows, a.cols);
    for (std::size_t k = 0; k < a.data.size(); ++k) out.data[k] = a.data[k];
    return out;
}
Array2<double> to_double(const Array2<double>& a) { return a; }
Field<double> to_double(const Field<float>& a) {
    Field<double> out(a.rows, a.cols);
    for (std::size_t k = 0; k < a.data.size(); ++k) out.data[k] = std::complex<double>(a.data[k]);
    return out;
}
Field<double> to_double(const Field<double>& a) { return a; }

template <class T>
void run_solvers(const DesignConfig& cfg, DesignResult& r, const DesignProgress& progress) {
    const DesignInputs& in = r.inputs;
    AngularSpectrum<T> prop(in.grid, cfg.distance, cfg.band_limit);
    const Array2<T> illum = cast<T>(in.illum), b = cast<T>(in.b), mask = cast<T>(in.mask), mmask = cast<T>(in.metrics_mask);
    const Array2<T> phi0 = cast<T>(r.phi0);
    const EnergyWeights weights{1.0, cfg.mu};

    r.initial_metrics = metrics(energy_and_grad<T>(phi0, illum, prop, b, mask, weights).field, b, mmask);

    auto timed = [&](const std::string& name, auto&& fn) {
        const auto t0 = std::chrono::steady_clock::now();
        Result<T> res = fn();
        SolverRun run;
        run.name = name;
        run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        run.phi = to_double(res.phi);
        run.field = to_double(res.field);
        run.history = std::move(res.history);
        run.shape_history = std::move(res.shape_history);
        run.efficiency_history = std::move(res.efficiency_history);
        run.metrics = metrics(res.field, b, mmask);
        r.runs.push_back(std::move(run));
    };
    auto cb = [&](const std::string& name) -> ProgressFn {
        if (!progress) return {};
        return [&, name](int it, double e) { return progress(name, it, e); };
    };

    if (cfg.run_gs) {
        GsParams gp = cfg.gs;
        gp.weights = weights;  // the logged history uses the same mu as the Adam runs
        timed("gs", [&] { return gerchberg_saxton<T>(phi0, illum, prop, b, mask, gp); });
    }

    OptimizeParams op;
    op.iters = cfg.iters;
    op.lr = cfg.lr;
    op.weights = weights;
    timed("adam", [&] { return optimize<T>(phi0, illum, prop, b, mask, op, cb("adam")); });

    if (cfg.levels > 0) {
        OptimizeParams oq = op;
        oq.quant.levels = cfg.levels;
        oq.quant.method = cfg.quant_method;
        oq.quant.start = cfg.quant_start;
        oq.quant.ramp = cfg.quant_ramp;
        oq.quant.choi_w = cfg.choi.choi_w;
        oq.quant.choi_gain0 = cfg.choi.choi_gain0;
        oq.quant.choi_gain1 = cfg.choi.choi_gain1;
        oq.quant.choi_tau0 = cfg.choi.choi_tau0;
        oq.quant.choi_c = cfg.choi.choi_c;
        oq.quant.choi_seed = cfg.seed;
        const std::string name = std::format("adam_q{}", cfg.levels);
        timed(name, [&] { return optimize<T>(phi0, illum, prop, b, mask, oq, cb(name)); });
    }
}

}  // namespace

DesignResult design(const Array2<double>& target, const DesignConfig& cfg, const DesignProgress& progress) {
    DesignResult r;
    r.config = cfg;
    r.inputs = prepare(target, cfg);
    r.sampling = sampling_for(cfg);
    if (cfg.init == InitMethod::tie) {
        Array2<double> i_source = r.inputs.illum;
        for (double& v : i_source.data) v *= v;
        r.phi0 = init_tie<double>(r.inputs.grid, cfg.distance, r.inputs.i_target, i_source);
    } else if (cfg.init == InitMethod::backprop) {
        AngularSpectrum<double> prop(r.inputs.grid, cfg.distance, cfg.band_limit);
        r.phi0 = init_backprop<double>(r.inputs.grid, prop, r.inputs.b);
    } else {
        r.phi0 = init_random<double>(r.inputs.grid, cfg.seed);
    }
    if (cfg.single_precision) run_solvers<float>(cfg, r, progress);
    else run_solvers<double>(cfg, r, progress);
    return r;
}

Json DesignResult::report() const {
    Json j;
    Json c;
    c.set("wavelength_m", config.wavelength);
    c.set("pitch_m", config.pitch);
    c.set("distance_m", config.distance);
    c.set("active_px", config.active);
    c.set("padded_px", inputs.grid.n);
    c.set("band_limit", config.band_limit);
    c.set("letterbox", config.letterbox);
    c.set("single_precision", config.single_precision);
    c.set("illumination", config.illum == IllumShape::square ? "square" : config.illum == IllumShape::disk ? "disk" : "gaussian");
    c.set("init", config.init == InitMethod::tie ? "tie" : config.init == InitMethod::random ? "random" : "backprop");
    c.set("seed", static_cast<int>(config.seed));
    c.set("iters", config.iters);
    c.set("lr", config.lr);
    c.set("mu", config.mu);
    c.set("levels", config.levels);
    c.set("quant_method", config.quant_method == QuantMethod::wyrowski ? "wyrowski" : "choi");
    c.set("quant_start", config.quant_start);
    c.set("quant_ramp", config.quant_ramp == QuantRamp::linear ? "linear" : "table");
    c.set("gs_iters", config.gs.iters);
    c.set("gs_phase_only_iters", config.gs.phase_only_iters);
    c.set("soft_edge_sigma_px", config.soft_edge_sigma);
    j.set("config", c);

    Json s;
    s.set("max_angle_deg", sampling.max_angle_rad * 180.0 / 3.14159265358979323846);
    s.set("spot_size_m", sampling.spot_size_m);
    s.set("spot_size_px", sampling.spot_size_px);
    s.set("signal_spread_m", sampling.signal_spread_m);
    s.set("window_extent_m", sampling.window_extent_m);
    s.set("fresnel_number", sampling.fresnel_number);
    s.set("wrap_misses_picture", sampling.wrap_misses_picture);
    s.set("spot_resolved", sampling.spot_resolved);
    std::vector<std::string> warnings;
    if (!sampling.wrap_misses_picture) warnings.push_back("light wrapping around the periodic window reaches the picture: enlarge the window, reduce the distance or the pitch");
    if (!sampling.spot_resolved) warnings.push_back("diffraction-limited spot is larger than one target pixel: finer detail is unreachable");
    s.set("warnings", warnings);
    j.set("sampling", s);

    const double pitch_m = config.pitch;
    auto metrics_json = [pitch_m](const Metrics& m) {
        Json mj;
        mj.set("efficiency", m.efficiency);
        mj.set("amplitude_rmse", m.amplitude_rmse);
        mj.set("speckle_contrast", m.speckle_contrast);
        mj.set("ncc", m.ncc);
        mj.set("psnr_db", m.psnr_db);
        mj.set("vortex_count", static_cast<int>(m.vortex_count));
        mj.set("vortex_count_bright", static_cast<int>(m.vortex_count_bright));
        mj.set("vortex_density_bright_per_mm2", m.vortex_density_bright / (pitch_m * pitch_m) * 1e-6);
        return mj;
    };
    j.set("initial", metrics_json(initial_metrics));
    Json rj;
    for (const auto& run : runs) {
        Json one;
        one.set("metrics", metrics_json(run.metrics));
        one.set("iterations", static_cast<int>(run.history.size()));
        one.set("seconds", run.seconds);
        one.set("final_energy", run.history.empty() ? 0.0 : run.history.back());
        rj.set(run.name, one);
    }
    j.set("runs", rj);
    return j;
}

std::string DesignResult::markdown_table() const {
    const double per_mm2 = 1e-6 / (config.pitch * config.pitch);
    std::string s = "| run | efficiency | amplitude RMSE | speckle contrast | NCC | PSNR (dB) | vortices in bright /mm² | iterations | time (s) |\n|---|---|---|---|---|---|---|---|---|\n";
    s += std::format("| initial ({}) | {:.4f} | {:.4f} | {:.3f} | {:.4f} | {:.2f} | {:.0f} | - | - |\n",
                     config.init == InitMethod::tie ? "TIE" : config.init == InitMethod::random ? "random" : "backprop", initial_metrics.efficiency, initial_metrics.amplitude_rmse,
                     initial_metrics.speckle_contrast, initial_metrics.ncc, initial_metrics.psnr_db, initial_metrics.vortex_density_bright * per_mm2);
    for (const auto& r : runs)
        s += std::format("| {} | {:.4f} | {:.4f} | {:.3f} | {:.4f} | {:.2f} | {:.0f} | {} | {:.1f} |\n", r.name, r.metrics.efficiency,
                         r.metrics.amplitude_rmse, r.metrics.speckle_contrast, r.metrics.ncc, r.metrics.psnr_db,
                         r.metrics.vortex_density_bright * per_mm2, r.history.size(), r.seconds);
    return s;
}

}  // namespace doe
