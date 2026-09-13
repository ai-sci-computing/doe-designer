/// @file pipeline.hpp
/// @brief The design pipeline: target image -> DOE phase, metrics and report.
#pragma once

#include "doe/array.hpp"
#include "doe/grid.hpp"
#include "doe/io.hpp"
#include "doe/metrics.hpp"
#include "doe/propagate.hpp"
#include "doe/solvers.hpp"

#include <functional>
#include <string>
#include <vector>

namespace doe {

/// Illumination amplitude on the active aperture.
enum class IllumShape { square, disk, gaussian };
/// Starting phase.
enum class InitMethod { tie, random, backprop };

/// Everything the command line exposes (defaults as documented in the manual).
struct DesignConfig {
    double wavelength = 532e-9;  ///< m
    double pitch = 8e-6;         ///< m
    double distance = 0.05;      ///< m, DOE plane to target plane
    int active = 512;            ///< active aperture in pixels; the target's longer side is resampled to this
    double pad = 1.0;            ///< minimum zero-padding factor; the window is set by the picture-clear rule (Grid::padded_for), this only enlarges it
    bool band_limit = true;      ///< Matsushima–Shimobaba band limit
    bool letterbox = true;       ///< non-square targets: keep the square window and force the strips beside the image dark; false = rectangular window, strips are "don't care"
    bool single_precision = false;  ///< run in float (transfer function still built in double)
    IllumShape illum = IllumShape::square;  ///< illumination amplitude shape
    double gaussian_waist = 0.5;  ///< Gaussian illumination 1/e amplitude radius as a fraction of the half aperture
    InitMethod init = InitMethod::tie;  ///< starting phase
    unsigned long long seed = 0;  ///< random init / Gumbel noise seed
    int iters = 400;              ///< Adam budget
    double lr = 0.05;             ///< Adam step
    double mu = 0.3;              ///< efficiency weight
    int levels = 0;               ///< quantization levels (0: continuous only)
    QuantMethod quant_method = QuantMethod::wyrowski;  ///< quantization method
    double quant_start = 0.4;     ///< fraction of the budget before quantization starts
    QuantRamp quant_ramp = QuantRamp::linear;  ///< capture ramp of the stepwise quantizer
    bool run_gs = true;           ///< also run the Gerchberg–Saxton baseline
    GsParams gs{40, 20};          ///< GS schedule
    double soft_edge_sigma = 0.0; ///< optional Gaussian blur of the target (pixels of the active grid)
    QuantParams choi{};           ///< Choi hyper-parameters (only the choi_* fields are used)
};

/// Arrays on the padded grid, all in double.
struct DesignInputs {
    Grid grid;                ///< padded grid
    Array2<double> illum;     ///< illumination amplitude (zero in the padding)
    Array2<double> b;         ///< target amplitude sqrt(I), zero outside the window
    Array2<double> mask;      ///< signal window of the energy: the active square (letterbox, dark strips) or the image rectangle (`letterbox == false`)
    Array2<double> metrics_mask;  ///< the image rectangle: the metrics are always evaluated on it, so that both modes are comparable
    Array2<double> i_target;  ///< target intensity (for TIE)
};

/// The grid a design with this configuration runs on (Grid::padded_for with the
/// configured distance and minimum padding); prepare(), design() and the
/// command line's pre-flight report all use it, so they cannot disagree.
Grid grid_for(const DesignConfig& cfg);

/// The sampling report for a configuration, on grid_for(cfg): what the command
/// line prints before a run and what design() stores in DesignResult::sampling.
SamplingReport sampling_for(const DesignConfig& cfg);

/// Resample the target so that its longer side is `active` pixels (aspect
/// ratio kept), embed it, build the square illumination, the energy window
/// (square with dark strips by default; the image rectangle with
/// `letterbox == false`) and the metrics mask (always the rectangle).
DesignInputs prepare(const Array2<double>& target, const DesignConfig& cfg);

/// One solver run.
struct SolverRun {
    std::string name;             ///< "gs", "adam", "adam_qN"
    Array2<double> phi;           ///< final DOE phase
    Field<double> field;          ///< target-plane field
    Metrics metrics;              ///< figures of merit of the final field
    std::vector<double> history;  ///< energy per iteration
    std::vector<double> shape_history;       ///< shape term per iteration
    std::vector<double> efficiency_history;  ///< efficiency term per iteration
    double seconds = 0.0;         ///< wall-clock time of the run
};

/// Result of design().
struct DesignResult {
    DesignConfig config;          ///< the configuration used
    DesignInputs inputs;          ///< prepared arrays
    SamplingReport sampling;      ///< sampling diagnostics
    Array2<double> phi0;          ///< starting phase
    Metrics initial_metrics;      ///< metrics of the starting phase (random or TIE)
    std::vector<SolverRun> runs;  ///< in order: gs (optional), adam, adam_qN (optional)
    /// JSON report: config, sampling report with warnings, metrics per run, timings.
    Json report() const;
    /// Markdown table comparing the runs (efficiency and RMSE in separate columns).
    std::string markdown_table() const;
};

/// Progress callback: (run name, iteration, energy) -> continue?
using DesignProgress = std::function<bool(const std::string&, int, double)>;

/// Run the pipeline: prepare, initialize, GS baseline, Adam, Adam with
/// quantization; float or double according to the config.
DesignResult design(const Array2<double>& target, const DesignConfig& cfg, const DesignProgress& progress = {});

}  // namespace doe
