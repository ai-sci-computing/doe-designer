#include "doe/cli.hpp"

#include "doe/image.hpp"
#include "doe/io.hpp"
#include "doe/render.hpp"
#include "doe/targets.hpp"

#include <charconv>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string_view>

namespace doe {
namespace {

double to_double(std::string_view opt, std::string_view v) {
    try {
        std::size_t pos = 0;
        const double d = std::stod(std::string(v), &pos);
        if (pos != v.size()) throw std::invalid_argument("");
        return d;
    } catch (...) {
        throw std::invalid_argument(std::format("option {} expects a number, got '{}'", opt, v));
    }
}
int to_int(std::string_view opt, std::string_view v) {
    int i = 0;
    const auto res = std::from_chars(v.data(), v.data() + v.size(), i);
    if (res.ec != std::errc{} || res.ptr != v.data() + v.size())
        throw std::invalid_argument(std::format("option {} expects an integer, got '{}'", opt, v));
    return i;
}

}  // namespace

std::string cli_usage() {
    return R"(doe_design: phase-only DOE from a target image

usage: doe_design --target IMAGE.png [options]
       doe_design --make-targets DIR [--active N]

  --target PATH        grayscale/RGB PNG target (longer side resampled to --active pixels, aspect ratio kept)
  --out DIR            output directory (default results)
  --distance M         DOE-to-target distance in meters (default 0.05)
  --wavelength M       vacuum wavelength (default 532e-9)
  --pitch M            DOE pixel pitch (default 8e-6)
  --active N           active aperture in pixels (default 512; the window is padded so that wrapped light misses the picture, 720 px at the defaults)
  --illum S            square | disk | gaussian illumination (default square)
  --init S             tie | random | backprop starting phase (default tie; backprop for photographs)
  --seed N             seed for random init / Gumbel noise (default 0)
  --iters N            Adam iterations (default 400)
  --lr X               Adam step size (default 0.05)
  --mu X               efficiency weight (default 0.3)
  --levels Q           phase levels for quantization, 0 = continuous (default 0)
  --quant S            wyrowski | choi quantization method (default wyrowski)
  --quant-start X      fraction of the budget before quantization starts (default 0.4)
  --quant-ramp S       linear | table capture ramp of the stepwise quantizer (default linear, Skeren 2002; table = Wyrowski's 10 held steps)
  --choi-gain A,B      Choi score scale ramp (default 300,1000, Supplement S2.3)
  --choi-tau X         Choi initial temperature (default 4, decays by e^-ln2)
  --no-gs              skip the Gerchberg-Saxton baseline
  --gs-cycles A,B      Gerchberg-Saxton cycles: A with the field zeroed outside the picture, then B with the amplitude free (default 20,20)
  --no-band-limit      disable the Matsushima-Shimobaba band limit
  --no-letterbox       non-square target: rectangular window, strips beside the image are "don't care" (default: strips forced dark)
  --single             single precision (float) solvers
  --soft-edge S        Gaussian blur of the target in pixels (default 0)
  --nz N               planes in the volume sweep (default 96)
  --view-size N        crop of the volume for the viewer (default min(active, 384))
  --threads N          FFTW threads (default 0 = all cores)
  --make-targets DIR   write the synthetic targets as PNGs and exit
  -h, --help           this text
)";
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions o;
    for (int a = 1; a < argc; ++a) {
        const std::string_view opt = argv[a];
        auto value = [&]() -> std::string_view {
            if (a + 1 >= argc) throw std::invalid_argument(std::format("option {} needs a value", opt));
            return argv[++a];
        };
        if (opt == "-h" || opt == "--help") o.help = true;
        else if (opt == "--target") o.target = value();
        else if (opt == "--make-targets") o.make_targets = value();
        else if (opt == "--out") o.out = value();
        else if (opt == "--distance") o.config.distance = to_double(opt, value());
        else if (opt == "--wavelength") o.config.wavelength = to_double(opt, value());
        else if (opt == "--pitch") o.config.pitch = to_double(opt, value());
        else if (opt == "--active") o.config.active = to_int(opt, value());
        else if (opt == "--illum") {
            const auto v = value();
            if (v == "square") o.config.illum = IllumShape::square;
            else if (v == "disk") o.config.illum = IllumShape::disk;
            else if (v == "gaussian") o.config.illum = IllumShape::gaussian;
            else throw std::invalid_argument(std::format("--illum: unknown shape '{}'", v));
        } else if (opt == "--init") {
            const auto v = value();
            if (v == "tie") o.config.init = InitMethod::tie;
            else if (v == "random") o.config.init = InitMethod::random;
            else if (v == "backprop") o.config.init = InitMethod::backprop;
            else throw std::invalid_argument(std::format("--init: unknown method '{}'", v));
        } else if (opt == "--seed") o.config.seed = static_cast<unsigned long long>(to_int(opt, value()));
        else if (opt == "--iters") o.config.iters = to_int(opt, value());
        else if (opt == "--lr") o.config.lr = to_double(opt, value());
        else if (opt == "--mu") o.config.mu = to_double(opt, value());
        else if (opt == "--levels") o.config.levels = to_int(opt, value());
        else if (opt == "--quant") {
            const auto v = value();
            if (v == "wyrowski") o.config.quant_method = QuantMethod::wyrowski;
            else if (v == "choi") o.config.quant_method = QuantMethod::choi;
            else throw std::invalid_argument(std::format("--quant: unknown method '{}'", v));
        } else if (opt == "--quant-start") o.config.quant_start = to_double(opt, value());
        else if (opt == "--quant-ramp") {
            const auto v = value();
            if (v == "linear") o.config.quant_ramp = QuantRamp::linear;
            else if (v == "table") o.config.quant_ramp = QuantRamp::table;
            else throw std::invalid_argument(std::format("--quant-ramp: unknown ramp '{}'", v));
        }
        else if (opt == "--choi-gain") {
            const std::string v(value());
            const auto comma = v.find(',');
            if (comma == std::string::npos) throw std::invalid_argument("--choi-gain expects A,B");
            o.config.choi.choi_gain0 = to_double(opt, v.substr(0, comma));
            o.config.choi.choi_gain1 = to_double(opt, v.substr(comma + 1));
        } else if (opt == "--choi-tau") o.config.choi.choi_tau0 = to_double(opt, value());
        else if (opt == "--no-gs") o.config.run_gs = false;
        else if (opt == "--gs-cycles") {
            const std::string v(value());
            const auto comma = v.find(',');
            if (comma == std::string::npos) throw std::invalid_argument("--gs-cycles expects A,B");
            const int zeroed = to_int(opt, v.substr(0, comma)), free_cycles = to_int(opt, v.substr(comma + 1));
            o.config.gs = doe::GsParams{zeroed + free_cycles, zeroed};
        }
        else if (opt == "--no-band-limit") o.config.band_limit = false;
        else if (opt == "--no-letterbox") o.config.letterbox = false;
        else if (opt == "--single") o.config.single_precision = true;
        else if (opt == "--soft-edge") o.config.soft_edge_sigma = to_double(opt, value());
        else if (opt == "--nz") o.nz = to_int(opt, value());
        else if (opt == "--view-size") o.view_size = to_int(opt, value());
        else if (opt == "--threads") o.threads = to_int(opt, value());
        else throw std::invalid_argument(std::format("unknown option '{}'", opt));
    }
    if (!o.help && o.target.empty() && o.make_targets.empty())
        throw std::invalid_argument("either --target IMAGE.png or --make-targets DIR is required (see --help)");
    return o;
}

namespace {
ImageU8 gray_png(const Array2<double>& a) {
    ImageU8 img;
    img.width = a.rows;
    img.height = a.cols;
    img.channels = 1;
    img.data.resize(a.rows * a.cols);
    for (std::size_t i = 0; i < a.rows; ++i)
        for (std::size_t j = 0; j < a.cols; ++j)
            img.data[j * a.rows + i] = static_cast<std::uint8_t>(std::lround(std::clamp(a(i, j), 0.0, 1.0) * 255.0));
    return img;
}
}  // namespace

std::vector<std::filesystem::path> write_targets(const std::filesystem::path& dir, std::size_t n) {
    std::filesystem::create_directories(dir);
    std::vector<std::filesystem::path> out;
    auto put = [&](const char* name, const Array2<double>& t) {
        const auto p = dir / name;
        write_png(p, gray_png(t));
        out.push_back(p);
    };
    put("cross_ring.png", targets::cross_plus_ring(n));
    put("spot_array_5x5.png", targets::spot_array(n, 5, 5, std::max(2.0, n / 64.0), 0.16));
    put("logo.png", targets::binary_logo(n, "DOE"));
    put("disk.png", targets::disk(n, 0.25));
    put("grating.png", targets::grating(n, std::max(4, static_cast<int>(n / 16))));
    return out;
}

std::vector<std::filesystem::path> write_outputs(const DesignResult& r, const std::filesystem::path& dir, int nz, std::size_t view) {
    std::filesystem::create_directories(dir);
    std::vector<std::filesystem::path> out;
    auto add = [&](const std::filesystem::path& p) { out.push_back(p); return p; };
    const SolverRun& final = r.runs.back();
    const auto active = static_cast<std::size_t>(r.config.active);

    // The DOE is the active aperture; the padding carries no illumination.
    write_npy(add(dir / "phase.npy"), crop_center(final.phi, active));
    write_png(add(dir / "phase.png"), render_phase(crop_center(final.phi, active)));
    Array2<double> I(final.field.rows, final.field.cols);
    for (std::size_t k = 0; k < I.data.size(); ++k) I.data[k] = std::norm(final.field.data[k]);
    write_png(add(dir / "reconstruction.png"), render_intensity(I, false, 40.0));
    write_png(add(dir / "target.png"), render_intensity(r.inputs.i_target, false, 40.0));
    for (const auto& run : r.runs) {
        write_npy(add(dir / ("phase_" + run.name + ".npy")), crop_center(run.phi, active));
        write_png(add(dir / ("phase_" + run.name + ".png")), render_phase(crop_center(run.phi, active)));
        Array2<double> Ir(run.field.rows, run.field.cols);
        for (std::size_t k = 0; k < Ir.data.size(); ++k) Ir.data[k] = std::norm(run.field.data[k]);
        write_png(add(dir / ("reconstruction_" + run.name + ".png")), render_intensity(Ir, false, 40.0));
    }

    write_text(add(dir / "report.json"), r.report().dump() + "\n");
    write_text(add(dir / "table.md"), r.markdown_table());

    SvgPlot plot;
    plot.title = "Convergence (normalized energy)";
    plot.x_label = "iteration";
    plot.y_label = "energy";
    plot.log_y = true;
    std::size_t longest = 0;
    for (const auto& run : r.runs) {
        plot.add_series(run.name, run.history);
        longest = std::max(longest, run.history.size());
    }
    write_text(add(dir / "convergence.svg"), plot.render());

    // The two terms of the energy as separate curves: the shape term (solid) is
    // what the solvers drive down, the efficiency term (dashed, same color)
    // shows the price paid in the window energy; one color per run.
    SvgPlot terms;
    terms.title = "Convergence: shape term (solid) and efficiency term (dashed)";
    terms.x_label = "iteration";
    terms.y_label = "energy term";
    terms.log_y = true;
    for (std::size_t si = 0; si < r.runs.size(); ++si) {
        const auto& run = r.runs[si];
        terms.add_series(run.name + " shape", run.shape_history, false, static_cast<int>(si));
        terms.add_series(run.name + " efficiency", run.efficiency_history, true, static_cast<int>(si));
    }
    write_text(add(dir / "convergence_terms.svg"), terms.render());

    std::vector<std::string> columns{"iteration"};
    for (const auto& run : r.runs) {
        columns.push_back(run.name);
        columns.push_back(run.name + "_shape");
        columns.push_back(run.name + "_efficiency");
    }
    auto at = [](const std::vector<double>& v, std::size_t k) { return k < v.size() ? v[k] : std::nan(""); };
    std::vector<std::vector<double>> rows;
    for (std::size_t k = 0; k < longest; ++k) {
        std::vector<double> row{static_cast<double>(k)};
        for (const auto& run : r.runs) {
            row.push_back(at(run.history, k));
            row.push_back(at(run.shape_history, k));
            row.push_back(at(run.efficiency_history, k));
        }
        rows.push_back(row);
    }
    write_csv(add(dir / "history.csv"), columns, rows);

    for (const auto& p : write_figures(r, dir, nz, view)) out.push_back(p);
    return out;
}

}  // namespace doe
