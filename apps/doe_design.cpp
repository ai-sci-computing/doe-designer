// doe_design: command-line front end of the design pipeline.
#include "doe/cli.hpp"
#include "doe/fft.hpp"
#include "doe/image.hpp"
#include "doe/version.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <format>

int main(int argc, char** argv) {
    doe::CliOptions opt;
    try {
        opt = doe::parse_cli(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "doe_design: %s\n\n%s", e.what(), doe::cli_usage().c_str());
        return 2;
    }
    if (opt.help) {
        std::fputs(doe::cli_usage().c_str(), stdout);
        return 0;
    }
    doe::set_fft_threads(opt.threads);
    try {
        if (!opt.make_targets.empty()) {
            auto files = doe::write_targets(opt.make_targets, static_cast<std::size_t>(opt.config.active));
            for (const auto& f : files) std::printf("wrote %s\n", f.string().c_str());
            return 0;
        }
        std::printf("doe_design %s  (FFTW %s, %d threads)\n", doe::version().c_str(), doe::fftw_version_string().c_str(), doe::fft_threads());
        auto target = doe::to_gray(doe::read_png(opt.target));
        std::printf("target %s: %zu x %zu px -> %d x %d active, padded window %d\n", opt.target.c_str(), target.rows, target.cols,
                    opt.config.active, opt.config.active, doe::grid_for(opt.config).n);

        // Sampling report before anything runs.
        const auto sr = doe::sampling_for(opt.config);  // the report design() stores as well: one computation, one grid
        std::printf("sampling: max angle %.2f deg, spot %.2f px (%.1f um), first-order spread %.2f mm, window %.2f mm (plate + half the spread %.2f mm), Fresnel number %.0f\n",
                    sr.max_angle_rad * 57.29577951, sr.spot_size_px, sr.spot_size_m * 1e6, sr.signal_spread_m * 1e3,
                    sr.window_extent_m * 1e3, (opt.config.active * opt.config.pitch + 0.5 * sr.signal_spread_m) * 1e3, sr.fresnel_number);
        if (!sr.spot_resolved) std::printf("WARNING: diffraction-limited spot exceeds one target pixel; finer detail is unreachable\n");
        if (!sr.wrap_misses_picture) std::printf("WARNING: light wrapping around the periodic window reaches the picture\n");

        int last_print = -1000;
        auto progress = [&](const std::string& name, int it, double e) {
            if (it - last_print >= 50 || it == 0) {
                std::printf("  %-8s iter %4d   E = %.4e\n", name.c_str(), it, e);
                last_print = it;
            }
            return true;
        };
        auto result = doe::design(target, opt.config, progress);
        auto files = doe::write_outputs(result, opt.out, opt.nz, static_cast<std::size_t>(std::max(opt.view_size, 0)));
        std::printf("\n%s\n", result.markdown_table().c_str());
        std::printf("wrote %zu files to %s\n", files.size(), opt.out.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "doe_design: error: %s\n", e.what());
        return 1;
    }
}
