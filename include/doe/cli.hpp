/// @file cli.hpp
/// @brief Command-line options of `doe_design` and the output writer.
#pragma once

#include "doe/pipeline.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace doe {

/// Parsed command line.
struct CliOptions {
    bool help = false;            ///< --help was given
    std::string target;           ///< target image path (PNG)
    std::string make_targets;     ///< if set: directory to write the synthetic targets to
    std::string out = "results";  ///< output directory
    DesignConfig config;          ///< design parameters
    int nz = 96;                  ///< planes of the volume sweep (rendering, phase 5)
    int view_size = 0;            ///< crop size of the volume (0: min(active, 384))
    int threads = 0;              ///< FFTW threads (0 = all cores)
};

/// Parse `argv`; throws std::invalid_argument with a message on bad input.
CliOptions parse_cli(int argc, char** argv);
/// Usage text listing every option with its default.
std::string cli_usage();

/// Write the synthetic targets (cross+ring, 5x5 spot array, logo, disk, grating) as 8-bit PNGs of side n.
std::vector<std::filesystem::path> write_targets(const std::filesystem::path& dir, std::size_t n);

/// Write the results of a design run into `dir`:
/// phase.npy / phase.png (final design, cropped to the active aperture),
/// phase_<run>.npy per run (cropped likewise), reconstruction.png and
/// target.png (full padded window), report.json, table.md, convergence.svg
/// (total energy per run), convergence_terms.svg (shape and efficiency terms
/// as separate curves), history.csv (all three per run).
/// Also writes the figures and volume.doev (see write_figures()) with `nz`
/// planes and a `view`-pixel crop (0 = min(active, 384)).
std::vector<std::filesystem::path> write_outputs(const DesignResult& result, const std::filesystem::path& dir, int nz = 96,
                                                 std::size_t view = 0);

}  // namespace doe
