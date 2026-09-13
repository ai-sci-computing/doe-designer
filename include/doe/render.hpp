/// @file render.hpp
/// @brief Volume sweep between the planes, cross-sections, vortex statistics,
/// the `.doev` volume file for the viewer, and the four standard figures.
#pragma once

#include "doe/array.hpp"
#include "doe/grid.hpp"
#include "doe/image.hpp"
#include "doe/pipeline.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace doe {

/// The wavefront through the box between the DOE plane (z = z0) and the
/// target plane (z = z1), cropped to `view x view` pixels around the axis.
/// Amplitudes and phases are stored in float; charges as int8.
struct VolumeData {
    std::size_t view = 0;              ///< crop side in pixels
    double pitch = 0.0;                ///< sample pitch (m)
    double wavelength = 0.0;           ///< wavelength (m)
    std::vector<double> z;             ///< plane positions (m), nz values from z0 to z1
    std::vector<Array2<float>> amplitude;  ///< |u| per plane
    std::vector<Array2<float>> phase;      ///< arg u per plane, (-pi, pi]
    std::vector<Array2<std::int8_t>> charges;  ///< vortex charge map per plane, (view-1)^2
    Array2<double> xz;                 ///< intensity along x through the center, one row per plane
    Array2<double> yz;                 ///< intensity along y through the center, one row per plane
    std::vector<std::size_t> vortex_count;  ///< vortices per plane
    std::vector<double> vortex_density;     ///< vortices per m^2 per plane
    Array2<double> doe_phase;          ///< DOE phase (cropped to the view), for the entry face
    Array2<double> target;             ///< target intensity (cropped), for the exit face

    /// Full-resolution source of the sweep (format v2), so a viewer can
    /// re-sweep with other plane counts, crops or distances from the file alone.
    struct Source {
        int n = 0;                     ///< padded grid side
        double pitch = 0;              ///< sample pitch (m)
        double wavelength = 0;         ///< wavelength (m)
        double distance = 0;           ///< design distance (m)
        bool band_limit = true;        ///< band limit used
        Array2<float> phase;           ///< DOE phase on the padded grid
        Array2<float> illum;           ///< illumination amplitude on the padded grid
        Array2<float> target;          ///< target intensity on the padded grid
    } source;                          ///< full-resolution source block (empty for v1 files)
    /// True when the full-resolution source is present.
    bool has_source() const { return source.n > 0 && source.phase.size() == static_cast<std::size_t>(source.n) * static_cast<std::size_t>(source.n); }
};

/// Propagate `u` to `nz` planes evenly spaced from z0 to z1 (streaming, one
/// plane in memory at a time) and collect the cropped planes, the center cuts
/// and the vortex statistics. Uses transfer_function() through
/// AngularSpectrum::sweep, i.e. the same band limit as the design propagator
/// (one definition of the transfer function).
VolumeData sweep_volume(const Field<double>& u, const Grid& grid, double z0, double z1, int nz, std::size_t view,
                        bool band_limit);

/// Copy the final DOE phase, the illumination, the target and the geometry of
/// a design into the volume's source block.
void attach_source(VolumeData& volume, const DesignResult& result);
/// Re-sweep from the source block: `nz` planes from 0 to `distance` (or the
/// stored distance when `distance <= 0`), cropped to `view`.
VolumeData sweep_from_source(const VolumeData& volume, int nz, std::size_t view, double distance = 0.0);

/// Write the binary `.doev` volume file (little-endian, format version 2;
/// version 1 files without the source block are still read).
void write_doev(const std::filesystem::path& path, const VolumeData& volume);
/// Read a `.doev` file written by write_doev().
VolumeData read_doev(const std::filesystem::path& path);

/// Vortex map: the target-plane intensity in gray with the plaquette residues
/// of the field marked (red = +1, blue = -1), cropped to `view` pixels.
ImageU8 render_vortex_map(const Field<double>& field, std::size_t view);

/// Write the four spec figures and the viewer data for a design result:
/// fig1_doe_phase.png (twilight, active aperture), fig2_target_reconstruction.png
/// (target | reconstruction | difference), fig3_xz_cross_section.png (log
/// intensity, z downwards), fig4_hsv_planes.png (a row of complex planes as
/// HSV), xz_cut.png / yz_cut.png, vortex_density.svg, volume.doev.
/// `view` = 0 selects min(active, 384).
std::vector<std::filesystem::path> write_figures(const DesignResult& result, const std::filesystem::path& dir, int nz,
                                                 std::size_t view = 0);

}  // namespace doe
