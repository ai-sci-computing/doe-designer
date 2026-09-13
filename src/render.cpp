#include "doe/render.hpp"

#include "doe/image.hpp"
#include "doe/io.hpp"
#include "doe/phase.hpp"
#include "doe/propagate.hpp"
#include "doe/vortex.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace doe {

VolumeData sweep_volume(const Field<double>& u, const Grid& grid, double z0, double z1, int nz, std::size_t view,
                        bool band_limit) {
    if (nz < 1) throw std::invalid_argument("sweep_volume: nz >= 1");
    if (view < 2 || view > static_cast<std::size_t>(grid.n)) throw std::invalid_argument("sweep_volume: bad view size");
    VolumeData vol;
    vol.view = view;
    vol.pitch = grid.pitch;
    vol.wavelength = grid.wavelength;
    vol.z.resize(static_cast<std::size_t>(nz));
    for (int p = 0; p < nz; ++p) vol.z[static_cast<std::size_t>(p)] = nz > 1 ? z0 + (z1 - z0) * p / (nz - 1) : z0;
    vol.amplitude.resize(vol.z.size());
    vol.phase.resize(vol.z.size());
    vol.charges.resize(vol.z.size());
    vol.xz = Array2<double>(vol.z.size(), view);
    vol.yz = Array2<double>(vol.z.size(), view);
    vol.vortex_count.resize(vol.z.size());
    vol.vortex_density.resize(vol.z.size());

    Grid crop_grid{static_cast<int>(view), grid.pitch, grid.wavelength};
    AngularSpectrum<double> A(grid, z1, band_limit);
    A.sweep(u, vol.z, [&](std::size_t p, const Field<double>& plane) {
        const Field<double> c = crop_center_field(plane, view);
        Array2<float>& amp = vol.amplitude[p];
        Array2<float>& ph = vol.phase[p];
        amp = Array2<float>(view, view);
        ph = Array2<float>(view, view);
        for (std::size_t k = 0; k < c.data.size(); ++k) {
            amp.data[k] = static_cast<float>(std::abs(c.data[k]));
            ph.data[k] = static_cast<float>(wrap_to_pi(std::arg(c.data[k])));
        }
        const std::size_t center = view / 2;
        for (std::size_t i = 0; i < view; ++i) {
            vol.xz(p, i) = std::norm(c(i, center));
            vol.yz(p, i) = std::norm(c(center, i));
        }
        // Vortex statistics on the stored (float) plane so the viewer's charges match its data.
        Field<double> stored(view, view);
        for (std::size_t k = 0; k < stored.data.size(); ++k) stored.data[k] = std::polar(double(amp.data[k]), double(ph.data[k]));
        const Array2<int> q = vortex_charge_map(stored);
        vol.charges[p] = Array2<std::int8_t>(q.rows, q.cols);
        for (std::size_t k = 0; k < q.data.size(); ++k) vol.charges[p].data[k] = static_cast<std::int8_t>(std::clamp(q.data[k], -127, 127));
        vol.vortex_count[p] = vortex_count(q);
        vol.vortex_density[p] = vortex_density(stored, crop_grid);
    });
    return vol;
}

// ---------------------------------------------------------------------------
// .doev
// ---------------------------------------------------------------------------

namespace {
template <class T>
void put(std::ofstream& f, const T& v) { f.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T>
void put_vec(std::ofstream& f, const std::vector<T>& v) { f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T))); }
template <class T>
T get(std::ifstream& f) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) throw std::runtime_error("read_doev: truncated file");
    return v;
}
template <class T>
void get_vec(std::ifstream& f, std::vector<T>& v) {
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    if (!f) throw std::runtime_error("read_doev: truncated file");
}
}  // namespace

void attach_source(VolumeData& vol, const DesignResult& r) {
    const auto& in = r.inputs;
    const std::size_t n = static_cast<std::size_t>(in.grid.n);
    vol.source.n = in.grid.n;
    vol.source.pitch = in.grid.pitch;
    vol.source.wavelength = in.grid.wavelength;
    vol.source.distance = r.config.distance;
    vol.source.band_limit = r.config.band_limit;
    vol.source.phase = Array2<float>(n, n);
    vol.source.illum = Array2<float>(n, n);
    vol.source.target = Array2<float>(n, n);
    const auto& phi = r.runs.back().phi;
    for (std::size_t k = 0; k < n * n; ++k) {
        vol.source.phase.data[k] = static_cast<float>(phi.data[k]);
        vol.source.illum.data[k] = static_cast<float>(in.illum.data[k]);
        vol.source.target.data[k] = static_cast<float>(in.i_target.data[k]);
    }
}

VolumeData sweep_from_source(const VolumeData& vol, int nz, std::size_t view, double distance) {
    if (!vol.has_source()) throw std::invalid_argument("sweep_from_source: the volume has no source block");
    const auto& src = vol.source;
    const std::size_t n = static_cast<std::size_t>(src.n);
    Grid grid{src.n, src.pitch, src.wavelength};
    Field<double> u(n, n);
    for (std::size_t k = 0; k < n * n; ++k) u.data[k] = std::polar(double(src.illum.data[k]), double(src.phase.data[k]));
    const double z1 = distance > 0.0 ? distance : src.distance;
    view = std::min<std::size_t>(std::max<std::size_t>(view, 8), n);
    VolumeData out = sweep_volume(u, grid, 0.0, z1, std::max(nz, 2), view, src.band_limit);
    Array2<double> phi(n, n), tgt(n, n);
    for (std::size_t k = 0; k < n * n; ++k) {
        phi.data[k] = src.phase.data[k];
        tgt.data[k] = src.target.data[k];
    }
    out.doe_phase = crop_center(phi, view);
    out.target = crop_center(tgt, view);
    out.source = src;
    return out;
}

void write_doev(const std::filesystem::path& path, const VolumeData& vol) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_doev: cannot open " + path.string());
    f.write("DOEV", 4);
    put<std::uint32_t>(f, 2);
    put<std::uint32_t>(f, static_cast<std::uint32_t>(vol.view));
    put<std::uint32_t>(f, static_cast<std::uint32_t>(vol.z.size()));
    put<double>(f, vol.pitch);
    put<double>(f, vol.wavelength);
    put_vec(f, vol.z);
    for (std::size_t p = 0; p < vol.z.size(); ++p) {
        put_vec(f, vol.amplitude[p].data);
        put_vec(f, vol.phase[p].data);
        put_vec(f, vol.charges[p].data);
    }
    put_vec(f, vol.xz.data);
    put_vec(f, vol.yz.data);
    std::vector<std::uint32_t> counts(vol.vortex_count.begin(), vol.vortex_count.end());
    put_vec(f, counts);
    put_vec(f, vol.vortex_density);
    const std::uint32_t has_faces = (vol.doe_phase.size() == vol.view * vol.view && vol.target.size() == vol.view * vol.view) ? 1 : 0;
    put<std::uint32_t>(f, has_faces);
    if (has_faces) {
        put_vec(f, vol.doe_phase.data);
        put_vec(f, vol.target.data);
    }
    // v2: source block
    const std::uint32_t has_source = vol.has_source() ? 1 : 0;
    put<std::uint32_t>(f, has_source);
    if (has_source) {
        put<std::uint32_t>(f, static_cast<std::uint32_t>(vol.source.n));
        put<double>(f, vol.source.pitch);
        put<double>(f, vol.source.wavelength);
        put<double>(f, vol.source.distance);
        put<std::uint32_t>(f, vol.source.band_limit ? 1 : 0);
        put_vec(f, vol.source.phase.data);
        put_vec(f, vol.source.illum.data);
        put_vec(f, vol.source.target.data);
    }
}

VolumeData read_doev(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_doev: cannot open " + path.string());
    char magic[4];
    f.read(magic, 4);
    if (!f || std::memcmp(magic, "DOEV", 4) != 0) throw std::runtime_error("read_doev: not a .doev file");
    const std::uint32_t version = get<std::uint32_t>(f);
    if (version != 1 && version != 2) throw std::runtime_error("read_doev: unsupported version");
    VolumeData vol;
    vol.view = get<std::uint32_t>(f);
    const std::size_t nz = get<std::uint32_t>(f);
    vol.pitch = get<double>(f);
    vol.wavelength = get<double>(f);
    vol.z.resize(nz);
    get_vec(f, vol.z);
    const std::size_t v2 = vol.view * vol.view, q2 = (vol.view - 1) * (vol.view - 1);
    vol.amplitude.assign(nz, Array2<float>(vol.view, vol.view));
    vol.phase.assign(nz, Array2<float>(vol.view, vol.view));
    vol.charges.assign(nz, Array2<std::int8_t>(vol.view - 1, vol.view - 1));
    for (std::size_t p = 0; p < nz; ++p) {
        get_vec(f, vol.amplitude[p].data);
        get_vec(f, vol.phase[p].data);
        get_vec(f, vol.charges[p].data);
    }
    (void)v2; (void)q2;
    vol.xz = Array2<double>(nz, vol.view);
    vol.yz = Array2<double>(nz, vol.view);
    get_vec(f, vol.xz.data);
    get_vec(f, vol.yz.data);
    std::vector<std::uint32_t> counts(nz);
    get_vec(f, counts);
    vol.vortex_count.assign(counts.begin(), counts.end());
    vol.vortex_density.resize(nz);
    get_vec(f, vol.vortex_density);
    if (get<std::uint32_t>(f) == 1) {
        vol.doe_phase = Array2<double>(vol.view, vol.view);
        vol.target = Array2<double>(vol.view, vol.view);
        get_vec(f, vol.doe_phase.data);
        get_vec(f, vol.target.data);
    }
    if (version >= 2 && get<std::uint32_t>(f) == 1) {
        vol.source.n = static_cast<int>(get<std::uint32_t>(f));
        vol.source.pitch = get<double>(f);
        vol.source.wavelength = get<double>(f);
        vol.source.distance = get<double>(f);
        vol.source.band_limit = get<std::uint32_t>(f) == 1;
        const std::size_t n = static_cast<std::size_t>(vol.source.n);
        vol.source.phase = Array2<float>(n, n);
        vol.source.illum = Array2<float>(n, n);
        vol.source.target = Array2<float>(n, n);
        get_vec(f, vol.source.phase.data);
        get_vec(f, vol.source.illum.data);
        get_vec(f, vol.source.target.data);
    }
    return vol;
}

// ---------------------------------------------------------------------------
// figures
// ---------------------------------------------------------------------------

namespace {

ImageU8 blank(std::size_t w, std::size_t h) {
    ImageU8 img;
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data.assign(w * h * 3, 255);
    return img;
}
void blit(ImageU8& dst, const ImageU8& src, std::size_t x0, std::size_t y0) {
    for (std::size_t y = 0; y < src.height; ++y)
        for (std::size_t x = 0; x < src.width; ++x)
            for (std::size_t c = 0; c < 3; ++c)
                dst.data[((y0 + y) * dst.width + x0 + x) * 3 + c] = src.data[(y * src.width + x) * 3 + c];
}
// nearest-neighbor vertical stretch so a few planes become a readable image
ImageU8 stretch_rows(const ImageU8& src, std::size_t factor) {
    ImageU8 out = blank(src.width, src.height * factor);
    for (std::size_t y = 0; y < out.height; ++y)
        for (std::size_t x = 0; x < src.width; ++x)
            for (std::size_t c = 0; c < 3; ++c) out.data[(y * out.width + x) * 3 + c] = src.data[((y / factor) * src.width + x) * 3 + c];
    return out;
}

}  // namespace

ImageU8 render_vortex_map(const Field<double>& field, std::size_t view) {
    const Field<double> c = crop_center_field(field, view);
    double imax = 0.0;
    for (const auto& z : c.data) imax = std::max(imax, std::norm(z));
    ImageU8 img = blank(view, view);
    for (std::size_t i = 0; i < view; ++i)
        for (std::size_t j = 0; j < view; ++j) {
            const auto g = static_cast<std::uint8_t>(std::lround(std::sqrt(imax > 0 ? std::norm(c(i, j)) / imax : 0.0) * 200.0));
            std::uint8_t* p = &img.data[(j * view + i) * 3];
            p[0] = p[1] = p[2] = g;
        }
    const Array2<int> q = vortex_charge_map(c);
    for (std::size_t i = 0; i < q.rows; ++i)
        for (std::size_t j = 0; j < q.cols; ++j) {
            if (q(i, j) == 0) continue;
            std::uint8_t* p = &img.data[(j * view + i) * 3];
            if (q(i, j) > 0) { p[0] = 255; p[1] = 40; p[2] = 40; }
            else { p[0] = 40; p[1] = 90; p[2] = 255; }
        }
    return img;
}

std::vector<std::filesystem::path> write_figures(const DesignResult& r, const std::filesystem::path& dir, int nz,
                                                 std::size_t view) {
    std::filesystem::create_directories(dir);
    std::vector<std::filesystem::path> out;
    auto add = [&](const std::filesystem::path& p) { out.push_back(p); return p; };
    const auto& in = r.inputs;
    const std::size_t active = static_cast<std::size_t>(r.config.active);
    if (view == 0) view = std::min<std::size_t>(active, 384);
    view = std::min<std::size_t>(view, static_cast<std::size_t>(in.grid.n));
    const SolverRun& final = r.runs.back();

    // Field leaving the DOE, and the sweep through the box.
    Field<double> u(in.grid.n, in.grid.n);
    for (std::size_t k = 0; k < u.data.size(); ++k) u.data[k] = std::polar(in.illum.data[k], final.phi.data[k]);
    VolumeData vol = sweep_volume(u, in.grid, 0.0, r.config.distance, nz, view, r.config.band_limit);
    vol.doe_phase = crop_center(final.phi, view);
    vol.target = crop_center(in.i_target, view);
    attach_source(vol, r);
    write_doev(add(dir / "volume.doev"), vol);

    // Figure 1: DOE phase, cyclic map, quantization steps visible.
    write_png(add(dir / "fig1_doe_phase.png"), render_phase(crop_center(final.phi, active)));

    // Figure 2: target | reconstruction | difference on the same scale.
    Array2<double> I(final.field.rows, final.field.cols);
    for (std::size_t k = 0; k < I.data.size(); ++k) I.data[k] = std::norm(final.field.data[k]);
    Array2<double> rec = crop_center(I, view), tgt = crop_center(in.i_target, view);
    double rmax = 0, tmax = 0;
    for (double v : rec.data) rmax = std::max(rmax, v);
    for (double v : tgt.data) tmax = std::max(tmax, v);
    Array2<double> rec_n = rec, diff(view, view);
    for (std::size_t k = 0; k < rec.data.size(); ++k) {
        rec_n.data[k] = rmax > 0 ? rec.data[k] / rmax : 0.0;
        diff.data[k] = std::abs(rec_n.data[k] - (tmax > 0 ? tgt.data[k] / tmax : 0.0));
    }
    {
        const std::size_t gap = 4;
        ImageU8 fig = blank(3 * view + 2 * gap, view);
        blit(fig, render_intensity(tgt, false, 40.0), 0, 0);
        blit(fig, render_intensity(rec_n, false, 40.0), view + gap, 0);
        blit(fig, render_intensity(diff, false, 40.0), 2 * (view + gap), 0);
        write_png(add(dir / "fig2_target_reconstruction.png"), fig);
    }

    // Figure 3: xz cross-section, log intensity, z downwards (DOE at the top).
    {
        // vol.xz is (nz x view): rows = z, cols = x. render_intensity maps rows to
        // the image width, so transpose to get x across and z down.
        Array2<double> t(view, vol.z.size());
        for (std::size_t p = 0; p < vol.z.size(); ++p)
            for (std::size_t i = 0; i < view; ++i) t(i, p) = vol.xz(p, i);
        ImageU8 img = render_intensity(t, true, 40.0);
        const std::size_t factor = std::max<std::size_t>(1, view / vol.z.size());
        write_png(add(dir / "fig3_xz_cross_section.png"), stretch_rows(img, factor));
        write_png(add(dir / "xz_cut.png"), stretch_rows(img, factor));
        Array2<double> ty(view, vol.z.size());
        for (std::size_t p = 0; p < vol.z.size(); ++p)
            for (std::size_t i = 0; i < view; ++i) ty(i, p) = vol.yz(p, i);
        write_png(add(dir / "yz_cut.png"), stretch_rows(render_intensity(ty, true, 40.0), factor));
    }

    // Figure 4: a row of complex planes as HSV (hue = phase, value = amplitude).
    {
        const std::size_t count = std::min<std::size_t>(vol.z.size(), 6), gap = 4;
        ImageU8 fig = blank(count * view + (count - 1) * gap, view);
        for (std::size_t c = 0; c < count; ++c) {
            const std::size_t p = count > 1 ? c * (vol.z.size() - 1) / (count - 1) : 0;
            Field<double> plane(view, view);
            for (std::size_t k = 0; k < plane.data.size(); ++k) plane.data[k] = std::polar(double(vol.amplitude[p].data[k]), double(vol.phase[p].data[k]));
            blit(fig, render_hsv(plane), c * (view + gap), 0);
        }
        write_png(add(dir / "fig4_hsv_planes.png"), fig);
    }

    write_png(add(dir / "vortex_map.png"), render_vortex_map(final.field, view));

    // Vortex density against z.
    {
        SvgPlot plot;
        plot.title = "Vortex density along the propagation axis";
        plot.x_label = "z (mm)";
        plot.y_label = "vortices per mm^2";
        std::vector<double> zmm(vol.z.size()), dens(vol.z.size());
        for (std::size_t p = 0; p < vol.z.size(); ++p) {
            zmm[p] = vol.z[p] * 1e3;
            dens[p] = vol.vortex_density[p] * 1e-6;
        }
        plot.add_series("vortex density", zmm, dens);
        write_text(add(dir / "vortex_density.svg"), plot.render());
    }
    return out;
}

}  // namespace doe
