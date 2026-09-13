#include "doe/viewer_core.hpp"

#include "doe/phase.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace doe {
namespace {

ImageU8 rgb(std::size_t w, std::size_t h) {
    ImageU8 img;
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data.assign(w * h * 3, 0);
    return img;
}
void put(ImageU8& img, std::size_t x, std::size_t y, RGB c) {
    std::uint8_t* p = &img.data[(y * img.width + x) * 3];
    p[0] = c.r;
    p[1] = c.g;
    p[2] = c.b;
}

}  // namespace

ImageU8 slice_texture(const VolumeData& v, std::size_t p, const SliceStyle& st) {
    if (p >= v.amplitude.size()) throw std::out_of_range("slice_texture: plane index");
    const Array2<float>& amp = v.amplitude[p];
    const Array2<float>& ph = v.phase[p];
    const std::size_t n = v.view;
    float amax = 0.f;
    for (float a : amp.data) amax = std::max(amax, a);
    ImageU8 img = rgb(n, n);
    const double inv_gamma = 1.0 / std::max(st.gamma, 1e-6);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double a = amax > 0 ? double(amp(i, j)) / amax : 0.0;
            switch (st.mode) {
                case SliceMode::intensity:
                    put(img, i, j, colormap_viridis(std::pow(a * a, inv_gamma)));
                    break;
                case SliceMode::log_intensity: {
                    const double db = a > 0 ? 20.0 * std::log10(a) : -1e300;
                    put(img, i, j, colormap_viridis(1.0 + db / st.floor_db));
                    break;
                }
                case SliceMode::hsv:
                    put(img, i, j, hsv_phasor(std::polar(a, double(ph(i, j)) + st.time_phase), 1.0));
                    break;
                case SliceMode::vortices: {
                    const auto g = static_cast<std::uint8_t>(std::lround(a * 200.0));
                    put(img, i, j, RGB{g, g, g});
                    break;
                }
                case SliceMode::real_part: {
                    const double re = std::clamp(a * std::cos(double(ph(i, j)) + st.time_phase), -1.0, 1.0);
                    // diverging map: -1 blue, 0 white, +1 red
                    const double m = std::abs(re);
                    auto q = [](double x) { return static_cast<std::uint8_t>(std::lround(std::clamp(x, 0.0, 1.0) * 255.0)); };
                    if (re >= 0) put(img, i, j, RGB{255, q(1.0 - m), q(1.0 - m)});
                    else put(img, i, j, RGB{q(1.0 - m), q(1.0 - m), 255});
                    break;
                }
            }
        }
    if (st.mode == SliceMode::vortices) {
        const Array2<std::int8_t>& q = v.charges[p];
        for (std::size_t i = 0; i < q.rows; ++i)
            for (std::size_t j = 0; j < q.cols; ++j) {
                if (q(i, j) > 0) put(img, i, j, RGB{255, 40, 40});
                else if (q(i, j) < 0) put(img, i, j, RGB{40, 90, 255});
            }
    }
    return img;
}

ImageU8 entry_face_texture(const VolumeData& v, EntryFace which) {
    if (v.doe_phase.size() != v.view * v.view) throw std::invalid_argument("entry_face_texture: no DOE phase in the volume");
    if (which == EntryFace::illuminated && v.has_source()) {
        const std::size_t n = static_cast<std::size_t>(v.source.n), view = v.view;
        const std::size_t i0 = (n - view) / 2, j0 = (n - view) / 2;
        Field<double> f(view, view);
        for (std::size_t i = 0; i < view; ++i)
            for (std::size_t j = 0; j < view; ++j)
                f(i, j) = std::polar(double(v.source.illum(i0 + i, j0 + j)), double(v.source.phase(i0 + i, j0 + j)));
        return render_hsv(f);
    }
    return render_phase(v.doe_phase);
}

ImageU8 exit_face_texture(const VolumeData& v, ExitFace which) {
    if (which == ExitFace::target) {
        if (v.target.size() != v.view * v.view) throw std::invalid_argument("exit_face_texture: no target in the volume");
        return render_intensity(v.target, false, 40.0);
    }
    if (v.amplitude.empty()) throw std::invalid_argument("exit_face_texture: empty volume");
    const Array2<float>& amp = v.amplitude.back();
    Array2<double> I(v.view, v.view);
    for (std::size_t k = 0; k < I.data.size(); ++k) I.data[k] = double(amp.data[k]) * double(amp.data[k]);
    return render_intensity(I, false, 40.0);
}

ImageU8 wall_texture(const VolumeData& v, Wall which, double floor_db) {
    const Array2<double>& cut = which == Wall::xz ? v.xz : v.yz;  // (nz x view)
    Array2<double> t(v.view, cut.rows);                             // width = view (x), height = nz (z)
    for (std::size_t p = 0; p < cut.rows; ++p)
        for (std::size_t i = 0; i < v.view; ++i) t(i, p) = cut(p, i);
    return render_intensity(t, true, floor_db);
}

void SliceAnimator::step(double dt) {
    if (!playing || planes_ < 2) return;
    pos_ += dir_ * speed * dt;
    const double last = static_cast<double>(planes_ - 1);
    if (bounce) {
        while (pos_ > last || pos_ < 0.0) {
            if (pos_ > last) { pos_ = 2 * last - pos_; dir_ = -1.0; }
            if (pos_ < 0.0) { pos_ = -pos_; dir_ = 1.0; }
        }
    } else {
        const double period = static_cast<double>(planes_);
        pos_ = std::fmod(pos_, period);
        if (pos_ < 0.0) pos_ += period;
    }
}

void SliceAnimator::seek(long plane) {
    const long last = static_cast<long>(planes_) - 1;
    pos_ = static_cast<double>(std::clamp(plane, 0L, std::max(last, 0L)));
    dir_ = 1.0;
}

std::size_t SliceAnimator::plane() const {
    const long idx = static_cast<long>(std::floor(pos_ + 1e-9));
    return static_cast<std::size_t>(std::clamp(idx, 0L, static_cast<long>(planes_) - 1));
}

ChangeClass change_class(const ViewerParams& a, const ViewerParams& b) {
    const bool design_changed = a.wavelength != b.wavelength || a.pitch != b.pitch || a.active != b.active || a.illum != b.illum ||
                                a.init != b.init || a.seed != b.seed || a.iters != b.iters || a.lr != b.lr ||
                                a.mu != b.mu || a.levels != b.levels || a.quant_method != b.quant_method || a.run_gs != b.run_gs;
    if (design_changed) return ChangeClass::redesign;
    const bool sweep_changed = a.distance != b.distance || a.nz != b.nz || a.view != b.view || a.z_end_factor != b.z_end_factor ||
                               a.band_limit != b.band_limit;
    return sweep_changed ? ChangeClass::repropagate : ChangeClass::none;
}

DesignConfig to_config(const ViewerParams& p) {
    DesignConfig c;
    c.wavelength = p.wavelength;
    c.pitch = p.pitch;
    c.distance = p.distance;
    c.active = p.active;
    c.illum = p.illum == 0 ? IllumShape::square : p.illum == 1 ? IllumShape::disk : IllumShape::gaussian;
    c.init = p.init == 0 ? InitMethod::tie : p.init == 1 ? InitMethod::random : InitMethod::backprop;
    c.seed = static_cast<unsigned long long>(std::max(p.seed, 0));
    c.iters = p.iters;
    c.lr = p.lr;
    c.mu = p.mu;
    c.levels = p.levels;
    c.quant_method = p.quant_method == 0 ? QuantMethod::wyrowski : QuantMethod::choi;
    c.band_limit = p.band_limit;
    c.run_gs = p.run_gs;
    return c;
}

}  // namespace doe
