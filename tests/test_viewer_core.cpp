// Viewer logic that does not need a GPU: how a stored plane becomes a texture
// in each display mode, the slice animator, the background job runner used
// for re-propagation and redesign, and which parameters need which.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/render.hpp"
#include "doe/viewer_core.hpp"

#include <chrono>
#include <cmath>
#include <numbers>
#include <thread>

using Catch::Matchers::WithinAbs;
constexpr double pi = std::numbers::pi;

namespace {
doe::VolumeData tiny_volume() {
    doe::VolumeData v;
    v.view = 4;
    v.pitch = 8e-6;
    v.wavelength = 532e-9;
    v.z = {0.0, 0.01, 0.02};
    for (int p = 0; p < 3; ++p) {
        doe::Array2<float> amp(4, 4, 0.f), ph(4, 4, 0.f);
        amp(1, 1) = 1.0f;          // one bright pixel
        amp(2, 2) = 0.1f;          // a dim one (-20 dB)
        ph(1, 1) = 0.0f;           // red
        ph(2, 2) = float(2 * pi / 3);  // green
        v.amplitude.push_back(amp);
        v.phase.push_back(ph);
        doe::Array2<std::int8_t> q(3, 3, 0);
        q(0, 0) = 1;
        q(2, 1) = -1;
        v.charges.push_back(q);
    }
    v.xz = doe::Array2<double>(3, 4, 0.0);
    v.yz = doe::Array2<double>(3, 4, 0.0);
    v.vortex_count = {2, 2, 2};
    v.vortex_density = {1, 1, 1};
    return v;
}
std::array<std::uint8_t, 3> px(const doe::ImageU8& im, std::size_t x, std::size_t y) {
    return {im.data[(y * im.width + x) * 3], im.data[(y * im.width + x) * 3 + 1], im.data[(y * im.width + x) * 3 + 2]};
}
}  // namespace

TEST_CASE("slice_texture: intensity, log intensity, HSV and vortex modes", "[viewer]") {
    auto v = tiny_volume();
    doe::SliceStyle st;
    st.mode = doe::SliceMode::intensity;
    auto lin = doe::slice_texture(v, 1, st);
    CHECK(lin.width == 4);
    CHECK(lin.channels == 3);
    CHECK(px(lin, 1, 1) == px(doe::render_intensity(doe::Array2<double>(1, 1, 1.0), false, 40.0), 0, 0));  // max color
    CHECK(px(lin, 0, 0)[0] == doe::colormap_viridis(0.0).r);  // zero
    // the dim pixel is 1 % in intensity: nearly the zero color in linear mode ...
    CHECK(px(lin, 2, 2)[0] < 80);
    st.mode = doe::SliceMode::log_intensity;
    st.floor_db = 40.0;
    auto lg = doe::slice_texture(v, 1, st);
    // ... but half way up the map in log mode (-20 dB of a 40 dB floor)
    CHECK(px(lg, 2, 2) == std::array<std::uint8_t, 3>{doe::colormap_viridis(0.5).r, doe::colormap_viridis(0.5).g, doe::colormap_viridis(0.5).b});
    st.mode = doe::SliceMode::hsv;
    auto hs = doe::slice_texture(v, 1, st);
    CHECK(px(hs, 1, 1) == std::array<std::uint8_t, 3>{255, 0, 0});
    CHECK(px(hs, 0, 0) == std::array<std::uint8_t, 3>{0, 0, 0});
    // time-harmonic animation rotates the hue: phase 0 advanced by 2 pi / 3 is green
    st.time_phase = 2 * pi / 3;
    auto hs2 = doe::slice_texture(v, 1, st);
    CHECK(px(hs2, 1, 1) == std::array<std::uint8_t, 3>{0, 255, 0});
    st.time_phase = 0.0;
    // real part, signed: +amplitude -> red, -amplitude -> blue, zero -> white
    st.mode = doe::SliceMode::real_part;
    auto re = doe::slice_texture(v, 1, st);
    CHECK(px(re, 1, 1) == std::array<std::uint8_t, 3>{255, 0, 0});     // Re(1 e^{i0}) = +1
    CHECK(px(re, 0, 0) == std::array<std::uint8_t, 3>{255, 255, 255});  // zero amplitude
    st.time_phase = pi;
    auto re2 = doe::slice_texture(v, 1, st);
    CHECK(px(re2, 1, 1) == std::array<std::uint8_t, 3>{0, 0, 255});     // Re(1 e^{i pi}) = -1
    st.time_phase = pi / 2;
    auto re3 = doe::slice_texture(v, 1, st);
    CHECK(px(re3, 1, 1)[0] > 250);  // Re = 0 -> white
    CHECK(px(re3, 1, 1)[2] > 250);
    st.time_phase = 0.0;
    st.mode = doe::SliceMode::vortices;
    auto vx = doe::slice_texture(v, 1, st);
    CHECK(px(vx, 0, 0) == std::array<std::uint8_t, 3>{255, 40, 40});   // +1 residue at plaquette (0,0)
    CHECK(px(vx, 2, 1) == std::array<std::uint8_t, 3>{40, 90, 255});   // -1 residue at plaquette (2,1)
    CHECK_THROWS(doe::slice_texture(v, 7, st));
}

TEST_CASE("face_textures: entry face (DOE phase) and exit face (target / reconstruction)", "[viewer]") {
    auto v = tiny_volume();
    v.doe_phase = doe::Array2<double>(4, 4, 0.0);
    v.target = doe::Array2<double>(4, 4, 0.0);
    v.target(1, 1) = 1.0;
    auto entry = doe::entry_face_texture(v);
    CHECK(entry.width == 4);
    CHECK(px(entry, 0, 0)[0] == doe::colormap_twilight(0.5).r);  // phase 0 -> middle of twilight
    auto exit_t = doe::exit_face_texture(v, doe::ExitFace::target);
    CHECK(px(exit_t, 1, 1)[0] == doe::colormap_viridis(1.0).r);
    auto exit_r = doe::exit_face_texture(v, doe::ExitFace::reconstruction);
    CHECK(px(exit_r, 1, 1)[0] == doe::colormap_viridis(1.0).r);  // last plane's intensity
    CHECK(px(exit_r, 2, 2)[0] < 80);
}

TEST_CASE("entry face with a source block: illuminated DOE field as HSV (value = illumination)", "[viewer]") {
    auto v = tiny_volume();
    v.doe_phase = doe::Array2<double>(4, 4, 0.0);
    v.target = doe::Array2<double>(4, 4, 0.0);
    v.source.n = 4;
    v.source.pitch = 8e-6;
    v.source.wavelength = 532e-9;
    v.source.distance = 0.02;
    v.source.phase = doe::Array2<float>(4, 4, 0.f);
    v.source.illum = doe::Array2<float>(4, 4, 0.f);
    v.source.target = doe::Array2<float>(4, 4, 0.f);
    v.source.illum(1, 1) = 1.0f;         // lit pixel, phase 0 -> red
    v.source.illum(2, 2) = 0.5f;
    v.source.phase(2, 2) = float(2 * pi / 3);  // half-lit, phase 2 pi / 3 -> half green
    REQUIRE(v.has_source());
    auto img = doe::entry_face_texture(v, doe::EntryFace::illuminated);
    CHECK(px(img, 1, 1) == std::array<std::uint8_t, 3>{255, 0, 0});
    CHECK(px(img, 0, 0) == std::array<std::uint8_t, 3>{0, 0, 0});     // dark: not illuminated
    CHECK(px(img, 2, 2) == std::array<std::uint8_t, 3>{0, 128, 0});
    auto ph = doe::entry_face_texture(v, doe::EntryFace::phase);      // the twilight phase map, as before
    CHECK(px(ph, 0, 0)[0] == doe::colormap_twilight(0.5).r);
    // without a source block the illuminated face falls back to the phase map
    v.source.n = 0;
    auto fb = doe::entry_face_texture(v, doe::EntryFace::illuminated);
    CHECK(px(fb, 0, 0)[0] == doe::colormap_twilight(0.5).r);
}

TEST_CASE("wall_textures: xz and yz cuts as log-intensity images with z along the image height", "[viewer]") {
    auto v = tiny_volume();
    v.xz(2, 3) = 1.0;  // last plane, x = 3
    auto xz = doe::wall_texture(v, doe::Wall::xz, 40.0);
    CHECK(xz.width == 4);   // x
    CHECK(xz.height == 3);  // z
    CHECK(px(xz, 3, 2)[0] == doe::colormap_viridis(1.0).r);
    CHECK(px(xz, 0, 0)[0] == doe::colormap_viridis(0.0).r);
}

TEST_CASE("SliceAnimator: plays at the given speed, wraps, pauses, seeks", "[viewer]") {
    doe::SliceAnimator a(10);  // 10 planes
    CHECK(a.plane() == 0);
    CHECK_FALSE(a.playing);
    a.step(1.0);
    CHECK(a.plane() == 0);  // paused
    a.playing = true;
    a.speed = 4.0;          // planes per second
    a.step(0.5);
    CHECK(a.plane() == 2);
    a.step(2.0);            // 8 more planes -> 10 -> wraps to 0
    CHECK(a.plane() == 0);
    a.seek(7);
    CHECK(a.plane() == 7);
    CHECK_THAT(a.position(), WithinAbs(7.0, 1e-12));
    a.seek(-3);
    CHECK(a.plane() == 0);
    a.seek(99);
    CHECK(a.plane() == 9);
    a.bounce = true;
    a.seek(9);
    a.step(0.5);  // +2 with bounce -> 7
    CHECK(a.plane() == 7);
}

TEST_CASE("BackgroundJob: runs off-thread, reports progress, hands over the result once, can be canceled", "[viewer]") {
    doe::BackgroundJob<int> job;
    CHECK_FALSE(job.running());
    CHECK_FALSE(job.ready());
    job.start([](doe::JobContext& ctx) {
        for (int i = 0; i < 20; ++i) {
            if (ctx.canceled()) return -1;
            ctx.progress(i / 20.0);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return 42;
    });
    CHECK(job.running());
    for (int k = 0; k < 500 && !job.ready(); ++k) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE(job.ready());
    CHECK(job.progress() >= 0.9);
    CHECK(job.take() == 42);
    CHECK_FALSE(job.ready());  // taken
    CHECK_FALSE(job.running());

    job.start([](doe::JobContext& ctx) {
        while (!ctx.canceled()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return -1;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    job.cancel();
    for (int k = 0; k < 500 && !job.ready(); ++k) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE(job.ready());
    CHECK(job.take() == -1);
    // starting while running is refused
    job.start([](doe::JobContext& ctx) { while (!ctx.canceled()) std::this_thread::sleep_for(std::chrono::milliseconds(1)); return 0; });
    CHECK_FALSE(job.start([](doe::JobContext&) { return 1; }));
    job.cancel();
    for (int k = 0; k < 500 && !job.ready(); ++k) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    (void)job.take();
}

TEST_CASE("ViewerParams: which changes need a redesign and which only a re-propagation", "[viewer]") {
    doe::ViewerParams a, b;
    CHECK(doe::change_class(a, b) == doe::ChangeClass::none);
    b.distance = a.distance * 1.5;
    CHECK(doe::change_class(a, b) == doe::ChangeClass::repropagate);  // the DOE is designed for a distance, but sweeping it is cheap
    b = a;
    b.nz = a.nz + 10;
    CHECK(doe::change_class(a, b) == doe::ChangeClass::repropagate);
    b = a;
    b.levels = 4;
    CHECK(doe::change_class(a, b) == doe::ChangeClass::redesign);
    b = a;
    b.mu = 1.0;
    CHECK(doe::change_class(a, b) == doe::ChangeClass::redesign);
    b = a;
    b.pitch = 6e-6;
    CHECK(doe::change_class(a, b) == doe::ChangeClass::redesign);
    // mapping to a DesignConfig keeps every field
    a.levels = 8;
    a.iters = 123;
    a.mu = 0.7;
    a.init = 2;
    a.illum = 2;
    auto cfg = doe::to_config(a);
    CHECK(cfg.levels == 8);
    CHECK(cfg.iters == 123);
    CHECK(cfg.mu == 0.7);
    CHECK(cfg.init == doe::InitMethod::backprop);
    CHECK(cfg.illum == doe::IllumShape::gaussian);
    CHECK(cfg.active == a.active);
}
