// doe_view: the animated wavefront box.
//
// A box spans the DOE plane (entry face, DOE phase) to the target plane (exit
// face, target or reconstruction). A slice plane travels through it showing
// the field at that z (intensity, log intensity, HSV phase/amplitude or the
// vortex map); the side walls carry the xz and yz log-intensity cuts. A
// parameter panel re-propagates the current DOE (cheap) or launches a new
// design on a worker thread (Run). Everything numerical lives in the core
// library (doe/viewer_core.hpp, doe/render.hpp, doe/pipeline.hpp); this file
// is GLFW / OpenGL 3.3 core / Dear ImGui plumbing only.
//
//   doe_view results/volume.doev
//   doe_view --target images/targets/cross_ring.png [design options]
//   doe_view ... --screenshot out.png      render a few frames offscreen and exit
#include "doe/cli.hpp"
#include "doe/fft.hpp"
#include "doe/image.hpp"
#include "doe/pipeline.hpp"
#include "doe/render.hpp"
#include "doe/version.hpp"
#include "doe/viewer_core.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// GL helpers
// ---------------------------------------------------------------------------

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        std::fprintf(stderr, "shader error: %s\n", log);
    }
    return s;
}

GLuint make_program() {
    const char* vs = R"(#version 330 core
        layout(location = 0) in vec3 pos;
        layout(location = 1) in vec2 uv;
        uniform mat4 mvp;
        out vec2 v_uv;
        void main() { v_uv = uv; gl_Position = mvp * vec4(pos, 1.0); })";
    const char* fs = R"(#version 330 core
        in vec2 v_uv;
        uniform sampler2D tex;
        uniform vec4 tint;      // rgb multiplier, alpha
        uniform int use_tex;
        out vec4 frag;
        void main() {
            vec4 c = use_tex == 1 ? texture(tex, v_uv) : vec4(1.0);
            frag = vec4(c.rgb * tint.rgb, tint.a);
        })";
    GLuint p = glCreateProgram();
    GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

struct Texture {
    GLuint id = 0;
    std::size_t w = 0, h = 0;
    void upload(const doe::ImageU8& img) {
        if (!id) glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, static_cast<GLsizei>(img.width), static_cast<GLsizei>(img.height), 0, GL_RGB, GL_UNSIGNED_BYTE, img.data.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        w = img.width;
        h = img.height;
    }
};

// A textured quad given by its four corners (counter-clockwise) and uv.
struct Quad {
    GLuint vao = 0, vbo = 0;
    void init() {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, 6 * 5 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
        glBindVertexArray(0);
    }
    void draw(const glm::vec3 c[4], bool flip_v = false) {
        const float v0 = flip_v ? 1.f : 0.f, v1 = flip_v ? 0.f : 1.f;
        const float data[30] = {c[0].x, c[0].y, c[0].z, 0, v0, c[1].x, c[1].y, c[1].z, 1, v0, c[2].x, c[2].y, c[2].z, 1, v1,
                                c[0].x, c[0].y, c[0].z, 0, v0, c[2].x, c[2].y, c[2].z, 1, v1, c[3].x, c[3].y, c[3].z, 0, v1};
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof data, data);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }
};

struct Lines {
    GLuint vao = 0, vbo = 0;
    std::vector<float> pts;
    void init() {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
        glBindVertexArray(0);
    }
    void add(glm::vec3 a, glm::vec3 b) {
        for (auto p : {a, b}) { pts.insert(pts.end(), {p.x, p.y, p.z, 0.f, 0.f}); }
    }
    void draw() {
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(pts.size() * sizeof(float)), pts.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(pts.size() / 5));
        glBindVertexArray(0);
        pts.clear();
    }
};

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct App {
    GLFWwindow* window = nullptr;
    GLuint program = 0;
    Quad quad;
    Lines lines;
    Texture tex_slice, tex_entry, tex_exit, tex_xz, tex_yz;

    std::optional<doe::VolumeData> volume;
    std::optional<doe::DesignResult> design;
    std::optional<doe::Array2<double>> target;  // grayscale target for redesigns
    std::string target_path;

    doe::ViewerParams params, applied;
    doe::SliceStyle style;
    doe::ExitFace exit_face = doe::ExitFace::reconstruction;
    doe::EntryFace entry_face = doe::EntryFace::illuminated;
    std::unique_ptr<doe::SliceAnimator> anim;
    std::size_t uploaded_plane = static_cast<std::size_t>(-1);
    doe::SliceStyle uploaded_style;
    bool time_anim = false;
    double omega = 2.0;  // rad/s for the time-harmonic hue rotation
    float z_compress = 0.25f;  // box depth relative to width
    float wall_alpha = 0.85f;
    bool show_walls = true, show_faces = true;

    // orbit camera
    float yaw = 0.8f, pitch = 0.45f, dist = 3.2f;
    bool dragging = false;
    double last_x = 0, last_y = 0;

    doe::BackgroundJob<doe::DesignResult> design_job;
    doe::BackgroundJob<doe::VolumeData> sweep_job;
    std::string status = "idle";

    void set_volume(doe::VolumeData v) {
        volume = std::move(v);
        anim = std::make_unique<doe::SliceAnimator>(volume->z.size());
        upload_entry();
        upload_exit();
        tex_xz.upload(doe::wall_texture(*volume, doe::Wall::xz, style.floor_db));
        tex_yz.upload(doe::wall_texture(*volume, doe::Wall::yz, style.floor_db));
        uploaded_plane = static_cast<std::size_t>(-1);
    }
    void upload_entry() {
        if (!volume || volume->doe_phase.size() != volume->view * volume->view) return;
        tex_entry.upload(doe::entry_face_texture(*volume, entry_face));
    }
    void upload_exit() {
        if (!volume) return;
        if (exit_face == doe::ExitFace::target && volume->target.size() != volume->view * volume->view) exit_face = doe::ExitFace::reconstruction;
        tex_exit.upload(doe::exit_face_texture(*volume, exit_face));
    }
    void upload_slice() {
        if (!volume || !anim) return;
        const std::size_t p = anim->plane();
        const bool style_changed = style.mode != uploaded_style.mode || style.floor_db != uploaded_style.floor_db ||
                                   style.gamma != uploaded_style.gamma || style.time_phase != uploaded_style.time_phase;
        if (p == uploaded_plane && !style_changed) return;
        tex_slice.upload(doe::slice_texture(*volume, p, style));
        uploaded_plane = p;
        uploaded_style = style;
    }

    // Launch a sweep of the current DOE (from the design, or from the file's
    // source block) with the current sweep parameters.
    bool can_sweep() const { return (design || (volume && volume->has_source())) && !sweep_job.running() && !design_job.running(); }
    void start_sweep() {
        if (!can_sweep()) return;
        doe::VolumeData src_holder;
        if (design) {
            src_holder.view = 1;
            doe::attach_source(src_holder, *design);
        } else {
            src_holder.source = volume->source;
        }
        const doe::ViewerParams p = params;
        status = "propagating...";
        sweep_job.start([src_holder, p](doe::JobContext& ctx) {
            ctx.progress(0.1);
            doe::VolumeData v = doe::sweep_from_source(src_holder, std::max(p.nz, 2), static_cast<std::size_t>(std::max(p.view, 8)), p.distance * p.z_end_factor);
            ctx.progress(1.0);
            return v;
        });
        applied = params;
    }
    // Launch a new design from the loaded target with the current parameters.
    void start_design() {
        if (!target || design_job.running() || sweep_job.running()) return;
        const doe::Array2<double> t = *target;
        const doe::DesignConfig cfg = doe::to_config(params);
        status = "designing...";
        design_job.start([t, cfg](doe::JobContext& ctx) {
            return doe::design(t, cfg, [&](const std::string&, int it, double) {
                ctx.progress(double(it) / std::max(cfg.iters, 1));
                return !ctx.canceled();
            });
        });
        applied = params;
    }
    void poll_jobs() {
        if (design_job.ready()) {
            design = design_job.take();
            status = "design done";
            start_sweep();
        }
        if (sweep_job.ready()) {
            set_volume(sweep_job.take());
            status = "ready";
        }
    }
};

// Box geometry: x, y in [-1, 1] (the view crop), z from 0 (DOE) to depth.
void draw_scene(App& app, int fb_w, int fb_h) {
    if (!app.volume) return;
    const float depth = 2.0f * app.z_compress;
    glm::mat4 proj = glm::perspective(glm::radians(40.0f), fb_w / float(std::max(fb_h, 1)), 0.05f, 50.0f);
    glm::vec3 center(0.f, 0.f, depth / 2);
    glm::vec3 eye = center + app.dist * glm::vec3(std::cos(app.pitch) * std::sin(app.yaw), std::sin(app.pitch), std::cos(app.pitch) * std::cos(app.yaw));
    glm::mat4 view = glm::lookAt(eye, center, glm::vec3(0, 1, 0));
    glm::mat4 mvp = proj * view;
    glUseProgram(app.program);
    glUniformMatrix4fv(glGetUniformLocation(app.program, "mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1i(glGetUniformLocation(app.program, "tex"), 0);
    const GLint tint = glGetUniformLocation(app.program, "tint"), use_tex = glGetUniformLocation(app.program, "use_tex");
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto face = [&](Texture& t, glm::vec3 c0, glm::vec3 c1, glm::vec3 c2, glm::vec3 c3, float alpha, bool flip = false) {
        if (!t.id) return;
        glBindTexture(GL_TEXTURE_2D, t.id);
        glUniform1i(use_tex, 1);
        glUniform4f(tint, 1, 1, 1, alpha);
        glm::vec3 c[4] = {c0, c1, c2, c3};
        app.quad.draw(c, flip);
    };
    // faces: entry at z = 0, exit at z = depth (x right, y up; texture x along i)
    if (app.show_faces) {
        face(app.tex_entry, {-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, 1.0f);
        face(app.tex_exit, {-1, -1, depth}, {1, -1, depth}, {1, 1, depth}, {-1, 1, depth}, 1.0f);
    }
    // walls: xz cut on the bottom (y = -1), yz cut on the side (x = +1); z runs along the texture height
    if (app.show_walls) {
        face(app.tex_xz, {-1, -1, 0}, {1, -1, 0}, {1, -1, depth}, {-1, -1, depth}, app.wall_alpha);
        face(app.tex_yz, {1, -1, 0}, {1, 1, 0}, {1, 1, depth}, {1, -1, depth}, app.wall_alpha);
    }
    // the moving slice
    if (app.anim) {
        const float z = depth * float(app.anim->position()) / float(std::max<std::size_t>(app.anim->planes() - 1, 1));
        glDepthMask(GL_FALSE);
        face(app.tex_slice, {-1, -1, z}, {1, -1, z}, {1, 1, z}, {-1, 1, z}, 0.95f);
        glDepthMask(GL_TRUE);
        // slice outline
        glUniform1i(use_tex, 0);
        glUniform4f(tint, 1.0f, 0.9f, 0.2f, 1.0f);
        app.lines.add({-1, -1, z}, {1, -1, z}); app.lines.add({1, -1, z}, {1, 1, z});
        app.lines.add({1, 1, z}, {-1, 1, z}); app.lines.add({-1, 1, z}, {-1, -1, z});
        app.lines.draw();
    }
    // box edges
    glUniform1i(use_tex, 0);
    glUniform4f(tint, 0.6f, 0.6f, 0.65f, 1.0f);
    for (float z : {0.f, depth}) {
        app.lines.add({-1, -1, z}, {1, -1, z}); app.lines.add({1, -1, z}, {1, 1, z});
        app.lines.add({1, 1, z}, {-1, 1, z}); app.lines.add({-1, 1, z}, {-1, -1, z});
    }
    for (float x : {-1.f, 1.f}) for (float y : {-1.f, 1.f}) app.lines.add({x, y, 0}, {x, y, depth});
    app.lines.draw();
    glDisable(GL_BLEND);
}

void draw_ui(App& app) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420, 760), ImGuiCond_FirstUseEver);
    ImGui::Begin("Wavefront box");
    ImGui::TextUnformatted(app.status.c_str());
    if (app.design_job.running() || app.sweep_job.running()) {
        ImGui::SameLine();
        ImGui::ProgressBar(float(app.design_job.running() ? app.design_job.progress() : app.sweep_job.progress()), ImVec2(120, 0));
        ImGui::SameLine();
        if (ImGui::SmallButton("cancel")) { app.design_job.cancel(); app.sweep_job.cancel(); }
    }
    if (app.volume && app.anim) {
        if (ImGui::CollapsingHeader("Playback", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button(app.anim->playing ? "Pause" : "Play")) app.anim->playing = !app.anim->playing;
            ImGui::SameLine();
            ImGui::Checkbox("bounce", &app.anim->bounce);
            int plane = static_cast<int>(app.anim->plane());
            if (ImGui::SliderInt("plane", &plane, 0, static_cast<int>(app.anim->planes()) - 1)) app.anim->seek(plane);
            ImGui::Text("z = %.2f mm of %.2f mm", app.volume->z[app.anim->plane()] * 1e3, app.volume->z.back() * 1e3);
            float speed = float(app.anim->speed);
            if (ImGui::SliderFloat("planes / s", &speed, 1.f, 60.f)) app.anim->speed = speed;
            const char* modes[] = {"intensity", "log intensity", "HSV (phase, amplitude)", "vortices", "real part (wave crests)"};
            int mode = static_cast<int>(app.style.mode);
            if (ImGui::Combo("slice", &mode, modes, 5)) app.style.mode = static_cast<doe::SliceMode>(mode);
            float floor_db = float(app.style.floor_db), gamma = float(app.style.gamma);
            if (ImGui::SliderFloat("log floor (dB)", &floor_db, 10.f, 80.f)) app.style.floor_db = floor_db;
            if (ImGui::SliderFloat("gamma", &gamma, 0.3f, 3.f)) app.style.gamma = gamma;
            ImGui::Checkbox("animate in time: Re[u e^{-i w t}] (real part / HSV modes)", &app.time_anim);
            float omega = float(app.omega);
            if (ImGui::SliderFloat("w (rad/s, display time)", &omega, 0.2f, 10.f)) app.omega = omega;
            ImGui::Text("vortices in this plane: %zu  (%.0f / mm^2)", app.volume->vortex_count[app.anim->plane()], app.volume->vortex_density[app.anim->plane()] * 1e-6);
            std::vector<float> dens(app.volume->vortex_density.begin(), app.volume->vortex_density.end());
            for (float& d : dens) d *= 1e-6f;
            ImGui::PlotLines("vortex density vs z", dens.data(), static_cast<int>(dens.size()), 0, nullptr, 0.f, FLT_MAX, ImVec2(0, 60));
        }
        if (ImGui::CollapsingHeader("Box")) {
            ImGui::SliderFloat("z compression", &app.z_compress, 0.05f, 2.f);
            ImGui::Checkbox("walls (xz, yz cuts)", &app.show_walls);
            ImGui::SliderFloat("wall opacity", &app.wall_alpha, 0.f, 1.f);
            ImGui::Checkbox("faces", &app.show_faces);
            int en = app.entry_face == doe::EntryFace::phase ? 0 : 1;
            if (ImGui::Combo("entry face", &en, "DOE phase\0illuminated DOE field (HSV)\0")) { app.entry_face = en == 0 ? doe::EntryFace::phase : doe::EntryFace::illuminated; app.upload_entry(); }
            int ef = app.exit_face == doe::ExitFace::target ? 0 : 1;
            if (ImGui::Combo("exit face", &ef, "target\0reconstruction\0")) { app.exit_face = ef == 0 ? doe::ExitFace::target : doe::ExitFace::reconstruction; app.upload_exit(); }
            ImGui::TextUnformatted("drag: orbit, scroll: zoom");
        }
    }
    if ((app.design || (app.volume && app.volume->has_source())) && ImGui::CollapsingHeader("Sweep (z resolution, crop, distance)", ImGuiTreeNodeFlags_DefaultOpen)) {
        doe::ViewerParams& p = app.params;
        ImGui::SliderInt("planes along z", &p.nz, 8, 512);
        ImGui::SliderInt("view crop (px)", &p.view, 32, 1024);
        float distance_mm = float(p.distance * 1e3);
        if (ImGui::SliderFloat("distance (mm)##sweep", &distance_mm, 1.f, 500.f, "%.1f", ImGuiSliderFlags_Logarithmic)) p.distance = distance_mm * 1e-3;
        float zf = float(p.z_end_factor);
        if (ImGui::SliderFloat("sweep to (x distance)", &zf, 0.2f, 3.f)) p.z_end_factor = zf;
        const bool busy = app.design_job.running() || app.sweep_job.running();
        if (!busy && ImGui::Button("Re-propagate")) app.start_sweep();
        if (!app.design) ImGui::TextWrapped("Re-propagation uses the full-resolution DOE stored in the file. Changing the DOE itself needs doe_view --target IMAGE.png.");
    }
    if (app.target && ImGui::CollapsingHeader("Design parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        doe::ViewerParams& p = app.params;
        float wl_nm = float(p.wavelength * 1e9), pitch_um = float(p.pitch * 1e6), mu = float(p.mu), lr = float(p.lr);
        if (ImGui::SliderFloat("wavelength (nm)", &wl_nm, 400.f, 1100.f)) p.wavelength = wl_nm * 1e-9;
        if (ImGui::SliderFloat("pitch (um)", &pitch_um, 1.f, 20.f)) p.pitch = pitch_um * 1e-6;
        ImGui::SliderInt("active (px)", &p.active, 64, 1024);
        ImGui::Combo("illumination", &p.illum, "square\0disk\0gaussian\0");
        ImGui::Combo("init", &p.init, "TIE (sparse targets)\0random\0backprop (photographs)\0");
        ImGui::InputInt("seed", &p.seed);
        ImGui::SliderInt("iterations", &p.iters, 10, 2000);
        if (ImGui::SliderFloat("lr", &lr, 0.005f, 0.2f)) p.lr = lr;
        if (ImGui::SliderFloat("mu (efficiency weight)", &mu, 0.f, 2.f)) p.mu = mu;
        ImGui::SliderInt("levels (0 = continuous)", &p.levels, 0, 16);
        ImGui::Combo("quantization", &p.quant_method, "wyrowski\0choi\0");
        ImGui::Checkbox("band limit", &p.band_limit);
        ImGui::SameLine();
        ImGui::Checkbox("GS baseline", &p.run_gs);
        const auto cls = doe::change_class(app.applied, p);
        const bool busy = app.design_job.running() || app.sweep_job.running();
        if (!busy) {
            if (ImGui::Button(cls == doe::ChangeClass::redesign ? "Run design (changed)" : "Run design")) app.start_design();
        }
        if (app.design) {
            if (ImGui::CollapsingHeader("Metrics of the last design")) {
                const auto& r = *app.design;
                ImGui::Text("spot %.2f px, Fresnel number %.0f%s", r.sampling.spot_size_px, r.sampling.fresnel_number, r.sampling.spot_resolved ? "" : "  (spot > 1 px!)");
                if (ImGui::BeginTable("m", 6, ImGuiTableFlags_Borders)) {
                    for (const char* h : {"run", "eff", "RMSE", "speckle", "NCC", "PSNR"}) { ImGui::TableNextColumn(); ImGui::TextUnformatted(h); }
                    for (const auto& run : r.runs) {
                        ImGui::TableNextColumn(); ImGui::TextUnformatted(run.name.c_str());
                        ImGui::TableNextColumn(); ImGui::Text("%.3f", run.metrics.efficiency);
                        ImGui::TableNextColumn(); ImGui::Text("%.3f", run.metrics.amplitude_rmse);
                        ImGui::TableNextColumn(); ImGui::Text("%.2f", run.metrics.speckle_contrast);
                        ImGui::TableNextColumn(); ImGui::Text("%.3f", run.metrics.ncc);
                        ImGui::TableNextColumn(); ImGui::Text("%.1f", run.metrics.psnr_db);
                    }
                    ImGui::EndTable();
                }
                std::vector<float> h(app.design->runs.back().history.begin(), app.design->runs.back().history.end());
                if (!h.empty()) ImGui::PlotLines("energy", h.data(), static_cast<int>(h.size()), 0, nullptr, 0.f, FLT_MAX, ImVec2(0, 60));
            }
        }
    }
    if (ImGui::Button("Screenshot (screenshot.png)")) {
        int w, h;
        glfwGetFramebufferSize(app.window, &w, &h);
        doe::ImageU8 img;
        img.width = static_cast<std::size_t>(w);
        img.height = static_cast<std::size_t>(h);
        img.channels = 3;
        img.data.resize(img.width * img.height * 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, img.data.data());
        // flip vertically
        for (std::size_t y = 0; y < img.height / 2; ++y)
            for (std::size_t x = 0; x < img.width * 3; ++x) std::swap(img.data[y * img.width * 3 + x], img.data[(img.height - 1 - y) * img.width * 3 + x]);
        doe::write_png("screenshot.png", img);
        app.status = "wrote screenshot.png";
    }
    ImGui::End();
}

void scroll_cb(GLFWwindow* w, double, double dy) {
    auto* app = static_cast<App*>(glfwGetWindowUserPointer(w));
    if (ImGui::GetIO().WantCaptureMouse) return;
    app->dist = std::clamp(app->dist * float(std::pow(0.9, dy)), 0.5f, 20.f);
}

}  // namespace

int main(int argc, char** argv) {
    // Options: a .doev file, or the doe_design options (--target ...), plus --screenshot PATH.
    std::string doev, screenshot;
    std::vector<char*> rest{argv[0]};
    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        if (s == "--screenshot" && a + 1 < argc) screenshot = argv[++a];
        else if (s.size() > 5 && s.substr(s.size() - 5) == ".doev") doev = s;
        else rest.push_back(argv[a]);
    }
    App app;
    if (doev.empty()) {
        doe::CliOptions opt;
        try {
            opt = doe::parse_cli(static_cast<int>(rest.size()), rest.data());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "doe_view: %s\nusage: doe_view FILE.doev | doe_view --target IMAGE.png [design options] [--screenshot out.png]\n", e.what());
            return 2;
        }
        if (opt.help) { std::printf("usage: doe_view FILE.doev | doe_view --target IMAGE.png [design options] [--screenshot out.png]\n%s", doe::cli_usage().c_str()); return 0; }
        try {
            app.target = doe::to_gray(doe::read_png(opt.target));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "doe_view: %s\n", e.what());
            return 1;
        }
        app.target_path = opt.target;
        doe::set_fft_threads(opt.threads);
        const auto& c = opt.config;
        app.params.wavelength = c.wavelength; app.params.pitch = c.pitch; app.params.distance = c.distance; app.params.active = c.active;
        app.params.illum = c.illum == doe::IllumShape::square ? 0 : c.illum == doe::IllumShape::disk ? 1 : 2;
        app.params.init = c.init == doe::InitMethod::tie ? 0 : c.init == doe::InitMethod::random ? 1 : 2; app.params.seed = static_cast<int>(c.seed);
        app.params.iters = c.iters; app.params.lr = c.lr; app.params.mu = c.mu; app.params.levels = c.levels;
        app.params.quant_method = c.quant_method == doe::QuantMethod::wyrowski ? 0 : 1; app.params.band_limit = c.band_limit; app.params.run_gs = c.run_gs;
        app.params.nz = opt.nz; app.params.view = opt.view_size > 0 ? opt.view_size : std::min(c.active, 384);
    }

    if (!glfwInit()) { std::fprintf(stderr, "doe_view: glfwInit failed (no display?)\n"); return 3; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    if (!screenshot.empty()) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    app.window = glfwCreateWindow(1280, 800, "doe_view - wavefront box", nullptr, nullptr);
    if (!app.window) { std::fprintf(stderr, "doe_view: cannot create an OpenGL 3.3 window\n"); glfwTerminate(); return 3; }
    glfwMakeContextCurrent(app.window);
    glfwSwapInterval(1);
    glfwSetWindowUserPointer(app.window, &app);
    glfwSetScrollCallback(app.window, scroll_cb);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini: fresh layout every start
    ImGui_ImplGlfw_InitForOpenGL(app.window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
    app.program = make_program();
    app.quad.init();
    app.lines.init();

    if (!doev.empty()) {
        try {
            app.set_volume(doe::read_doev(doev));
            app.status = "loaded " + doev;
            app.params.nz = static_cast<int>(app.volume->z.size());
            app.params.view = static_cast<int>(app.volume->view);
            if (app.volume->has_source()) {
                app.params.distance = app.volume->source.distance;
                app.params.pitch = app.volume->source.pitch;
                app.params.wavelength = app.volume->source.wavelength;
                app.params.band_limit = app.volume->source.band_limit;
            }
            app.applied = app.params;
        }
        catch (const std::exception& e) { std::fprintf(stderr, "doe_view: %s\n", e.what()); return 1; }
    } else {
        app.start_design();
    }

    double last = glfwGetTime();
    int frames = 0;
    while (!glfwWindowShouldClose(app.window)) {
        glfwPollEvents();
        const double now = glfwGetTime(), dt = now - last;
        last = now;
        app.poll_jobs();
        if (app.anim) {
            app.anim->step(dt);
            if (app.time_anim) app.style.time_phase = std::fmod(app.style.time_phase + app.omega * dt, 2 * 3.14159265358979);
            app.upload_slice();
        }
        // orbit
        if (!ImGui::GetIO().WantCaptureMouse) {
            double x, y;
            glfwGetCursorPos(app.window, &x, &y);
            const bool down = glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            if (down && app.dragging) {
                // grab convention, as in the web viewer: a drag to the right turns the box to the right
                // (the camera orbits to the left), a drag down brings the top of the box toward the viewer
                app.yaw -= float(x - app.last_x) * 0.005f;
                app.pitch = std::clamp(app.pitch + float(y - app.last_y) * 0.005f, -1.5f, 1.5f);
            }
            app.dragging = down;
            app.last_x = x;
            app.last_y = y;
        }
        int fb_w, fb_h;
        glfwGetFramebufferSize(app.window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);
        glClearColor(0.08f, 0.08f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        draw_scene(app, fb_w, fb_h);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        draw_ui(app);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(app.window);

        if (!screenshot.empty()) {
            // wait for the data, render a few frames, save, exit
            const bool busy = app.design_job.running() || app.sweep_job.running() || !app.volume;
            if (!busy && ++frames >= 3) {
                doe::ImageU8 img;
                img.width = static_cast<std::size_t>(fb_w);
                img.height = static_cast<std::size_t>(fb_h);
                img.channels = 3;
                img.data.resize(img.width * img.height * 3);
                glPixelStorei(GL_PACK_ALIGNMENT, 1);
                glReadBuffer(GL_FRONT);
                glReadPixels(0, 0, fb_w, fb_h, GL_RGB, GL_UNSIGNED_BYTE, img.data.data());
                for (std::size_t y = 0; y < img.height / 2; ++y)
                    for (std::size_t x = 0; x < img.width * 3; ++x) std::swap(img.data[y * img.width * 3 + x], img.data[(img.height - 1 - y) * img.width * 3 + x]);
                doe::write_png(screenshot, img);
                std::printf("wrote %s\n", screenshot.c_str());
                break;
            }
        }
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(app.window);
    glfwTerminate();
    return 0;
}
