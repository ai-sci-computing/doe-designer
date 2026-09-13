/// @file viewer_core.hpp
/// @brief GPU-independent logic of the box viewer: textures for the slice, faces
/// and walls, the slice animator, a background job runner and the parameter panel model.
#pragma once

#include "doe/image.hpp"
#include "doe/pipeline.hpp"
#include "doe/render.hpp"

#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

namespace doe {

/// What the moving slice shows.
enum class SliceMode { intensity, log_intensity, hsv, vortices, real_part };

/// Display settings of the slice.
struct SliceStyle {
    SliceMode mode = SliceMode::hsv;  ///< display mode
    double floor_db = 40.0;           ///< log mode: dynamic range below the plane maximum
    double gamma = 1.0;               ///< intensity modes: value^(1/gamma) before color mapping
    double time_phase = 0.0;          ///< HSV mode: hue rotation @f$\omega t@f$ of the time-harmonic animation @f$\mathrm{Re}\,u e^{-i\omega t}@f$
};

/// RGB texture of plane `p` of the volume in the given style. Intensity is
/// normalized to the plane's own maximum; HSV hue = phase + time_phase,
/// value = amplitude / max; vortices: gray intensity with red (+) / blue (-)
/// residues of the stored charge map; real_part: the physical field
/// @f$\mathrm{Re}\,[u\,e^{i\,\mathrm{time\_phase}}]@f$ (i.e. @f$\mathrm{Re}\,u e^{-i\omega t}@f$
/// with time_phase = @f$-\omega t@f$ up to sign) on a blue–white–red map,
/// so crests and troughs are seen moving when time_phase advances.
ImageU8 slice_texture(const VolumeData& v, std::size_t p, const SliceStyle& style);

/// What the entry face shows.
enum class EntryFace { phase, illuminated };
/// Entry face: the DOE phase with the twilight map, or (with a v2 source
/// block) the illuminated DOE field as HSV, hue = phase and value =
/// illumination amplitude, so that a disk or Gaussian illumination is visible.
/// Falls back to the phase map when no source block is present.
ImageU8 entry_face_texture(const VolumeData& v, EntryFace which = EntryFace::phase);

/// What the exit face shows.
enum class ExitFace { target, reconstruction };
/// Exit face: target intensity or the last plane's intensity (viridis, linear).
ImageU8 exit_face_texture(const VolumeData& v, ExitFace which);

/// Which side wall.
enum class Wall { xz, yz };
/// Side wall: the center cut as a log-intensity image, x (or y) along the
/// width and z along the height (row 0 = DOE plane).
ImageU8 wall_texture(const VolumeData& v, Wall which, double floor_db);

/// Plays the slice through the planes.
struct SliceAnimator {
    /// Animator over `planes` planes, starting paused at plane 0.
    explicit SliceAnimator(std::size_t planes) : planes_(planes) {}
    bool playing = false;  ///< advance on step()
    bool bounce = false;   ///< reverse at the ends instead of wrapping
    double speed = 8.0;    ///< planes per second
    /// Advance by `dt` seconds when playing.
    void step(double dt);
    /// Jump to a plane (clamped).
    void seek(long plane);
    /// Current plane index.
    std::size_t plane() const;
    /// Fractional position.
    double position() const { return pos_; }
    /// Number of planes.
    std::size_t planes() const { return planes_; }

private:
    std::size_t planes_;
    double pos_ = 0.0;
    double dir_ = 1.0;
};

/// Passed to a background job: cancellation flag and progress sink.
class JobContext {
public:
    /// True once cancel() was requested on the job.
    bool canceled() const { return stop_.load(); }
    /// Report progress in [0, 1].
    void progress(double p) { progress_.store(p); }

private:
    template <class T> friend class BackgroundJob;
    std::atomic<bool> stop_{false};
    std::atomic<double> progress_{0.0};
};

/// Runs one function on a worker thread and hands its result back to the UI
/// thread. One job at a time; start() refuses while a job runs.
template <class T>
class BackgroundJob {
public:
    BackgroundJob() = default;
    ~BackgroundJob() { cancel(); join(); }
    BackgroundJob(const BackgroundJob&) = delete;
    BackgroundJob& operator=(const BackgroundJob&) = delete;

    /// Start `fn` on a new thread. Returns false if a job is still running.
    bool start(std::function<T(JobContext&)> fn) {
        if (running()) return false;
        join();
        ctx_ = std::make_unique<JobContext>();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            result_.reset();
        }
        running_.store(true);
        thread_ = std::thread([this, fn = std::move(fn)] {
            T r = fn(*ctx_);
            std::lock_guard<std::mutex> lock(mutex_);
            result_ = std::move(r);
            running_.store(false);
        });
        return true;
    }
    /// Ask the job to stop (it must poll JobContext::canceled()).
    void cancel() { if (ctx_) ctx_->stop_.store(true); }
    /// A job is executing.
    bool running() const { return running_.load(); }
    /// A result is waiting to be taken.
    bool ready() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return result_.has_value();
    }
    /// Progress of the running job in [0, 1].
    double progress() const { return ctx_ ? ctx_->progress_.load() : 0.0; }
    /// Take the result (once).
    T take() {
        join();
        std::lock_guard<std::mutex> lock(mutex_);
        T r = std::move(*result_);
        result_.reset();
        return r;
    }

private:
    void join() { if (thread_.joinable()) thread_.join(); }
    std::thread thread_;
    std::unique_ptr<JobContext> ctx_;
    std::atomic<bool> running_{false};
    mutable std::mutex mutex_;
    std::optional<T> result_;
};

/// The parameter panel of the viewer (plain values, ImGui-friendly).
struct ViewerParams {
    double wavelength = 532e-9;  ///< m
    double pitch = 8e-6;         ///< m
    double distance = 0.05;      ///< m
    int active = 256;            ///< px
    int illum = 0;               ///< 0 square, 1 disk, 2 gaussian
    int init = 0;                ///< 0 TIE, 1 random, 2 backprop
    int seed = 0;                ///< seed
    int iters = 300;             ///< Adam iterations
    double lr = 0.05;            ///< Adam step
    double mu = 0.3;             ///< efficiency weight
    int levels = 0;              ///< quantization levels
    int quant_method = 0;        ///< 0 wyrowski, 1 choi
    bool band_limit = true;      ///< band limit
    bool run_gs = false;         ///< GS baseline too
    int nz = 96;                 ///< planes of the sweep
    int view = 256;              ///< crop of the sweep
    double z_end_factor = 1.0;   ///< sweep to z_end_factor * distance
};

/// How much work a parameter change implies.
enum class ChangeClass { none, repropagate, redesign };
/// Compare two parameter sets: the sweep-only fields (distance, nz, view,
/// z_end_factor, band_limit) cost n_z transforms; everything else changes the
/// design and needs a new optimization.
ChangeClass change_class(const ViewerParams& before, const ViewerParams& after);
/// The DesignConfig behind the panel values.
DesignConfig to_config(const ViewerParams& p);

}  // namespace doe
