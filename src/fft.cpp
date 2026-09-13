#include "doe/fft.hpp"

#include <utility>

#include "parallel.hpp"

#include <fftw3.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace doe {
namespace {

std::mutex& planner_mutex() {
    static std::mutex m;
    return m;
}

int& thread_setting() {
    static int n = 0;  // 0 = hardware concurrency
    return n;
}

int resolved_threads() {
    const int n = thread_setting();
    if (n > 0) return n;
    return std::max(1u, std::thread::hardware_concurrency());
}

void ensure_threads_initialized() {
    static const bool ok = [] { return fftw_init_threads() != 0 && fftwf_init_threads() != 0; }();
    if (!ok) throw std::runtime_error("FFTW: thread initialization failed");
}

// FFTW's double and single precision APIs are separate C symbol families;
// this trait maps the working precision onto them.
template <class T>
struct fftw_traits;

template <>
struct fftw_traits<double> {
    using plan = fftw_plan;
    using cplx = fftw_complex;
    static plan make(int n, cplx* in, cplx* out, int sign, unsigned flags) {
        return fftw_plan_dft_2d(n, n, in, out, sign, flags);
    }
    static void exec(plan p, cplx* in, cplx* out) { fftw_execute_dft(p, in, out); }
    static void destroy(plan p) { fftw_destroy_plan(p); }
};

template <>
struct fftw_traits<float> {
    using plan = fftwf_plan;
    using cplx = fftwf_complex;
    static plan make(int n, cplx* in, cplx* out, int sign, unsigned flags) {
        return fftwf_plan_dft_2d(n, n, in, out, sign, flags);
    }
    static void exec(plan p, cplx* in, cplx* out) { fftwf_execute_dft(p, in, out); }
    static void destroy(plan p) { fftwf_destroy_plan(p); }
};

constexpr unsigned planner_flags() {
#ifdef DOE_FFTW_MEASURE
    return FFTW_MEASURE | FFTW_UNALIGNED;
#else
    return FFTW_ESTIMATE | FFTW_UNALIGNED;
#endif
}

}  // namespace

void set_fft_threads(int n) { thread_setting() = std::max(n, 0); }
int fft_threads() { return resolved_threads(); }

namespace par {
void set_threads(int n) { set_fft_threads(n); }
int threads() { return resolved_threads(); }
}  // namespace par

template <class T>
Fft2<T>::Fft2(std::size_t n) : n_(n) {
    if (n == 0) throw std::invalid_argument("Fft2: size must be positive");
    using tr = fftw_traits<T>;
    // Scratch buffers are only needed to create the plans; FFTW_MEASURE would
    // overwrite them, which is why they are not the caller's arrays.
    Field<T> a(n, n), b(n, n);
    auto* pa = reinterpret_cast<typename tr::cplx*>(a.data.data());
    auto* pb = reinterpret_cast<typename tr::cplx*>(b.data.data());
    std::lock_guard<std::mutex> lock(planner_mutex());
    ensure_threads_initialized();
    fftw_plan_with_nthreads(resolved_threads());
    fftwf_plan_with_nthreads(resolved_threads());
    fwd_ = tr::make(static_cast<int>(n), pa, pb, FFTW_FORWARD, planner_flags());
    inv_ = tr::make(static_cast<int>(n), pa, pb, FFTW_BACKWARD, planner_flags());
    if (!fwd_ || !inv_) throw std::runtime_error("Fft2: FFTW planning failed");
}

template <class T>
Fft2<T>::~Fft2() {
    using tr = fftw_traits<T>;
    std::lock_guard<std::mutex> lock(planner_mutex());
    if (fwd_) tr::destroy(static_cast<typename tr::plan>(fwd_));
    if (inv_) tr::destroy(static_cast<typename tr::plan>(inv_));
}

template <class T>
Fft2<T>::Fft2(Fft2&& other) noexcept : n_(other.n_), fwd_(other.fwd_), inv_(other.inv_) {
    other.fwd_ = other.inv_ = nullptr;
    other.n_ = 0;
}

template <class T>
Fft2<T>& Fft2<T>::operator=(Fft2&& other) noexcept {
    // Swap, so the plans this object held are destroyed with `other` when it dies
    // (and a self-assignment is a no-op); no explicit destructor call on a live object.
    std::swap(n_, other.n_);
    std::swap(fwd_, other.fwd_);
    std::swap(inv_, other.inv_);
    return *this;
}

template <class T>
void Fft2<T>::execute(void* plan, const Field<T>& in, Field<T>& out) const {
    using tr = fftw_traits<T>;
    if (in.rows != n_ || in.cols != n_) throw std::invalid_argument("Fft2: input has the wrong shape");
    if (&in == &out) throw std::invalid_argument("Fft2: in-place execution is not supported");
    if (!out.same_shape(in)) out = Field<T>(n_, n_);
    // std::complex<T> is layout-compatible with T[2] (C++ standard [complex.numbers]),
    // which is exactly FFTW's fftw(f)_complex.
    auto* pin = reinterpret_cast<typename tr::cplx*>(const_cast<std::complex<T>*>(in.data.data()));
    auto* pout = reinterpret_cast<typename tr::cplx*>(out.data.data());
    tr::exec(static_cast<typename tr::plan>(plan), pin, pout);
    // Orthonormal scaling: 1/sqrt(n*n) = 1/n on each direction.
    const T scale = T(1) / static_cast<T>(n_);
    for (auto& z : out.data) z *= scale;
}

template <class T>
void Fft2<T>::forward(const Field<T>& in, Field<T>& out) const { execute(fwd_, in, out); }

template <class T>
void Fft2<T>::inverse(const Field<T>& in, Field<T>& out) const { execute(inv_, in, out); }

template <class T>
Field<T> Fft2<T>::forward(const Field<T>& in) const {
    Field<T> out(n_, n_);
    forward(in, out);
    return out;
}

template <class T>
Field<T> Fft2<T>::inverse(const Field<T>& in) const {
    Field<T> out(n_, n_);
    inverse(in, out);
    return out;
}

template class Fft2<float>;
template class Fft2<double>;

}  // namespace doe
