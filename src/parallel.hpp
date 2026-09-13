// Element-wise parallel loops (OpenMP when available). Loops carry no
// reductions except through deterministic per-thread partial sums combined in
// thread order, so results do not depend on the thread count beyond rounding.
#pragma once

#include <cstddef>
#include <vector>

#ifdef DOE_HAVE_OPENMP
#include <omp.h>
#endif

namespace doe::par {

/// Threads OpenMP loops use (set together with the FFT threads).
void set_threads(int n);
int threads();

/// for (i = 0; i < n; ++i) fn(i), in parallel.
template <class F>
void for_each(std::size_t n, F&& fn) {
#ifdef DOE_HAVE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads())
    for (long i = 0; i < static_cast<long>(n); ++i) fn(static_cast<std::size_t>(i));
#else
    for (std::size_t i = 0; i < n; ++i) fn(i);
#endif
}

/// Deterministic parallel sum of fn(i) for i < n: per-thread partials over
/// contiguous blocks, combined in block order.
template <class F>
double sum(std::size_t n, F&& fn) {
    const int t = threads();
    std::vector<double> partial(static_cast<std::size_t>(t), 0.0);
#ifdef DOE_HAVE_OPENMP
#pragma omp parallel num_threads(t)
    {
        const int id = omp_get_thread_num(), nt = omp_get_num_threads();
        const std::size_t lo = n * static_cast<std::size_t>(id) / static_cast<std::size_t>(nt);
        const std::size_t hi = n * static_cast<std::size_t>(id + 1) / static_cast<std::size_t>(nt);
        double s = 0.0;
        for (std::size_t i = lo; i < hi; ++i) s += fn(i);
        partial[static_cast<std::size_t>(id)] = s;
    }
#else
    for (std::size_t i = 0; i < n; ++i) partial[0] += fn(i);
#endif
    double total = 0.0;
    for (double s : partial) total += s;
    return total;
}

}  // namespace doe::par
