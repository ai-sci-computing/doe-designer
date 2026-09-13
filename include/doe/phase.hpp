/// @file phase.hpp
/// @brief Phase wrapping to the principal interval.
#pragma once

#include <cmath>
#include <numbers>

namespace doe {

/// Wrap @f$x@f$ to @f$(-\pi, \pi]@f$ (the value @f$-\pi@f$ is represented as @f$\pi@f$,
/// matching `std::arg(-1) == pi`). Used by the solvers, the quantizers and the
/// vortex counter so every phase has one canonical representation.
inline double wrap_to_pi(double x) {
    constexpr double two_pi = 2.0 * std::numbers::pi;
    double r = x - two_pi * std::floor((x + std::numbers::pi) / two_pi);  // [-pi, pi)
    if (r <= -std::numbers::pi) r += two_pi;                             // -> (-pi, pi]
    return r;
}

}  // namespace doe
