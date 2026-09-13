#include "doe/grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace doe {

int next_smooth(int n) {
    if (n <= 1) return 1;
    const std::int64_t limit = 2LL * n;
    std::vector<std::int64_t> smooth{1};
    for (std::int64_t f : {2, 3, 5, 7}) {
        std::vector<std::int64_t> grown;
        for (std::int64_t s : smooth)
            for (std::int64_t v = s; v <= limit; v *= f) grown.push_back(v);
        smooth.swap(grown);
    }
    std::int64_t best = limit;
    for (std::int64_t s : smooth)
        if (s >= n && s < best) best = s;
    return static_cast<int>(best);
}

Grid Grid::padded(int active, double pitch, double wavelength, double pad) {
    return Grid{next_smooth(static_cast<int>(std::ceil(pad * active))), pitch, wavelength};
}

Grid Grid::padded_for(int active, double pitch, double wavelength, double distance, double pad_min) {
    // Picture-clear rule: S >= D + d tan(theta_max), see the header.
    const double theta = std::asin(std::min(1.0, wavelength / (2.0 * pitch)));
    const double required = (active * pitch + distance * std::tan(theta)) / pitch;
    const int n = std::max(static_cast<int>(std::ceil(pad_min * active)), static_cast<int>(std::ceil(required - 1e-9)));
    return Grid{next_smooth(std::max(n, 1)), pitch, wavelength};
}

double Grid::k() const { return 2.0 * std::numbers::pi / wavelength; }

double Grid::f(std::size_t i) const {
    // numpy.fft.fftfreq: non-negative bins first, then the negative ones.
    const std::int64_t ii = static_cast<std::int64_t>(i);
    const std::int64_t half = (static_cast<std::int64_t>(n) + 1) / 2;
    const std::int64_t bin = ii < half ? ii : ii - n;
    return static_cast<double>(bin) / (n * pitch);
}

double Grid::max_diffraction_angle() const {
    const double f_nyq = 1.0 / (2.0 * pitch);
    return std::asin(std::min(1.0, f_nyq * wavelength));
}

}  // namespace doe
