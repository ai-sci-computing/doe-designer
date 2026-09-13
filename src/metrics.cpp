#include "doe/metrics.hpp"

#include "doe/energy.hpp"
#include "doe/vortex.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace doe {

template <class T>
Metrics metrics(const Field<T>& v, const Array2<T>& b, const Array2<T>& mask) {
    const std::size_t N = v.data.size();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Metrics m{0.0, 0.0, nan, nan, nan, 0, 0, nan};

    // Amplitudes and the least-squares scale on the window.
    Array2<T> v_abs(v.rows, v.cols);
    double total = 0.0, signal = 0.0, b_max = 0.0;
    for (std::size_t k = 0; k < N; ++k) {
        const double I = std::norm(std::complex<double>(v.data[k]));
        v_abs.data[k] = T(std::sqrt(I));
        total += I;
        signal += double(mask.data[k]) * I;
        b_max = std::max(b_max, double(b.data[k]));
    }
    m.efficiency = signal / (total + 1e-30);
    const double s = optimal_scale(v_abs, b, mask);

    // Window statistics: amplitude residual, intensities a (field) and t (target).
    double err = 0.0, peak_t = 0.0, n_w = 0.0, sum_a = 0.0, sum_t = 0.0;
    std::vector<double> a, t, bright;
    for (std::size_t k = 0; k < N; ++k) {
        if (!(mask.data[k] > 0)) continue;
        const double f = double(v_abs.data[k]), tb = s * double(b.data[k]);
        err += (f - tb) * (f - tb);
        peak_t = std::max(peak_t, tb);
        n_w += 1.0;
        a.push_back(f * f);
        t.push_back(tb * tb);
        sum_a += f * f;
        sum_t += tb * tb;
        if (double(b.data[k]) > 0.5 * b_max) bright.push_back(f * f);
    }
    if (n_w == 0.0) return m;
    m.amplitude_rmse = std::sqrt(err / n_w) / (peak_t + 1e-12);

    // Speckle contrast sigma / mean of the intensity on the bright pixels (population std).
    if (!bright.empty()) {
        double mean = 0.0;
        for (double x : bright) mean += x;
        mean /= double(bright.size());
        double var = 0.0;
        for (double x : bright) var += (x - mean) * (x - mean);
        var /= double(bright.size());
        m.speckle_contrast = mean > 0.0 ? std::sqrt(var) / mean : nan;
    }

    // Pearson correlation of intensities; undefined when either side is constant.
    const double mean_a = sum_a / n_w, mean_t = sum_t / n_w;
    double cov = 0.0, var_a = 0.0, var_t = 0.0, mse = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        cov += (a[k] - mean_a) * (t[k] - mean_t);
        var_a += (a[k] - mean_a) * (a[k] - mean_a);
        var_t += (t[k] - mean_t) * (t[k] - mean_t);
        mse += (a[k] - t[k]) * (a[k] - t[k]);
    }
    mse /= n_w;
    const double tiny = 1e-24 * (mean_a * mean_a + mean_t * mean_t) * n_w;
    m.ncc = (var_a > tiny && var_t > tiny) ? cov / std::sqrt(var_a * var_t) : nan;
    const double peak = peak_t * peak_t;
    m.psnr_db = mse > 0.0 ? 10.0 * std::log10(peak * peak / mse) : std::numeric_limits<double>::infinity();

    // Vortices: plaquettes inside the window / inside the bright features.
    const Array2<int> q = vortex_charge_map(v);
    std::size_t bright_plaquettes = 0;
    for (std::size_t i = 0; i < q.rows; ++i)
        for (std::size_t j = 0; j < q.cols; ++j) {
            auto in_window = [&](std::size_t ii, std::size_t jj) { return mask(ii, jj) > 0; };
            auto is_bright = [&](std::size_t ii, std::size_t jj) { return double(b(ii, jj)) > 0.5 * b_max; };
            const bool win = in_window(i, j) && in_window(i + 1, j) && in_window(i + 1, j + 1) && in_window(i, j + 1);
            const bool bri = is_bright(i, j) && is_bright(i + 1, j) && is_bright(i + 1, j + 1) && is_bright(i, j + 1);
            if (bri) ++bright_plaquettes;
            if (q(i, j) != 0) {
                if (win) ++m.vortex_count;
                if (bri) ++m.vortex_count_bright;
            }
        }
    m.vortex_density_bright = bright_plaquettes > 0 ? double(m.vortex_count_bright) / double(bright_plaquettes) : nan;
    return m;
}

template Metrics metrics<float>(const Field<float>&, const Array2<float>&, const Array2<float>&);
template Metrics metrics<double>(const Field<double>&, const Array2<double>&, const Array2<double>&);

}  // namespace doe
