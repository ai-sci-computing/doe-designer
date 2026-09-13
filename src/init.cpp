#include "doe/init.hpp"

#include "doe/fft.hpp"
#include "doe/phase.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>

namespace doe {

Array2<double> poisson_periodic(const Grid& grid, const Array2<double>& rhs) {
    const std::size_t n = static_cast<std::size_t>(grid.n);
    if (rhs.rows != n || rhs.cols != n) throw std::invalid_argument("poisson_periodic: rhs shape != grid");
    const std::size_t N = n * n;

    // Project to zero mean: the periodic Laplacian has the constants as its
    // null space, so a rhs with non-zero mean has no solution.
    double mean = 0.0;
    for (double x : rhs.data) mean += x;
    mean /= static_cast<double>(N);

    Field<double> r(n, n);
    for (std::size_t k = 0; k < N; ++k) r.data[k] = rhs.data[k] - mean;

    Fft2<double> fft(n);
    Field<double> R = fft.forward(r);
    const double four_pi2 = 4.0 * std::numbers::pi * std::numbers::pi;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double fx = grid.f(i), fy = grid.f(j);
            const double lap = -four_pi2 * (fx * fx + fy * fy);  // symbol of the Laplacian
            R(i, j) = (i == 0 && j == 0) ? 0.0 : R(i, j) / lap;  // DC: zero mean solution
        }
    Field<double> phi = fft.inverse(R);
    Array2<double> out(n, n);
    for (std::size_t k = 0; k < N; ++k) out.data[k] = phi.data[k].real();
    return out;
}

template <class T>
Array2<T> init_tie(const Grid& grid, double distance, const Array2<T>& i_target, const Array2<T>& i_source) {
    const std::size_t n = static_cast<std::size_t>(grid.n);
    if (i_target.rows != n || i_target.cols != n || !i_target.same_shape(i_source))
        throw std::invalid_argument("init_tie: shapes do not match the grid");
    const std::size_t N = n * n;

    double src_max = 0.0, src_sum = 0.0, tgt_sum = 0.0;
    for (std::size_t k = 0; k < N; ++k) {
        src_max = std::max(src_max, double(i_source.data[k]));
        tgt_sum += double(i_target.data[k]);
    }
    const double floor = 1e-6 * src_max;
    Array2<double> src(n, n);
    for (std::size_t k = 0; k < N; ++k) {
        src.data[k] = std::max(double(i_source.data[k]), floor);
        src_sum += src.data[k];
    }
    // Teague eq. (4) with I = I0, dI/dz ~ (I_t - I0)/d:  lap(phi) = (k/d) (1 - I_t/I0)
    const double scale = src_sum / (tgt_sum + 1e-30);  // target rescaled to the source energy
    Array2<double> rhs(n, n);
    const double k_over_d = grid.k() / distance;
    for (std::size_t k = 0; k < N; ++k) rhs.data[k] = k_over_d * (1.0 - scale * double(i_target.data[k]) / src.data[k]);

    Array2<double> phi = poisson_periodic(grid, rhs);
    Array2<T> out(n, n);
    for (std::size_t k = 0; k < N; ++k) out.data[k] = T(phi.data[k]);
    return out;
}

template <class T>
Array2<T> init_backprop(const Grid& grid, const AngularSpectrum<T>& prop, const Array2<double>& b) {
    const std::size_t n = static_cast<std::size_t>(grid.n);
    if (b.rows != n || b.cols != n) throw std::invalid_argument("init_backprop: target shape != grid");
    Field<T> v(n, n);
    for (std::size_t k = 0; k < v.data.size(); ++k) v.data[k] = std::complex<T>(T(b.data[k]), T(0));
    const Field<T> u = prop.adjoint(v);
    Array2<T> phi(n, n);
    for (std::size_t k = 0; k < phi.data.size(); ++k) phi.data[k] = T(wrap_to_pi(std::arg(std::complex<double>(u.data[k]))));
    return phi;
}

template <class T>
Array2<T> init_random(const Grid& grid, unsigned long long seed) {
    const std::size_t n = static_cast<std::size_t>(grid.n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(-std::numbers::pi, std::numbers::pi);
    Array2<T> phi(n, n);
    for (auto& x : phi.data) x = T(uni(rng));
    return phi;
}

template Array2<float> init_tie<float>(const Grid&, double, const Array2<float>&, const Array2<float>&);
template Array2<double> init_tie<double>(const Grid&, double, const Array2<double>&, const Array2<double>&);
template Array2<float> init_backprop<float>(const Grid&, const AngularSpectrum<float>&, const Array2<double>&);
template Array2<double> init_backprop<double>(const Grid&, const AngularSpectrum<double>&, const Array2<double>&);
template Array2<float> init_random<float>(const Grid&, unsigned long long);
template Array2<double> init_random<double>(const Grid&, unsigned long long);

}  // namespace doe
