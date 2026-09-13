#include "doe/quantize.hpp"

#include "doe/phase.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>

namespace doe {
namespace {

constexpr double pi = std::numbers::pi;

double wrap(double x) { return wrap_to_pi(x); }

// Index z of the nearest level and the signed distance to it (on the circle).
struct Nearest {
    int z;
    double dist;
};
Nearest nearest_level(double phi, int q) {
    const double D = 2.0 * pi / q;
    // (z - 0.5) D <= phi + pi < (z + 0.5) D  ->  z = round((phi + pi) / D), z == q wraps to 0
    const double t = (wrap(phi) + pi) / D;
    int z = static_cast<int>(std::floor(t + 0.5));
    if (z >= q) z -= q;
    if (z < 0) z += q;
    const double level = -pi + z * D;
    return {z, wrap(phi - level)};
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

}  // namespace

template <class T>
void project_to_levels(Array2<T>& phi, int q) {
    if (q < 1) throw std::invalid_argument("project_to_levels: q must be >= 1");
    const Levels L{q};
    for (auto& x : phi.data) x = T(wrap(L.level(nearest_level(double(x), q).z)));
}

template <class T>
std::size_t project_stepwise(Array2<T>& phi, int q, double epsilon) {
    if (q < 1) throw std::invalid_argument("project_stepwise: q must be >= 1");
    const Levels L{q};
    const double capture = 0.5 * epsilon * L.delta();  // Wyrowski eq. (22): |phi - level| < 0.5 e D
    std::size_t moved = 0;
    for (auto& x : phi.data) {
        const Nearest nl = nearest_level(double(x), q);
        if (std::abs(nl.dist) < capture) {
            x = T(wrap(L.level(nl.z)));
            ++moved;
        }
    }
    return moved;
}

double wyrowski_epsilon(int p, int P) {
    static const double table[10] = {0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0};
    if (P < 1 || p < 1 || p > P) throw std::invalid_argument("wyrowski_epsilon: need 1 <= p <= P");
    if (P == 10) return table[p - 1];
    if (P == 1) return 1.0;
    // Interpolate the paper's table linearly in the fraction p/P, mapping
    // p = 1 -> table[0] and p = P -> table[9] = 1.
    const double s = static_cast<double>(p - 1) / static_cast<double>(P - 1) * 9.0;
    const int i = std::min(static_cast<int>(std::floor(s)), 8);
    const double f = s - i;
    return table[i] + f * (table[i + 1] - table[i]);
}

double capture_fraction(QuantRamp ramp, int p, int P) {
    if (P < 1 || p < 1 || p > P) throw std::invalid_argument("capture_fraction: need 1 <= p <= P");
    if (ramp == QuantRamp::table) return wyrowski_epsilon(p, P);
    return static_cast<double>(p) / static_cast<double>(P);  // Skeren et al. 2002 eq. (11)
}

double GumbelSoftmaxQuantizer::score(double phi, int l) const {
    const double d = wrap(phi - Levels{q}.level(l));  // signed angular difference on the circle
    const double s = sigmoid(w * d);
    return s * (1.0 - s);
}

double GumbelSoftmaxQuantizer::value(double phi, const double* gumbel) const {
    // eq. (7): softmax over (score_l + g_l) / tau, computed stably
    std::vector<double> z(q);
    double zmax = -1e300;
    for (int l = 0; l < q; ++l) {
        z[l] = (gain * score(phi, l) + gumbel[l]) / tau;
        zmax = std::max(zmax, z[l]);
    }
    double sum = 0.0, acc = 0.0;
    for (int l = 0; l < q; ++l) {
        z[l] = std::exp(z[l] - zmax);
        sum += z[l];
    }
    for (int l = 0; l < q; ++l) acc += Levels{q}.level(l) * z[l] / sum;  // eq. (6)
    return acc;
}

double GumbelSoftmaxQuantizer::derivative(double phi, const double* gumbel) const {
    const Levels L{q};
    std::vector<double> G(q), ds(q);
    double zmax = -1e300;
    for (int l = 0; l < q; ++l) {
        G[l] = (gain * score(phi, l) + gumbel[l]) / tau;
        zmax = std::max(zmax, G[l]);
        const double d = wrap(phi - L.level(l));
        const double s = sigmoid(w * d);
        ds[l] = gain * w * s * (1.0 - s) * (1.0 - 2.0 * s);  // d (gain * score_l) / d phi
    }
    double sum = 0.0;
    for (int l = 0; l < q; ++l) {
        G[l] = std::exp(G[l] - zmax);
        sum += G[l];
    }
    for (int l = 0; l < q; ++l) G[l] /= sum;
    // d q_hat / d phi = sum_l Q_l sum_m G_l (delta_lm - G_m) ds_m / tau
    double gbar = 0.0;
    for (int m = 0; m < q; ++m) gbar += G[m] * ds[m];
    double out = 0.0;
    for (int l = 0; l < q; ++l) out += L.level(l) * G[l] * (ds[l] - gbar) / tau;
    return out;
}

Array2<double> GumbelSoftmaxQuantizer::sample_noise(std::size_t n, unsigned long long seed) const {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(1e-12, 1.0);
    Array2<double> g(n, static_cast<std::size_t>(q));
    for (auto& x : g.data) x = -std::log(-std::log(uni(rng)));  // Gumbel(0, 1)
    return g;
}

template <class T>
Array2<T> surrogate_gradient(const Array2<T>& grad_q, const Array2<T>& phi, const GumbelSoftmaxQuantizer& Q,
                             const Array2<double>& noise) {
    if (!grad_q.same_shape(phi) || noise.rows != phi.data.size() || noise.cols != static_cast<std::size_t>(Q.q))
        throw std::invalid_argument("surrogate_gradient: shape mismatch");
    Array2<T> g(phi.rows, phi.cols);
    for (std::size_t k = 0; k < phi.data.size(); ++k)
        g.data[k] = T(double(grad_q.data[k]) * Q.derivative(double(phi.data[k]), &noise(k, 0)));
    return g;
}

template void project_to_levels<float>(Array2<float>&, int);
template void project_to_levels<double>(Array2<double>&, int);
template std::size_t project_stepwise<float>(Array2<float>&, int, double);
template std::size_t project_stepwise<double>(Array2<double>&, int, double);
template Array2<float> surrogate_gradient<float>(const Array2<float>&, const Array2<float>&, const GumbelSoftmaxQuantizer&,
                                                 const Array2<double>&);
template Array2<double> surrogate_gradient<double>(const Array2<double>&, const Array2<double>&,
                                                   const GumbelSoftmaxQuantizer&, const Array2<double>&);

}  // namespace doe
