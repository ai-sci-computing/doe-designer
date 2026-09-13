#include "doe/energy.hpp"

#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace doe {

template <class T>
double optimal_scale(const Array2<T>& v_abs, const Array2<T>& b, const Array2<T>& mask) {
    // Reductions in double even for T = float: a float32 accumulation
    // over N^2 terms drifts enough across thousands of iterations to move the
    // scale factor visibly.
    double num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < v_abs.data.size(); ++k) {
        const double w = mask.data[k], bb = b.data[k];
        num += w * bb * double(v_abs.data[k]);
        den += w * bb * bb;
    }
    return den > 0.0 ? num / den : 1.0;
}

template <class T>
EnergyResult<T> energy_and_grad(const Array2<T>& phi, const Array2<T>& illum, const AngularSpectrum<T>& prop,
                                const Array2<T>& b, const Array2<T>& mask, EnergyWeights w) {
    if (!phi.same_shape(illum) || !phi.same_shape(b) || !phi.same_shape(mask))
        throw std::invalid_argument("energy_and_grad: array shapes differ");
    const std::size_t N = phi.data.size();

    // u = illum e^{i phi},  v = A u,  f = |v|
    Field<T> u(phi.rows, phi.cols);
    par::for_each(N, [&](std::size_t k) { u.data[k] = std::polar(illum.data[k], phi.data[k]); });
    Field<T> v = prop.forward(u);

    // amplitudes once (reused by the gradient), then the four reductions in double
    std::vector<double> f(N);
    par::for_each(N, [&](std::size_t k) { f[k] = std::abs(std::complex<double>(v.data[k])); });
    double p = par::sum(N, [&](std::size_t k) { return double(mask.data[k]) * double(b.data[k]) * f[k]; });
    double c = par::sum(N, [&](std::size_t k) { return double(mask.data[k]) * double(b.data[k]) * double(b.data[k]); });
    double s2 = par::sum(N, [&](std::size_t k) { return double(mask.data[k]) * f[k] * f[k]; });
    double e_in = par::sum(N, [&](std::size_t k) { return double(illum.data[k]) * double(illum.data[k]); });
    c = std::max(c, 1e-30);
    s2 = std::max(s2, 1e-30);
    e_in = std::max(e_in, 1e-30);

    EnergyResult<T> r;
    r.shape = w.shape * (1.0 - p * p / (c * s2));            // Fienup 1997 eq. (20)
    r.efficiency = w.efficiency * (1.0 - s2 / e_in);          // 1 - diffraction efficiency
    r.energy = r.shape + r.efficiency;
    r.scale = p / c;  // c is clamped to 1e-30 above

    // dE/df_m = w_s 2 p w_m (p f_m - b_m s2) / (c s2^2) - w_e 2 w_m f_m / E_in ;
    // dE/dv* = dE/df * v / (2 |v|)  (the 1/2 is d|v|/dv*, not canceled).
    // v/|v| is the unit phasor; exact zeros (vortex cores) are guarded.
    const double k_shape = w.shape * 2.0 * p / (c * s2 * s2);
    const double k_eff = w.efficiency * 2.0 / e_in;
    Field<T> g_v(phi.rows, phi.cols);
    par::for_each(N, [&](std::size_t k) {
        const double wm = mask.data[k];
        if (wm == 0.0) {
            g_v.data[k] = std::complex<T>(0, 0);
            return;
        }
        const std::complex<double> vk(v.data[k]);
        const double fk = f[k];
        const double dEdf = k_shape * wm * (p * fk - double(b.data[k]) * s2) - k_eff * wm * fk;
        const double safe = std::max(fk, 1e-12);
        g_v.data[k] = std::complex<T>(dEdf * vk / (2.0 * safe));
    });
    // G = A^H dE/dv*,  dE/dphi = 2 Im(conj(u) G)
    Field<T> G = prop.adjoint(g_v);
    r.grad = Array2<T>(phi.rows, phi.cols);
    par::for_each(N, [&](std::size_t k) {
        r.grad.data[k] = T(2.0 * std::imag(std::conj(std::complex<double>(u.data[k])) * std::complex<double>(G.data[k])));
    });
    r.field = std::move(v);
    return r;
}

template double optimal_scale<float>(const Array2<float>&, const Array2<float>&, const Array2<float>&);
template double optimal_scale<double>(const Array2<double>&, const Array2<double>&, const Array2<double>&);
template EnergyResult<float> energy_and_grad<float>(const Array2<float>&, const Array2<float>&,
                                                    const AngularSpectrum<float>&, const Array2<float>&,
                                                    const Array2<float>&, EnergyWeights);
template EnergyResult<double> energy_and_grad<double>(const Array2<double>&, const Array2<double>&,
                                                      const AngularSpectrum<double>&, const Array2<double>&,
                                                      const Array2<double>&, EnergyWeights);

}  // namespace doe
