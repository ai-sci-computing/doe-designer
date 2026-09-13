#include "doe/solvers.hpp"

#include "doe/phase.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>

namespace doe {

template <class T>
Adam<T>::Adam(std::size_t rows, std::size_t cols, AdamParams params)
    : p_(params), m_(rows, cols, 0.0), v_(rows, cols, 0.0) {}

template <class T>
void Adam<T>::step(Array2<T>& x, const Array2<T>& g) {
    if (x.rows != m_.rows || x.cols != m_.cols || g.rows != m_.rows || g.cols != m_.cols)
        throw std::invalid_argument("Adam::step: shape mismatch");
    ++t_;
    const double c1 = 1.0 - std::pow(p_.beta1, t_), c2 = 1.0 - std::pow(p_.beta2, t_);
    for (std::size_t k = 0; k < m_.data.size(); ++k) {
        const double gk = double(g.data[k]);
        m_.data[k] = p_.beta1 * m_.data[k] + (1.0 - p_.beta1) * gk;
        v_.data[k] = p_.beta2 * v_.data[k] + (1.0 - p_.beta2) * gk * gk;
        const double mhat = m_.data[k] / c1, vhat = v_.data[k] / c2;
        x.data[k] = T(double(x.data[k]) - p_.lr * mhat / (std::sqrt(vhat) + p_.eps));
    }
}

template <class T>
void wrap_phase(Array2<T>& phi) {
    for (auto& x : phi.data) x = T(wrap_to_pi(double(x)));
}

template <class T>
Result<T> gerchberg_saxton(const Array2<T>& phi0, const Array2<T>& illum, const AngularSpectrum<T>& prop,
                           const Array2<T>& b, const Array2<T>& mask, const GsParams& params) {
    const int q = params.quant.levels;
    // Wyrowski §3.C: Q cycles for each of the steps p = 1..P-1, one cycle for p = P.
    const int P = std::max(params.quant.steps, 1), Q = std::max(params.quant.cycles_per_step, 1);
    const int quant_cycles = q > 0 ? Q * (P - 1) + 1 : 0;
    const int iters = params.iters + quant_cycles;
    const std::size_t N = phi0.data.size();
    Result<T> r;
    r.phi = phi0;
    Field<T> u(phi0.rows, phi0.cols), v_new(phi0.rows, phi0.cols);
    Array2<T> v_abs(phi0.rows, phi0.cols);
    for (int it = 0; it < iters; ++it) {
        // Forward: v = A(illum e^{i phi}); log the normalized energy for comparison with Adam.
        auto e = energy_and_grad<T>(r.phi, illum, prop, b, mask, params.weights);
        r.history.push_back(e.energy);
        r.shape_history.push_back(e.shape);
        r.efficiency_history.push_back(e.efficiency);
        const Field<T>& v = e.field;
        for (std::size_t k = 0; k < N; ++k) v_abs.data[k] = T(std::abs(std::complex<double>(v.data[k])));
        const double s = optimal_scale(v_abs, b, mask);

        // Fienup's projection residual: squared distance of v to the constraint
        // set of this cycle. For X that set is {c b on W, 0 outside}, so the
        // outside energy counts; for X' only the window does. Non-increasing
        // within a stage when A is unitary (error reduction, Fienup 1982).
        const bool free_outside = it >= params.phase_only_iters || it >= params.iters;
        double res = 0.0;
        for (std::size_t k = 0; k < N; ++k) {
            const double f = double(v_abs.data[k]);
            if (mask.data[k] > 0) {
                const double d = f - s * double(b.data[k]);
                res += d * d;
            } else if (!free_outside) {
                res += f * f;
            }
        }
        r.residual_history.push_back(res);

        // (i) target-plane projection (Wyrowski eqs. 7 / 10): modulus c_j b on the
        // window, phase kept; outside the window zero (X, first stage) or
        // untouched (X', amplitude freedom).
        for (std::size_t k = 0; k < N; ++k) {
            const std::complex<double> vk(v.data[k]);
            const double a = std::max(double(v_abs.data[k]), 1e-12);
            const std::complex<double> phase = vk / a;
            const bool in_window = mask.data[k] > 0;
            const double amp = in_window ? s * double(b.data[k]) : (free_outside ? double(v_abs.data[k]) : 0.0);
            v_new.data[k] = std::complex<T>(amp * phase);
        }
        // (ii) DOE-plane projection (Wyrowski eq. 8): keep only the phase; |u| = illum is re-imposed by the forward model
        Field<T> back = prop.adjoint(v_new);
        for (std::size_t k = 0; k < N; ++k) r.phi.data[k] = T(wrap_to_pi(std::arg(std::complex<double>(back.data[k]))));
        // Quantization stage: Q_Z^(p) (Wyrowski eq. 22) replaces the unit-modulus
        // operator. Table ramp: step p advances every Q cycles, the last step is
        // direct. Linear ramp (Skeren et al. 2002 eq. 11): every one of the J
        // cycles is a step, epsilon = j / J.
        if (q > 0 && it >= params.iters) {
            const int j = it - params.iters + 1;
            const double eps = params.quant.ramp == QuantRamp::linear ? capture_fraction(QuantRamp::linear, j, quant_cycles)
                                                                       : capture_fraction(QuantRamp::table, std::min(P, 1 + (j - 1) / Q), P);
            project_stepwise(r.phi, q, eps);
        }
    }
    if (q > 0) project_to_levels(r.phi, q);
    wrap_phase(r.phi);
    for (std::size_t k = 0; k < N; ++k) u.data[k] = std::polar(illum.data[k], r.phi.data[k]);
    r.field = prop.forward(u);
    return r;
}

template <class T>
Result<T> optimize(const Array2<T>& phi0, const Array2<T>& illum, const AngularSpectrum<T>& prop,
                   const Array2<T>& b, const Array2<T>& mask, const OptimizeParams& params,
                   const ProgressFn& progress) {
    Result<T> r;
    r.phi = phi0;
    AdamParams ap = params.adam;
    ap.lr = params.lr;
    Adam<T> adam(phi0.rows, phi0.cols, ap);
    const QuantParams& qp = params.quant;
    const int q = qp.levels;
    const int it0 = q > 0 ? static_cast<int>(std::floor(qp.start * params.iters)) : params.iters;
    const int remaining = std::max(params.iters - it0, 1);
    const int P = std::max(qp.steps, 1);
    for (int it = 0; it < params.iters; ++it) {
        const bool quantizing = q > 0 && it >= it0;
        if (quantizing && qp.method == QuantMethod::choi) {
            // Choi eq. (5): forward with the hard quantizer, backward through the relaxation.
            Array2<T> phi_q = r.phi;
            project_to_levels(phi_q, q);
            auto e = energy_and_grad<T>(phi_q, illum, prop, b, mask, params.weights);
            r.history.push_back(e.energy);
            r.shape_history.push_back(e.shape);
            r.efficiency_history.push_back(e.efficiency);
            if (progress && !progress(it, e.energy)) break;
            const double frac = remaining > 1 ? double(it - it0) / double(remaining - 1) : 1.0;
            // Supplement S2.3: score scale 300 -> 1000, tau = tau0 exp(-c frac), w from the level spacing.
            const double w = qp.choi_w > 0 ? qp.choi_w : 4.0 * q / (2.0 * std::numbers::pi);
            const GumbelSoftmaxQuantizer Q{q, w, qp.choi_tau0 * std::exp(-qp.choi_c * frac),
                                           qp.choi_gain0 + (qp.choi_gain1 - qp.choi_gain0) * frac};
            const Array2<double> noise = Q.sample_noise(r.phi.data.size(), qp.choi_seed + 7919ULL * static_cast<unsigned long long>(it + 1));
            Array2<T> g = surrogate_gradient(e.grad, r.phi, Q, noise);
            adam.step(r.phi, g);
            wrap_phase(r.phi);
            continue;
        }
        auto e = energy_and_grad<T>(r.phi, illum, prop, b, mask, params.weights);
        r.history.push_back(e.energy);
        r.shape_history.push_back(e.shape);
        r.efficiency_history.push_back(e.efficiency);
        if (progress && !progress(it, e.energy)) break;
        adam.step(r.phi, e.grad);
        wrap_phase(r.phi);  // stay on the torus
        if (quantizing) {
            // Wyrowski eq. (22) as a projected gradient step. Linear ramp (Skeren
            // et al. 2002 eq. 11): every iteration of the stage is a step, epsilon
            // = (it - it0 + 1) / remaining. Table ramp: step p of P grows with the budget.
            const double eps = qp.ramp == QuantRamp::linear
                                   ? capture_fraction(QuantRamp::linear, it - it0 + 1, remaining)
                                   : capture_fraction(QuantRamp::table, std::min(P, 1 + (it - it0) * P / remaining), P);
            project_stepwise(r.phi, q, eps);
        }
    }
    if (q > 0) project_to_levels(r.phi, q);
    Field<T> u(phi0.rows, phi0.cols);
    for (std::size_t k = 0; k < u.data.size(); ++k) u.data[k] = std::polar(illum.data[k], r.phi.data[k]);
    r.field = prop.forward(u);
    return r;
}

template class Adam<float>;
template class Adam<double>;
template void wrap_phase<float>(Array2<float>&);
template void wrap_phase<double>(Array2<double>&);
template Result<float> gerchberg_saxton<float>(const Array2<float>&, const Array2<float>&, const AngularSpectrum<float>&,
                                               const Array2<float>&, const Array2<float>&, const GsParams&);
template Result<double> gerchberg_saxton<double>(const Array2<double>&, const Array2<double>&, const AngularSpectrum<double>&,
                                                 const Array2<double>&, const Array2<double>&, const GsParams&);
template Result<float> optimize<float>(const Array2<float>&, const Array2<float>&, const AngularSpectrum<float>&,
                                       const Array2<float>&, const Array2<float>&, const OptimizeParams&, const ProgressFn&);
template Result<double> optimize<double>(const Array2<double>&, const Array2<double>&, const AngularSpectrum<double>&,
                                         const Array2<double>&, const Array2<double>&, const OptimizeParams&, const ProgressFn&);

}  // namespace doe
