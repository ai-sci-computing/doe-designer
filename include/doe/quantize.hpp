/// @file quantize.hpp
/// @brief Phase quantization onto @f$Z@f$ levels: Wyrowski's stepwise operator and
/// Choi's Gumbel-Softmax relaxation.
///
/// Two documented methods are offered:
/// - Wyrowski 1990 (@cite wyrowski1990): quantization as a projection that is
///   introduced stepwise during the iteration (eqs. 14, 15, 22, 23);
/// - Choi et al. 2022 (@cite choi2022): the hard quantizer in the forward pass
///   and the gradient of a Gumbel-Softmax relaxation in the backward pass
///   (eqs. 5-8), with the softmax temperature annealed.
#pragma once

#include "doe/array.hpp"

#include <cstddef>
#include <numbers>
#include <vector>

namespace doe {

/// The @f$Z@f$ phase levels of Wyrowski 1990 eq. (14):
/// @f$\{-\pi, -\pi + \Delta, \dots, \pi - \Delta\}@f$ with @f$\Delta = 2\pi/Z@f$.
struct Levels {
    int q;  ///< number of levels @f$Z \ge 1@f$
    /// Level spacing @f$\Delta = 2\pi / Z@f$.
    double delta() const { return 2.0 * std::numbers::pi / q; }
    /// Level @f$z@f$, @f$z = 0 \dots Z-1@f$.
    double level(int z) const { return -std::numbers::pi + z * delta(); }
};

/// Direct quantizer @f$Q_Z@f$ (Wyrowski eq. 15): every phase is replaced by
/// its nearest level on the circle, i.e. level @f$z@f$ for
/// @f$(z - 0.5)\Delta \le \varphi + \pi < (z + 0.5)\Delta@f$, with @f$z = Z@f$
/// identified with @f$z = 0@f$. Results are wrapped to @f$(-\pi, \pi]@f$
/// (so the level @f$-\pi@f$ is written as @f$\pi@f$). Idempotent.
template <class T>
void project_to_levels(Array2<T>& phi, int q);

/// Stepwise quantizer @f$Q_Z^{(p)}@f$ (Wyrowski eq. 22): a phase is
/// projected onto level @f$z@f$ only if
/// @f$(z - 0.5\,\epsilon)\Delta < \varphi + \pi < (z + 0.5\,\epsilon)\Delta@f$,
/// i.e. within @f$\pm\frac{\epsilon\Delta}{2}@f$ of the level; all other
/// phases are left unchanged. @f$\epsilon = 0@f$ changes nothing,
/// @f$\epsilon = 1@f$ is the direct quantizer.
/// @return the number of phases that were projected.
template <class T>
std::size_t project_stepwise(Array2<T>& phi, int q, double epsilon);

/// The capture fraction @f$\epsilon(p)@f$ of step @f$p = 1 \dots P@f$
/// (Wyrowski eq. 23: @f$0 < \epsilon(1) < \dots < \epsilon(P) = 1@f$).
/// For @f$P = 10@f$ this is the paper's table 0.3, 0.5, 0.6, 0.7, 0.75, 0.8,
/// 0.85, 0.9, 0.95, 1; for other @f$P@f$ the table is interpolated linearly
/// in @f$p/P@f$.
double wyrowski_epsilon(int p, int P);

/// How the capture fraction grows over the quantization stage.
///
/// Škereň, Richter and Fiala 2002 (@cite skeren2002) compared the ways of
/// spending a fixed number of cycles on the widening and found the finest
/// ramp best: a new, slightly larger capture range in every cycle, none of
/// them repeated (their approach I, eq. 11), against holding each range for
/// several cycles (approach III, Wyrowski's arrangement) and against
/// repeating the whole ramp (approach II). Measured here with Adam,
/// the linear ramp lowers the final energy by 1 to 3 % at
/// 2, 4 and 8 levels against the held table.
enum class QuantRamp {
    table,   ///< Wyrowski's table (eq. 23), @f$P@f$ steps each held for several iterations
    linear   ///< Škereň et al. eq. (11): @f$\epsilon(p) = p/P@f$ with one step per iteration of the stage
};

/// Capture fraction of step @f$p = 1 \dots P@f$ for a ramp: wyrowski_epsilon()
/// for QuantRamp::table, @f$p/P@f$ for QuantRamp::linear (Škereň et al. 2002
/// eq. 11, @f$\epsilon(n) = \frac{\pi}{L}\,\frac{n}{N}@f$ as a fraction of the half
/// spacing). Throws unless @f$1 \le p \le P@f$.
double capture_fraction(QuantRamp ramp, int p, int P);

/// Gumbel-Softmax relaxation of the quantizer (Choi et al. 2022 eqs. 6-8).
///
/// @f[
///   \hat q(\varphi) = \sum_{l=1}^{L} Q_l\, G_l(\mathrm{score}(\varphi, Q)), \qquad
///   G_l(z) = \frac{\exp((z_l + g_l)/\tau)}{\sum_{l'} \exp((z_{l'} + g_{l'})/\tau)}, \qquad
///   \mathrm{score}_l = \sigma(w\,\delta(\varphi, Q_l))\,\big(1 - \sigma(w\,\delta(\varphi, Q_l))\big),
/// @f]
/// with @f$g_l \sim \mathrm{Gumbel}(0,1)@f$, @f$\sigma@f$ the logistic
/// sigmoid, @f$\delta@f$ the signed angular difference on the circle and
/// @f$w@f$ a scale factor. The score peaks (at 1/4) on the level itself, so
/// for @f$\tau \to 0@f$ and zero noise @f$\hat q@f$ is the nearest level.
/// The paper uses the exact quantizer in the forward pass and
/// @f$\mathrm d\hat q/\mathrm d\varphi@f$ in the backward pass (eq. 5); the
/// noise "prevent[s] the optimization from getting stuck in local minima" and
/// @f$\tau@f$ is annealed. The derivative is analytic:
/// @f$\mathrm d\hat q/\mathrm d\varphi = \sum_l Q_l \sum_m G_l(\delta_{lm} - G_m)\,\mathrm{score}_m'/\tau@f$
/// with @f$\mathrm{score}_m' = w\,\sigma'(1 - 2\sigma)@f$.
struct GumbelSoftmaxQuantizer {
    int q;             ///< number of levels @f$L@f$
    double w;          ///< score scale @f$w@f$
    double tau;        ///< softmax temperature @f$\tau@f$
    double gain = 1.0; ///< logit gain applied to the score before the softmax (1 = eqs. 6-8 as printed)

    /// @f$\mathrm{score}_l(\varphi)@f$ of eq. (8) for level @f$l@f$.
    double score(double phi, int l) const;
    /// @f$\hat q(\varphi)@f$ of eq. (6) for one phase; `gumbel` points to @f$L@f$ noise values.
    double value(double phi, const double* gumbel) const;
    /// @f$\mathrm d\hat q/\mathrm d\varphi@f$ for one phase.
    double derivative(double phi, const double* gumbel) const;
    /// Draw an (n x L) array of Gumbel(0,1) samples, @f$g = -\log(-\log U)@f$, from a seeded generator.
    Array2<double> sample_noise(std::size_t n, unsigned long long seed) const;
};

/// Surrogate gradient of Choi et al. eq. (5): the gradient with respect to
/// the quantized phase, multiplied elementwise by @f$\mathrm d\hat q/\mathrm d\varphi@f$
/// at the continuous phase. `noise` is (N x L) as produced by sample_noise().
template <class T>
Array2<T> surrogate_gradient(const Array2<T>& grad_q, const Array2<T>& phi, const GumbelSoftmaxQuantizer& Q,
                             const Array2<double>& noise);

/// @cond INTERNAL
extern template void project_to_levels<float>(Array2<float>&, int);
extern template void project_to_levels<double>(Array2<double>&, int);
extern template std::size_t project_stepwise<float>(Array2<float>&, int, double);
extern template std::size_t project_stepwise<double>(Array2<double>&, int, double);
extern template Array2<float> surrogate_gradient<float>(const Array2<float>&, const Array2<float>&,
                                                        const GumbelSoftmaxQuantizer&, const Array2<double>&);
extern template Array2<double> surrogate_gradient<double>(const Array2<double>&, const Array2<double>&,
                                                          const GumbelSoftmaxQuantizer&, const Array2<double>&);
/// @endcond

}  // namespace doe
