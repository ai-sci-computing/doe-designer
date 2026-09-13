/// @file energy.hpp
/// @brief Design energy of a phase-only DOE and its Wirtinger gradient.
///
/// Notation: @f$u = \mathrm{illum}\,e^{i\varphi}@f$ is the
/// field leaving the DOE, @f$v = A u@f$ the field at the target plane,
/// @f$f = |v|@f$ its amplitude, @f$b = \sqrt{I_{\text{target}}}@f$ the
/// target amplitude and @f$W@f$ (a 0/1 mask) the signal window. With
/// @f[
///   p = \sum_W b f, \qquad c = \sum_W b^2, \qquad s_2 = \sum_W f^2,
///   \qquad E_{\text{in}} = \sum |\mathrm{illum}|^2
/// @f]
/// the energy is
/// @f[
///   E(\varphi) = w_s\Big(1 - \frac{p^2}{c\, s_2}\Big) + w_e\Big(1 - \frac{s_2}{E_{\text{in}}}\Big).
/// @f]
///
/// **Shape term.** It is the normalized mean-square error between @f$f@f$
/// and @f$b@f$ on the window, minimized over a real multiplicative constant:
/// Fienup 1997 @cite fienup1997 eq. (20),
/// @f$ \min_{a_R} E^2 = 1 - (\mathrm{Re}\, r_{fg})^2 / (r_{gg}(0,0)\, r_{ff}(0,0)) @f$,
/// with the cross-correlation at zero shift @f$r_{fg} = p@f$ and the
/// auto-correlations @f$r_{ff} = c@f$, @f$r_{gg} = s_2@f$ (no translation
/// search: the target is registered). It is the squared cosine between
/// @f$f@f$ and @f$b@f$, invariant under @f$f \to a f@f$.
///
/// Why not the scale-matched residual @f$\sum_W (f - s b)^2@f$ with the
/// least-squares @f$s = p/c@f$: substituting @f$s@f$ gives @f$s_2 - p^2/c@f$,
/// which is minimized by driving @f$f \to 0@f$ on the window (all energy
/// into the free "don't care" region, @f$s \to 0@f$, residual @f$\to 0@f$).
/// Measured in the reference implementation: efficiency 0.25 at a random
/// start, 1.3e-5 after optimizing that form, while its loss curve looked
/// like healthy convergence. Dividing by @f$s_2@f$ removes
/// the degeneracy, and T13 guards it.
///
/// **Efficiency term.** The shape term is blind to magnitude, so the
/// diffraction efficiency (energy in the window over total energy,
/// Wyrowski & Bryngdahl 1988 @cite wyrowski1988, Wyrowski 1990
/// @cite wyrowski1990) gets its own term. @f$A@f$ is a partial isometry
/// and @f$|u|@f$ is fixed, so the total energy is (essentially) constant and
/// maximizing the window energy means maximizing @f$s_2@f$; @f$w_e = \mu@f$
/// is the fidelity/efficiency trade-off (Ripoll et al. 2004 @cite ripoll2004).
///
/// **Gradient** (Wirtinger / CR calculus @cite kreutzdelgado2009;
/// Chakravarthula et al. 2019 @cite chakravarthula2019 eqs. 10-14). With
/// @f$w_m = W_m@f$,
/// @f[
///   \frac{\partial E}{\partial f_m}
///     = \frac{2 p\, w_m}{c\, s_2^2}\big(p f_m - b_m s_2\big)
///       - \frac{2\mu\, w_m f_m}{E_{\text{in}}}, \qquad
///   \frac{\partial E}{\partial \bar v} = \frac{\partial E}{\partial f}\,\frac{v}{2|v|}, \qquad
///   G = A^H \frac{\partial E}{\partial \bar v}, \qquad
///   \frac{\partial E}{\partial \varphi} = 2\,\mathrm{Im}\big(\bar u \odot G\big).
/// @f]
/// The @f$\tfrac12@f$ is @f$\partial |v| / \partial \bar v@f$ and is not
/// canceled, because @f$\partial E/\partial f@f$ already carries the full
/// @f$f@f$-derivative. Working with the amplitude rather
/// than the intensity converges markedly better: the intensity form has a
/// vanishing gradient in dark regions.
///
/// All reductions accumulate in double regardless of T.
#pragma once

#include "doe/array.hpp"
#include "doe/propagate.hpp"

namespace doe {

/// Weights of the two energy terms. `shape` is normally 1; `efficiency` is
/// the trade-off @f$\mu@f$ (0.3 by default). Either can be set to
/// zero to test a term in isolation (T4 runs each term alone).
struct EnergyWeights {
    double shape = 1.0;       ///< weight of the Fienup shape term
    double efficiency = 0.3;  ///< weight @f$\mu@f$ of the efficiency term
};

/// Least-squares scale @f$ s = \sum_W b\,|v| \big/ \sum_W b^2 @f$ matching
/// @f$|v|@f$ to @f$b@f$ on the window (the real constant @f$a_R@f$ of
/// Fienup 1997 @cite fienup1997; the scale factor @f$s@f$ of Peng et al.
/// 2020 @cite peng2020 eq. 2). Returns 1 when the window is empty.
template <class T>
double optimal_scale(const Array2<T>& v_abs, const Array2<T>& b, const Array2<T>& mask);

/// Value, terms and gradient of the energy at one phase.
template <class T>
struct EnergyResult {
    double energy;       ///< @f$E@f$ = shape + efficiency
    double shape;        ///< @f$w_s (1 - p^2/(c s_2))@f$
    double efficiency;   ///< @f$w_e (1 - s_2/E_{\text{in}})@f$
    double scale;        ///< least-squares scale @f$p/c@f$ (reported, not used in E)
    Array2<T> grad;      ///< @f$\partial E/\partial\varphi@f$
    Field<T> field;      ///< @f$v = A(\mathrm{illum}\,e^{i\varphi})@f$, reused by callers
};

/// Evaluate the energy and its gradient with respect to the phase.
/// @param phi   DOE phase (radians), padded grid
/// @param illum illumination amplitude on the DOE (zero in the padding)
/// @param prop  propagator DOE plane -> target plane
/// @param b     target amplitude @f$\sqrt{I_{\text{target}}}@f$ on the padded grid
/// @param mask  signal window (1 inside, 0 in the free region)
/// @param w     term weights
template <class T>
EnergyResult<T> energy_and_grad(const Array2<T>& phi, const Array2<T>& illum, const AngularSpectrum<T>& prop,
                                const Array2<T>& b, const Array2<T>& mask, EnergyWeights w = {});

/// @cond INTERNAL
extern template double optimal_scale<float>(const Array2<float>&, const Array2<float>&, const Array2<float>&);
extern template double optimal_scale<double>(const Array2<double>&, const Array2<double>&, const Array2<double>&);
extern template EnergyResult<float> energy_and_grad<float>(const Array2<float>&, const Array2<float>&,
                                                           const AngularSpectrum<float>&, const Array2<float>&,
                                                           const Array2<float>&, EnergyWeights);
extern template EnergyResult<double> energy_and_grad<double>(const Array2<double>&, const Array2<double>&,
                                                             const AngularSpectrum<double>&, const Array2<double>&,
                                                             const Array2<double>&, EnergyWeights);
/// @endcond

}  // namespace doe
