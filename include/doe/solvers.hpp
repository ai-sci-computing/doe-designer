/// @file solvers.hpp
/// @brief Phase retrieval solvers: Adam on the energy, Gerchberg–Saxton baseline.
#pragma once

#include "doe/array.hpp"
#include "doe/energy.hpp"
#include "doe/propagate.hpp"
#include "doe/quantize.hpp"

#include <functional>
#include <vector>

namespace doe {

/// Hyper-parameters of Adam (Kingma & Ba 2015 @cite kingma2015, Algorithm 1;
/// their defaults are @f$\beta_1 = 0.9@f$, @f$\beta_2 = 0.999@f$,
/// @f$\epsilon = 10^{-8}@f$; the step size is problem specific).
struct AdamParams {
    double lr = 0.05;      ///< step size @f$\alpha@f$
    double beta1 = 0.9;    ///< decay of the first moment
    double beta2 = 0.999;  ///< decay of the second raw moment
    double eps = 1e-8;     ///< @f$\epsilon@f$ in the denominator
};

/// Adam optimizer state for one array of parameters.
///
/// Implements Algorithm 1 of @cite kingma2015 verbatim:
/// @f[
///   m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad
///   v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad
///   \hat m_t = m_t/(1-\beta_1^t),\quad \hat v_t = v_t/(1-\beta_2^t),\quad
///   \theta_t = \theta_{t-1} - \alpha\, \hat m_t / (\sqrt{\hat v_t} + \epsilon).
/// @f]
/// Moments are kept in double regardless of T.
template <class T>
class Adam {
public:
    /// State for a rows x cols parameter array.
    Adam(std::size_t rows, std::size_t cols, AdamParams params = {});
    /// One update of `x` with gradient `g` (in place).
    void step(Array2<T>& x, const Array2<T>& g);
    /// Number of steps taken so far (@f$t@f$).
    int iteration() const { return t_; }

private:
    AdamParams p_;
    Array2<double> m_, v_;
    int t_ = 0;
};

/// Result of a solver run.
template <class T>
struct Result {
    Array2<T> phi;                          ///< final DOE phase, wrapped to @f$(-\pi, \pi]@f$
    Field<T> field;                         ///< target-plane field of the final phase
    std::vector<double> history;            ///< normalized energy per iteration (comparable across solvers)
    std::vector<double> shape_history;      ///< the shape term of `history` (Fienup 1997 eq. 20), same length
    std::vector<double> efficiency_history; ///< the efficiency term of `history`, @f$\mu(1 - s_2/E_{\text{in}})@f$, same length
    std::vector<double> residual_history;   ///< GS only: @f$\sum_W (|v| - s b)^2@f$ per iteration
};

/// Which documented quantization method a solver uses.
enum class QuantMethod {
    wyrowski,  ///< stepwise projection @f$Q_Z^{(p)}@f$, Wyrowski 1990 eqs. 22-23 (GS and Adam)
    choi       ///< hard quantizer forward, Gumbel-Softmax surrogate gradient backward, Choi et al. 2022 eqs. 5-8 (Adam only)
};

/// Quantization settings shared by both solvers. `levels == 0` means continuous.
struct QuantParams {
    int levels = 0;                               ///< number of phase levels @f$Z@f$ (0: no quantization)
    QuantMethod method = QuantMethod::wyrowski;   ///< method (GS always uses Wyrowski's operator)
    double start = 0.4;                           ///< Adam: fraction of the budget before quantization starts (spec §8: ~40 %)
    QuantRamp ramp = QuantRamp::linear;           ///< capture ramp: linear (Škereň et al. 2002, one step per iteration) or Wyrowski's held table
    int steps = 10;                               ///< table ramp: Wyrowski's @f$P@f$ (his example uses 10)
    int cycles_per_step = 5;                      ///< GS: Wyrowski's @f$Q@f$ cycles per step, @f$J = Q(P-1)+1@f$ quantization cycles for both ramps
    double choi_w = 0.0;                          ///< Choi: score width @f$w@f$; 0 = 4 / level spacing (S2.3: "tuned considering the number of phase levels")
    double choi_gain0 = 300.0;                    ///< Choi: score scale at the start of the stage (S2.3: "increased from 300 to 1,000")
    double choi_gain1 = 1000.0;                   ///< Choi: score scale at the end of the stage
    double choi_tau0 = 4.0;                       ///< Choi: initial softmax temperature (S2.3: @f$\tau_0 \sim 4@f$)
    double choi_c = 0.6931471805599453;           ///< Choi: decay in @f$\tau = \tau_0 e^{-c\, t/t_{max}}@f$ (S2.3: @f$c \sim \ln 2@f$)
    unsigned long long choi_seed = 0;             ///< Choi: seed of the Gumbel noise
};

/// Parameters of the iterative Fourier-transform (Gerchberg–Saxton) baseline.
struct GsParams {
    int iters = 40;              ///< analog cycles (Wyrowski: "approximately 10-20 iteration cycles in both steps")
    int phase_only_iters = 20;   ///< cycles with Wyrowski's operator X (field zeroed outside the window) before switching to X'
    QuantParams quant{};         ///< optional quantization stage appended after the analog cycles
    EnergyWeights weights{};     ///< weights of the logged energy history only (GS itself uses projections, not the energy)
};

/// Iterative Fourier-transform algorithm with amplitude freedom outside the
/// signal window (Gerchberg & Saxton 1972 @cite gerchberg1972; Wyrowski 1990
/// @cite wyrowski1990 §2; Wyrowski & Bryngdahl 1988 @cite wyrowski1988).
///
/// Each cycle alternates two projections. At the target plane
/// (Wyrowski eqs. 7, 9, 10): keep the phase of @f$v@f$, replace its modulus
/// on the window @f$W@f$ by @f$c_j\,b@f$ with the least-squares factor
/// @f[ c_j = rac{\sum_W b\,|v_j|}{\sum_W b^2} \qquad 	ext{(eq. 9)} @f]
/// and, outside the window, either zero it (operator @f$X@f$, eq. 7:
/// "exclusive utilization of the phase freedom") or leave it untouched
/// (operator @f$X^{\prime}@f$, eq. 10: amplitude freedom, Fienup's "don't care"
/// region @cite fienup1980). At the DOE plane keep only the phase
/// (operator @f$U@f$, eq. 8: unit modulus).
///
/// Wyrowski's procedure runs @f$X@f$ ‘in the first iteration cycles until
/// the exclusive utilization of the phase freedom is nearly depleted; i.e.,
/// the iteration stagnates’ and then @f$X^{\prime}@f$, ‘approximately 10-20 iteration
/// cycles in both steps’. The first stage is what drives energy into the
/// window. Under @f$X^{\prime}@f$ the least-squares factor follows the window field,
/// so every further cycle trades efficiency for fidelity without bound
/// (measured on the 512 px disk target at the defaults: 0.99 after stage 1,
/// then 0.93 / 0.89 / 0.86 / 0.81 after 10 / 20 / 30 / 60 cycles of @f$X^{\prime}@f$,
/// while the shape term falls from 0.041 to 0.014 and then slowly; at unit-test
/// scale with the band limit off the slide is steeper, 0.95 to 0.23 in 30 cycles). Wyrowski's 10-20 cycles per stage bound that slide
/// and are the default. With a unitary propagator (band limit off) each stage
/// is Fienup's error-reduction algorithm @cite fienup1982, whose projection
/// residual (distance to that stage's constraint set) never increases.
///
/// **Quantization** (Wyrowski §3.C, eqs. 22-23): after the analog cycles the
/// unit-modulus operator is replaced by the stepwise quantizer
/// @f$Q_Z^{(p)}@f$ (project_stepwise() with @f$\epsilon(p)@f$ from
/// wyrowski_epsilon()) for @f$P@f$ steps of @f$Q@f$ cycles each, the last
/// step being a single cycle with the direct quantizer: @f$J = Q(P-1)+1@f$
/// quantization cycles, all under @f$X^{\prime}@f$.
///
/// It is block coordinate descent on the same energy the gradient solver
/// minimizes, which makes it an honest baseline rather than a straw man.
template <class T>
Result<T> gerchberg_saxton(const Array2<T>& phi0, const Array2<T>& illum, const AngularSpectrum<T>& prop,
                           const Array2<T>& b, const Array2<T>& mask, const GsParams& params);

/// Parameters of the gradient solver.
struct OptimizeParams {
    int iters = 400;           ///< iteration budget
    double lr = 0.05;          ///< Adam step size
    EnergyWeights weights{};   ///< shape / efficiency weights
    AdamParams adam{};         ///< remaining Adam hyper-parameters (lr is taken from `lr`)
    QuantParams quant{};       ///< optional quantization by continuation after `quant.start` of the budget
};

/// Progress callback: (iteration, energy) -> continue?
using ProgressFn = std::function<bool(int, double)>;

/// Adam directly on the phase. After each step the phase is
/// wrapped, @f$\varphi \mapsto \arg e^{i\varphi}@f$, to stay on the torus.
/// The energy history is the value of energy_and_grad() at each iterate.
///
/// **Quantization by continuation** starts after `quant.start` of the budget
/// (the design first finds a good continuous solution, then is squeezed onto
/// the levels while the data term can still react):
/// - Wyrowski: the remaining iterations are split into @f$P@f$ steps; after
///   every Adam step the phase is passed through @f$Q_Z^{(p)}@f$ (projected
///   gradient step), and the final phase through the direct quantizer.
/// - Choi: the energy and its gradient are evaluated at the hard-quantized
///   phase, the gradient is multiplied by @f$\mathrm d\hat q/\mathrm d\varphi@f$
///   of the Gumbel-Softmax relaxation (fresh noise every iteration; score scale
///   ramped linearly from `choi_gain0` to `choi_gain1` and temperature
///   @f$\tau = \tau_0 e^{-c\,t/t_{max}}@f$ over the stage, Supplement S2.3 of
///   @cite choi2022), Adam updates the continuous phase, and the final phase
///   is hard-quantized.
template <class T>
Result<T> optimize(const Array2<T>& phi0, const Array2<T>& illum, const AngularSpectrum<T>& prop,
                   const Array2<T>& b, const Array2<T>& mask, const OptimizeParams& params,
                   const ProgressFn& progress = {});

/// Wrap every element to @f$(-\pi, \pi]@f$.
template <class T>
void wrap_phase(Array2<T>& phi);

/// @cond INTERNAL
extern template class Adam<float>;
extern template class Adam<double>;
extern template void wrap_phase<float>(Array2<float>&);
extern template void wrap_phase<double>(Array2<double>&);
extern template Result<float> gerchberg_saxton<float>(const Array2<float>&, const Array2<float>&, const AngularSpectrum<float>&,
                                                      const Array2<float>&, const Array2<float>&, const GsParams&);
extern template Result<double> gerchberg_saxton<double>(const Array2<double>&, const Array2<double>&, const AngularSpectrum<double>&,
                                                        const Array2<double>&, const Array2<double>&, const GsParams&);
extern template Result<float> optimize<float>(const Array2<float>&, const Array2<float>&, const AngularSpectrum<float>&,
                                              const Array2<float>&, const Array2<float>&, const OptimizeParams&, const ProgressFn&);
extern template Result<double> optimize<double>(const Array2<double>&, const Array2<double>&, const AngularSpectrum<double>&,
                                                const Array2<double>&, const Array2<double>&, const OptimizeParams&, const ProgressFn&);
/// @endcond

}  // namespace doe
