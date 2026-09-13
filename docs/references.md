# Traceability: code → published source

Every numerical routine cites the equation it implements. This table is the index;
the Doxygen comment of each symbol carries the same `@cite` key and equation number,
and the last column names the test that checks it against an analytic solution,
an operator identity, or frozen values of the reference implementation.
Keys refer to `references.bib`.

The JavaScript port in `viewer-web/core/` (web viewer) implements the same routines with the
same sources; instead of its own analytic tests for every line it is cross-checked against the
C++ binary in `viewer-web/test/core_pipeline.test.js` (phases, metrics, planes and vortex
charges of a `doe_design` run), plus the analytic tests of `core_propagate.test.js` and
`core_energy.test.js` (FFT against a naive DFT, Goodman eq. 3-78 bin by bin, Matsushima band
limit, adjointness, finite-difference gradient, Poisson manufactured solution, analytic vortices).

| Symbol | Source | Equation / section | Verified by |
|---|---|---|---|
| `doe::next_smooth` | fftw_manual; matsushima2009 §2.2 (zero padding) | "small primes (2, 3, 5, and 7)" | `test_next_smooth` (T11, brute force ≤ 5000) |
| `doe::Fft2` | numpy `norm="ortho"` convention, FFTW sign convention | unitary scaling 1/n per direction | `test_fft` (Parseval, delta/constant, phasor bin, float vs double) |
| `doe::Grid` | goodman2017 eq. 3-66 (direction cosines ↔ frequencies) | `x_i = (i − n/2) p`, `fftfreq` order, `asin(λ/2p)` | `test_grid` |
| `doe::Grid::padded_for` | matsushima2009 §2.2 (zero padding of the periodic window); picture-clear rule `S ≥ N p + d tan θmax` (wrapped light misses the picture) | `n = next_smooth(max(pad·N, (N p + d tan θ)/p))`, 720 px at the defaults | `test_grid`, `test_pipeline` "prepare" |
| `doe::transfer_function` | goodman2017 eqs. 3-71, 3-72, 3-77, 3-78; matsushima2009 eqs. 11–14, §2.2 | `H = exp[i 2π (z/λ) √(1 − (λfx)² − (λfy)²)]`, evanescent → 0, `f_lim = 1/(λ√((2z/S)²+1))`, rect window | `test_propagate` "Goodman eq. 3-78 bin by bin", "band limit is Matsushima & Shimobaba eq. 13", "built in double" |
| `doe::AngularSpectrum::forward/adjoint` | kreutzdelgado2009 (adjoint of a unitary-conjugated multiplier) | `A^H = F⁻¹ H̄ F` | T1, T2, T12 (adjointness, isometry), oracle parity with `optics.py` |
| `doe::AngularSpectrum::volume` | — (shares one forward FFT) | | T10 (≤ 1e-14 vs `forward` per plane) |
| zero padding factor 2 | matsushima2009 §2.2 | | T7 (border ring, wrapped energy vs 4× padding) |
| analytic: tilted beam | goodman2017 eq. 3-66 | centroid shift `d tan(asin λ f0)` | `test_propagate` "tilted beam" (1e-4) |
| analytic: Talbot revival | montgomery1967 (self-imaging condition `(k − k_z) z = 2π`), rayleigh1881 (paraxial `2p²/λ`), kyvalsky2003 | `z_T = 2π/(k − k_z(f0))` | `test_propagate` "Talbot" (1e-9, full and half distance) |
| analytic: Gaussian beam | siegman1986 ch. 17 | `w(z) = w0 √(1 + (z/zR)²)`, `zR = π w0²/λ` | T3 (1e-3, spec allows 1 %) |
| analytic: circular aperture on axis | bornwolf1999 §8.3; wyant_fresnel eq. 5 | `U/U0 = e^{ikz} − (z/R) e^{ikR}`, `R = √(z²+a²)` | `test_propagate` "on-axis intensity" (2 %) |
| `doe::sampling_report` | — (sampling diagnostics) | `λd/D`, `2d tanθ`, `D²/(λd)` | `test_propagate` "sampling_report" |
| `doe::optimal_scale` | fienup1997 (real constant @f$a_R@f$); peng2020 eq. 2 (scale @f$s@f$); wyrowski1990 eq. 9 (@f$c_j@f$) | `s = Σ_W b|v| / Σ_W b²` | `test_energy` "optimal_scale", oracle |
| `doe::energy_and_grad` shape term | fienup1997 eq. 20 | `1 − (Re r_fg)² / (r_gg r_ff)` | `test_energy` invariance, T4 shape alone, oracle |
| `doe::energy_and_grad` efficiency term | wyrowski1988, wyrowski1990, ripoll2004 | `μ (1 − s₂/E_in)` | T4 efficiency alone |
| `doe::energy_and_grad` gradient | kreutzdelgado2009; chakravarthula2019 eqs. 10–14 | `∂E/∂v̄ = ∂E/∂f · v/(2|v|)`, `G = A^H ∂E/∂v̄`, `∂E/∂φ = 2 Im(ū G)` | T4 sum, h-sweep, oracle gradient samples |
| `doe::metrics` | wyrowski1988 (efficiency); goodman2007 ch. 2–3, aagedal1996 (speckle contrast) | | `test_metrics` analytic cases, oracle |
| `doe::poisson_periodic` | — (spectral inversion of the periodic Laplacian) | `φ̂ = r̂ / (−4π²|f|²)`, DC = 0 | T6 manufactured solution (rounding level) |
| `doe::init_tie` | teague1983 eq. 4 (sign for the `e^{+ikz}` carrier of his eq. 1); paganin1998; zuo2020 | `∇²φ = (k/d)(1 − I_t/I₀)` | `test_init` focusing check with the propagator |
| `doe::Adam` | kingma2015 Algorithm 1 | moment updates, bias correction, defaults 0.9 / 0.999 / 1e-8 | `test_solvers` hand-computed sequence, quadratic |
| `doe::gerchberg_saxton` | gerchberg1972; wyrowski1990 §2 eqs. 7–10 (X, U, c_j, X'); fienup1980 (free region); fienup1982 (error reduction) | two stages, least-squares `c_j` | fixed point, per-stage monotone residual, efficiency-slide measurements |
| `doe::optimize` | kingma2015 (Adam on φ, wrap to the torus) | | energy falls, efficiency guard (T13 at unit scale) |
| `doe::Levels`, `project_to_levels` | wyrowski1990 eqs. 14, 15 | `{−π + zΔ}`, nearest level | `test_quantize` |
| `doe::project_stepwise`, `wyrowski_epsilon` | wyrowski1990 eqs. 22, 23 + his ε table for P = 10; §3.C schedule `J = Q(P−1)+1` | capture `|φ − level| < ε Δ/2` | `test_quantize`, `test_quantized_solvers` (T5, T9) |
| `doe::QuantRamp`, `capture_fraction` | skeren2002 eq. 11 (linear ramp ε = p/P, one step per iteration, their approach I; default); wyrowski1990 eq. 23 (held table) | `ε(p) = p/P` or the table | `test_quantize`, `test_quantized_solvers` (T5, T9) |
| `doe::GumbelSoftmaxQuantizer`, `surrogate_gradient` | choi2022 eqs. 5–8 | forward hard `q`, backward `dq̂/dφ` of the Gumbel-Softmax relaxation | FD ratio 1 ± 1e-6, T9 with Choi |
| `doe::read_png/write_png`, `to_gray` | libpng; Rec. 601 luma | | `test_image` round trips |
| `doe::colormap_twilight/viridis` | matplotlib LUTs (sampled once, `src/colormap_lut.hpp`) | | endpoints pinned |
| `doe::hsv_phasor`, `render_hsv` | — (hue = phase, value = amplitude) | | known phasors |
| `doe::write_npy/read_npy` | NumPy `.npy` format v1.0 | | round trip; loaded with `numpy.load` (dev check) |
| `doe::targets::spot_array` | dammann1971, krackhardt1989 (fan-out targets) | | `test_targets` |
| `doe::vortex_charge_map` | goldstein1988 (residue: wrapped differences around four adjacent points, 0 / ±1 cycle); fried1992 (branch points of the phase, eq. 1); berry2000 eq. 2.4 (charge sign) | plaquette sum / 2π | `test_vortex` analytic vortices ±1, winding 2, smooth field 0 |
| `doe::vortex_density` | berry2000 eq. 4.6: `d₂ = K₂/(4π)` for isotropic random waves | | `test_vortex` ring-spectrum speckle, 10 % |
| `doe::AngularSpectrum::sweep`, `sweep_volume` | matsushima2009 (same transfer_function per plane) | | `test_render` sweep == volume (T10), plane 0 = input, last plane = reconstruction |
| figures 1–4, cuts, vortex-density plot | — | | `test_render` files and sizes |
| `doe::init_backprop` | gerchberg1972, fienup1980 (start from the back-propagated target); wyrowski1990 eq. 13 (target with a random phase) | `φ0 = arg(A^H b)` (flat target phase) | `test_init` "init_backprop" |
| `doe::init_random` | wyrowski1990 eq. 13 (random start phase) | uniform in `(−π, π]`, seeded | `test_init` "init_random" |
| `doe::resample`, `embed`, `crop_center` | — (bilinear resampling; zero embedding in the padded window) | | `test_image` "resample", "embed", "crop_center" |
| `doe::targets::cross_plus_ring`, `disk`, `grating`, `binary_logo`, `soft_edges` | — (synthetic test targets; built-in 5×7 font) | | `test_targets` |
| `doe::render_phase`, `render_intensity` | — (cyclic map for phase; viridis, optional log scale with a dB floor) | | `test_image` "render_phase / render_intensity / render_hsv" |
| `doe::render_vortex_map` | goldstein1988 residues drawn on the intensity (red +1, blue −1) | | `test_render` "sweep_volume", `test_cli` "write_outputs" |
| `doe::write_doev`, `read_doev`, `attach_source`, `sweep_from_source` | — (`.doev` v1/v2 layout) | | `test_render` "doev" round trips, v2 re-sweep |
| `doe::slice_texture`, face and wall textures (`viewer_core`) | — (same HSV and log-intensity mappings as the figures) | | `test_viewer_core` |
| `doe::SvgPlot` (convergence, vortex density) | — | total energy per run; shape and efficiency terms as separate curves | `test_io` "svg_line_plot", `test_cli` "write_outputs" (history.csv columns) |
| quantization-noise analysis (`docs/quantization_noise.py`, README "Grating") | goodman1970 (false images of phase quantization); wyrowski1990 §3.A, eq. 24 | `e^{iφ} → sinc(1/Z) e^{iφ} + q`, `η_Z = sinc²(1/Z) η` | measured 0.189 = 1 − sinc²(1/4) on all five designs |
