# Where does the quantization noise of a 4-level design go, and why does the grating suffer most?
#
# Rounding a phase to Z levels turns exp(i phi) into sinc(1/Z) exp(i phi) plus "false images"
# (Goodman & Silvestri 1970; Wyrowski 1990 Sec. 3.A and eq. 24). This script takes the
# continuous Adam design of each README target (results/readme/<target>/phase_adam.npy),
# rounds it to 4 levels, propagates both, and splits the rounded field into
# sinc(1/4) * (continuous field) + residual. It prints where the residual power lands and the
# signal-to-haze ratio in the bright pixels. Self-contained: numpy and ImageMagick only.
#
#   python3 docs/quantization_noise.py            (after the README runs, see README "Results")
import subprocess, sys, pathlib, numpy as np

root = pathlib.Path(__file__).resolve().parent.parent
Z, a, lam, pitch, d = 4, 512, 532e-9, 8e-6, 0.05
n = 720                                    # padded window of a 512 px design at 50 mm (picture-clear rule, Grid::padded_for)
sinc = np.sin(np.pi / Z) / (np.pi / Z)

# angular spectrum with the Matsushima-Shimobaba band limit, as in src/propagate.cpp
f = np.fft.fftfreq(n, pitch)
fx, fy = np.meshgrid(f, f, indexing='ij')
arg = 1.0 - (lam * fx) ** 2 - (lam * fy) ** 2
H = np.where(arg > 0, np.exp(1j * 2 * np.pi * d / lam * np.sqrt(np.maximum(arg, 0))), 0)
flim = 1.0 / (lam * np.sqrt((2 * d / (n * pitch)) ** 2 + 1))
H = np.where((np.abs(fx) <= flim) & (np.abs(fy) <= flim), H, 0)
forward = lambda u: np.fft.ifft2(H * np.fft.fft2(u))

def read_gray(path):
    pgm = subprocess.run(['magick', str(path), '-colorspace', 'Gray', '-depth', '8', 'pgm:-'], capture_output=True, check=True).stdout
    parts = pgm.split(b'\n', 3); w, h = map(int, parts[1].split())
    return (np.frombuffer(parts[3][:w * h], dtype=np.uint8).reshape(h, w).astype(float) / 255.0).T  # (i = x, j = y)

def embed(x):
    out = np.zeros((n, n), dtype=x.dtype); i0 = (n - x.shape[0]) // 2; j0 = (n - x.shape[1]) // 2
    out[i0:i0 + x.shape[0], j0:j0 + x.shape[1]] = x; return out

def ncc(I, T, m):
    x = I[m] - I[m].mean(); y = T[m] - T[m].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))

def contrast(I, m):
    return float(I[m].std() / I[m].mean())

print(f"{'target':15s} {'bright':>6s} | {'eff':>5s} {'ncc':>5s} {'C':>5s} -> {'eff':>5s} {'ncc':>5s} {'C':>5s} | {'noise/tot':>9s} {'in win':>6s} {'in dark':>7s} | {'S/H bright':>10s} {'2sqrt(H/S)':>10s}")
for name in ['cross_ring', 'spot_array_5x5', 'logo', 'disk', 'grating']:
    phi = np.load(root / 'results/readme' / name / 'phase_adam.npy')
    tgt = read_gray(root / 'images/targets' / f'{name}.png'); tgt /= tgt.max()
    T = embed(tgt); W = embed(np.ones((a, a), bool)); illum = embed(np.ones((a, a)))
    phi_r = -np.pi + np.round((phi + np.pi) / (2 * np.pi / Z)) * (2 * np.pi / Z)
    vc = forward(illum * np.exp(1j * embed(phi))); vr = forward(illum * np.exp(1j * embed(phi_r)))
    Ic, Ir = np.abs(vc) ** 2, np.abs(vr) ** 2
    R = np.abs(vr - sinc * vc) ** 2                       # false images
    dark, bright = W & (T < 0.5), W & (T >= 0.5)
    S = (sinc ** 2 * Ic)[bright].mean(); Hz = R[bright].mean()   # signal and haze per bright pixel
    print(f"{name:15s} {bright.sum() / W.sum():6.3f} | {Ic[W].sum() / Ic.sum():5.3f} {ncc(Ic, T, W):5.3f} {contrast(Ic, bright):5.3f} -> "
          f"{Ir[W].sum() / Ir.sum():5.3f} {ncc(Ir, T, W):5.3f} {contrast(Ir, bright):5.3f} | "
          f"{R.sum() / illum.sum():9.3f} {R[W].sum() / R.sum():6.3f} {R[dark].sum() / R[W].sum():7.3f} | {S / Hz:10.1f} {2 * np.sqrt(Hz / S):10.3f}")
print(f"1 - sinc(1/{Z})^2 = {1 - sinc ** 2:.3f}   (Wyrowski 1990 eq. 24: efficiency of the quantized structure = sinc(1/Z)^2 x analog efficiency)")
