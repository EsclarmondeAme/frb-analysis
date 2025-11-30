import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# load data
# ------------------------------------------------------------
df = pd.read_csv("frbs.csv")

if "mjd" not in df.columns:
    raise ValueError("frbs.csv must contain an 'mjd' column")

mjd = df["mjd"].dropna().values
N = len(mjd)
if N == 0:
    raise ValueError("no valid MJD values found")

# ------------------------------------------------------------
# compute sidereal phase
# sidereal day = 86164.0905 s = 0.99726958 days
# phase = (MJD % sidereal_period) / sidereal_period
# ------------------------------------------------------------
sidereal_day = 0.99726958
phase = ((mjd / sidereal_day) % 1.0)

# ------------------------------------------------------------
# compute Fourier harmonics 1..4
# R_n = sqrt(A_n^2 + B_n^2)
# ------------------------------------------------------------
harmonics = []
n_max = 4

for n in range(1, n_max + 1):
    A_n = (2.0 / N) * np.sum(np.cos(2 * np.pi * n * phase))
    B_n = (2.0 / N) * np.sum(np.sin(2 * np.pi * n * phase))
    R_n = np.sqrt(A_n**2 + B_n**2)
    harmonics.append((n, A_n, B_n, R_n))

# ------------------------------------------------------------
# reconstruct modulation curve using Fourier series
# f(phi) = 1 + Σ (A_n cos 2πnφ + B_n sin 2πnφ)
# ------------------------------------------------------------
phi_grid = np.linspace(0, 1, 1000)
curve = np.ones_like(phi_grid)

for n, A_n, B_n, _ in harmonics:
    curve += A_n * np.cos(2 * np.pi * n * phi_grid) + \
             B_n * np.sin(2 * np.pi * n * phi_grid)

# ------------------------------------------------------------
# plot histogram + harmonic-reconstructed curve
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))

# histogram of real phases
plt.hist(phase, bins=40, density=True, alpha=0.45, label="FRB sidereal phase histogram")

# overlay harmonic reconstruction
plt.plot(phi_grid, curve / np.trapz(curve, phi_grid),   # normalize
         linewidth=2.2, label="harmonic reconstruction (n=1..4)")

plt.xlabel("sidereal phase")
plt.ylabel("probability density")
plt.title("FRB sidereal-phase modulation (harmonic reconstruction)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("frb_sidereal_phase_modulation.png", dpi=180)
print("saved plot → frb_sidereal_phase_modulation.png")
