import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# load neutrino data
# ------------------------------------------------------------

nu = pd.read_csv("neutrinos_clean.csv")

if "mjd" not in nu.columns:
    raise ValueError("neutrinos_clean.csv must contain MJD column")

# drop invalid rows
nu = nu.dropna(subset=["mjd"])
mjd = nu["mjd"].values
N = len(mjd)

print(f"neutrinos with valid MJD: {N}")

# ------------------------------------------------------------
# compute sidereal phase
# ------------------------------------------------------------

# integer part = day, fractional part * 1.002737909 = sidereal day fraction
mjd_frac = mjd % 1.0
sidereal_phase = (mjd_frac * 1.002737909) % 1.0

# ------------------------------------------------------------
# compute harmonics
# ------------------------------------------------------------

def harmonic_coeff(phi, n):
    A = np.mean(np.cos(2 * np.pi * n * phi))
    B = np.mean(np.sin(2 * np.pi * n * phi))
    return A, B, np.sqrt(A*A + B*B)

print("------------------------------------------------------------")
print("neutrino sidereal harmonics (n = 1..4)")
print("------------------------------------------------------------")

harm = []
for n in range(1, 5):
    A, B, R = harmonic_coeff(sidereal_phase, n)
    harm.append((n, A, B, R))
    print(f" n={n}:  A_n={A:+.4f}  B_n={B:+.4f}  R_n={R:.4f}")

# ------------------------------------------------------------
# monte carlo null test
# ------------------------------------------------------------

print("------------------------------------------------------------")
print("monte carlo test (uniform phases)")
print("------------------------------------------------------------")

rng = np.random.default_rng(12345)
MC = 20000

mc_R = np.zeros((4, MC))

for k in range(MC):
    rand_phi = rng.random(N)
    for i, n in enumerate([1,2,3,4]):
        A = np.mean(np.cos(2 * np.pi * n * rand_phi))
        B = np.mean(np.sin(2 * np.pi * n * rand_phi))
        mc_R[i, k] = np.sqrt(A*A + B*B)

for i, (n, A, B, R) in enumerate(harm):
    p = np.mean(mc_R[i] >= R)
    print(f" n={n}:  p(R_rand >= R_obs) = {p:.4f}")

# ------------------------------------------------------------
# plot histogram
# ------------------------------------------------------------

plt.figure(figsize=(10,5))
plt.hist(sidereal_phase, bins=30, density=True, alpha=0.5)
plt.xlabel("sidereal phase")
plt.ylabel("probability density")
plt.title("neutrino sidereal phase histogram")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("neutrino_sidereal_phase_histogram.png", dpi=150)
print("[info] saved → neutrino_sidereal_phase_histogram.png")
print("[done]")
