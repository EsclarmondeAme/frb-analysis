import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. "Planck-like" C_ell power spectrum (toy but realistic)
# ------------------------------------------------------------

def planck_like_cl(ell):
    """
    smooth analytic approximation to the Planck TT power spectrum.
    purely a toy model so we can test directional modulation
    without downloading real Planck data.
    """
    ell = np.asarray(ell, dtype=float)

    # envelope (roughly right shape)
    envelope = 1.0 / (1.0 + (ell / 150.0) ** 2)

    # acoustic oscillations
    oscill = 1.0 + 0.25 * np.sin(ell / 55.0) + 0.12 * np.sin(ell / 28.0)

    # high-l damping
    damping = np.exp(-(ell / 1800.0) ** 2)

    return 1000.0 * envelope * oscill * damping


ells = np.arange(30, 801)
Cl_data = planck_like_cl(ells)

# toy 2% fractional errors to mimic Planck
sigma_Cl = 0.02 * Cl_data

# ------------------------------------------------------------
# 2. simple directional modulation model
# ------------------------------------------------------------

def modulated_cl(Cl, ells, A, n):
    """
    directional modulation of the power spectrum:

        C_l -> C_l * (1 + A * (l / 800)^n)

    A : modulation amplitude
    n : harmonic index (1 = dipole, 2 = quadrupole)
    """
    shape = (ells / 800.0) ** n
    return Cl * (1.0 + A * shape)


def chi2(data, model, sigma):
    return np.sum(((data - model) / sigma) ** 2)


A_grid = np.linspace(-0.3, 0.3, 1201)  # enough to explore reasonable amplitudes


def fit_modulation(n):
    chi2_vals = []
    for A in A_grid:
        model = modulated_cl(Cl_data, ells, A, n)
        chi2_vals.append(chi2(Cl_data, model, sigma_Cl))
    chi2_vals = np.array(chi2_vals)

    # best-fit amplitude
    best_idx = np.argmin(chi2_vals)
    A_best = A_grid[best_idx]
    chi2_min = chi2_vals[best_idx]
    dchi2 = chi2_vals - chi2_min

    # estimate 1-sigma error from Δχ² = 1 on each side
    # (if grid doesn’t reach that, we mark it as unknown)
    def find_side(start, step):
        i = start
        while 0 <= i < len(A_grid):
            if dchi2[i] >= 1.0:
                return abs(A_grid[i] - A_best)
            i += step
        return np.nan

    sigma_plus = find_side(best_idx + 1, +1)
    sigma_minus = find_side(best_idx - 1, -1)

    # average the two sides if both exist
    if np.isfinite(sigma_plus) and np.isfinite(sigma_minus):
        sigma_A = 0.5 * (sigma_plus + sigma_minus)
    elif np.isfinite(sigma_plus):
        sigma_A = sigma_plus
    elif np.isfinite(sigma_minus):
        sigma_A = sigma_minus
    else:
        sigma_A = np.nan

    if np.isfinite(sigma_A) and sigma_A > 0:
        significance = abs(A_best) / sigma_A
    else:
        significance = np.nan

    return A_best, sigma_A, significance, dchi2


# ------------------------------------------------------------
# 3. run fits for dipole (n=1) and quadrupole (n=2)
# ------------------------------------------------------------

A1_best, sigA1, sig1, dchi2_1 = fit_modulation(1)
A2_best, sigA2, sig2, dchi2_2 = fit_modulation(2)

print("============================================================")
print("CMB high-ℓ directional modulation test (toy Planck spectrum)")
print("multipole range: 30 ≤ l ≤ 800")
print("============================================================")
print(f"dipole modulation (n=1):")
print(f"  A1 = {A1_best:+.4f} ± {sigA1:.4f}   → significance ≈ {sig1:.2f} σ")
print("------------------------------------------------------------")
print(f"quadrupole modulation (n=2):")
print(f"  A2 = {A2_best:+.4f} ± {sigA2:.4f}   → significance ≈ {sig2:.2f} σ")
print("============================================================")

# ------------------------------------------------------------
# 4. plot likelihood curves
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(A_grid, dchi2_1, label="dipole (n=1) Δχ²")
plt.plot(A_grid, dchi2_2, label="quadrupole (n=2) Δχ²")
plt.axhline(1.0, linestyle="--", linewidth=1, label="Δχ² = 1 (1σ)")
plt.axvline(A1_best, color="C0", linestyle=":", linewidth=1)
plt.axvline(A2_best, color="C1", linestyle=":", linewidth=1)
plt.xlabel("modulation amplitude A")
plt.ylabel("Δχ² relative to best fit")
plt.title("CMB directional modulation likelihood (toy model)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cmb_modulation_likelihood.png", dpi=150)
print("[info] saved plot → cmb_modulation_likelihood.png")
print("[info] done.")
