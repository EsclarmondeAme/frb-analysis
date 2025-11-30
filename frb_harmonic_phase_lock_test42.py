import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm
import sys

# ------------------------------------------------------------
# harmonic model
# ------------------------------------------------------------
def harmonic(phi, C, A1, ph1, A2, ph2):
    return C * (1 + A1*np.cos(phi - ph1) + A2*np.cos(2*phi - ph2))

# ------------------------------------------------------------
def fit_phases(phi, weights=None):
    if weights is None:
        weights = np.ones_like(phi)

    hist, edges = np.histogram(phi, bins=36, weights=weights)
    centers = 0.5*(edges[1:] + edges[:-1])

    # avoid zero bins
    hist = np.clip(hist, 1e-6, None)

    # initial guesses
    C0 = np.mean(hist)
    A1_0 = 0.1
    A2_0 = 0.1
    ph1_0 = np.median(phi)
    ph2_0 = np.median(phi)

    try:
        popt, _ = curve_fit(harmonic, centers, hist,
                            p0=[C0, A1_0, ph1_0, A2_0, ph2_0],
                            maxfev=20000)
    except:
        return None, None

    _, A1, ph1, A2, ph2 = popt
    return float(ph1), float(ph2)

# ------------------------------------------------------------
def circular_distance(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 2*np.pi - d)

# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_harmonic_phase_lock_test42.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])

    # unified-axis phi
    phi = np.deg2rad(df["phi_unified"].values)
    theta = df["theta_unified"].values
    z = df["z_est"].values

    # define subsets
    subsets = {
        "all": np.ones(len(df), dtype=bool),
        "z_0_0.3": (z >= 0.0) & (z < 0.3),
        "z_0.3_0.6": (z >= 0.3) & (z < 0.6),
        "z_0.6_1.0": (z >= 0.6) & (z < 1.0),
        "theta_25_40": (theta >= 25) & (theta < 40),
        "theta_40_60": (theta >= 40) & (theta < 60),
    }

    # jackknife sky quadrants
    quad_edges = [ -180, -90, 0, 90, 180 ]
    for i in range(4):
        lo, hi = quad_edges[i], quad_edges[i+1]
        subsets[f"Q{i+1}"] = (np.rad2deg(phi) >= lo) & (np.rad2deg(phi) < hi)

    # --------------------------------------------------------
    # fit phases per subset
    # --------------------------------------------------------
    phases = []
    labels = []

    for name, mask in subsets.items():
        if np.sum(mask) < 30:
            continue

        ph1, ph2 = fit_phases(phi[mask])
        if ph1 is None:
            continue

        phases.append([ph1, ph2])
        labels.append(name)

    phases = np.array(phases)
    ph1_vals = phases[:,0]
    ph2_vals = phases[:,1]

    # compute RMS phase scatter
    mean_ph1 = np.angle(np.sum(np.exp(1j*ph1_vals)))
    mean_ph2 = np.angle(np.sum(np.exp(1j*ph2_vals)))

    S1 = np.sqrt(np.mean(circular_distance(ph1_vals, mean_ph1)**2))
    S2 = np.sqrt(np.mean(circular_distance(ph2_vals, mean_ph2)**2))

    S_real = S1 + S2

    # --------------------------------------------------------
    # monte carlo null
    # --------------------------------------------------------
    Nmc = 100000
    S_null = []

    for _ in tqdm(range(Nmc)):
        phi_sh = np.random.permutation(phi)
        phs = []
        for name, mask in subsets.items():
            if np.sum(mask) < 30:
                continue
            ph1, ph2 = fit_phases(phi_sh[mask])
            if ph1 is None:
                break
            phs.append([ph1, ph2])
        if len(phs) == 0:
            continue

        phs = np.array(phs)
        m1 = np.angle(np.sum(np.exp(1j*phs[:,0])))
        m2 = np.angle(np.sum(np.exp(1j*phs[:,1])))
        S1n = np.sqrt(np.mean(circular_distance(phs[:,0], m1)**2))
        S2n = np.sqrt(np.mean(circular_distance(phs[:,1], m2)**2))
        S_null.append(S1n + S2n)

    S_null = np.array(S_null)
    p = np.mean(S_null <= S_real)

    # --------------------------------------------------------
    print("="*60)
    print("FRB HARMONIC PHASE-LOCK STABILITY TEST (TEST 42)")
    print("="*60)
    print(f"number of subsets used = {len(ph1_vals)}")
    print(f"observed phase-lock statistic S_real = {S_real:.4f}")
    print(f"null mean S = {np.mean(S_null):.4f}")
    print(f"null std  S = {np.std(S_null):.4f}")
    print(f"p-value(S_null <= S_real) = {p:.4f}")
    print("-"*60)
    if p < 0.05:
        print("interpretation: strong phase-locking – coherent helical modes.")
    elif p < 0.3:
        print("interpretation: weak-to-moderate coherence.")
    else:
        print("interpretation: no evidence of coherent phase-locking.")
    print("="*60)
    print("test 42 complete.")
    print("="*60)


if __name__ == "__main__":
    main()
