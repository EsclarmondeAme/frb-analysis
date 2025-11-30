import sys
import numpy as np
import pandas as pd
from scipy.special import sph_harm
from scipy.signal import welch
from scipy.optimize import curve_fit

# ============================================================
# Test 107A — Time-Resolved Harmonic Mode PSD + Q-factor
# ============================================================

def lorentzian(f, f0, gamma, A):
    return A * (gamma**2) / ((f - f0)**2 + gamma**2)

def compute_alms(thetas, phis, lmax):
    """return dict[(l,m)] = a_lm (complex)"""
    alms = {}
    N = len(thetas)
    for l in range(lmax+1):
        for m in range(-l, l+1):
            Ylm = sph_harm(m, l, phis, thetas)
            a_lm = np.sum(Ylm) / N
            alms[(l,m)] = a_lm
    return alms

def main():
    if len(sys.argv) < 2:
        print("usage: python frb_harmonic_psd_Qfactor_test107A.py frbs_unified.csv")
        sys.exit(1)

    infile = sys.argv[1]

    print("============================================================")
    print(" Test 107A — Time-Resolved Harmonic PSD + Q-factor")
    print("============================================================")

    print(f"[info] loading: {infile}")
    df = pd.read_csv(infile)

    ra = np.radians(df["ra"].values)
    dec = np.radians(df["dec"].values)
    mjd = df["mjd"].values

    # convert to spherical coords
    thetas = np.pi / 2 - dec
    phis = ra

    print(f"[info] N_FRB = {len(df)}")
    print(f"[info] MJD range = {mjd.min()} – {mjd.max()}")

    # time-chunking
    N_chunk = 8
    idx_sorted = np.argsort(mjd)
    mjd_sorted = mjd[idx_sorted]
    th_sorted = thetas[idx_sorted]
    ph_sorted = phis[idx_sorted]

    chunks = np.array_split(np.arange(len(df)), N_chunk)

    lmax = 8
    print(f"[info] lmax = {lmax}")
    print(f"[info] using {N_chunk} time chunks")

    # store time series of a_lm(t)
    alm_time = { (l,m): [] for l in range(lmax+1) for m in range(-l,l+1) }
    t_vals = []

    print("[info] computing a_lm(t) time series...")
    for ci, chunk in enumerate(chunks):
        th_c = th_sorted[chunk]
        ph_c = ph_sorted[chunk]
        mjd_c = mjd_sorted[chunk]

        t_vals.append(np.mean(mjd_c))
        alms = compute_alms(th_c, ph_c, lmax)

        for key in alm_time.keys():
            alm_time[key].append(alms[key])

        print(f"[info] chunk {ci}: N={len(chunk)}, MJD={mjd_c.min():.1f}-{mjd_c.max():.1f}")

    t_vals = np.array(t_vals)

    # build PSD and Q-factor
    print("[info] computing PSD and Q-factor per mode...")
    results = []

    for l in range(lmax+1):
        for m in range(-l, l+1):
            a_series = np.array(alm_time[(l,m)], dtype=complex)
            real_series = a_series.real

            # PSD using Welch
            freqs, Pxx = welch(real_series, nperseg=min(8, len(real_series)))

            # find peak
            peak_idx = np.argmax(Pxx)
            f0 = freqs[peak_idx]

            # fit Lorentzian around peak
            try:
                fit_range = slice(max(0, peak_idx-2), min(len(freqs), peak_idx+3))
                popt, _ = curve_fit(lorentzian, freqs[fit_range], Pxx[fit_range],
                                    p0=[f0, 0.1*(freqs[1]-freqs[0]), Pxx[peak_idx]])
                f0_fit, gamma_fit, A_fit = popt
                Q = f0_fit / (2*gamma_fit) if gamma_fit>0 else 0.0
            except:
                f0_fit = gamma_fit = A_fit = Q = np.nan

            results.append((l, m, f0_fit, gamma_fit, Q, np.max(Pxx)))

    print("------------------------------------------------------------")
    print(" RESULTS (Harmonic PSD peaks and Q-factor)")
    print("------------------------------------------------------------")
    for (l, m, f0, g, Q, Pmax) in results:
        print(f"(l,m)=({l:2d},{m:3d}) | f0={f0:.5f} | gamma={g:.5f} | Q={Q:.2f} | Pmax={Pmax:.3e}")

    # ============================================================
    # save results in dictionary form for Test 107B
    # ============================================================

    results_dict = {}

    for (l, m, f0_fit, gamma_fit, Q_fit, Pmax_fit) in results:
        results_dict[(l, m)] = {
            "f0": float(f0_fit) if f0_fit == f0_fit else np.nan,
            "gamma": float(gamma_fit) if gamma_fit == gamma_fit else np.nan,
            "Q": float(Q_fit) if Q_fit == Q_fit else np.nan,
            "Pmax": float(Pmax_fit) if Pmax_fit == Pmax_fit else np.nan,
        }

    np.save("test107A_results.npy", results_dict, allow_pickle=True)
    print("[saved] test107A_results.npy")




    print("============================================================")
    print(" test 107A complete")
    print("============================================================")











if __name__ == "__main__":
    main()
