#!/usr/bin/env python3
import numpy as np
import argparse
from scipy.special import sph_harm
from tqdm import tqdm

def compute_alm(ra, dec, lmax):
    ph = np.radians(ra)
    th = np.radians(90 - dec)
    alm = {}

    for l in range(lmax+1):
        for m in range(-l, l+1):
            Y = sph_harm(m, l, ph, th)
            alm[(l,m)] = np.sum(Y).real
    return alm

def build_bins(vals, nbins=10):
    mn, mx = np.min(vals), np.max(vals)
    if mx == mn:
        return np.array([mn]), np.array([0]) 
    edges = np.linspace(mn, mx, nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    return centers, edges

def transitions(amplitudes, edges):
    idx = np.digitize(amplitudes, edges) - 1
    idx[idx < 0] = 0
    idx[idx >= len(edges)-1] = len(edges)-2
    return idx

def cycle_current(P, centers):
    C = 0.0
    nb = len(centers)
    for i in range(nb):
        for j in range(i+1, nb):
            dP = P[i,j] - P[j,i]
            dE = centers[j] - centers[i]
            C += dP * dE
    return C

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog")
    parser.add_argument("--lmax", type=int, default=8)
    parser.add_argument("--n_null", type=int, default=2000)
    args = parser.parse_args()

    data = np.genfromtxt(args.catalog, delimiter=",", names=True)
    ra = data["ra"]
    dec = data["dec"]
    mjd = data["mjd"]

    print("====================================================")
    print("Test 108 — Mode-resolved detailed-balance violation")
    print("====================================================")
    print(f"[info] N_FRB = {len(ra)}")

    # time-chunk split (same as 107)
    mjd_min = np.min(mjd)
    mjd_max = np.max(mjd)
    chunks = []
    edges = np.linspace(mjd_min, mjd_max, 9)

    for i in range(8):
        mask = (mjd >= edges[i]) & (mjd < edges[i+1])
        chunks.append(np.where(mask)[0])
        print(f"[info] chunk {i}: N={len(chunks[-1])}, MJD={edges[i]:.1f}-{edges[i+1]:.1f}")

    # compute a_lm(t)
    print("[info] computing a_lm time series...")
    time_series = {}  # (l,m) → [a(t0), a(t1), ...]
    lmax = args.lmax

    for l in range(lmax+1):
        for m in range(-l, l+1):
            series = []
            for idx in chunks:
                if len(idx) == 0:
                    series.append(0.0)
                    continue
                alm = compute_alm(ra[idx], dec[idx], lmax=l)
                series.append(alm[(l,m)])
            time_series[(l,m)] = np.array(series)

    # results
    results = {}

    print("[info] computing real cycle currents...")
    for l in range(lmax+1):
        for m in range(-l, l+1):

            a = time_series[(l,m)]
            centers, edges = build_bins(a, nbins=10)
            ids = transitions(a, edges)

            nb = len(centers)
            P = np.zeros((nb, nb))

            # estimate transition probabilities
            for i in range(len(ids)-1):
                P[ids[i], ids[i+1]] += 1
            if P.sum() > 0:
                P /= P.sum()

            C_real = cycle_current(P, centers)

            # null ensemble
            C_null = np.zeros(args.n_null)
            for k in range(args.n_null):
                shuffled = np.random.permutation(a)
                ids_s = transitions(shuffled, edges)
                Pn = np.zeros((nb, nb))
                for i in range(len(ids_s)-1):
                    Pn[ids_s[i], ids_s[i+1]] += 1
                if Pn.sum() > 0:
                    Pn /= Pn.sum()
                C_null[k] = cycle_current(Pn, centers)

            p = (np.sum(C_null >= C_real) + 1) / (args.n_null + 1)

            results[(l,m)] = (C_real, C_null.mean(), C_null.std(), p)

            print(f"(l,m)=({l:2d},{m:3d}) | C={C_real:+.4e} | null_mean={C_null.mean():.4e} | p={p:.5f}")

    print("====================================================")
    print("test 108 complete")
    print("====================================================")

if __name__ == "__main__":
    main()
