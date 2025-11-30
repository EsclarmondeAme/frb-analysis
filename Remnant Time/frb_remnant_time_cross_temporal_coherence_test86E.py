#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
from scipy.special import sph_harm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------
# parse UTC → seconds
# ---------------------------------------------------------
def parse_time_to_seconds(t):
    try:
        return datetime.fromisoformat(t).timestamp()
    except:
        return np.nan

# ---------------------------------------------------------
# correlation helper
# ---------------------------------------------------------
def corrcoef(x, y):
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    return np.mean(xm * ym) / (np.std(x) * np.std(y))

# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main(fname):

    print("==========================================================")
    print(" CROSS–TEMPORAL PHASE MEMORY TEST (TEST 86E)")
    print("==========================================================")

    df = pd.read_csv(fname)
    N = len(df)
    print(f"[info] N_FRB = {N}")

    # -----------------------------------------------------
    # extract observation time
    # -----------------------------------------------------
    time_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "utc" in lc or "time" in lc or "mjd" in lc:
            time_col = c
            break

    if time_col is None:
        print("[err] no usable time column detected (utc/time/mjd).")
        sys.exit(1)

    print(f"[info] time column: {time_col}")

    t_raw = df[time_col].astype(str).values
    t_sec = np.array([parse_time_to_seconds(x) for x in t_raw], dtype=float)

    if np.any(np.isnan(t_sec)):
        print("[err] could not parse some timestamps.")
        sys.exit(1)

    # -----------------------------------------------------
    # extract coordinates: ALWAYS convert RA/Dec → Galactic
    # -----------------------------------------------------
    if ("ra" not in df.columns) or ("dec" not in df.columns):
        print("[err] frbs_unified.csv must have ra and dec.")
        sys.exit(1)

    ra = df["ra"].astype(float).values
    dec = df["dec"].astype(float).values

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    sky = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    gal = sky.galactic
    l_deg = gal.l.value
    b_deg = gal.b.value

    print("[info] converted RA/Dec → Galactic (l,b).")

    # -----------------------------------------------------
    # remnant–time SIGN label (fallback hemisphere)
    # -----------------------------------------------------
    if "remnant" in df.columns:
        R = np.sign(df["remnant"].values.astype(float))
        R[R == 0] = +1
        print("[info] using remnant scalar column.")
    else:
        # unified axis from earlier tests
        l0 = np.radians(160.0)
        b0 = np.radians(-0.5)

        l_rad = np.radians(l_deg)
        b_rad = np.radians(b_deg)

        x  = np.cos(b_rad)*np.cos(l_rad)
        y  = np.cos(b_rad)*np.sin(l_rad)
        z  = np.sin(b_rad)

        x0 = np.cos(b0)*np.cos(l0)
        y0 = np.cos(b0)*np.sin(l0)
        z0 = np.sin(b0)

        R = np.sign(x*x0 + y*y0 + z*z0)
        print("[info] using unified-axis hemisphere sign (fallback).")

    # -----------------------------------------------------
    # build spherical harmonics (phase field)
    # -----------------------------------------------------
    theta = np.radians(90 - b_deg)
    phi   = np.radians(l_deg)

    l_max = 8
    Y = []
    for ell in range(l_max+1):
        for m in range(-ell, ell+1):
            Y.append(sph_harm(m, ell, phi, theta))
    Y = np.vstack(Y)
    phases = np.angle(Y)

    # -----------------------------------------------------
    # build FRB pairs
    # -----------------------------------------------------
    i_idx, j_idx = np.triu_indices(N, k=1)

    # time separation
    dt = np.abs(t_sec[i_idx] - t_sec[j_idx])

    # phase alignment
    dphase = phases[:, i_idx] - phases[:, j_idx]
    G = np.mean(np.cos(dphase), axis=0)

    # sign relationship
    same = (R[i_idx] == R[j_idx])
    opp  = (R[i_idx] != R[j_idx])

    # -----------------------------------------------------
    # restrict to LARGE time gaps = top 20%
    # -----------------------------------------------------
    thresh = np.percentile(dt, 80)
    large = (dt >= thresh)

    same_large = same & large
    opp_large  = opp  & large

    print("----------------------------------------------------------")
    print(f"[info] large-time threshold = {thresh:.3e} sec")
    print(f"[info] pairs in same-large : {np.sum(same_large)}")
    print(f"[info] pairs in opp-large  : {np.sum(opp_large)}")
    print("----------------------------------------------------------")

    rho_same = corrcoef(dt[same_large], G[same_large]) if np.sum(same_large)>10 else np.nan
    rho_opp  = corrcoef(dt[opp_large],  G[opp_large])  if np.sum(opp_large)>10 else np.nan

    print(f"[info] rho_same_large = {rho_same:.6e}")
    print(f"[info] rho_opp_large  = {rho_opp:.6e}")

    # -----------------------------------------------------
    # null test: shuffle times ONLY
    # -----------------------------------------------------
    n_null = 2000
    rho_same_null = []
    rho_opp_null  = []

    for k in range(n_null):
        t_perm = np.random.permutation(t_sec)
        dt_null = np.abs(t_perm[i_idx] - t_perm[j_idx])

        large_null = (dt_null >= thresh)

        same_ln = same & large_null
        opp_ln  = opp  & large_null

        if np.sum(same_ln) > 10:
            rho_same_null.append(corrcoef(dt_null[same_ln], G[same_ln]))
        if np.sum(opp_ln) > 10:
            rho_opp_null.append(corrcoef(dt_null[opp_ln],  G[opp_ln]))

    rho_same_null = np.array(rho_same_null)
    rho_opp_null  = np.array(rho_opp_null)

    p_same = np.mean(np.abs(rho_same_null) >= abs(rho_same)) if not np.isnan(rho_same) else np.nan
    p_opp  = np.mean(np.abs(rho_opp_null)  >= abs(rho_opp))  if not np.isnan(rho_opp)  else np.nan

    print("----------------------------------------------------------")
    print(" NULL RESULTS")
    print(f" same-large: mean={np.mean(rho_same_null):.6e}, std={np.std(rho_same_null):.6e}, p={p_same:.6f}")
    print(f" opp-large : mean={np.mean(rho_opp_null):.6e}, std={np.std(rho_opp_null):.6e}, p={p_opp:.6f}")
    print("----------------------------------------------------------")

    print("interpretation:")
    print("  - rho_same_large ≈ 0 with high p → no internal temporal dependence.")
    print("  - rho_opp_large  significantly ≠ 0 → cross-hemisphere large-gap coherence.")
    print("    This is exactly the predicted signature of projected time-compression.")
    print("==========================================================")
    print("test 86E complete.")
    print("==========================================================")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_cross_temporal_coherence_test86E.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
