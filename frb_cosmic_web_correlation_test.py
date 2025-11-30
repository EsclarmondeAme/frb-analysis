import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import pearsonr, ks_2samp


def load_axes(path="axes.json"):
    """
    load unified axis (galactic l,b) from axes.json
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"could not find {path} in current directory")
    with open(path, "r") as f:
        axes = json.load(f)

    if "unified_axis" not in axes:
        raise KeyError("axes.json has no 'unified_axis' entry")

    l0 = axes["unified_axis"]["l"]
    b0 = axes["unified_axis"]["b"]
    return float(l0), float(b0)


def angle_from_unified_axis(ra_deg, dec_deg, l0_deg, b0_deg):
    """
    compute angular separation (in deg) of a sky position from the unified axis
    defined by (l0,b0) in galactic coordinates.
    """
    axis_coord = SkyCoord(l=l0_deg * u.deg, b=b0_deg * u.deg, frame="galactic")
    frb_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").galactic
    sep = frb_coord.separation(axis_coord)
    return sep.deg


def main():
    print("=======================================================")
    print("          frb cosmic-web / supergalactic test          ")
    print("=======================================================")

    # ----------------------------------------------------
    # load frb catalogue
    # ----------------------------------------------------
    if not os.path.exists("frbs.csv"):
        print("error: frbs.csv not found in current directory.")
        return

    frb = pd.read_csv("frbs.csv")

    # tolerate both 'z' and 'z_est' but we do not actually need z here
    if "z" in frb.columns:
        z_col = "z"
    elif "z_est" in frb.columns:
        z_col = "z_est"
    else:
        z_col = None

    required = {"ra", "dec"}
    if not required.issubset(frb.columns):
        print("error: frbs.csv must contain columns 'ra' and 'dec'.")
        return

    # drop rows without coordinates
    frb = frb.dropna(subset=["ra", "dec"])
    if len(frb) == 0:
        print("error: no frbs with valid ra/dec after dropna.")
        return

    # ----------------------------------------------------
    # load unified axis
    # ----------------------------------------------------
    try:
        l0, b0 = load_axes("axes.json")
    except Exception as e:
        print(f"error loading axes.json: {e}")
        return

    print("loaded: frbs.csv")
    print(f"unified axis (galactic): l = {l0:.2f} deg, b = {b0:.2f} deg")
    print("-------------------------------------------------------")

    # ----------------------------------------------------
    # compute theta (angle from unified axis) and SGB
    # ----------------------------------------------------
    ra = frb["ra"].to_numpy(dtype=float)
    dec = frb["dec"].to_numpy(dtype=float)

    # angle from unified axis
    theta = angle_from_unified_axis(ra, dec, l0, b0)

    # supergalactic latitude
    coords_eq = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    coords_sg = coords_eq.supergalactic
    sgb = coords_sg.sgb.deg
    sgb_abs = np.abs(sgb)

    frb["theta_axis"] = theta
    frb["sgb"] = sgb
    frb["sgb_abs"] = sgb_abs

    # ----------------------------------------------------
    # define the main anisotropy shell (20–60 deg)
    # ----------------------------------------------------
    shell_min = 20.0
    shell_max = 60.0
    shell_mask = (theta >= shell_min) & (theta <= shell_max)

    n_shell = shell_mask.sum()
    n_total = len(frb)

    if n_shell < 10:
        print(f"warning: only {n_shell} events in 20–60 deg shell; "
              f"results may be noisy.")

    print("shell selection (around unified axis):")
    print(f"  theta in [{shell_min:.1f}, {shell_max:.1f}] deg")
    print(f"  n_shell = {n_shell} of {n_total}")
    print("-------------------------------------------------------")

    # ----------------------------------------------------
    # 1) theta–|SGB| correlation inside the shell
    # ----------------------------------------------------
    theta_shell = theta[shell_mask]
    sgb_shell = sgb_abs[shell_mask]

    if len(theta_shell) < 3:
        print("not enough points in shell for correlation test.")
        return

    r_real, p_pearson = pearsonr(theta_shell, sgb_shell)

    print("theta–|sgb| correlation inside shell:")
    print(f"  pearson r = {r_real:.4f}, p = {p_pearson:.4g}")
    print("-------------------------------------------------------")
    print("running monte-carlo by shuffling |sgb| within shell...")

    rng = np.random.default_rng(seed=42)
    n_mc = 5000
    r_mc = np.empty(n_mc, dtype=float)

    for i in range(n_mc):
        shuffled = rng.permutation(sgb_shell)
        r_mc[i] = pearsonr(theta_shell, shuffled)[0]

    # two-sided p-value based on |r|
    p_mc = float(np.mean(np.abs(r_mc) >= np.abs(r_real)))

    print("-------------------------------------------------------")
    print("monte-carlo correlation null (shuffled |sgb|):")
    print(f"  p_mc(|r_null| >= |r_real|) = {p_mc:.5f}")
    print("-------------------------------------------------------")

    # ----------------------------------------------------
    # 2) KS test: |SGB| in-shell vs out-of-shell
    # ----------------------------------------------------
    sgb_out = sgb_abs[~shell_mask]
    if len(sgb_out) >= 3:
        ks_stat, ks_p = ks_2samp(sgb_shell, sgb_out)
        print("ks test: |sgb| distribution shell vs outside:")
        print(f"  ks statistic = {ks_stat:.4f}, p = {ks_p:.4g}")
    else:
        ks_stat, ks_p = np.nan, np.nan
        print("not enough events outside shell for ks test.")
    print("-------------------------------------------------------")

    # ----------------------------------------------------
    # make diagnostic plot: theta vs |sgb|
    # ----------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.scatter(theta, sgb_abs, s=15, alpha=0.6, label="all frbs")

    # highlight shell events
    plt.scatter(theta_shell, sgb_shell, s=25, alpha=0.9, label="shell: 20–60 deg")

    plt.axvspan(shell_min, shell_max, alpha=0.1, label="shell region")
    plt.xlabel(r"angle from unified axis $\theta$ (deg)")
    plt.ylabel(r"supergalactic latitude $|{\rm sgb}|$ (deg)")

    plt.title("frb shell vs supergalactic plane")
    plt.legend()
    plt.tight_layout()
    outname = "frb_cosmic_web_correlation.png"
    plt.savefig(outname, dpi=150)
    plt.close()

    # ----------------------------------------------------
    # scientific verdict
    # ----------------------------------------------------
    print("scientific interpretation:")
    print("-------------------------------------------------------")

    # correlation verdict
    if p_mc > 0.05 and p_pearson > 0.05:
        print(
            "no significant correlation is found between shell angle θ and\n"
            "distance from the supergalactic plane |sgb|. the warped shell\n"
            "does not preferentially sit in the local supergalactic plane."
        )
    else:
        print(
            "a statistically significant correlation is found between θ and |sgb|.\n"
            "this would suggest a link between the warped shell and the local\n"
            "supergalactic geometry (cosmic web)."
        )

    # ks verdict
    if not np.isnan(ks_p):
        if ks_p > 0.05:
            print(
                "the |sgb| distribution of shell frbs is statistically consistent\n"
                "with that of the rest of the sky; there is no strong evidence\n"
                "that the shell occupies a special band in supergalactic latitude."
            )
        else:
            print(
                "the |sgb| distribution of shell frbs differs significantly from\n"
                "the rest of the sky; this would indicate a preferred band in\n"
                "supergalactic latitude."
            )

    print("-------------------------------------------------------")
    print(f"saved plot: {outname}")
    print("analysis complete.")
    print("=======================================================")


if __name__ == "__main__":
    main()
