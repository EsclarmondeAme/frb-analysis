#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, pearsonr
from astropy.coordinates import SkyCoord
import astropy.units as u

# -------------------------------------------------------
# configuration
# -------------------------------------------------------

FRB_CSV = "frbs.csv"

# unified axis (galactic), from your analysis
UNIFIED_L = 159.85  # deg
UNIFIED_B = -0.51   # deg

# shell where the warped / lopsided structure lives
SHELL_THETA_MIN = 20.0  # deg
SHELL_THETA_MAX = 60.0  # deg

# observables to test
OBSERVABLES = [
    ("dm", "Dispersion measure DM"),
    ("width", "Pulse width"),
    ("fluence", "Fluence"),
    ("energy_proxy", "Energy proxy ~ fluence * DM^2"),
]

# -------------------------------------------------------
# helpers
# -------------------------------------------------------


def load_frbs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure lowercase column names for safety
    df.columns = [c.lower() for c in df.columns]

    required = ["ra", "dec", "dm", "width", "fluence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing required columns in {path}: {missing}")

    # optional redshift / z_est
    if "z_est" in df.columns and "z" not in df.columns:
        df["z"] = df["z_est"]

    # energy proxy
    dm = df["dm"].astype(float)
    flu = df["fluence"].astype(float)
    df["energy_proxy"] = flu * dm**2

    return df


def add_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    ra = df["ra"].values
    dec = df["dec"].values

    sky_eq = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    gal = sky_eq.galactic
    sg = sky_eq.supergalactic

    df["l"] = gal.l.deg
    df["b"] = gal.b.deg
    df["sgl"] = sg.sgl.deg
    df["sgb"] = sg.sgb.deg
    df["abs_sgb"] = np.abs(df["sgb"].values)

    # angular distance to unified axis
    axis = SkyCoord(l=UNIFIED_L * u.deg, b=UNIFIED_B * u.deg, frame="galactic")
    frb_gal = SkyCoord(l=df["l"].values * u.deg,
                       b=df["b"].values * u.deg,
                       frame="galactic")
    theta = frb_gal.separation(axis).deg
    df["theta_axis"] = theta

    return df


def summarise_array(x: np.ndarray) -> str:
    if x.size == 0:
        return "N=0"
    return f"N={x.size}, mean={np.nanmean(x):.3g}, median={np.nanmedian(x):.3g}"


# -------------------------------------------------------
# main analysis
# -------------------------------------------------------


def main():
    print("=======================================================")
    print("       frb observable-coherence test")
    print("=======================================================")

    # load catalogue
    try:
        frb = load_frbs(FRB_CSV)
    except Exception as e:
        print("error: could not load frb catalogue:")
        print(e)
        return

    frb = add_coordinate_columns(frb)

    # define shell / outside
    theta = frb["theta_axis"].values
    in_shell = (theta >= SHELL_THETA_MIN) & (theta <= SHELL_THETA_MAX)
    out_shell = ~in_shell

    n_shell = in_shell.sum()
    n_out = out_shell.sum()

    print(f"loaded: {FRB_CSV}")
    print("-------------------------------------------------------")
    print(f"unified axis (galactic): l = {UNIFIED_L:.2f} deg, b = {UNIFIED_B:.2f} deg")
    print("shell definition:")
    print(f"  theta in [{SHELL_THETA_MIN:.1f}, {SHELL_THETA_MAX:.1f}] deg")
    print(f"  n_shell   = {n_shell}")
    print(f"  n_outside = {n_out}")
    print("-------------------------------------------------------")

    results = []

    for col, label in OBSERVABLES:
        if col not in frb.columns:
            print(f"warning: observable '{col}' not found, skipping.")
            continue

        shell_vals = frb.loc[in_shell, col].astype(float).to_numpy()
        out_vals = frb.loc[out_shell, col].astype(float).to_numpy()

        shell_vals = shell_vals[np.isfinite(shell_vals)]
        out_vals = out_vals[np.isfinite(out_vals)]

        print(f"observable: {label} ({col})")
        print(f"  shell:   {summarise_array(shell_vals)}")
        print(f"  outside: {summarise_array(out_vals)}")

        if shell_vals.size >= 5 and out_vals.size >= 5:
            ks_res = ks_2samp(shell_vals, out_vals, alternative="two-sided")
            ks_stat, ks_p = ks_res.statistic, ks_res.pvalue
        else:
            ks_stat, ks_p = np.nan, np.nan

        # correlation with |sgb| inside shell
        abs_sgb_shell = frb.loc[in_shell, "abs_sgb"].astype(float).to_numpy()
        mask_finite = np.isfinite(shell_vals) & np.isfinite(abs_sgb_shell)
        if mask_finite.sum() >= 5:
            r_sgb, p_sgb = pearsonr(abs_sgb_shell[mask_finite],
                                    shell_vals[mask_finite])
        else:
            r_sgb, p_sgb = np.nan, np.nan

        print(f"  ks(shell vs outside): D = {ks_stat:.3f}, p = {ks_p:.3g}")
        print(f"  corr(|sgb|, {col}) in shell: r = {r_sgb:.3f}, p = {p_sgb:.3g}")
        print("-------------------------------------------------------")

        results.append({
            "observable": col,
            "label": label,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "r_sgb": r_sgb,
            "p_sgb": p_sgb,
            "mean_shell": np.nanmean(shell_vals) if shell_vals.size else np.nan,
            "mean_out": np.nanmean(out_vals) if out_vals.size else np.nan,
            "med_shell": np.nanmedian(shell_vals) if shell_vals.size else np.nan,
            "med_out": np.nanmedian(out_vals) if out_vals.size else np.nan,
        })

    # basic verdict logic
    sig_shell = any(
        (not np.isnan(r["ks_p"])) and (r["ks_p"] < 1e-2)
        for r in results
    )
    sig_sgb = any(
        (not np.isnan(r["p_sgb"])) and (r["p_sgb"] < 1e-2 and abs(r["r_sgb"]) > 0.2)
        for r in results
    )

    print("scientific interpretation:")
    print("-------------------------------------------------------")
    if not sig_shell and not sig_sgb:
        print("no strong evidence that FRB observables (DM, width, fluence, energy)")
        print("change between the warped shell and the rest of the sky, or with")
        print("|sgb| inside the shell. this favours a primarily geometric /")
        print("positional anisotropy rather than a strong change in physical")
        print("burst properties.")
    elif sig_shell and not sig_sgb:
        print("some observables differ between shell and outside, but there is")
        print("no clear monotonic trend with |sgb| inside the shell. this could")
        print("indicate a weak physical coherence tied to the shell geometry.")
    elif not sig_shell and sig_sgb:
        print("observables show a correlation with |sgb| inside the shell, but")
        print("the shell vs outside distributions are not dramatically different.")
        print("this suggests a subtle modulation by supergalactic latitude.")
    else:
        print("both shell vs outside and |sgb|-correlation tests show significant")
        print("differences for at least one observable. this favours a scenario")
        print("in which the warped shell traces not only positional overdensity")
        print("but also changes in FRB physical properties along the cosmic web.")

    # ---------------------------------------------------
    # simple diagnostic plots
    # ---------------------------------------------------
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()

        # theta vs |sgb|
        axes[0].scatter(frb["theta_axis"], frb["abs_sgb"], s=8, alpha=0.5)
        axes[0].axvspan(SHELL_THETA_MIN, SHELL_THETA_MAX,
                        color="grey", alpha=0.2, label="shell")
        axes[0].set_xlabel(r"$\theta$ from unified axis (deg)")
        axes[0].set_ylabel(r"$|{\rm SGB}|$ (deg)")
        axes[0].legend(loc="best")
        axes[0].set_title("geometry: theta vs |sgb|")

        # one observable vs theta (energy proxy if available)
        if "energy_proxy" in frb.columns:
            y = frb["energy_proxy"].values
            axes[1].scatter(frb["theta_axis"], y, s=8, alpha=0.5)
            axes[1].axvspan(SHELL_THETA_MIN, SHELL_THETA_MAX,
                            color="grey", alpha=0.2)
            axes[1].set_xlabel(r"$\theta$ (deg)")
            axes[1].set_ylabel("energy proxy")
            axes[1].set_title("energy proxy vs theta")
        else:
            axes[1].axis("off")

        # shell vs outside distributions for DM and width if present
        if "dm" in frb.columns:
            dm_shell = frb.loc[in_shell, "dm"].values
            dm_out = frb.loc[out_shell, "dm"].values
            bins = np.histogram(
                np.concatenate([dm_shell, dm_out]),
                bins=30
            )[1]
            axes[2].hist(dm_shell, bins=bins, alpha=0.5, label="shell")
            axes[2].hist(dm_out, bins=bins, alpha=0.5, label="outside")
            axes[2].set_xlabel("DM")
            axes[2].set_ylabel("count")
            axes[2].set_title("DM distribution")
            axes[2].legend(loc="best")
        else:
            axes[2].axis("off")

        if "width" in frb.columns:
            w_shell = frb.loc[in_shell, "width"].values
            w_out = frb.loc[out_shell, "width"].values
            bins = np.histogram(
                np.concatenate([w_shell, w_out]),
                bins=30
            )[1]
            axes[3].hist(w_shell, bins=bins, alpha=0.5, label="shell")
            axes[3].hist(w_out, bins=bins, alpha=0.5, label="outside")
            axes[3].set_xlabel("width")
            axes[3].set_ylabel("count")
            axes[3].set_title("width distribution")
            axes[3].legend(loc="best")
        else:
            axes[3].axis("off")

        plt.tight_layout()
        outname = "frb_observable_coherence.png"
        plt.savefig(outname, dpi=200)
        plt.close(fig)
        print("saved plot:", outname)
    except Exception as e:
        print("warning: failed to generate plots:", e)

    print("analysis complete.")
    print("=======================================================")


if __name__ == "__main__":
    main()
