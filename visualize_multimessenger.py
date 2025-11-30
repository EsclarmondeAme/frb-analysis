import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def safe_read_csv(path):
    """read a csv if it exists, otherwise return None."""
    if not os.path.isfile(path):
        print(f"[info] file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"[info] loaded {len(df)} rows from {path}")
        return df
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}")
        return None


def load_all_data():
    """load frbs, neutrinos, uhecr, and coincidence tables if present."""
    frbs = safe_read_csv("frbs.csv")
    nus = safe_read_csv("neutrinos_clean.csv")
    crs = safe_read_csv("uhecr.csv")
    coinc = safe_read_csv("coincidences_enhanced.csv")
    wide = safe_read_csv("coincidences_wide_enhanced.csv")
    triples = safe_read_csv("triple_coincidences_enhanced.csv")

    # parse utc columns if present
    for df, name in [
        (frbs, "frbs"),
        (nus, "neutrinos"),
        (crs, "uhecr"),
        (coinc, "coincidences"),
        (wide, "coincidences_wide"),
        (triples, "triple_coincidences"),
    ]:
        if df is not None and "utc" in df.columns:
            df["utc"] = pd.to_datetime(df["utc"], errors="coerce")
        # some tables use frb_time / nu_time / cr_time instead of utc
        if df is not None:
            for col in ["frb_time", "nu_time", "cr_time"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

    return frbs, nus, crs, coinc, wide, triples


def plot_sky_map(frbs, nus, crs, save_path="sky_map.png"):
    """scatter plot of ra/dec for all three messengers."""
    if frbs is None and nus is None and crs is None:
        print("[info] no sky data available, skipping sky map")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("sky map: frbs, neutrinos, uhecr")

    if frbs is not None:
        ax.scatter(frbs["ra"], frbs["dec"], s=10, alpha=0.6, label="frb")

    if nus is not None:
        ax.scatter(nus["ra"], nus["dec"], s=25, marker="x", alpha=0.8, label="neutrino")

    if crs is not None:
        ax.scatter(crs["ra"], crs["dec"], s=20, marker="^", alpha=0.8, label="uhecr")

    ax.set_xlabel("ra [deg]")
    ax.set_ylabel("dec [deg]")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"[info] saved sky map to {save_path}")


def plot_time_energy(frbs, nus, crs, save_path="time_energy.png"):
    """time vs energy (or flux) for all messengers."""
    if frbs is None and nus is None and crs is None:
        print("[info] no time-energy data available, skipping")
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.suptitle("time vs energy / flux")

    # frbs: utc vs flux
    ax = axes[0]
    if frbs is not None and "utc" in frbs.columns:
        ax.scatter(frbs["utc"], frbs.get("flux", 1.0), s=10, alpha=0.6)
        ax.set_ylabel("frb flux (arb)")
    else:
        ax.text(0.5, 0.5, "no frb time data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("frb")

    ax.grid(True, linewidth=0.3, alpha=0.5)

    # neutrinos: utc vs energy_tev
    ax = axes[1]
    if nus is not None and "utc" in nus.columns and "energy_tev" in nus.columns:
        ax.scatter(nus["utc"], nus["energy_tev"], s=20, alpha=0.7)
        ax.set_ylabel("nu energy [tev]")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "no neutrino energy/time", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("neutrinos")

    ax.grid(True, linewidth=0.3, alpha=0.5)

    # uhecr: utc vs energy_eev
    ax = axes[2]
    if crs is not None and "utc" in crs.columns and "energy_eev" in crs.columns:
        ax.scatter(crs["utc"], crs["energy_eev"], s=20, alpha=0.7)
        ax.set_ylabel("cr energy [eev]")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "no uhecr energy/time", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("uhecr")

    ax.grid(True, linewidth=0.3, alpha=0.5)

    axes[-1].set_xlabel("time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    print(f"[info] saved time-energy plot to {save_path}")


def plot_score_vs_redshift(coinc, triples, save_path="score_redshift.png"):
    """coincidence / triple score vs redshift scatter plots."""
    if coinc is None and triples is None:
        print("[info] no coincidence tables, skipping score vs redshift")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("coincidence score vs redshift")

    # left: pair coincidences
    ax = axes[0]
    if coinc is not None and "frb_z" in coinc.columns and "coinc_score" in coinc.columns:
        z = coinc["frb_z"].replace([np.inf, -np.inf], np.nan)
        s = coinc["coinc_score"]
        mask = z.notna() & s.notna()
        ax.scatter(z[mask], s[mask], s=30, alpha=0.7)
        ax.set_xlabel("frb z_est")
        ax.set_ylabel("coinc score")
        ax.set_title("frb + neutrino")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "no pair coincidence scores", ha="center", va="center", transform=ax.transAxes)

    # right: triple coincidences
    ax = axes[1]
    if triples is not None and "frb_z" in triples.columns and "triple_score" in triples.columns:
        z = triples["frb_z"].replace([np.inf, -np.inf], np.nan)
        s = triples["triple_score"]
        mask = z.notna() & s.notna()
        ax.scatter(z[mask], s[mask], s=30, alpha=0.7)
        ax.set_xlabel("frb z_est")
        ax.set_ylabel("triple score")
        ax.set_title("frb + nu + uhecr")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "no triple coincidence scores", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    print(f"[info] saved score vs redshift plot to {save_path}")


def plot_angular_histograms(coinc, triples, save_path="angular_histograms.png"):
    """plot histograms of angular separations from coincidence tables."""
    if coinc is None and triples is None:
        print("[info] no coincidence tables, skipping angular histograms")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("angular separation histograms")

    ax = axes[0]
    if coinc is not None and "angular_deg" in coinc.columns:
        ang = coinc["angular_deg"].dropna()
        ax.hist(ang, bins=20, alpha=0.8)
        ax.set_xlabel("frb–nu angle [deg]")
        ax.set_ylabel("count")
        ax.set_title("pair coincidences")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "no angular_deg data", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1]
    if triples is not None and "ang_nu_cr_deg" in triples.columns:
        ang_nc = triples["ang_nu_cr_deg"].dropna()
        ax.hist(ang_nc, bins=20, alpha=0.8)
        ax.set_xlabel("nu–cr angle [deg]")
        ax.set_ylabel("count")
        ax.set_title("triple coincidences")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "no triple angular data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    print(f"[info] saved angular histograms to {save_path}")


def main():
    print("visualizing multi-messenger data")
    print("=" * 60)

    frbs, nus, crs, coinc, wide, triples = load_all_data()

    plot_sky_map(frbs, nus, crs, save_path="sky_map.png")
    plot_time_energy(frbs, nus, crs, save_path="time_energy.png")
    plot_score_vs_redshift(coinc, triples, save_path="score_redshift.png")
    plot_angular_histograms(coinc, triples, save_path="angular_histograms.png")

    print("\nall plots saved in the current folder.")
    print("you can open the png files to inspect them.")
    print("=" * 60)

    # show plots interactively if you want
    try:
        plt.show()
    except Exception:
        # in case environment is non-interactive
        pass


if __name__ == "__main__":
    main()
