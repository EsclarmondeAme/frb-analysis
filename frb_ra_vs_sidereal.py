import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.time import Time
import astropy.units as u


def sidereal_phase_from_mjd(mjd_array):
    """
    convert MJD times to sidereal phase (0..1) at greenwich.
    """
    t = Time(mjd_array, format="mjd", scale="utc")
    gmst = t.sidereal_time("mean", "greenwich")   # Angle with hourangle units
    gmst_hours = gmst.hour                        # convert to hours
    phase = (gmst_hours / 24.0) % 1.0
    return phase



def harmonic_amplitude(phases, n=1, n_mc=20000, random_state=12345):
    """
    compute n-th harmonic amplitude R_n and a Monte Carlo p-value
    under the null hypothesis of uniform phases on [0,1).
    """
    phi = np.asarray(phases) * 2.0 * np.pi
    A_n = np.mean(np.cos(n * phi))
    B_n = np.mean(np.sin(n * phi))
    R_n = np.sqrt(A_n**2 + B_n**2)

    rng = np.random.default_rng(random_state)
    N = len(phi)
    mc_R = []
    for _ in range(n_mc):
        rnd = rng.random(N) * 2.0 * np.pi
        A_r = np.mean(np.cos(n * rnd))
        B_r = np.mean(np.sin(n * rnd))
        mc_R.append(np.sqrt(A_r**2 + B_r**2))
    mc_R = np.array(mc_R)
    p = np.mean(mc_R >= R_n)
    return A_n, B_n, R_n, p


def main():
    print("============================================================")
    print("FRB RA vs sidereal-phase test (CHIME Catalog 1 → frbs.csv)")
    print("============================================================")

    # load frbs.csv produced earlier
    frb = pd.read_csv("frbs.csv")

    # need RA (deg) and MJD
    needed_cols = {"ra", "mjd"}
    if not needed_cols.issubset(frb.columns):
        raise ValueError(f"frbs.csv is missing required columns {needed_cols}")

    frb_clean = frb.dropna(subset=["ra", "mjd"]).copy()
    N = len(frb_clean)
    print(f"FRBs with valid RA & MJD: {N}")

    ra_deg = frb_clean["ra"].to_numpy()
    mjd = frb_clean["mjd"].to_numpy()

    # RA-phase (0..1) and sidereal phase from times
    phi_ra = (ra_deg / 360.0) % 1.0
    phi_sid = sidereal_phase_from_mjd(mjd)

    # correlation between RA-phase and sidereal phase
    r = np.corrcoef(phi_ra, phi_sid)[0, 1]
    print("------------------------------------------------------------")
    print(f"correlation between RA phase and sidereal phase: r = {r:.3f}")

    # harmonic analysis for n = 1..4
    print("------------------------------------------------------------")
    print("harmonic amplitudes and p-values (RA vs sidereal)")
    print("n    set      A_n        B_n        R_n        p(R_rand >= R_n)")
    print("------------------------------------------------------------")

    for n in range(1, 5):
        A_ra, B_ra, R_ra, p_ra = harmonic_amplitude(phi_ra, n=n)
        A_si, B_si, R_si, p_si = harmonic_amplitude(phi_sid, n=n)
        print(
            f"{n:1d}   RA    {A_ra:+.4f}   {B_ra:+.4f}   {R_ra:.4f}   {p_ra:7.4f}"
        )
        print(
            f"    SID   {A_si:+.4f}   {B_si:+.4f}   {R_si:.4f}   {p_si:7.4f}"
        )
        print("")

    # histogram comparison plot
    bins = 24
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        phi_sid,
        bins=bins,
        range=(0.0, 1.0),
        density=True,
        alpha=0.5,
        label="sidereal phase (from MJD)",
    )
    ax.hist(
        phi_ra,
        bins=bins,
        range=(0.0, 1.0),
        density=True,
        histtype="step",
        linewidth=2.0,
        label="RA phase = RA / 360°",
    )

    ax.set_xlabel("phase (0..1)")
    ax.set_ylabel("probability density")
    ax.set_title("FRB RA-phase vs sidereal-phase distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig("frb_ra_vs_sidereal.png")
    print("------------------------------------------------------------")
    print("saved comparison plot → frb_ra_vs_sidereal.png")

    # scatter plot of φ_sid vs φ_RA
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(phi_ra, phi_sid, s=10, alpha=0.6)
    ax2.set_xlabel("RA phase (RA / 360°)")
    ax2.set_ylabel("sidereal phase from MJD")
    ax2.set_title("FRB RA phase vs sidereal phase")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("frb_ra_vs_sidereal_scatter.png")
    print("saved scatter → frb_ra_vs_sidereal_scatter.png")
    print("============================================================")


if __name__ == "__main__":
    main()
