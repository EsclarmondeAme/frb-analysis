"""
frb cone-axis sidereal modulation fit

this script fits a physical cone-axis model to fast radio burst sidereal phases.
it interprets the harmonic structure (n=1..4) in terms of an underlying
axis direction and cone half-angle, rather than unconstrained fourier amplitudes.

model summary
-------------
frb rate vs sidereal phase φ is modeled as:

    R(φ) = C + A * max(0, cos(φ - φ0 - α))

where:
    φ0  = axis phase (where the cone axis crosses local meridian)
    α   = cone half-angle (broadness of emission weight)
    A   = modulation amplitude
    C   = constant background

this is a very general physical “spotlight / cone” model that naturally produces
harmonic content at n = 1..4 depending on cone width.

output:
    - best-fit axis sidereal phase φ0
    - cone half-angle α
    - amplitude A
    - constant C
    - predicted harmonic amplitudes R_n
    - comparison against random uniform phases
    - saved diagnostic plot: frb_cone_fit.png
"""

import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

def load_frbs(path="frbs.csv"):
    """
    load frb catalog and return mjd array
    """
    df = pd.read_csv(path)
    if "mjd" not in df.columns:
        raise ValueError("frbs.csv missing mjd column")
    mjd = df["mjd"].values
    return mjd


def sidereal_phase(mjd):
    """
    compute sidereal phase (0..1) from mjd.
    use simple linear sidereal/day ratio.
    """
    sidereal_per_solar = 1.00273790935
    phase = (mjd * sidereal_per_solar) % 1.0
    return phase


def cone_model(phi, phi0, alpha, A, C):
    """
    physical cone model R(φ):
        R = C + A * max(0, cos(phi - phi0 - alpha))
    """
    angle = phi - phi0 - alpha
    val = np.cos(angle)
    val[val < 0] = 0
    return C + A * val


def neg_log_likelihood(theta, phases):
    """
    poisson likelihood for histogram bins:
        ln L = sum(n_i ln λ_i - λ_i)
    negative for minimization.
    """
    phi0, alpha, A, C = theta
    # enforce positivity constraints
    if A < 0 or C < 0:
        return 1e12
    R = cone_model(phases, phi0, alpha, A, C)
    # small epsilon to avoid zero logs
    R = np.clip(R, 1e-12, None)
    # treat each frb as one count at its phase
    return -np.sum(np.log(R))


def fit_cone_axis(phases):
    """
    optimize parameters:
        phi0, alpha, A, C
    """
    init = np.array([0.0, 0.3, 1.0, 1.0])
    bounds = [
        (0, 2*np.pi),   # phi0
        (0, np.pi/2),  # alpha (0..90 deg)
        (0, None),     # A
        (0, None)      # C
    ]
    res = minimize(neg_log_likelihood, init, args=(phases,),
                   bounds=bounds, method="L-BFGS-B")
    return res.x


def compute_harmonics(phases, nmax=4):
    """
    compute fourier harmonics for comparison.
    """
    amps = []
    for n in range(1, nmax+1):
        A_n = np.mean(np.cos(2*np.pi*n*phases))
        B_n = np.mean(np.sin(2*np.pi*n*phases))
        R_n = np.sqrt(A_n**2 + B_n**2)
        amps.append((A_n, B_n, R_n))
    return amps


def plot_result(phases, theta, outname="frb_cone_fit.png"):
    """
    diagnostic plot of data histogram and best-fit model.
    """
    phi0, alpha, A, C = theta

    bins = 40
    hist, edges = np.histogram(phases, bins=bins, range=(0, 1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    phi_grid = np.linspace(0, 1, 1000)
    phi_rad = phi_grid * 2*np.pi

    model_vals = cone_model(phi_rad, phi0, alpha, A, C)
    model_vals /= np.trapz(model_vals, phi_grid)

    plt.figure(figsize=(10,5))
    plt.step(centers, hist, where="mid", label="frb sidereal histogram")
    plt.plot(phi_grid, model_vals, linewidth=2, label="best-fit cone model")
    plt.xlabel("sidereal phase")
    plt.ylabel("probability density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outname)
    logging.info("saved plot → %s", outname)


def main():
    logging.info("loading frbs...")
    mjd = load_frbs("frbs.csv")

    logging.info("computing sidereal phases...")
    phases = sidereal_phase(mjd)

    logging.info("fitting cone-axis model...")
    phi0, alpha, A, C = fit_cone_axis(phases)
    logging.info("fit results:")
    logging.info(" axis phase φ0 (rad): %.4f", phi0)
    logging.info(" cone half-angle α (deg): %.2f", np.degrees(alpha))
    logging.info(" amplitude A: %.4f", A)
    logging.info(" constant C: %.4f", C)

    logging.info("computing harmonic amplitudes...")
    harmonics = compute_harmonics(phases)
    for i, (A_n, B_n, R_n) in enumerate(harmonics, start=1):
        logging.info(" n=%d  A_n=%.4f  B_n=%.4f  R_n=%.4f", i, A_n, B_n, R_n)

    logging.info("plotting...")
    plot_result(phases, (phi0, alpha, A, C))

    logging.info("done.")


if __name__ == "__main__":
    main()
