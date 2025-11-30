import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm


# ------------------------------------------------------------
# logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------
def sidereal_phase_from_mjd(mjd):
    """
    convert MJD to sidereal phase in [0,1).
    using standard approximation:

    GMST(hours) ≈ 18.697374558 + 24.06570982441908 * (MJD - 51544.5)

    then sidereal phase = GMST/24 mod 1
    """
    gmst_hours = 18.697374558 + 24.06570982441908 * (mjd - 51544.5)
    phi = (gmst_hours / 24.0) % 1.0
    return phi


def frb_cone_model(phi, phi0, alpha, A, C):
    """
    FRB sidereal cone-selection model:
        S(phi) = C + A * max( cos(phi - phi0) - cos(alpha), 0 )
    """
    return C + A * np.maximum(np.cos(2*np.pi*(phi - phi0)) - np.cos(alpha), 0)


def neutrino_dipole_model(phi, phi0, D, E):
    """
    neutrino dipole model:
        N(phi) = E + D * cos(phi - phi0)
    """
    return E + D * np.cos(2*np.pi*(phi - phi0))


# ------------------------------------------------------------
# model fitting
# ------------------------------------------------------------
def neg_log_likelihood_frb(params, phi):
    """
    FRB negative log-likelihood under cone model.
    """
    phi0, alpha, logA, logC = params
    A = np.exp(logA)
    C = np.exp(logC)

    model = frb_cone_model(phi, phi0, alpha, A, C)
    if np.any(model <= 0):
        return np.inf

    return -np.sum(np.log(model))


def neg_log_likelihood_nu(params, phi):
    """
    neutrino negative log-likelihood under dipole model.
    """
    phi0, logD, logE = params
    D = np.exp(logD)
    E = np.exp(logE)

    model = neutrino_dipole_model(phi, phi0, D, E)
    if np.any(model <= 0):
        return np.inf

    return -np.sum(np.log(model))


def fit_frb(phi):
    """
    fit FRB cone-selection model.
    """
    x0 = [0.2, 0.5, np.log(1e-3), np.log(1e-3)]
    bounds = [(0,1), (0, np.pi), (-20,10), (-20,10)]

    res = minimize(neg_log_likelihood_frb, x0, args=(phi,), bounds=bounds)
    return res


def fit_nu(phi):
    """
    fit neutrino dipole model.
    """
    x0 = [0.3, np.log(1e-3), np.log(1e-3)]
    bounds = [(0,1), (-20,10), (-20,10)]

    res = minimize(neg_log_likelihood_nu, x0, args=(phi,), bounds=bounds)
    return res


def joint_neg_log_likelihood(params, phi_frb, phi_nu):
    """
    joint negative log-likelihood with shared axis phi0:

        FRB uses cone (alpha,A,C)
        neutrinos use dipole (D,E)

    params:
        phi0 shared axis
        alpha
        logA, logC
        logD, logE
    """
    phi0, alpha, logA, logC, logD, logE = params

    A = np.exp(logA)
    C = np.exp(logC)
    D = np.exp(logD)
    E = np.exp(logE)

    model_frb = frb_cone_model(phi_frb, phi0, alpha, A, C)
    model_nu  = neutrino_dipole_model(phi_nu, phi0, D, E)

    if np.any(model_frb <= 0) or np.any(model_nu <= 0):
        return np.inf

    L = -np.sum(np.log(model_frb)) - np.sum(np.log(model_nu))
    return L


def fit_joint(phi_frb, phi_nu):
    """
    fit joint model with shared axis phi0.
    """
    x0 = [0.3, 0.3, np.log(1e-3), np.log(1e-3), np.log(1e-3), np.log(1e-3)]
    bounds = [
        (0,1),          # phi0
        (0, np.pi),     # alpha
        (-20,10),       # logA
        (-20,10),       # logC
        (-20,10),       # logD
        (-20,10)        # logE
    ]

    res = minimize(joint_neg_log_likelihood, x0,
                   args=(phi_frb, phi_nu), bounds=bounds)
    return res


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    logging.info("loading FRBs...")
    frb = pd.read_csv("frbs.csv")
    frb = frb.dropna(subset=["mjd"])
    logging.info(f"FRBs with valid MJD: {len(frb)}")

    logging.info("loading neutrinos...")
    nu = pd.read_csv("neutrinos.csv")
    nu = nu.dropna(subset=["mjd"])
    logging.info(f"neutrinos with valid MJD: {len(nu)}")

    phi_frb = sidereal_phase_from_mjd(frb["mjd"].values)
    phi_nu  = sidereal_phase_from_mjd(nu["mjd"].values)

    logging.info("------------------------------------------------------------")
    logging.info("fitting FRB cone model...")
    frb_fit = fit_frb(phi_frb)
    logging.info(f"FRB fit success: {frb_fit.success}")

    logging.info("fitting neutrino dipole model...")
    nu_fit = fit_nu(phi_nu)
    logging.info(f"neutrino fit success: {nu_fit.success}")

    logging.info("------------------------------------------------------------")
    logging.info("fitting joint model (shared axis)...")
    joint_fit = fit_joint(phi_frb, phi_nu)
    logging.info(f"joint fit success: {joint_fit.success}")

    # likelihoods
    L_frb  = -frb_fit.fun
    L_nu   = -nu_fit.fun
    L_sep  = L_frb + L_nu       # separate-axis
    L_joint = -joint_fit.fun    # joint-axis

    dchi2 = 2*(L_sep - L_joint)

    logging.info("------------------------------------------------------------")
    logging.info("fit results:")
    logging.info(f"separate-axis  logL = {L_sep:.2f}")
    logging.info(f"joint-axis     logL = {L_joint:.2f}")
    logging.info(f"Δχ² = {dchi2:.2f}")

    # significance: 1 extra parameter (phi0 same instead of independent)
    sigma = np.sqrt(dchi2) if dchi2 > 0 else 0
    logging.info(f"significance ≈ {sigma:.2f} σ")
    logging.info("------------------------------------------------------------")

    # ------------------------------------------------------------
    # plot likelihood curve vs phi0
    # ------------------------------------------------------------
    phi_grid = np.linspace(0,1,200)
    L_curve = []

    for ph in phi_grid:
        p = joint_fit.x.copy()
        p[0] = ph
        L_curve.append(-joint_neg_log_likelihood(p, phi_frb, phi_nu))

    plt.figure(figsize=(10,5))
    plt.plot(phi_grid, L_curve, label="joint log-likelihood")
    plt.axvline(joint_fit.x[0], color="red", linestyle="--", label="best φ0")
    plt.xlabel("sidereal phase φ0")
    plt.ylabel("log-likelihood")
    plt.legend()
    plt.title("joint FRB + neutrino axis likelihood")
    plt.tight_layout()
    plt.savefig("frb_neutrino_joint_axis_likelihood_curve.png")

    logging.info("[INFO] saved → frb_neutrino_joint_axis_likelihood_curve.png")
    logging.info("[done]")


if __name__ == "__main__":
    main()
