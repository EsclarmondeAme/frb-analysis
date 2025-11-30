import numpy as np

# -------------------------------------------------------
# cross-layer physics helpers for coincidence scoring
# -------------------------------------------------------

# characteristic energy scale (TeV) for strong coupling
E_STAR_TEV = 10.0      # tune later if you like
ALPHA = 1.0            # power-law index in g(E)
E_CUT_TEV = 30.0       # exponential cutoff scale (TeV)

# overall coupling normalisation (relative; absolute scale not important
# for ranking coincidences)
ETA0 = 1e-4

# coherence scales (tuneable)
TAU_COH_SEC = 60.0     # characteristic time coherence (~ 1 minute)
THETA_COH_DEG = 5.0    # characteristic angular coherence (~ 5 degrees)


def coupling_efficiency(E_tev):
    """
    cross-layer coupling efficiency η(E):
        η(E) = η0 * (E/E*)^α * exp(-E_cut / E)

    E_tev can be scalar or array-like (TeV).
    """
    E = np.asarray(E_tev, dtype=float)
    # avoid division by zero
    E = np.clip(E, 1e-3, None)
    return ETA0 * (E / E_STAR_TEV) ** ALPHA * np.exp(-E_CUT_TEV / E)


def time_coherence_weight(dt_sec):
    """
    gaussian time coherence weight:
        w_t = exp( - (Δt / τ_coh)^2 )

    dt_sec can be scalar or array-like (seconds).
    """
    dt = np.abs(np.asarray(dt_sec, dtype=float))
    return np.exp(-(dt / TAU_COH_SEC) ** 2)


def angular_coherence_weight(theta_deg):
    """
    gaussian angular coherence weight:
        w_a = exp( - (θ / θ_coh)^2 )

    theta_deg can be scalar or array-like (degrees).
    """
    th = np.abs(np.asarray(theta_deg, dtype=float))
    return np.exp(-(th / THETA_COH_DEG) ** 2)
