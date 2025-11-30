import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------------------
# helper: compute local grain intensity (same as Test 45/46)
# ----------------------------------------------------------
def compute_grain_intensity(ra, dec, k=20):
    n = len(ra)
    G = np.zeros(n)

    for i in range(n):
        dra = (ra - ra[i]) * np.cos(np.deg2rad(dec[i]))
        ddec = (dec - dec[i])
        dist2 = dra*dra + ddec*ddec
        idx = np.argsort(dist2)[1:k+1]
        G[i] = np.mean(np.sqrt(dist2[idx]))
    return G


# ----------------------------------------------------------
# helper: spatial anisotropy gradient (Test 47 style)
# ----------------------------------------------------------
def compute_spatial_gradient(theta, phi):
    # convert to radians for safe spacing
    th = np.deg2rad(theta)
    ph = np.deg2rad(phi)

    # compute gradients separately
    dth = np.gradient(th)
    dph = np.gradient(ph)

    # gradient magnitude
    return np.sqrt(dth * dth + dph * dph)



# ----------------------------------------------------------
# compute energy gradient encoding (Test 48)
# ----------------------------------------------------------
def energy_gradient_encoding(df):
    fields = ["dm", "snr", "fluence", "width", "z_est"]
    X = df[fields].values
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    slopes = np.gradient(Xn, axis=0)
    return slopes


# ----------------------------------------------------------
# compute cross-energy interactions (Test 49)
# ----------------------------------------------------------
def cross_energy_terms(df):
    fields = ["dm", "snr", "fluence", "width", "z_est"]
    X = df[fields].values
    pairs = []

    for i in range(len(fields)):
        for j in range(i+1, len(fields)):
            pairs.append(X[:, i] * X[:, j])

    return np.vstack(pairs).T


# ----------------------------------------------------------
# compute harmonic helicity coefficients (Test 41)
# ----------------------------------------------------------
def harmonic_features(phi):
    a1 = np.cos(np.deg2rad(phi))
    b1 = np.sin(np.deg2rad(phi))
    a2 = np.cos(2*np.deg2rad(phi))
    b2 = np.sin(2*np.deg2rad(phi))
    return np.vstack([a1, b1, a2, b2]).T


# ----------------------------------------------------------
# MAIN: Unified Latent-Geometry Field (LGM)
# ----------------------------------------------------------
def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python frb_unified_latent_geometry_field_test50.py frbs_unified.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    print(f"loaded {len(df)} FRBs")

    # --------------------------------------
    # 1. Compute all components
    # --------------------------------------
    print("computing grain intensity ...")
    G = compute_grain_intensity(df["ra"].values, df["dec"].values)

    print("computing spatial gradient ...")
    SAG = compute_spatial_gradient(df["theta_unified"].values,
                                   df["phi_unified"].values)

    print("computing energy gradients ...")
    EGR = energy_gradient_encoding(df)
    EGR = EGR.sum(axis=1)

    print("computing cross energy interactions ...")
    CE = cross_energy_terms(df)

    print("computing harmonic helicity features ...")
    HF = harmonic_features(df["phi_unified"].values)

    # --------------------------------------
    # build feature matrix
    # --------------------------------------
    X = np.column_stack([G, SAG, EGR, CE, HF])
    y = df["theta_unified"].values  # target field

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # unified LGM model
    model = LinearRegression()
    model.fit(Xs, y)
    y_pred = model.predict(Xs)

    RSS_LGM = mean_squared_error(y, y_pred) * len(y)

    # --------------------------------------
    # Monte Carlo comparison
    # --------------------------------------
    N_MC = 5000
    RSS_null = np.zeros(N_MC)

    print("running Monte Carlo null ...")
    for i in tqdm(range(N_MC)):
        y_shuf = np.random.permutation(y)
        model.fit(Xs, y_shuf)
        RSS_null[i] = mean_squared_error(y_shuf, model.predict(Xs)) * len(y)

    rss_mean = np.mean(RSS_null)
    rss_std  = np.std(RSS_null)

    p_value = np.mean(RSS_null <= RSS_LGM)

    # --------------------------------------
    # results
    # --------------------------------------
    print("====================================================================")
    print(" FRB UNIFIED LATENT-GEOMETRY FIELD (LGM) TEST (TEST 50)")
    print("====================================================================")
    print(f"RSS_LGM      = {RSS_LGM:.6f}")
    print(f"null mean    = {rss_mean:.6f}")
    print(f"null std     = {rss_std:.6f}")
    print(f"p-value      = {p_value:.6f}")
    print("--------------------------------------------------------------------")
    print("interpretation:")
    print(" - low p => unified latent field explains structure beyond chance")
    print(" - high p => grain, energy gradients, anisotropy do not unify")
    print("====================================================================")
    print("test 50 complete.")
    print("====================================================================")


if __name__ == "__main__":
    main()
