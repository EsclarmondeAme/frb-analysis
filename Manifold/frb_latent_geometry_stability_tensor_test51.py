import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------
# grain intensity
# -----------------------------------------------------
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

# -----------------------------------------------------
# spatial anisotropy gradient
# -----------------------------------------------------
def compute_spatial_gradient(theta, phi):
    th = np.deg2rad(theta)
    ph = np.deg2rad(phi)
    dth = np.gradient(th)
    dph = np.gradient(ph)
    return np.sqrt(dth*dth + dph*dph)

# -----------------------------------------------------
# energy gradient encoding
# -----------------------------------------------------
def energy_gradient_encoding(df):
    fields = ["dm", "snr", "fluence", "width", "z_est"]
    X = df[fields].values
    Xn = (X - X.mean(axis=0)) / X.std(axis=0)
    slopes = np.gradient(Xn, axis=0)
    return slopes.sum(axis=1)

# -----------------------------------------------------
# cross energy interactions
# -----------------------------------------------------
def cross_energy_terms(df):
    fields = ["dm", "snr", "fluence", "width", "z_est"]
    X = df[fields].values
    out = []
    for i in range(len(fields)):
        for j in range(i+1, len(fields)):
            out.append(X[:, i] * X[:, j])
    return np.vstack(out).T

# -----------------------------------------------------
# harmonic features
# -----------------------------------------------------
def harmonic_features(phi):
    ph = np.deg2rad(phi)
    return np.vstack([
        np.cos(ph), np.sin(ph),
        np.cos(2*ph), np.sin(2*ph)
    ]).T

# -----------------------------------------------------
# build feature matrix
# -----------------------------------------------------
def build_features(df):
    G   = compute_grain_intensity(df["ra"].values, df["dec"].values)
    SAG = compute_spatial_gradient(df["theta_unified"], df["phi_unified"])
    EGR = energy_gradient_encoding(df)
    CE  = cross_energy_terms(df)
    HF  = harmonic_features(df["phi_unified"])
    return np.column_stack([G, SAG, EGR, CE, HF])

# -----------------------------------------------------
# MAIN — latent geometry stability tensor
# -----------------------------------------------------
def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python frb_latent_geometry_stability_tensor_test51.py frbs_unified.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    print(f"loaded {len(df)} FRBs")

    # baseline features
    X = build_features(df)
    y = df["theta_unified"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    base_model = LinearRegression().fit(Xs, y)
    y0 = base_model.predict(Xs)

    # ==================================================
    # compute the five stability components
    # ==================================================

    N = len(df)
    B = 200  # bootstrap iterations

    T_resample = []
    T_energy   = []
    T_sky      = []
    T_feature  = []
    T_cross    = []

    print("computing stability tensor ...")

    for _ in tqdm(range(B)):
        # --- resample ---
        idx = np.random.choice(N, N, replace=True)
        Xb = Xs[idx]
        yb = y[idx]
        mb = LinearRegression().fit(Xb, yb)
        yb_pred = mb.predict(Xs)
        T_resample.append(mean_squared_error(y0, yb_pred))

        # --- energy jitter ---
        J = df.copy()
        J[["dm","snr","fluence","width","z_est"]] *= (1 + 0.05*np.random.randn(N,5))
        Xj = build_features(J)
        Xj = scaler.transform(Xj)
        yj_pred = base_model.predict(Xj)
        T_energy.append(mean_squared_error(y0, yj_pred))

        # --- sky jitter ---
        J2 = df.copy()
        J2["ra"]  += np.random.randn(N)*0.05
        J2["dec"] += np.random.randn(N)*0.05
        Xs2 = build_features(J2)
        Xs2 = scaler.transform(Xs2)
        ys2_pred = base_model.predict(Xs2)
        T_sky.append(mean_squared_error(y0, ys2_pred))

        # --- feature masking ---
        Xm = Xs.copy()
        mask_col = np.random.randint(0, Xs.shape[1])
        Xm[:, mask_col] = 0
        ym_pred = base_model.predict(Xm)
        T_feature.append(mean_squared_error(y0, ym_pred))

        # --- cross perturbation ---
        Xc = Xs.copy()
        Xc += 0.05*np.random.randn(*Xc.shape)
        yx_pred = base_model.predict(Xc)
        T_cross.append(mean_squared_error(y0, yx_pred))

    # aggregate
    T_resample = np.mean(T_resample)
    T_energy   = np.mean(T_energy)
    T_sky      = np.mean(T_sky)
    T_feature  = np.mean(T_feature)
    T_cross    = np.mean(T_cross)

    S_total = T_resample + T_energy + T_sky + T_feature + T_cross

    # --------------------------------------------------
    # Monte Carlo null
    # --------------------------------------------------
    print("running Monte Carlo null ...")
    N_MC = 20000
    null = np.zeros(N_MC)

    for i in tqdm(range(N_MC)):
        y_shuf = np.random.permutation(y)
        base = LinearRegression().fit(Xs, y_shuf)
        yb = base.predict(Xs)
        null[i] = mean_squared_error(y_shuf, yb)

    p = np.mean(null <= S_total)

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------
    print("===================================================================")
    print(" FRB LATENT GEOMETRY STABILITY TENSOR (TEST 51)")
    print("===================================================================")
    print(f"T_resample = {T_resample:.6f}")
    print(f"T_energy   = {T_energy:.6f}")
    print(f"T_sky      = {T_sky:.6f}")
    print(f"T_feature  = {T_feature:.6f}")
    print(f"T_cross    = {T_cross:.6f}")
    print("-------------------------------------------------------------------")
    print(f"S_total    = {S_total:.6f}")
    print(f"null mean  = {np.mean(null):.6f}")
    print(f"null std   = {np.std(null):.6f}")
    print(f"p-value    = {p:.6f}")
    print("-------------------------------------------------------------------")
    print("interpretation:")
    print(" - low p: unified latent geometry is stable under perturbations")
    print(" - high p: field collapses => likely accidental")
    print("===================================================================")
    print("test 51 complete.")
    print("===================================================================")

if __name__ == "__main__":
    main()
