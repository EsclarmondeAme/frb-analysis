import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

# =============================================================
# LOAD UNIFIED CATALOG (SAME LOGIC AS 3D MODEL & RESIDUAL MODEL)
# =============================================================
def load_catalog():
    paths = [
        "chime_frb_catalog1.csv",
        "frbs.csv"
    ]
    dfs = []
    for p in paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))

    df = pd.concat(dfs, ignore_index=True)

    # dm column
    dm_col = None
    for cand in ["dm", "dm_fitb", "bonsai_dm"]:
        if cand in df.columns:
            dm_col = cand
            break
    if dm_col is None:
        raise ValueError("No DM column found in unified catalog!")

    df = df.rename(columns={dm_col: "dm"})
    df = df.dropna(subset=["ra", "dec", "dm"])

    # rough comoving distance estimate (scaled DM)
    df["r"] = df["dm"] * 1.0

    return df


# =============================================================
# CONVERT CATALOG TO 3D CARTESIAN
# =============================================================
def to_xyz(df):
    coords = SkyCoord(ra=df["ra"].values * u.deg,
                      dec=df["dec"].values * u.deg,
                      distance=df["r"].values * u.Mpc,
                      frame="icrs")

    x = coords.cartesian.x.value
    y = coords.cartesian.y.value
    z = coords.cartesian.z.value
    return x, y, z


# =============================================================
# GRID FRBS INTO 3D DENSITY CUBE
# =============================================================
def build_cube(x, y, z, Nr=40, Ntheta=64, Nphi=128):
    # spherical conversion
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / np.clip(r, 1e-6, None))     # 0..pi
    phi = np.mod(np.arctan2(y, x), 2*np.pi)           # 0..2pi

    r_bins = np.linspace(0, r.max(), Nr + 1)
    th_bins = np.linspace(0, np.pi, Ntheta + 1)
    ph_bins = np.linspace(0, 2*np.pi, Nphi + 1)

    cube, _ = np.histogramdd(
        (r, theta, phi),
        bins=(r_bins, th_bins, ph_bins)
    )

    return cube, r_bins, th_bins, ph_bins


# =============================================================
# PROJECT SHELL INTO ANGULAR MAP
# =============================================================
def project_shell(shell):
    return shell  # already Ntheta × Nphi


# =============================================================
# MAIN PCA DRIVER
# =============================================================
def main():
    print("INFO | ======================================================")
    print("INFO | FRB 3D RESIDUAL PCA SHELLS")
    print("INFO | ======================================================")

    df = load_catalog()
    print(f"INFO | loaded unified catalog with N = {len(df)}")

    x, y, z = to_xyz(df)
    print("INFO | converted to 3d cartesian")

    cube, r_bins, th_bins, ph_bins = build_cube(x, y, z)
    print(f"INFO | 3d cube built with shape = {cube.shape}")

    cube_s = gaussian_filter(cube, sigma=1.0)
    print("INFO | cube smoothed")

    # build radial expectation (selection function)
    radial_exp = cube_s.mean(axis=(1,2), keepdims=True)
    cube_res = cube_s - radial_exp
    print("INFO | residual cube computed")

    Nr, Ntheta, Nphi = cube_res.shape
    maps = []

    # flatten each shell
    for i in range(Nr):
        shell = cube_res[i]
        flat = shell.flatten()
        maps.append(flat)

    M = np.array(maps)
    M = M - M.mean(axis=0, keepdims=True)

    # PCA
    C = np.cov(M, rowvar=True)
    eigvals, eigvecs = np.linalg.eigh(C)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    np.savetxt("frb_3d_pca_shell_eigenvalues.txt", eigvals)

    print("INFO | PCA complete.")
    print("INFO | top eigenvalues:", eigvals[:6])

    # =============================================================
    # SAVE FIRST FEW PCA MODES AS SKY MAPS
    # =============================================================
    Nmodes = 6
    for k in range(Nmodes):
        mode = eigvecs[:, k]
        shell_k = mode[None, :]  # project eigenvector to angular space

        # expand eigenvector back to 2D shape
        full_maps = []
        for i in range(Nr):
            m = M[i] * mode[i]

            full_maps.append(m)

        map2d = np.sum(np.array(full_maps), axis=0)
        map2d = map2d.reshape(Ntheta, Nphi)

        plt.figure(figsize=(10,4))
        plt.imshow(map2d, origin="lower", aspect="auto",
                   extent=[0,360,0,180], cmap="coolwarm")
        plt.colorbar(label="PCA mode intensity")
        plt.title(f"PCA eigenmode k={k}")
        plt.xlabel("phi (deg)")
        plt.ylabel("theta (deg)")
        plt.tight_layout()
        plt.savefig(f"frb_3d_pca_mode_{k}.png")
        plt.close()

    print("INFO | saved PCA modes: frb_3d_pca_mode_0..5.png")
    print("INFO | PCA shells complete.")


if __name__ == "__main__":
    main()
