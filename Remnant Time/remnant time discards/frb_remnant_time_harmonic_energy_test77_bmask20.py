#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd

# --------------------------------------------
# spherical → Cartesian
# --------------------------------------------
def sph_to_xyz(ra, dec):
    ra  = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T

# --------------------------------------------
# unified-axis projection (given theta,phi)
# --------------------------------------------
def axis_to_xyz(theta, phi):
    t = np.radians(theta)
    p = np.radians(phi)
    x = np.sin(t) * np.cos(p)
    y = np.sin(t) * np.sin(p)
    z = np.cos(t)
    return np.array([x, y, z])

# --------------------------------------------
# small spherical harmonic energy estimator
# --------------------------------------------
def harmonic_energy(X, ell_list=[1,2,3,4]):
    """
    very low-ell harmonic energy using dot-products with
    simple spherical basis functions.
    This is a surrogate spectral estimator — stable for N~300–1000.
    """
    x = X[:,0]; y = X[:,1]; z = X[:,2]
    E = 0.0

    # ℓ=1 components ~ dipole-like
    if 1 in ell_list:
        E += np.mean(x)**2 + np.mean(y)**2 + np.mean(z)**2

    # ℓ=2 surrogate (quadrupole traces)
    if 2 in ell_list:
        Q1 = np.mean( x**2 - y**2 )
        Q2 = np.mean( 2*x*y )
        Q3 = np.mean( 2*x*z )
        Q4 = np.mean( 2*y*z )
        Q5 = np.mean( 3*z**2 - 1 )
        E += Q1**2 + Q2**2 + Q3**2 + Q4**2 + Q5**2

    # ℓ=3+4 simple polynomial basis (stable approximate)
    if 3 in ell_list:
        E += np.mean(x*y*z)**2
    if 4 in ell_list:
        E += np.mean((x**2 + y**2 - 2*z**2))**2

    return E

# --------------------------------------------
# Monte Carlo null — axis rotations
# --------------------------------------------
def random_axis():
    u = np.random.uniform(-1,1)
    phi = np.random.uniform(0,2*np.pi)
    z = u
    r = np.sqrt(1-z*z)
    return np.array([r*np.cos(phi), r*np.sin(phi), z])

def project_sign(X, axis_xyz):
    return X @ axis_xyz

# --------------------------------------------
# main
# --------------------------------------------
def main(path, n_mc=2000):

    print("================================================")
    print(" FRB REMNANT-TIME HARMONIC ENERGY DISTORTION TEST (77A)")
    print(" Galactic mask: |b| >= 20°")
    print("================================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values
    Rrem = df["theta_unified"].values * 0  # placeholder, corrected below

    # actual remnant-time residual (R>0 or R<0)
    # user’s unified file stores remnant-time as:
    #   z_est  → redshift estimate
    #   dm     → dispersion measure
    # we extract R as:
    R = df["z_est"].values - (df["dm"].values / 1000.0)
    # sign only
    signR = np.sign(R)

    # ----------------------------------------
    # mask |b|>=20°
    # compute Galactic latitude
    import astropy.coordinates as coord
    import astropy.units as u
    sky = coord.SkyCoord(ra=RA*u.deg, dec=Dec*u.deg, frame='icrs')
    b = sky.galactic.b.deg
    mask = np.abs(b) >= 20.0
    RA  = RA[mask]
    Dec = Dec[mask]
    signR = signR[mask]

    print(f"[info] original N={len(df)}, after mask N={len(signR)}")

    # ----------------------------------------
    # prepare coordinates
    X = sph_to_xyz(RA, Dec)
    X_pos = X[signR > 0]
    X_neg = X[signR < 0]

    if len(X_pos)==0 or len(X_neg)==0:
        print("[error] no positive or negative R hemisphere after masking.")
        sys.exit(1)

    # ----------------------------------------
    # real harmonic energies
    E_pos = harmonic_energy(X_pos)
    E_neg = harmonic_energy(X_neg)
    D_real = E_pos - E_neg

    # ----------------------------------------
    # Monte Carlo null: rotate hemisphere assignment
    D_null = np.zeros(n_mc)
    for k in range(n_mc):
        axis = random_axis()
        sign_rand = np.sign(X @ axis)
        Xp = X[sign_rand > 0]
        Xn = X[sign_rand < 0]
        if len(Xp)==0 or len(Xn)==0:
            D_null[k] = 0
        else:
            D_null[k] = harmonic_energy(Xp) - harmonic_energy(Xn)

    # ----------------------------------------
    # statistics
    meanN = np.mean(D_null)
    stdN  = np.std(D_null)
    if stdN == 0:
        p = 1.0
    else:
        p = np.mean( np.abs(D_null - meanN) >= np.abs(D_real - meanN) )

    # ----------------------------------------
    # output
    print("------------------------------------------------")
    print(f"E_pos (R>0)   = {E_pos}")
    print(f"E_neg (R<0)   = {E_neg}")
    print(f"D_real        = {D_real}")
    print("------------------------------------------------")
    print(f"null mean D   = {meanN}")
    print(f"null std D    = {stdN}")
    print(f"p-value       = {p}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> harmonic-energy distribution differs between")
    print("            remnant-time hemispheres (robust).")
    print("  high p -> harmonic energies symmetric; consistent with isotropy.")
    print("================================================")
    print("test 77A complete.")
    print("================================================")


if __name__ == "__main__":
    main(sys.argv[1])
