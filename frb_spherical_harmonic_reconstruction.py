import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y   # replacement for deprecated sph_harm
from astropy.coordinates import SkyCoord, Galactocentric, Galactic, ICRS
import astropy.units as u

###############################################
# CONFIG
###############################################
UNIFIED_L = 159.85    # unified axis l, b (deg)
UNIFIED_B = -0.51
LMAX = 4               # reconstruct up to ℓ = 4
NGRID = 200            # grid resolution for map reconstruction
MC_TRIALS = 2000       # monte-carlo trials

###############################################
# coordinate transforms
###############################################
def compute_axis_frame(ra_deg, dec_deg):
    coords = SkyCoord(ra_deg*u.deg, dec_deg*u.deg, frame="icrs")
    gal = coords.galactic
    
    # convert to axis-aligned frame
    axis = SkyCoord(l=UNIFIED_L*u.deg, b=UNIFIED_B*u.deg, frame="galactic")
    # rotate into axis frame using position angle rotation
    lon = gal.l.radian
    lat = gal.b.radian

    # convert to cartesian
    x = np.cos(lat)*np.cos(lon)
    y = np.cos(lat)*np.sin(lon)
    z = np.sin(lat)

    # axis unit vector
    ax = np.cos(axis.b.radian)*np.cos(axis.l.radian)
    ay = np.cos(axis.b.radian)*np.sin(axis.l.radian)
    az = np.sin(axis.b.radian)

    # rotation axis (north pole to unified axis)
    # compute rotated zenith angle and azimuth
    # spherical law: new_z = dot(v, axis)
    new_z = x*ax + y*ay + z*az
    new_theta = np.arccos(np.clip(new_z, -1, 1))

    # azimuth: compute orthonormal basis
    # axis basis vectors
    # note: we only need approximate φ for spherical harmonic evaluation
    xp = ax
    yp = ay
    zp = az

    # cross to find phi-like angle
    # project into plane perpendicular to axis
    vx = x - new_z * xp
    vy = y - new_z * yp
    vz = z - new_z * zp
    
    phi = np.arctan2(vy, vx)

    return new_theta, phi


###############################################
# compute spherical harmonic expansion
###############################################
def compute_alm(theta, phi, lmax):
    alm = {}

    for l in range(lmax+1):
        for m in range(-l, l+1):
            Ylm = sph_harm_y(m, l, phi, theta)
            coeff = np.sum(Ylm)
            alm[(l,m)] = coeff

    return alm


###############################################
# reconstruct map from a_lm
###############################################
def reconstruct_map(alm, lmax, Ngrid=200):
    theta_grid = np.linspace(0, np.pi, Ngrid)
    phi_grid = np.linspace(-np.pi, np.pi, Ngrid)
    map_grid = np.zeros((Ngrid, Ngrid))

    for i, th in enumerate(theta_grid):
        for j, ph in enumerate(phi_grid):
            val = 0
            for l in range(lmax+1):
                for m in range(-l, l+1):
                    val += alm[(l,m)] * sph_harm_y(m, l, ph, th)
            map_grid[i,j] = val.real

    return theta_grid, phi_grid, map_grid


###############################################
# mask functions
###############################################
def galactic_mask(theta, phi, ra, dec, cut=20):
    coord = SkyCoord(ra*u.deg, dec*u.deg)
    return np.abs(coord.galactic.b.deg) >= cut

def ecliptic_mask(theta, phi, ra, dec, cut=20):
    coord = SkyCoord(ra*u.deg, dec*u.deg)
    beta = coord.barycentrictrueecliptic.lat.deg
    return np.abs(beta) >= cut

def supergalactic_mask(theta, phi, ra, dec, cut=20):
    coord = SkyCoord(ra*u.deg, dec*u.deg)
    sgb = coord.supergalactic.b.deg
    return np.abs(sgb) >= cut

###############################################
# monte-carlo significance
###############################################
def mc_significance(theta, phi, lmax, real_power):
    powers = []

    n = len(theta)
    for _ in range(MC_TRIALS):
        th_rand = np.arccos(1 - 2*np.random.rand(n))
        ph_rand = np.random.uniform(-np.pi, np.pi, n)
        alm_null = compute_alm(th_rand, ph_rand, lmax)

        power = np.zeros(lmax+1)
        for l in range(lmax+1):
            Cl = 0
            for m in range(-l, l+1):
                Cl += np.abs(alm_null[(l,m)])**2
            power[l] = Cl / (2*l+1)
        powers.append(power)

    powers = np.array(powers)
    pvals = np.mean(powers >= real_power, axis=0)
    return pvals


###############################################
# main
###############################################
def main():
    df = pd.read_csv("frbs.csv").dropna(subset=["ra","dec"])

    # axis frame
    theta, phi = compute_axis_frame(df["ra"].values, df["dec"].values)

    ###############################################
    # compute real a_lm
    ###############################################
    alm = compute_alm(theta, phi, LMAX)

    # power spectrum
    Cl = np.zeros(LMAX+1)
    for l in range(LMAX+1):
        total = 0
        for m in range(-l, l+1):
            total += np.abs(alm[(l,m)])**2
        Cl[l] = total / (2*l+1)

    ###############################################
    # monte-carlo significance
    ###############################################
    pvals = mc_significance(theta, phi, LMAX, Cl)

    print("===== FULL HARMONIC RESULT (ℓ ≤ 4) =====")
    for l in range(1, LMAX+1):
        print(f"ℓ={l}: C_l_real={Cl[l]:.4e}   p_MC={pvals[l]:.4f}")

    ###############################################
    # reconstruct map
    ###############################################
    thg, phg, map_grid = reconstruct_map(alm, LMAX, NGRID)

    plt.figure(figsize=(10,6))
    plt.imshow(map_grid, origin="lower", extent=[-180,180,0,180], cmap="viridis")
    plt.colorbar()
    plt.title("FRB harmonic reconstruction (ℓ ≤ 4)")
    plt.xlabel("phi (deg)")
    plt.ylabel("theta (deg)")
    plt.savefig("frb_harmonic_map.png", dpi=200)
    plt.close()

    ###############################################
    # save coefficients
    ###############################################
    with open("frb_alm_lmax4.txt","w") as f:
        for l in range(LMAX+1):
            for m in range(-l, l+1):
                f.write(f"{l}\t{m}\t{alm[(l,m)].real:.6e}\t{alm[(l,m)].imag:.6e}\n")

    print("saved: frb_harmonic_map.png, frb_alm_lmax4.txt")
    print("analysis complete.")


if __name__ == "__main__":
    main()
