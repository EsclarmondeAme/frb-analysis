import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time
import os

# ------------------------------------------------------------
# helper: compute sidereal phase from mjd
# ------------------------------------------------------------
def sidereal_phase(mjd):
    # sidereal day = 23h 56m 4.0916s = 0.99726957 solar days
    sid_day = 0.99726957
    return (mjd / sid_day) % 1.0

# ------------------------------------------------------------
# helper: convert ra, dec to unit vector
# ------------------------------------------------------------
def radec_to_vec(ra_deg, dec_deg):
    c = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
    x = np.cos(c.dec.radian)*np.cos(c.ra.radian)
    y = np.cos(c.dec.radian)*np.sin(c.ra.radian)
    z = np.sin(c.dec.radian)
    return np.array([x, y, z]).T

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    print("="*60)
    print("sidereal axis mapping")
    print("="*60)

    df = pd.read_csv("frbs.csv")

    if not {"ra", "dec", "mjd"}.issubset(df.columns):
        print("missing columns ra, dec, mjd in frbs.csv")
        return

    ra = df["ra"].values
    dec = df["dec"].values
    mjd = df["mjd"].values

    # compute sidereal phases
    sid = sidereal_phase(mjd)

    # build unit vectors
    vecs = radec_to_vec(ra, dec)

    # number of bins across 0..1 sidereal phase
    nbins = 24   # 24 → 1 hour resolution
    bins = np.linspace(0, 1, nbins+1)

    # store dominant sky direction per bin
    dominant_dirs = []

    # prepare folder for animation frames
    if not os.path.exists("sidereal_axis_frames"):
        os.makedirs("sidereal_axis_frames")

    print("computing axes for each sidereal bin...")

    for i in range(nbins):
        lo, hi = bins[i], bins[i+1]
        mask = (sid >= lo) & (sid < hi)
        subset = vecs[mask]

        if len(subset) == 0:
            dominant_dirs.append(None)
            continue

        # direction = normalized average vector
        m = subset.mean(axis=0)
        n = np.linalg.norm(m)
        if n > 0:
            m = m / n

        dominant_dirs.append(m)

        # plot frame
        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot(111, projection='mollweide')

        # plot FRBs
        coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs').galactic
        l = coords.l.wrap_at(180*u.deg).radian
        b = coords.b.radian
        ax.scatter(l, b, s=4, c='lightblue', alpha=0.5)

        # plot axis for this bin
        if dominant_dirs[i] is not None:
            vx, vy, vz = dominant_dirs[i]
            caxis = SkyCoord(x=vx, y=vy, z=vz,
                             unit=(u.one, u.one, u.one),
                             representation_type="cartesian").represent_as("spherical")

            gl = caxis.lon.wrap_at(180*u.deg).radian
            gb = caxis.lat.radian
            ax.scatter(gl, gb, c='red', s=200, marker='*', label='axis')

        ax.set_title(f"sidereal phase {lo:.2f} - {hi:.2f}")
        ax.grid(True, alpha=0.3)
        plt.savefig(f"sidereal_axis_frames/frame_{i:02d}.png", dpi=150)
        plt.close()

    print("computing global sidereal dipole...")

    # average all non-null axes
    valid = [v for v in dominant_dirs if v is not None]
    global_vec = np.mean(valid, axis=0)
    global_vec /= np.linalg.norm(global_vec)

    # convert to coordinates
    gv = SkyCoord(x=global_vec[0], y=global_vec[1], z=global_vec[2],
                  unit=(u.one, u.one, u.one),
                  representation_type="cartesian").represent_as("spherical")

    global_ra = gv.lon.deg % 360
    global_dec = gv.lat.deg

    print("")
    print("global sidereal dipole axis:")
    print(f"ra  = {global_ra:.2f} deg")
    print(f"dec = {global_dec:.2f} deg")

    # ------------------------------------------------------------
    # plot global map
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111, projection='mollweide')
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs').galactic
    l = coords.l.wrap_at(180*u.deg).radian
    b = coords.b.radian
    ax.scatter(l, b, s=4, c='steelblue', alpha=0.5)

    gl = gv.lon.wrap_at(180*u.deg).radian
    gb = gv.lat.radian
    ax.scatter(gl, gb, c='gold', s=200, marker='*', edgecolor='black',
               label='global sidereal axis')

    ax.set_title("global sidereal dipole axis")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.savefig("sidereal_axis_map.png", dpi=160)
    plt.close()

    print("")
    print("saved → sidereal_axis_map.png")
    print("saved → sidereal_axis_frames/*.png")
    print("done.")
    print("="*60)


if __name__ == "__main__":
    main()
