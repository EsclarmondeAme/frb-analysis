import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter

# =====================================================
# CONFIG
# =====================================================

NR   = 40
NTH  = 64
NPH  = 128
SMOOTH_SIGMA = 1.0

isos = [0.5, 1.0, 2.0, 3.0]   # isosurfaces (residual sigma units)

# unified catalog file
CAT_FILE = "unified_frb_catalog.csv"

# =====================================================
# utilities
# =====================================================

def load_catalog():
    df = pd.read_csv(CAT_FILE)

    # must contain: ra, dec, dm, z_est (or DM-derived proxy)
    for col in ["ra", "dec", "dm"]:
        if col not in df.columns:
            raise ValueError(f"catalog missing required column: {col}")

    # assign coarse distance if no redshift
    if "z_est" in df.columns:
        z = df["z_est"].to_numpy()
    else:
        # estimate “distance” from DM
        z = 0.001 * df["dm"].to_numpy()

    ra  = df["ra"].to_numpy()
    dec = df["dec"].to_numpy()
    return ra, dec, z


def frb_to_xyz(ra, dec, r):
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    x = r * np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra))
    y = r * np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra))
    z = r * np.sin(np.deg2rad(dec))
    return x, y, z


def build_cube(xs, ys, zs):
    cube = np.zeros((NR, NTH, NPH), dtype=float)

    # spherical coordinates
    r   = np.sqrt(xs**2 + ys**2 + zs**2)
    th  = np.arccos(zs / np.clip(r, 1e-9, None))      # 0..pi
    ph  = np.mod(np.arctan2(ys, xs), 2*np.pi)         # 0..2pi

    # indices
    ri  = np.clip((r  / r.max()) * (NR-1),  0, NR-1 ).astype(int)
    thi = np.clip((th / np.pi) * (NTH-1),   0, NTH-1).astype(int)
    phi = np.clip((ph / (2*np.pi)) * (NPH-1), 0, NPH-1).astype(int)

    for i in range(len(xs)):
        cube[ri[i], thi[i], phi[i]] += 1

    return cube


def compute_residual_cube(cube):
    # expectation under isotropy = radial average
    mean_r = cube.mean(axis=(1,2))   # shape (NR,)
    exp = np.repeat(mean_r[:,None,None], NTH, axis=1)
    exp = np.repeat(exp, NPH, axis=2)

    residual = cube - exp
    std = residual.std()
    return residual / std


def save_mesh(vertices, faces, outname):
    # simple ply writer
    with open(outname, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for p in faces:
            f.write(f"3 {p[0]} {p[1]} {p[2]}\n")


# =====================================================
# MAIN
# =====================================================

def main():
    print("loading unified catalog…")
    ra, dec, z = load_catalog()
    print(f"N = {len(ra)}")

    print("converting to 3d…")
    xs, ys, zs = frb_to_xyz(ra, dec, z)

    print("building 3d cube…")
    cube = build_cube(xs, ys, zs)

    print("smoothing…")
    cube = gaussian_filter(cube, SMOOTH_SIGMA)

    print("computing residual cube…")
    res = compute_residual_cube(cube)

    # normalize cube radial size to physical cartesian cube
    Rmax = 1.0
    xv = np.linspace(-Rmax, Rmax, NR)
    yv = np.linspace(-Rmax, Rmax, NTH)
    zv = np.linspace(-Rmax, Rmax, NPH)

    # marching cubes requires 3d cartesian grid
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')

    # run isosurface extraction
    for iso in isos:
        print(f"extracting isosurface at level = {iso}…")
        verts, faces, normals, values = marching_cubes(res, level=iso)

        out_png = f"frb_3d_isosurface_{iso:.1f}.png"
        out_ply = f"frb_3d_isosurface_{iso:.1f}.ply"

        # plot
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(verts[:,0], verts[:,1], faces, verts[:,2],
                        linewidth=0.2, antialiased=True, alpha=0.85)

        ax.set_title(f"FRB 3D Isosurface (residual={iso:.1f})")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"saved: {out_png}")

        save_mesh(verts, faces, out_ply)
        print(f"saved mesh: {out_ply}")

    print("marching-cubes 3d isosurface model complete.")


if __name__ == "__main__":
    main()
