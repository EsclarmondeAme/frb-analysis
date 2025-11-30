import numpy as np
import matplotlib.pyplot as plt

# we reuse the helpers from the 3d density script
from frb_3d_density_model import (
    load_catalog,
    convert_to_3d,
    build_density_cube,
    smooth_cube,
)


def compute_radial_expectation(cube):
    """
    given a smoothed 3d density cube n(r, theta, phi),
    compute the mean radial profile n_iso(r) by averaging over angles.
    """
    # cube shape: (nr, ntheta, nphi)
    radial_mean = cube.mean(axis=(1, 2))
    return radial_mean


def build_residual_cube(cube, radial_mean):
    """
    build a residual significance cube:

        residual = (n_obs - n_iso(r)) / sqrt(n_iso(r))

    so positive values are over-densities, negative are deficits.
    """
    # broadcast radial profile over angles
    expected = radial_mean[:, None, None]
    eps = 1e-6
    residual = (cube - expected) / np.sqrt(expected + eps)
    return residual


def project_angular(residual_cube, theta_edges, phi_edges, fname="frb_3d_residual_ang.png"):
    """
    integrate residual significance along radius and make an angular map.
    """
    # sum significance along r (you could also use mean; sum emphasises coherent structure)
    ang_resid = residual_cube.sum(axis=0)

    # symmetric colour range
    vmax = np.nanmax(np.abs(ang_resid))
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        ang_resid,
        origin="lower",
        aspect="auto",
        extent=[phi_edges[0], phi_edges[-1], theta_edges[0], theta_edges[-1]],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("residual significance Σ (summed over r)")

    ax.set_xlabel("phi (deg)")
    ax.set_ylabel("theta (deg)")
    ax.set_title("FRB 3d residual map (after removing radial selection)")

    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"saved: {fname}")


def main():
    print("loading unified catalog…")
    df = load_catalog()
    print("N =", len(df))

    print("converting to 3d…")
    r, theta, phi = convert_to_3d(df)

    print("building 3d density cube…")
    cube, r_edges, theta_edges, phi_edges = build_density_cube(r, theta, phi)

    print("smoothing cube…")
    cube_s = smooth_cube(cube, sigma=1.5)

    print("computing radial expectation (isotropic selection)…")
    radial_mean = compute_radial_expectation(cube_s)

    print("building residual cube…")
    resid_cube = build_residual_cube(cube_s, radial_mean)

    # basic stats
    finite = np.isfinite(resid_cube)
    print("residual stats: mean = {:.3f}, std = {:.3f}, max = {:.3f}, min = {:.3f}".format(
        np.nanmean(resid_cube[finite]),
        np.nanstd(resid_cube[finite]),
        np.nanmax(resid_cube[finite]),
        np.nanmin(resid_cube[finite]),
    ))

    print("projecting to angular residual map…")
    project_angular(resid_cube, theta_edges, phi_edges)

    print("3d residual model complete.")


if __name__ == "__main__":
    main()
