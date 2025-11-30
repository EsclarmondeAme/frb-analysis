"""
frb_3d_residual_harmonics.py

take the same unified 3d frb density model as in frb_3d_density_model.py,
remove the mean radial selection function, and compute spherical harmonic
coefficients of the residual field (l <= 4).

outputs:
    - frb_3d_residual_harmonic_coeffs.txt  (list of a_lm for residuals)
    - frb_3d_residual_harmonic_map.png     (angular residual map built from a_lm)
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from frb_3d_density_model import (
    load_catalog,
    convert_to_3d,
    build_density_cube,
    smooth_cube,
    angular_harmonic_reconstruction,
)

L_MAX = 4
SMOOTH_SIGMA = 1.0  # same as in frb_3d_density_model


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def build_residual_cube(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    take a smoothed 3d density cube n(r,theta,phi) and remove the
    mean radial selection function.

    returns
    -------
    residual_norm : ndarray
        residuals normalized by sqrt(expectation) (where expectation > 0),
        shape (Nr, Ntheta, Nphi)
    expected : ndarray
        isotropic expectation cube (same shape as cube)
    """
    # average over angles to get n(r)
    radial_profile = cube.mean(axis=(1, 2))  # shape (Nr,)

    # build isotropic expectation cube
    expected = radial_profile[:, None, None]

    # residuals
    residual = cube - expected

    # normalize by sqrt(expectation) to get something like "sigma" units
    with np.errstate(divide="ignore", invalid="ignore"):
        residual_norm = np.where(
            expected > 0.0,
            residual / np.sqrt(expected),
            0.0,
        )

    # simple stats for sanity
    m = float(np.mean(residual_norm))
    s = float(np.std(residual_norm))
    logging.info(
        "residual cube stats (normalized): mean = %.3f, std = %.3f", m, s
    )

    return residual_norm, expected


def save_coeffs(coeffs: dict[tuple[int, int], complex], filename: str) -> None:
    """
    save a_lm coefficients to a text file.
    """
    lines = []
    for (l, m), a in sorted(coeffs.items()):
        lines.append(
            f"l={l:2d}  m={m:3d}  "
            f"Re(a_lm)={a.real: .6e}  Im(a_lm)={a.imag: .6e}  |a_lm|={abs(a): .6e}"
        )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# residual 3d harmonic coefficients (after radial selection removal)\n")
        for line in lines:
            f.write(line + "\n")

    logging.info("saved: %s", filename)


def summarize_power(coeffs: dict[tuple[int, int], complex], Lmax: int) -> None:
    """
    print a quick summary of which m dominates at each l.
    """
    logging.info("multipole summary (residual field):")
    for l in range(1, Lmax + 1):
        amps = []
        for m in range(-l, l + 1):
            a = coeffs.get((l, m), 0.0 + 0.0j)
            amps.append((m, abs(a)))
        if not amps:
            continue
        m_dom, a_dom = max(amps, key=lambda x: x[1])
        logging.info(
            "  l=%d: dominant m=%2d, |a_lm|=%.3e", l, m_dom, a_dom
        )


def plot_residual_map(
    ang_map: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    filename: str,
) -> None:
    """
    plot the angular residual map reconstructed from the residual cube.
    """
    plt.figure(figsize=(9, 4))
    extent = (phi_edges[0], phi_edges[-1], theta_edges[0], theta_edges[-1])

    im = plt.imshow(
        ang_map,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
    )

    plt.xlabel("phi (deg)")
    plt.ylabel("theta (deg)")
    plt.title("FRB 3d residual harmonic map (after radial selection removal)")
    cbar = plt.colorbar(im)
    cbar.set_label("residual amplitude (summed over r)")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    logging.info("saved: %s", filename)


def main() -> None:
    setup_logging()

    logging.info("=" * 70)
    logging.info("FRB 3D RESIDUAL HARMONICS (L <= %d)", L_MAX)
    logging.info("=" * 70)

    # 1. load unified catalog (same helper as 3d density script)
    df = load_catalog()
    logging.info("loaded unified catalog with N = %d frbs", len(df))

    # 2. convert to 3d spherical coordinates
    r, theta, phi = convert_to_3d(df)
    logging.info("converted to 3d coordinates")

    # 3. build 3d density cube and smooth
    cube, r_edges, theta_edges, phi_edges = build_density_cube(r, theta, phi)
    logging.info(
        "density cube built: shape = (Nr=%d, Ntheta=%d, Nphi=%d)",
        cube.shape[0],
        cube.shape[1],
        cube.shape[2],
    )

    cube_s = smooth_cube(cube, sigma=SMOOTH_SIGMA)
    logging.info("cube smoothed with sigma = %.2f", SMOOTH_SIGMA)

    # 4. remove radial selection (build residual cube)
    residual_cube, expected = build_residual_cube(cube_s)

    # 5. compute spherical harmonics of residual field
    coeffs, ang_map, TH, PH = angular_harmonic_reconstruction(
        residual_cube,
        theta_edges,
        phi_edges,
        Lmax=L_MAX,
    )

    # 6. save coefficients and map, and print a small summary
    save_coeffs(coeffs, "frb_3d_residual_harmonic_coeffs.txt")
    summarize_power(coeffs, L_MAX)

    plot_residual_map(
        ang_map,
        theta_edges,
        phi_edges,
        "frb_3d_residual_harmonic_map.png",
    )

    logging.info("3d residual harmonic analysis complete.")


if __name__ == "__main__":
    main()
