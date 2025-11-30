#!/usr/bin/env python3
import numpy as np

# ============================================================
# utilities
# ============================================================
def angdist(l1, b1, l2, b2):
    """great-circle separation in degrees between two galactic directions."""
    l1 = np.deg2rad(l1)
    b1 = np.deg2rad(b1)
    l2 = np.deg2rad(l2)
    b2 = np.deg2rad(b2)
    return np.rad2deg(
        np.arccos(
            np.sin(b1)*np.sin(b2) +
            np.cos(b1)*np.cos(b2)*np.cos(l1 - l2)
        )
    )


def random_isotropic_axis():
    """draw a random direction on the sphere: (l,b) in degrees."""
    # l uniform in [0,360)
    l = np.random.uniform(0.0, 360.0)
    # sin(b) uniform in [-1,1]
    u = np.random.uniform(-1.0, 1.0)
    b = np.rad2deg(np.arcsin(u))
    return l, b


def max_pairwise_separation(axes):
    """axes is array of shape (N,2): (l,b) deg; return max separation between any pair."""
    N = len(axes)
    max_sep = 0.0
    for i in range(N):
        for j in range(i+1, N):
            l1, b1 = axes[i]
            l2, b2 = axes[j]
            d = angdist(l1, b1, l2, b2)
            if d > max_sep:
                max_sep = d
    return max_sep


# ============================================================
# main
# ============================================================
def main():
    print("=====================================================================")
    print("UNIFIED AXIS STACKING TEST")
    print("=====================================================================")
    print("testing whether multiple claimed preferred axes are unusually aligned")
    print("compared to random independent sky directions.")
    print("---------------------------------------------------------------------")

    # --------------------------------------------------------
    # define your axes here (l,b in degrees, galactic)
    #
    # you can extend this list as you add datasets.
    # label, l, b
    # --------------------------------------------------------
    axes_info = [
        ("CMB hemispherical asymmetry", 152.62,   4.03),   # from your earlier script
        ("FRB unified axis",            159.85,  -0.51),   # from unified_best_fit_axis.py
        ("Clock sidereal modulation",   163.54,  -3.93),   # from refined_unified_axis_test
        # add more like:
        # ("radio dipole", l_xx, b_xx),
        # ("NVSS number-count dipole", l_yy, b_yy),
    ]

    labels = [x[0] for x in axes_info]
    axes   = np.array([[x[1], x[2]] for x in axes_info])
    N = len(axes)

    print("axes included:")
    for lbl, (l,b) in zip(labels, axes):
        print(f"  - {lbl:30s}  (l={l:7.2f}°, b={b:7.2f}°)")
    print("---------------------------------------------------------------------")

    # --------------------------------------------------------
    # observed pairwise separations
    # --------------------------------------------------------
    print("pairwise angular separations (degrees):")
    for i in range(N):
        for j in range(i+1, N):
            d = angdist(axes[i,0], axes[i,1], axes[j,0], axes[j,1])
            print(f"  {labels[i]:30s} ↔ {labels[j]:30s} : {d:6.2f}°")
    print("---------------------------------------------------------------------")

    obs_max_sep = max_pairwise_separation(axes)
    print(f"maximum pairwise separation between all axes: {obs_max_sep:.2f}°")
    print("smaller values mean tighter clustering.")
    print("---------------------------------------------------------------------")

    # --------------------------------------------------------
    # monte carlo null: N independent isotropic axes
    # --------------------------------------------------------
    sims = 200000
    max_seps = np.empty(sims, dtype=float)

    for k in range(sims):
        rnd_axes = np.array([random_isotropic_axis() for _ in range(N)])
        max_seps[k] = max_pairwise_separation(rnd_axes)

    # fraction of random realizations as tight or tighter than observed
    p_value = np.mean(max_seps <= obs_max_sep)

    print("null distribution summary (isotropic independent axes):")
    print(f"  mean max separation   = {np.mean(max_seps):6.2f}°")
    print(f"  median max separation = {np.median(max_seps):6.2f}°")
    print(f"  10th percentile       = {np.percentile(max_seps, 10):6.2f}°")
    print(f"  1st percentile        = {np.percentile(max_seps,  1):6.2f}°")
    print("---------------------------------------------------------------------")
    print(f"observed max separation = {obs_max_sep:6.2f}°")
    print(f"p-value (null max_sep ≤ observed) = {p_value:.4e}")
    print("---------------------------------------------------------------------")

    # --------------------------------------------------------
    # scientific verdict
    # --------------------------------------------------------
    print("=====================================================================")
    print("SCIENTIFIC VERDICT")
    print("=====================================================================")
    if p_value < 1e-3:
        print("the stacked axes are extremely tightly clustered compared to random.")
        print("→ strong evidence for a common preferred direction.")
    elif p_value < 0.01:
        print("the stacked axes are significantly more clustered than random.")
        print("→ moderate-to-strong evidence for a shared preferred axis.")
    elif p_value < 0.05:
        print("the stacked axes are mildly more clustered than random.")
        print("→ suggestive evidence for a common axis, but not conclusive.")
    else:
        print("the degree of clustering is compatible with random axes.")
        print("→ no strong evidence that these axes share a single physical origin.")
    print("=====================================================================")
    print("analysis complete.")
    print("=====================================================================")


if __name__ == "__main__":
    main()
