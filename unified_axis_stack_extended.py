import json
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import itertools

# ----------------------------------------------------
# angular distance (degrees)
# ----------------------------------------------------
def angdist(l1, b1, l2, b2):
    c1 = SkyCoord(l=l1*u.deg, b=b1*u.deg, frame='galactic')
    c2 = SkyCoord(l=l2*u.deg, b=b2*u.deg, frame='galactic')
    return c1.separation(c2).deg

# ----------------------------------------------------
# load axes.json
# ----------------------------------------------------
def load_axes(json_path="axes.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["axes"]

# ----------------------------------------------------
# null simulation for N independent isotropic axes
# ----------------------------------------------------
def simulate_null(n_axes, nsim=50000):
    max_seps = []
    for _ in range(nsim):
        # random galactic directions
        l = np.random.uniform(0, 360, n_axes)
        b = np.degrees(np.arcsin(np.random.uniform(-1, 1, n_axes)))  

        # compute max separation
        seps = []
        for i, j in itertools.combinations(range(n_axes), 2):
            seps.append(angdist(l[i], b[i], l[j], b[j]))
        max_seps.append(max(seps))

    return np.array(max_seps)

# ----------------------------------------------------
# plotting
# ----------------------------------------------------
def plot_axes(axes, save_path="axis_stack_sky.png"):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection="aitoff")
    ax.grid(True)

    for ax0 in axes:
        l = np.radians(ax0["l"] - 180)        # center plot
        b = np.radians(ax0["b"])
        ax.scatter(l, b, marker='o', s=40)
        ax.text(l, b, ax0["name"], fontsize=8)

    plt.title("stacked axes: galactic aitoff projection", pad=20)
    plt.savefig(save_path, dpi=300)
    plt.close()

# ----------------------------------------------------
# main
# ----------------------------------------------------
def main():
    print("="*70)
    print("UNIFIED AXIS STACK — EXTENDED MULTI-DATASET TEST")
    print("="*70)

    axes = load_axes()
    n = len(axes)

    print(f"axes loaded: {n}")
    for a in axes:
        print(f" - {a['name']:30s}  (l={a['l']:.2f}°, b={a['b']:.2f}°)")
    print("-"*70)

    # pairwise separations
    print("pairwise separations (deg):")
    seps = []
    for a1, a2 in itertools.combinations(axes, 2):
        d = angdist(a1['l'], a1['b'], a2['l'], a2['b'])
        seps.append(d)
        print(f" {a1['name']}  ↔  {a2['name']} :  {d:.2f}°")

    max_sep = max(seps)
    print("-"*70)
    print(f"maximum pairwise separation: {max_sep:.2f}°")
    print("-"*70)

    # null test
    print("generating isotropic null distribution...")
    null = simulate_null(n, nsim=50000)

    pval = np.mean(null <= max_sep)
    mean_null = np.mean(null)
    perc1 = np.percentile(null, 1)
    perc10 = np.percentile(null, 10)

    print(f"mean max_sep(null)      = {mean_null:.2f}°")
    print(f"1% max_sep(null)        = {perc1:.2f}°")
    print(f"10% max_sep(null)       = {perc10:.2f}°")
    print(f"observed max_sep        = {max_sep:.2f}°")
    print(f"p-value                 = {pval:.5f}")
    print("-"*70)

    # verdict
    print("="*70)
    print("scientific verdict")
    print("="*70)

    if pval < 0.01:
        print("the clustered axes are extremely unlikely under isotropy.")
        print("→ strong evidence for a shared preferred direction.")
    elif pval < 0.05:
        print("axes show mild-to-moderate clustering beyond random expectation.")
    else:
        print("axes are consistent with independent random directions.")
    print("="*70)

    # save sky map
    plot_axes(axes)
    print("saved: axis_stack_sky.png")
    print("="*70)


if __name__ == "__main__":
    main()
