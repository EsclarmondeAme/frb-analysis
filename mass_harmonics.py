import math
import itertools

import numpy as np

# ------------------------------------------------------------
# standard model particle masses (approx, in MeV)
# source: rough PDG values (good enough for ratio analysis)
# ------------------------------------------------------------

particles = [
    # leptons
    ("electron",        0.51099895),
    ("muon",           105.6583755),
    ("tau",           1776.86),

    # neutrinos: only upper bounds / tiny effective masses
    # we skip them here because values are too uncertain

    # quarks (current masses, very approximate, MeV)
    ("up",               2.2),
    ("down",             4.7),
    ("strange",         96.0),
    ("charm",        1270.0),
    ("bottom",       4180.0),
    ("top",        172760.0),

    # gauge bosons (in MeV)
    ("W",            80379.0),
    ("Z",            91187.6),

    # higgs (in MeV)
    ("Higgs",      125100.0),
]

# convert to numpy for convenience
names = [p[0] for p in particles]
masses = np.array([p[1] for p in particles], dtype=float)

print("standard model mass list (MeV):")
print("================================")
for n, m in particles:
    print(f"{n:8s} : {m:10.4f} MeV")
print("================================\n")


# ------------------------------------------------------------
# helper: best rational approximation p/q with small integers
# ------------------------------------------------------------

def best_small_ratio(x, max_n=10):
    """
    given a positive real x, find integers p, q <= max_n
    that best approximate x ≈ p/q.

    returns (p, q, approx_value, rel_error)
    """
    best_p, best_q = 0, 1
    best_err = float("inf")
    best_val = 0.0

    for p in range(1, max_n + 1):
        for q in range(1, max_n + 1):
            val = p / q
            rel_err = abs(x - val) / val
            if rel_err < best_err:
                best_err = rel_err
                best_val = val
                best_p, best_q = p, q

    return best_p, best_q, best_val, best_err


# ------------------------------------------------------------
# scan all unordered pairs for simple ratios
# ------------------------------------------------------------

print("searching for simple mass ratios (p/q with p,q ≤ 10)...")
print("")

results = []

for (i, (name_i, m_i)), (j, (name_j, m_j)) in itertools.combinations(enumerate(particles), 2):
    ratio = max(m_i, m_j) / min(m_i, m_j)
    p, q, approx, rel_err = best_small_ratio(ratio, max_n=10)

    # keep only "good" matches (within 2% relative error)
    if rel_err < 0.02:
        results.append({
            "pair": (name_i, name_j),
            "ratio": ratio,
            "p": p,
            "q": q,
            "approx": approx,
            "rel_err": rel_err,
        })

# sort by relative error (best matches first)
results = sorted(results, key=lambda r: r["rel_err"])

if not results:
    print("no strong small-integer ratio matches found (within 2% tolerance).")
else:
    print("strong small-integer ratio candidates (within ~2%):")
    print("===================================================")
    for r in results:
        a, b = r["pair"]
        ratio = r["ratio"]
        p, q = r["p"], r["q"]
        approx = r["approx"]
        err = r["rel_err"]
        print(
            f"{a:8s} / {b:8s} ≈ {ratio:8.4f} "
            f"≈ {p:2d}/{q:2d} = {approx:6.3f} "
            f"(rel error ≈ {err*100:4.2f}%)"
        )

    print("===================================================\n")


# ------------------------------------------------------------
# very rough "harmonicity" score
# count how many ratios fall within epsilon of simple p/q
# ------------------------------------------------------------

epsilon = 0.02  # 2%

num_pairs = len(particles) * (len(particles) - 1) // 2
num_good = len(results)

print(f"total particle pairs checked: {num_pairs}")
print(f"pairs with simple p/q ≤ 10 within {epsilon*100:.1f}%: {num_good}")
if num_pairs > 0:
    print(f"fraction: {num_good / num_pairs:.3f}")
print("")

print("note:")
print("- if this fraction is very small, masses look 'random' w.r.t. simple harmonics.")
print("- if it were unusually large, that could hint at hidden structure.")
print("- to really claim anything, we’d compare this to many random mass sets with similar ranges.")
