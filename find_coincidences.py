import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

from cross_layer_physics import (
    coupling_efficiency,
    time_coherence_weight,
    angular_coherence_weight,
)



print("=" * 60)
print("multi-messenger coincidence analysis")
print("frb + neutrino (with direction, energy, redshift diagnostics)")
print("=" * 60)

# ---------------------------------------------------------
# helper: angular distance on a sphere
# ---------------------------------------------------------
def angular_distance(ra1, dec1, ra2, dec2):
    r1, d1 = radians(ra1), radians(dec1)
    r2, d2 = radians(ra2), radians(dec2)

    a = sin((d2 - d1) / 2) ** 2 + cos(d1) * cos(d2) * sin((r2 - r1) / 2) ** 2
    return 2 * atan2(sqrt(a), sqrt(1 - a)) * (180 / np.pi)


# ---------------------------------------------------------
# load data
# ---------------------------------------------------------
print("\nloading data...")
frbs = pd.read_csv("frbs.csv")
neutrinos = pd.read_csv("neutrinos_clean.csv")






frbs["utc"] = pd.to_datetime(frbs["utc"], utc=True)

neutrinos["utc"] = pd.to_datetime(neutrinos["utc"], format="mixed", utc=True)

print(f"{len(frbs)} frbs loaded")
print(f"{len(neutrinos)} neutrinos loaded")

# overlapping window
overlap_start = max(frbs["utc"].min(), neutrinos["utc"].min())
overlap_end = min(frbs["utc"].max(), neutrinos["utc"].max())

frb_ov = frbs[(frbs["utc"] >= overlap_start) & (frbs["utc"] <= overlap_end)]
nu_ov = neutrinos[(neutrinos["utc"] >= overlap_start) & (neutrinos["utc"] <= overlap_end)]

print("\noverlap range:")
print(overlap_start, " → ", overlap_end)
print(f"frbs in range: {len(frb_ov)}")
print(f"neutrinos in range: {len(nu_ov)}")

# ---------------------------------------------------------
# detection parameters
# ---------------------------------------------------------
base_time_window = 60  # seconds
max_ang_sep = 5        # degrees; relatively tight

background_energy_mean = frb_ov["flux"].mean() if "flux" in frb_ov else 1.0

coincidences = []

print("\nsearching for coincidences...")

for _, frb in frb_ov.iterrows():
    frb_flux = frb.get("flux", 1.0)
    frb_z = frb.get("z_est", np.nan)

    for _, nu in nu_ov.iterrows():
        # dynamic time window based on neutrino energy / signalness
        nu_sig = nu.get("signalness", 0.5)
        energy_factor = max(0.2, min(3.0, 1.0 / (nu_sig + 0.1)))
        dyn_time_window = base_time_window * energy_factor

        dt = abs((frb["utc"] - nu["utc"]).total_seconds())
        if dt > dyn_time_window:
            continue

        ang = angular_distance(frb["ra"], frb["dec"], nu["ra"], nu["dec"])
        if ang > max_ang_sep:
            continue

        # energy-weighted score
        # --- new physical score model ---
        nu_E_tev = nu.get("energy_tev", np.nan)

        # physical coherence weights
        w_E   = coupling_efficiency(nu_E_tev) if np.isfinite(nu_E_tev) else 1.0
        w_t   = time_coherence_weight(dt)
        w_ang = angular_coherence_weight(ang)

        # astrophysical base weight
        base = 0.0

        if np.isfinite(frb_flux):
            base += np.log10(1.0 + max(frb_flux, 0.0))

        if np.isfinite(frb_z):
            base += 0.3 * max(frb_z, 0.0)

        if np.isfinite(nu_sig):
            base += 0.5 * nu_sig

        coinc_score = base * w_E * w_t * w_ang


        coincidences.append({
            "frb_name": frb["name"],
            "frb_time": frb["utc"],
            "nu_id": nu["event_id"],
            "nu_time": nu["utc"],
            "time_diff_s": dt,
            "dyn_window_s": dyn_time_window,
            "angular_deg": ang,
            "frb_flux": frb_flux,
            "frb_z": frb_z,
            "nu_signalness": nu_sig,
            "coinc_score": coinc_score,

        })

# ---------------------------------------------------------
# results
# ---------------------------------------------------------
print("\n" + "=" * 60)
print(f"found {len(coincidences)} coincidences")
print("=" * 60)

if len(coincidences) == 0:
    print("\nno coincidences found")
    print("this does not rule out cross-layer events")
    print("may require:")
    print("  - more data")
    print("  - softer angular / time cuts")
    print("  - tuned energy weighting")
else:
    df = pd.DataFrame(coincidences)
    df.to_csv("coincidences_enhanced.csv", index=False)

    print("\nstrongest coincidences:")
    print(df.sort_values("coinc_score", ascending=False).head(10).to_string(index=False))

    # approximate false alarm expectation
    days = (overlap_end - overlap_start).total_seconds() / 86400.0
    expected_random = (len(frb_ov) * len(nu_ov) * (2 * base_time_window)) / (days * 86400.0)

    print("\nexpected random coincidences:", round(expected_random, 3))
    print("observed:", len(df))

    if len(df) > expected_random * 2:
        print("\npossible anomalous excess")

    # -----------------------------------------------------
    # redshift diagnostics: how do coincidences depend on z?
    # -----------------------------------------------------
    if "z_est" in frb_ov.columns:
        all_z = frb_ov["z_est"].replace([np.inf, -np.inf], np.nan).dropna()
        coinc_z = df["frb_z"].replace([np.inf, -np.inf], np.nan).dropna()

        if len(all_z) > 5 and len(coinc_z) > 0:
            print("\nredshift diagnostics:")
            print(f"  all frbs:    mean z = {all_z.mean():.3f}, std = {all_z.std():.3f}")
            print(f"  coinc frbs:  mean z = {coinc_z.mean():.3f}, std = {coinc_z.std():.3f}")

            # correlation between score and z (using finite values)
            z_for_corr = df["frb_z"].fillna(all_z.mean())
            scores = df["coinc_score"]
            corr = np.corrcoef(z_for_corr, scores)[0, 1]

            print(f"  corr(score, z) ≈ {corr:.3f}")
            if abs(corr) < 0.2:
                print("  → coincidence strength is roughly redshift-independent")
                print("    (consistent with non-geometric / cross-layer origin)")
            else:
                print("  → coincidence strength shows redshift dependence")
                print("    (more like standard flux-limited astrophysical selection)")

print("\nanalysis complete")
print("=" * 60)
