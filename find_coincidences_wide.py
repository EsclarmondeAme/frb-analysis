import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

print("=" * 60)
print("wide coincidence search (frb + neutrino, with redshift diagnostics)")
print("=" * 60)

def angular_distance(ra1, dec1, ra2, dec2):
    r1, d1 = radians(ra1), radians(dec1)
    r2, d2 = radians(ra2), radians(dec2)
    a = sin((d2 - d1) / 2) ** 2 + cos(d1) * cos(d2) * sin((r2 - r1) / 2) ** 2
    return 2 * atan2(sqrt(a), sqrt(1 - a)) * (180 / np.pi)

print("\nloading data...")
frbs = pd.read_csv("frbs.csv")
neutrinos = pd.read_csv("neutrinos_clean.csv")
frbs["utc"] = pd.to_datetime(frbs["utc"], format="mixed", utc=True)


neutrinos["utc"] = pd.to_datetime(neutrinos["utc"], format="mixed", utc=True)



print(f"{len(frbs)} frbs")
print(f"{len(neutrinos)} neutrinos")

start = max(frbs["utc"].min(), neutrinos["utc"].min())
end = min(frbs["utc"].max(), neutrinos["utc"].max())

frb_ov = frbs[(frbs["utc"] >= start) & (frbs["utc"] <= end)]
nu_ov = neutrinos[(neutrinos["utc"] >= start) & (neutrinos["utc"] <= end)]

print("\noverlap:")
print(start, "→", end)
print(f"frbs in overlap: {len(frb_ov)}")
print(f"neutrinos in overlap: {len(nu_ov)}")

base_time_window = 60   # seconds
max_ang_sep = 10        # degrees, wider
background_energy_mean = frb_ov["flux"].mean() if "flux" in frb_ov else 1.0

coinc = []
print("\nsearching wide window coincidences...")

for _, frb in frb_ov.iterrows():
    frb_flux = frb.get("flux", 1.0)
    frb_z = frb.get("z_est", np.nan)

    for _, nu in nu_ov.iterrows():
        nu_sig = nu.get("signalness", 0.5)
        energy_factor = max(0.2, min(3.0, 1.0 / (nu_sig + 0.1)))
        dyn_window = base_time_window * energy_factor

        dt = abs((frb["utc"] - nu["utc"]).total_seconds())
        if dt > dyn_window:
            continue

        ang = angular_distance(frb["ra"], frb["dec"], nu["ra"], nu["dec"])
        if ang > max_ang_sep:
            continue

        # physics-inspired coherence & coupling weights
        nu_E_tev = nu.get("energy_tev", np.nan)

        w_E   = coupling_efficiency(nu_E_tev) if np.isfinite(nu_E_tev) else 1.0
        w_t   = time_coherence_weight(dt)
        w_ang = angular_coherence_weight(ang)

        # astrophysical base weight: brighter FRB, moderate redshift, stronger neutrino
        base = 0.0

        if np.isfinite(frb_flux):
            base += np.log10(1.0 + max(frb_flux, 0.0))

        if np.isfinite(frb_z):
            base += 0.3 * max(frb_z, 0.0)

        if np.isfinite(nu_sig):
            base += 0.5 * nu_sig

        score = base * w_E * w_t * w_ang


        coinc.append({
            "frb_name": frb["name"],
            "frb_time": frb["utc"],
            "nu_id": nu["event_id"],
            "nu_time": nu["utc"],
            "time_diff_s": dt,
            "dyn_window_s": dyn_window,
            "angular_deg": ang,
            "frb_flux": frb_flux,
            "frb_z": frb_z,
            "nu_signalness": nu_sig,
            "score": score,
        })

print("\n" + "=" * 60)
print("results")
print("=" * 60)
print("found", len(coinc), "coincidences")

if len(coinc) == 0:
    print("\nno matches (wide search)")
    print("try:")
    print("  • increasing angular window")
    print("  • adding uhecr data")
    print("  • using longer time windows (hours or days)")
else:
    df = pd.DataFrame(coinc)
    df.to_csv("coincidences_wide_enhanced.csv", index=False)

    print("\ntop coincidences:")
    print(df.sort_values("score", ascending=False).head(10).to_string(index=False))

    days = (end - start).total_seconds() / 86400.0
    expected_random = (len(frb_ov) * len(nu_ov) * (2 * base_time_window)) / (days * 86400.0)

    print("\nexpected random matches:", round(expected_random, 3))
    print("observed:", len(df))

    if len(df) > expected_random * 3:
        print("\npossible excess beyond background")

    # redshift diagnostics
    if "z_est" in frb_ov.columns:
        all_z = frb_ov["z_est"].replace([np.inf, -np.inf], np.nan).dropna()
        coinc_z = df["frb_z"].replace([np.inf, -np.inf], np.nan).dropna()

        if len(all_z) > 5 and len(coinc_z) > 0:
            print("\nredshift diagnostics (wide search):")
            print(f"  all frbs:    mean z = {all_z.mean():.3f}, std = {all_z.std():.3f}")
            print(f"  coinc frbs:  mean z = {coinc_z.mean():.3f}, std = {coinc_z.std():.3f}")

            z_for_corr = df["frb_z"].fillna(all_z.mean())
            scores = df["score"]
            corr = np.corrcoef(z_for_corr, scores)[0, 1]
            print(f"  corr(score, z) ≈ {corr:.3f}")

            if abs(corr) < 0.2:
                print("  → wide coincidences are roughly redshift-independent")
            else:
                print("  → wide coincidences show redshift dependence")

print("\ndone")
print("=" * 60)
