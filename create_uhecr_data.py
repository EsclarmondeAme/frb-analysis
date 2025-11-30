import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cross_layer_config import CROSS_LAYER_SOURCES

print("creating uhecr dataset")
print("=" * 60)

np.random.seed(44)

# ---------------------------------------------------------
# parameters
# ---------------------------------------------------------
n_uhecr = 60  # ~20 per year for 3 years
start_date = datetime(2019, 1, 1)
duration_days = 365 * 3

inject_anisotropy = False
anisotropy_ra = 130        # degrees
anisotropy_dec = -40
anisotropy_strength = 0.25

inject_cross_layer = False
cross_layer_fraction = 0.015
energy_boost = 50          # multiply EeV energy

# ---------------------------------------------------------
# timestamps
# ---------------------------------------------------------
timestamps = [
    start_date + timedelta(days=float(np.random.uniform(0, duration_days)))
    for _ in range(n_uhecr)
]

# ---------------------------------------------------------
# sky distribution (Auger-like)
# ---------------------------------------------------------
# Declination range of Auger: roughly −90 to +20 deg
decs = np.random.uniform(-90, 20, n_uhecr)

# RA is effectively uniform
ras = np.random.uniform(0, 360, n_uhecr)

# anisotropy injection
if inject_anisotropy:
    mask = np.random.rand(n_uhecr) < anisotropy_strength
    ras[mask] = np.random.normal(anisotropy_ra, 5, mask.sum())
    decs[mask] = np.random.normal(anisotropy_dec, 5, mask.sum())

# ---------------------------------------------------------
# energies (EeV)
# ---------------------------------------------------------
# Ultra-high-energy CRs follow a power-law: ~E^-3
energies = np.random.pareto(a=3.0, size=n_uhecr) + 1.0
energies = energies * 10      # base energy ~10 EeV
energies = np.clip(energies, 10, 250)

# cross-layer synthetic injections
if inject_cross_layer:
    mask = np.random.rand(n_uhecr) < cross_layer_fraction
    energies[mask] *= energy_boost
    energies = np.clip(energies, 10, 10000)  # avoid overflow

# ---------------------------------------------------------
# magnetic deflection (crucial for realism)
# ---------------------------------------------------------
# UHECRs are deflected by galactic magnetic fields
# typical deflection ~ few degrees at 50 EeV
deflection_deg = np.random.normal(3, 1.0, n_uhecr)
ras = (ras + np.random.normal(0, deflection_deg)) % 360
decs = np.clip(decs + np.random.normal(0, deflection_deg), -90, 90)

# ---------------------------------------------------------
# zenith angle (Auger)
# ---------------------------------------------------------
zenith = np.random.uniform(0, 55, n_uhecr)

# ---------------------------------------------------------
# pack dataframe
# ---------------------------------------------------------
df = pd.DataFrame({
    "name": [f"UHECR{i:03d}" for i in range(n_uhecr)],
    "utc": [t.strftime("%Y-%m-%d %H:%M:%S.%f") for t in timestamps],
    "mjd": [(t - datetime(1858, 11, 17)).total_seconds()/86400 for t in timestamps],
    "ra": ras,
    "dec": decs,
    "energy_eev": energies,
    "zenith_deg": zenith,
    "z_est": np.nan   # maintain consistency across all messengers
})

# ---------------------------------------------------------
# inject cross-layer uhecr events
# ---------------------------------------------------------
extra_rows = []
for src in CROSS_LAYER_SOURCES:
    # one ultra-high-energy cosmic ray per source
    # delayed by hours to days after t0
    delay_days = float(np.random.uniform(0.2, 3.0))
    t = src["t0"] + timedelta(days=delay_days)

    energy_eev = float(np.random.lognormal(mean=np.log(80), sigma=0.3))

    # small magnetic deflection
    ra = (src["ra"] + np.random.normal(0, 3)) % 360
    dec = np.clip(src["dec"] + np.random.normal(0, 3), -90, 90)

    zenith_deg = float(np.random.uniform(0, 55))

    extra_rows.append({
        "name": f"{src['id']}_CR1",
        "utc": t.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "mjd": (t - datetime(1858, 11, 17)).total_seconds()/86400,
        "ra": ra,
        "dec": dec,
        "energy_eev": energy_eev,
        "zenith_deg": zenith_deg,
        "z_est": np.nan
    })

if extra_rows:
    extra_df = pd.DataFrame(extra_rows)
    df = pd.concat([df, extra_df], ignore_index=True)




df = df.sort_values("utc").reset_index(drop=True)
df.to_csv("uhecr.csv", index=False)

# ---------------------------------------------------------
# summary
# ---------------------------------------------------------
print(f"\ngenerated {len(df)} uhecr events")
print("time span:", df['utc'].min(), "→", df['utc'].max())
print("energy range (eev):", df['energy_eev'].min(), "→", df['energy_eev'].max())

if inject_anisotropy:
    print("anisotropy injected at:", anisotropy_ra, anisotropy_dec)

if inject_cross_layer:
    print(f"cross-layer uhecr injected ({cross_layer_fraction*100:.1f}%)")

print("\ndata saved as uhecr.csv")
print("=" * 60)
