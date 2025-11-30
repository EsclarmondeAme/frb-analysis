import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cross_layer_config import CROSS_LAYER_SOURCES

print("creating realistic chime frb dataset")
print("=" * 60)

np.random.seed(42)

# ---------------------------------------------------------
# parameters
# ---------------------------------------------------------
n_frbs = 600               # realistic sample size for 2–3 years
inject_anisotropy = False  # set to True for tests
anisotropy_ra = 140        # degrees (example hotspot)
anisotropy_dec = 35        # degrees
anisotropy_strength = 0.25 # 0–1

inject_cross_layer = True   # we now want real aligned cross-layer frbs
cross_layer_fraction = 0.0  # no random ones anymore
frequency_gap = 50          # boost factor for cross-layer flux


include_redshift = True    # adds approximate cosmology distance

# ---------------------------------------------------------
# time window (2019–2021)
# ---------------------------------------------------------
start_date = datetime(2019, 1, 1)
end_date = datetime(2021, 12, 31)
days = (end_date - start_date).days

timestamps = [
    start_date + timedelta(days=float(np.random.uniform(0, days)))
    for _ in range(n_frbs)
]

# ---------------------------------------------------------
# declinations: realistic CHIME telescope sensitivity
# peak around +50°, gaussian falloff
# ---------------------------------------------------------
decs = np.random.normal(50, 18, n_frbs)
decs = np.clip(decs, -11, 90)

# ---------------------------------------------------------
# right ascension: uniform, but allow optional hotspot
# ---------------------------------------------------------
ra_uniform = np.random.uniform(0, 360, n_frbs)

if inject_anisotropy:
    ra_aniso = np.random.normal(anisotropy_ra, 5, n_frbs)
    dec_aniso = np.random.normal(anisotropy_dec, 5, n_frbs)
    mask = np.random.rand(n_frbs) < anisotropy_strength
    ra_uniform[mask] = ra_aniso[mask]
    decs[mask] = dec_aniso[mask]

ras = ra_uniform

# ---------------------------------------------------------
# dispersion measure (DM): gamma-like distribution
# ---------------------------------------------------------
dm_vals = np.random.gamma(shape=3, scale=150, size=n_frbs) + 100
dm_vals = np.clip(dm_vals, 100, 3000)

# ---------------------------------------------------------
# flux distribution: log-normal
# ---------------------------------------------------------
flux_vals = np.random.lognormal(mean=np.log(2), sigma=1.2, size=n_frbs)
flux_vals = np.clip(flux_vals, 0.3, 200)

# synthetic cross-layer bursts: inject brighter events
if inject_cross_layer:
    mask = np.random.rand(n_frbs) < cross_layer_fraction
    flux_vals[mask] *= frequency_gap

# ---------------------------------------------------------
# FRB width
# ---------------------------------------------------------
width_vals = np.random.gamma(2, 2, n_frbs) + 0.5
width_vals = np.clip(width_vals, 0.3, 50)

# ---------------------------------------------------------
# SNR distribution
# ---------------------------------------------------------
snr_vals = np.random.uniform(8, 100, n_frbs)

# ---------------------------------------------------------
# approximate redshift from DM (very rough)
# z ≈ DM / 1200  (based on Ioka 2003 & Macquart 2020)
# ---------------------------------------------------------
if include_redshift:
    z_vals = dm_vals / 1200
else:
    z_vals = np.full(n_frbs, np.nan)





# ---------------------------------------------------------
# pack into dataframe
# ---------------------------------------------------------
df = pd.DataFrame({
    "name": [f"FRB{t.year}{i:04d}" for i, t in enumerate(timestamps)],
    "utc": [t.strftime("%Y-%m-%d %H:%M:%S.%f") for t in timestamps],
    "mjd": [(t - datetime(1858, 11, 17)).total_seconds() / 86400 for t in timestamps],
    "ra": ras,
    "dec": decs,
    "dm": dm_vals,
    "flux": flux_vals,
    "width": width_vals,
    "snr": snr_vals,
    "z_est": z_vals     # added redshift estimate
})

df = df.sort_values("utc").reset_index(drop=True)
df.to_csv("chime_realistic_frbs.csv", index=False)


# ---------------------------------------------------------
# inject coordinated cross-layer frbs at shared sources
# ---------------------------------------------------------
if inject_cross_layer:
    extra_rows = []
    for src in CROSS_LAYER_SOURCES:
        # make 2 bright bursts per source near t0
        for k in range(2):
            t = src["t0"] + timedelta(seconds=float(np.random.uniform(-10, 10)))
            dm_val = float(np.random.gamma(shape=3, scale=150) + 300)
            flux_val = float(np.random.lognormal(mean=np.log(5), sigma=0.5) * frequency_gap)
            width_val = float(np.random.gamma(2, 1) + 0.5)
            snr_val = float(np.random.uniform(50, 150))

            extra_rows.append({
                "name": f"{src['id']}_FRB{k+1}",
                "utc": t.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "mjd": (t - datetime(1858, 11, 17)).total_seconds()/86400,
                "ra": src["ra"],
                "dec": src["dec"],
                "dm": dm_val,
                "flux": flux_val,
                "width": width_val,
                "snr": snr_val,
                "z_est": dm_val / 1200.0
            })

    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        df = pd.concat([df, extra_df], ignore_index=True)


# ---------------------------------------------------------
# summary
# ---------------------------------------------------------
print(f"\ngenerated {len(df)} realistic chime frbs")
print("time span:", df['utc'].min(), "→", df['utc'].max())
print("dec range:", df['dec'].min(), "→", df['dec'].max())
print("flux range:", df['flux'].min(), "→", df['flux'].max())
print("dm range:", df['dm'].min(), "→", df['dm'].max())

if inject_anisotropy:
    print("\nanisotropy injected around:", anisotropy_ra, anisotropy_dec)

if inject_cross_layer:
    print(f"\ninserted synthetic cross-layer bursts ({cross_layer_fraction*100:.1f}%)")

print("\ndata saved to chime_realistic_frbs.csv")
print("=" * 60)
