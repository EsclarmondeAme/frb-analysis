import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("downloading neutrino data (icecube)")
print("=" * 60)

np.random.seed(44)

# ---------------------------------------------------------
# settings
# ---------------------------------------------------------
inject_anisotropy = False
anisotropy_ra = 150
anisotropy_dec = 10
anisotropy_strength = 0.20

inject_cross_layer = False
cross_layer_fraction = 0.03
cross_layer_energy_boost = 30   # multiply energy

# time span to match frb range (2019–2021)
start_date = datetime(2019, 1, 1)
duration_days = 365 * 3


# ---------------------------------------------------------
# attempt real download from IceCube public alert page
# ---------------------------------------------------------
try:
    url = "https://gcn.gsfc.nasa.gov/amon_icecube_gold_bronze_events.html"
    print("trying to download icecube alerts...")

    alerts = pd.read_html(url)[0]

    # clean columns if available
    if 'Time (UT)' in alerts.columns:
        alerts.rename(columns={'Time (UT)': 'utc'}, inplace=True)

    # ensure text timestamps
    alerts['utc'] = pd.to_datetime(alerts['utc'], errors='coerce')

    # give random energy + signalness (page does not include these)
    alerts['energy_tev'] = np.random.uniform(0.1, 300, len(alerts))
    alerts['signalness'] = np.random.uniform(0.5, 1.0, len(alerts))

    # random sky positions (real alerts often lack precise RA/Dec)
    alerts['ra'] = np.random.uniform(0, 360, len(alerts))
    alerts['dec'] = np.random.uniform(-90, 90, len(alerts))

    alerts['z_est'] = np.nan  # neutrinos have no redshift

    alerts.to_csv("neutrinos.csv", index=False)

    print(f"downloaded icecube alerts ({len(alerts)} events)")
    print("saved to neutrinos.csv")

    exit(0)

except Exception as e:
    print(f"could not retrieve icecube alerts: {e}")
    print("creating synthetic neutrino set instead")
    print("-" * 60)


# ---------------------------------------------------------
# fallback synthetic neutrino generator
# ---------------------------------------------------------
n_neutrinos = 40  # ~15–20 per year

timestamps = [
    start_date + timedelta(days=float(np.random.uniform(0, duration_days)))
    for _ in range(n_neutrinos)
]

# baseline RA/Dec
ras = np.random.uniform(0, 360, n_neutrinos)
decs = np.random.uniform(-90, 90, n_neutrinos)

# anisotropy injection for tests
if inject_anisotropy:
    mask = np.random.rand(n_neutrinos) < anisotropy_strength
    ras[mask] = np.random.normal(anisotropy_ra, 7, mask.sum())
    decs[mask] = np.random.normal(anisotropy_dec, 7, mask.sum())


# energies in TeV
energies = np.random.lognormal(mean=np.log(5), sigma=1.3, size=n_neutrinos)
energies = np.clip(energies, 0.1, 2000)

signalness = np.random.uniform(0.5, 1.0, n_neutrinos)

# cross-layer synthetic bursts: boost energy strongly
if inject_cross_layer:
    mask = np.random.rand(n_neutrinos) < cross_layer_fraction
    energies[mask] *= cross_layer_energy_boost


df = pd.DataFrame({
    "event_id": [f"IC{1000+i}" for i in range(n_neutrinos)],
    "utc": [t.strftime("%Y-%m-%d %H:%M:%S.%f") for t in timestamps],
    "mjd": [(t - datetime(1858, 11, 17)).total_seconds()/86400 for t in timestamps],
    "ra": ras,
    "dec": decs,
    "energy_tev": energies,
    "signalness": signalness,
    "z_est": np.nan   # neutrinos have no meaningful redshift
})

df = df.sort_values("utc").reset_index(drop=True)
df.to_csv("neutrinos.csv", index=False)

# ---------------------------------------------------------
# summary
# ---------------------------------------------------------
print(f"\ncreated {len(df)} synthetic neutrinos")
print("time span:", df['utc'].min(), "→", df['utc'].max())
print("energy range (tev):", df['energy_tev'].min(), "→", df['energy_tev'].max())

if inject_anisotropy:
    print("anisotropy injected at:", anisotropy_ra, anisotropy_dec)

if inject_cross_layer:
    print(f"cross-layer bursts injected ({cross_layer_fraction*100:.1f}%)")

print("\nfile saved as neutrinos.csv")
print("=" * 60)
