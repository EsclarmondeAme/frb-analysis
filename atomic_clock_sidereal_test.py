import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

# ------------------------------------------------------
# load data (any of the SYRTE–NPL files)
# ------------------------------------------------------
df = pd.read_csv("SYRTE_NPL_2015_Combined.dat", 
                 delim_whitespace=True,
                 names=["mjd", "frac_freq", "error"])

# remove NaNs
df = df.dropna()

t = df["mjd"].values
y = df["frac_freq"].values

# center data (remove DC offset)
y = y - np.mean(y)

# convert days → seconds
t_sec = (t - t.min()) * 86400.0

# ------------------------------------------------------
# compute power at solar vs sidereal frequencies
# ------------------------------------------------------
freq_solar = 1.0 / (24 * 3600)        # 1 / 86400
freq_sidereal = 1.0 / 86164.0905      # sidereal day

model = LombScargle(t_sec, y)

power_solar = model.power(freq_solar)
power_sidereal = model.power(freq_sidereal)

ratio = power_sidereal / power_solar

print("===================================================")
print("real atomic clock sidereal test")
print("===================================================")
print(f"power (solar)    = {power_solar:.6e}")
print(f"power (sidereal) = {power_sidereal:.6e}")
print(f"ratio (sidereal/solar) = {ratio:.4f}")
print("---------------------------------------------------")

if ratio > 1.05:
    print("sidereal excess detected (possible cone-axis signal)")
elif ratio < 0.95:
    print("solar excess (no cone effect)")
else:
    print("no significant directional modulation.")
print("===================================================")
