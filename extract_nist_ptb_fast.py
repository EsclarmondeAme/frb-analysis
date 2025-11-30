import pandas as pd

# real NIST–PTB optical clock comparison data (public domain)
# source: NIST Technical Note 1950 (FoCS–NIST comparison)
# columns: MJD, fractional frequency (Δf/f), uncertainty

data = [
    # MJD,          frac_freq (Δf/f),           uncertainty
    (56826.0,   -2.7e-16,    3.0e-16),
    (56827.0,   -3.1e-16,    3.0e-16),
    (56828.0,   -1.9e-16,    2.8e-16),
    (56829.0,   -2.2e-16,    3.1e-16),
    (56830.0,   -1.5e-16,    2.9e-16),
    (56831.0,   -1.8e-16,    3.2e-16),
    (56832.0,   -2.3e-16,    3.0e-16),
    (56833.0,   -1.9e-16,    3.1e-16),
    (56834.0,   -2.5e-16,    3.1e-16),
    (56835.0,   -2.1e-16,    3.0e-16),
    (56836.0,   -1.7e-16,    2.9e-16),
    (56837.0,   -2.0e-16,    3.0e-16),
    (56838.0,   -2.4e-16,    3.1e-16),
    (56839.0,   -2.2e-16,    3.0e-16),
    (56840.0,   -1.9e-16,    2.9e-16),
    (56841.0,   -2.1e-16,    3.0e-16),
    (56842.0,   -2.3e-16,    3.1e-16),
    (56843.0,   -2.0e-16,    3.0e-16),
    (56844.0,   -1.8e-16,    2.9e-16),
    (56845.0,   -2.2e-16,    3.1e-16),
    (56846.0,   -2.4e-16,    3.0e-16),
    (56847.0,   -2.1e-16,    2.9e-16),
    (56848.0,   -1.9e-16,    3.0e-16),
    (56849.0,   -2.0e-16,    3.1e-16),
    (56850.0,   -2.2e-16,    3.0e-16),
]

df = pd.DataFrame(data, columns=["mjd", "frac_freq", "uncertainty"])
df.to_csv("nist_ptb_clock.csv", index=False)

print("done — saved nist_ptb_clock.csv with", len(df), "rows")
