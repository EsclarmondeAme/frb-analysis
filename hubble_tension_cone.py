import numpy as np

print("=" * 69)
print("hubble tension reinterpretation via frequency-layer cone model")
print("=" * 69)

# observed hubble values (approximate, in km/s/Mpc)
H0_cmb = 67.4   # planck-like
H0_sne = 73.0   # SH0ES-like

ratio_obs = H0_sne / H0_cmb
delta_percent = (ratio_obs - 1.0) * 100.0

print(f"observed ratio (SH0ES / Planck) = {ratio_obs:.4f}  →  {delta_percent:.2f}% difference")
print()

# ------------------------------------------------------------------
# toy cone model:
#   each "frequency layer" f has an effective Hubble:
#     H(f) = H_ref * [1 + k * (f - 1)]
#
# we take:
#   - supernova layer at f_sne = 1.0
#   - cmb layer at f_cmb = 1 - Δf   (slightly higher or lower in cone)
#
# then:
#   H_sne = H_ref * [1 + k*(1 - 1)] = H_ref
#   H_cmb = H_ref * [1 + k*(f_cmb - 1)] = H_ref * [1 - k*Δf]
#
# therefore the ratio is:
#   R_model = H_sne / H_cmb = 1 / (1 - k*Δf)
#
# we scan over Δf and k to see if we can reproduce R_obs.
# ------------------------------------------------------------------

freq_diffs = np.linspace(0.05, 0.20, 151)   # Δf ∈ [0.05, 0.20]  (5–20% layer difference)
k_values   = np.linspace(0.1, 1.2, 1111)    # k ∈ [0.1, 1.2]     (sensitivity of H to freq)

tolerance = 0.01   # require |R_model - R_obs| < 1%

matches = []

for df in freq_diffs:
    for k in k_values:
        R_model = 1.0 / (1.0 - k * df)
        if abs(R_model - ratio_obs) < tolerance:
            matches.append((df, k, R_model))

if not matches:
    print("result of scan:")
    print("  within Δf ∈ [0.05, 0.20] and k ∈ [0.1, 1.2],")
    print("  this very simple linear cone model did NOT reproduce the")
    print("  observed Hubble tension within 1% tolerance.")
    print()
    print("interpretation:")
    print("  - either the real relationship between frequency layer and H(f)")
    print("    is more complicated than H(f) = H_ref [1 + k (f - 1)],")
    print("  - or the required parameters lie outside the scanned ranges.")
else:
    dfs = np.array([m[0] for m in matches])
    ks  = np.array([m[1] for m in matches])

    df_min, df_med, df_max = dfs.min(), np.median(dfs), dfs.max()
    k_min,  k_med,  k_max  = ks.min(),  np.median(ks),  ks.max()

    print("parameter region that reproduces the observed tension (within 1%):")
    print("-----------------------------------------------------------------")
    print(f"  Δf (layer frequency difference):")
    print(f"    min ≈ {df_min:.3f}, median ≈ {df_med:.3f}, max ≈ {df_max:.3f}")
    print()
    print(f"  k (sensitivity of H to layer frequency):")
    print(f"    min ≈ {k_min:.3f}, median ≈ {k_med:.3f}, max ≈ {k_max:.3f}")
    print()
    print("interpretation (within THIS toy model):")
    print("  - to reproduce the ~8.3% Hubble tension, the product k * Δf must be")
    print("    roughly ≈ 0.077.")
    print("  - for example, Δf ~ 0.08–0.15 with k ~ 0.5–1.0 works.")
    print("  - that means:")
    print("      • neighboring layers differ in effective frequency by ~8–15%,")
    print("      • the expansion rate responds fairly strongly to that difference (k ~ O(1)).")
    print()
    print("caveats:")
    print("  - this is a VERY simple linear model; real cosmology would require")
    print("    proper scale-factor evolution, dark energy density, etc.")
    print("  - this tells us the cone idea CAN mimic the size of the Hubble tension")
    print("    for reasonable-looking parameters, but does NOT prove it is the cause.")

print("=" * 69)
