import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tqdm import tqdm
import sys

"""
TEST 45 – FRB PLANCK-GRAIN MICRO-CLUSTERING TEST
------------------------------------------------
Goal:
    Detect ultra-small angular-scale clustering patterns which would indicate
    granular structure in the underlying spacetime / emission geometry.

Inputs:
    - FRB catalog with 'theta_unified' and 'phi_unified' columns.

Output:
    - micro-scale pair separation spectrum
    - Allan variance over angular windows
    - null distribution from isotropic MC
"""

# ------------------------------------------------------------
# angular separation on the sphere
# ------------------------------------------------------------
def angsep(t1, p1, t2, p2):
    dphi = np.deg2rad(p1 - p2)
    t1 = np.deg2rad(t1)
    t2 = np.deg2rad(t2)
    return np.rad2deg(np.arccos(
        np.clip(np.sin(t1)*np.sin(t2) + np.cos(t1)*np.cos(t2)*np.cos(dphi), -1, 1)
    ))


def main():
    if len(sys.argv) < 2:
        print("usage: python frb_planck_grain_microclustering_test45.py frbs_unified.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    print("detected FRBs:", len(df))

    theta = df["theta_unified"].values
    phi   = df["phi_unified"].values
    n = len(theta)

    # ------------------------------------------------------------
    # compute all pairwise separations
    # ------------------------------------------------------------
    coords = np.vstack([theta, phi]).T
    seps = []
    print("computing pairwise separations...")
    for i in range(n):
        for j in range(i+1, n):
            seps.append(angsep(theta[i], phi[i], theta[j], phi[j]))
    seps = np.array(seps)

    # ------------------------------------------------------------
    # MICRO-SCALE SPECTRUM
    # ------------------------------------------------------------
    bins = np.linspace(0, 2, 200)  # 0–2 deg resolution
    hist_real, _ = np.histogram(seps, bins=bins)

    # ------------------------------------------------------------
    # ALLAN VARIANCE ON MICRO-SCALE BINS
    # ------------------------------------------------------------
    diffs = np.diff(hist_real)
    allan_var_real = 0.5 * np.mean(diffs**2)

    # ------------------------------------------------------------
    # Null Monte Carlo
    # ------------------------------------------------------------
    print("running Monte Carlo isotropic (N=5000)...")
    N_MC = 5000

    allan_null = []
    minsep_null = []

    for _ in tqdm(range(N_MC)):
        # random sky: keep number of points fixed
        t_rand = np.degrees(np.arccos(1 - 2*np.random.rand(n)))   # random theta
        p_rand = np.random.rand(n)*360                             # random phi

        # random separations
        seps_r = []
        for i in range(n):
            for j in range(i+1, n):
                seps_r.append(angsep(t_rand[i], p_rand[i], t_rand[j], p_rand[j]))
        seps_r = np.array(seps_r)

        # Allan variance for null
        hist_r, _ = np.histogram(seps_r, bins=bins)
        diffs_r = np.diff(hist_r)
        allan_null.append(0.5 * np.mean(diffs_r**2))

        # min-separation test
        minsep_null.append(seps_r.min())

    allan_null = np.array(allan_null)
    minsep_null = np.array(minsep_null)

    # ------------------------------------------------------------
    # Observed values
    # ------------------------------------------------------------
    allan_obs = allan_var_real
    minsep_obs = seps.min()

    # ------------------------------------------------------------
    # p-values
    # ------------------------------------------------------------
    p_allan = np.mean(allan_null >= allan_obs)  # fixed: >= for high Allan variance
    p_minsep = np.mean(minsep_null <= minsep_obs)

    # ------------------------------------------------------------
    # *** NEW: IDENTIFY ZERO-SEPARATION PAIRS ***
    # ------------------------------------------------------------
    print("\n" + "="*68)
    print("IDENTIFYING CLOSE PAIRS (separation < 0.01 degrees)")
    print("="*68)
    
    close_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            sep = angsep(theta[i], phi[i], theta[j], phi[j])
            if sep < 0.01:  # less than 0.01 degrees (essentially identical)
                close_pairs.append({
                    'idx1': i,
                    'idx2': j,
                    'sep': sep,
                    'theta1': theta[i],
                    'phi1': phi[i],
                    'theta2': theta[j],
                    'phi2': phi[j]
                })
    
    if len(close_pairs) > 0:
        print(f"\nFound {len(close_pairs)} pairs with separation < 0.01 deg:\n")
        
        for k, pair in enumerate(close_pairs[:20]):  # show first 20
            i = pair['idx1']
            j = pair['idx2']
            print(f"Pair #{k+1}:")
            print(f"  FRB indices: {i} and {j}")
            print(f"  Separation: {pair['sep']:.8f} degrees")
            print(f"  Position 1 (unified): θ={pair['theta1']:.6f}°, φ={pair['phi1']:.6f}°")
            print(f"  Position 2 (unified): θ={pair['theta2']:.6f}°, φ={pair['phi2']:.6f}°")
            
            # Show original RA/Dec if available
            if 'ra' in df.columns and 'dec' in df.columns:
                print(f"  RA/Dec 1: {df['ra'].iloc[i]:.6f}°, {df['dec'].iloc[i]:.6f}°")
                print(f"  RA/Dec 2: {df['ra'].iloc[j]:.6f}°, {df['dec'].iloc[j]:.6f}°")
            
            # Show FRB names if available
            if 'frb_name' in df.columns:
                print(f"  FRB 1 name: {df['frb_name'].iloc[i]}")
                print(f"  FRB 2 name: {df['frb_name'].iloc[j]}")
            elif 'name' in df.columns:
                print(f"  FRB 1 name: {df['name'].iloc[i]}")
                print(f"  FRB 2 name: {df['name'].iloc[j]}")
            
            # Check if they might be repeaters (same position)
            ra_diff = abs(df['ra'].iloc[i] - df['ra'].iloc[j]) if 'ra' in df.columns else None
            dec_diff = abs(df['dec'].iloc[i] - df['dec'].iloc[j]) if 'dec' in df.columns else None
            
            if ra_diff is not None and dec_diff is not None:
                if ra_diff < 0.001 and dec_diff < 0.001:
                    print(f"  *** LIKELY REPEATER: Positions identical within 0.001° ***")
                else:
                    print(f"  RA diff: {ra_diff:.6f}°, Dec diff: {dec_diff:.6f}°")
            
            print()
        
        if len(close_pairs) > 20:
            print(f"... and {len(close_pairs) - 20} more pairs (not shown)")
        
        # Summary statistics
        print("\n" + "-"*68)
        print("CLOSE PAIR SUMMARY:")
        print(f"  Total pairs < 0.01°: {len(close_pairs)}")
        print(f"  Mean separation: {np.mean([p['sep'] for p in close_pairs]):.8f}°")
        print(f"  Min separation: {np.min([p['sep'] for p in close_pairs]):.8f}°")
        print(f"  Max separation: {np.max([p['sep'] for p in close_pairs]):.8f}°")
        
    else:
        print("\nNo pairs found with separation < 0.01 degrees")
        print("(Note: Minimum observed separation was {:.6f} degrees)".format(minsep_obs))

    # ------------------------------------------------------------
    # print main results
    # ------------------------------------------------------------
    print("\n" + "="*68)
    print("FRB PLANCK-GRAIN MICRO-CLUSTERING TEST (TEST 45)")
    print("="*68)
    print(f"N_FRB = {n}")
    print("------------------------------------------------------------")
    print(f"observed Allan variance: {allan_obs:.6f}")
    print(f"null mean Allan variance: {allan_null.mean():.6f}")
    print(f"p-value (granularity): {p_allan:.6f}")
    print("------------------------------------------------------------")
    print(f"observed minimum separation: {minsep_obs:.6f} deg")
    print(f"null mean minimum separation: {minsep_null.mean():.6f}")
    print(f"p-value (min-sep anomaly): {p_minsep:.6f}")
    print("------------------------------------------------------------")
    print("interpretation:")
    print(" - high p(Allan) with large Allan variance: strong micro-scale clustering")
    print(" - low p(min-sep): pairs much closer than random expectation")
    
    if len(close_pairs) > 0:
        # Check if likely repeaters
        if 'ra' in df.columns and 'dec' in df.columns:
            identical_positions = sum(
                1 for p in close_pairs 
                if abs(df['ra'].iloc[p['idx1']] - df['ra'].iloc[p['idx2']]) < 0.001 
                and abs(df['dec'].iloc[p['idx1']] - df['dec'].iloc[p['idx2']]) < 0.001
            )
            
            if identical_positions > 0:
                print(f"\n *** {identical_positions} pairs have IDENTICAL positions (< 0.001°)")
                print(" *** These are likely REPEATING FRBs (same source detected multiple times)")
            
            if identical_positions < len(close_pairs):
                distinct_close = len(close_pairs) - identical_positions
                print(f"\n *** {distinct_close} pairs are CLOSE but DISTINCT positions")
                print(" *** This could indicate angular quantization / lattice structure")
    
    print("="*68)
    print("test 45 complete.")
    print("="*68)


if __name__ == "__main__":
    main()