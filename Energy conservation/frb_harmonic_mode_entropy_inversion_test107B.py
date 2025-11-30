import numpy as np
import pandas as pd
import sys
from scipy.special import sph_harm
from scipy.stats import entropy

# ================================================================
# Test 107B — Per-mode Entropy Production (σ_lm) and Inversion (I_lm)
# ================================================================

if len(sys.argv) < 3:
    print("usage: python frb_harmonic_mode_entropy_inversion_test107B.py frbs_unified.csv test107A_results.npy")
    sys.exit(1)

catalog_file = sys.argv[1]
qfile        = sys.argv[2]

print("============================================================")
print("Test 107B — Harmonic Mode Entropy & Inversion (σ_lm, I_lm)")
print("============================================================")

# ----------------------------
# load catalog
# ----------------------------
df = pd.read_csv(catalog_file)
N = len(df)
print(f"[info] loaded catalog: N_FRB = {N}")

# coords
ras  = np.deg2rad(df['ra'].values)
decs = np.deg2rad(df['dec'].values)
thetas = np.pi/2 - decs
phis   = ras

# fluence for inversion
F = df['fluence'].values
F_med = np.median(F)
F_hi  = np.percentile(F, 75)  # "high" fluence bin
F_lo  = np.percentile(F, 50)  # "mid" bin

# times
t = df['mjd'].values
t_min, t_max = t.min(), t.max()

# match chunking with Test 107A
NUM_CHUNKS = 8
edges = np.linspace(t_min, t_max, NUM_CHUNKS+1)

print(f"[info] time range MJD = {t_min:.1f} – {t_max:.1f}")
print(f"[info] chunks = {NUM_CHUNKS}")

# ----------------------------
# load 107A mode table
# ----------------------------
print(f"[info] loading Q-factor table: {qfile}")
Qtable = np.load(qfile, allow_pickle=True).item()
# expected Qtable[(l,m)] = dict(f0=..., gamma=..., Q=..., Pmax=...)

# ----------------------------
# harmonic parameters
# ----------------------------
lmax = 8
print(f"[info] lmax = {lmax}")

# store mode series
# a_lm[ chunk ][ l ][ m ] = complex amplitude
a_lm = [[{} for _ in range(lmax+1)] for __ in range(NUM_CHUNKS)]

# ----------------------------
# compute a_lm per chunk (same as 107A)
# ----------------------------

for ci in range(NUM_CHUNKS):
    lo, hi = edges[ci], edges[ci+1]
    mask = (t >= lo) & (t < hi)
    idx  = np.where(mask)[0]

    print(f"[info] chunk {ci}: N={len(idx)}, MJD={lo:.1f}-{hi:.1f}")

    if len(idx) == 0:
        # fill zeros
        for l in range(lmax+1):
            for m in range(-l, l+1):
                a_lm[ci][l][m] = 0+0j
        continue

    th = thetas[idx]
    ph = phis[idx]
    Fl = F[idx]

    for l in range(lmax+1):
        for m in range(-l, l+1):
            Y = sph_harm(m, l, ph, th)
            a = np.sum(Fl * np.conj(Y))
            a_lm[ci][l][m] = a

# ----------------------------
# build transitions per mode
# ----------------------------
# we track amplitude transitions chunk→chunk
# define microstates by phase quadrant of a_lm


def state_of(a):
    phase = np.angle(a)
    if phase < 0:
        phase += 2*np.pi
    return int((phase / (2*np.pi)) * 8)  # 0..7 angular bins


# initialize sigma and inversion
sigma_lm = {}
I_lm     = {}

# inversion bins
hi_mask = F >= F_hi
mid_mask = (F >= F_lo) & (F < F_hi)

print("[info] computing σ_lm and I_lm...")

for l in range(lmax+1):
    for m in range(-l, l+1):

        # ----------------------------
        # compute inversion per mode
        # ----------------------------
        # project all FRBs into |Y_lm|^2 weighting
        Y_all = sph_harm(m, l, phis, thetas)
        w = np.abs(Y_all)**2

        N_hi  = np.sum(w[hi_mask])
        N_mid = np.sum(w[mid_mask]) + 1e-9

        I_val = N_hi / N_mid
        I_lm[(l,m)] = I_val

        # ----------------------------
        # compute entropy production σ
        # using chunk transitions
        # ----------------------------
        # build transition counts
        Tmat = np.zeros((8,8))

        for ci in range(NUM_CHUNKS-1):
            a1 = a_lm[ci][l][m]
            a2 = a_lm[ci+1][l][m]

            s1 = state_of(a1)
            s2 = state_of(a2)
            Tmat[s1, s2] += 1

        # normalize rows to probabilities W
        W = Tmat / (Tmat.sum(axis=1, keepdims=True) + 1e-12)

        # stationary distribution P ~ row-sum
        P = Tmat.sum(axis=1)
        P = P / (P.sum() + 1e-12)

        # entropy production
        sigma = 0.0
        for i in range(8):
            for j in range(8):
                if W[i,j] > 0 and W[j,i] > 0:
                    sigma += P[i] * W[i,j] * np.log(W[i,j] / W[j,i])

        sigma_lm[(l,m)] = sigma

# ----------------------------
# null Monte Carlo (isotropic sky)
# ----------------------------
print("[info] building null ensemble (N=2000)...")

N_null = 2000
sigma_null = { (l,m): [] for l in range(lmax+1) for m in range(-l, l+1) }
I_null     = { (l,m): [] for l in range(lmax+1) for m in range(-l, l+1) }

rng = np.random.default_rng(123)

for trial in range(N_null):
    if trial % 200 == 0:
        print(f"  null {trial}/{N_null}")

    # random directions
    u = rng.random(N)
    ph_rand = rng.random(N)*2*np.pi
    th_rand = np.arccos(2*u - 1)
    Yrand = {}  # cache

    # random fluence shuffle
    Fshuf = rng.permutation(F)

    # chunkwise a_lm for this trial
    a_rand = [[{} for _ in range(lmax+1)] for __ in range(NUM_CHUNKS)]

    for ci in range(NUM_CHUNKS):
        lo, hi = edges[ci], edges[ci+1]
        mask = (t >= lo) & (t < hi)
        idx = np.where(mask)[0]

        if len(idx)==0:
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    a_rand[ci][l][m] = 0+0j
            continue

        th = th_rand[idx]
        ph = ph_rand[idx]
        Fl = Fshuf[idx]

        for l in range(lmax+1):
            for m in range(-l,l+1):
                Y = sph_harm(m, l, ph, th)
                a = np.sum(Fl * np.conj(Y))
                a_rand[ci][l][m] = a

    # per-mode null sigma & I
    for l in range(lmax+1):
        for m in range(-l, l+1):

            # inversion
            Y_all = sph_harm(m,l,ph_rand,th_rand)
            w = np.abs(Y_all)**2
            N_hi  = np.sum(w[Fshuf>=F_hi])
            N_mid = np.sum(w[(Fshuf>=F_lo)&(Fshuf<F_hi)]) + 1e-9
            I_n = N_hi / N_mid
            I_null[(l,m)].append(I_n)

            # transitions
            Tmat = np.zeros((8,8))
            for ci in range(NUM_CHUNKS-1):
                a1 = a_rand[ci][l][m]
                a2 = a_rand[ci+1][l][m]
                s1 = state_of(a1)
                s2 = state_of(a2)
                Tmat[s1,s2]+=1
            W = Tmat/(Tmat.sum(axis=1,keepdims=True)+1e-12)
            P = Tmat.sum(axis=1)
            P = P/(P.sum()+1e-12)

            sig=0.0
            for i in range(8):
                for j in range(8):
                    if W[i,j]>0 and W[j,i]>0:
                        sig += P[i]*W[i,j]*np.log(W[i,j]/W[j,i])

            sigma_null[(l,m)].append(sig)

# ----------------------------
# compute p-values
# ----------------------------
print("------------------------------------------------------------")
print("RESULTS (σ_lm, I_lm with p-values)")
print("------------------------------------------------------------")

for l in range(lmax+1):
    for m in range(-l,l+1):
        s = sigma_lm[(l,m)]
        i = I_lm[(l,m)]

        null_s = np.array(sigma_null[(l,m)])
        null_i = np.array(I_null[(l,m)])

        p_s = np.mean(null_s >= s)
        p_i = np.mean(null_i >= i)

        print(f"(l,m)=({l:2d},{m:3d}) | sigma={s: .4e} (p={p_s:.4f}) | I={i: .4e} (p={p_i:.4f})")

print("============================================================")
print("test 107B complete")
print("============================================================")
