#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test 99 — Harmonic-Phase Rotation Robustness of the Joint Entropy Deficit (Test 91)

Purpose:
    Determine whether the Test 91 joint-entropy deficit depends on the
    absolute phase orientation φ_h, or whether the structure would remain
    under arbitrary rotations φ_h → φ_h + Δ.

    If the deficit appears only near Δ = 0 (the true phase alignment),
    the structure is physically meaningful.
    If the deficit persists under many Δ, the effect may be tied to
    binning or phase conventions rather than a true cosmic correlation.

Pure version:
    - no fallbacks
    - no apply_filter
"""

import argparse
import logging
import numpy as np
import pandas as pd


# -------------------------------------------------------------
# utilities
# -------------------------------------------------------------

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")


def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"theta_u", "phi_h", "rt_sign"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"catalog missing required columns: {sorted(missing)}")
    return df


def bin_var(x, edges):
    idx = np.digitize(x, edges) - 1
    idx[(x < edges[0]) | (x >= edges[-1])] = -1
    return idx


def shannon_entropy_joint(theta_u, rt_sign, phi_h):
    theta_edges = np.array([0,20,35,50,90,180])
    rt_edges    = np.array([-2,0,2])
    phi_edges   = np.linspace(0,2*np.pi,12+1)

    phi_h = np.mod(phi_h, 2*np.pi)

    bT = bin_var(theta_u, theta_edges)
    bR = bin_var(rt_sign, rt_edges)
    bP = bin_var(phi_h,  phi_edges)

    ok = (bT>=0)&(bR>=0)&(bP>=0)
    if not np.any(ok):
        return np.nan

    C = np.zeros((5,2,12), dtype=int)
    for t,r,p in zip(bT[ok], bR[ok], bP[ok]):
        C[t,r,p] += 1

    P = C[C>0].astype(float)/np.sum(C)
    return -np.sum(P*np.log(P))


def test91_core(theta_u, rt_sign, phi_h, n_null=2000, seed=1):
    rng = np.random.default_rng(seed)

    H_real = shannon_entropy_joint(theta_u, rt_sign, phi_h)

    rt_vals  = rt_sign.copy()
    phi_vals = phi_h.copy()

    H_null = np.empty(n_null)
    for i in range(n_null):
        rt_s  = rng.permutation(rt_vals)
        phi_s = rng.permutation(phi_vals)
        H_null[i] = shannon_entropy_joint(theta_u, rt_s, phi_s)

    return (H_real,
            float(np.mean(H_null)),
            float(np.std(H_null, ddof=1)),
            float(np.mean(H_null <= H_real)))


# -------------------------------------------------------------
# Test 99 — harmonic-phase rotation
# -------------------------------------------------------------

def run_test99(df: pd.DataFrame, n_steps: int, n_null: int, seed: int):

    theta  = df["theta_u"].to_numpy(float)
    rt     = df["rt_sign"].to_numpy(float)
    phi_h0 = df["phi_h"].to_numpy(float)

    deltas = np.linspace(0, 2*np.pi, n_steps, endpoint=False)

    H_vals = np.empty(n_steps)
    p_vals = np.empty(n_steps)

    for i, d in enumerate(deltas):
        phi_rot = np.mod(phi_h0 + d, 2*np.pi)
        H, nm, ns, p = test91_core(theta, rt, phi_rot,
                                   n_null=n_null,
                                   seed=seed + i*1000)
        H_vals[i] = H
        p_vals[i] = p

    return deltas, H_vals, p_vals


def parse_args():
    p = argparse.ArgumentParser(
        description="Test 99 — Harmonic-Phase Rotation Robustness (Test 91)"
    )
    p.add_argument("catalog", type=str,
                   help="frbs_unified_for_test91.csv")
    p.add_argument("--n-steps", type=int, default=180,
                   help="number of phase-rotation steps over [0,2π)")
    p.add_argument("--n-null", type=int, default=2000,
                   help="null shuffles per Δ")
    p.add_argument("--seed", type=int, default=1,
                   help="rng seed base")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_catalog(args.catalog)
    logging.info(f"loaded catalog: {args.catalog}, N={len(df)}")

    deltas, H_vals, p_vals = run_test99(
        df, args.n_steps, args.n_null, args.seed
    )

    print("="*70)
    print("Test 99 — Harmonic-Phase Rotation Robustness")
    print("="*70)
    print("Δ(rad)     H(Δ)        p_deficit(Δ)")
    print("-"*70)
    for d, H, p in zip(deltas, H_vals, p_vals):
        print(f"{d:7.4f}   {H:10.6f}   {p:10.6f}")
    print("="*70)
    print("interpretation:")
    print("  - If the strong entropy deficit occurs only near Δ≈0 (the true phase),")
    print("    the phase-remnant-angle structure is physically meaningful.")
    print("  - If the deficit persists for many Δ, the signal may be tied to")
    print("    binning or coordinate conventions.")
    print("="*70)


if __name__ == "__main__":
    main()
