#!/usr/bin/env python3

"""
master_run.py

runs the full multi-messenger cross-layer detection pipeline:

1. download frbs
2. download neutrinos
3. clean neutrino data
4. generate realistic backup frbs (if needed)
5. generate synthetic uhecr data (fallback)
6. run coincidence search (tight)
7. run coincidence search (wide)
8. run triple-coincidence search
"""

import os
import subprocess

def run(script):
    print(f"\n--- running: {script} ---")
    subprocess.run([
        r"C:\Users\ratec\AppData\Local\Programs\Python\Python312\python.exe",
        script
    ], check=True)


def file_exists(fname):
    return os.path.isfile(fname) and os.path.getsize(fname) > 0


# 1. download/update frbs
run("download_frbs.py")

# 2. optional: alternate frb downloader
run("download_frbs_v2.py")

# 3. download neutrino alerts
run("download_neutrinos.py")

# 4. clean neutrino data
run("process_neutrinos_v2.py")

# 5. generate realistic frb fallback (only if frbs.csv missing)
if not file_exists("frbs.csv"):
    run("create_realistic_chime.py")

# 6. generate uhecr fallback (only if uhecr.csv missing)
if not file_exists("uhecr.csv"):
    run("create_uhecr_data.py")

# 7. run coincidence search
run("find_coincidences.py")

# 8. run wide coincidence search
run("find_coincidences_wide.py")

# 9. run triple-coincidence search
run("find_triple_coincidences.py")

print("\n--- pipeline complete ---")
