[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17778004.svg)](https://doi.org/10.5281/zenodo.17778004)

FRB Analysis Toolkit

A unified, reproducible environment for analyzing the current sample of Fast Radio Bursts (FRBs).
This repository consolidates all code, data products, simulations, and statistical tools used across
geometry, harmonic structure, temporal analysis, null testing, and multi-domain robustness checks.

The codebase is designed to be fully transparent, versioned, and archival through Zenodo.

Overview

This repository provides:

sky and coordinate transformation utilities

harmonic and multipole analysis modules

radial, shell, and geometric structure tests

temporal and frequency-domain exploration tools

null simulations and footprint-corrected robustness tests

multi-messenger comparison tools

full unified FRB dataset (CSV)

auxiliary maps, diagnostic figures, and summary products

All scripts are self-contained and can be executed independently or combined into a full pipeline.

Data Included

unified FRB catalog (CSV)

detector-specific sky catalogs (Parkes, CHIME, ASKAP, etc.)

neutrino and auxiliary comparison datasets

harmonic coefficients, PCA shells, and diagnostic maps

summary statistics and null distributions

Reproducibility

All results in this repository can be reproduced directly by running the corresponding Python scripts.


To ensure deterministic output:

python script_name.py input_dataset.csv


For most modules, a standard unified dataset is included:

frbs_unified.csv

Citation

If you use this repository, please cite:

EsclarmondeAme, FRB Analysis Toolkit, Zenodo (2025).
DOI: 10.5281/zenodo.17778004

BibTeX:

@software{frb_analysis_2025,
  author       = {EsclarmondeAme},
  title        = {FRB Analysis Toolkit},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17778004},
  url          = {https://doi.org/10.5281/zenodo.17778004}
}

License

All material is provided for research and non-commercial use.
Additional licensing may be added in future releases.

Versioning

This repository is archived through Zenodo.
Each GitHub release automatically receives a permanent DOI.

Current archived version: v2.1

Contact

For questions or collaborations, open an Issue on the repository.
