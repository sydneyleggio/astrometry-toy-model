# astrometry-toy-model

A small-scale numerical testbed for detecting a stochastic gravitational-wave background (GWB) through astrometric deflections of stars, worked out as the direct analog of the pulsar-timing-array (PTA) detection problem.

Pulsar timing arrays search for a gravitational-wave background hidden in the correlated timing residuals of many pulsars, with the expected pair correlation given by the Hellings-Downs curve. This project asks the same question for astrometry: if a GWB is instead measured through tiny apparent position shifts of stars on the sky, how do the same detection statistics behave, and does the PTA intuition, that a broad pairwise ("Hellings-Downs-like") correlation eventually beats a simpler per-star ("Common Process") statistic, carry over?

This repository is the prototype and validation sandbox for that question. It works at small to moderate star counts with full spherical geometry, and computes the key quantities two independent ways wherever possible, so a bug in one implementation cannot hide undetected. The goal is to get the math right at a scale where everything can be checked against a dense, brute-force alternative, before scaling the same framework up to survey-realistic star fields.

## What's actually being computed

- **Star fields** are sampled uniformly on the sphere, or on a smaller circular patch, using true great-circle geometry rather than a flat-sky approximation.
- **The overlap function**, the astrometric analog of the Hellings-Downs curve, is computed via a vectorized associated-Legendre recurrence that streams over multipole l instead of storing every term. This keeps memory use manageable even when the closest star pair pushes l_max into the tens of thousands.
- **Common Process (CP) SNR**, a per-star statistic, is inverted in closed form via a Sherman-Morrison identity, so the full weak-signal, intermediate-plateau, and strong-signal behavior comes out of one exact expression rather than separate limiting cases.
- **Hellings-Downs-like (HD) SNR**, a per-pair statistic, is computed for the full (non-diagonal) covariance matrix two independent ways: a dense eigendecomposition for smaller problems, and a matrix-free iterative solver (MINRES, with GMRES as a fallback) that never forms the full covariance matrix, for larger ones. Both use the same full noise convention and are checked against each other rather than trusted individually. Negative eigenvalues can and do show up here; that reflects the estimator covariance itself rather than a bug, and is consistent with the PTA literature.

## Repository layout

| File | What it does |
|---|---|
| `main.py` | Core physics: spherical star-field sampling, exact pairwise angular separations, the overlap function, and the full, weak, and intermediate CP SNR curves. Also includes a diagonal-only approximation of the HD SNR and diagnostics that check computed curves against known analytic slopes and plateaus. |
| `hd_full_matrix_snr.py` | The full (non-diagonal) HD covariance SNR, solved two independent ways, dense and matrix-free, and cross-checked against each other. |
| `SLOW_hd_full_matrix_snr.py` | Reference implementation kept for validating the optimized solver above. |
| `cp_full_curve_snr.py`, `hd_curve_case3approx_snr.py`, `hd_intermediate_sum.py`, `gamma_tilde_ab_sum.py` | Standalone scripts isolating individual pieces of the CP or HD SNR calculation for testing and debugging. |
| `gammavstheta.py`, `plot_cp_hd.py`, `comparison_plots_curveapprox.py`, `C_parallelplot.py` | Plotting and comparison utilities for the overlap function and SNR curves. |
| `main_astropycoords.py` | Star-field construction using Astropy's coordinate framework, as a cross-check on the custom spherical geometry used elsewhere. |

## Running it

```bash
pip install numpy scipy matplotlib astropy
python main.py                 # CP full curve, plus a diagonal-only HD approximation
python hd_full_matrix_snr.py   # full (non-diagonal) HD covariance SNR
```

Both scripts save a `.png` of the resulting SNR curves, tagged with the star count and field of view used, so different parameter sweeps do not overwrite each other.

## Status

This is exploratory, methods-development code, not a production pipeline. It exists to answer "does this math actually work" at a scale small enough to check by hand or against a brute-force alternative, before the same framework runs on survey-realistic star fields.

Developed as part of ongoing research with Kris Pardo (USC) and Jeffrey Hazboun (Oregon State) on astrometric detection of the gravitational-wave background.


