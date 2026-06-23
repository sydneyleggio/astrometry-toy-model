"""
compare_snr_runs_fullmatrix.py
====================
Overlay CP and HD SNR curves from multiple hd_full_matrix_snr.py runs,
using the saved .npz data (not pixel-extracted images) for exact curves.

Works for any of the three comparison types you've been doing:
  - Same FoV, different N        (N-sweep)
  - Same N, different FoV         (FoV-sweep)
  - N and FoV scaled together     (fixed-density sweep)

The script doesn't need to know which type of sweep it is — it just reads
N_STARS/FIELD_SIZE_DEG out of each .npz file's saved metadata and builds
labels and colors automatically.

USAGE
-----
Edit the FILES list below to point at your .npz files, then run:

    python3 compare_snr_runs.py

Or import and call directly:

    from compare_snr_runs import plot_comparison
    plot_comparison(["run1.npz", "run2.npz", ...], out_path="my_overlay.png")
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; works locally and on the cluster
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# EDIT THIS LIST — paths to the .npz files you want to compare
# ------------------------------------------------------------
FILES = [
    "hd_full_matrix_snr_N100_FoV10.npz",
    "hd_full_matrix_snr_N200_FoV10.npz",
    "hd_full_matrix_snr_N300_FoV10.npz",
    "hd_full_matrix_snr_N400_FoV10.npz",
    "hd_full_matrix_snr_N500_FoV10.npz",
]

# Output filename for the comparison plot
OUT_PATH = "snr_comparison.png"


def load_run(path):
    """Load one saved run and return a dict of arrays + metadata."""
    d = np.load(path)
    return {
        "r_vals": d["r_vals"],
        "rho_cp": d["rho_cp"],
        "rho_hd": d["rho_hd"],
        "N_STARS": int(d["N_STARS"]),
        "FIELD_SIZE_DEG": float(d["FIELD_SIZE_DEG"]),
        "PHYSICAL_RATIO": float(d["PHYSICAL_RATIO"]),
    }


def make_label(run, varying):
    """Build a legend label showing only the parameter(s) that actually vary."""
    n, fov = run["N_STARS"], run["FIELD_SIZE_DEG"]
    if varying == "N":
        return f"N={n}"
    if varying == "FoV":
        return f"FoV={fov:g}\u00b0"
    return f"N={n}, FoV={fov:g}\u00b0"


def detect_sweep_type(runs):
    """
    Figure out which of the three sweep types this is, purely from the
    metadata already stored in each .npz file:
      - 'N'      : FIELD_SIZE_DEG is the same across all runs, N_STARS varies
      - 'FoV'    : N_STARS is the same across all runs, FIELD_SIZE_DEG varies
      - 'joint'  : both vary (e.g. fixed-density sweep)
    """
    n_vals = {r["N_STARS"] for r in runs}
    fov_vals = {r["FIELD_SIZE_DEG"] for r in runs}

    if len(fov_vals) == 1 and len(n_vals) > 1:
        return "N"
    if len(n_vals) == 1 and len(fov_vals) > 1:
        return "FoV"
    return "joint"


def plot_comparison(files, out_path="snr_comparison.png", title=None):
    """
    Build one overlay figure comparing CP and HD curves across all given
    .npz files. Color-grades each family (CP in blues, HD in oranges) by
    run order, so trends across the sweep are visible at a glance.
    """
    runs = [load_run(f) for f in files]
    sweep_type = detect_sweep_type(runs)

    # Sort runs by the parameter that's actually varying, so color gradients
    # and legend order follow a sensible progression rather than file order.
    if sweep_type == "N":
        runs.sort(key=lambda r: r["N_STARS"])
    elif sweep_type == "FoV":
        runs.sort(key=lambda r: r["FIELD_SIZE_DEG"])
    else:
        runs.sort(key=lambda r: (r["N_STARS"], r["FIELD_SIZE_DEG"]))

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    cp_cmap = plt.colormaps["Blues"]
    hd_cmap = plt.colormaps["Oranges"]
    shades = np.linspace(0.35, 0.95, max(len(runs), 2))

    for run, shade in zip(runs, shades):
        label = make_label(run, sweep_type)
        ax.loglog(
            run["r_vals"], run["rho_cp"],
            color=cp_cmap(shade), lw=2.2,
            label=fr"$\rho_{{\rm CP}}$, {label}",
        )
    for run, shade in zip(runs, shades):
        label = make_label(run, sweep_type)
        ax.loglog(
            run["r_vals"], run["rho_hd"],
            color=hd_cmap(shade), lw=2.2, ls="--",
            label=fr"$\rho_{{\rm HD}}$, {label}",
        )

    # All runs share the same physical operating point (it only depends on
    # fixed instrument/GW constants, not on N or FoV), so one vertical line
    # suffices regardless of sweep type.
    phys_r = runs[0]["PHYSICAL_RATIO"]
    ax.axvline(phys_r, color="k", lw=1.2, ls=":", alpha=0.7,
               label=rf"physical $r={phys_r:.1e}$")

    ax.set_xlabel(r"$P_{\rm gw}(f_l)\,/\,P_n(f_l)$", fontsize=13)
    ax.set_ylabel(r"$\rho$", fontsize=13)

    if title is None:
        if sweep_type == "N":
            fov = runs[0]["FIELD_SIZE_DEG"]
            title = f"CP and HD Full SNR Curves \u2014 N sweep (FoV = {fov:g}\u00b0)"
        elif sweep_type == "FoV":
            n = runs[0]["N_STARS"]
            title = f"CP and HD Full SNR Curves \u2014 FoV sweep (N = {n})"
        else:
            title = "CP and HD Full SNR Curves \u2014 Joint N/FoV Scaling"
    ax.set_title(title, fontsize=13)

    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"Saved comparison plot to {out_path}  (detected sweep type: {sweep_type})")
    plt.close(fig)


if __name__ == "__main__":
    plot_comparison(FILES, out_path=OUT_PATH)
