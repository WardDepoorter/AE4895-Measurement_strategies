"""
XRD Diffractogram Plotter — NaBr
Reads Rigaku .asc format and plots 2θ vs intensity.

Usage:
    python plot_xrd.py

Requires: numpy, matplotlib
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams["font.size"] = 14
import re
import os
cd = os.getcwd()
# ── File to plot ──────────────────────────────────────────────────────────────
ASC_FILE_NAbr = cd+"/XRD/NaBr_Theta_2-Theta.asc"
ASC_FILE_NACl = cd+"/XRD/NaCl_Theta_2-Theta.asc"
# ── Parse the .asc file ───────────────────────────────────────────────────────
def parse_asc(filepath):
    """Parse a Rigaku ASCII (.asc) XRD file and return (two_theta, counts)."""
    start = stop = step = None
    counts = []
    in_data = False

    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Read scan parameters from header
            if line.startswith("*START"):
                start = float(line.split("=")[1])
            elif line.startswith("*STOP"):
                stop = float(line.split("=")[1])
            elif line.startswith("*STEP"):
                step = float(line.split("=")[1])
            elif line.startswith("*COUNT"):
                in_data = True   # next non-header lines are intensity data
                continue

            # Data lines: comma-separated integers (4 per row)
            if in_data and re.match(r"^\d", line):
                counts.extend([int(v.strip()) for v in line.split(",")])

    two_theta = np.arange(len(counts)) * step + start
    return two_theta, np.array(counts)


two_theta_NaBr, intensity_NaBr = parse_asc(ASC_FILE_NAbr)
two_theta_NaCl, intensity_NaCl = parse_asc(ASC_FILE_NACl)



# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8.5))

ax.plot(two_theta_NaBr, intensity_NaBr, linewidth=0.9, color="#003D7C", label="NaBr")
ax.plot(two_theta_NaCl, intensity_NaCl, linewidth=0.9, color="#8B0000", label="NaCl")

ax.set_xlabel("2θ (deg  )", fontsize=14)
ax.set_ylabel("Intensity (counts)", fontsize=14)
#ax.set_title("XRD Diffractogram — NaBr\n"  "Cu Kα, λ = 1.5406 Å  |  Rigaku MiniFlex", fontsize=13, pad=10)

ax.set_xlim(two_theta_NaBr[0], two_theta_NaBr[-1])
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=14)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")
ax.spines[["top", "right"]].set_visible(False)
plt.legend(fontsize=14)

plt.tight_layout()
plt.savefig("XRD.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved → NaBr_XRD.png")
# Normalize
intensity_NaBr_norm = intensity_NaBr / intensity_NaBr.max()
intensity_NaCl_norm = intensity_NaCl / intensity_NaBr.max()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8.5))
ax.plot(two_theta_NaBr, intensity_NaBr_norm, linewidth=0.9, color="#003D7C", label="NaBr")
ax.plot(two_theta_NaCl, intensity_NaCl_norm, linewidth=0.9, color="#8B0000", label="NaCl")

ax.set_xlabel("2θ (deg)", fontsize=14)
ax.set_ylabel("Normalised Intensity", fontsize=14)
ax.set_xlim(two_theta_NaBr[0], two_theta_NaBr[-1])
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=14)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")
ax.spines[["top", "right"]].set_visible(False)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("XRD_norm.png", dpi=200, bbox_inches="tight")
plt.show()