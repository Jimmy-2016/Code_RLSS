"""
Plot the recovery results from SimpleRecovery.py:
  - recovered-vs-true scatter per parameter (with identity line + Pearson r)
  - cross-correlation matrix of the recovered parameters (off-diagonal trade-offs)
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# which optimizer's results to plot: cma (default) | anneal | gp
METHOD = sys.argv[1] if len(sys.argv) > 1 else "cma"

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "SavedData", f"Recovery_{METHOD}.pkl"), "rb") as f:
    true_params, fit_params, names = pickle.load(f)

# ---- recovered vs true -----------------------------------------------------
fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
for i, ax in enumerate(axes):
    t, f = true_params[:, i], fit_params[:, i]
    r, p = pearsonr(t, f)
    ax.scatter(t, f, c="k", s=110, zorder=3)
    lo = min(t.min(), f.min())
    hi = max(t.max(), f.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=3)
    ax.set_title(f"{names[i]}\nr = {r:.2f} (p = {p:.3f})")
    ax.set_xlabel("true")
    ax.set_ylabel("recovered")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
fig.suptitle(f"Recovery — {METHOD}", y=1.02)
fig.tight_layout()

# ---- cross-correlation of recovered params (off-diagonal confusion) --------
C = np.corrcoef(fit_params.T)
fig2, ax2 = plt.subplots(figsize=(5, 4))
im = ax2.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
ax2.set_xticks(range(len(names)))
ax2.set_yticks(range(len(names)))
ax2.set_xticklabels(names)
ax2.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(names)):
        ax2.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                 color="k", fontsize=10)
ax2.set_title("Recovered-parameter correlation")
fig2.colorbar(im, ax=ax2)
fig2.tight_layout()

print("Pearson r (true vs recovered):")
for i, nm in enumerate(names):
    r, _ = pearsonr(true_params[:, i], fit_params[:, i])
    print(f"  {nm:7s}: {r:.3f}")

plt.show()
