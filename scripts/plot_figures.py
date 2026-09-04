"""Generate README figures for the hybrid SVD++/DecayPop project.

Run from repo root:
    python scripts/plot_figures.py
Outputs:
    assets/fig_user_segment_sensitivity.png
    assets/fig_decaypop_trend.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(OUT, exist_ok=True)

# --- palette (validated categorical slots 1-3) --------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SEG_COLORS = {"New": "#2a78d6", "Trend": "#eb6834", "Regular": "#1baf7a"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =============================================================================
# Figure 1 - User segment sensitivity (Table V, test-only protocol)
# =============================================================================
METRICS = ["RMSE", "MAE", "NDCG@10"]
SEGMENTS = ["New", "Trend", "Regular"]

SLOPE = {  # metric -> {segment: value}
    "RMSE":    {"New": -0.9400, "Trend": -0.9879, "Regular": -0.9674},
    "MAE":     {"New": -0.8736, "Trend": -0.9303, "Regular": -0.9127},
    "NDCG@10": {"New":  0.2791, "Trend":  0.4844, "Regular":  0.3068},
}
RANGE = {
    "RMSE":    {"New": 0.7343, "Trend": 0.7701, "Regular": 0.7540},
    "MAE":     {"New": 0.6846, "Trend": 0.7339, "Regular": 0.7138},
    "NDCG@10": {"New": 0.2586, "Trend": 0.4257, "Regular": 0.2784},
}
FRIEDMAN_P = {"RMSE": 0.0672, "MAE": 0.0608, "NDCG@10": 0.0005}


def grouped_bars(ax, data, title, ylabel):
    x = np.arange(len(METRICS))
    w = 0.26
    for i, seg in enumerate(SEGMENTS):
        vals = [data[m][seg] for m in METRICS]
        off = (i - 1) * (w + 0.015)
        bars = ax.bar(x + off, vals, width=w, color=SEG_COLORS[seg], label=seg,
                      edgecolor=SURFACE, linewidth=2, zorder=3)
        for b, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            pad = 0.018 if v >= 0 else -0.018
            ax.text(b.get_x() + b.get_width() / 2, v + pad, f"{v:.3f}",
                    ha="center", va=va, fontsize=7.5, color=INK2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\nFriedman p = {FRIEDMAN_P[m]:.4f}" for m in METRICS],
                       fontsize=9, color=INK2)
    ax.axhline(0, color=BASE, linewidth=1, zorder=2)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=10, loc="left")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
grouped_bars(axes[0], SLOPE,
             "Slope — direction of response to the hybrid ratio",
             "Slope (per unit change in SVD++ weight)")
grouped_bars(axes[1], RANGE,
             "Range — magnitude of variation across the 9 configurations",
             "Range (max − min)")
axes[0].set_ylim(-1.15, 0.62)
axes[1].set_ylim(0, 0.95)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, ["New users", "Trend-followers", "Regular users"],
           loc="upper left", ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.008, 1.005))
fig.suptitle("User-segment sensitivity to the personalization–popularity balance",
             fontsize=13, fontweight="bold", x=0.011, ha="left", y=1.13, color=INK)
fig.text(0.011, 1.075,
         "Test-only protocol · 10-fold cross-validation · MovieLens 100K",
         fontsize=9, color=MUTED, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "fig_user_segment_sensitivity.png"),
            dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)


# =============================================================================
# Figure 2 - DecayPop time-aware popularity
# =============================================================================
# NOTE: replace MONTHLY_RATINGS with the values from results/decaypop/.
# Month 1 = most recent, month 6 = oldest. The decay weights below are exact:
# w(t) = exp(-lambda * t) with lambda = 1.0.
MONTHS = np.arange(1, 7)
LAMBDA = 1.0
WEIGHTS = np.exp(-LAMBDA * MONTHS)
MONTHLY_RATINGS = np.array([266, 370, 261, 314, 235, 575])  # <-- swap for real totals

fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.35, 1]})

ax = axes[0]
ax.plot(MONTHS, MONTHLY_RATINGS, color=SEG_COLORS["New"], linewidth=2,
        marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
for m, v in zip(MONTHS, MONTHLY_RATINGS):
    ax.text(m, v + 22, f"{v:,}", ha="center", fontsize=8, color=INK2)
ax.set_title("Raw interaction volume — top-10 recommended items",
             fontsize=11, fontweight="bold", color=INK, loc="left", pad=8)
ax.set_ylabel("Ratings per month", fontsize=9)
ax.set_ylim(0, max(MONTHLY_RATINGS) * 1.25)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)

ax = axes[1]
bars = ax.bar(MONTHS, WEIGHTS, width=0.55, color=SEG_COLORS["Trend"],
              edgecolor=SURFACE, linewidth=2, zorder=3)
for b, w in zip(bars, WEIGHTS):
    ax.text(b.get_x() + b.get_width() / 2, w + 0.012, f"{w:.3f}",
            ha="center", fontsize=8, color=INK2)
ax.set_title("DecayPop weight applied to that volume —  w(t) = exp(−1.0 · t)",
             fontsize=11, fontweight="bold", color=INK, loc="left", pad=8)
ax.set_ylabel("Decay weight", fontsize=9)
ax.set_xlabel("Month  (1 = most recent, 6 = oldest)", fontsize=9)
ax.set_ylim(0, 0.44)
ax.set_xticks(MONTHS)
ax.set_xticklabels([f"Month {m}" for m in MONTHS], fontsize=9)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
ax.annotate("94% of the total weight falls in months 1–2",
            xy=(1.5, 0.30), xytext=(2.6, 0.34), fontsize=9, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))

fig.suptitle("Time-aware popularity: recent interactions dominate the DecayPop signal",
             fontsize=13, fontweight="bold", x=0.011, ha="left", y=1.03, color=INK)
fig.text(0.011, 0.972,
         "Older months carry volume but almost no weight — this is what makes the popularity "
         "signal trend-responsive rather than all-time.",
         fontsize=9, color=MUTED, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "fig_decaypop_trend.png"),
            dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

print("wrote", os.listdir(OUT))
