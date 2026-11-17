"""Extra analysis + figures for the interview prep document.
Reuses every variable from src/pipeline.py (same split, same models) by executing it silently.
Usage: SEED=0 python src/prep_analysis.py   (figures are only written for SEED=0)
"""
import contextlib, io, json, os
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("src/pipeline.py", encoding="utf-8").read())

import scipy.sparse as sp
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

os.makedirs("outputs/prep", exist_ok=True)
OUT = {}

# ---------- 1. Ridge with pairwise (epistasis-style) features: tests the "interactions" hypothesis ----------
def pair_feats(seqs):
    idx = np.array([[AA.index(c) for c in s] for s in seqs])         # n x 4
    n = len(idx)
    rows, cols = [], []
    for j in range(4):                                               # main effects (same as the 80 one-hot columns)
        rows.append(np.arange(n)); cols.append(j * 20 + idx[:, j])
    base = 80
    for a in range(4):
        for b in range(a + 1, 4):                                    # one 20x20 block per pair of positions
            rows.append(np.arange(n)); cols.append(base + idx[:, a] * 20 + idx[:, b]); base += 400
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, base))

Ptr, Pva, Pte, Ppool = (pair_feats(d.Variants) for d in (tr, va, te, pool))
rows = []
for a in [0.1, 0.3, 1, 3, 10, 30]:
    m = Ridge(alpha=a).fit(Ptr, ytr)
    rows.append((a, spearmanr(yva, m.predict(Pva))[0]))
alpha_pair = max(rows, key=lambda r: r[1])[0]
ridge_pair = Ridge(alpha=alpha_pair).fit(Ptr, ytr)
p_pair_te, p_pair_pool = ridge_pair.predict(Pte), ridge_pair.predict(Ppool)
OUT["pair_ridge_alpha"] = alpha_pair
OUT["test_metrics"] = {"mean": results["mean baseline"], "ridge": results["one-hot Ridge"],
                       "ridge_pairs": metrics(yte, p_pair_te), "mlp": results["MLP (3-seed mean)"]}
thr = np.quantile(ypool, 0.99)
def top_hits(p):
    i = np.argsort(-p)[:20]
    return dict(mean_fitness=float(ypool[i].mean()), hits=int((ypool[i] >= thr).sum()))
OUT["candidates"] = {"ridge": top_hits(p_ridge_pool), "ridge_pairs": top_hits(p_pair_pool), "mlp": top_hits(p_mlp_pool)}

# ---------- 2. Error by true-fitness band ----------
edges = [0, 0.01, 0.1, 1, np.inf]
labels = ["< 0.01", "0.01 – 0.1", "0.1 – 1", "> 1"]
band = np.digitize(yte, edges[1:-1])
preds = {"Mean baseline": np.full_like(yte, ytr.mean()), "Ridge": p_ridge_te, "MLP": p_mlp_te}
OUT["bands"] = {"labels": labels, "n": [int((band == b).sum()) for b in range(4)],
                "mae": {k: [float(np.abs(yte - p)[band == b].mean()) for b in range(4)] for k, p in preds.items()}}

# ---------- 3. Reliability numbers ----------
OUT["seed_std_rho"] = float(rho)
json.dump(OUT, open(f"outputs/prep/prep_numbers_seed{SEED}.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
if SEED != 0:
    raise SystemExit

# =====================  FIGURES (seed 0 only)  =====================
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "figure.dpi": 150})
# one palette for every figure: navy family + grey, one muted red for reference lines
BLUE, TEAL, AMBER, CRIMSON = "#4c78a8", "#59a89c", "#e8993e", "#d1495b"   # Ridge, Ridge+pairs, neural net, true winners / top-1% line
GREY, DIAG = "#a9b2bf", "#6b7280"                                           # baseline / random, guide lines
def save(fig, name):
    fig.savefig(f"outputs/prep/{name}.png", dpi=220, bbox_inches="tight"); fig.savefig(f"outputs/prep/{name}.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Fig 1: from the protein chain to one row of data ----
SLOT, NEUTRAL, TXT = "#9b7bb8", "#8a94a6", "#2d3748"       # purple = "the four slots / changed letters" (amber is reserved for the neural net)
full = df.copy()
best = full.loc[full.Fitness.idxmax()]
zero = full[(full.Fitness == 0)].iloc[0]
mid = full[(full.Fitness > 0.4) & (full.Fitness < 0.6)].iloc[0]
rows_ = [("VDGV", 1.0, "natural version"), ("VDAV", float(full[full.Variants == "VDAV"].Fitness.iloc[0]), "1 letter changed"),
         (mid.Variants, float(mid.Fitness), "4 letters changed"), (zero.Variants, 0.0, "4 letters changed"),
         (best.Variants, float(best.Fitness), "best in the data")]
fig, ax = plt.subplots(figsize=(6.6, 4.1))
ax.set_xlim(0, 10); ax.set_ylim(-0.7, 11.8); ax.axis("off")
# panel A: the chain
ax.text(0.05, 11.4, "A.  A protein is a long chain of building blocks. Each block is written as one letter.", fontsize=8.4, fontweight="bold", color=TXT, va="center")
items = [37, 38, 39, 40, 41, 42, 43, None, 52, 53, 54, 55, 56]
slot_letters = {39: "V", 40: "D", 41: "G", 54: "V"}
xs = np.arange(len(items)) * 0.66 + 0.5; yc = 9.7
ax.plot([xs[0] - 0.35, xs[-1] + 0.35], [yc, yc], color="#cbd5e0", lw=2, zorder=0)
for x, pos in zip(xs, items):
    if pos is None:
        ax.text(x, yc, "···", ha="center", va="center", fontsize=10, color=NEUTRAL, bbox=dict(fc="white", ec="none", pad=1.5), zorder=2)
    elif pos in slot_letters:
        ax.scatter([x], [yc], s=330, color=SLOT, zorder=2, edgecolor="none")
        ax.text(x, yc, slot_letters[pos], ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=3)
        ax.text(x, 8.95, f"slot {pos}", ha="center", va="center", fontsize=7, color=SLOT, fontweight="bold")
    else:
        ax.scatter([x], [yc], s=75, color="#e2e8f0", edgecolor="#cbd5e0", zorder=2)
ax.text(0.05, 8.3, "Only the four purple slots ever change. The rest of the chain stays exactly the same.", fontsize=7.8, color=TXT, va="center")
# panel B: rows
ax.text(0.05, 7.35, "B.  One row of data = one four-letter combination and its measured score", fontsize=8.4, fontweight="bold", color=TXT, va="center")
ax.text(0.05, 6.85, "Each slot holds one of 20 letters: 20 × 20 × 20 × 20 = 160,000 possible combinations; 149,361 are in the data.", fontsize=7.6, color="#4a5568", va="center")
for j, pos in enumerate([39, 40, 41, 54]):
    ax.text(0.4 + j * 0.62, 6.15, str(pos), ha="center", va="center", fontsize=7, color=SLOT, fontweight="bold")
ax.text(5.1, 6.15, "measured score (fitness)", ha="left", va="center", fontsize=7, color="#718096")
x_ref = 5.1 + np.log10(101) / np.log10(877.2) * 3.6
for i, (sq, f, note) in enumerate(rows_):
    y = 5.15 - i * 1.0
    for j, ch in enumerate(sq):
        ref = "VDGV"[j] == ch
        ax.add_patch(FancyBboxPatch((0.15 + j * 0.62, y - 0.36), 0.5, 0.72, boxstyle="round,pad=0.02", fc="#e2e8f0" if ref else SLOT, ec="none"))
        ax.text(0.4 + j * 0.62, y, ch, ha="center", va="center", fontsize=10, fontweight="bold", color=TXT if ref else "white")
    ax.text(2.85, y, note, va="center", fontsize=7.5, color="#4a5568")
    w = max(np.log10(1 + f * 100) / np.log10(1 + 876.2) * 3.6, 0.02)
    ax.add_patch(plt.Rectangle((5.1, y - 0.22), w, 0.44, fc=NEUTRAL, ec="none"))
    ax.text(5.1 + w + 0.1, y, f"{f:.3g}", va="center", fontsize=8, bbox=dict(fc="white", ec="none", pad=0.6))
ax.plot([x_ref, x_ref], [0.2, 5.7], ls="--", lw=0.9, color=CRIMSON)
ax.text(x_ref, -0.15, "VDGV's score = 1 (the reference)", ha="center", fontsize=7.3, color=CRIMSON)
save(fig, "fig1_rows")

# ---- Fig 2: the whole landscape, sorted ----
fs = np.sort(df.Fitness.values)
q = np.linspace(0, 100, len(fs))
fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.plot(q, fs, color=BLUE, lw=1.4)
ax.set_yscale("symlog", linthresh=1e-3, linscale=0.4)
ax.axhline(1, color=CRIMSON, ls="--", lw=1)
ax.set_xlabel("variants, sorted from weakest to strongest (%)"); ax.set_ylabel("fitness")
ax.set_yticks([0, 1e-2, 1e-1, 1]); ax.set_yticklabels(["0", ".01", ".1", "1"])
zero_pct = (df.Fitness == 0).mean() * 100; beat_pct = (df.Fitness > 1).mean() * 100
ax.annotate(f"{zero_pct:.0f}% score exactly 0", xy=(zero_pct / 2, 0), xytext=(3, 0.06), fontsize=7.5, arrowprops=dict(arrowstyle="-", color=GREY, lw=.6))
ax.annotate(f"only {beat_pct:.1f}% beat\nthe reference", xy=(100 - beat_pct / 2, 2), xytext=(40, 1.9), fontsize=7.5, color=CRIMSON,
            arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=.6))
save(fig, "fig2_landscape")
OUT_LANDSCAPE = dict(zero_pct=zero_pct, beat_pct=beat_pct, med=float(df.Fitness.median()))

# ---- Fig 3: the four sets (consulting-style: proportional bar + usage table) ----
NAVY, MID, LIGHT, PALE, INK = "#8aa94f", "#6b7a99", "#b08968", "#c77c9a", "#2d3748"   # set colours; amber stays neural-net only
sets = [("Train", 2000, NAVY, "Fits the model's weights", "Yes"),
        ("Validation", 2000, MID, "Picks settings; decides when to stop", "Yes"),
        ("Test", 5000, LIGHT, "Final report card", "No"),
        ("Pool", 20000, PALE, "Stands in for unmeasured candidates", "No")]
N = len(df)
fig = plt.figure(figsize=(6.6, 2.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
x0, x1, yb, hb = 3, 97, 33.5, 3.2
scale = (x1 - x0) / N; x = x0
for name, n, col, *_ in sets + [("Unused", N - 29000, "#eceff1")]:
    w = n * scale
    ax.add_patch(plt.Rectangle((x, yb), w, hb, fc=col, ec="white", lw=0.6))
    if name == "Unused":
        ax.text(x + w / 2, yb + hb / 2, f"Not used in this experiment: {n:,} variants ({n / N:.0%})", ha="center", va="center", fontsize=7.3, color="#718096")
    x += w
ax.text(x0, 39, "All 149,361 measured variants, drawn to scale", fontsize=8.3, fontweight="bold", color=INK, va="center")
# table
cx = [3, 17, 27, 62, 85]; heads = ["Set", "Variants", "Role", "May steer model choices?"]
heads_x = [cx[0], cx[1], cx[2], cx[3]]
ax.text(cx[0], 27.4, "Set", fontsize=7.8, fontweight="bold", color=INK); ax.text(cx[1] + 6, 27.4, "Variants", fontsize=7.8, fontweight="bold", color=INK, ha="right")
ax.text(cx[2] + 5, 27.4, "Role", fontsize=7.8, fontweight="bold", color=INK); ax.text(cx[4] - 1, 27.4, "May steer model\nchoices?", fontsize=7.8, fontweight="bold", color=INK, ha="left", va="center", linespacing=0.95)
ax.plot([x0, x1], [25.3, 25.3], color=INK, lw=0.9)
for i, (name, n, col, role, steer) in enumerate(sets):
    y = 21.3 - i * 5.4
    ax.add_patch(plt.Rectangle((cx[0], y - 1.0), 1.1, 2.2, fc=col, ec="none"))
    ax.text(cx[0] + 2.2, y + 0.1, name, fontsize=8, va="center", color=INK, fontweight="bold")
    ax.text(cx[1] + 6, y + 0.1, f"{n:,}", fontsize=8, va="center", ha="right", color=INK)
    ax.text(cx[2] + 5, y + 0.1, role, fontsize=8, va="center", color=INK)
    ax.text(cx[4] - 1, y + 0.1, steer, fontsize=8, va="center", color="#2f855a" if steer == "Yes" else "#a63d40", fontweight="bold")
    ax.plot([x0, x1], [y - 2.6, y - 2.6], color="#cbd5e0", lw=0.5)
ax.plot([x0, x1], [y - 2.6, y - 2.6], color=INK, lw=0.9)
fig.savefig("outputs/prep/fig3_split.png", dpi=220, bbox_inches="tight"); fig.savefig("outputs/prep/fig3_split.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Fig 4: predicted vs true on test ----
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
for a, p, t in [(axs[0], p_ridge_te, "One-hot + Ridge"), (axs[1], p_mlp_te, "Small neural net (3-seed mean)")]:
    hi = yte > 1
    a.scatter(yte[~hi], p[~hi], s=3, alpha=0.35, color=GREY, label="true fitness ≤ 1", rasterized=True)
    a.scatter(yte[hi], p[hi], s=5, alpha=0.7, color=CRIMSON, label="true fitness > 1", rasterized=True)
    a.set_xscale("symlog", linthresh=1e-2); a.set_yscale("symlog", linthresh=1e-2)
    a.plot([0, 9], [0, 9], color=DIAG, lw=0.9, ls="--"); a.set_title(t)
    a.set_xlabel("true fitness (test set)")
    a.text(0.03, 0.95, f"Spearman {metrics(yte, p)['Spearman']:.2f}\nMAE {metrics(yte, p)['MAE']:.3f}", transform=a.transAxes, va="top", fontsize=8)
axs[0].set_ylabel("predicted fitness"); axs[0].legend(loc="lower right", fontsize=7, markerscale=2, frameon=False)
plt.tight_layout(); save(fig, "fig4_pred_vs_true")

# ---- Fig 5: error by band ----
fig, ax = plt.subplots(figsize=(3.3, 2.6))
w = 0.26
for k, (name, col) in enumerate([("Mean baseline", GREY), ("Ridge", BLUE), ("MLP", AMBER)]):
    ax.bar(np.arange(4) + (k - 1) * w, OUT["bands"]["mae"][name], w, label=name if name != "MLP" else "Neural net", color=col)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{l}\n(n={n:,})" for l, n in zip(labels, OUT["bands"]["n"])], fontsize=7)
ax.set_xlabel("true fitness band (test set)"); ax.set_ylabel("mean absolute error"); ax.set_yscale("log")
ax.legend(fontsize=7, frameon=False, loc="upper left")
plt.tight_layout(); save(fig, "fig5_error_bands")

# ---- Fig 6: candidate round ----
rng = np.random.default_rng(7)
fig, ax = plt.subplots(figsize=(6.6, 2.7))
ax.hist(np.clip(ypool, 1e-4, None), bins=np.logspace(-4, 1, 70), weights=np.full(len(ypool), 1 / len(ypool)), alpha=0.0)
groups = [("Random 20\n(one draw)", rng.choice(len(ypool), 20, replace=False), GREY),
          ("Ridge top 20", np.argsort(-p_ridge_pool)[:20], BLUE),
          ("Ridge + pair\nfeatures top 20", np.argsort(-p_pair_pool)[:20], TEAL),
          ("Neural net top 20", np.argsort(-p_mlp_pool)[:20], AMBER),
          ("Oracle top 20\n(ceiling)", np.argsort(-ypool)[:20], "#2d3748")]
for i, (n, ix, col) in enumerate(groups):
    y = ypool[ix]
    hollow = n.startswith("Oracle")
    ax.scatter(np.maximum(y, 3e-4), np.full(20, i) + rng.uniform(-0.18, 0.18, 20), s=16, facecolor="none" if hollow else col, edgecolor=col if hollow else "none", alpha=0.9)
    ax.text(14, i, f"{int((y >= thr).sum())}/20 hits", va="center", fontsize=8, color="#2d3748", fontweight="bold")
ax.axvline(thr, color=CRIMSON, ls="--", lw=1); ax.text(thr * 1.06, -0.62, "pool top-1% line", color=CRIMSON, fontsize=7.5, va="center")
ax.set_xscale("log"); ax.set_xlim(2.5e-4, 40); ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups], fontsize=8)
ax.set_xlabel("true fitness of the 20 picked variants (log scale; zeros drawn at the left edge)"); ax.set_ylim(4.5, -0.9)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
plt.tight_layout(); save(fig, "fig6_candidates")

# ---- Fig 7: disagreement vs error ----
fig, ax = plt.subplots(figsize=(3.3, 2.6))
e = np.maximum(abs_err, 1e-4); s_ = np.maximum(std_te, 1e-4); hi = yte > 1
ax.scatter(s_[~hi], e[~hi], s=3, alpha=0.3, color=GREY, rasterized=True, label="true fitness ≤ 1")
ax.scatter(s_[hi], e[hi], s=6, alpha=0.8, color=CRIMSON, rasterized=True, label="true fitness > 1")
bins = np.quantile(s_, np.linspace(0, 1, 11)); mid_x, med_y = [], []
for lo, hi_ in zip(bins[:-1], bins[1:]):
    m = (s_ >= lo) & (s_ <= hi_); mid_x.append(np.median(s_[m])); med_y.append(np.median(e[m]))
ax.plot(mid_x, med_y, color=BLUE, lw=1.6, marker="o", ms=3, label="median per decile")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("disagreement between the 3 networks (std)"); ax.set_ylabel("|error| of their average")
ax.legend(fontsize=6.5, frameon=False, loc="lower right"); ax.text(0.03, 0.95, f"Spearman {rho:.2f}", transform=ax.transAxes, va="top", fontsize=8)
plt.tight_layout(); save(fig, "fig7_disagreement")
json.dump({**OUT, "landscape": OUT_LANDSCAPE, "fig1_rows": [(s, f) for s, f, _ in rows_]}, open("outputs/prep/prep_numbers_seed0.json", "w"), indent=1)
print("figures done", OUT_LANDSCAPE, rows_)
