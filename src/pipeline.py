# %% [markdown]
# # GB1 four-position fitness prediction (small, reproducible demo)
# Question: with only a small set of measured variants, can simple models predict other variants
# and choose a better next batch than random sampling?
#
# Design decisions fixed BEFORE looking at candidate results:
# - Target y = continuous Fitness (regression, no binarisation). Input X = the 4 letters only (one-hot).
# - Random split of the full measured set: train 2,000 / val 2,000 / test 5,000 / candidate pool 20,000, disjoint.
# - Val chooses settings (Ridge alpha by val Spearman; MLP early stopping by val MSE). Test is only reported.
# - Candidate round: pick top 20 by predicted score from the pool, then reveal actual fitness.
#   Score = mean actual fitness of the 20 picks, and hits = picks within the pool's top 1% fitness.
#   Compare with 5,000 random draws of 20 from the same pool. Candidate results are NOT used to choose models.

# %%
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import matplotlib.pyplot as plt

import os, warnings
warnings.filterwarnings("ignore", message="An input array is constant")
SEED = int(os.environ.get("SEED", 0))
AA = "ACDEFGHIKLMNPQRSTVWY"
torch.set_num_threads(4)

df = pd.read_csv("data/raw/four_mutations_full_data.csv", usecols=["Variants", "Fitness"])
assert df.Variants.is_unique and df.Variants.str.len().eq(4).all() and df.Fitness.notna().all()

# %% [markdown]
# ## 1. Split (disjoint by construction: one random permutation cut into consecutive blocks)

# %%
N_TRAIN, N_VAL, N_TEST, N_POOL = 2000, 2000, 5000, 20000
perm = np.random.default_rng(SEED).permutation(len(df))
cuts = np.cumsum([N_TRAIN, N_VAL, N_TEST, N_POOL])
parts = np.split(perm[:cuts[-1]], cuts[:-1])
tr, va, te, pool = (df.iloc[p].reset_index(drop=True) for p in parts)
allidx = np.concatenate(parts)
assert len(set(allidx)) == len(allidx)  # no overlap
for name, d in [("train", tr), ("val", va), ("test", te), ("pool", pool)]:
    print(f"{name:5s} n={len(d):6d}  mean={d.Fitness.mean():.4f} median={d.Fitness.median():.4f} "
          f"frac>1={np.mean(d.Fitness > 1):.4f} frac==0={np.mean(d.Fitness == 0):.3f}")

# %% [markdown]
# ## 2. One-hot encoding: 4 positions x 20 amino acids = 80 binary features (fixed alphabet, so nothing is fitted on data)

# %%
def onehot(seqs):
    X = np.zeros((len(seqs), 4 * 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            X[i, j * 20 + AA.index(ch)] = 1.0
    return X

Xtr, Xva, Xte, Xpool = (onehot(d.Variants) for d in (tr, va, te, pool))
ytr, yva, yte, ypool = (d.Fitness.values.astype(np.float32) for d in (tr, va, te, pool))
print("X shape", Xtr.shape, " each row sums to", Xtr.sum(1)[:3], "(one letter per position)")

def metrics(y, p):
    return dict(MAE=float(np.mean(np.abs(y - p))), RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
                Spearman=float(spearmanr(y, p)[0]) if np.std(p) > 0 else float("nan"))

results = {}

# %% [markdown]
# ## 3. Baseline: predict the training mean for everything

# %%
results["mean baseline"] = metrics(yte, np.full_like(yte, ytr.mean()))
print("train mean =", ytr.mean(), results["mean baseline"], "(Spearman undefined: constant prediction)")

# %% [markdown]
# ## 4. One-hot + Ridge (alpha picked on validation Spearman)

# %%
rows = []
for a in [0.01, 0.1, 1, 10, 100, 1000]:
    m = Ridge(alpha=a).fit(Xtr, ytr)
    rows.append((a, metrics(yva, m.predict(Xva))))
    print(f"alpha={a:<7} val", {k: round(v, 4) for k, v in rows[-1][1].items()})
best_alpha = max(rows, key=lambda r: r[1]["Spearman"])[0]
ridge = Ridge(alpha=best_alpha).fit(Xtr, ytr)
p_ridge_te, p_ridge_pool = ridge.predict(Xte), ridge.predict(Xpool)
results["one-hot Ridge"] = metrics(yte, p_ridge_te)
print("chosen alpha:", best_alpha, " test:", results["one-hot Ridge"])

# %% [markdown]
# ## 5. Small PyTorch MLP (same one-hot input), 3 seeds
# Core loop: forward -> loss -> zero_grad -> backward -> optimizer.step. `model.eval()` + `torch.no_grad()` for evaluation.

# %%
def make_mlp():
    return nn.Sequential(nn.Linear(80, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

def train_mlp(seed, max_epochs=300, patience=30):
    torch.manual_seed(seed)
    model, loss_fn = make_mlp(), nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=128, shuffle=True, generator=g)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            pred = model(xb).squeeze(1)       # forward
            loss = loss_fn(pred, yb)          # loss
            opt.zero_grad()                   # clear old gradients
            loss.backward()                   # gradients
            opt.step()                        # update parameters
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv).squeeze(1), yv).item()
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, epoch + 1, best

def predict(model, X):
    with torch.no_grad():
        return model(torch.tensor(X)).squeeze(1).numpy()

models, te_preds, pool_preds = [], [], []
for s in [0, 1, 2]:
    m, ep, vloss = train_mlp(s)
    models.append(m)
    te_preds.append(predict(m, Xte)); pool_preds.append(predict(m, Xpool))
    print(f"seed {s}: stopped at epoch {ep}, best val MSE {vloss:.4f}, test", {k: round(v, 4) for k, v in metrics(yte, te_preds[-1]).items()})
te_preds, pool_preds = np.array(te_preds), np.array(pool_preds)
p_mlp_te, p_mlp_pool = te_preds.mean(0), pool_preds.mean(0)
results["MLP (single seed 0)"] = metrics(yte, te_preds[0])
results["MLP (3-seed mean)"] = metrics(yte, p_mlp_te)

# %% [markdown]
# ## 6. Test results (test set was not used to choose anything)

# %%
res = pd.DataFrame(results).T.round(4)
print(res.to_string())
res.to_csv("outputs/test_metrics.csv")

# %% [markdown]
# ## 7. Simple reliability check: does disagreement between the 3 MLPs relate to error?
# Only a partial proxy (initialisation variance); not a calibrated interval, and models can be consistently wrong.

# %%
std_te = te_preds.std(0)
abs_err = np.abs(yte - p_mlp_te)
rho = spearmanr(std_te, abs_err)[0]
q = np.quantile(std_te, [0.25, 0.75])
print(f"Spearman(seed std, |error|) = {rho:.3f}")
print(f"median |error|: low-disagreement quartile = {np.median(abs_err[std_te <= q[0]]):.4f}, "
      f"high-disagreement quartile = {np.median(abs_err[std_te >= q[1]]):.4f}")
hi = yte > 1
print(f"mean seed-std on true fitness>1: {std_te[hi].mean():.4f} vs others: {std_te[~hi].mean():.4f}")
print(f"mean |error| on true fitness>1: {abs_err[hi].mean():.4f} vs others: {abs_err[~hi].mean():.4f}")

# %% [markdown]
# ## 8. One retrospective candidate round (K = 20), same pool for every strategy

# %%
K, N_RANDOM = 20, 5000
thr = np.quantile(ypool, 0.99)   # "hit" = within the pool's top 1% of true fitness
def score(idx):
    return dict(mean_fitness=float(ypool[idx].mean()), hits_top1pct=int((ypool[idx] >= thr).sum()))
cand = {
    "Ridge top-20": score(np.argsort(-p_ridge_pool)[:K]),
    "MLP 3-seed top-20": score(np.argsort(-p_mlp_pool)[:K]),
    "Oracle top-20 (ceiling)": score(np.argsort(-ypool)[:K]),
}
rng = np.random.default_rng(123)
rand = [score(rng.choice(len(ypool), K, replace=False)) for _ in range(N_RANDOM)]
rm = np.array([r["mean_fitness"] for r in rand]); rh = np.array([r["hits_top1pct"] for r in rand])
cand["Random 20 (mean of 5000 draws)"] = dict(mean_fitness=float(rm.mean()), hits_top1pct=float(rh.mean()))
cdf = pd.DataFrame(cand).T
print(f"pool: n={len(ypool)}, top-1% threshold = {thr:.3f}, pool mean fitness = {ypool.mean():.4f}, expected random hits = {K*0.01:.2f}")
print(cdf.round(4).to_string())
print(f"random mean_fitness 5th-95th pct: {np.percentile(rm,5):.4f} - {np.percentile(rm,95):.4f}; 95th pct hits: {np.percentile(rh,95):.0f}")
for name in ["Ridge top-20", "MLP 3-seed top-20"]:
    print(f"{name}: fraction of random draws with mean_fitness >= this strategy = {(rm >= cand[name]['mean_fitness']).mean():.4f}")
cdf.to_csv("outputs/candidate_round.csv")
top = np.argsort(-p_mlp_pool)[:K]
print("MLP picks (variant, predicted, true):")
print(", ".join(f"{pool.Variants[i]}({p_mlp_pool[i]:.2f}/{ypool[i]:.2f})" for i in top))
print("distinct amino acids per position among picks:", [len(set(pool.Variants[i][j] for i in top)) for j in range(4)])

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, p, t in [(ax[0], p_ridge_te, "Ridge"), (ax[1], p_mlp_te, "MLP (3-seed mean)")]:
    a.scatter(yte, p, s=4, alpha=0.4); a.set_xlabel("true fitness (test)"); a.set_ylabel("predicted"); a.set_title(t)
    lim = max(yte.max(), p.max()); a.plot([0, lim], [0, lim], "r--", lw=1)
ax[2].scatter(std_te, abs_err, s=4, alpha=0.4); ax[2].set_xlabel("std across 3 seeds"); ax[2].set_ylabel("|error| (MLP mean)")
ax[2].set_title(f"disagreement vs error (rho={rho:.2f})")
plt.tight_layout(); plt.savefig("outputs/results.png", dpi=120)
