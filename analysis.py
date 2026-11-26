"""Plastic waste generation across Indian states: cluster-based assessment.
Run from the repository root:  python analysis.py
Reads data/*.csv, writes results/*.csv|json and figures/*.png (all relative paths)."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
(ROOT/"results").mkdir(exist_ok=True); (ROOT/"figures").mkdir(exist_ok=True)
SEED = 42
df = pd.read_csv(ROOT/"data/plastic_waste_cpcb.csv")
YEARS = list(df.columns[1:]); V = df[YEARS].to_numpy(float)
res = {}

# ---------- features ----------
def endpoints(row):
    idx = np.where(~np.isnan(row))[0]
    return idx[0], idx[-1]
def features(V):
    mean = np.nanmean(V, axis=1)
    first = np.array([V[i, endpoints(V[i])[0]] for i in range(len(V))])
    last = np.array([V[i, endpoints(V[i])[1]] for i in range(len(V))])
    span = np.array([endpoints(V[i])[1]-endpoints(V[i])[0] for i in range(len(V))])
    return mean, first, last, span
mean, first, last, span = features(V)
growth = 100*(last-first)/first
F = pd.DataFrame({"State_UT": df.State_UT, "mean_generation": mean, "growth_pct": growth,
                  "n_obs": (~np.isnan(V)).sum(1), "span_years": span})

def kfit(X, k, seed=SEED, n_init=10):
    return KMeans(k, n_init=n_init, random_state=seed).fit(X)
def sweep(X):
    return {k: float(silhouette_score(X, kfit(X, k).labels_)) for k in range(2, 7)}
def scale(A): return StandardScaler().fit_transform(A)

# ---------- 1. main solution ----------
X = scale(np.c_[F.mean_generation, F.growth_pct])
sil = sweep(X); res["silhouette_by_k"] = sil
K = max(sil, key=sil.get); km = kfit(X, K); ref = km.labels_
F["cluster"] = ref
res["selected_k"] = K; res["silhouette"] = sil[K]
res["inertia_k1_8"] = {k: float(kfit(X, k).inertia_) for k in range(1, 9)}
tab1 = F.groupby("cluster").agg(n=("State_UT","size"), mean_generation=("mean_generation","mean"),
                                mean_growth=("growth_pct","mean")).round(2)
tab1.to_csv(ROOT/"results/table1_clusters.csv"); F.to_csv(ROOT/"results/state_level_results.csv", index=False)
res["table1"] = tab1.reset_index().to_dict("records")
res["cluster_members"] = {int(c): F.State_UT[F.cluster==c].tolist() for c in sorted(F.cluster.unique())}

# ---------- 2. missingness ----------
miss = pd.DataFrame({"year": YEARS, "missing": np.isnan(V).sum(0)})
miss.to_csv(ROOT/"results/table_missingness.csv", index=False)
res["missing_by_year"] = dict(zip(YEARS, miss.missing.tolist()))
res["span_distribution"] = F.span_years.value_counts().sort_index().to_dict()
res["n_states_missing_any"] = int(np.isnan(V).any(1).sum())

# ---------- 3. stability across seeds and bootstrap ----------
rng = np.random.default_rng(SEED)
seed_ari = [adjusted_rand_score(ref, kfit(X, K, seed=s, n_init=1).labels_) for s in range(100)]
res["seed_ari_mean"] = float(np.mean(seed_ari)); res["seed_ari_min"] = float(np.min(seed_ari))
B = 500; boot_ari = []; boot_k = []; raw = np.c_[F.mean_generation, F.growth_pct]
for b in range(B):
    i = rng.integers(0, len(raw), len(raw))
    sc = StandardScaler().fit(raw[i]); Xb = sc.transform(raw[i])
    if len(np.unique(i)) < 8: continue
    s = {}
    for k in (2, 3, 4, 5, 6):
        try: s[k] = silhouette_score(Xb, KMeans(k, n_init=5, random_state=b).fit(Xb).labels_)
        except Exception: pass
    boot_k.append(max(s, key=s.get))
    kb = KMeans(K, n_init=5, random_state=b).fit(Xb)
    boot_ari.append(adjusted_rand_score(ref, kb.predict(sc.transform(raw))))
res["bootstrap_ari_mean"] = float(np.mean(boot_ari)); res["bootstrap_ari_sd"] = float(np.std(boot_ari))
res["bootstrap_ari_p5_p95"] = [float(np.percentile(boot_ari, 5)), float(np.percentile(boot_ari, 95))]
res["bootstrap_k_share"] = {int(k): float(np.mean(np.array(boot_k)==k)) for k in range(2, 7)}

# null baselines for the silhouette: random Gaussian features; independently permuted features
null_g = [silhouette_score(scale(rng.normal(size=(len(X), 2))), kfit(scale(rng.normal(size=(len(X), 2))), K, n_init=3).labels_) for _ in range(0)]
null_g, null_p = [], []
for _ in range(300):
    Z = scale(rng.normal(size=(len(X), 2))); null_g.append(silhouette_score(Z, kfit(Z, K, n_init=3).labels_))
    Zp = scale(np.c_[rng.permutation(raw[:,0]), rng.permutation(raw[:,1])]); null_p.append(silhouette_score(Zp, kfit(Zp, K, n_init=3).labels_))
res["null_gaussian_sil_mean_p95"] = [float(np.mean(null_g)), float(np.percentile(null_g, 95))]
res["null_permuted_sil_mean_p95"] = [float(np.mean(null_p)), float(np.percentile(null_p, 95))]

# ---------- 4. skew treatment / alternative preprocessing ----------
def run_variant(name, A, idx=None):
    Xv = scale(A); s = sweep(Xv); kv = max(s, key=s.get)
    lab3 = kfit(Xv, 3).labels_
    base = ref if idx is None else ref[idx]
    sizes = sorted(np.bincount(lab3).tolist(), reverse=True)
    return {"variant": name, "n": len(Xv), "best_k": kv, "sil_best": round(s[kv], 3),
            "sil_k3": round(s[3], 3), "ari_k3_vs_original": round(adjusted_rand_score(base, lab3), 3), "sizes_k3": sizes}
rows = []
rows.append(run_variant("Original (mean, endpoint growth %)", raw))
ratio = np.log(last/first)
rows.append(run_variant("Log10 mean + log growth ratio", np.c_[np.log10(mean), ratio]))
cagr = np.where(span > 0, np.log(last/first)/np.where(span>0, span, 1), 0.0)
rows.append(run_variant("Log10 mean + annualised log growth", np.c_[np.log10(mean), cagr]))
rank = lambda a: pd.Series(a).rank().to_numpy()
rows.append(run_variant("Rank (percentile) scaling of both", np.c_[rank(mean), rank(growth)]))
rows.append(run_variant("Log10 mean only", np.c_[np.log10(mean)]))
# common window 2018-19..2020-21 (Meghalaya lacks 2020-21 -> excluded)
cw = ~np.isnan(V[:, 2:]).any(1)
Vw = V[cw][:, 2:]; mw = Vw.mean(1); gw = 100*(Vw[:, -1]-Vw[:, 0])/Vw[:, 0]
rows.append(run_variant("Common window 2018-19 to 2020-21 (n=%d)" % cw.sum(), np.c_[mw, gw], idx=np.where(cw)[0]))
rows.append(run_variant("Common window, log scales", np.c_[np.log10(mw), np.log(Vw[:, -1]/Vw[:, 0])], idx=np.where(cw)[0]))
# complete-case 2016-17..2020-21
cc = ~np.isnan(V).any(1); Vc = V[cc]
rows.append(run_variant("Complete cases only, 2016-17 to 2020-21 (n=%d)" % cc.sum(),
            np.c_[Vc.mean(1), 100*(Vc[:, -1]-Vc[:, 0])/Vc[:, 0]], idx=np.where(cc)[0]))
rows.append(run_variant("Complete cases only, log scales (n=%d)" % cc.sum(),
            np.c_[np.log10(Vc.mean(1)), np.log(Vc[:, -1]/Vc[:, 0])], idx=np.where(cc)[0]))
pd.DataFrame(rows).to_csv(ROOT/"results/table_sensitivity.csv", index=False)
res["sensitivity"] = rows
# alternative algorithms on original features
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
res["ari_ward_vs_kmeans"] = float(adjusted_rand_score(ref, AgglomerativeClustering(K, linkage="ward").fit_predict(X)))
res["ari_gmm_vs_kmeans"] = float(adjusted_rand_score(ref, GaussianMixture(K, random_state=SEED, n_init=5).fit(X).predict(X)))

# ---------- 5. regression (drivers) ----------
cov = pd.read_csv(ROOT/"data/state_covariates_census2011.csv")
D = F.merge(cov, on="State_UT", validate="1:1")
y = np.log10(D.mean_generation.to_numpy())
P = np.c_[np.log10(D.Population_2011), D.Urban_pct_2011]
res["corr_logpop_logwaste"] = float(np.corrcoef(P[:,0], y)[0,1]); res["corr_urban_logwaste"] = float(np.corrcoef(P[:,1], y)[0,1])
res["corr_pop_urban"] = float(np.corrcoef(P[:,0], P[:,1])[0,1])
loo = LeaveOneOut()
models = {"Mean-only baseline": None,
          "OLS": make_pipeline(StandardScaler(), LinearRegression()),
          "Ridge (alpha by inner CV)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 2, 30))),
          "Lasso (alpha by inner CV)": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=SEED, max_iter=20000)),
          "Random forest": RandomForestRegressor(300, min_samples_leaf=2, random_state=SEED)}
reg = []; preds = {}
for name, m in models.items():
    if m is None:
        p = np.array([y[np.arange(len(y)) != i].mean() for i in range(len(y))])
    else:
        p = cross_val_predict(m, P, y, cv=loo)
    e = y - p; preds[name] = p
    # repeated 5-fold CV for a mean +/- sd of RMSE
    rm = []
    for r in range(20):
        fe = []
        for tr, te in KFold(5, shuffle=True, random_state=r).split(P):
            pt = np.full(len(te), y[tr].mean()) if m is None else __import__("sklearn.base", fromlist=["clone"]).clone(m).fit(P[tr], y[tr]).predict(P[te])
            fe.append(np.mean((y[te]-pt)**2))
        rm.append(np.sqrt(np.mean(fe)))
    # paired bootstrap CI of RMSE difference vs baseline (LOOCV errors)
    eb = y - preds["Mean-only baseline"]
    d = [np.sqrt(np.mean(e[i]**2)) - np.sqrt(np.mean(eb[i]**2)) for i in (rng.integers(0, len(e), len(e)) for _ in range(2000))]
    reg.append({"model": name, "loocv_rmse": round(float(np.sqrt(np.mean(e**2))), 3),
                "kfold_rmse_mean": round(float(np.mean(rm)), 3), "kfold_rmse_sd": round(float(np.std(rm)), 3),
                "loocv_r2": round(float(1-np.sum(e**2)/np.sum((y-y.mean())**2)), 3),
                "diff_vs_baseline": round(float(np.sqrt(np.mean(e**2))-np.sqrt(np.mean(eb**2))), 3),
                "diff_ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]})
pd.DataFrame(reg).to_csv(ROOT/"results/table_regression.csv", index=False); res["regression"] = reg
# in-sample OLS coefficients and ridge/lasso details
import statsmodels.api as sm
ols = sm.OLS(y, sm.add_constant(StandardScaler().fit_transform(P))).fit()
res["ols"] = {"r2": float(ols.rsquared), "adj_r2": float(ols.rsquared_adj), "f_p": float(ols.f_pvalue),
              "coef_std": ols.params.tolist(), "p": ols.pvalues.tolist(), "ci": ols.conf_int().tolist()}
ridge_alphas, lasso_alphas = [], []
for tr, te in loo.split(P):
    r = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 2, 30))).fit(P[tr], y[tr]); ridge_alphas.append(r[-1].alpha_)
    l = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=SEED, max_iter=20000)).fit(P[tr], y[tr]); lasso_alphas.append(l[-1].alpha_)
res["ridge_alpha_median_range"] = [float(np.median(ridge_alphas)), float(min(ridge_alphas)), float(max(ridge_alphas))]
res["lasso_alpha_median_range"] = [float(np.median(lasso_alphas)), float(min(lasso_alphas)), float(max(lasso_alphas))]
res["rf_importance"] = dict(zip(["log10 population", "urban %"], RandomForestRegressor(300, min_samples_leaf=2, random_state=SEED).fit(P, y).feature_importances_.round(3).tolist()))
# error analysis on best model by LOOCV
best = min(reg, key=lambda r: r["loocv_rmse"])["model"]; res["best_regression"] = best
D["loocv_pred_log10"] = preds[best]; D["loocv_resid"] = y - preds[best]
D.sort_values("loocv_resid", key=np.abs, ascending=False).head(6)[["State_UT","mean_generation","loocv_pred_log10","loocv_resid"]].to_csv(ROOT/"results/table_top_residuals.csv", index=False)
res["top_residuals"] = D.reindex(D.loocv_resid.abs().sort_values(ascending=False).index).head(6)[["State_UT","mean_generation","loocv_resid"]].round(3).to_dict("records")
# per-capita (kg per person per year) by cluster - descriptive
D["kg_per_capita"] = D.mean_generation*1000/D.Population_2011
res["percapita_by_cluster"] = D.groupby("cluster").kg_per_capita.agg(["median","min","max"]).round(2).reset_index().to_dict("records")
res["percapita_top_bottom"] = {"top": D.nlargest(3,"kg_per_capita")[["State_UT","kg_per_capita"]].round(2).to_dict("records"),
                               "bottom": D.nsmallest(3,"kg_per_capita")[["State_UT","kg_per_capita"]].round(2).to_dict("records")}
D.to_csv(ROOT/"results/state_level_with_covariates.csv", index=False)

# ---------- figures ----------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})
colors = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax[0].plot(list(sil), list(sil.values()), "o-", color="#1f4e79"); ax[0].axvline(K, ls=":", c="grey")
ax[0].set(xlabel="Number of clusters K", ylabel="Silhouette score", title="(a) Silhouette, K = 2-6")
ks = list(res["inertia_k1_8"]); ax[1].plot(ks, list(res["inertia_k1_8"].values()), "o-", color="#7f3b08")
ax[1].set(xlabel="Number of clusters K", ylabel="Within-cluster SS", title="(b) Elbow diagnostic, K = 1-8")
plt.tight_layout(); plt.savefig(ROOT/"figures/fig1_k_selection.png"); plt.close()

plt.figure(figsize=(7.2, 2.8)); Z = linkage(X, "ward"); dendrogram(Z, no_labels=True, color_threshold=0, above_threshold_color="#444")
plt.ylabel("Ward distance"); plt.title("Ward dendrogram (standardised features; leaf labels omitted)"); plt.tight_layout()
plt.savefig(ROOT/"figures/fig2_dendrogram.png"); plt.close()

fig, ax = plt.subplots(figsize=(5.6, 3.6))
for c in sorted(F.cluster.unique()):
    s = F[F.cluster==c]; ax.scatter(s.mean_generation, s.growth_pct, s=28, c=colors[c], label=f"Cluster {c} (n={len(s)})", alpha=.85)
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=10); ax.set(xlabel="Mean generation (tonnes/year, log axis)", ylabel="Endpoint growth (%, symlog axis)")
ax.legend(frameon=False); plt.tight_layout(); plt.savefig(ROOT/"figures/fig3_clusters.png"); plt.close()

top = F.nlargest(10, "mean_generation").iloc[::-1]
fig, ax = plt.subplots(figsize=(5.6, 3.3)); ax.barh(top.State_UT, top.mean_generation/1000, color=[colors[c] for c in top.cluster])
ax.set(xlabel="Mean generation (thousand tonnes/year)"); plt.tight_layout(); plt.savefig(ROOT/"figures/fig4_top10.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
ax[0].hist(seed_ari, bins=np.linspace(0, 1.0001, 11), color="#1f4e79"); ax[0].set(title="(a) 100 seeds, single init", xlabel="ARI vs reference", ylabel="Count")
ax[1].hist(boot_ari, bins=np.linspace(-0.05, 1.0001, 15), color="#7f3b08"); ax[1].set(title="(b) 500 bootstrap resamples", xlabel="ARI vs reference")
ax[2].hist(null_g, bins=20, color="#999", alpha=.8, label="Random Gaussian"); ax[2].hist(null_p, bins=20, color="#ccb", alpha=.8, label="Permuted features")
ax[2].axvline(sil[K], c="red", ls="--", label="Observed"); ax[2].set(title="(c) Null silhouette, K=3", xlabel="Silhouette"); ax[2].legend(frameon=False, fontsize=6)
plt.tight_layout(); plt.savefig(ROOT/"figures/fig5_stability.png"); plt.close()

fig, ax = plt.subplots(figsize=(4.6, 3.6)); b = preds[best]
ax.scatter(y, b, c=[colors[c] for c in D.cluster], s=26); lim = [min(y.min(), b.min())-.1, max(y.max(), b.max())+.1]; ax.plot(lim, lim, "k:", lw=.8)
for _, r in D.reindex(D.loocv_resid.abs().sort_values(ascending=False).index).head(4).iterrows():
    ax.annotate(r.State_UT.replace(" and Daman and Diu","").replace("Dadra and Nagar Haveli","DNH&DD"), (np.log10(r.mean_generation), r.loocv_pred_log10), fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.set(xlabel="Observed log10 mean generation", ylabel=f"LOOCV prediction ({best.split(' (')[0]})"); plt.tight_layout()
plt.savefig(ROOT/"figures/fig6_regression.png"); plt.close()

json.dump(res, open(ROOT/"results/summary.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=1, default=float)[:6000])
