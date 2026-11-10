import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import json

df = pd.read_csv("/home/roshan/Pictures/aditya ML/RS_Session_266_AU_1159_B_and_C_1.csv")
years = ['2016-17','2017-18','2018-19','2019-20','2020-21']
for y in years:
    df[y] = pd.to_numeric(df[y], errors='coerce')

# mean generation (tonnes/yr) across available years
df['mean_gen'] = df[years].mean(axis=1)
# growth: last vs first available
def growth(row):
    vals = row[years].dropna()
    if len(vals) < 2: return np.nan
    return (vals.iloc[-1]-vals.iloc[0])/vals.iloc[0]*100
df['growth_pct'] = df.apply(growth, axis=1)

feat = df[['mean_gen','growth_pct']].copy()
feat = feat.fillna(feat.median())
X = StandardScaler().fit_transform(feat)

# KMeans choose k by silhouette
best=None
for k in range(2,7):
    km=KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    s=silhouette_score(X, km.labels_)
    if best is None or s>best[0]:
        best=(s,k,km)
s,k,km=best
df['kmeans_cluster']=km.labels_

# Hierarchical
hc=AgglomerativeClustering(n_clusters=k).fit(X)
df['hier_cluster']=hc.labels_

df.to_csv("/home/roshan/Pictures/aditya ML/analysis/results.csv", index=False)
summary = {
  "n_states": len(df),
  "best_k": int(k),
  "silhouette": float(s),
  "cluster_means": df.groupby('kmeans_cluster')[['mean_gen','growth_pct']].mean().round(2).to_dict('index')
}
with open("/home/roshan/Pictures/aditya ML/analysis/summary.json","w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
