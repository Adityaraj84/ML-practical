import pandas as pd, numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

df = pd.read_csv("/home/roshan/Pictures/aditya ML/RS_Session_266_AU_1159_B_and_C_1.csv")
years = ['2016-17','2017-18','2018-19','2019-20','2020-21']
for y in years: df[y]=pd.to_numeric(df[y], errors='coerce')
df['mean_gen']=df[years].mean(axis=1)
def growth(r):
    v=r[years].dropna()
    return (v.iloc[-1]-v.iloc[0])/v.iloc[0]*100 if len(v)>=2 else np.nan
df['growth_pct']=df.apply(growth, axis=1)

X=StandardScaler().fit_transform(df[['mean_gen','growth_pct']].fillna(df[['mean_gen','growth_pct']].median()))
best=None
for k in range(2,7):
    km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X)
    s=silhouette_score(X,km.labels_)
    if best is None or s>best[0]: best=(s,k,km)
s,k,km=best
df['kmeans_cluster']=km.labels_
df['hier_cluster']=AgglomerativeClustering(n_clusters=k).fit(X).labels_

# plots
fig,ax=plt.subplots(figsize=(8,6))
sc=ax.scatter(df['mean_gen'],df['growth_pct'],c=df['kmeans_cluster'],cmap='tab10')
for _,r in df.iterrows():
    ax.annotate(r['State/UT-wise'],(r['mean_gen'],r['growth_pct']),fontsize=6)
ax.set_xlabel('Mean plastic waste (tonnes/yr)'); ax.set_ylabel('Growth %')
ax.set_title('State clusters by waste generation & growth')
fig.tight_layout(); fig.savefig("/home/roshan/Pictures/aditya ML/analysis/clusters.png")

df.to_csv("/home/roshan/Pictures/aditya ML/analysis/results.csv",index=False)
json.dump({"n":len(df),"k":int(k),"silhouette":float(s)}, open("/home/roshan/Pictures/aditya ML/analysis/summary.json","w"), indent=2)
print("done", len(df), k, round(s,3))
