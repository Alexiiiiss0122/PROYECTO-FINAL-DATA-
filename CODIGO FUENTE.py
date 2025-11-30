import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# ================================
# 1. CREAR CARPETA DE SALIDA
# ================================
OUTDIR = "proyecto_entrega"
os.makedirs(OUTDIR, exist_ok=True)
print("\n=== Carpeta creada:", OUTDIR, "===\n")

# ================================
# 2. CARGAR BASE
# ================================
df_original = pd.read_csv("Base_limpia_Bank.csv")
df = df_original.copy()
print("=== Primeras filas del dataset ===")
print(df.head(), "\n")

# ================================
# 3. LIMPIEZA
# ================================
print("=== Limpieza de datos... ===")
df["saldo"] = df["saldo"].astype(str).str.replace(",", "").str.strip()
df["saldo"] = pd.to_numeric(df["saldo"], errors="coerce")
df["saldo"] = df["saldo"].fillna(df["saldo"].median())

for col in ["Edad", "campaign", "contactos_previos"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

df["suscrito_bin"] = df["suscrito"].map({"yes": 1, "no": 0})

print("\n=== Datos numéricos LIMPIOS ===")
print(df.describe(), "\n")

df.to_csv(f"{OUTDIR}/base_limpia_procesada.csv", index=False)

# ================================
# 4. EDA NUMÉRICO
# ================================
num_cols = ["Edad", "saldo", "campaign", "contactos_previos"]
desc = df[num_cols].describe().transpose()
desc.to_csv(f"{OUTDIR}/eda_descriptivo_numerico.csv")

print("=== Resumen descriptivo numérico ===")
print(desc, "\n")

# Histogramas y boxplots
def plot_num(col):
    plt.figure(figsize=(8,4))
    plt.hist(df[col], bins=30)
    plt.title(f"Histograma de {col}")
    plt.savefig(f"{OUTDIR}/hist_{col}.png")
    plt.close()

    plt.figure(figsize=(6,3))
    plt.boxplot(df[col], vert=False)
    plt.title(f"Boxplot de {col}")
    plt.savefig(f"{OUTDIR}/box_{col}.png")
    plt.close()

for c in num_cols:
    plot_num(c)

print("=== Histogramas y Boxplots generados ===\n")

# ================================
# 5. EDA CATEGÓRICO
# ================================
cat_cols = ["trabajo","estado_civil","educacion","credito_vivienda",
            "prestamo","contacto","poutcome","suscrito"]

print("=== Conteos categóricos principales ===")
for col in cat_cols:
    vc = df[col].value_counts().head(10)
    print(f"\n--- {col} ---")
    print(vc)
    vc.to_csv(f"{OUTDIR}/counts_{col}.csv")

    plt.figure(figsize=(8,4))
    vc.plot(kind="bar")
    plt.title(f"Frecuencias de {col}")
    plt.savefig(f"{OUTDIR}/bar_{col}.png")
    plt.close()

print("\n=== Gráficas categóricas generadas ===\n")

# ================================
# 6. CORRELACIÓN
# ================================
corr = df[num_cols].corr()

print("=== Matriz de correlación ===")
print(corr, "\n")

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(num_cols)), num_cols, rotation=45)
plt.yticks(range(len(num_cols)), num_cols)
plt.title("Matriz de correlación")
plt.savefig(f"{OUTDIR}/heatmap_corr.png")
plt.close()

corr.to_csv(f"{OUTDIR}/corr.csv")

# ================================
# 7. PREPROCESAMIENTO PARA CLUSTERING
# ================================
print("=== Preparando datos para clustering... ===")

df_red = df.copy()

for col in cat_cols:
    top = df_red[col].value_counts().nlargest(8).index
    df_red[col] = df_red[col].where(df_red[col].isin(top), "Other")

X_cat = pd.get_dummies(df_red[cat_cols], drop_first=True)
X_num = StandardScaler().fit_transform(df_red[num_cols])

X = np.hstack([X_num, X_cat.values])

# ================================
# 8. BUSCAR EL MEJOR K
# ================================
print("\n=== Buscando mejor K con silhouette ===")
sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil_scores[k] = silhouette_score(X, labels)
    print(f"K={k} → silhouette={sil_scores[k]:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
best_sil = sil_scores[best_k]

print(f"\n>>> Mejor K encontrado: {best_k} (silhouette={best_sil:.4f})\n")

# ================================
# 9. ENTRENAR K-MEANS FINAL
# ================================
print("=== Entrenando modelo KMeans final... ===")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=15)
df_red["cluster"] = kmeans.fit_predict(X)

print("\n=== Distribución de clientes por cluster ===")
print(df_red["cluster"].value_counts(), "\n")

# ================================
# 10. PCA PARA GRAFICAR
# ================================
print("=== Generando PCA para visualización ===")
pca = PCA(n_components=2)
p2 = pca.fit_transform(X)
df_red["pca1"], df_red["pca2"] = p2[:,0], p2[:,1]

plt.figure(figsize=(7,6))
for cl in df_red["cluster"].unique():
    temp = df_red[df_red["cluster"]==cl]
    plt.scatter(temp["pca1"], temp["pca2"], s=10, label=f"Cluster {cl}")
plt.legend()
plt.title("Clusters en PCA")
plt.savefig(f"{OUTDIR}/pca_clusters.png")
plt.close()

print("=== PCA guardado ===\n")

# ================================
# 11. PERFILES DE CLUSTER
# ================================
profiles = df_red.groupby("cluster").agg(
    count=("Edad","size"),
    edad_mean=("Edad","mean"),
    saldo_mean=("saldo","mean"),
    camp_mean=("campaign","mean"),
    prev_mean=("contactos_previos","mean"),
    suscrito_rate=("suscrito_bin","mean")
)

profiles.to_csv(f"{OUTDIR}/cluster_profiles.csv")

print("=== Perfiles de cluster ===")
print(profiles, "\n")

# ================================
# 12. CLIENTES POTENCIALES
# ================================
overall_rate = df_red["suscrito_bin"].mean()
target_clusters = profiles[profiles["suscrito_rate"]>overall_rate].index.tolist()

print(f"=== Tasa global de suscripción: {overall_rate:.4f} ===")
print("Clusters objetivo:", target_clusters, "\n")

# Distancias
dist = cdist(X, kmeans.cluster_centers_)
df_red["dist_center"] = dist[np.arange(len(X)), df_red["cluster"]]

pot = df_red[(df_red["suscrito_bin"]==0) & (df_red["cluster"].isin(target_clusters))]
top50 = pot.sort_values("dist_center").head(50)

print("=== TOP 50 CLIENTES POTENCIALES ===")
print(top50[["Edad","saldo","cluster","dist_center"]].head(50))

top50.to_csv(f"{OUTDIR}/top50_potenciales.csv", index=False)

# ================================
# 13. REPORTE TXT
# ================================
with open(f"{OUTDIR}/reporte.txt","w",encoding="utf-8") as f:
    f.write(
f"""
PROYECTO FINAL – CLUSTERING Y CLIENTES POTENCIALES

Registros totales: {len(df)}
Mejor k encontrado: {best_k}
Silhouette: {best_sil:.4f}

Tasa global de suscripción: {overall_rate:.4f}
Clusters objetivo: {target_clusters}

Archivos generados en carpeta '{OUTDIR}':
- Histogramas
- Boxplots
- Barras categóricas
- Heatmap
- PCA clusters
- Perfiles de cluster
- Top 50 clientes potenciales
"""
    )

print("\n=== REPORTE GENERADO CORRECTAMENTE ===")
print("Todos los archivos están GUARDADOS en la carpeta:", OUTDIR)
