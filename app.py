"""
app.py — Application Streamlit : Classification vs Clustering sur MNIST
Adapté du notebook — Version complète avec :
  • Section 4 NOUVELLE : Idée Centrale (mapping hongrois, pureté des clusters)
  • Split fixe 50k/10k/10k (fidèle au notebook)
  • Comparaison multi-modèles : K-Means+PCA  vs  K-Means+UMAP  vs  GMM+UMAP
  • Normalisation L2 optionnelle
  • Tableau de pureté des clusters
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from scipy.optimize import linear_sum_assignment

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="MNIST : Classification vs Clustering",
    page_icon="🔢",
    initial_sidebar_state="expanded",
)

SEED = 42
np.random.seed(SEED)

PALETTE_10 = [
    "#E63946", "#F4A261", "#2A9D8F", "#264653", "#8338EC",
    "#FB5607", "#3A86FF", "#FF006E", "#06D6A0", "#FFB703",
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2d2d4e;
}
section[data-testid="stSidebar"] * { color: #e0e0ff !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.85rem; letter-spacing: 0.05em;
    text-transform: uppercase; color: #a0a0cc !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 2rem !important; font-weight: 700 !important;
    color: #4fc3f7 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important; text-transform: uppercase;
    letter-spacing: 0.08em; color: #9e9ebb !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d0d1a; border-radius: 12px;
    padding: 6px; gap: 4px; border: 1px solid #1e1e3a;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    font-size: 0.9rem; color: #8888aa;
    border-radius: 8px; padding: 8px 20px; transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important; font-weight: 600 !important;
}

.custom-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d2d5e; border-left: 4px solid #667eea;
    border-radius: 8px; padding: 16px 20px; margin: 12px 0;
    font-size: 0.92rem; line-height: 1.6; color: #d0d0ee;
}
.custom-box.success { border-left-color: #06D6A0; }
.custom-box.warning { border-left-color: #FFB703; }
.custom-box.danger  { border-left-color: #E63946; }
.custom-box.gold    { border-left-color: #FFD700; }

.main-title {
    font-family: 'Space Mono', monospace; font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 30%, #f093fb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 4px;
}
.sub-title { font-size: 1rem; color: #8888aa; margin-top: 0; }

.fancy-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #667eea, #f093fb, transparent);
    margin: 24px 0; border: none;
}

.compare-table {
    width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.9rem;
}
.compare-table th {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; padding: 12px 16px; text-align: center;
    font-weight: 600; font-size: 0.95rem;
}
.compare-table th:first-child { border-radius: 8px 0 0 0; }
.compare-table th:last-child  { border-radius: 0 8px 0 0; }
.compare-table td {
    background: #12122a; color: #d0d0ee;
    padding: 10px 16px; border-bottom: 1px solid #1e1e3a; text-align: center;
}
.compare-table td:first-child {
    text-align: left; font-weight: 500; color: #a0a0cc; background: #0d0d20;
}

.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.03em;
}
.badge-blue   { background: #1a3a5e; color: #4fc3f7; border: 1px solid #2a5a8e; }
.badge-purple { background: #2a1a4e; color: #b39ddb; border: 1px solid #4a3a7e; }
.badge-green  { background: #0a3a2a; color: #4caf50; border: 1px solid #1a6a4a; }
.badge-orange { background: #3a2a0a; color: #ffb74d; border: 1px solid #7a5a2a; }
.badge-gold   { background: #2a2000; color: #FFD700; border: 1px solid #5a4a00; }

.golden-rule {
    background: linear-gradient(135deg, #1a1000, #2a1a00);
    border: 2px solid #FFB703; border-radius: 12px;
    padding: 20px 28px; margin: 20px 0;
    font-size: 1.15rem; font-weight: 600; color: #FFD54F;
    text-align: center; font-family: 'Space Mono', monospace;
    letter-spacing: 0.03em; box-shadow: 0 0 30px rgba(255,183,3,0.15);
}

.idea-box {
    background: linear-gradient(135deg, #0a1a0a, #1a2a0a);
    border: 2px solid #06D6A0; border-radius: 12px;
    padding: 20px 28px; margin: 16px 0; color: #a0ffcc;
    font-family: 'Space Mono', monospace; font-size: 0.95rem;
    box-shadow: 0 0 25px rgba(6,214,160,0.1);
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important; transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def cluster_accuracy(y_true, y_pred, n_classes=10):
    """
    Calcule l'accuracy du clustering via l'algorithme hongrois (linear_sum_assignment).
    Trouve le meilleur mapping bijection cluster→label pour maximiser les bons appariements.
    Retourne : (accuracy %, dict {cluster_id: digit})
    """
    cost_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_lbl, pred_lbl in zip(y_true, y_pred):
        if pred_lbl < n_classes:
            cost_matrix[pred_lbl, true_lbl] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapping = dict(zip(row_ind, col_ind))
    y_mapped = np.array([mapping.get(c, 0) for c in y_pred])
    acc = accuracy_score(y_true, y_mapped) * 100
    return acc, mapping


def _dark_axes(ax):
    """Applique le style sombre uniforme à un axe matplotlib."""
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2d2d4e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT ET ENTRAÎNEMENT (cachés)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="⏳ Chargement de MNIST...")
def charger_mnist():
    """
    Charge MNIST — tente Keras en premier (rapide, ~5s),
    puis fallback sur fetch_openml si Keras n'est pas disponible.
    Split fidèle au notebook : 50k train / 10k val / 10k test.
    Normalisation pixel : [0,255] → [0,1].
    """
    try:
        # ── Option rapide : Keras / TensorFlow ──────────────────────────────
        from tensorflow.keras.datasets import mnist as keras_mnist
        (X_tr_raw, y_tr_raw), (X_test_raw, y_test_raw) = keras_mnist.load_data()

        X_tr_raw   = X_tr_raw.reshape(-1, 784).astype(np.float32)   / 255.0
        X_test_raw = X_test_raw.reshape(-1, 784).astype(np.float32) / 255.0
        y_tr_raw   = y_tr_raw.astype(int)
        y_test_raw = y_test_raw.astype(int)

        X_train, y_train = X_tr_raw[:50_000], y_tr_raw[:50_000]
        X_val,   y_val   = X_tr_raw[50_000:], y_tr_raw[50_000:]  # 10 000
        X_test,  y_test  = X_test_raw,         y_test_raw         # 10 000

    except Exception:
        # ── Fallback : fetch_openml (sklearn) ───────────────────────────────
        from sklearn.datasets import fetch_openml
        mnist  = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X_full = mnist.data.astype(np.float32) / 255.0
        y_full = mnist.target.astype(int)

        X_train, y_train = X_full[:50_000],          y_full[:50_000]
        X_val,   y_val   = X_full[50_000:60_000],    y_full[50_000:60_000]
        X_test,  y_test  = X_full[60_000:70_000],    y_full[60_000:70_000]

    return X_train, y_train, X_val, y_val, X_test, y_test


@st.cache_resource(show_spinner="🌲 Entraînement du Random Forest (~1–2 min)...")
def entrainer_random_forest(n_estimators: int, max_depth: int):
    X_train, y_train, X_val, y_val, X_test, y_test = charger_mnist()
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=5, min_samples_leaf=2,
        n_jobs=-1, random_state=SEED,
    )
    rf.fit(X_train, y_train)
    train_pred = rf.predict(X_train)
    val_pred   = rf.predict(X_val)
    test_pred  = rf.predict(X_test)
    return rf, train_pred, val_pred, test_pred


@st.cache_resource(show_spinner="🔍 PCA + K-Means en cours...")
def entrainer_clustering(n_components_pca: int, n_clusters: int, use_l2: bool):
    """
    Pipeline : StandardScaler [→ L2 Normalizer optionnel] → PCA → K-Means.
    Opère sur les 10 000 images du test set (fidèle au notebook).
    """
    _, _, _, _, X_test, y_test = charger_mnist()
    X_cluster = X_test        # 10 000 images
    y_true    = y_test

    # Pipeline de preprocessing
    steps = [("std", StandardScaler())]
    if use_l2:
        steps.append(("l2", Normalizer(norm="l2")))
    steps.append(("pca", PCA(n_components=n_components_pca, random_state=SEED)))
    pipe = Pipeline(steps)
    X_pca = pipe.fit_transform(X_cluster)

    pca_model = pipe.named_steps["pca"]

    kmeans = KMeans(
        n_clusters=n_clusters, init="k-means++",
        n_init=10, max_iter=300, random_state=SEED,
    )
    cluster_labels = kmeans.fit_predict(X_pca)

    # Métriques
    n_eval = min(3000, len(X_pca))
    sil = silhouette_score(X_pca[:n_eval], cluster_labels[:n_eval], random_state=SEED)
    ari = adjusted_rand_score(y_true, cluster_labels)
    nmi = normalized_mutual_info_score(y_true, cluster_labels)
    acc_map, mapping = cluster_accuracy(y_true, cluster_labels, n_classes=max(n_clusters, 10))

    return pca_model, kmeans, X_pca, cluster_labels, y_true, X_cluster, sil, ari, nmi, acc_map, mapping


@st.cache_data(show_spinner="📊 Méthode du coude...")
def calculer_elbow(n_components_pca: int, use_l2: bool):
    _, _, _, _, X_test, _ = charger_mnist()
    steps = [("std", StandardScaler())]
    if use_l2:
        steps.append(("l2", Normalizer(norm="l2")))
    steps.append(("pca", PCA(n_components=n_components_pca, random_state=SEED)))
    pipe = Pipeline(steps)
    X_pca = pipe.fit_transform(X_test)
    K_range, inerties = range(2, 13), []
    for k in K_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=3, random_state=SEED)
        km.fit(X_pca)
        inerties.append(km.inertia_)
    return list(K_range), inerties


@st.cache_data(show_spinner="📐 Projection t-SNE (patient, ~1–2 min)...")
def calculer_tsne(n_components_pca: int, n_clusters: int, use_l2: bool, n_tsne: int = 2000):
    _, _, X_pca, cluster_labels, y_true, _, _, _, _, _, _ = entrainer_clustering(
        n_components_pca, n_clusters, use_l2
    )
    n_pts = min(n_tsne, len(X_pca))
    tsne  = TSNE(n_components=2, perplexity=35, n_iter=800,
                 random_state=SEED, init="pca", learning_rate="auto")
    X_2d  = tsne.fit_transform(X_pca[:n_pts])
    return X_2d, cluster_labels[:n_pts], y_true[:n_pts]


@st.cache_resource(show_spinner="🔬 Comparaison multi-modèles (K-Means PCA / GMM PCA)...")
def comparer_modeles(n_components_pca: int, use_l2: bool):
    """
    Compare K-Means+PCA vs GMM+PCA (sans UMAP pour éviter la dépendance externe).
    Les deux s'appuient sur le même preprocessing.
    """
    _, _, _, _, X_test, y_test = charger_mnist()

    steps = [("std", StandardScaler())]
    if use_l2:
        steps.append(("l2", Normalizer(norm="l2")))
    steps.append(("pca", PCA(n_components=n_components_pca, random_state=SEED)))
    pipe   = Pipeline(steps)
    X_proc = pipe.fit_transform(X_test)

    configs = {
        "K-Means PCA (baseline)": KMeans(n_clusters=10, n_init=10, random_state=SEED),
        "K-Means PCA (n_init=50)": KMeans(n_clusters=10, n_init=50, max_iter=500,
                                           tol=1e-5, random_state=SEED),
        "GMM PCA (full)": GaussianMixture(n_components=10, covariance_type="full",
                                          n_init=5, random_state=SEED),
    }

    results = {}
    for name, model in configs.items():
        labels = model.fit_predict(X_proc)
        acc, _ = cluster_accuracy(y_test, labels)
        ari    = adjusted_rand_score(y_test, labels)
        nmi    = normalized_mutual_info_score(y_test, labels)
        sil    = silhouette_score(X_proc[:3000], labels[:3000])
        results[name] = {"Accuracy": acc, "ARI": ari, "NMI": nmi, "Silhouette": sil}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <span style='font-family: Space Mono, monospace; font-size: 1.1rem;
                     font-weight: 700; color: #a0a0ff;'>⚙️ HYPERPARAMÈTRES</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**🌲 Random Forest**")
    n_estimators = st.slider("n_estimators", 50, 300, 150, 25,
                             help="Nombre d'arbres de décision.")
    max_depth = st.slider("max_depth", 10, 50, 30, 5,
                          help="Profondeur maximale de chaque arbre.")

    st.divider()
    st.markdown("**📉 PCA**")
    n_components_pca = st.slider("Composantes PCA", 10, 100, 50, 5)
    use_l2 = st.toggle("Normalisation L2", value=False,
                       help="Ajoute un Normalizer(l2) après StandardScaler (notebook cell 14).")

    st.divider()
    st.markdown("**🔵 K-Means**")
    n_clusters = st.slider("K (clusters)", 5, 15, 10, 1)

    st.divider()
    st.markdown("**🗺️ t-SNE**")
    n_tsne = st.slider("Points t-SNE", 500, 3000, 1500, 250)

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#5555aa; text-align:center; padding:8px;'>
        Split fixe (notebook) :<br>
        Train 50k / Val 10k / Test 10k<br><br>
        Modifier un paramètre déclenche le recalcul.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-title'>MNIST : Classification vs Clustering</div>
<p class='sub-title'>
    Analyse comparative complète — Random Forest (supervisé)
    vs PCA + K-Means (non supervisé) — avec mapping hongrois &amp; comparaison multi-modèles
</p>
""", unsafe_allow_html=True)
st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖼️ Exploration",
    "🎯 Classification",
    "🔍 Clustering",
    "💡 Idée Centrale",
    "🧭 Comparatif",
])


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — EXPLORATION
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📚 Le Dataset MNIST")
    st.markdown("""
    <div class='custom-box'>
        <strong>MNIST</strong> contient <strong>70 000 images 28×28</strong> de chiffres manuscrits
        (0–9). Chaque image est aplatie en <strong>784 features</strong>. Le notebook utilise
        un split fixe : <strong>50 000 train / 10 000 validation / 10 000 test</strong>
        — contrairement à un sous-échantillonnage dynamique.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Chargement de MNIST..."):
        X_train, y_train, X_val, y_val, X_test, y_test = charger_mnist()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏋️ Train",       f"{len(X_train):,}", "50 000")
    col2.metric("✔️ Validation",   f"{len(X_val):,}",   "10 000")
    col3.metric("🧪 Test",         f"{len(X_test):,}",  "10 000")
    col4.metric("📐 Features",     "784",               "28×28 px")
    col5.metric("🏷️ Classes",      "10",                "Chiffres 0–9")

    st.divider()
    st.subheader("🖼️ Un exemple par classe (+ histogramme d'intensité)")

    # Récupère un exemple par classe (fidèle au notebook)
    class_examples = {}
    for i in range(len(X_train)):
        lbl = y_train[i]
        if lbl not in class_examples:
            class_examples[lbl] = X_train[i].reshape(28, 28)
        if len(class_examples) == 10:
            break

    fig_ex, axes_ex = plt.subplots(2, 10, figsize=(18, 4.2), facecolor="#0d0d1a")
    fig_ex.suptitle("Un exemple de chaque chiffre (ligne 1) + histogramme d'intensité (ligne 2)",
                    color="white", fontsize=12, fontweight="bold", y=1.01)
    for digit in range(10):
        img = class_examples[digit]
        axes_ex[0, digit].imshow(img, cmap="plasma", interpolation="nearest")
        axes_ex[0, digit].set_title(f"Chiffre {digit}", fontsize=9,
                                    color="#FFB703", fontweight="bold")
        axes_ex[0, digit].axis("off")
        axes_ex[1, digit].hist(img.flatten(), bins=20, color=PALETTE_10[digit],
                               alpha=0.85, edgecolor="none")
        axes_ex[1, digit].set_xlabel("Intensité", fontsize=7, color="#a0a0cc")
        axes_ex[1, digit].set_yticks([])
        axes_ex[1, digit].set_facecolor("#0d0d1a")
        axes_ex[1, digit].tick_params(colors="white")
        for sp in axes_ex[1, digit].spines.values():
            sp.set_color("#1e1e2e")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig_ex)
    plt.close(fig_ex)

    st.divider()
    st.subheader("📈 Distribution des classes (Train Set)")

    unique_cls, counts = np.unique(y_train, return_counts=True)
    fig_dist, ax_dist = plt.subplots(figsize=(10, 3.5), facecolor="#0d0d1a")
    _dark_axes(ax_dist)
    bars = ax_dist.bar(unique_cls, counts,
                       color=[PALETTE_10[c] for c in unique_cls], alpha=0.9,
                       edgecolor="#1e1e3a", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax_dist.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                     str(cnt), ha="center", va="bottom",
                     fontsize=9, color="white", fontweight="bold")
    ax_dist.set_xlabel("Classe", color="#a0a0cc", fontsize=11)
    ax_dist.set_ylabel("Nombre d'images", color="#a0a0cc", fontsize=11)
    ax_dist.set_title("Répartition des 10 classes dans le train set (50 000 images)",
                      color="white", fontsize=12, fontweight="bold")
    ax_dist.grid(True, axis="y", alpha=0.15, color="white")
    plt.tight_layout()
    st.pyplot(fig_dist)
    plt.close(fig_dist)


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🌲 Random Forest — Approche Supervisée")
    st.markdown("""
    <div class='custom-box'>
        <strong>Paramètres notebook :</strong> n_estimators=150, max_depth=30,
        min_samples_split=5, min_samples_leaf=2, n_jobs=-1.<br>
        Entraîné sur <strong>50 000 images étiquetées</strong>.
        Métrique principale : <strong>Accuracy</strong>.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Entraînement du Random Forest..."):
        rf, train_pred, val_pred, test_pred = entrainer_random_forest(n_estimators, max_depth)
        X_train, y_train, X_val, y_val, X_test, y_test = charger_mnist()

    train_acc = accuracy_score(y_train, train_pred) * 100
    val_acc   = accuracy_score(y_val,   val_pred)   * 100
    test_acc  = accuracy_score(y_test,  test_pred)  * 100

    st.divider()
    st.subheader("🏆 Résultats")
    c1, c2, c3 = st.columns(3)
    c1.metric("🏋️ Accuracy Train",     f"{train_acc:.2f}%")
    c2.metric("✔️ Accuracy Validation", f"{val_acc:.2f}%",
              delta=f"{val_acc - train_acc:.2f}% vs train")
    c3.metric("🧪 Accuracy Test",       f"{test_acc:.2f}%",
              delta=f"{test_acc - val_acc:.2f}% vs val")

    # --- Matrice de confusion ---
    st.divider()
    st.subheader("🎯 Matrice de Confusion (Test Set)")

    cm = confusion_matrix(y_test, test_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(9, 7), facecolor="#0d0d1a")
    ax_cm.set_facecolor("#0d0d1a")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10),
                linewidths=0.4, ax=ax_cm,
                cbar_kws={"label": "Nombre d'images", "shrink": 0.8})
    ax_cm.set_title("Matrice de Confusion — Random Forest (Test 10k)",
                    color="white", fontsize=13, fontweight="bold")
    ax_cm.set_xlabel("Classe Prédite",  fontsize=11, color="#a0a0cc")
    ax_cm.set_ylabel("Classe Réelle",   fontsize=11, color="#a0a0cc")
    ax_cm.tick_params(colors="white")
    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # --- Feature Importance ---
    st.divider()
    st.subheader("🔥 Importance des Pixels (28×28)")

    feature_importance   = rf.feature_importances_.reshape(28, 28)
    importances_sorted   = np.sort(rf.feature_importances_)[::-1]

    fig_fi, (ax_fi1, ax_fi2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d0d1a")
    for ax in (ax_fi1, ax_fi2):
        _dark_axes(ax)

    im = ax_fi1.imshow(feature_importance, cmap="hot", interpolation="bilinear")
    ax_fi1.set_title("Carte d'importance des pixels", color="white",
                     fontsize=11, fontweight="bold")
    ax_fi1.set_xlabel("Colonne pixel", color="#a0a0cc")
    ax_fi1.set_ylabel("Ligne pixel",   color="#a0a0cc")
    cbar = plt.colorbar(im, ax=ax_fi1, shrink=0.85)
    cbar.set_label("Importance relative", color="#a0a0cc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.ax.yaxis.set_tick_params(color="white")

    ax_fi2.bar(range(1, 21), importances_sorted[:20],
               color=[PALETTE_10[i % 10] for i in range(20)],
               alpha=0.9, edgecolor="#0d0d1a", linewidth=0.5)
    ax_fi2.set_title("Top 20 des pixels les plus discriminants",
                     color="white", fontsize=11, fontweight="bold")
    ax_fi2.set_xlabel("Rang", color="#a0a0cc")
    ax_fi2.set_ylabel("Importance", color="#a0a0cc")
    ax_fi2.grid(True, axis="y", alpha=0.15, color="white")

    fig_fi.suptitle("Feature Importance — Random Forest",
                    color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_fi)
    plt.close(fig_fi)

    st.markdown("""
    <div class='custom-box success'>
        💡 Les pixels les plus discriminants sont <strong>centraux (zone 8–20)</strong>.
        Les coins sont quasi-toujours vides et n'apportent aucune information.
    </div>
    """, unsafe_allow_html=True)

    # --- Rapport de classification ---
    st.divider()
    st.subheader("📋 Rapport de Classification par Classe")

    rows = []
    for digit in range(10):
        mask = y_test == digit
        preds = test_pred[mask]
        p  = np.sum(preds == digit) / max(np.sum(test_pred == digit), 1)
        r  = np.sum(preds == digit) / max(np.sum(mask), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        rows.append({"Chiffre": digit, "Précision": f"{p:.3f}",
                     "Rappel": f"{r:.3f}", "F1-Score": f"{f1:.3f}",
                     "Support": int(np.sum(mask))})
    st.dataframe(pd.DataFrame(rows).set_index("Chiffre"), width="stretch")

    # --- Images bien / mal classées ---
    st.divider()
    st.subheader("✅ Bien classées vs ❌ Mal classées")

    correct_idx   = np.where(test_pred == y_test)[0]
    incorrect_idx = np.where(test_pred != y_test)[0]
    rng_ex = np.random.RandomState(777)
    show_c = rng_ex.choice(correct_idx,   size=min(10, len(correct_idx)),   replace=False)
    show_w = rng_ex.choice(incorrect_idx, size=min(10, len(incorrect_idx)), replace=False)

    fig_bm, axes_bm = plt.subplots(2, 10, figsize=(18, 4.5), facecolor="#0d0d1a")
    fig_bm.suptitle("Ligne 1 : Bien classées ✓ | Ligne 2 : Mal classées ✗",
                    color="white", fontsize=12, fontweight="bold", y=1.01)
    test_imgs = X_test.reshape(-1, 28, 28)
    for i, idx in enumerate(show_c[:10]):
        axes_bm[0, i].imshow(test_imgs[idx], cmap="Greens", interpolation="nearest")
        axes_bm[0, i].set_title(f"✓ {y_test[idx]}", color="#06D6A0",
                                fontsize=10, fontweight="bold")
        axes_bm[0, i].axis("off")
    for i, idx in enumerate(show_w[:10]):
        axes_bm[1, i].imshow(test_imgs[idx], cmap="Reds", interpolation="nearest")
        axes_bm[1, i].set_title(f"✗ R:{y_test[idx]} P:{test_pred[idx]}",
                                color="#E63946", fontsize=8, fontweight="bold")
        axes_bm[1, i].axis("off")
    plt.tight_layout(pad=0.3)
    st.pyplot(fig_bm)
    plt.close(fig_bm)


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔵 PCA + K-Means — Approche Non Supervisée")

    l2_label = "activée ✅" if use_l2 else "désactivée ❌"
    st.markdown(f"""
    <div class='custom-box warning'>
        <strong>🚫 Labels masqués.</strong> Pipeline :
        <code>StandardScaler → {"L2 Normalizer → " if use_l2 else ""}PCA({n_components_pca}D) → K-Means(K={n_clusters})</code>
        <br>
        Normalisation L2 : <strong>{l2_label}</strong> (bascule dans la sidebar)
        <br>
        Données : <strong>10 000 images du test set</strong> (fidèle au notebook).
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Clustering en cours..."):
        pca_model, kmeans_model, X_pca, cluster_labels, y_true, X_raw, \
            sil, ari, nmi, acc_map, mapping = entrainer_clustering(
                n_components_pca, n_clusters, use_l2)

    # Métriques
    st.divider()
    st.subheader("📊 Métriques d'Évaluation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔷 Silhouette",       f"{sil:.4f}", help="[-1,1] — 1=parfait")
    c2.metric("🎯 ARI",              f"{ari:.4f}", help="[0,1]")
    c3.metric("🔗 NMI",              f"{nmi:.4f}", help="[0,1]")
    c4.metric("📐 Inertie K-Means",  f"{kmeans_model.inertia_:,.0f}")

    # PCA variance
    st.divider()
    st.subheader("📉 Analyse PCA — Variance Expliquée")

    cumvar = np.cumsum(pca_model.explained_variance_ratio_)
    n_comp = pca_model.n_components_

    fig_pca, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="#0d0d1a")
    for ax in (ax_p1, ax_p2):
        _dark_axes(ax)

    ax_p1.plot(range(1, n_comp+1), pca_model.explained_variance_ratio_*100,
               color="#4fc3f7", linewidth=2)
    ax_p1.fill_between(range(1, n_comp+1), pca_model.explained_variance_ratio_*100,
                       alpha=0.25, color="#4fc3f7")
    ax_p1.set_title("Variance par composante", color="white", fontsize=11, fontweight="bold")
    ax_p1.set_xlabel("Composante", color="#a0a0cc")
    ax_p1.set_ylabel("Variance (%)", color="#a0a0cc")
    ax_p1.grid(True, alpha=0.1, color="white")

    ax_p2.plot(range(1, n_comp+1), cumvar*100, color="#f093fb",
               linewidth=2, marker="o", markersize=3)
    ax_p2.axhline(y=80, color="#FFB703", linestyle="--", alpha=0.8,
                  label="Seuil 80%", linewidth=1.5)
    ax_p2.set_title("Variance cumulée", color="white", fontsize=11, fontweight="bold")
    ax_p2.set_xlabel("Nombre de composantes", color="#a0a0cc")
    ax_p2.set_ylabel("Variance cumulée (%)", color="#a0a0cc")
    ax_p2.legend(facecolor="#1a1a2e", edgecolor="#2d2d4e", labelcolor="white")
    ax_p2.grid(True, alpha=0.1, color="white")

    fig_pca.suptitle(f"PCA : {n_comp} composantes → {cumvar[-1]*100:.1f}% de variance",
                     color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_pca)
    plt.close(fig_pca)

    # Elbow
    st.divider()
    st.subheader("📐 Méthode du Coude")

    with st.spinner("Calcul des inerties..."):
        K_range, inerties = calculer_elbow(n_components_pca, use_l2)

    fig_el, ax_el = plt.subplots(figsize=(9, 4.5), facecolor="#0d0d1a")
    _dark_axes(ax_el)
    ax_el.plot(K_range, inerties, "o-", color="#4fc3f7", linewidth=2.5,
               markersize=8, label="Inertie")
    ax_el.axvline(x=10, color="#FFB703", linestyle="--", alpha=0.9,
                  linewidth=2, label="K=10 (nb classes réelles)")
    if n_clusters != 10:
        ax_el.axvline(x=n_clusters, color="#06D6A0", linestyle=":",
                      alpha=0.9, linewidth=2, label=f"K={n_clusters} (sélection)")
    ax_el.set_title("Elbow Method — Choix du K optimal",
                    color="white", fontsize=13, fontweight="bold")
    ax_el.set_xlabel("K", color="#a0a0cc", fontsize=11)
    ax_el.set_ylabel("Inertie", color="#a0a0cc", fontsize=11)
    ax_el.legend(facecolor="#1a1a2e", edgecolor="#2d2d4e", labelcolor="white")
    ax_el.set_xticks(K_range)
    ax_el.grid(True, alpha=0.1, color="white")
    plt.tight_layout()
    st.pyplot(fig_el)
    plt.close(fig_el)

    # t-SNE
    st.divider()
    st.subheader("🗺️ Projection t-SNE 2D")

    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        run_tsne = st.button("🚀 Calculer la projection t-SNE", key="btn_tsne")
    with col_info:
        st.markdown(f"""
        <div class='custom-box' style='margin:0; padding:10px 16px;'>
            ⏱️ <strong>{n_tsne} points</strong> — environ 1–2 min.
        </div>
        """, unsafe_allow_html=True)

    if run_tsne or "tsne_ok" in st.session_state:
        st.session_state["tsne_ok"] = True
        with st.spinner("t-SNE en cours..."):
            X_2d, cl_tsne, yt_tsne = calculer_tsne(n_components_pca, n_clusters, use_l2, n_tsne)

        fig_ts, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor="#0d0d1a")
        for ax in (ax_t1, ax_t2):
            _dark_axes(ax)
            ax.spines["bottom"].set_color("#1a1a2e")
            ax.spines["left"].set_color("#1a1a2e")

        for cid in range(n_clusters):
            m = cl_tsne == cid
            ax_t1.scatter(X_2d[m, 0], X_2d[m, 1], c=PALETTE_10[cid % 10],
                          s=6, alpha=0.65, label=f"Cluster {cid}")
        ax_t1.set_title("Clusters K-Means (sans labels)",
                        color="white", fontsize=11, fontweight="bold")
        ax_t1.legend(markerscale=3, fontsize=8, ncol=2,
                     facecolor="#1a1a2e", edgecolor="#2d2d4e", labelcolor="white")
        ax_t1.set_xlabel("Dim 1", color="#a0a0cc")
        ax_t1.set_ylabel("Dim 2", color="#a0a0cc")
        ax_t1.grid(True, alpha=0.07, color="white")

        for digit in range(10):
            m = yt_tsne == digit
            ax_t2.scatter(X_2d[m, 0], X_2d[m, 1], c=PALETTE_10[digit],
                          s=6, alpha=0.65, label=f"Chiffre {digit}")
        ax_t2.set_title("Vraies classes MNIST (référence)",
                        color="white", fontsize=11, fontweight="bold")
        ax_t2.legend(markerscale=3, fontsize=8, ncol=2,
                     facecolor="#1a1a2e", edgecolor="#2d2d4e", labelcolor="white")
        ax_t2.set_xlabel("Dim 1", color="#a0a0cc")
        ax_t2.set_ylabel("Dim 2", color="#a0a0cc")
        ax_t2.grid(True, alpha=0.07, color="white")

        fig_ts.suptitle(f"t-SNE ({n_tsne} pts) — PCA {n_comp}D → 2D",
                        color="white", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_ts)
        plt.close(fig_ts)
    else:
        st.info("👆 Cliquez pour lancer la projection t-SNE.")

    # Images par cluster
    st.divider()
    st.subheader("🖼️ Images Représentatives par Cluster (5 plus proches du centroïde)")

    centroids      = kmeans_model.cluster_centers_
    X_imgs         = X_raw.reshape(-1, 28, 28)
    n_show         = min(n_clusters, 10)

    fig_cl, axes_cl = plt.subplots(n_show, 5, figsize=(10, 2.2*n_show), facecolor="#0d0d1a")
    if n_show == 1:
        axes_cl = np.expand_dims(axes_cl, 0)
    fig_cl.suptitle(f"5 images représentatives des {n_show} premiers clusters",
                    color="white", fontsize=12, fontweight="bold")

    for cid in range(n_show):
        mask_c      = cluster_labels == cid
        idx_c       = np.where(mask_c)[0]
        dists       = np.linalg.norm(X_pca[idx_c] - centroids[cid], axis=1)
        closest     = idx_c[np.argsort(dists)[:5]]
        majority    = int(np.bincount(y_true[idx_c]).argmax())
        for j, img_idx in enumerate(closest):
            axes_cl[cid, j].imshow(X_imgs[img_idx], cmap="plasma", interpolation="nearest")
            axes_cl[cid, j].axis("off")
            if j == 0:
                axes_cl[cid, 0].set_ylabel(
                    f"C{cid} ≈ {majority}", fontsize=9, rotation=0,
                    labelpad=38, va="center",
                    color=PALETTE_10[cid % 10], fontweight="bold")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig_cl)
    plt.close(fig_cl)


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 4 — IDÉE CENTRALE  ← NOUVEAU (Section 4 du notebook)
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("💡 Idée Centrale — Mapping Optimal via l'Algorithme Hongrois")

    st.markdown("""
    <div class='idea-box'>
        ✨ <strong>Question clé :</strong> De combien de points le clustering non supervisé
        est-il en retard sur la classification supervisée ?<br><br>
        On utilise l'<strong>algorithme hongrois</strong> (<code>linear_sum_assignment</code>)
        pour trouver le meilleur mapping bijection <em>cluster → chiffre</em>
        qui maximise les bons appariements.
        L'accuracy résultante est la <strong>meilleure accuracy possible</strong>
        pour le clustering — en connaissant les labels <em>a posteriori</em>.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Calcul du mapping hongrois..."):
        _, _, _, _, X_test_c, y_test_c = charger_mnist()
        _, _, _, cluster_labels_c, y_true_c, _, _, _, _, acc_map_c, mapping_c = \
            entrainer_clustering(n_components_pca, n_clusters, use_l2)
        rf_c, _, _, test_pred_c = entrainer_random_forest(n_estimators, max_depth)
        test_acc_c = accuracy_score(y_test_c, test_pred_c) * 100

    st.divider()
    st.subheader("📊 Mapping Cluster → Chiffre & Pureté")

    # Tableau de pureté
    rows_purity = []
    for cid in sorted(mapping_c.keys()):
        digit    = mapping_c[cid]
        mask_c_i = cluster_labels_c == cid
        if np.sum(mask_c_i) == 0:
            continue
        purity = np.mean(y_true_c[mask_c_i] == digit) * 100
        size   = int(np.sum(mask_c_i))
        rows_purity.append({
            "Cluster": cid,
            "→ Chiffre": digit,
            "Pureté (%)": f"{purity:.1f}%",
            "Taille cluster": size,
        })

    df_purity = pd.DataFrame(rows_purity)
    st.dataframe(df_purity, width="stretch")

    # Graphique pureté par cluster
    st.divider()
    st.subheader("📊 Pureté de Chaque Cluster")

    purities = [float(r["Pureté (%)"].replace("%","")) for r in rows_purity]
    cids     = [r["Cluster"] for r in rows_purity]
    digits   = [r["→ Chiffre"] for r in rows_purity]

    fig_pur, ax_pur = plt.subplots(figsize=(10, 4), facecolor="#0d0d1a")
    _dark_axes(ax_pur)
    bars_pur = ax_pur.bar(range(len(cids)), purities,
                          color=[PALETTE_10[d % 10] for d in digits],
                          alpha=0.9, edgecolor="#0d0d1a", linewidth=0.5)
    for bar, p, d in zip(bars_pur, purities, digits):
        ax_pur.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{p:.0f}%\n(→{d})", ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold")
    ax_pur.set_xticks(range(len(cids)))
    ax_pur.set_xticklabels([f"C{c}" for c in cids], color="white")
    ax_pur.set_ylim(0, 115)
    ax_pur.set_xlabel("Cluster", color="#a0a0cc")
    ax_pur.set_ylabel("Pureté (%)", color="#a0a0cc")
    ax_pur.set_title("Pureté de chaque cluster (après mapping hongrois)",
                     color="white", fontsize=12, fontweight="bold")
    ax_pur.grid(True, axis="y", alpha=0.15, color="white")
    plt.tight_layout()
    st.pyplot(fig_pur)
    plt.close(fig_pur)

    # Comparaison RF vs Clustering
    st.divider()
    st.subheader("⚖️ RF vs Clustering — Graphique de l'Idée Centrale")

    ecart = test_acc_c - acc_map_c

    fig_idea, ax_idea = plt.subplots(figsize=(8, 5), facecolor="#0d0d1a")
    _dark_axes(ax_idea)

    models   = ["Random Forest\n(Supervisé)", f"K-Means+PCA\n(Non supervisé\n+ mapping hongrois)"]
    accs     = [test_acc_c, acc_map_c]
    colors_i = ["#3A86FF", "#FFB703"]

    bars_i = ax_idea.bar(models, accs, color=colors_i, alpha=0.85,
                         edgecolor="#0d0d1a", linewidth=0.5, width=0.5)
    for bar, acc in zip(bars_i, accs):
        ax_idea.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{acc:.2f}%", ha="center", va="bottom",
                     fontsize=13, color="white", fontweight="bold",
                     fontfamily="monospace")

    # Annotation de l'écart
    ax_idea.annotate(
        f"Écart : {ecart:.2f} pts\n= coût de l'absence de labels",
        xy=(1, acc_map_c), xytext=(0.5, (test_acc_c + acc_map_c)/2),
        ha="center", color="#E63946", fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.5),
    )

    ax_idea.set_ylim(0, 115)
    ax_idea.set_ylabel("Accuracy (%)", color="#a0a0cc", fontsize=11)
    ax_idea.set_title(
        "Idée Centrale : Accuracy RF vs Clustering (après mapping optimal)",
        color="white", fontsize=12, fontweight="bold"
    )
    ax_idea.grid(True, axis="y", alpha=0.15, color="white")
    plt.tight_layout()
    st.pyplot(fig_idea)
    plt.close(fig_idea)

    # Résumé textuel
    st.markdown(f"""
    <div class='custom-box gold'>
        <strong>📌 Résumé chiffré :</strong><br><br>
        • <strong>Random Forest</strong> (supervisé, 50k labels) → Accuracy test :
          <span style='color:#4fc3f7; font-family:monospace;'>{test_acc_c:.2f}%</span><br>
        • <strong>K-Means+PCA</strong> (non supervisé, 0 label) → Accuracy après mapping :
          <span style='color:#FFB703; font-family:monospace;'>{acc_map_c:.2f}%</span><br>
        • <strong>Écart</strong> :
          <span style='color:#E63946; font-family:monospace;'>{ecart:.2f} pts</span>
          = coût de l'absence de labels<br><br>
        → Le clustering retrouve les groupes <strong>SANS AUCUN label</strong> avec seulement
        <strong>{ecart:.1f} points</strong> d'écart sur un modèle supervisé entraîné sur 50 000
        exemples. C'est là toute la puissance — et la limite — de l'apprentissage non supervisé.
    </div>
    """, unsafe_allow_html=True)

    # --- Comparaison multi-modèles ---
    st.divider()
    st.subheader("🔬 Comparaison Multi-Modèles de Clustering")

    st.markdown("""
    <div class='custom-box'>
        Comparison de 3 stratégies de clustering sur la même projection PCA :<br>
        <strong>K-Means baseline</strong> vs <strong>K-Means optimisé (n_init=50)</strong>
        vs <strong>GMM (full covariance)</strong>.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Comparaison multi-modèles..."):
        results_comp = comparer_modeles(n_components_pca, use_l2)

    # Tableau
    df_comp = pd.DataFrame(results_comp).T
    df_comp.index.name = "Modèle"
    df_comp["Accuracy"] = df_comp["Accuracy"].apply(lambda x: f"{x:.2f}%")
    df_comp["ARI"]       = df_comp["ARI"].apply(lambda x: f"{x:.4f}")
    df_comp["NMI"]       = df_comp["NMI"].apply(lambda x: f"{x:.4f}")
    df_comp["Silhouette"]= df_comp["Silhouette"].apply(lambda x: f"{x:.4f}")
    st.dataframe(df_comp, width="stretch")

    # Graphique 4 métriques
    fig_comp, axes_comp = plt.subplots(1, 4, figsize=(16, 5), facecolor="#0d0d1a")
    metrics_c = ["Accuracy", "ARI", "NMI", "Silhouette"]
    colors_c  = ["#3A86FF", "#FF006E", "#06D6A0"]

    raw_results = comparer_modeles(n_components_pca, use_l2)  # dict with floats

    for ax_c, metric in zip(axes_comp, metrics_c):
        _dark_axes(ax_c)
        names_c = list(raw_results.keys())
        vals_c  = [raw_results[n][metric] / (100.0 if metric == "Accuracy" else 1.0)
                   for n in names_c]
        bars_c  = ax_c.bar(range(len(names_c)), vals_c, color=colors_c, alpha=0.85,
                           edgecolor="#0d0d1a", linewidth=0.5)
        ax_c.set_xticks(range(len(names_c)))
        ax_c.set_xticklabels(
            [n.split(" ")[0]+"…" if len(n)>15 else n for n in names_c],
            rotation=18, ha="right", fontsize=7.5, color="white")
        ax_c.set_title(metric, color="white", fontweight="bold", fontsize=10)
        ax_c.set_ylim(0, 1.1)
        ax_c.grid(axis="y", alpha=0.15, color="white")
        for bar, v in zip(bars_c, vals_c):
            ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f"{v:.3f}", ha="center", va="bottom",
                      fontsize=8, color="white", fontweight="bold")

    fig_comp.suptitle("Comparaison multi-modèles de clustering (sur données PCA)",
                      color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 5 — COMPARATIF FINAL
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🧭 Tableau de Bord Comparatif Final")

    try:
        rf_final = accuracy_score(y_test_c, test_pred_c) * 100
    except Exception:
        rf_final = test_acc_c

    try:
        sil_f = sil; ari_f = ari; nmi_f = nmi; acc_f = acc_map_c
    except Exception:
        sil_f = ari_f = nmi_f = acc_f = 0.0

    st.markdown(f"""
    <table class='compare-table'>
        <thead>
            <tr>
                <th>Critère</th>
                <th>🎯 Classification (Random Forest)</th>
                <th>🔍 Clustering (K-Means + PCA)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Type d'apprentissage</td>
                <td><span class='badge badge-blue'>Supervisé</span></td>
                <td><span class='badge badge-purple'>Non Supervisé</span></td>
            </tr>
            <tr>
                <td>Labels requis ?</td>
                <td>✅ Oui — 50 000 étiquettes</td>
                <td>❌ Non — masqués totalement</td>
            </tr>
            <tr>
                <td>Algorithme</td>
                <td>Random Forest ({n_estimators} arbres, depth={max_depth})</td>
                <td>StandardScaler {"→ L2 " if use_l2 else ""}→ PCA({n_components_pca}D) → K-Means(K={n_clusters})</td>
            </tr>
            <tr>
                <td>Accuracy principale</td>
                <td><strong style='color:#4fc3f7; font-family:monospace;'>
                    {rf_final:.2f}%</strong></td>
                <td><strong style='color:#FFB703; font-family:monospace;'>
                    {acc_f:.2f}% (mapping hongrois)</strong></td>
            </tr>
            <tr>
                <td>Métriques additionnelles</td>
                <td>Matrice confusion, F1, Précision, Rappel, Feature Importance</td>
                <td>Silhouette : {sil_f:.4f} | ARI : {ari_f:.4f} | NMI : {nmi_f:.4f}</td>
            </tr>
            <tr>
                <td>Nouveauté vs v1</td>
                <td>Split fixe 50k/10k/10k (notebook)</td>
                <td>Mapping hongrois + pureté + multi-modèles</td>
            </tr>
            <tr>
                <td>Quand l'utiliser ?</td>
                <td>Données <strong>labellisées</strong></td>
                <td>Données <strong>inconnues</strong> / exploratoire</td>
            </tr>
            <tr>
                <td>Interprétabilité</td>
                <td><span class='badge badge-green'>Haute</span> — Feature Importance</td>
                <td><span class='badge badge-orange'>Moyenne</span> — t-SNE, pureté</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='golden-rule'>
        ⚖️ LA RÈGLE D'OR<br><br>
        🏷️ Labels disponibles &nbsp;→&nbsp;
        <span style='color:#4fc3f7'>CLASSIFICATION</span>
        &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        ❓ Données inconnues &nbsp;→&nbsp;
        <span style='color:#FFB703'>CLUSTERING</span>
        <br>
        <span style='font-size:0.8rem; color:#aa9933; font-weight:400;'>
            (puis éventuellement Classification une fois les groupes identifiés)
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#0a1628,#1a2a4e);
                    border:1px solid #2a4a8e; border-radius:12px; padding:20px;'>
            <div style='font-family:Space Mono,monospace; font-size:1rem;
                        color:#4fc3f7; font-weight:700; margin-bottom:12px;'>
                🎯 CLASSIFICATION — RÉSUMÉ
            </div>
        """, unsafe_allow_html=True)
        st.metric("Accuracy Test",    f"{rf_final:.2f}%")
        st.metric("Algorithme",       "Random Forest")
        st.metric("Arbres",           f"{n_estimators}")
        st.metric("Profondeur max",   f"{max_depth}")
        st.metric("Split train",      "50 000 images")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a0a28,#2e1a4e);
                    border:1px solid #6a2a8e; border-radius:12px; padding:20px;'>
            <div style='font-family:Space Mono,monospace; font-size:1rem;
                        color:#b39ddb; font-weight:700; margin-bottom:12px;'>
                🔍 CLUSTERING — RÉSUMÉ
            </div>
        """, unsafe_allow_html=True)
        st.metric("Accuracy (mapping)", f"{acc_f:.2f}%")
        st.metric("Silhouette Score",   f"{sil_f:.4f}")
        st.metric("ARI",                f"{ari_f:.4f}")
        st.metric("NMI",                f"{nmi_f:.4f}")
        st.metric("Clusters K",         f"{n_clusters}")
        st.metric("Composantes PCA",    f"{n_components_pca}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("📝 Conclusion")
    st.markdown(f"""
    <div class='custom-box success'>
        <strong>📌 Synthèse :</strong><br><br>
        • <strong>Random Forest</strong> (50k labels) → <strong>{rf_final:.2f}% accuracy</strong>
          sur 10 000 images de test.<br>
        • <strong>K-Means+PCA</strong> (0 label, K={n_clusters}) → <strong>{acc_f:.2f}% accuracy</strong>
          après mapping hongrois optimal — avec un Silhouette de {sil_f:.4f}, ARI={ari_f:.4f}.<br>
        • L'écart de <strong>{rf_final - acc_f:.2f} points</strong> représente exactement
          le <em>coût de l'absence de supervision</em> : ce que l'on perd en ne fournissant
          aucun label au modèle.<br><br>
        • La projection <strong>t-SNE</strong> confirme visuellement que les deux approches
          identifient les mêmes structures latentes dans les données.
    </div>
    <div class='custom-box' style='border-left-color:#b39ddb; margin-top:16px;'>
        💡 <strong>Ces deux approches sont complémentaires, pas concurrentes.</strong><br>
        Le clustering explore ; la classification exploite.
        Utilisez d'abord le clustering pour comprendre vos données,
        puis la classification quand les labels sont disponibles.
    </div>
    """, unsafe_allow_html=True)
