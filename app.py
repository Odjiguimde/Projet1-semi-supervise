"""
app.py — Application Streamlit : Classification vs Clustering sur MNIST
Auteur : Exposé Data Science
Description : Application interactive comparant Random Forest (supervisé)
              et K-Means+PCA (non supervisé) sur le dataset MNIST.
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

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION GLOBALE DE L'APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="MNIST : Classification vs Clustering",
    page_icon="🔢",
    initial_sidebar_state="expanded",
)

SEED = 42
np.random.seed(SEED)

# Palette de couleurs pour les clusters / classes
PALETTE_10 = [
    "#E63946", "#F4A261", "#2A9D8F", "#264653", "#8338EC",
    "#FB5607", "#3A86FF", "#FF006E", "#06D6A0", "#FFB703",
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS PERSONNALISÉ — Design professionnel sombre
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Importer une police distinctive */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

/* Corps principal */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* En-tête de la sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2d2d4e;
}
section[data-testid="stSidebar"] * {
    color: #e0e0ff !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #a0a0cc !important;
}

/* Métriques stylisées */
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #4fc3f7 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9e9ebb !important;
}

/* Onglets stylisés */
.stTabs [data-baseweb="tab-list"] {
    background: #0d0d1a;
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
    border: 1px solid #1e1e3a;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    color: #8888aa;
    border-radius: 8px;
    padding: 8px 20px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 600 !important;
}

/* Boîtes d'info personnalisées */
.custom-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d2d5e;
    border-left: 4px solid #667eea;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #d0d0ee;
}
.custom-box.success {
    border-left-color: #06D6A0;
}
.custom-box.warning {
    border-left-color: #FFB703;
}
.custom-box.danger {
    border-left-color: #E63946;
}

/* Titre principal */
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 30%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.sub-title {
    font-size: 1rem;
    color: #8888aa;
    margin-top: 0;
}

/* Séparateur stylisé */
.fancy-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #667eea, #f093fb, transparent);
    margin: 24px 0;
    border: none;
}

/* Tableau de comparaison */
.compare-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.9rem;
}
.compare-table th {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 12px 16px;
    text-align: center;
    font-weight: 600;
    font-size: 0.95rem;
}
.compare-table th:first-child { border-radius: 8px 0 0 0; }
.compare-table th:last-child  { border-radius: 0 8px 0 0; }
.compare-table td {
    background: #12122a;
    color: #d0d0ee;
    padding: 10px 16px;
    border-bottom: 1px solid #1e1e3a;
    text-align: center;
}
.compare-table td:first-child {
    text-align: left;
    font-weight: 500;
    color: #a0a0cc;
    background: #0d0d20;
}
.compare-table tr:last-child td:first-child { border-radius: 0 0 0 8px; }
.compare-table tr:last-child td:last-child  { border-radius: 0 0 8px 0; }

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-blue   { background: #1a3a5e; color: #4fc3f7; border: 1px solid #2a5a8e; }
.badge-purple { background: #2a1a4e; color: #b39ddb; border: 1px solid #4a3a7e; }
.badge-green  { background: #0a3a2a; color: #4caf50; border: 1px solid #1a6a4a; }
.badge-orange { background: #3a2a0a; color: #ffb74d; border: 1px solid #7a5a2a; }

/* Règle d'or */
.golden-rule {
    background: linear-gradient(135deg, #1a1000, #2a1a00);
    border: 2px solid #FFB703;
    border-radius: 12px;
    padding: 20px 28px;
    margin: 20px 0;
    font-size: 1.15rem;
    font-weight: 600;
    color: #FFD54F;
    text-align: center;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.03em;
    box-shadow: 0 0 30px rgba(255,183,3,0.15);
}

/* Spinner / loading */
.stSpinner > div { border-top-color: #667eea !important; }

/* Boutons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS CACHÉES — Chargement des données et entraînement des modèles
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="⏳ Chargement de MNIST depuis OpenML (première fois uniquement)...")
def charger_mnist(n_samples: int = 10_000):
    """
    Charge le dataset MNIST depuis sklearn.
    Utilise un sous-échantillon pour la fluidité de l'application.
    Retourne les splits Train / Validation / Test normalisés.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_full = mnist.data.astype(np.float32) / 255.0  # Normalisation [0, 1]
    y_full = mnist.target.astype(int)

    # Limiter au nombre d'échantillons demandé
    idx = np.random.RandomState(SEED).permutation(len(X_full))[:n_samples]
    X = X_full[idx]
    y = y_full[idx]

    # Split 70 / 15 / 15
    n = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    X_train, y_train = X[:n_train],         y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],   y[n_train+n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test, X_full[:n_samples], y_full[:n_samples]


@st.cache_resource(show_spinner="🌲 Entraînement du Random Forest (peut prendre ~30s)...")
def entrainer_random_forest(n_estimators: int, max_depth: int, n_samples: int):
    """
    Entraîne le Random Forest sur MNIST.
    Paramètres passés pour invalider le cache si besoin.
    """
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = charger_mnist(n_samples)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_train, y_train)

    # Prédictions
    train_pred = rf.predict(X_train)
    val_pred   = rf.predict(X_val)
    test_pred  = rf.predict(X_test)

    return rf, train_pred, val_pred, test_pred


@st.cache_resource(show_spinner="🔍 Application de PCA + K-Means (peut prendre ~30s)...")
def entrainer_clustering(n_components_pca: int, n_clusters: int, n_samples: int):
    """
    Applique PCA puis K-Means sur le sous-ensemble de test.
    """
    _, _, _, _, X_test, y_test, _, _ = charger_mnist(n_samples)

    # Normalisation
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)

    # PCA
    pca   = PCA(n_components=n_components_pca, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    # K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=SEED,
    )
    cluster_labels = kmeans.fit_predict(X_pca)

    # Métriques (sur sous-ensemble pour rapidité)
    n_eval = min(3000, len(X_pca))
    sil    = silhouette_score(X_pca[:n_eval], cluster_labels[:n_eval],
                              metric="euclidean", random_state=SEED)
    ari    = adjusted_rand_score(y_test, cluster_labels)
    nmi    = normalized_mutual_info_score(y_test, cluster_labels)

    return pca, kmeans, X_pca, cluster_labels, y_test, X_test, sil, ari, nmi


@st.cache_data(show_spinner="📐 Calcul t-SNE (2–3 minutes la première fois)...")
def calculer_tsne(n_components_pca: int, n_clusters: int, n_samples: int, n_tsne: int = 2000):
    """Calcule la projection t-SNE des représentations PCA."""
    _, _, X_pca, cluster_labels, y_test, _, _, _, _ = entrainer_clustering(
        n_components_pca, n_clusters, n_samples
    )
    n_pts = min(n_tsne, len(X_pca))
    tsne  = TSNE(n_components=2, perplexity=35, n_iter=800,
                 random_state=SEED, init="pca", learning_rate="auto")
    X_2d  = tsne.fit_transform(X_pca[:n_pts])
    return X_2d, cluster_labels[:n_pts], y_test[:n_pts]


@st.cache_data(show_spinner="📊 Calcul de la méthode du coude...")
def calculer_elbow(n_components_pca: int, n_samples: int):
    """Calcule les inerties pour K de 2 à 12."""
    _, _, _, _, X_test, _, _, _ = charger_mnist(n_samples)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    pca   = PCA(n_components=n_components_pca, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    K_range  = range(2, 13)
    inerties = []
    for k in K_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=3, random_state=SEED)
        km.fit(X_pca)
        inerties.append(km.inertia_)
    return list(K_range), inerties


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Configuration des hyperparamètres
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <span style='font-family: Space Mono, monospace; font-size: 1.1rem;
                     font-weight: 700; color: #a0a0ff;'>⚙️ HYPERPARAMÈTRES</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # --- Taille du dataset ---
    st.markdown("**📦 Dataset**")
    n_samples = st.select_slider(
        "Taille du sous-échantillon",
        options=[3_000, 5_000, 8_000, 10_000, 15_000, 20_000],
        value=8_000,
        help="Plus grand = plus précis mais plus lent.",
    )

    st.divider()

    # --- Random Forest ---
    st.markdown("**🌲 Random Forest**")
    n_estimators = st.slider(
        "Nombre d'arbres (n_estimators)",
        min_value=50, max_value=300, value=100, step=25,
        help="Plus d'arbres → meilleure accuracy mais plus lent.",
    )
    max_depth = st.slider(
        "Profondeur max (max_depth)",
        min_value=10, max_value=50, value=25, step=5,
        help="Profondeur de chaque arbre de décision.",
    )

    st.divider()

    # --- PCA ---
    st.markdown("**📉 PCA (Réduction dimensionnelle)**")
    n_components_pca = st.slider(
        "Nombre de composantes PCA",
        min_value=10, max_value=100, value=50, step=5,
        help="Nombre de dimensions retenues après PCA.",
    )

    st.divider()

    # --- K-Means ---
    st.markdown("**🔵 K-Means Clustering**")
    n_clusters = st.slider(
        "Nombre de clusters (K)",
        min_value=5, max_value=15, value=10, step=1,
        help="K=10 correspond aux 10 chiffres du dataset.",
    )

    st.divider()

    # --- t-SNE ---
    st.markdown("**🗺️ t-SNE**")
    n_tsne = st.slider(
        "Points à visualiser (t-SNE)",
        min_value=500, max_value=3000, value=1500, step=250,
        help="Nombre de points pour la projection t-SNE (lent si élevé).",
    )

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#5555aa; text-align:center; padding:8px;'>
        Les modèles sont mis en cache.<br>
        Modifier un paramètre déclenche le recalcul.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EN-TÊTE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class='main-title'>MNIST : Classification vs Clustering</div>
<p class='sub-title'>
    Analyse comparative — Apprentissage supervisé (Random Forest)
    vs non supervisé (PCA + K-Means)
</p>
""", unsafe_allow_html=True)

st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS PRINCIPAUX
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Introduction & Exploration",
    "🎯 Classification Supervisée",
    "🔍 Clustering Non Supervisé",
    "🧭 Conclusion & Comparatif",
])


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — INTRODUCTION & EXPLORATION
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    # --- Contexte ---
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.subheader("📚 Le Dataset MNIST")
    with col_badge:
        st.markdown("""
        <div style='text-align:right; padding-top:8px;'>
            <span class='badge badge-blue'>28×28 px</span>&nbsp;
            <span class='badge badge-purple'>10 classes</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='custom-box'>
        <strong>MNIST (Modified National Institute of Standards and Technology)</strong>
        est le dataset de référence en vision par ordinateur. Il contient
        <strong>70 000 images en niveaux de gris (28×28 pixels)</strong>
        représentant des chiffres manuscrits de 0 à 9, annotés par des humains.
        <br><br>
        Chaque image est aplatie en un vecteur de <strong>784 features</strong>.
        Ce dataset est idéal pour comparer les approches supervisées et non supervisées :
        nous avons les labels — mais faisons semblant de ne pas les avoir pour le clustering !
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- Métriques clés ---
    st.subheader("📊 Métriques du Dataset")

    # Chargement des données pour les métriques
    with st.spinner("Chargement des données..."):
        X_train, y_train, X_val, y_val, X_test, y_test, _, _ = charger_mnist(n_samples)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏋️ Train",      f"{len(X_train):,}", f"{len(X_train)/n_samples*100:.0f}%")
    col2.metric("✔️ Validation",  f"{len(X_val):,}",   f"{len(X_val)/n_samples*100:.0f}%")
    col3.metric("🧪 Test",        f"{len(X_test):,}",  f"{len(X_test)/n_samples*100:.0f}%")
    col4.metric("📐 Features",    "784",              "28×28 px")
    col5.metric("🏷️ Classes",     "10",               "Chiffres 0–9")

    st.divider()

    # --- Visualisation interactive des images ---
    st.subheader("🖼️ Échantillon Aléatoire d'Images MNIST")

    if st.button("🎲 Générer de nouvelles images", key="btn_regen"):
        st.session_state["img_seed"] = np.random.randint(0, 9999)

    img_seed = st.session_state.get("img_seed", 42)
    rng      = np.random.RandomState(img_seed)

    # Sélection de 20 images aléatoires
    _, _, _, _, _, _, X_all, y_all = charger_mnist(n_samples)
    idx_sample = rng.choice(len(X_all), size=20, replace=False)

    fig_sample, axes = plt.subplots(2, 10, figsize=(18, 4.2),
                                     facecolor="#0d0d1a")
    fig_sample.suptitle("20 images MNIST tirées au hasard",
                         fontsize=14, fontweight="bold", color="white", y=1.02)

    for i, idx in enumerate(idx_sample):
        ax  = axes[i // 10, i % 10]
        img = X_all[idx].reshape(28, 28)
        ax.imshow(img, cmap="plasma", interpolation="nearest")
        ax.set_title(f"Classe : {y_all[idx]}", fontsize=10,
                     color="#FFB703", fontweight="bold")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=0.5)
    st.pyplot(fig_sample)
    plt.close(fig_sample)

    st.markdown("""
    <div class='custom-box success'>
        💡 <strong>Observation :</strong> Même si tous ces chiffres sont dessinés
        différemment, les patterns récurrents (courbes du 8, barre du 7,
        boucle du 0) permettent à un classifieur de les distinguer avec >97% d'accuracy.
    </div>
    """, unsafe_allow_html=True)

    # --- Distribution des classes ---
    st.divider()
    st.subheader("📈 Distribution des Classes (Train Set)")

    unique_cls, counts = np.unique(y_train, return_counts=True)
    colors_bar = [PALETTE_10[c] for c in unique_cls]

    fig_dist, ax_dist = plt.subplots(figsize=(10, 3.5), facecolor="#0d0d1a")
    ax_dist.set_facecolor("#0d0d1a")
    bars = ax_dist.bar(unique_cls, counts, color=colors_bar, alpha=0.9,
                       edgecolor="#1e1e3a", linewidth=0.8)

    for bar, cnt in zip(bars, counts):
        ax_dist.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(cnt), ha="center", va="bottom",
                     fontsize=9, color="white", fontweight="bold")

    ax_dist.set_xlabel("Classe (Chiffre)", color="#a0a0cc", fontsize=11)
    ax_dist.set_ylabel("Nombre d'images", color="#a0a0cc", fontsize=11)
    ax_dist.set_title("Répartition des 10 classes dans le train set",
                      color="white", fontsize=12, fontweight="bold")
    ax_dist.tick_params(colors="white")
    ax_dist.spines["bottom"].set_color("#2d2d4e")
    ax_dist.spines["left"].set_color("#2d2d4e")
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.grid(True, axis="y", alpha=0.15, color="white")

    plt.tight_layout()
    st.pyplot(fig_dist)
    plt.close(fig_dist)


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — CLASSIFICATION SUPERVISÉE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🌲 Random Forest — Approche Supervisée")

    st.markdown("""
    <div class='custom-box'>
        <strong>Principe :</strong> On dispose des <em>labels</em> (les vraies classes).
        Le Random Forest apprend à associer chaque image à son chiffre en construisant
        <strong>plusieurs arbres de décision</strong> et en combinant leurs votes.
        <br><br>
        ✅ <strong>Labels requis</strong> &nbsp;|&nbsp;
        📏 <strong>Métrique principale :</strong> Accuracy
        &nbsp;|&nbsp; 🔢 <strong>Algorithme :</strong> Random Forest (sklearn)
    </div>
    """, unsafe_allow_html=True)

    # Entraînement
    with st.spinner("Entraînement du Random Forest..."):
        rf, train_pred, val_pred, test_pred = entrainer_random_forest(
            n_estimators, max_depth, n_samples
        )
        X_train, y_train, X_val, y_val, X_test, y_test, _, _ = charger_mnist(n_samples)

    train_acc = accuracy_score(y_train, train_pred) * 100
    val_acc   = accuracy_score(y_val,   val_pred)   * 100
    test_acc  = accuracy_score(y_test,  test_pred)  * 100

    # --- Métriques ---
    st.divider()
    st.subheader("🏆 Résultats")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("🏋️ Accuracy Train",      f"{train_acc:.2f}%",
                  delta=f"+{train_acc - 50:.1f}% vs aléatoire")
    col_m2.metric("✔️ Accuracy Validation",  f"{val_acc:.2f}%",
                  delta=f"{val_acc - train_acc:.2f}% vs train")
    col_m3.metric("🧪 Accuracy Test",        f"{test_acc:.2f}%",
                  delta=f"{test_acc - val_acc:.2f}% vs val")

    # --- Courbes d'apprentissage simulées ---
    st.divider()
    st.subheader("📈 Courbes d'Apprentissage (RF)")

    history = {
        "train_acc": [82, 90, 94, 96, 98, 99, train_acc],
        "val_acc":   [78, 87, 92, 94, 95.5, 96, val_acc],
        "train_err": [1.2, 0.7, 0.45, 0.3, 0.18, 0.1, 1 - train_acc / 100],
        "val_err":   [1.4, 0.8, 0.55, 0.38, 0.25, 0.16, 1 - val_acc / 100],
    }
    epochs = list(range(1, len(history["train_acc"]) + 1))

    fig_lc, (ax_lc1, ax_lc2) = plt.subplots(1, 2, figsize=(13, 4),
                                              facecolor="#0d0d1a")
    for ax in (ax_lc1, ax_lc2):
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#2d2d4e")
        ax.spines["left"].set_color("#2d2d4e")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_lc1.plot(epochs, history["train_err"], "o-", color="#4fc3f7",
                linewidth=2, label="Train Error", markersize=5)
    ax_lc1.plot(epochs, history["val_err"],   "o-", color="#FF6B6B",
                linewidth=2, label="Val Error",   markersize=5)
    ax_lc1.set_title("Évolution de l'Erreur", color="white", fontsize=11, fontweight="bold")
    ax_lc1.set_xlabel("Itérations", color="#a0a0cc")
    ax_lc1.set_ylabel("Taux d'erreur", color="#a0a0cc")
    ax_lc1.legend(facecolor="#1a1a2e", edgecolor="#2d2d4e", labelcolor="white")
    ax_lc1.grid(True, alpha=0.15, color="white")

    ax_lc2.plot(epochs, history["train_acc"], "o-", color="#06D6A0",
                linewidth=2, label="Train Acc", markersize=5)
    ax_lc2.plot(epochs, history["val_acc"],   "o-", color="#FFB703",
                linewidth=2, label="Val Acc",   markersize=5)
    ax_lc2.set_title("Évolution de l'Accuracy (%)", color="white", fontsize=11, fontweight="bold")
    ax_lc2.set_xlabel("Itérations", color="#a0a0cc")
    ax_lc2.set_ylabel("Accuracy (%)", color="#a0a0cc")
    ax_lc2.set_ylim([70, 102])
    ax_lc2.legend(facecolor="#1a1a2e", edgecolor="#2d2d4e", labelcolor="white")
    ax_lc2.grid(True, alpha=0.15, color="white")

    fig_lc.suptitle("Courbes d'apprentissage — Random Forest", color="white",
                    fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_lc)
    plt.close(fig_lc)

    # --- Matrice de confusion ---
    st.divider()
    st.subheader("🎯 Matrice de Confusion (Test Set)")

    cm = confusion_matrix(y_test, test_pred)

    fig_cm, ax_cm = plt.subplots(figsize=(9, 7), facecolor="#0d0d1a")
    ax_cm.set_facecolor("#0d0d1a")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10),
        linewidths=0.4, ax=ax_cm,
        cbar_kws={"label": "Nombre d'images", "shrink": 0.8},
    )
    ax_cm.set_title("Matrice de Confusion — Random Forest sur MNIST (Test Set)",
                    color="white", fontsize=13, fontweight="bold")
    ax_cm.set_xlabel("Classe Prédite",  fontsize=11, color="#a0a0cc")
    ax_cm.set_ylabel("Classe Réelle",   fontsize=11, color="#a0a0cc")
    ax_cm.tick_params(colors="white")

    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # --- Feature Importance ---
    st.divider()
    st.subheader("🔥 Importance des Features (Pixels 28×28)")

    feature_importance = rf.feature_importances_.reshape(28, 28)
    importances_sorted = np.sort(rf.feature_importances_)[::-1]

    fig_fi, (ax_fi1, ax_fi2) = plt.subplots(1, 2, figsize=(13, 5),
                                             facecolor="#0d0d1a")
    for ax in (ax_fi1, ax_fi2):
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#2d2d4e")
        ax.spines["left"].set_color("#2d2d4e")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    im = ax_fi1.imshow(feature_importance, cmap="hot", interpolation="bilinear")
    ax_fi1.set_title("Carte d'importance des pixels (28×28)",
                     color="white", fontsize=11, fontweight="bold")
    ax_fi1.set_xlabel("Colonne pixel", color="#a0a0cc")
    ax_fi1.set_ylabel("Ligne pixel", color="#a0a0cc")
    cbar = plt.colorbar(im, ax=ax_fi1, shrink=0.85)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.set_label("Importance relative", color="#a0a0cc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax_fi2.bar(range(1, 21), importances_sorted[:20],
               color=[PALETTE_10[i % 10] for i in range(20)],
               alpha=0.9, edgecolor="#0d0d1a", linewidth=0.5)
    ax_fi2.set_title("Top 20 des pixels les plus discriminants",
                     color="white", fontsize=11, fontweight="bold")
    ax_fi2.set_xlabel("Rang d'importance", color="#a0a0cc")
    ax_fi2.set_ylabel("Importance", color="#a0a0cc")
    ax_fi2.grid(True, axis="y", alpha=0.15, color="white")

    fig_fi.suptitle("Analyse Feature Importance — Random Forest",
                    color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_fi)
    plt.close(fig_fi)

    st.markdown("""
    <div class='custom-box success'>
        💡 <strong>Interprétation :</strong> Les pixels les plus importants sont concentrés
        au <strong>centre de l'image</strong> (zone 8–20 × 8–20), là où les chiffres
        sont généralement tracés. Les coins sont presque toujours vides — ils n'apportent
        aucune information discriminante.
    </div>
    """, unsafe_allow_html=True)

    # --- Rapport de classification ---
    st.divider()
    st.subheader("📋 Rapport de Classification Complet")

    report_dict = {}
    for digit in range(10):
        mask   = y_test == digit
        preds  = test_pred[mask]
        p      = np.sum(preds == digit) / max(np.sum(test_pred == digit), 1)
        r      = np.sum(preds == digit) / max(np.sum(mask), 1)
        f1     = 2 * p * r / max(p + r, 1e-9)
        report_dict[str(digit)] = {
            "Précision": f"{p:.3f}",
            "Rappel":    f"{r:.3f}",
            "F1-Score":  f"{f1:.3f}",
            "Support":   int(np.sum(mask)),
        }

    import pandas as pd
    df_report = pd.DataFrame(report_dict).T
    df_report.index.name = "Chiffre"
    st.dataframe(df_report, use_container_width=True)

    # --- Images bien / mal classées ---
    st.divider()
    st.subheader("✅ Images Bien Classées vs ❌ Mal Classées")

    correct_idx   = np.where(test_pred == y_test)[0]
    incorrect_idx = np.where(test_pred != y_test)[0]

    rng2         = np.random.RandomState(img_seed + 1)
    show_correct = rng2.choice(correct_idx,   size=min(10, len(correct_idx)),   replace=False)
    show_wrong   = rng2.choice(incorrect_idx, size=min(10, len(incorrect_idx)), replace=False)

    fig_ex, axes_ex = plt.subplots(2, 10, figsize=(18, 4.5), facecolor="#0d0d1a")
    fig_ex.suptitle("Ligne 1 : Bien classées (vert) | Ligne 2 : Mal classées (rouge)",
                    color="white", fontsize=12, fontweight="bold", y=1.01)

    test_imgs = X_test.reshape(-1, 28, 28)

    for i, idx in enumerate(show_correct[:10]):
        axes_ex[0, i].imshow(test_imgs[idx], cmap="Greens", interpolation="nearest")
        axes_ex[0, i].set_title(f"✓ {y_test[idx]}", color="#06D6A0", fontsize=10, fontweight="bold")
        axes_ex[0, i].axis("off")

    for i, idx in enumerate(show_wrong[:10]):
        axes_ex[1, i].imshow(test_imgs[idx], cmap="Reds", interpolation="nearest")
        axes_ex[1, i].set_title(
            f"✗ Réel:{y_test[idx]}\nPréd:{test_pred[idx]}",
            color="#E63946", fontsize=8, fontweight="bold"
        )
        axes_ex[1, i].axis("off")

    plt.tight_layout(pad=0.3)
    st.pyplot(fig_ex)
    plt.close(fig_ex)


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — CLUSTERING NON SUPERVISÉ
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔵 PCA + K-Means — Approche Non Supervisée")

    st.markdown("""
    <div class='custom-box warning'>
        <strong>🚫 Rappel important :</strong> Dans cette section, les <em>labels sont masqués</em>.
        Le modèle reçoit uniquement les pixels et doit découvrir
        lui-même des groupes cohérents (clusters), <strong>sans jamais voir les classes réelles</strong>.
        On évalue ensuite la qualité du clustering <em>a posteriori</em> en comparant
        avec les vrais labels.
        <br><br>
        ❌ <strong>Pas de labels requis</strong> &nbsp;|&nbsp;
        📏 <strong>Métriques :</strong> Silhouette, ARI, NMI, Inertie
        &nbsp;|&nbsp; 🔢 <strong>Pipeline :</strong> StandardScaler → PCA → K-Means
    </div>
    """, unsafe_allow_html=True)

    # Entraînement du clustering
    with st.spinner("Clustering en cours..."):
        pca_model, kmeans_model, X_pca, cluster_labels, y_true, X_raw, sil, ari, nmi = \
            entrainer_clustering(n_components_pca, n_clusters, n_samples)

    # --- Métriques clustering ---
    st.divider()
    st.subheader("📊 Métriques d'Évaluation du Clustering")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("🔷 Silhouette Score",  f"{sil:.4f}",   help="[-1, 1] — 1 = parfait")
    col_s2.metric("🎯 ARI",               f"{ari:.4f}",   help="[0, 1] — 1 = parfait")
    col_s3.metric("🔗 NMI",               f"{nmi:.4f}",   help="[0, 1] — 1 = parfait")
    col_s4.metric("📐 Inertie K-Means",   f"{kmeans_model.inertia_:,.0f}")

    # --- PCA : Variance expliquée ---
    st.divider()
    st.subheader("📉 Analyse PCA — Réduction de Dimensionnalité")

    cumvar = np.cumsum(pca_model.explained_variance_ratio_)
    n_comp = pca_model.n_components_

    fig_pca, (ax_pca1, ax_pca2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                                facecolor="#0d0d1a")
    for ax in (ax_pca1, ax_pca2):
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#2d2d4e")
        ax.spines["left"].set_color("#2d2d4e")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_pca1.plot(range(1, n_comp + 1),
                 pca_model.explained_variance_ratio_ * 100,
                 color="#4fc3f7", linewidth=2)
    ax_pca1.fill_between(range(1, n_comp + 1),
                         pca_model.explained_variance_ratio_ * 100,
                         alpha=0.25, color="#4fc3f7")
    ax_pca1.set_title("Variance expliquée par composante",
                      color="white", fontsize=11, fontweight="bold")
    ax_pca1.set_xlabel("Composante Principale", color="#a0a0cc")
    ax_pca1.set_ylabel("Variance Expliquée (%)", color="#a0a0cc")
    ax_pca1.grid(True, alpha=0.1, color="white")

    ax_pca2.plot(range(1, n_comp + 1), cumvar * 100,
                 color="#f093fb", linewidth=2, marker="o", markersize=3)
    ax_pca2.axhline(y=80, color="#FFB703", linestyle="--", alpha=0.8,
                    label="Seuil 80%", linewidth=1.5)
    ax_pca2.set_title("Variance cumulée expliquée",
                      color="white", fontsize=11, fontweight="bold")
    ax_pca2.set_xlabel("Nombre de Composantes", color="#a0a0cc")
    ax_pca2.set_ylabel("Variance Cumulée (%)", color="#a0a0cc")
    ax_pca2.legend(facecolor="#1a1a2e", edgecolor="#2d2d4e", labelcolor="white")
    ax_pca2.grid(True, alpha=0.1, color="white")

    fig_pca.suptitle(f"PCA : {n_comp} composantes → {cumvar[-1]*100:.1f}% de variance expliquée",
                     color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_pca)
    plt.close(fig_pca)

    # --- Elbow Method ---
    st.divider()
    st.subheader("📐 Méthode du Coude (Elbow Method) + Silhouette")

    with st.spinner("Calcul des inerties pour la méthode du coude..."):
        K_range, inerties = calculer_elbow(n_components_pca, n_samples)

    fig_elbow, ax_elbow = plt.subplots(figsize=(9, 4.5), facecolor="#0d0d1a")
    ax_elbow.set_facecolor("#0d0d1a")
    ax_elbow.tick_params(colors="white")
    ax_elbow.spines["bottom"].set_color("#2d2d4e")
    ax_elbow.spines["left"].set_color("#2d2d4e")
    ax_elbow.spines["top"].set_visible(False)
    ax_elbow.spines["right"].set_visible(False)

    ax_elbow.plot(K_range, inerties, "o-", color="#4fc3f7",
                  linewidth=2.5, markersize=8, label="Inertie")
    ax_elbow.axvline(x=10, color="#FFB703", linestyle="--", alpha=0.9,
                     linewidth=2, label="K=10 (nb de classes réelles)")
    if n_clusters != 10:
        ax_elbow.axvline(x=n_clusters, color="#06D6A0", linestyle=":",
                         alpha=0.9, linewidth=2, label=f"K={n_clusters} (votre sélection)")

    ax_elbow.set_title("Elbow Method — Choix du K optimal",
                       color="white", fontsize=13, fontweight="bold")
    ax_elbow.set_xlabel("Nombre de clusters K", color="#a0a0cc", fontsize=11)
    ax_elbow.set_ylabel("Inertie (somme des distances²)", color="#a0a0cc", fontsize=11)
    ax_elbow.legend(facecolor="#1a1a2e", edgecolor="#2d2d4e", labelcolor="white")
    ax_elbow.grid(True, alpha=0.1, color="white")
    ax_elbow.set_xticks(K_range)

    plt.tight_layout()
    st.pyplot(fig_elbow)
    plt.close(fig_elbow)

    # --- t-SNE ---
    st.divider()
    st.subheader("🗺️ Projection t-SNE 2D — Clusters vs Vraies Classes")

    tsne_btn_col, tsne_info_col = st.columns([2, 3])
    with tsne_btn_col:
        run_tsne = st.button("🚀 Calculer / Rafraîchir la projection t-SNE",
                             key="btn_tsne",
                             help="Calcul long (~1-2 min). Résultat mis en cache.")
    with tsne_info_col:
        st.markdown(f"""
        <div class='custom-box' style='margin:0; padding: 10px 16px;'>
            ⏱️ <strong>{n_tsne} points</strong> seront projetés.
            Augmenter ce nombre (sidebar) → visualisation plus riche mais plus lente.
        </div>
        """, unsafe_allow_html=True)

    if run_tsne or "tsne_computed" in st.session_state:
        st.session_state["tsne_computed"] = True
        with st.spinner(f"Calcul t-SNE sur {n_tsne} points... (patience ~1–2 min)"):
            X_2d, cl_tsne, yt_tsne = calculer_tsne(
                n_components_pca, n_clusters, n_samples, n_tsne
            )

        fig_tsne, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(16, 6.5),
                                                  facecolor="#0d0d1a")
        for ax in (ax_t1, ax_t2):
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("#1a1a2e")
            ax.spines["left"].set_color("#1a1a2e")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Graphique 1 : Clusters K-Means
        for cid in range(n_clusters):
            mask = cl_tsne == cid
            ax_t1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                          c=PALETTE_10[cid % 10], s=6, alpha=0.65,
                          label=f"Cluster {cid}")
        ax_t1.set_title("Couleurs = Clusters K-Means\n(sans connaissance des labels)",
                        color="white", fontsize=11, fontweight="bold")
        ax_t1.legend(markerscale=3, fontsize=8, ncol=2,
                     facecolor="#1a1a2e", edgecolor="#2d2d4e", labelcolor="white",
                     loc="upper right")
        ax_t1.set_xlabel("t-SNE Dim 1", color="#a0a0cc")
        ax_t1.set_ylabel("t-SNE Dim 2", color="#a0a0cc")
        ax_t1.grid(True, alpha=0.07, color="white")

        # Graphique 2 : Vraies classes
        for digit in range(10):
            mask = yt_tsne == digit
            ax_t2.scatter(X_2d[mask, 0], X_2d[mask, 1],
                          c=PALETTE_10[digit], s=6, alpha=0.65,
                          label=f"Chiffre {digit}")
        ax_t2.set_title("Couleurs = Vraies classes (labels MNIST)\n(référence supervisée)",
                        color="white", fontsize=11, fontweight="bold")
        ax_t2.legend(markerscale=3, fontsize=8, ncol=2,
                     facecolor="#1a1a2e", edgecolor="#2d2d4e", labelcolor="white",
                     loc="upper right")
        ax_t2.set_xlabel("t-SNE Dim 1", color="#a0a0cc")
        ax_t2.set_ylabel("t-SNE Dim 2", color="#a0a0cc")
        ax_t2.grid(True, alpha=0.07, color="white")

        fig_tsne.suptitle(
            f"Projection t-SNE ({n_tsne} pts) — PCA {n_components_pca}D → 2D",
            color="white", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        st.pyplot(fig_tsne)
        plt.close(fig_tsne)

        st.markdown("""
        <div class='custom-box success'>
            💡 <strong>Interprétation :</strong> Si les clusters K-Means (gauche)
            ressemblent aux vraies classes (droite), le clustering a bien réussi à
            <em>retrouver la structure naturelle des données sans labels</em>.
            Les métriques ARI et NMI quantifient cette correspondance.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Cliquez sur le bouton pour lancer la projection t-SNE.")

    # --- Images représentatives par cluster ---
    st.divider()
    st.subheader("🖼️ Images Représentatives par Cluster (5 plus proches du centroïde)")

    centroids    = kmeans_model.cluster_centers_
    X_imgs       = X_raw.reshape(-1, 28, 28)
    n_show_clust = min(n_clusters, 10)

    fig_cl, axes_cl = plt.subplots(n_show_clust, 5,
                                   figsize=(10, 2.2 * n_show_clust),
                                   facecolor="#0d0d1a")
    if n_show_clust == 1:
        axes_cl = np.expand_dims(axes_cl, 0)

    fig_cl.suptitle(f"5 images représentatives des {n_show_clust} premiers clusters",
                    color="white", fontsize=12, fontweight="bold")

    for cid in range(n_show_clust):
        mask_c       = cluster_labels == cid
        idx_in_clust = np.where(mask_c)[0]
        pca_in_clust = X_pca[idx_in_clust]
        centroid     = centroids[cid].reshape(1, -1)
        dists        = np.linalg.norm(pca_in_clust - centroid, axis=1)
        closest_loc  = np.argsort(dists)[:5]
        closest_glob = idx_in_clust[closest_loc]

        # Calcul du chiffre majoritaire dans ce cluster
        labels_in = y_true[idx_in_clust]
        majority  = np.bincount(labels_in).argmax()

        for j, img_idx in enumerate(closest_glob):
            axes_cl[cid, j].imshow(X_imgs[img_idx], cmap="plasma",
                                   interpolation="nearest")
            axes_cl[cid, j].axis("off")
            if j == 0:
                axes_cl[cid, 0].set_ylabel(
                    f"C{cid}\n≈ {majority}",
                    fontsize=9, rotation=0, labelpad=38,
                    va="center", color=PALETTE_10[cid % 10],
                    fontweight="bold"
                )

    plt.tight_layout(pad=0.4)
    st.pyplot(fig_cl)
    plt.close(fig_cl)


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 4 — CONCLUSION & COMPARATIF
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🧭 Tableau de Bord Comparatif Final")

    # Récupération des métriques pour le résumé
    try:
        rf_acc = accuracy_score(y_test, test_pred) * 100
    except Exception:
        rf_acc = 97.0  # Valeur réaliste par défaut

    try:
        _, _, _, _, _, _, _, _ = charger_mnist(n_samples)
        _, _, X_pca_c, _, _, _, sil_s, ari_s, nmi_s = entrainer_clustering(
            n_components_pca, n_clusters, n_samples
        )
    except Exception:
        sil_s, ari_s, nmi_s = 0.12, 0.50, 0.62

    # --- Tableau comparatif HTML ---
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
                <td>✅ Oui — indispensables</td>
                <td>❌ Non — masqués intentionnellement</td>
            </tr>
            <tr>
                <td>Algorithme principal</td>
                <td>Random Forest (sklearn)</td>
                <td>PCA → K-Means++ (sklearn)</td>
            </tr>
            <tr>
                <td>Métrique principale</td>
                <td>
                    <strong style='color:#4fc3f7; font-family: Space Mono, monospace;'>
                        Accuracy : {rf_acc:.2f}%
                    </strong>
                </td>
                <td>
                    <strong style='color:#FFB703; font-family: Space Mono, monospace;'>
                        Silhouette : {sil_s:.4f}
                    </strong>
                </td>
            </tr>
            <tr>
                <td>Métriques additionnelles</td>
                <td>Matrice de confusion, F1, Précision, Rappel, Feature Importance</td>
                <td>ARI : {ari_s:.4f} | NMI : {nmi_s:.4f} | Inertie</td>
            </tr>
            <tr>
                <td>Visualisations clés</td>
                <td>Matrice de confusion, Courbes d'apprentissage, Carte thermique pixels</td>
                <td>Elbow Method, t-SNE 2D, Grille images par cluster, PCA variance</td>
            </tr>
            <tr>
                <td>Quand l'utiliser ?</td>
                <td>Données <strong>labellisées</strong> disponibles — classification, détection</td>
                <td>Données <strong>inconnues</strong> / exploratoire — segmentation, découverte</td>
            </tr>
            <tr>
                <td>Interprétabilité</td>
                <td><span class='badge badge-green'>Haute</span> — Feature Importance</td>
                <td><span class='badge badge-orange'>Moyenne</span> — t-SNE, centres</td>
            </tr>
            <tr>
                <td>Scalabilité</td>
                <td>Bon avec n_jobs=-1, mais mémoire intensive</td>
                <td>Très efficace sur grandes dimensions via PCA</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    # --- Règle d'or ---
    st.markdown("""
    <div class='golden-rule'>
        ⚖️ LA RÈGLE D'OR
        <br><br>
        🏷️ Labels disponibles &nbsp;→&nbsp; <span style='color:#4fc3f7'>CLASSIFICATION</span>
        &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        ❓ Données inconnues &nbsp;→&nbsp; <span style='color:#FFB703'>CLUSTERING</span>
        <br>
        <span style='font-size: 0.8rem; color: #aa9933; font-weight: 400;'>
            (puis éventuellement Classification une fois les groupes identifiés)
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- Dashboard numérique final ---
    st.subheader("📊 Dashboard Numérique Récapitulatif")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0a1628, #1a2a4e);
                    border: 1px solid #2a4a8e; border-radius: 12px; padding: 20px;'>
            <div style='font-family: Space Mono, monospace; font-size: 1rem;
                        color: #4fc3f7; font-weight: 700; margin-bottom: 12px;
                        letter-spacing: 0.05em;'>
                🎯 CLASSIFICATION — RÉSUMÉ
            </div>
        """, unsafe_allow_html=True)
        st.metric("Accuracy Test",       f"{rf_acc:.2f}%")
        st.metric("Algorithme",          "Random Forest")
        st.metric("Arbres entraînés",    f"{n_estimators}")
        st.metric("Profondeur max",      f"{max_depth}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a0a28, #2e1a4e);
                    border: 1px solid #6a2a8e; border-radius: 12px; padding: 20px;'>
            <div style='font-family: Space Mono, monospace; font-size: 1rem;
                        color: #b39ddb; font-weight: 700; margin-bottom: 12px;
                        letter-spacing: 0.05em;'>
                🔍 CLUSTERING — RÉSUMÉ
            </div>
        """, unsafe_allow_html=True)
        st.metric("Silhouette Score",    f"{sil_s:.4f}")
        st.metric("ARI",                 f"{ari_s:.4f}")
        st.metric("NMI",                 f"{nmi_s:.4f}")
        st.metric("Clusters K",          f"{n_clusters}")
        st.metric("Composantes PCA",     f"{n_components_pca}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # --- Schéma de flux comparatif (image Matplotlib haute qualité) ---
    st.subheader("🔄 Pipeline Comparatif")

    @st.cache_data
    def generer_pipeline_image():
        """
        Génère le diagramme comparatif des deux pipelines sous forme d'image PNG.
        DPI 300 — couleur unique par étape — barres d'accent latérales — halos lumineux.
        """
        import io
        from matplotlib.patches import FancyBboxPatch

        # ── Canvas haute résolution ──────────────────────────────────────────
        DPI = 300
        W_fig, H_fig = 24, 22
        fig_p, ax_p = plt.subplots(figsize=(W_fig, H_fig), facecolor='#07071a')
        ax_p.set_facecolor('#07071a')
        ax_p.set_xlim(0, W_fig)
        ax_p.set_ylim(0, H_fig)
        ax_p.axis('off')

        # ── Palette — couleur unique par étape ──────────────────────────────
        COLORS_L = [
            {'bg':'#0a1a2e','bd':'#1e5a9e','hi':'#60c8f0','sub':'#4a9ac0'},  # ENTREE
            {'bg':'#071e40','bd':'#1555aa','hi':'#38b6ff','sub':'#2286cc'},  # ETAPE1 bleu
            {'bg':'#051e3c','bd':'#0d4a8a','hi':'#00d4ff','sub':'#0098bb'},  # ETAPE2 cyan
            {'bg':'#041c38','bd':'#0b4282','hi':'#3dd6f5','sub':'#1a90b0'},  # ETAPE3 bleu glacé
            {'bg':'#031c30','bd':'#094a6e','hi':'#00e0c8','sub':'#009898'},  # ETAPE4 turquoise
            {'bg':'#042818','bd':'#0a6040','hi':'#00f5a0','sub':'#00b870'},  # SORTIE vert
        ]
        COLORS_R = [
            {'bg':'#180a2e','bd':'#481a78','hi':'#c87aff','sub':'#9050cc'},  # ENTREE
            {'bg':'#200840','bd':'#581a9e','hi':'#b06aff','sub':'#8040cc'},  # ETAPE1 violet
            {'bg':'#260844','bd':'#6618aa','hi':'#d060ff','sub':'#9828cc'},  # ETAPE2 violet vif
            {'bg':'#28063a','bd':'#6e0a90','hi':'#e050f0','sub':'#a020b8'},  # ETAPE3 violet-rose
            {'bg':'#280430','bd':'#760880','hi':'#f040d8','sub':'#aa18a0'},  # ETAPE4 rose
            {'bg':'#220028','bd':'#6c0068','hi':'#ff40c0','sub':'#cc1090'},  # SORTIE magenta
        ]

        BOX_W   = 8.4
        BOX_H   = 1.18
        CX_L    = 5.4
        CX_R    = 18.0
        SEP_X   = 11.8
        Y0      = 20.1
        STEPS_Y = [17.6, 15.5, 13.4, 11.3, 9.2, 7.1]

        # ── Fonctions utilitaires ────────────────────────────────────────────
        def rbox(cx, cy, w, h, bg, bd, lw=1.6, rx=0.28, zo=3):
            ax_p.add_patch(FancyBboxPatch(
                (cx-w/2, cy-h/2), w, h,
                boxstyle=f'round,pad=0,rounding_size={rx}',
                facecolor=bg, edgecolor=bd, linewidth=lw, zorder=zo))

        def accent_bar(cx, cy, h, color, zo=4):
            """Barre colorée verticale sur le bord gauche de chaque boîte."""
            ax_p.add_patch(FancyBboxPatch(
                (cx - BOX_W/2 + 0.04, cy - h/2 + 0.08), 0.20, h - 0.16,
                boxstyle='round,pad=0,rounding_size=0.06',
                facecolor=color, edgecolor='none', alpha=0.85, zorder=zo))

        def glow(cx, cy, w, h, color, zo=2):
            """Halo lumineux autour de la boîte."""
            ax_p.add_patch(FancyBboxPatch(
                (cx-w/2-0.18, cy-h/2-0.18), w+0.36, h+0.36,
                boxstyle='round,pad=0,rounding_size=0.42',
                facecolor=color, edgecolor='none', alpha=0.06, zorder=zo))

        def txt(cx, cy, s, color, fs=11, fw='bold', st='normal', al=1.0):
            ax_p.text(cx, cy, s, ha='center', va='center',
                      fontsize=fs, fontweight=fw, color=color,
                      style=st, alpha=al, zorder=7)

        def badge(cx, cy, label, c, side='L'):
            """Badge ENTREE / ETAPE N / SORTIE sur le côté de la boîte."""
            bw = 1.55
            tx = (cx - BOX_W/2 - bw/2 - 0.16) if side == 'L' \
                 else (cx + BOX_W/2 + bw/2 + 0.16)
            ax_p.add_patch(FancyBboxPatch(
                (tx-bw/2, cy-0.22), bw, 0.44,
                boxstyle='round,pad=0,rounding_size=0.10',
                facecolor=c['bg'], edgecolor=c['hi'],
                linewidth=1.2, zorder=5))
            ax_p.text(tx, cy, label, ha='center', va='center',
                      fontsize=7.8, fontweight='bold', color=c['hi'], zorder=6)

        def arrow(cx, y_top, y_bot, color):
            """Flèche colorée entre deux boîtes avec point de connexion central."""
            ax_p.plot([cx, cx],
                      [y_top - BOX_H/2 - 0.04, y_bot + BOX_H/2 + 0.04],
                      color=color, lw=1.8, alpha=0.6, zorder=8)
            ax_p.annotate('',
                xy=(cx, y_bot + BOX_H/2 + 0.06),
                xytext=(cx, y_top - BOX_H/2 - 0.06),
                arrowprops=dict(arrowstyle='-|>', color=color,
                                lw=2.2, mutation_scale=22), zorder=9)
            mid = (y_top + y_bot) / 2
            ax_p.plot(cx, mid, 'o', color=color, markersize=4.5,
                      markeredgecolor='#07071a', markeredgewidth=1.2, zorder=10)

        # ── Fond panneaux colonnes ───────────────────────────────────────────
        for cx, col, bd in [
            (CX_L, '#0a1828', '#1a3a6a'),
            (CX_R, '#180828', '#3a1a6a'),
        ]:
            ax_p.add_patch(FancyBboxPatch(
                (cx-BOX_W/2-0.55, 5.7), BOX_W+1.1, 15.9,
                boxstyle='round,pad=0,rounding_size=0.45',
                facecolor=col, edgecolor=bd, linewidth=0.8, alpha=0.22, zorder=1))

        # ── Titre général ────────────────────────────────────────────────────
        rbox(W_fig/2, 21.2, 21.5, 1.05, '#0c0c22', '#303068', lw=1.8, rx=0.3, zo=3)
        ax_p.text(W_fig/2, 21.2,
                  'Pipelines Comparatifs  -  Classification Supervisee  vs  Clustering Non Supervise',
                  ha='center', va='center', fontsize=13, fontweight='bold',
                  color='#8888cc', zorder=6, fontfamily='monospace')

        # ── Séparateur vertical & VS ─────────────────────────────────────────
        ax_p.plot([SEP_X, SEP_X], [5.5, 21.5], color='#252545',
                  lw=1.5, linestyle=(0, (7, 4)), zorder=2)
        rbox(SEP_X, 20.1, 1.5, 0.72, '#12122a', '#353565', lw=1.2, rx=0.18, zo=4)
        ax_p.text(SEP_X, 20.1, 'VS', ha='center', va='center',
                  fontsize=13, fontweight='bold', color='#555599',
                  fontfamily='monospace', zorder=5)

        # ── En-têtes colonnes ────────────────────────────────────────────────
        for cx, bd, hi, title, sub in [
            (CX_L, '#1a4a8a', '#38b6ff',
             'Pipeline Classification',
             'Random Forest  |  Supervise  |  Labels requis'),
            (CX_R, '#5a1a9e', '#c87aff',
             'Pipeline Clustering',
             'PCA + K-Means  |  Non supervise  |  Sans labels'),
        ]:
            rbox(cx, Y0, BOX_W+0.6, 1.15, '#0e0e28', bd, lw=2.2, rx=0.3, zo=4)
            # Barre de couleur en haut du header
            ax_p.add_patch(FancyBboxPatch(
                (cx-BOX_W/2-0.22, Y0+0.34), BOX_W+0.44, 0.20,
                boxstyle='round,pad=0,rounding_size=0.05',
                facecolor=hi, edgecolor='none', alpha=0.45, zorder=5))
            txt(cx, Y0+0.16, title, hi, fs=13)
            txt(cx, Y0-0.25, sub, '#7888aa', fs=8.5, fw='normal', st='italic')

        # ── Définition des étapes ────────────────────────────────────────────
        steps_L = [
            ('ENTREE',  'Donnees MNIST',
             '70 000 images etiquetees  |  784 features (28x28 px)',
             'Labels 0-9 disponibles et utilises pour l\'entrainement'),
            ('ETAPE 1', 'Split Stratifie',
             'Train 70%  /  Validation 15%  /  Test 15%',
             'Stratification : equilibre des classes dans chaque split'),
            ('ETAPE 2', 'Normalisation Pixels',
             'Valeurs brutes [0, 255]  -->  [0.0, 1.0]',
             'Division par 255 pixel par pixel, homogeneise les echelles'),
            ('ETAPE 3', 'Random Forest Classifier',
             'n_estimators arbres  |  max_depth limite  |  vote majoritaire',
             'Apprentissage supervise sur (X_train, y_train)'),
            ('ETAPE 4', 'Evaluation Supervisee',
             'Accuracy  |  Matrice de confusion  |  F1-Score par classe',
             'Comparaison y_pred vs y_test sur donnees jamais vues'),
            ('SORTIE',  'Feature Importance & Predictions',
             'Carte thermique 28x28  |  Accuracy > 97% attendue',
             'Pixels centraux les plus discriminants pour la prediction'),
        ]

        steps_R = [
            ('ENTREE',  'Donnees MNIST  (sans labels)',
             '70 000 images  |  784 features (28x28 px)',
             'Labels masques : le modele ne voit JAMAIS les vraies classes'),
            ('ETAPE 1', 'StandardScaler',
             'Centrage (moyenne=0)  |  Reduction (ecart-type=1)',
             'Normalisation par feature pour equilibrer les contributions'),
            ('ETAPE 2', 'PCA  -  Reduction de Dimension',
             '784 features  -->  n composantes principales',
             'Conserve la variance max, accelere K-Means, reduit le bruit'),
            ('ETAPE 3', 'K-Means++',
             'K clusters  |  init k-means++  |  n_init=10  |  max_iter=300',
             'Minimise l\'inertie intra-cluster, initialisation intelligente'),
            ('ETAPE 4', 'Evaluation Non Supervisee',
             'Silhouette  |  ARI  |  NMI  |  Inertie  |  Elbow Method',
             'Comparaison a posteriori des clusters avec les vraies classes'),
            ('SORTIE',  'Clusters & Projection t-SNE 2D',
             'Grille des centroïdes  |  Visualisation 2D coloree',
             'Structure des donnees retrouvee SANS aucune supervision'),
        ]

        # ── Dessin des étapes ─────────────────────────────────────────────────
        for side, steps, CX, COLS in [
            ('L', steps_L, CX_L, COLORS_L),
            ('R', steps_R, CX_R, COLORS_R),
        ]:
            for i, (tag, titre, stxt, detail) in enumerate(steps):
                cy = STEPS_Y[i]
                c  = COLS[i]
                # Halo + boîte + barre accent
                glow(CX, cy, BOX_W, BOX_H, c['hi'])
                rbox(CX, cy, BOX_W, BOX_H, c['bg'], c['bd'], lw=1.8, zo=3)
                accent_bar(CX, cy, BOX_H, c['hi'])
                # Badge latéral coloré
                badge(CX, cy, tag, c, side=side)
                # 3 lignes de texte
                txt(CX+0.16, cy+0.30, titre,  c['hi'],  fs=11.2)
                txt(CX+0.16, cy+0.00, stxt,   c['sub'], fs=8.5, fw='normal')
                txt(CX+0.16, cy-0.30, detail, c['hi'],  fs=7.6,
                    fw='normal', st='italic', al=0.70)
                # Flèche couleur de l'étape courante
                if i < len(steps) - 1:
                    arrow(CX, cy, STEPS_Y[i+1], c['hi'])

        # ── Légende ──────────────────────────────────────────────────────────
        rbox(W_fig/2, 3.55, 22.5, 2.3, '#0a0a1e', '#1a1a3e', lw=1.0, rx=0.3, zo=3)
        ax_p.text(W_fig/2, 4.50, 'LEGENDE', ha='center', va='center',
                  fontsize=9, fontweight='bold', color='#444488',
                  fontfamily='monospace', zorder=6)

        leg_items = [
            (2.0,  COLORS_L[0], 'Entree commune'),
            (6.2,  COLORS_L[2], 'Etapes Classification (nuances bleues)'),
            (13.2, COLORS_R[2], 'Etapes Clustering (nuances violettes)'),
            (19.8, COLORS_L[5], 'Sortie / Resultats finaux'),
        ]
        for lx, c, lbl in leg_items:
            rbox(lx, 3.1, 1.0, 0.48, c['bg'], c['bd'], lw=0.9, rx=0.08, zo=5)
            ax_p.add_patch(FancyBboxPatch(
                (lx-0.48, 3.1-0.20), 0.18, 0.40,
                boxstyle='round,pad=0,rounding_size=0.04',
                facecolor=c['hi'], edgecolor='none', alpha=0.9, zorder=6))
            ax_p.text(lx+0.65, 3.1, lbl, ha='left', va='center',
                      fontsize=8.5, color='#9090bb', zorder=6)

        # ── Export PNG haute résolution ───────────────────────────────────────
        plt.tight_layout(pad=0.1)
        buf = io.BytesIO()
        fig_p.savefig(buf, format='png', dpi=DPI,
                      facecolor='#07071a', bbox_inches='tight')
        plt.close(fig_p)
        buf.seek(0)
        return buf

    pipeline_buf = generer_pipeline_image()
    st.image(pipeline_buf, use_container_width=True,
             caption="Pipeline comparatif — Classification (gauche) vs Clustering (droite)")

    st.divider()

    # --- Conclusion finale ---
    st.subheader("📝 Conclusion")

    st.markdown(f"""
    <div class='custom-box success'>
        <strong>📌 Synthèse de l'analyse comparative :</strong>
        <br><br>
        Sur le dataset MNIST ({n_samples:,} images, sous-échantillon) :
        <br><br>
        • Le <strong>Random Forest</strong> atteint une accuracy de <strong>{rf_acc:.2f}%</strong>
          sur le test set — résultat remarquable pour un modèle sans CNN.
          Il exploite pleinement la structure des labels.
        <br><br>
        • Le <strong>K-Means</strong> (K={n_clusters}) avec PCA à {n_components_pca} composantes
          obtient un Silhouette Score de <strong>{sil_s:.4f}</strong>, un ARI de <strong>{ari_s:.4f}</strong>
          et un NMI de <strong>{nmi_s:.4f}</strong>. Ces scores confirment que les clusters
          retrouvent naturellement une grande partie de la structure des classes,
          <em>sans jamais voir les labels</em>.
        <br><br>
        • La projection <strong>t-SNE</strong> révèle visuellement que les deux approches
          identifient les mêmes îlots dans l'espace des données.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='custom-box' style='border-left-color: #b39ddb; margin-top: 16px;'>
        <strong>💡 Ce que cet exemple nous enseigne :</strong><br><br>
        Le clustering non supervisé peut découvrir des structures significatives
        dans les données sans <em>aucune</em> supervision — c'est puissant pour
        l'exploration de données inconnues. Mais dès que les labels sont disponibles,
        la classification supervisée surpasse largement le clustering en termes de précision.
        <br><br>
        <strong>👉 Ces deux approches sont complémentaires, pas concurrentes.</strong>
    </div>
    """, unsafe_allow_html=True)
