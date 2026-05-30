# app.py
"""
Phase 5 — Démonstrateur Applicatif
Système d'aide au tri radiologique — ChestMNIST+
Lancement : streamlit run app.py
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration de la page (doit être le premier appel Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RadioScan AI — Aide au tri radiologique",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CHEST_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia",
]
N_CLASSES = len(CHEST_CLASSES)

# Seuil P95 d'anomalie (calibré à l'entraînement — à ajuster)
ANOMALY_THRESHOLD_P95 = 0.012
ANOMALY_THRESHOLD_P99 = 0.025

CHECKPOINT_DIR = Path("checkpoints")
DEVICE = torch.device("cpu")  # Interface toujours sur CPU pour la démo

# ---------------------------------------------------------------------------
# Styles CSS personnalisés
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .pathology-high {
        background-color: #fdecea;
        border-left: 4px solid #e53935;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 3px 0;
        font-weight: 600;
        color: #b71c1c;
    }
    .pathology-mid {
        background-color: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 3px 0;
        font-weight: 500;
        color: #f57f17;
    }
    .pathology-low {
        background-color: #f1f8e9;
        border-left: 4px solid #7cb342;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 3px 0;
        color: #33691e;
    }
    .anomaly-alert {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .anomaly-ok {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #856404;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e3a5f;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# SECTION MOCK / CHARGEMENT RÉEL DES MODÈLES
# Remplacer les fonctions mock par les imports réels de vos scripts :
#   from train_models import SimpleCNN, ResNetTL, ViTHybrid, build_model
#   from anomaly_detector import ConvAutoencoder
#   from multimodal_poc import MultimodalFusionModel
# ===========================================================================

class _MockClassifier(nn.Module):
    """Modèle de classification simulé — remplacer par le vrai modèle."""
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        torch.manual_seed(42)
        self.fc = nn.Linear(1, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simule des logits réalistes selon l'image
        pixel_mean = x.mean().item()
        torch.manual_seed(int(pixel_mean * 10000) % 2**31)
        return torch.randn(1, N_CLASSES) * 2


class _MockAutoencoder(nn.Module):
    """Autoencodeur simulé — remplacer par ConvAutoencoder."""
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x) * 0.05
        return x + noise, torch.zeros(1)


class _MockMultimodal(nn.Module):
    """Modèle multimodal simulé — remplacer par MultimodalFusionModel."""
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(int(txt.sum().item() * 100) % 2**31)
        return torch.randn(1, self.n_classes) * 2.5


@st.cache_resource(show_spinner="Chargement des modèles…")
def load_models(architecture: str = "simple_cnn") -> dict:
    """Charge les vrais modèles depuis les checkpoints. Fallback aux mocks si absent."""
    # Vérifie si les checkpoints Phase 2 & 3 existent
    checkpoint_classifier = CHECKPOINT_DIR / f"best_{architecture}.pt"
    checkpoint_ae = CHECKPOINT_DIR / "best_autoencoder.pt"
    checkpoint_mm = CHECKPOINT_DIR / "best_Multimodal.pt"

    # Si les checkpoints supervisé/anomalie n'existent pas, fallback aux mocks
    if not checkpoint_classifier.exists() or not checkpoint_ae.exists():
        log.warning(
            "Checkpoints Phase 2/3 manquants (%s ou %s) — fallback aux mocks",
            checkpoint_classifier.exists(), checkpoint_ae.exists()
        )
        return {
            "classifier": _MockClassifier().eval(),
            "autoencoder": _MockAutoencoder().eval(),
            "multimodal": _MockMultimodal().eval(),
            "architecture": architecture,
            "is_mock": True,
        }

    try:
        from train_models import build_model, TrainConfig
        from anomaly_detector import ConvAutoencoder
        from multimodal_poc import MultimodalFusionModel
        import joblib

        # Classifieur supervisé
        cfg = TrainConfig(architecture=architecture, image_size=64)
        classifier = build_model(cfg)
        ckpt = torch.load(checkpoint_classifier, map_location=DEVICE, weights_only=True)
        classifier.load_state_dict(ckpt["model_state_dict"])
        classifier.eval()

        # Autoencodeur
        ae = ConvAutoencoder(latent_dim=128, image_size=64).to(DEVICE)
        ae_ckpt = torch.load(checkpoint_ae, map_location=DEVICE, weights_only=True)
        ae.load_state_dict(ae_ckpt["state_dict"])
        ae.eval()

        # Multimodal (optionnel — peut venir du POC Phase 4)
        multimodal = None
        if checkpoint_mm.exists():
            try:
                multimodal = MultimodalFusionModel(vocab_size=5000, n_classes=14).to(DEVICE)
                mm_ckpt = torch.load(checkpoint_mm, map_location=DEVICE, weights_only=True)
                multimodal.load_state_dict(mm_ckpt["state_dict"])
                multimodal.eval()
            except Exception as e:
                log.warning("Multimodal non chargé (%s) — mode mock", e)
                multimodal = _MockMultimodal().eval()
        else:
            multimodal = _MockMultimodal().eval()

        # TF-IDF (optionnel)
        tfidf = None
        try:
            tfidf = joblib.load(CHECKPOINT_DIR / "tfidf_vectorizer.pkl")
        except FileNotFoundError:
            pass

        return {
            "classifier": classifier,
            "autoencoder": ae,
            "multimodal": multimodal,
            "tfidf": tfidf,
            "architecture": architecture,
            "is_mock": False,
        }
    except Exception as e:
        log.warning("Impossible de charger modèles supervisé/anomalie (%s) — fallback aux mocks", e)
        return {
            "classifier": _MockClassifier().eval(),
            "autoencoder": _MockAutoencoder().eval(),
            "multimodal": _MockMultimodal().eval(),
            "architecture": architecture,
            "is_mock": True,
        }


def preprocess_image(pil_img: Image.Image, size: int = 64) -> torch.Tensor:
    """Prétraitement image → tenseur normalisé (1, 1, H, W)."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.498], std=[0.248]),
    ])
    return transform(pil_img).unsqueeze(0)


def predict_classification(
    models: dict,
    img_tensor: torch.Tensor,
) -> dict[str, float]:
    """
    Inférence supervisée — retourne les probabilités par pathologie.

    TODO : Pour utiliser le vrai modèle, cette fonction est déjà correcte
    tant que build_model() renvoie un modèle avec la même interface forward().
    """
    with torch.no_grad():
        logits = models["classifier"](img_tensor)
        probs  = torch.sigmoid(logits).squeeze().numpy()
    return {cls: float(prob) for cls, prob in zip(CHEST_CLASSES, probs)}


def predict_anomaly(
    models: dict,
    img_tensor: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """
    Score d'anomalie = MSE(original, reconstruit).
    Retourne (score, image_reconstruite).
    """
    with torch.no_grad():
        reconstructed, _ = models["autoencoder"](img_tensor)
        score = float(((img_tensor - reconstructed) ** 2).mean().item())
    return score, reconstructed.squeeze()


def encode_text_tfidf(text: str, vocab_size: int = 5000) -> torch.Tensor:
    """
    Encodage TF-IDF simplifié pour la démo.
    TODO : Charger le vrai TfidfVectorizer fité à l'entraînement :
        import joblib
        tfidf = joblib.load("checkpoints/tfidf_vectorizer.pkl")
        vec = tfidf.transform([text]).toarray().astype(np.float32)
        return torch.tensor(vec)
    """
    # Simulation : hash des mots → vecteur creux
    words = text.lower().split()
    vec = np.zeros(vocab_size, dtype=np.float32)
    for word in words:
        idx = hash(word) % vocab_size
        vec[idx] += 1.0 / (len(words) + 1e-6)
    return torch.tensor(vec).unsqueeze(0)


def predict_multimodal(
    models: dict,
    img_tensor: torch.Tensor,
    report_text: str,
) -> dict[str, float]:
    """Inférence multimodale (image + texte)."""
    txt_tensor = encode_text_tfidf(report_text)
    with torch.no_grad():
        logits = models["multimodal"](img_tensor, txt_tensor)
        probs  = torch.sigmoid(logits).squeeze().numpy()
    return {cls: float(prob) for cls, prob in zip(CHEST_CLASSES, probs)}


# ===========================================================================
# COMPOSANTS UI
# ===========================================================================

def render_sidebar() -> dict:
    """Sidebar de configuration — retourne les paramètres choisis."""
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/lungs.png",
        width=60,
    )
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    arch = st.sidebar.selectbox(
        "🧠 Architecture supervisée",
        options=["simple_cnn", "resnet", "vit"],
        format_func=lambda x: {
            "simple_cnn": "SimpleCNN (from scratch)",
            "resnet": "ResNet-50 (Transfer Learning)",
            "vit": "ViT-B/16 (Vision Transformer)",
        }[x],
        index=1,
    )

    threshold = st.sidebar.slider(
        "🎯 Seuil de détection (probabilité)",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Probabilité minimale pour signaler une pathologie",
    )

    anomaly_p = st.sidebar.selectbox(
        "⚠️ Seuil anomalie",
        options=["P95", "P99"],
        index=0,
        help="Percentile calibré sur le test set",
    )
    anomaly_threshold = ANOMALY_THRESHOLD_P95 if anomaly_p == "P95" else ANOMALY_THRESHOLD_P99

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Statut des modèles**")

    models = load_models(architecture=arch)
    if models.get("is_mock"):
        with st.sidebar.expander("⚠️ Mode démo — voir pourquoi"):
            st.write("Les vrais modèles n'ont pas pu être chargés. Causes possibles :")
            st.write("- Checkpoints manquants dans `checkpoints/`")
            st.write("- Import échoué (dépendances manquantes)")
            st.write("- Train/anomaly_detector/multimodal_poc modules non disponibles")
            st.code("ls checkpoints/*.pt", language="bash")
    else:
        st.sidebar.success(f"✅ Modèles réels chargés ({arch})")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='font-size:0.75rem; color:#6c757d;'>
    <b>RadioScan AI v1.0</b><br>
    Projet Deep Learning Médical<br>
    EFREI Paris — 2025/2026
    </div>
    """, unsafe_allow_html=True)

    return {
        "architecture": arch,
        "threshold": threshold,
        "anomaly_threshold": anomaly_threshold,
        "models": models,
    }


def render_pathology_results(
    probs: dict[str, float],
    threshold: float,
    title: str = "Prédictions",
    color: str = "#1e3a5f",
) -> None:
    """Affiche les probabilités de pathologies avec code couleur."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

    detected = {k: v for k, v in probs.items() if v >= threshold}
    n_detected = len(detected)

    if n_detected == 0:
        st.success(f"✅ Aucune pathologie détectée (seuil : {threshold:.0%})")
    else:
        st.error(f"🔴 {n_detected} pathologie(s) détectée(s) au seuil {threshold:.0%}")

    # Tri par probabilité décroissante
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    for cls, prob in sorted_probs:
        col_bar, col_val = st.columns([4, 1])
        with col_bar:
            if prob >= threshold:
                css_class = "pathology-high" if prob >= 0.7 else "pathology-mid"
                icon = "🔴" if prob >= 0.7 else "🟡"
            else:
                css_class = "pathology-low"
                icon = "🟢"

            st.markdown(
                f'<div class="{css_class}">{icon} {cls}</div>',
                unsafe_allow_html=True,
            )
        with col_val:
            st.markdown(f"<div style='padding-top:6px; font-weight:600;'>{prob:.1%}</div>",
                        unsafe_allow_html=True)
        st.progress(min(prob, 1.0))


def render_anomaly_gauge(score: float, threshold: float) -> None:
    """Affiche le score d'anomalie avec jauge et alerte conditionnelle."""
    st.markdown('<div class="section-title">🔍 Score d\'anomalie (Autoencodeur)</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score MSE", f"{score:.5f}", delta=None)
    with col2:
        st.metric("Seuil P95", f"{ANOMALY_THRESHOLD_P95:.5f}", delta=None)
    with col3:
        normalized = min(score / (ANOMALY_THRESHOLD_P99 * 2), 1.0)
        st.metric("Niveau", f"{normalized:.0%}", delta=None)

    # Barre de progression normalisée
    normalized_display = min(score / (ANOMALY_THRESHOLD_P99 * 1.5), 1.0)
    st.progress(normalized_display)

    # Alerte
    if score >= ANOMALY_THRESHOLD_P99:
        st.markdown(
            '<div class="anomaly-alert">🚨 ANOMALIE CRITIQUE — Score très élevé (>P99)<br>'
            'Image fortement hors distribution. Révision urgente recommandée.</div>',
            unsafe_allow_html=True,
        )
    elif score >= ANOMALY_THRESHOLD_P95:
        st.warning(
            f"⚠️ **Anomalie détectée** — Score ({score:.5f}) dépasse le seuil P95 ({threshold:.5f}). "
            "Cette radiographie présente des caractéristiques inhabituelles."
        )
    else:
        st.markdown(
            '<div class="anomaly-ok">✅ Image dans la distribution normale<br>'
            f'Score MSE = {score:.5f} (inférieur au seuil P95)</div>',
            unsafe_allow_html=True,
        )


def render_reconstruction_comparison(
    original: Image.Image,
    reconstructed_tensor: torch.Tensor,
) -> None:
    """Affiche l'image originale vs reconstruite côte à côte."""
    with st.expander("🔬 Visualiser la reconstruction (Autoencodeur)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original, caption="Image originale", use_container_width=True)
        with col2:
            recon_np = reconstructed_tensor.numpy()
            recon_np = (recon_np - recon_np.min()) / (recon_np.max() - recon_np.min() + 1e-8)
            recon_img = Image.fromarray((recon_np * 255).astype(np.uint8))
            st.image(recon_img, caption="Reconstruction AE", use_container_width=True)
        with col3:
            diff = np.abs(
                np.array(original.resize((64, 64)).convert("L")).astype(float) / 255.0
                - recon_np
            )
            diff_norm = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
            diff_img = Image.fromarray(diff_norm)
            st.image(diff_img, caption="Carte d'erreur (|orig - recon|)",
                     use_container_width=True)


def render_multimodal_comparison(
    probs_img: dict[str, float],
    probs_mm: dict[str, float],
    threshold: float,
) -> None:
    """Tableau comparatif Image seule vs Multimodal."""
    st.markdown('<div class="section-title">🔀 Comparaison Image seule vs Multimodal</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        render_pathology_results(probs_img, threshold, title="📷 Image seule (CNN)")
    with col2:
        render_pathology_results(probs_mm, threshold, title="📷+📝 Multimodal (Image + Texte)")

    # Delta le plus significatif
    deltas = {
        cls: abs(probs_mm.get(cls, 0) - probs_img.get(cls, 0))
        for cls in CHEST_CLASSES
    }
    top_cls = max(deltas, key=deltas.get)
    top_delta = deltas[top_cls]

    if top_delta > 0.05:
        direction = "↑" if probs_mm[top_cls] > probs_img[top_cls] else "↓"
        st.info(
            f"💡 **Impact du texte :** La pathologie **{top_cls}** voit sa probabilité "
            f"{direction} de {top_delta:.1%} grâce au compte-rendu radiologique."
        )


# ===========================================================================
# APPLICATION PRINCIPALE
# ===========================================================================

def main() -> None:
    # --- Header ---
    st.markdown('<div class="main-header">🫁 RadioScan AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Système d\'aide au tri radiologique — '
        'Classification multi-label & Détection d\'anomalies</div>',
        unsafe_allow_html=True,
    )

    # Disclaimer médical
    st.markdown("""
    <div class="disclaimer">
    ⚕️ <b>Avertissement médical :</b> Cet outil est un démonstrateur académique.
    Il ne constitue pas un dispositif médical certifié et ne doit pas être utilisé
    pour des décisions cliniques réelles. Tout résultat doit être validé par un
    radiologue qualifié.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Sidebar ---
    config = render_sidebar()
    models    = config["models"]
    threshold = config["threshold"]
    anomaly_threshold = config["anomaly_threshold"]

    # --- Layout principal ---
    left_col, right_col = st.columns([1, 2], gap="large")

    # ============================
    # COLONNE GAUCHE — Inputs
    # ============================
    with left_col:
        st.markdown("### 📤 Charger une radiographie")
        uploaded_file = st.file_uploader(
            "Formats acceptés : JPG, PNG, DICOM (converti)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            pil_img = Image.open(uploaded_file).convert("RGB")
            st.image(pil_img, caption=f"Radiographie : {uploaded_file.name}",
                     use_container_width=True)
            st.caption(f"Dimensions : {pil_img.size[0]}×{pil_img.size[1]} px")

        st.markdown("---")
        st.markdown("### 📝 Compte-rendu radiologique (optionnel)")
        report_text = st.text_area(
            "Coller ou saisir le texte du rapport",
            placeholder=(
                "Ex: Bilateral lungs are clear. No pleural effusion. "
                "Heart size is within normal limits. No pneumothorax identified."
            ),
            height=150,
            label_visibility="collapsed",
        )
        if report_text:
            word_count = len(report_text.split())
            st.caption(f"📊 {word_count} mots saisis")

        analyze_btn = st.button(
            "🔬 Analyser la radiographie",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

        if not uploaded_file:
            st.info("👆 Chargez une radiographie pour commencer l'analyse.")
            # Image de démonstration — charge une vraie image OpenI
            with st.expander("💡 Utiliser une image de démonstration"):
                demo_img_path = None
                # Cherche une vraie image OpenI
                openi_dir = Path("data/openi/openi_images")
                if openi_dir.exists():
                    images = list(openi_dir.glob("*.png"))
                    if images:
                        demo_img_path = images[0]
                        st.write(f"📁 Image : `{demo_img_path.name}`")

                if demo_img_path:
                    # Affiche et propose de télécharger la vraie image
                    demo_img = Image.open(demo_img_path)
                    st.image(demo_img, caption="Image OpenI réelle", use_container_width=True)
                    with open(demo_img_path, "rb") as f:
                        st.download_button(
                            "⬇️ Télécharger image OpenI réelle",
                            data=f.read(),
                            file_name=demo_img_path.name,
                            mime="image/png",
                        )
                else:
                    # Fallback : image synthétique si pas d'OpenI
                    st.warning("❌ Pas de dataset OpenI trouvé. Télécharge une image manuelle.")
                    st.text("Lance d'abord : python download_openi.py --skip_images")

    # ============================
    # COLONNE DROITE — Résultats
    # ============================
    with right_col:
        if uploaded_file and analyze_btn:
            pil_img = Image.open(uploaded_file)

            with st.spinner("Analyse en cours… ⏳"):
                t0 = time.time()

                # Prétraitement
                img_tensor = preprocess_image(pil_img, size=64)

                # Inférence classification supervisée
                probs_img = predict_classification(models, img_tensor)

                # Inférence anomalie
                anomaly_score, recon_tensor = predict_anomaly(models, img_tensor)

                # Inférence multimodale (si texte fourni)
                probs_mm = None
                if report_text.strip():
                    probs_mm = predict_multimodal(models, img_tensor, report_text)

                elapsed = time.time() - t0

            st.success(f"✅ Analyse complète en **{elapsed:.2f}s** | Modèle : `{config['architecture']}`")

            # --- Onglets de résultats ---
            tab1, tab2, tab3 = st.tabs([
                "🏥 Classification supervisée",
                "🔍 Détection d'anomalies",
                "🔀 Analyse multimodale",
            ])

            with tab1:
                # Métriques résumé
                detected = {k: v for k, v in probs_img.items() if v >= threshold}
                m1, m2, m3 = st.columns(3)
                m1.metric("Pathologies détectées", len(detected))
                m2.metric("Score max", f"{max(probs_img.values()):.1%}")
                m3.metric("Seuil utilisé", f"{threshold:.0%}")

                render_pathology_results(probs_img, threshold,
                                         title="Probabilités par pathologie")

                with st.expander("📋 Données brutes (JSON)", expanded=False):
                    st.json({k: round(v, 4) for k, v in
                             sorted(probs_img.items(), key=lambda x: x[1], reverse=True)})

            with tab2:
                render_anomaly_gauge(anomaly_score, anomaly_threshold)
                st.markdown("")
                render_reconstruction_comparison(
                    pil_img.convert("L"),
                    recon_tensor.detach(),
                )

                with st.expander("ℹ️ Comment interpréter le score d'anomalie ?"):
                    st.markdown("""
                    Le score d'anomalie est la **MSE (Mean Squared Error)** entre l'image
                    originale et sa reconstruction par l'Autoencodeur convolutif.

                    - **Score faible** (< P95) : l'image ressemble aux radiographies
                      d'entraînement → cas probable dans la distribution
                    - **Score entre P95 et P99** : image atypique, mérite attention
                    - **Score > P99** : image très hors distribution, priorité à la révision

                    *Les seuils P95 et P99 ont été calibrés sur le test set de ChestMNIST+.*
                    """)

            with tab3:
                if probs_mm is not None:
                    render_multimodal_comparison(probs_img, probs_mm, threshold)

                    with st.expander("📋 Données brutes multimodal (JSON)", expanded=False):
                        comparison = {
                            cls: {
                                "image_only": round(probs_img[cls], 4),
                                "multimodal": round(probs_mm[cls], 4),
                                "delta": round(probs_mm[cls] - probs_img[cls], 4),
                            }
                            for cls in CHEST_CLASSES
                        }
                        st.json(comparison)
                else:
                    st.info(
                        "💬 **Aucun compte-rendu fourni.**\n\n"
                        "Renseignez un rapport radiologique dans le champ de gauche "
                        "pour activer la prédiction multimodale et la comparer "
                        "à la classification image seule."
                    )
                    st.markdown("""
                    **Exemple de compte-rendu à coller :**
                    ```
                    Bilateral lungs demonstrate clear lung fields without
                    consolidation, pleural effusion, or pneumothorax.
                    Cardiac silhouette is within normal limits.
                    No acute cardiopulmonary process.
                    ```
                    """)

        elif uploaded_file and not analyze_btn:
            st.info("👈 Cliquez sur **Analyser la radiographie** pour lancer l'inférence.")

        else:
            # Placeholder avec schéma du pipeline
            st.markdown("### 🏗️ Architecture du système")
            st.markdown("""
            ```
            📷 Radiographie
                    │
                    ├──► 🧠 CNN Supervisé ──► 14 probabilités de pathologies
                    │
                    ├──► 🔄 Autoencodeur ──► Score d'anomalie (MSE)
                    │
            📝 Rapport (optionnel)
                    │
                    └──► 🔀 Fusion Multimodale ──► Prédictions enrichies
            ```
            """)

            st.markdown("### 📈 Modèles disponibles")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.info("**SimpleCNN**\nEntraîné from scratch\n4 blocs Conv-BN-ReLU")
            with col_b:
                st.success("**ResNet-50**\nTransfer Learning\nImageNet pretrained")
            with col_c:
                st.warning("**ViT-B/16**\nVision Transformer\nImageNet pretrained")

    # --- Footer ---
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#6c757d; font-size:0.8rem;'>
    RadioScan AI • Projet Deep Learning Médical • EFREI Paris 2025/2026 •
    Données : ChestMNIST+ (MedMNIST v3) & OpenI
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
