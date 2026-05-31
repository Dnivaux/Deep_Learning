"""
Phase 5 — Interface
Système d'aide au tri radiologique — ChestMNIST+
Lancement : streamlit run app.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RadioScan AI",
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

ANOMALY_THRESHOLD_P95 = 0.012
ANOMALY_THRESHOLD_P99 = 0.025

CHECKPOINT_DIR = Path("checkpoints")
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 1rem 0 1.25rem 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.25rem;
}
.app-logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; flex-shrink: 0;
}
.app-title {
    font-size: 1.5rem; font-weight: 700;
    color: #0f172a; letter-spacing: -0.4px; margin: 0;
}
.app-subtitle {
    font-size: 0.82rem; color: #64748b; margin: 2px 0 0 0;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em;
    color: #94a3b8; margin-bottom: 10px; margin-top: 4px;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #94a3b8;
    padding: 10px 16px; border-radius: 6px;
    font-size: 0.8rem; color: #64748b; line-height: 1.55;
}

/* ── Badges ── */
.badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 999px; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.02em;
}
.badge-demo { background:#fef3c7; color:#92400e; border:1px solid #fde68a; }
.badge-live { background:#dcfce7; color:#14532d; border:1px solid #bbf7d0; }

/* ── Carte architecture ── */
.arch-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 16px;
    height: 100%;
    position: relative;
    transition: box-shadow .15s;
}
.arch-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,.08); }
.arch-card-accent {
    width: 36px; height: 4px;
    border-radius: 2px;
    margin-bottom: 12px;
}
.arch-card-title {
    font-size: 0.95rem; font-weight: 700;
    color: #0f172a; margin: 0 0 4px 0;
}
.arch-card-sub {
    font-size: 0.75rem; color: #64748b;
    margin: 0 0 12px 0;
}
.arch-card-stat {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: #475569;
    padding: 5px 0; border-top: 1px solid #f1f5f9;
}
.arch-card-stat span { font-weight: 600; color: #0f172a; }

/* ── Carte pipeline ── */
.pipeline-step {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    display: flex; align-items: flex-start; gap: 12px;
}
.pipeline-icon {
    width: 36px; height: 36px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.pipeline-title {
    font-size: 0.85rem; font-weight: 600;
    color: #0f172a; margin: 0 0 2px 0;
}
.pipeline-desc {
    font-size: 0.77rem; color: #64748b; margin: 0; line-height: 1.4;
}

/* ── Carte stat dataset ── */
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.stat-value {
    font-size: 1.4rem; font-weight: 700;
    color: #0f172a; letter-spacing: -0.5px;
}
.stat-label {
    font-size: 0.75rem; color: #64748b;
    margin-top: 2px;
}
.stat-sub {
    font-size: 0.7rem; color: #94a3b8;
    margin-top: 1px;
}

/* ── Pathologie rows ── */
.path-row {
    display: flex; align-items: center;
    padding: 7px 12px; border-radius: 6px;
    margin: 4px 0; font-size: 0.87rem;
    border: 1px solid transparent;
}
.path-high { background:#fef2f2; border-color:#fca5a5; color:#7f1d1d; font-weight:600; }
.path-mid  { background:#fffbeb; border-color:#fcd34d; color:#78350f; font-weight:500; }
.path-low  { background:#f8fafc; border-color:#e2e8f0; color:#475569; }
.path-indicator {
    width: 8px; height: 8px; border-radius: 50%;
    margin-right: 10px; flex-shrink: 0;
}
.ind-high { background:#ef4444; }
.ind-mid  { background:#f59e0b; }
.ind-low  { background:#94a3b8; }

/* ── Anomalie ── */
.anomaly-critical {
    background:#fef2f2; border:1px solid #fca5a5;
    border-left:4px solid #dc2626; padding:14px 18px;
    border-radius:8px; color:#7f1d1d; font-weight:600; font-size:.95rem;
}
.anomaly-warn {
    background:#fffbeb; border:1px solid #fcd34d;
    border-left:4px solid #f59e0b; padding:14px 18px;
    border-radius:8px; color:#78350f; font-weight:500; font-size:.95rem;
}
.anomaly-ok {
    background:#f0fdf4; border:1px solid #bbf7d0;
    border-left:4px solid #22c55e; padding:14px 18px;
    border-radius:8px; color:#14532d; font-weight:500; font-size:.95rem;
}

/* ── Barre de progression custom ── */
.prog-track {
    background: #e2e8f0;
    border-radius: 4px;
    height: 4px;
    margin: 3px 0 10px 0;
    overflow: hidden;
}
.prog-fill {
    height: 4px;
    border-radius: 4px;
    transition: width .3s ease;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] { font-size:.85rem; font-weight:500; color:#64748b; }
.stTabs [aria-selected="true"] { color:#0f172a; font-weight:600; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

/* ── Metric card override ── */
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# MODÈLES
# ===========================================================================

class _MockClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        torch.manual_seed(42)
        self.fc = nn.Linear(1, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixel_mean = x.mean().item()
        torch.manual_seed(int(pixel_mean * 10000) % 2**31)
        return torch.randn(1, N_CLASSES) * 2


class _MockAutoencoder(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x + torch.randn_like(x) * 0.05, torch.zeros(1)


class _MockMultimodal(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(int(txt.sum().item() * 100) % 2**31)
        return torch.randn(1, self.n_classes) * 2.5


@st.cache_resource(show_spinner="Chargement des modèles…")
def load_models(architecture: str = "simple_cnn") -> dict:
    ck_clf = CHECKPOINT_DIR / f"best_{architecture}.pt"
    ck_ae  = CHECKPOINT_DIR / "best_autoencoder.pt"
    ck_mm  = CHECKPOINT_DIR / "best_Multimodal.pt"

    if not ck_clf.exists() or not ck_ae.exists():
        log.warning("Checkpoints Phase 2/3 manquants — mode démo")
        return {"classifier": _MockClassifier().eval(),
                "autoencoder": _MockAutoencoder().eval(),
                "multimodal":  _MockMultimodal().eval(),
                "architecture": architecture, "is_mock": True}

    try:
        from train_models import build_model, TrainConfig
        from anomaly_detector import ConvAutoencoder
        from multimodal_poc import MultimodalFusionModel
        import joblib

        cfg = TrainConfig(architecture=architecture, image_size=64)
        clf = build_model(cfg)
        clf.load_state_dict(torch.load(ck_clf, map_location=DEVICE,
                                       weights_only=True)["model_state_dict"])
        clf.eval()

        ae = ConvAutoencoder(latent_dim=128, image_size=64).to(DEVICE)
        ae.load_state_dict(torch.load(ck_ae, map_location=DEVICE,
                                      weights_only=True)["state_dict"])
        ae.eval()

        mm = _MockMultimodal().eval()
        if ck_mm.exists():
            try:
                mm = MultimodalFusionModel(vocab_size=5000, n_classes=14).to(DEVICE)
                mm.load_state_dict(torch.load(ck_mm, map_location=DEVICE,
                                              weights_only=True)["state_dict"])
                mm.eval()
            except Exception as e:
                log.warning("Multimodal non chargé : %s", e)
                mm = _MockMultimodal().eval()

        tfidf = None
        try:
            tfidf = joblib.load(CHECKPOINT_DIR / "tfidf_vectorizer.pkl")
        except FileNotFoundError:
            pass

        return {"classifier": clf, "autoencoder": ae, "multimodal": mm,
                "tfidf": tfidf, "architecture": architecture, "is_mock": False}

    except Exception as e:
        log.warning("Chargement échoué (%s) — mode démo", e)
        return {"classifier": _MockClassifier().eval(),
                "autoencoder": _MockAutoencoder().eval(),
                "multimodal":  _MockMultimodal().eval(),
                "architecture": architecture, "is_mock": True}


# ===========================================================================
# INFÉRENCE
# ===========================================================================

def preprocess_image(pil_img: Image.Image, size: int = 64) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.498], std=[0.248]),
    ])
    return t(pil_img).unsqueeze(0)


def predict_classification(models: dict, img_tensor: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        probs = torch.sigmoid(models["classifier"](img_tensor)).squeeze().numpy()
    return {cls: float(p) for cls, p in zip(CHEST_CLASSES, probs)}


def predict_anomaly(models: dict, img_tensor: torch.Tensor) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        recon, _ = models["autoencoder"](img_tensor)
        score = float(((img_tensor - recon) ** 2).mean())
    return score, recon.squeeze()


def encode_text_tfidf(text: str, vocab_size: int = 5000) -> torch.Tensor:
    words = text.lower().split()
    vec   = np.zeros(vocab_size, dtype=np.float32)
    for w in words:
        vec[hash(w) % vocab_size] += 1.0 / (len(words) + 1e-6)
    return torch.tensor(vec).unsqueeze(0)


def predict_multimodal(models: dict, img_tensor: torch.Tensor,
                       text: str) -> dict[str, float]:
    with torch.no_grad():
        probs = torch.sigmoid(
            models["multimodal"](img_tensor, encode_text_tfidf(text))
        ).squeeze().numpy()
    return {cls: float(p) for cls, p in zip(CHEST_CLASSES, probs)}


# ===========================================================================
# COMPOSANTS UI
# ===========================================================================

def render_sidebar() -> dict:
    st.sidebar.markdown(
        "<div style='font-size:1rem;font-weight:700;color:#0f172a;"
        "margin-bottom:4px;'>RadioScan AI</div>"
        "<div style='font-size:0.75rem;color:#94a3b8;margin-bottom:1rem;'>"
        "Aide au tri radiologique — v1.0</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    st.sidebar.markdown("<div class='section-label'>Architecture</div>",
                        unsafe_allow_html=True)
    arch = st.sidebar.selectbox(
        "Architecture", label_visibility="collapsed",
        options=["simple_cnn", "resnet", "vit"],
        format_func=lambda x: {
            "simple_cnn": "SimpleCNN — from scratch",
            "resnet":     "ResNet-50 — Transfer Learning",
            "vit":        "ViT-B/16 — Vision Transformer",
        }[x], index=1,
    )

    st.sidebar.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='section-label'>Seuil de détection</div>",
                        unsafe_allow_html=True)
    threshold = st.sidebar.slider(
        "Seuil", label_visibility="collapsed",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Probabilité minimale pour signaler une pathologie",
    )
    st.sidebar.caption(f"Seuil actuel : **{threshold:.0%}**")

    st.sidebar.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='section-label'>Seuil anomalie</div>",
                        unsafe_allow_html=True)
    anomaly_p = st.sidebar.radio(
        "Percentile", label_visibility="collapsed",
        options=["P95", "P99"], horizontal=True,
        help="Percentile calibré sur le test set",
    )
    anomaly_threshold = ANOMALY_THRESHOLD_P95 if anomaly_p == "P95" else ANOMALY_THRESHOLD_P99

    st.sidebar.markdown("---")
    models = load_models(architecture=arch)

    st.sidebar.markdown("<div class='section-label'>Statut des modèles</div>",
                        unsafe_allow_html=True)
    if models.get("is_mock"):
        st.sidebar.markdown(
            "<span class='badge badge-demo'>Mode démo — données simulées</span>",
            unsafe_allow_html=True,
        )
        st.sidebar.caption(
            "Checkpoints manquants. Lancez `train_models.py` "
            "puis `anomaly_detector.py` pour charger les vrais modèles."
        )
    else:
        st.sidebar.markdown(
            "<span class='badge badge-live'>Modèles entraînés chargés</span>",
            unsafe_allow_html=True,
        )
        st.sidebar.caption(f"Architecture active : **{arch}**")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:0.72rem;color:#94a3b8;line-height:1.7;'>"
        "EFREI Paris — Deep Learning Médical<br>"
        "2025 / 2026<br>"
        "Données : ChestMNIST+ · OpenI (NIH)"
        "</div>",
        unsafe_allow_html=True,
    )

    return {"architecture": arch, "threshold": threshold,
            "anomaly_threshold": anomaly_threshold, "models": models}


def render_pathology_results(probs: dict[str, float], threshold: float,
                              title: str = "Probabilités par pathologie") -> None:
    st.markdown(f"<div class='section-label'>{title}</div>", unsafe_allow_html=True)

    detected = [k for k, v in probs.items() if v >= threshold]
    if detected:
        st.markdown(
            f"<div style='font-size:.84rem;color:#7f1d1d;font-weight:600;"
            f"margin-bottom:10px;'>{len(detected)} pathologie(s) "
            f"au seuil {threshold:.0%}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='font-size:.84rem;color:#14532d;font-weight:500;"
            f"margin-bottom:10px;'>Aucune pathologie au seuil {threshold:.0%}</div>",
            unsafe_allow_html=True)

    rows_html = ""
    for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        if prob >= threshold and prob >= 0.7:
            row_c, ind_c, bar_color = "path-high", "ind-high", "#ef4444"
        elif prob >= threshold:
            row_c, ind_c, bar_color = "path-mid", "ind-mid", "#f59e0b"
        else:
            row_c, ind_c, bar_color = "path-low", "ind-low", "#94a3b8"

        pct = min(prob * 100, 100)
        rows_html += (
            f"<div class='path-row {row_c}'>"
            f"<span class='path-indicator {ind_c}'></span>"
            f"<span style='flex:1'>{cls}</span>"
            f"<span style='font-variant-numeric:tabular-nums;font-size:.84rem;'>"
            f"{prob:.1%}</span></div>"
            f"<div class='prog-track'>"
            f"<div class='prog-fill' style='width:{pct:.1f}%;background:{bar_color};'></div>"
            f"</div>"
        )
    st.markdown(rows_html, unsafe_allow_html=True)


def render_anomaly_gauge(score: float, threshold: float) -> None:
    st.markdown("<div class='section-label'>Score d'anomalie — Autoencodeur convolutif</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Score MSE", f"{score:.5f}")
    c2.metric("Seuil P95", f"{ANOMALY_THRESHOLD_P95:.5f}")
    c3.metric("Seuil P99", f"{ANOMALY_THRESHOLD_P99:.5f}")

    normalized_pct = min(score / (ANOMALY_THRESHOLD_P99 * 1.5), 1.0) * 100
    if score >= ANOMALY_THRESHOLD_P99:
        bar_color = "#dc2626"
    elif score >= ANOMALY_THRESHOLD_P95:
        bar_color = "#f59e0b"
    else:
        bar_color = "#22c55e"
    st.markdown(
        f"<div class='prog-track' style='height:8px;margin:8px 0 16px 0;'>"
        f"<div class='prog-fill' style='width:{normalized_pct:.1f}%;"
        f"background:{bar_color};height:8px;'></div></div>",
        unsafe_allow_html=True,
    )

    if score >= ANOMALY_THRESHOLD_P99:
        st.markdown(
            "<div class='anomaly-critical'>Anomalie critique — score supérieur au P99."
            "<br><span style='font-weight:400;font-size:.85rem;'>"
            "Image fortement hors distribution. Révision prioritaire recommandée.</span></div>",
            unsafe_allow_html=True)
    elif score >= ANOMALY_THRESHOLD_P95:
        st.markdown(
            f"<div class='anomaly-warn'>Image atypique — score ({score:.5f}) "
            f"supérieur au seuil P95 ({threshold:.5f})."
            f"<br><span style='font-weight:400;font-size:.85rem;'>"
            f"Caractéristiques inhabituelles détectées.</span></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='anomaly-ok'>Image dans la distribution normale — "
            f"score MSE = {score:.5f}."
            f"<br><span style='font-weight:400;font-size:.85rem;'>"
            f"Inférieur au seuil P95 ({threshold:.5f}).</span></div>",
            unsafe_allow_html=True)


def render_reconstruction_comparison(original: Image.Image,
                                      recon_tensor: torch.Tensor) -> None:
    with st.expander("Visualiser la reconstruction (Autoencodeur)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(original, caption="Image originale", use_container_width=True)
        with c2:
            rn = recon_tensor.numpy()
            rn = (rn - rn.min()) / (rn.max() - rn.min() + 1e-8)
            st.image(Image.fromarray((rn * 255).astype(np.uint8)),
                     caption="Reconstruction AE", use_container_width=True)
        with c3:
            diff = np.abs(
                np.array(original.resize((64, 64)).convert("L")).astype(float) / 255.0 - rn
            )
            diff_img = Image.fromarray((diff / (diff.max() + 1e-8) * 255).astype(np.uint8))
            st.image(diff_img, caption="Carte d'erreur |orig − recon|",
                     use_container_width=True)


def render_multimodal_comparison(probs_img: dict[str, float],
                                  probs_mm: dict[str, float],
                                  threshold: float) -> None:
    st.markdown(
        "<div class='section-label'>Image seule vs Multimodal (Image + Texte)</div>",
        unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        render_pathology_results(probs_img, threshold, title="CNN — Image seule")
    with c2:
        render_pathology_results(probs_mm,  threshold, title="Fusion — Image + Rapport")

    deltas = {cls: abs(probs_mm.get(cls, 0) - probs_img.get(cls, 0))
              for cls in CHEST_CLASSES}
    top_cls, top_delta = max(deltas.items(), key=lambda x: x[1])
    if top_delta > 0.05:
        direction = "augmente" if probs_mm[top_cls] > probs_img[top_cls] else "diminue"
        st.info(
            f"**Impact du compte-rendu :** la probabilité de **{top_cls}** "
            f"{direction} de {top_delta:.1%} grâce au rapport radiologique."
        )


def render_placeholder(models: dict) -> None:
    """Écran d'accueil quand aucune image n'est chargée."""

    # ── Pipeline ──
    st.markdown("<div class='section-label'>Pipeline de traitement</div>",
                unsafe_allow_html=True)

    steps = [
        ("#dbeafe", "#1d4ed8", "Entrée",
         "Radiographie (JPG / PNG)",
         "Chargez une image dans le panneau gauche pour démarrer l'analyse."),
        ("#dcfce7", "#16a34a", "Étape 1",
         "Classification supervisée",
         "CNN prédit 14 probabilités de pathologies via BCEWithLogitsLoss."),
        ("#fef3c7", "#d97706", "Étape 2",
         "Détection d'anomalies",
         "L'autoencodeur calcule MSE(original, reconstruit) et compare aux seuils P95/P99."),
        ("#f3e8ff", "#9333ea", "Étape 3",
         "Fusion multimodale",
         "Le rapport texte est combiné à l'image via Late Fusion (TF-IDF + CNN)."),
    ]

    cols = st.columns(4)
    for col, (bg, accent, label, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="pipeline-step" style="background:{bg}22;border-color:{accent}33;">
                <div>
                    <div style="font-size:.62rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:.08em;color:{accent};margin-bottom:4px;">
                        {label}
                    </div>
                    <div class="pipeline-title">{title}</div>
                    <div class="pipeline-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    # ── Architectures ──
    st.markdown("<div class='section-label'>Architectures disponibles</div>",
                unsafe_allow_html=True)

    arch_data = [
        {
            "color": "#1d4ed8",
            "title": "SimpleCNN",
            "sub": "Entraîné from scratch sur ChestMNIST+",
            "stats": [
                ("Blocs Conv-BN-ReLU", "4"),
                ("Paramètres", "~2.1 M"),
                ("AUC-ROC (val)", "0.741"),
                ("Temps / époque", "~4 min"),
            ],
        },
        {
            "color": "#16a34a",
            "title": "ResNet-50",
            "sub": "Transfer Learning — pré-entraîné ImageNet",
            "stats": [
                ("Blocs résiduels", "16"),
                ("Paramètres", "~25 M"),
                ("AUC-ROC (val)", "~0.840"),
                ("Fine-tuning", "Couche finale"),
            ],
        },
        {
            "color": "#9333ea",
            "title": "ViT-B/16",
            "sub": "Vision Transformer — pré-entraîné ImageNet",
            "stats": [
                ("Transformer blocks", "12"),
                ("Paramètres", "~86 M"),
                ("AUC-ROC (val)", "~0.862"),
                ("Patches 16×16", "16 tokens"),
            ],
        },
    ]

    cols = st.columns(3)
    for col, arch in zip(cols, arch_data):
        with col:
            stats_html = "".join(
                f"<div class='arch-card-stat'>{k}<span>{v}</span></div>"
                for k, v in arch["stats"]
            )
            st.markdown(f"""
            <div class="arch-card">
                <div class="arch-card-accent"
                     style="background:{arch['color']};"></div>
                <div class="arch-card-title">{arch['title']}</div>
                <div class="arch-card-sub">{arch['sub']}</div>
                {stats_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    # ── Dataset stats ──
    st.markdown("<div class='section-label'>Datasets utilisés</div>",
                unsafe_allow_html=True)

    dataset_stats = [
        ("78 468", "Images train", "ChestMNIST+"),
        ("22 433", "Images test",  "ChestMNIST+"),
        ("14",     "Pathologies",  "Multi-label"),
        ("7 428",  "Paires img/texte", "OpenI (NIH)"),
        ("3 851",  "Cas distincts",    "OpenI (NIH)"),
        ("16.2 %", "Avec pathologie",  "OpenI"),
    ]

    cols = st.columns(6)
    for col, (val, label, source) in zip(cols, dataset_stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
                <div class="stat-sub">{source}</div>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# APPLICATION PRINCIPALE
# ===========================================================================

def main() -> None:
    # ── Header ──
    st.markdown("""
    <div class="app-header">
        <div class="app-logo">&#x1FAC1;</div>
        <div>
            <p class="app-title">RadioScan AI</p>
            <p class="app-subtitle">
                Classification multi-label &amp; Détection d'anomalies —
                ChestMNIST+ &amp; OpenI
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>Avertissement :</strong> démonstrateur académique uniquement.
        Ne constitue pas un dispositif médical certifié. Tout résultat doit être
        validé par un radiologue qualifié avant toute décision clinique.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

    # ── Sidebar ──
    config            = render_sidebar()
    models            = config["models"]
    threshold         = config["threshold"]
    anomaly_threshold = config["anomaly_threshold"]

    # ── Layout ──
    left_col, right_col = st.columns([1, 2], gap="large")

    # ─────────────── GAUCHE ───────────────
    with left_col:
        st.markdown("<div class='section-label'>Radiographie</div>",
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Formats acceptés : JPG, PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            pil_img = Image.open(uploaded_file).convert("RGB")
            st.image(pil_img, caption=uploaded_file.name, use_container_width=True)
            st.caption(f"{pil_img.size[0]} × {pil_img.size[1]} px")

        st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Compte-rendu radiologique (optionnel)</div>",
                    unsafe_allow_html=True)
        report_text = st.text_area(
            "Compte-rendu", label_visibility="collapsed",
            placeholder=(
                "Ex : Bilateral lungs are clear. No pleural effusion. "
                "Heart size is within normal limits."
            ),
            height=130,
        )
        if report_text:
            st.caption(f"{len(report_text.split())} mots")

        st.markdown("<div style='margin-top:.75rem'></div>", unsafe_allow_html=True)
        analyze_btn = st.button(
            "Analyser la radiographie",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

        if not uploaded_file:
            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            with st.expander("Utiliser une image de démonstration"):
                openi_dir     = Path("data/openi/openi_images")
                demo_img_path = None
                if openi_dir.exists():
                    imgs = list(openi_dir.glob("*.png"))
                    if imgs:
                        demo_img_path = imgs[0]
                if demo_img_path:
                    st.image(Image.open(demo_img_path),
                             caption=demo_img_path.name, use_container_width=True)
                    with open(demo_img_path, "rb") as f:
                        st.download_button("Télécharger cette image", data=f.read(),
                                           file_name=demo_img_path.name, mime="image/png")
                else:
                    st.caption("Dataset OpenI non trouvé. Lancez `python download_openi.py`.")

    # ─────────────── DROITE ───────────────
    with right_col:
        if uploaded_file and analyze_btn:
            pil_img = Image.open(uploaded_file)

            with st.spinner("Analyse en cours…"):
                t0 = time.time()
                img_tensor    = preprocess_image(pil_img, size=64)
                probs_img     = predict_classification(models, img_tensor)
                anomaly_score, recon_tensor = predict_anomaly(models, img_tensor)
                probs_mm      = (predict_multimodal(models, img_tensor, report_text)
                                 if report_text.strip() else None)
                elapsed = time.time() - t0

            st.caption(
                f"Analyse en {elapsed:.2f} s  ·  {config['architecture']}  ·  "
                f"{'Données simulées' if models['is_mock'] else 'Modèles entraînés'}"
            )

            tab1, tab2, tab3 = st.tabs([
                "Classification supervisée",
                "Détection d'anomalies",
                "Analyse multimodale",
            ])

            with tab1:
                detected = {k: v for k, v in probs_img.items() if v >= threshold}
                m1, m2, m3 = st.columns(3)
                m1.metric("Pathologies détectées", len(detected))
                m2.metric("Score maximum",         f"{max(probs_img.values()):.1%}")
                m3.metric("Seuil appliqué",        f"{threshold:.0%}")
                st.markdown("<div style='margin-top:.75rem'></div>", unsafe_allow_html=True)
                render_pathology_results(probs_img, threshold)
                with st.expander("Données brutes (JSON)"):
                    st.json({k: round(v, 4) for k, v in
                             sorted(probs_img.items(), key=lambda x: x[1], reverse=True)})

            with tab2:
                render_anomaly_gauge(anomaly_score, anomaly_threshold)
                st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
                render_reconstruction_comparison(pil_img.convert("L"), recon_tensor.detach())
                with st.expander("Comment interpréter le score d'anomalie ?"):
                    st.markdown("""
                    Le score est la **MSE** entre l'image originale et sa reconstruction
                    par l'autoencodeur, entraîné uniquement sur le train set.

                    | Plage | Interprétation |
                    |---|---|
                    | Score < P95 | Image dans la distribution d'entraînement |
                    | P95 ≤ Score < P99 | Image atypique — attention recommandée |
                    | Score ≥ P99 | Hors distribution — révision prioritaire |

                    *Seuils calibrés sur le test set de ChestMNIST+ (22 433 images).*
                    """)

            with tab3:
                if probs_mm is not None:
                    render_multimodal_comparison(probs_img, probs_mm, threshold)
                    with st.expander("Données brutes (JSON)"):
                        st.json({
                            cls: {"image_only": round(probs_img[cls], 4),
                                  "multimodal":  round(probs_mm[cls],  4),
                                  "delta":       round(probs_mm[cls] - probs_img[cls], 4)}
                            for cls in CHEST_CLASSES
                        })
                else:
                    st.info(
                        "Renseignez un compte-rendu radiologique dans le panneau gauche "
                        "pour activer la prédiction multimodale."
                    )
                    st.markdown("""
                    **Exemple de rapport :**
                    ```
                    Bilateral lungs demonstrate clear lung fields without consolidation,
                    pleural effusion, or pneumothorax. Cardiac silhouette within normal
                    limits. No acute cardiopulmonary process identified.
                    ```
                    """)

        elif uploaded_file and not analyze_btn:
            st.info("Cliquez sur **Analyser la radiographie** pour lancer l'inférence.")

        else:
            render_placeholder(models)

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#94a3b8;font-size:.76rem;'>"
        "RadioScan AI &nbsp;·&nbsp; Projet Deep Learning Médical &nbsp;·&nbsp; "
        "EFREI Paris 2025/2026 &nbsp;·&nbsp; "
        "ChestMNIST+ (MedMNIST v3) &amp; OpenI (NIH)"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
