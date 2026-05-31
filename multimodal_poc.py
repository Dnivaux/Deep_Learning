# multimodal_poc.py
"""
Phase 4 — Preuve de Concept Multimodale (Image + Texte)
Dataset : OpenI (Indiana University Chest X-ray + Radiology Reports)
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_pipeline import setup_mlflow, setup_reproducibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels directement depuis ChestMNIST (14 classes, pas de mapping = pas de perte)
# Chargés dynamiquement depuis medmnist.INFO pour garantir la cohérence
from medmnist import INFO as _MEDMNIST_INFO
LABEL_NAMES: list[str] = list(_MEDMNIST_INFO["chestmnist"]["label"].values())
N_CLASSES = len(LABEL_NAMES)  # 14


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultimodalConfig:
    # Données
    data_root: str = "data/openi"
    image_size: int = 64
    max_text_features: int = 5000       # Taille vocabulaire TF-IDF
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # Entraînement
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    num_workers: int = 0

    # Architecture
    img_embed_dim: int = 256
    txt_embed_dim: int = 256
    dropout: float = 0.3

    # MLflow
    mlflow_experiment: str = "ChestMNIST_Multimodal_POC"
    mlflow_uri: str = "sqlite:///mlflow.db"
    output_dir: str = "multimodal_outputs"
    checkpoint_dir: str = "checkpoints"

    # Si True : utilise des données synthétiques pour tester le pipeline
    use_synthetic: bool = False


# ---------------------------------------------------------------------------
# 2. Chargement & Préparation des Données OpenI
# ---------------------------------------------------------------------------

def _build_label_based_report(label_vec: list[int], rng: np.random.Generator) -> str:
    """
    Génère un compte-rendu radiologique réaliste à partir des VRAIS labels.
    Les textes sont ancrés sur les pathologies réellement présentes dans l'image.
    """
    vocab_normal = [
        "No acute cardiopulmonary process identified.",
        "Bilateral lungs are clear without consolidation or effusion.",
        "Heart size is within normal limits. No pneumothorax.",
        "Normal chest radiograph. No acute findings.",
        "Lungs are well expanded and clear. No pleural effusion.",
    ]
    vocab_pathology = {
        "Atelectasis":         ["Atelectasis noted at the lung base.", "Linear atelectasis present.", "Platelike atelectasis identified."],
        "Cardiomegaly":        ["Cardiomegaly is present.", "Enlarged cardiac silhouette noted.", "Cardiac enlargement observed."],
        "Effusion":            ["Pleural effusion present.", "Blunting of the costophrenic angle.", "Small pleural effusion identified."],
        "Infiltration":        ["Infiltrates are seen.", "Patchy opacity in the lung field.", "Airspace disease identified."],
        "Mass":                ["Pulmonary mass identified.", "Rounded opacity concerning for mass.", "Nodular density seen."],
        "Nodule":              ["Pulmonary nodule present.", "Small nodule identified.", "Calcified nodule seen."],
        "Pneumonia":           ["Consolidation consistent with pneumonia.", "Lobar consolidation present.", "Pneumonia cannot be excluded."],
        "Pneumothorax":        ["Pneumothorax identified.", "Visceral pleural line visible.", "Hyperlucency consistent with pneumothorax."],
        "Consolidation":       ["Consolidation is present.", "Airspace consolidation noted.", "Lobar consolidation identified."],
        "Edema":               ["Pulmonary edema present.", "Vascular congestion noted.", "Interstitial edema identified."],
        "Emphysema":           ["Emphysema noted.", "Hyperinflation present.", "Air trapping identified."],
        "Fibrosis":            ["Pulmonary fibrosis present.", "Interstitial markings increased.", "Fibrotic changes noted."],
        "Pleural Thickening":  ["Pleural thickening identified.", "Thickened pleura noted.", "Pleural irregularity seen."],
        "Hernia":              ["Hernia identified.", "Diaphragmatic hernia present.", "Bowel loops in chest."],
    }

    active = [LABEL_NAMES[i] for i, v in enumerate(label_vec) if v == 1 and i < len(LABEL_NAMES)]

    if not active:
        return rng.choice(vocab_normal)

    parts = [rng.choice(vocab_pathology[lbl]) for lbl in active if lbl in vocab_pathology]
    return " ".join(parts) if parts else rng.choice(vocab_normal)


def _build_from_chestmnist(data_root: str, chestmnist_root: str = "data",
                            size: int = 64, max_samples: int = 3000) -> None:
    """
    Construit le dataset multimodal en réutilisant les VRAIES images ChestMNIST+
    déjà téléchargées, avec des rapports textuels générés depuis les vrais labels.

    Garantie : chaque image est une vraie radiographie thoracique (pas du bruit).
    Vérification anti-bruit : rejet des images avec variance trop faible.
    """
    from medmnist import ChestMNIST

    # LABEL_NAMES est déjà les 14 classes ChestMNIST — pas de mapping nécessaire
    log.info("Construction du dataset multimodal depuis ChestMNIST+ (images réelles)…")
    log.info("Labels utilisés (%d) : %s", N_CLASSES, LABEL_NAMES)
    Path(data_root).mkdir(parents=True, exist_ok=True)
    img_dir = Path(data_root) / "images"
    img_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)
    records = []

    for split in ["train", "val", "test"]:
        try:
            ds = ChestMNIST(split=split, download=False, size=size, root=chestmnist_root)
        except RuntimeError:
            try:
                ds = ChestMNIST(split=split, download=True, size=size, root=chestmnist_root)
            except Exception as e:
                log.error("Impossible de charger ChestMNIST split=%s : %s", split, e)
                continue

        log.info("  Split %s : %d images", split, len(ds))
        n_split = min(len(ds), max_samples // 3)

        for i in range(n_split):
            pil_img, label_arr = ds[i]

            # Vérification anti-bruit
            img_np = np.array(pil_img) if not isinstance(pil_img, np.ndarray) else pil_img
            if img_np.std() < 5.0:
                continue

            # Sauvegarde image
            img_path = str(img_dir / f"{split}_{i:05d}.png")
            if isinstance(pil_img, Image.Image):
                pil_img.save(img_path)
            else:
                Image.fromarray(img_np).save(img_path)

            # Labels directs ChestMNIST (14 classes) — pas de mapping
            raw = label_arr.flatten() if hasattr(label_arr, "flatten") else np.array(label_arr)
            label_vec = [int(v) for v in raw[:N_CLASSES]]
            # Compléter si nécessaire
            while len(label_vec) < N_CLASSES:
                label_vec.append(0)

            # Rapport textuel basé sur les vrais labels
            report = _build_label_based_report(label_vec, rng)

            records.append({
                "text": report,
                "image_path": img_path,
                "labels": label_vec,
                "source": f"chestmnist_{split}",
            })

        if len(records) >= max_samples:
            break

    if not records:
        raise RuntimeError(
            "Aucune image valide trouvée dans ChestMNIST. "
            "Vérifiez que le dataset est téléchargé dans le dossier 'data/'."
        )

    meta_path = Path(data_root) / "records.json"
    with open(meta_path, "w") as f:
        json.dump(records, f)

    # Diagnostic de distribution des classes
    lbl_mat = np.array([r["labels"] for r in records])
    n_pos_total = sum(1 for r in records if sum(r["labels"]) > 0)
    log.info("Dataset prêt : %d paires | %d avec pathologie (%.0f%%)",
             len(records), n_pos_total, 100 * n_pos_total / len(records))
    log.info("Positifs par classe :")
    for i, lbl in enumerate(LABEL_NAMES):
        n = int(lbl_mat[:, i].sum())
        log.info("  %-25s : %4d (%.1f%%)", lbl, n, 100 * n / len(records))

    # Vérification image[0]
    check_img = np.array(Image.open(records[0]["image_path"]))
    log.info("Image[0] — mean=%.1f std=%.1f | '%s...'",
             check_img.mean(), check_img.std(), records[0]["text"][:60])


class OpenIDataset(Dataset):
    """
    Dataset PyTorch pour paires (Image, Texte_TF-IDF, Labels).

    Le texte est déjà vectorisé (np.ndarray float32) car TfidfVectorizer
    est fitté une seule fois sur le train set puis appliqué à val/test.
    """

    def __init__(
        self,
        records: list[dict],
        tfidf_matrix: np.ndarray,
        image_transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.tfidf_matrix = tfidf_matrix       # (N, vocab_size) float32
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]

        # Image
        img = Image.open(rec["image_path"]).convert("L")
        img_tensor = self.image_transform(img)

        # Texte (déjà vectorisé)
        txt_tensor = torch.tensor(self.tfidf_matrix[idx], dtype=torch.float32)

        # Labels
        lbl_tensor = torch.tensor(rec["labels"], dtype=torch.float32)

        return img_tensor, txt_tensor, lbl_tensor


def get_openi_dataloaders(cfg: MultimodalConfig) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Charge OpenI, vectorise le texte (TF-IDF fit sur train uniquement),
    et retourne les DataLoaders Train/Val/Test.

    Garanties anti-leakage :
    - TF-IDF fitté UNIQUEMENT sur le train set.
    - Split aléatoire avec seed fixe avant toute transformation.
    """
    data_root = cfg.data_root
    meta_path = Path(data_root) / "records.json"

    # Chargement ou construction depuis ChestMNIST (vraies images)
    if not meta_path.exists():
        log.info("Construction du dataset depuis les vraies images ChestMNIST+…")
        _build_from_chestmnist(
            data_root=data_root,
            chestmnist_root="data",   # dossier où ChestMNIST est déjà téléchargé
            size=cfg.image_size,
            max_samples=3000,
        )

    # Vérification d'intégrité : rejette si toutes les images sont du bruit
    with open(meta_path) as f:
        probe = json.load(f)
    probe_records = probe["records"] if isinstance(probe, dict) and "records" in probe else probe
    sample_img = np.array(Image.open(probe_records[0]["image_path"]))
    if sample_img.std() < 5.0:
        log.error(
            "BRUIT DÉTECTÉ dans les images (std=%.2f). "
            "Supprimez le dossier '%s' et relancez pour reconstruire.",
            sample_img.std(), data_root,
        )
        raise RuntimeError(
            f"Images corrompues (bruit blanc) dans '{data_root}'. "
            f"Supprimez le dossier et relancez le script."
        )

    with open(meta_path) as f:
        raw = json.load(f)

    # Supporte les deux formats :
    # - Ancien (liste) : [{image_path, text, labels}, ...]
    # - Nouveau OpenI  : {label_names: [...], records: [...]}
    if isinstance(raw, dict) and "records" in raw:
        all_records = raw["records"]
        # Synchronise LABEL_NAMES si le fichier en contient
        if "label_names" in raw:
            global LABEL_NAMES, N_CLASSES
            LABEL_NAMES = raw["label_names"]
            N_CLASSES   = len(LABEL_NAMES)
            log.info("Labels chargés depuis records.json : %d classes", N_CLASSES)
    else:
        all_records = raw

    log.info("Dataset chargé : %d échantillons", len(all_records))

    # Split stratifié sur la somme des labels (normal vs anormal)
    is_normal = [int(sum(r["labels"]) == 0) for r in all_records]
    indices = list(range(len(all_records)))

    train_idx, tmp_idx = train_test_split(
        indices, test_size=cfg.val_ratio + cfg.test_ratio,
        stratify=is_normal, random_state=cfg.seed,
    )
    val_ratio_adj = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    is_normal_tmp = [is_normal[i] for i in tmp_idx]
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=1 - val_ratio_adj,
        stratify=is_normal_tmp, random_state=cfg.seed,
    )

    train_records = [all_records[i] for i in train_idx]
    val_records   = [all_records[i] for i in val_idx]
    test_records  = [all_records[i] for i in test_idx]

    log.info("Split — Train: %d | Val: %d | Test: %d",
             len(train_records), len(val_records), len(test_records))

    # TF-IDF : fit sur train uniquement (anti-leakage)
    log.info("Vectorisation TF-IDF (max_features=%d)…", cfg.max_text_features)
    tfidf = TfidfVectorizer(
        max_features=cfg.max_text_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_texts = [r["text"] for r in train_records]
    val_texts   = [r["text"] for r in val_records]
    test_texts  = [r["text"] for r in test_records]

    X_train = tfidf.fit_transform(train_texts).toarray().astype(np.float32)
    X_val   = tfidf.transform(val_texts).toarray().astype(np.float32)
    X_test  = tfidf.transform(test_texts).toarray().astype(np.float32)

    # Transforms image
    img_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_ds = OpenIDataset(train_records, X_train, img_transform)
    val_ds   = OpenIDataset(val_records,   X_val,   img_transform)
    test_ds  = OpenIDataset(test_records,  X_test,  img_transform)

    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, generator=gen)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers)

    metadata = {
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "tfidf_vocab_size": len(tfidf.vocabulary_),
        "n_classes": N_CLASSES, "label_names": LABEL_NAMES,
    }
    return train_loader, val_loader, test_loader, metadata, tfidf


# ---------------------------------------------------------------------------
# 3. Architectures
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """
    Petit CNN extrayant un embedding de dimension img_embed_dim.
    Utilisé seul (ImageOnly) et comme branche image du modèle multimodal.
    """

    def __init__(self, embed_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class TextEncoder(nn.Module):
    """
    MLP appliqué sur des features TF-IDF précomputées.
    TF-IDF est léger, interprétable et très rapide sur CPU.
    Alternative à un BERT : plus lourd mais meilleure représentation contextuelle.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vocab_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ImageOnlyModel(nn.Module):
    """Modèle Baseline 1 : classification depuis l'image uniquement."""

    def __init__(self, n_classes: int, embed_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = ImageEncoder(embed_dim, dropout)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        # texts ignoré — interface unifiée pour la boucle d'entraînement
        return self.head(self.encoder(images))


class TextOnlyModel(nn.Module):
    """Modèle Baseline 2 : classification depuis le texte uniquement."""

    def __init__(self, vocab_size: int, n_classes: int,
                 embed_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = TextEncoder(vocab_size, embed_dim, dropout)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        # images ignoré — interface unifiée
        return self.head(self.encoder(texts))


class MultimodalFusionModel(nn.Module):
    """
    Modèle Multimodal — Fusion Tardive par Concaténation.

    Architecture :
        Image  → ImageEncoder → embed_img (256-d) ─┐
                                                     ├─ concat (512-d) → Classifier → logits
        Texte  → TextEncoder  → embed_txt (256-d) ─┘

    Choix de la fusion tardive :
    - Chaque modalité possède son propre espace de représentation
    - La concaténation est différentiable end-to-end
    - Le classifieur apprend quels aspects de chaque modalité sont pertinents
    - Alternative (fusion intermédiaire) : cross-attention — meilleure mais ~10× plus lourde
    """

    def __init__(self, vocab_size: int, n_classes: int,
                 img_embed_dim: int = 256, txt_embed_dim: int = 256,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.img_encoder = ImageEncoder(img_embed_dim, dropout)
        self.txt_encoder = TextEncoder(vocab_size, txt_embed_dim, dropout)

        fused_dim = img_embed_dim + txt_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, n_classes),
        )

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        img_emb = self.img_encoder(images)   # (B, 256)
        txt_emb = self.txt_encoder(texts)    # (B, 256)
        fused = torch.cat([img_emb, txt_emb], dim=1)  # (B, 512)
        return self.classifier(fused)


def build_pos_weight_openi(train_loader: DataLoader) -> torch.Tensor:
    """Calcule pos_weight = n_neg/n_pos par classe pour BCEWithLogitsLoss."""
    all_labels = [lbl.numpy() for _, _, lbl in train_loader]
    lbl_mat = np.concatenate(all_labels, axis=0)
    n = lbl_mat.shape[0]
    n_pos = lbl_mat.sum(axis=0)
    n_neg = n - n_pos
    pw = np.clip(n_neg / np.clip(n_pos, 1, None), 1.0, 30.0)
    return torch.tensor(pw, dtype=torch.float32).to(DEVICE)


# ---------------------------------------------------------------------------
# 4. Boucles d'entraînement
# ---------------------------------------------------------------------------

def train_epoch_mm(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: optim.Optimizer,
    epoch: int,
    model_name: str,
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"[{model_name}] Train Ep{epoch:02d}",
                leave=False, dynamic_ncols=True)
    for images, texts, labels in pbar:
        images = images.to(DEVICE, non_blocking=True)
        texts  = texts.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images, texts)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate_epoch_mm(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    epoch: int,
    model_name: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for images, texts, labels in tqdm(loader, desc=f"[{model_name}] Val   Ep{epoch:02d}",
                                      leave=False, dynamic_ncols=True):
        images = images.to(DEVICE); texts = texts.to(DEVICE)
        labels_f = labels.float().to(DEVICE)

        logits = model(images, texts)
        total_loss += criterion(logits, labels_f).item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.numpy())

    probs_mat  = np.concatenate(all_probs,  axis=0)
    labels_mat = np.concatenate(all_labels, axis=0)
    preds_mat  = (probs_mat >= 0.5).astype(int)

    valid_aucs, valid_aps = [], []
    for i in range(N_CLASSES):
        if labels_mat[:, i].sum() == 0:
            continue
        valid_aucs.append(roc_auc_score(labels_mat[:, i], probs_mat[:, i]))
        valid_aps.append(average_precision_score(labels_mat[:, i], probs_mat[:, i]))

    return {
        "val_loss":      round(total_loss / len(loader), 5),
        "val_auc_macro": round(float(np.mean(valid_aucs)) if valid_aucs else 0.0, 4),
        "val_map":       round(float(np.mean(valid_aps))  if valid_aps  else 0.0, 4),
        "val_f1_macro":  round(f1_score(labels_mat, preds_mat, average="macro",  zero_division=0), 4),
        "val_f1_micro":  round(f1_score(labels_mat, preds_mat, average="micro",  zero_division=0), 4),
    }


def train_one_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: MultimodalConfig,
    pos_weight: torch.Tensor,
) -> tuple[dict, list[dict]]:
    """Boucle complète pour un modèle + early stopping + checkpoint."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_auc = 0.0
    patience_counter = 0
    history = []
    ckpt_path = Path(cfg.checkpoint_dir) / f"best_{model_name}.pt"

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch_mm(model, train_loader, criterion, optimizer, epoch, model_name)
        val_metrics = validate_epoch_mm(model, val_loader, criterion, epoch, model_name)

        val_metrics["train_loss"] = round(train_loss, 5)
        val_metrics["epoch_time_s"] = round(time.time() - t0, 1)
        history.append(val_metrics)

        scheduler.step(val_metrics["val_auc_macro"])

        log.info(
            "[%s] Ep%02d — loss=%.4f | AUC=%.4f | F1=%.4f",
            model_name, epoch, train_loss,
            val_metrics["val_auc_macro"], val_metrics["val_f1_macro"],
        )

        current_auc = val_metrics["val_auc_macro"]
        if current_auc > best_auc or not ckpt_path.exists():
            best_auc = current_auc
            patience_counter = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "best_auc": best_auc}, ckpt_path)
            log.info("  ✓ [%s] Checkpoint sauvegardé (AUC=%.4f)", model_name, best_auc)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                log.info("[%s] Early stopping epoch %d", model_name, epoch)
                break

    # Recharge meilleur état (le checkpoint existe toujours car on sauvegarde au moins epoch 1)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        best_epoch = ckpt["epoch"]
    else:
        # Sécurité : sauvegarde l'état actuel si rien n'a été sauvegardé
        log.warning("[%s] Aucun checkpoint trouvé — sauvegarde de l'état final.", model_name)
        torch.save({"epoch": 1, "state_dict": model.state_dict(), "best_auc": best_auc}, ckpt_path)
        best_epoch = 1

    return {"best_auc": best_auc, "best_epoch": best_epoch}, history


# ---------------------------------------------------------------------------
# 5. Comparaison & Visualisation
# ---------------------------------------------------------------------------

def plot_comparison(results: dict[str, dict], output_dir: Path,
                    log_to_mlflow: bool = True) -> None:
    """Graphique comparatif des trois modèles (AUC, F1, mAP)."""
    models   = list(results.keys())
    metrics  = ["val_auc_macro", "val_f1_macro", "val_map"]
    labels   = ["AUC-ROC Macro", "F1 Macro", "mAP"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (m_name, color) in enumerate(zip(models, colors)):
        vals = [results[m_name]["best_val_metrics"].get(k, 0) for k in metrics]
        bars = ax.bar(x + i * width, vals, width, label=m_name, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison Image / Texte / Multimodal — OpenI POC")
    ax.legend()
    plt.tight_layout()

    path = output_dir / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if log_to_mlflow:
        mlflow.log_artifact(str(path), artifact_path="comparison")
    log.info("Graphique comparatif sauvegardé : %s", path)


def plot_training_curves(histories: dict[str, list[dict]], output_dir: Path,
                         log_to_mlflow: bool = True) -> None:
    """Courbes de loss et AUC par epoch pour les trois modèles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"ImageOnly": "#1f77b4", "TextOnly": "#ff7f0e", "Multimodal": "#2ca02c"}

    for m_name, history in histories.items():
        epochs = range(1, len(history) + 1)
        axes[0].plot(epochs, [h["val_loss"] for h in history],
                     label=m_name, color=colors[m_name])
        axes[1].plot(epochs, [h["val_auc_macro"] for h in history],
                     label=m_name, color=colors[m_name])

    axes[0].set_title("Val Loss par epoch"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss"); axes[0].legend()
    axes[1].set_title("Val AUC-ROC par epoch"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC"); axes[1].legend()

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if log_to_mlflow:
        mlflow.log_artifact(str(path), artifact_path="comparison")


# ---------------------------------------------------------------------------
# 6. Pipeline principal
# ---------------------------------------------------------------------------

def run_multimodal_poc(cfg: MultimodalConfig) -> dict:
    setup_reproducibility(cfg.seed)
    setup_mlflow(experiment_name=cfg.mlflow_experiment, tracking_uri=cfg.mlflow_uri)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("PHASE 4 — MULTIMODAL POC | device: %s", DEVICE)
    log.info("=" * 65)

    # --- Données ---
    train_loader, val_loader, test_loader, metadata, tfidf = get_openi_dataloaders(cfg)
    vocab_size = metadata["tfidf_vocab_size"]
    pos_weight = build_pos_weight_openi(train_loader)

    # --- Trois modèles ---
    model_registry = {
        "ImageOnly": ImageOnlyModel(N_CLASSES, cfg.img_embed_dim, cfg.dropout),
        "TextOnly":  TextOnlyModel(vocab_size, N_CLASSES, cfg.txt_embed_dim, cfg.dropout),
        "Multimodal": MultimodalFusionModel(
            vocab_size, N_CLASSES,
            cfg.img_embed_dim, cfg.txt_embed_dim, cfg.dropout,
        ),
    }

    all_results  = {}
    all_histories = {}

    for model_name, model in model_registry.items():
        model = model.to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("\n>>> Entraînement : %s (%s params)", model_name, f"{n_params:,}")

        with mlflow.start_run(run_name=f"{model_name}_sz{cfg.image_size}"):
            mlflow.log_params({
                **asdict(cfg),
                "model_name": model_name,
                "n_params": n_params,
                "vocab_size": vocab_size,
                "device": str(DEVICE),
            })

            summary, history = train_one_model(
                model, model_name, train_loader, val_loader, cfg, pos_weight
            )

            # Métriques finales sur val (dernière valeur = meilleur ckpt)
            best_epoch_idx = summary["best_epoch"] - 1
            best_val_metrics = history[min(best_epoch_idx, len(history) - 1)]

            for step, h in enumerate(history, 1):
                mlflow.log_metrics({k: v for k, v in h.items()
                                    if isinstance(v, float)}, step=step)

            mlflow.log_metrics({
                "best_val_auc_macro": summary["best_auc"],
                "best_epoch": summary["best_epoch"],
            })
            mlflow.pytorch.log_model(model, artifact_path=f"model_{model_name}")

            # Historique JSON
            hist_path = Path(cfg.checkpoint_dir) / f"history_{model_name}.json"
            with open(hist_path, "w") as f:
                json.dump(history, f, indent=2)
            mlflow.log_artifact(str(hist_path))

        all_results[model_name] = {**summary, "best_val_metrics": best_val_metrics}
        all_histories[model_name] = history

    # --- Visualisations comparatives ---
    output_dir = Path(cfg.output_dir)
    with mlflow.start_run(run_name="comparison_summary"):
        plot_comparison(all_results, output_dir)
        plot_training_curves(all_histories, output_dir)

        # Tableau récapitulatif
        print("\n" + "=" * 70)
        print(f"{'MODÈLE':<15} {'AUC Macro':>12} {'F1 Macro':>10} {'mAP':>10} {'Epochs':>8}")
        print("=" * 70)
        for m_name, res in all_results.items():
            vm = res["best_val_metrics"]
            print(f"{m_name:<15} {vm['val_auc_macro']:>12.4f} "
                  f"{vm['val_f1_macro']:>10.4f} {vm['val_map']:>10.4f} "
                  f"{res['best_epoch']:>8}")
        print("=" * 70 + "\n")

        summary_path = output_dir / "poc_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        mlflow.log_artifact(str(summary_path))

    log.info("POC Multimodal terminé. Outputs : %s", cfg.output_dir)
    return all_results


# ---------------------------------------------------------------------------
# 7. Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 — Multimodal POC OpenI")
    parser.add_argument("--data_root",   type=str, default="data/openi")
    parser.add_argument("--image_size",  type=int, default=64, choices=[64, 128])
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_epochs",  type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--img_embed_dim", type=int, default=256)
    parser.add_argument("--txt_embed_dim", type=int, default=256)
    parser.add_argument("--max_text_features", type=int, default=5000)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--patience",    type=int, default=5)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir",  type=str, default="multimodal_outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--mlflow_experiment", type=str, default="ChestMNIST_Multimodal_POC")
    parser.add_argument("--mlflow_uri",  type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Utilise des données synthétiques (test pipeline sans OpenI)")

    args = parser.parse_args()
    cfg = MultimodalConfig(**vars(args))
    results = run_multimodal_poc(cfg)
