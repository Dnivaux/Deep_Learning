# train_models.py
"""
Phase 2 — Modélisation Supervisée : Classification Multi-label ChestMNIST+
Architectures : SimpleCNN | ResNet50-TL | ViT-B/16
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
)
from torch.utils.data import DataLoader
from torchvision import models

from data_pipeline import (
    get_chestmnist_dataloaders,
    setup_mlflow,
    setup_reproducibility,
    CHEST_CLASSES,
    DATASET_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = len(CHEST_CLASSES)  # 14


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    architecture: str = "resnet"          # "simple_cnn" | "resnet" | "vit"
    image_size: int = 224
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    freeze_backbone: bool = False          # Transfer Learning uniquement
    dropout: float = 0.3
    seed: int = 42
    num_workers: int = 0
    data_root: str = "data"
    checkpoint_dir: str = "checkpoints"
    mlflow_experiment: str = "ChestMNIST_Phase2"
    mlflow_uri: str = "sqlite:///mlflow.db"  # DB backend (évite FutureWarning)
    patience: int = 7                      # Early stopping
    scheduler: str = "cosine"             # "cosine" | "plateau" | "none"
    download: bool = False


# ---------------------------------------------------------------------------
# 2. Architectures
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    CNN entraîné from scratch.
    4 blocs Conv-BN-ReLU-MaxPool + tête de classification.
    """

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Bloc 1 : 1→32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),             # H/2

            # Bloc 2 : 32→64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),             # H/4

            # Bloc 3 : 64→128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),             # H/8

            # Bloc 4 : 128→256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),   # → (256, 4, 4) quelle que soit l'entrée
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),      # logits bruts (pas de sigmoid)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ChestMNIST est en niveaux de gris (1 canal)
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        return self.classifier(self.features(x))


class ResNetTL(nn.Module):
    """
    ResNet-50 pré-entraîné ImageNet avec Transfer Learning.
    La tête FC est remplacée par une couche → 14 sorties.
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)

        # Adaptation canal 1→3 via conv initiale
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if freeze_backbone:
            for name, param in backbone.named_parameters():
                if "layer4" not in name and "fc" not in name:
                    param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        return self.model(x)


class ViTHybrid(nn.Module):
    """
    Vision Transformer ViT-B/16 pré-entraîné ImageNet.
    La tête MLP est remplacée pour 14 sorties.
    Compatible avec les images niveaux de gris.
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        vit = models.vit_b_16(weights=weights)

        # ViT attend 224×224 et 3 canaux — on adapte la projection d'entrée
        # patch_size=16, hidden_dim=768
        original_proj = vit.conv_proj  # Conv2d(3, 768, 16, 16)
        vit.conv_proj = nn.Conv2d(
            1, original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
        )
        # Initialisation : moyenne des 3 canaux originaux
        with torch.no_grad():
            vit.conv_proj.weight = nn.Parameter(
                original_proj.weight.mean(dim=1, keepdim=True)
            )

        if freeze_backbone:
            for name, param in vit.named_parameters():
                if "heads" not in name and "encoder.layers.11" not in name:
                    param.requires_grad = False

        in_features = vit.heads.head.in_features
        vit.heads = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_classes),
        )
        self.model = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        return self.model(x)


def build_model(cfg: TrainConfig) -> nn.Module:
    """Factory : instancie le bon modèle selon cfg.architecture."""
    arch = cfg.architecture.lower()
    if arch == "simple_cnn":
        model = SimpleCNN(dropout=cfg.dropout)
    elif arch == "resnet":
        model = ResNetTL(freeze_backbone=cfg.freeze_backbone, dropout=cfg.dropout)
    elif arch == "vit":
        model = ViTHybrid(freeze_backbone=cfg.freeze_backbone, dropout=cfg.dropout)
    else:
        raise ValueError(f"Architecture inconnue : {arch}. Choisir parmi: simple_cnn | resnet | vit")

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Modèle %s — paramètres entraînables : %s", arch, f"{n_params:,}")
    return model


# ---------------------------------------------------------------------------
# 3. Loss pondérée (gestion du déséquilibre)
# ---------------------------------------------------------------------------

def build_pos_weight(train_loader: DataLoader) -> torch.Tensor:
    """
    Calcule pos_weight[i] = n_négatifs_i / n_positifs_i pour BCEWithLogitsLoss.

    Stratégie : pour chaque classe i sous-représentée (peu de positifs),
    on amplifie le gradient des exemples positifs d'un facteur = ratio neg/pos.
    Cela évite que le modèle prédit systématiquement 0 sur les classes rares.

    pos_weight est DIFFÉRENT des class_weights de l'EDA :
    - EDA weights = 1/fréquence (normalisé, pour pondérer les classes entre elles)
    - pos_weight = neg/pos (pour rééquilibrer BCE par classe séparément)
    """
    log.info("Calcul des pos_weight sur le train set…")
    all_labels: list[np.ndarray] = []
    for _, labels in train_loader:
        all_labels.append(labels.numpy())

    label_matrix = np.concatenate(all_labels, axis=0)  # (N, 14)
    n_samples = label_matrix.shape[0]
    n_pos = label_matrix.sum(axis=0)                   # positifs par classe
    n_neg = n_samples - n_pos                          # négatifs par classe

    pos_weight = n_neg / np.clip(n_pos, 1, None)       # évite division par zéro
    pos_weight = np.clip(pos_weight, 1.0, 50.0)        # cap à 50x pour stabilité

    for i, cls in enumerate(CHEST_CLASSES):
        log.info("  %-30s pos=%5d  neg=%5d  pw=%.1f",
                 cls, int(n_pos[i]), int(n_neg[i]), pos_weight[i])

    return torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)


# ---------------------------------------------------------------------------
# 4. Optimiseur & Scheduler
# ---------------------------------------------------------------------------

def build_optimizer_scheduler(
    model: nn.Module,
    cfg: TrainConfig,
) -> tuple[optim.Optimizer, object]:
    """Construit l'optimiseur AdamW + le scheduler choisi."""
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    if cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs, eta_min=1e-6
        )
    elif cfg.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    else:
        scheduler = None

    return optimizer, scheduler


# ---------------------------------------------------------------------------
# 5. Boucles d'entraînement
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: optim.Optimizer,
    epoch: int,
) -> dict[str, float]:
    """
    Une epoch d'entraînement.
    Retourne : {loss_mean, lr}
    """
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"Train Ep{epoch:02d}", leave=False,
                unit="batch", dynamic_ncols=True)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping pour stabilité (utile surtout pour ViT)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": total_loss / n_batches, "lr": lr}


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    epoch: int,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Epoch de validation.
    Retourne : loss, AUC-ROC macro, F1 macro, mAP + AUC par classe.
    """
    model.eval()
    total_loss = 0.0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for images, labels in tqdm(loader, desc=f"Val   Ep{epoch:02d}", leave=False,
                               unit="batch", dynamic_ncols=True):
        images = images.to(DEVICE, non_blocking=True)
        labels_f = labels.float().to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels_f)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    probs_mat = np.concatenate(all_probs, axis=0)    # (N, 14)
    labels_mat = np.concatenate(all_labels, axis=0)  # (N, 14)
    preds_mat = (probs_mat >= threshold).astype(int)

    # AUC-ROC par classe (ignorer classes avec 0 positifs dans ce split)
    auc_per_class: dict[str, float] = {}
    ap_per_class: dict[str, float] = {}
    valid_aucs, valid_aps = [], []

    for i, cls in enumerate(CHEST_CLASSES):
        if labels_mat[:, i].sum() == 0:
            continue
        auc = roc_auc_score(labels_mat[:, i], probs_mat[:, i])
        ap = average_precision_score(labels_mat[:, i], probs_mat[:, i])
        auc_per_class[cls] = round(auc, 4)
        ap_per_class[cls] = round(ap, 4)
        valid_aucs.append(auc)
        valid_aps.append(ap)

    auc_macro = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    map_score = float(np.mean(valid_aps)) if valid_aps else 0.0

    # F1 macro multi-label
    f1_macro = f1_score(labels_mat, preds_mat, average="macro", zero_division=0)
    f1_micro = f1_score(labels_mat, preds_mat, average="micro", zero_division=0)

    metrics = {
        "val_loss": total_loss / len(loader),
        "val_auc_macro": round(auc_macro, 4),
        "val_map": round(map_score, 4),
        "val_f1_macro": round(f1_macro, 4),
        "val_f1_micro": round(f1_micro, 4),
    }

    log.info(
        "  [Val Epoch %d] loss=%.4f | AUC=%.4f | mAP=%.4f | F1_macro=%.4f",
        epoch, metrics["val_loss"], auc_macro, map_score, f1_macro,
    )
    return metrics, auc_per_class


# ---------------------------------------------------------------------------
# 6. Boucle principale d'entraînement
# ---------------------------------------------------------------------------

def run_training(cfg: TrainConfig) -> dict:
    """
    Orchestre l'entraînement complet d'une architecture avec tracking MLflow.
    """
    setup_reproducibility(cfg.seed)
    setup_mlflow(experiment_name=cfg.mlflow_experiment, tracking_uri=cfg.mlflow_uri)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("ENTRAÎNEMENT — architecture: %s | device: %s", cfg.architecture, DEVICE)
    log.info("=" * 60)

    # --- Chargement données ---
    train_loader, val_loader, _, metadata = get_chestmnist_dataloaders(
        size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        data_root=cfg.data_root,
        download=cfg.download,
    )

    # --- Modèle, loss, optimiseur ---
    model = build_model(cfg)
    pos_weight = build_pos_weight(train_loader)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)

    run_name = f"{cfg.architecture}_sz{cfg.image_size}_lr{cfg.learning_rate}_ep{cfg.num_epochs}"

    with mlflow.start_run(run_name=run_name):
        # Log hyperparamètres
        mlflow.log_params({**asdict(cfg), "device": str(DEVICE), "n_classes": N_CLASSES})
        mlflow.log_params({
            "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })

        best_auc = 0.0
        patience_counter = 0
        history: list[dict] = []

        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.time()

            train_metrics = train_epoch(model, train_loader, criterion, optimizer, epoch)
            val_metrics, auc_per_class = validate_epoch(model, val_loader, criterion, epoch)

            elapsed = time.time() - t0
            epoch_metrics = {**train_metrics, **val_metrics, "epoch_time_s": round(elapsed, 1)}
            history.append(epoch_metrics)

            # --- MLflow : log métriques par step ---
            mlflow.log_metrics(epoch_metrics, step=epoch)
            # Log AUC par classe (1 fois tous les 5 epochs pour ne pas surcharger)
            if epoch % 5 == 0:
                mlflow.log_metrics(
                    {f"auc_{cls.replace(' ', '_')}": v for cls, v in auc_per_class.items()},
                    step=epoch,
                )

            # --- Scheduler ---
            if scheduler is not None:
                if cfg.scheduler == "plateau":
                    scheduler.step(val_metrics["val_auc_macro"])
                else:
                    scheduler.step()

            # --- Early stopping & checkpoint ---
            current_auc = val_metrics["val_auc_macro"]
            if current_auc > best_auc:
                best_auc = current_auc
                patience_counter = 0
                ckpt_path = Path(cfg.checkpoint_dir) / f"best_{cfg.architecture}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "config": asdict(cfg),
                }, ckpt_path)
                log.info("  ✓ Nouveau meilleur AUC: %.4f — checkpoint sauvegardé", best_auc)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    log.info("Early stopping déclenché à l'epoch %d (patience=%d)", epoch, cfg.patience)
                    break

        # --- Log du meilleur modèle dans MLflow ---
        best_ckpt = torch.load(
            Path(cfg.checkpoint_dir) / f"best_{cfg.architecture}.pt",
            map_location=DEVICE,
            weights_only=True,
        )
        model.load_state_dict(best_ckpt["model_state_dict"])
        mlflow.pytorch.log_model(model, artifact_path=f"model_{cfg.architecture}")

        # Métriques finales
        mlflow.log_metrics({
            "best_val_auc_macro": best_auc,
            "best_epoch": best_ckpt["epoch"],
        })

        # Historique JSON
        history_path = Path(cfg.checkpoint_dir) / f"history_{cfg.architecture}.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        mlflow.log_artifact(str(history_path))

        log.info("Entraînement terminé. Meilleur AUC-ROC macro : %.4f", best_auc)

    return {"best_auc": best_auc, "history": history}


# ---------------------------------------------------------------------------
# 7. Argparse & Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Phase 2 — Entraînement ChestMNIST+")
    parser.add_argument("--architecture", type=str, default="resnet",
                        choices=["simple_cnn", "resnet", "vit"],
                        help="Architecture à entraîner")
    parser.add_argument("--image_size", type=int, default=224, choices=[64, 128, 224])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Geler le backbone (TL uniquement)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"])
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--mlflow_experiment", type=str, default="ChestMNIST_Phase2")
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--download", action="store_true")

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    results = run_training(cfg)