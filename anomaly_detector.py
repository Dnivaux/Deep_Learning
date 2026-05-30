# anomaly_detector.py
"""
Étape 3 — Détection d'anomalies non-supervisée
Autoencodeur convolutif sur ChestMNIST+
Score d'anomalie = MSE(image_originale, image_reconstruite)
"""

from __future__ import annotations

import json
import logging
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline import (
    get_chestmnist_dataloaders,
    setup_mlflow,
    setup_reproducibility,
    CHEST_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

@dataclass
class AEConfig:
    image_size: int = 64
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    latent_dim: int = 128
    seed: int = 42
    num_workers: int = 0
    data_root: str = "data"
    output_dir: str = "ae_outputs"
    checkpoint_dir: str = "checkpoints"
    mlflow_experiment: str = "ChestMNIST_AnomalyDetection"
    mlflow_uri: str = "sqlite:///mlflow.db"
    patience: int = 5
    top_k_anomalies: int = 5
    download: bool = False
    # Si True : entraîne uniquement sur les images "normales" (aucune pathologie)
    normal_only: bool = False


# ---------------------------------------------------------------------------
# 2. Architecture Autoencodeur
# ---------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    """
    Encoder convolutif : image → vecteur latent.
    Taille image : (1, H, W) → (latent_dim,)
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Bloc 1 : 1 → 32, H/2
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloc 2 : 32 → 64, H/4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloc 3 : 64 → 128, H/8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloc 4 : 128 → 256, H/16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Pour image 64×64 : feature map = (256, 4, 4)
        # Pour image 128×128 : feature map = (256, 8, 8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        h = self.conv_blocks(x)
        h = self.adaptive_pool(h)
        h = h.flatten(1)
        return self.fc(h)


class ConvDecoder(nn.Module):
    """
    Decoder convolutif : vecteur latent → image reconstruite.
    (latent_dim,) → (1, H, W)
    """

    def __init__(self, latent_dim: int = 128, image_size: int = 64) -> None:
        super().__init__()
        self.image_size = image_size
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv_blocks = nn.Sequential(
            # Bloc 1 : 256 → 128, ×2
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Bloc 2 : 128 → 64, ×2
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Bloc 3 : 64 → 32, ×2
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Bloc 4 : 32 → 1, ×2  (retour à H×W)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # sortie en [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.deconv_blocks(h)


class ConvAutoencoder(nn.Module):
    """Autoencodeur complet Encoder + Decoder."""

    def __init__(self, latent_dim: int = 128, image_size: int = 64) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim, image_size)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne (reconstruction, code_latent)."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ---------------------------------------------------------------------------
# 3. Filtrage images "normales"
# ---------------------------------------------------------------------------

def _filter_normal_samples(dataset) -> torch.utils.data.Subset:
    """
    Retourne un Subset contenant uniquement les images sans aucune pathologie
    (tous les labels à 0). Utile pour entraîner l'AE sur distribution normale.
    """
    normal_indices = [
        i for i, (_, label) in enumerate(dataset)
        if label.sum() == 0
    ]
    log.info(
        "Filtrage normal_only : %d/%d images sans pathologie (%.1f%%)",
        len(normal_indices), len(dataset),
        100 * len(normal_indices) / len(dataset),
    )
    return torch.utils.data.Subset(dataset, normal_indices)


# ---------------------------------------------------------------------------
# 4. Boucles entraînement
# ---------------------------------------------------------------------------

def train_epoch_ae(
    model: ConvAutoencoder,
    loader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: optim.Optimizer,
    epoch: int,
) -> float:
    """Une epoch d'entraînement. Retourne la loss moyenne."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Train AE Ep{epoch:02d}", leave=False,
                unit="batch", dynamic_ncols=True)
    for images, _ in pbar:
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        images = images.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        reconstructed, _ = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate_epoch_ae(
    model: ConvAutoencoder,
    loader: DataLoader,
    criterion: nn.MSELoss,
    epoch: int,
) -> float:
    """Validation — retourne la MSE moyenne."""
    model.eval()
    total_loss = 0.0

    for images, _ in tqdm(loader, desc=f"Val   AE Ep{epoch:02d}", leave=False,
                          unit="batch", dynamic_ncols=True):
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        images = images.to(DEVICE, non_blocking=True)
        reconstructed, _ = model(images)
        total_loss += criterion(reconstructed, images).item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# 5. Calcul des scores d'anomalie sur le Test set
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_anomaly_scores(
    model: ConvAutoencoder,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule le score d'anomalie (MSE par image) sur le loader fourni.

    Returns:
        scores      : (N,) MSE par image
        originals   : (N, 1, H, W) images originales dénormalisées
        reconstructs: (N, 1, H, W) images reconstruites
        labels      : (N, 14) labels multi-label
    """
    model.eval()
    all_scores, all_orig, all_recon, all_labels = [], [], [], []

    for images, labels in tqdm(loader, desc="Scoring anomalies", dynamic_ncols=True):
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        images = images.to(DEVICE)
        recon, _ = model(images)

        # MSE par image : mean sur (C, H, W)
        mse = ((images - recon) ** 2).mean(dim=(1, 2, 3))

        all_scores.append(mse.cpu().numpy())
        all_orig.append(images.cpu().numpy())
        all_recon.append(recon.cpu().numpy())
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_scores),
        np.concatenate(all_orig),
        np.concatenate(all_recon),
        np.concatenate(all_labels),
    )


# ---------------------------------------------------------------------------
# 6. Visualisations
# ---------------------------------------------------------------------------

def plot_reconstruction_distribution(
    scores: np.ndarray,
    output_dir: Path,
    log_to_mlflow: bool = True,
) -> None:
    """Histogramme de la distribution des scores d'anomalie."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=100, color="#1f77b4", alpha=0.8, edgecolor="none")
    ax.axvline(np.percentile(scores, 95), color="orange", linestyle="--",
               label=f"P95 = {np.percentile(scores, 95):.5f}")
    ax.axvline(np.percentile(scores, 99), color="red", linestyle="--",
               label=f"P99 = {np.percentile(scores, 99):.5f}")
    ax.set_xlabel("Score d'anomalie (MSE reconstruction)")
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution des scores d'anomalie — Test set")
    ax.legend()
    plt.tight_layout()

    path = output_dir / "anomaly_score_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if log_to_mlflow:
        mlflow.log_artifact(str(path), artifact_path="anomaly")
    log.info("Distribution sauvegardée : %s", path)


def plot_top_anomalies(
    scores: np.ndarray,
    originals: np.ndarray,
    reconstructs: np.ndarray,
    labels: np.ndarray,
    top_k: int,
    output_dir: Path,
    log_to_mlflow: bool = True,
) -> None:
    """
    Sauvegarde les top_k images les plus anormales :
    colonne gauche = originale, colonne droite = reconstruction.
    """
    top_indices = np.argsort(scores)[::-1][:top_k]

    fig, axes = plt.subplots(top_k, 2, figsize=(6, top_k * 3))
    fig.suptitle(f"Top {top_k} anomalies (MSE la plus haute)", fontsize=14)

    for row, idx in enumerate(top_indices):
        orig = originals[idx, 0]
        recon = reconstructs[idx, 0]
        pathologies = [CHEST_CLASSES[i] for i, v in enumerate(labels[idx]) if v == 1]
        label_str = ", ".join(pathologies) if pathologies else "Aucune"

        axes[row, 0].imshow(orig, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Originale | MSE={scores[idx]:.5f}\n{label_str}",
                                fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(recon, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Reconstruite", fontsize=8)
        axes[row, 1].axis("off")

    plt.tight_layout()
    path = output_dir / "top_anomalies.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if log_to_mlflow:
        mlflow.log_artifact(str(path), artifact_path="anomaly")
    log.info("Top anomalies sauvegardées : %s", path)


def plot_random_reconstructions(
    originals: np.ndarray,
    reconstructs: np.ndarray,
    output_dir: Path,
    n: int = 8,
    log_to_mlflow: bool = True,
) -> None:
    """Grille de reconstructions aléatoires pour évaluation visuelle."""
    indices = np.random.choice(len(originals), size=n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))
    fig.suptitle("Reconstructions aléatoires (haut = original, bas = reconstruction)")

    for col, idx in enumerate(indices):
        axes[0, col].imshow(originals[idx, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, col].axis("off")
        axes[1, col].imshow(reconstructs[idx, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, col].axis("off")

    plt.tight_layout()
    path = output_dir / "random_reconstructions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if log_to_mlflow:
        mlflow.log_artifact(str(path), artifact_path="anomaly")
    log.info("Reconstructions aléatoires sauvegardées : %s", path)


# ---------------------------------------------------------------------------
# 7. Pipeline principal
# ---------------------------------------------------------------------------

def run_anomaly_detection(cfg: AEConfig) -> dict:
    setup_reproducibility(cfg.seed)
    setup_mlflow(experiment_name=cfg.mlflow_experiment, tracking_uri=cfg.mlflow_uri)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("AUTOENCODEUR — device: %s | latent_dim: %d", DEVICE, cfg.latent_dim)
    log.info("=" * 60)

    # --- Chargement données ---
    train_loader, val_loader, test_loader, metadata = get_chestmnist_dataloaders(
        size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        data_root=cfg.data_root,
        download=cfg.download,
    )

    # Filtrage optionnel images normales
    if cfg.normal_only:
        train_dataset = train_loader.dataset
        normal_dataset = _filter_normal_samples(train_dataset)
        train_loader = DataLoader(
            normal_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )

    # --- Modèle ---
    model = ConvAutoencoder(latent_dim=cfg.latent_dim, image_size=cfg.image_size)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Autoencodeur : %s paramètres", f"{n_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    run_name = f"AE_sz{cfg.image_size}_ldim{cfg.latent_dim}_ep{cfg.num_epochs}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**asdict(cfg), "device": str(DEVICE), "n_params": n_params})

        best_val_loss = float("inf")
        patience_counter = 0
        history = []

        # --- Entraînement ---
        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.time()
            train_loss = train_epoch_ae(model, train_loader, criterion, optimizer, epoch)
            val_loss = validate_epoch_ae(model, val_loader, criterion, epoch)
            elapsed = time.time() - t0

            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]

            epoch_metrics = {
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": lr,
                "epoch_time_s": round(elapsed, 1),
            }
            history.append(epoch_metrics)
            mlflow.log_metrics(epoch_metrics, step=epoch)

            log.info(
                "Epoch %d/%d — train_loss=%.6f | val_loss=%.6f | lr=%.2e | %.1fs",
                epoch, cfg.num_epochs, train_loss, val_loss, lr, elapsed,
            )

            # Checkpoint meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_path = Path(cfg.checkpoint_dir) / "best_autoencoder.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": asdict(cfg),
                }, ckpt_path)
                log.info("  ✓ Meilleure val_loss: %.6f — checkpoint sauvegardé", best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    log.info("Early stopping à l'epoch %d", epoch)
                    break

        # --- Chargement meilleur modèle ---
        best_ckpt = torch.load(
            Path(cfg.checkpoint_dir) / "best_autoencoder.pt",
            map_location=DEVICE,
            weights_only=True,
        )
        model.load_state_dict(best_ckpt["model_state_dict"])

        # --- Scoring anomalies sur Test set ---
        log.info("Calcul des scores d'anomalie sur le Test set…")
        scores, originals, reconstructs, test_labels = compute_anomaly_scores(
            model, test_loader
        )

        # Statistiques
        anomaly_stats = {
            "mean_mse": float(np.mean(scores)),
            "std_mse": float(np.std(scores)),
            "p50_mse": float(np.percentile(scores, 50)),
            "p95_mse": float(np.percentile(scores, 95)),
            "p99_mse": float(np.percentile(scores, 99)),
            "best_val_loss": best_val_loss,
        }
        mlflow.log_metrics(anomaly_stats)
        log.info("Stats anomalie test — mean=%.6f | p95=%.6f | p99=%.6f",
                 anomaly_stats["mean_mse"], anomaly_stats["p95_mse"], anomaly_stats["p99_mse"])

        # --- Visualisations ---
        plot_reconstruction_distribution(scores, output_dir)
        plot_top_anomalies(scores, originals, reconstructs, test_labels,
                           cfg.top_k_anomalies, output_dir)
        plot_random_reconstructions(originals, reconstructs, output_dir)

        # Sauvegarde scores JSON
        scores_path = output_dir / "anomaly_scores.json"
        top_k_idx = np.argsort(scores)[::-1][:cfg.top_k_anomalies].tolist()
        with open(scores_path, "w") as f:
            json.dump({
                "stats": anomaly_stats,
                "top_k_indices": top_k_idx,
                "top_k_scores": [float(scores[i]) for i in top_k_idx],
                "top_k_labels": [
                    [CHEST_CLASSES[j] for j, v in enumerate(test_labels[i]) if v == 1]
                    for i in top_k_idx
                ],
            }, f, indent=2)
        mlflow.log_artifact(str(scores_path), artifact_path="anomaly")

        # Log modèle MLflow
        mlflow.pytorch.log_model(model, artifact_path="model_autoencoder")

        # Historique
        history_path = Path(cfg.checkpoint_dir) / "history_ae.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        mlflow.log_artifact(str(history_path))

    log.info("Détection d'anomalies terminée. Outputs : %s", output_dir)
    return {"anomaly_stats": anomaly_stats, "scores": scores}


# ---------------------------------------------------------------------------
# 8. Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Étape 3 — Autoencodeur ChestMNIST+")
    parser.add_argument("--image_size", type=int, default=64, choices=[64, 128, 224])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="ae_outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--mlflow_experiment", type=str, default="ChestMNIST_AnomalyDetection")
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--top_k_anomalies", type=int, default=5)
    parser.add_argument("--normal_only", action="store_true",
                        help="Entraîne uniquement sur les images sans pathologie")
    parser.add_argument("--download", action="store_true")

    args = parser.parse_args()
    cfg = AEConfig(**vars(args))
    results = run_anomaly_detection(cfg)
