# data_pipeline.py
"""
Phase 1 — Data Pipeline & EDA
Système d'aide au tri radiologique — ChestMNIST+
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader

import medmnist
from medmnist import ChestMNIST, INFO
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME = "chestmnist"
VALID_SIZES = (28, 64, 128, 224)
CHEST_CLASSES: list[str] = list(INFO[DATASET_NAME]["label"].values())  # 14 pathologies


# ---------------------------------------------------------------------------
# 1. Reproductibilité
# ---------------------------------------------------------------------------

def setup_reproducibility(seed: int = 42) -> None:
    """Fixe toutes les sources d'aléatoire pour garantir la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info("Seed fixée à %d sur random / numpy / torch / cuda.", seed)


# ---------------------------------------------------------------------------
# 2. MLflow
# ---------------------------------------------------------------------------

def setup_mlflow(
    experiment_name: str = "ChestMNIST_Pipeline",
    tracking_uri: str = "mlruns",
) -> str:
    """
    Configure MLflow en mode local.

    Returns:
        run_id du run actif.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    log.info("MLflow configuré — URI : %s / Expérience : %s", tracking_uri, experiment_name)
    return tracking_uri


# ---------------------------------------------------------------------------
# 3. Transforms
# ---------------------------------------------------------------------------

def _compute_train_stats(
    dataset: ChestMNIST,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Calcule mean et std pixel sur le train set UNIQUEMENT
    (évite la fuite statistique vers val/test).
    """
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    all_pixels: list[torch.Tensor] = []
    for imgs, _ in loader:
        # imgs shape: (B, C, H, W) — float après ToTensor
        all_pixels.append(imgs.view(imgs.size(0), imgs.size(1), -1))
    concat = torch.cat(all_pixels, dim=0)  # (N, C, H*W)
    mean = concat.mean(dim=(0, 2)).tolist()
    std = concat.std(dim=(0, 2)).tolist()
    log.info("Stats train — mean: %s | std: %s", mean, std)
    return tuple(mean), tuple(std)


def get_transforms(
    split: str,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    image_size: int = 224,
) -> transforms.Compose:
    """
    Retourne les transformations PyTorch selon le split.

    - Train : augmentation + normalisation.
    - Val / Test : redimensionnement + normalisation uniquement.
    """
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ]

    if split == "train":
        augment = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
        return transforms.Compose(augment)

    return transforms.Compose(base)


# ---------------------------------------------------------------------------
# 4. Chargement des données
# ---------------------------------------------------------------------------

def get_chestmnist_dataloaders(
    size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    data_root: str = "data",
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Télécharge ChestMNIST+ et retourne les DataLoaders Train / Val / Test.

    Les splits sont ceux fournis officiellement par MedMNIST pour éviter
    tout data leakage. Aucune re-séparation manuelle n'est effectuée.

    Args:
        size:        Résolution des images (64, 128 ou 224).
        batch_size:  Taille de batch.
        num_workers: Workers pour le DataLoader.
        seed:        Seed du générateur de shuffle (reproductibilité).
        data_root:   Répertoire local de stockage.
        download:    Si True, télécharge les données. Si False, charge depuis le cache local.

    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    if size not in VALID_SIZES:
        raise ValueError(f"size doit être dans {VALID_SIZES}, reçu : {size}")

    Path(data_root).mkdir(parents=True, exist_ok=True)

    # Chargement minimal sans transform pour calculer les stats sur le train
    log.info("Téléchargement / chargement de ChestMNIST+ (size=%d)…", size)
    train_raw = ChestMNIST(
        split="train",
        transform=transforms.ToTensor(),
        download=download,
        size=size,
        root=data_root,
    )
    mean, std = _compute_train_stats(train_raw)

    # Rechargement avec transforms définitifs
    train_dataset = ChestMNIST(
        split="train",
        transform=get_transforms("train", mean, std, size),
        download=False,
        size=size,
        root=data_root,
    )
    val_dataset = ChestMNIST(
        split="val",
        transform=get_transforms("val", mean, std, size),
        download=False,
        size=size,
        root=data_root,
    )
    test_dataset = ChestMNIST(
        split="test",
        transform=get_transforms("test", mean, std, size),
        download=False,
        size=size,
        root=data_root,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    metadata = {
        "dataset": DATASET_NAME,
        "image_size": size,
        "n_classes": len(CHEST_CLASSES),
        "task": INFO[DATASET_NAME]["task"],
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "mean": list(mean),
        "std": list(std),
        "batch_size": batch_size,
        "seed": seed,
    }

    log.info(
        "Datasets — Train: %d | Val: %d | Test: %d",
        metadata["n_train"], metadata["n_val"], metadata["n_test"],
    )
    return train_loader, val_loader, test_loader, metadata


# ---------------------------------------------------------------------------
# 5. EDA
# ---------------------------------------------------------------------------

def perform_eda(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: str = "eda_outputs",
    log_to_mlflow: bool = True,
) -> dict:
    """
    Analyse Exploratoire des Données sur ChestMNIST+.

    Calcule :
    - Distribution des labels (multi-label) par split.
    - Taux de positifs par pathologie (déséquilibre).
    - Class weights pour une future BCEWithLogitsLoss pondérée.
    - Sauvegarde les graphiques et loggue dans MLflow.

    Returns:
        Dictionnaire de métriques EDA.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    splits = {"train": train_loader, "val": val_loader, "test": test_loader}
    eda_metrics: dict = {}

    for split_name, loader in splits.items():
        log.info("EDA — analyse du split '%s'…", split_name)
        all_labels: list[np.ndarray] = []

        for _, labels in loader:
            all_labels.append(labels.numpy())

        label_matrix = np.concatenate(all_labels, axis=0)  # (N, 14)
        n_samples = label_matrix.shape[0]

        # Fréquence positive par classe
        pos_rate = label_matrix.mean(axis=0)          # (14,)
        pos_count = label_matrix.sum(axis=0).astype(int)

        # Class weights : inverse de la fréquence (clip pour éviter ±inf)
        class_weights = 1.0 / np.clip(pos_rate, 1e-6, 1.0)
        class_weights /= class_weights.sum()          # normalisation

        # Ratio de déséquilibre max / min
        imbalance_ratio = float(pos_rate.max() / np.clip(pos_rate.min(), 1e-6, 1.0))

        split_metrics = {
            "n_samples": int(n_samples),
            "positive_rate_per_class": {
                cls: float(round(pos_rate[i], 4))
                for i, cls in enumerate(CHEST_CLASSES)
            },
            "positive_count_per_class": {
                cls: int(pos_count[i])
                for i, cls in enumerate(CHEST_CLASSES)
            },
            "class_weights": {
                cls: float(round(class_weights[i], 4))
                for i, cls in enumerate(CHEST_CLASSES)
            },
            "imbalance_ratio": round(imbalance_ratio, 2),
            "mean_labels_per_sample": float(round(label_matrix.sum(axis=1).mean(), 3)),
        }
        eda_metrics[split_name] = split_metrics

        # --- Graphique distribution ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Taux positifs
        ax = axes[0]
        colors = ["#d62728" if r > 0.3 else "#1f77b4" for r in pos_rate]
        ax.barh(CHEST_CLASSES, pos_rate, color=colors)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.6, label="50 %")
        ax.set_xlabel("Taux de positifs")
        ax.set_title(f"[{split_name.upper()}] Distribution des pathologies (N={n_samples})")
        ax.legend()

        # Class weights
        ax2 = axes[1]
        ax2.barh(CHEST_CLASSES, class_weights, color="#2ca02c")
        ax2.set_xlabel("Poids de classe (normalisé)")
        ax2.set_title(f"[{split_name.upper()}] Class weights — ratio déséquilibre : {imbalance_ratio:.1f}x")

        plt.tight_layout()
        fig_path = Path(output_dir) / f"eda_{split_name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Graphique sauvegardé : %s", fig_path)

        if log_to_mlflow:
            mlflow.log_artifact(str(fig_path), artifact_path="eda")

    # Sauvegarde JSON
    json_path = Path(output_dir) / "eda_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eda_metrics, f, indent=2, ensure_ascii=False)

    if log_to_mlflow:
        mlflow.log_artifact(str(json_path), artifact_path="eda")
        # Log scalaires clés dans MLflow
        for split_name, metrics in eda_metrics.items():
            mlflow.log_metric(f"{split_name}_n_samples", metrics["n_samples"])
            mlflow.log_metric(f"{split_name}_imbalance_ratio", metrics["imbalance_ratio"])

    log.info("EDA terminée. Métriques : %s", json_path)
    _print_eda_summary(eda_metrics)
    return eda_metrics


def _print_eda_summary(eda_metrics: dict) -> None:
    """Affiche un résumé lisible dans la console."""
    print("\n" + "=" * 60)
    print("RÉSUMÉ EDA — ChestMNIST+")
    print("=" * 60)
    for split, m in eda_metrics.items():
        print(f"\n[{split.upper()}] N={m['n_samples']} | "
              f"Déséquilibre max/min: {m['imbalance_ratio']}x | "
              f"Labels/sample (moy): {m['mean_labels_per_sample']}")
        print(f"  {'Pathologie':<30} {'Taux pos':>10} {'Poids classe':>14}")
        print(f"  {'-'*56}")
        for cls in CHEST_CLASSES:
            rate = m["positive_rate_per_class"][cls]
            weight = m["class_weights"][cls]
            flag = " ⚠ déséquilibré" if rate < 0.05 else ""
            print(f"  {cls:<30} {rate:>10.3f} {weight:>14.4f}{flag}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 6. Point d'entrée principal
# ---------------------------------------------------------------------------

def run_pipeline(
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    data_root: str = "data",
    output_dir: str = "eda_outputs",
    mlflow_experiment: str = "ChestMNIST_Pipeline",
    mlflow_uri: str = "mlruns",
    download: bool = False,
) -> dict:
    """
    Exécute le pipeline complet Phase 1 :
    reproductibilité → MLflow → chargement → EDA.
    """
    setup_reproducibility(seed)
    setup_mlflow(experiment_name=mlflow_experiment, tracking_uri=mlflow_uri)

    with mlflow.start_run(run_name=f"pipeline_size{image_size}_seed{seed}"):
        # Log des hyperparamètres pipeline
        mlflow.log_params({
            "image_size": image_size,
            "batch_size": batch_size,
            "seed": seed,
            "dataset": DATASET_NAME,
        })

        train_loader, val_loader, test_loader, metadata = get_chestmnist_dataloaders(
            size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            data_root=data_root,
            download=download,
        )

        # Log métadonnées dataset
        mlflow.log_params({
            "n_train": metadata["n_train"],
            "n_val": metadata["n_val"],
            "n_test": metadata["n_test"],
            "n_classes": metadata["n_classes"],
        })

        eda_metrics = perform_eda(
            train_loader, val_loader, test_loader,
            output_dir=output_dir,
            log_to_mlflow=True,
        )

    return {"metadata": metadata, "eda": eda_metrics}


if __name__ == "__main__":
    results = run_pipeline(
        image_size=224,
        batch_size=32,
        num_workers=0,   # 0 pour Windows (évite les erreurs de fork)
        seed=42,
    )