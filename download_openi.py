# download_openi.py
"""
Téléchargement et parsing du dataset OpenI (NIH NLM)
Images : https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
Rapports XML : https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tar.gz

Produit : data/openi/records.json avec paires (image_path, text, labels)
Usage   : python download_openi.py
"""

from __future__ import annotations

import json
import logging
import tarfile
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs officielles NIH
# ---------------------------------------------------------------------------
URLS = {
    "images_png": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
    "reports_xml_candidates": [
        "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz",
        "https://openi.nlm.nih.gov/imgs/collections/ecgen-radiology.tar.gz",
        "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_xml.tgz",
    ],
}

DATA_ROOT    = Path("data/openi")
IMAGES_DIR   = DATA_ROOT / "images"
REPORTS_DIR  = DATA_ROOT / "reports"
RECORDS_PATH = DATA_ROOT / "records.json"

# Labels à extraire des MeSH tags OpenI (subset des pathologies communes)
OPENI_LABEL_MAP = {
    "Atelectasis":        "Atelectasis",
    "Cardiomegaly":       "Cardiomegaly",
    "Effusion":           "Effusion",
    "Infiltrate":         "Infiltration",
    "Infiltration":       "Infiltration",
    "Mass":               "Mass",
    "Nodule":             "Nodule",
    "Pneumonia":          "Pneumonia",
    "Pneumothorax":       "Pneumothorax",
    "Consolidation":      "Consolidation",
    "Edema":              "Edema",
    "Emphysema":          "Emphysema",
    "Fibrosis":           "Fibrosis",
    "Pleural Thickening": "Pleural Thickening",
    "Hernia":             "Hernia",
}
LABEL_NAMES = sorted(set(OPENI_LABEL_MAP.values()))
N_CLASSES   = len(LABEL_NAMES)


# ---------------------------------------------------------------------------
# Utilitaires de téléchargement
# ---------------------------------------------------------------------------

class _TqdmProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download(url: str, dest: Path) -> Path:
    """Télécharge url → dest avec barre de progression."""
    dest.mkdir(parents=True, exist_ok=True)   # crée dest lui-même (pas juste parent)
    filename = dest / Path(url).name
    if filename.exists():
        log.info("Déjà téléchargé : %s", filename)
        return filename

    log.info("Téléchargement : %s", url)
    with _TqdmProgress(unit="B", unit_scale=True, miniters=1,
                       desc=Path(url).name) as t:
        urllib.request.urlretrieve(url, filename, reporthook=t.update_to)
    return filename


def _extract(archive: Path, dest: Path) -> None:
    """Extrait une archive .tgz/.tar.gz vers dest."""
    if dest.exists() and any(dest.iterdir()):
        log.info("Déjà extrait : %s", dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Extraction : %s → %s", archive, dest)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in tqdm(members, desc="Extraction", unit="fichier"):
            tar.extract(m, path=dest)


# ---------------------------------------------------------------------------
# Parsing des rapports XML OpenI
# ---------------------------------------------------------------------------

def _parse_xml_report(xml_path: Path) -> dict | None:
    """
    Parse un rapport XML OpenI.

    Structure XML attendue :
        <eCitation>
          <uId id="CXR123"/>
          <MeSH><major>Atelectasis</major>...</MeSH>
          <AbstractText Label="FINDINGS">...</AbstractText>
          <AbstractText Label="IMPRESSION">...</AbstractText>
        </eCitation>

    Retourne un dict ou None si le fichier est invalide.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    # ID du cas (correspond au nom des images PNG)
    uid_el = root.find(".//uId")
    case_id = uid_el.get("id") if uid_el is not None else xml_path.stem

    # Texte : FINDINGS + IMPRESSION
    findings   = ""
    impression = ""
    for el in root.findall(".//AbstractText"):
        label = el.get("Label", "").upper()
        text  = (el.text or "").strip()
        if label == "FINDINGS":
            findings = text
        elif label == "IMPRESSION":
            impression = text

    full_text = " ".join(filter(None, [findings, impression])).strip()
    if not full_text:
        return None

    # Labels depuis les MeSH tags
    label_vec = [0] * N_CLASSES
    for mesh_el in root.findall(".//MeSH/major"):
        raw_lbl = (mesh_el.text or "").strip()
        # Normalisation : supprime les suffixes "/diagnosis" etc.
        raw_lbl = raw_lbl.split("/")[0].strip()
        mapped  = OPENI_LABEL_MAP.get(raw_lbl)
        if mapped and mapped in LABEL_NAMES:
            label_vec[LABEL_NAMES.index(mapped)] = 1

    return {
        "case_id":  case_id,
        "text":     full_text,
        "labels":   label_vec,
        "findings": findings,
        "impression": impression,
    }


# ---------------------------------------------------------------------------
# Construction du records.json
# ---------------------------------------------------------------------------

def build_records(
    images_dir: Path,
    reports_dir: Path,
    records_path: Path,
    min_img_std: float = 5.0,
) -> list[dict]:
    """
    Associe chaque rapport XML à ses images PNG.
    Un rapport peut correspondre à plusieurs images (frontal + latéral).
    """
    # Recherche récursive des XML
    xml_files = list(reports_dir.rglob("*.xml"))
    log.info("Rapports XML trouvés : %d", len(xml_files))

    # Index images par case_id
    # Nommage OpenI : CXR<id>_IM-<num>-<view>.png
    img_index: dict[str, list[Path]] = {}
    for img_path in images_dir.rglob("*.png"):
        # Extrait l'ID du cas depuis le nom de fichier
        parts = img_path.stem.split("_")
        case_id = parts[0]  # ex: "CXR123"
        img_index.setdefault(case_id, []).append(img_path)

    log.info("Images PNG indexées : %d cas distincts", len(img_index))

    records  = []
    n_no_img = 0
    n_noise  = 0

    for xml_path in tqdm(xml_files, desc="Parsing rapports", unit="xml"):
        parsed = _parse_xml_report(xml_path)
        if parsed is None:
            continue

        case_id = parsed["case_id"]
        img_paths = img_index.get(case_id, [])

        if not img_paths:
            n_no_img += 1
            continue

        for img_path in img_paths:
            # Vérification anti-bruit
            try:
                img_np = np.array(Image.open(img_path).convert("L"))
            except Exception:
                continue

            if img_np.std() < min_img_std:
                n_noise += 1
                continue

            records.append({
                "text":       parsed["text"],
                "image_path": str(img_path),
                "labels":     parsed["labels"],
                "case_id":    case_id,
                "source":     "openi_nlm",
            })

    log.info("Rapports sans image : %d | Images rejetées (bruit) : %d", n_no_img, n_noise)

    # Diagnostic de distribution
    if records:
        lbl_mat = np.array([r["labels"] for r in records])
        n_with  = sum(1 for r in records if sum(r["labels"]) > 0)
        log.info("Records valides : %d | Avec pathologie : %d (%.1f%%)",
                 len(records), n_with, 100 * n_with / len(records))
        log.info("Positifs par classe :")
        for i, lbl in enumerate(LABEL_NAMES):
            n = int(lbl_mat[:, i].sum())
            log.info("  %-25s : %4d (%.1f%%)", lbl, n, 100 * n / len(records))

        records_path.parent.mkdir(parents=True, exist_ok=True)
        with open(records_path, "w", encoding="utf-8") as f:
            json.dump({"label_names": LABEL_NAMES, "records": records}, f, indent=2)
        log.info("records.json sauvegardé : %s", records_path)
    else:
        log.error("Aucun record valide ! Vérifiez les chemins et le format des fichiers.")

    return records


# ---------------------------------------------------------------------------
# Vérification visuelle
# ---------------------------------------------------------------------------

def verify_sample(records: list[dict], n: int = 3) -> None:
    """Affiche les stats des n premières images pour vérifier qu'elles ne sont pas du bruit."""
    log.info("\n=== Vérification des %d premières images ===", n)
    for i, rec in enumerate(records[:n]):
        img = np.array(Image.open(rec["image_path"]).convert("L"))
        active = [LABEL_NAMES[j] for j, v in enumerate(rec["labels"]) if v == 1]
        log.info(
            "Image %d — mean=%.1f | std=%.1f | min=%d | max=%d",
            i, img.mean(), img.std(), img.min(), img.max(),
        )
        log.info("  Texte  : %s...", rec["text"][:80])
        log.info("  Labels : %s", active if active else ["Normal"])


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Téléchargement dataset OpenI NIH")
    parser.add_argument("--data_root", default="data/openi")
    parser.add_argument("--skip_images", action="store_true",
                        help="Sauter le téléchargement des images PNG (déjà fait)")
    parser.add_argument("--skip_reports", action="store_true",
                        help="Sauter le téléchargement des rapports XML (déjà fait)")
    args = parser.parse_args()

    DATA_ROOT    = Path(args.data_root)
    IMAGES_DIR   = DATA_ROOT / "openi_images"   # séparé des images ChestMNIST
    REPORTS_DIR  = DATA_ROOT / "reports"
    archives_dir = DATA_ROOT / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    # 1. Images PNG
    archive_img = archives_dir / "NLMCXR_png.tgz"
    if not args.skip_images:
        archive_img = _download(URLS["images_png"], archives_dir)
    else:
        log.info("Téléchargement images ignoré (--skip_images)")

    # 2. Rapports XML — toujours tenté sauf si --skip_reports
    archive_xml = None

    # Cherche d'abord une archive XML déjà présente localement
    for xml_path in archives_dir.glob("*.tgz"):
        if any(k in xml_path.name.lower() for k in ("report", "xml", "ecgen", "radiology")):
            archive_xml = xml_path
            log.info("Archive XML trouvée localement : %s", archive_xml)
            break

    if archive_xml is None and not args.skip_reports:
        for url in URLS["reports_xml_candidates"]:
            try:
                log.info("Tentative téléchargement XML : %s", url)
                archive_xml = _download(url, archives_dir)
                log.info("✅ Rapports XML téléchargés : %s", archive_xml.name)
                break
            except Exception as e:
                log.warning("  Échec : %s", e)

    if archive_xml is None:
        log.error(
            "Rapports XML introuvables. Télécharge manuellement :\n"
            "  %s\net place le fichier dans : %s",
            URLS["reports_xml_candidates"][0], archives_dir,
        )

    # 3. Extraction images
    if archive_img.exists():
        _extract(archive_img, IMAGES_DIR)
    else:
        log.error("Archive images introuvable : %s", archive_img)

    # Diagnostic : affiche la structure extraite
    png_files = list(IMAGES_DIR.rglob("*.png"))
    log.info("Images PNG extraites : %d fichiers (recherche récursive)", len(png_files))
    if png_files:
        log.info("Exemple de chemin : %s", png_files[0])

    # 4. Extraction rapports
    if archive_xml:
        _extract(archive_xml, REPORTS_DIR)
        xml_files = list(REPORTS_DIR.rglob("*.xml"))
        log.info("Fichiers XML extraits : %d", len(xml_files))

    # 3. Construction du dataset
    records = build_records(IMAGES_DIR, REPORTS_DIR, DATA_ROOT / "records.json")
    log.info("Chemin images utilisé : %s", IMAGES_DIR)

    # 4. Vérification
    if records:
        verify_sample(records)
        log.info("\n✅ Dataset OpenI prêt ! Lance maintenant :")
        log.info("   python multimodal_poc.py --image_size 64 --num_epochs 5 --batch_size 32")
