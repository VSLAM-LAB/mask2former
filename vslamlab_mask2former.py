"""
VSLAM-LAB capability: Mask2Former static/dynamic masks (entry point of github.com/VSLAM-LAB/mask2former).

Generates a per-frame binary static/dynamic mask for every rgb frame of the given VSLAM-LAB sequence folders:
Mask2Former (HuggingFace transformers, COCO-panoptic checkpoint) segments each frame, and any pixel belonging to a
movable "thing" class (person, vehicles, animals - see DYNAMIC_CLASSES) is flagged dynamic. Mask convention:
1 = static point, 0 = dynamic point, 8-bit L PNG with the same filename as its source frame.

Artifact (inside each sequence folder): one mask folder per rgb stream found in rgb.csv (path_rgb_0 ->
mask2former_0, path_rgb_1 -> mask2former_1, ...), each with a mask2former_<i>/.mask2former_complete marker; a
stream with the marker is skipped unless --overwrite. rgb.csv is never modified. If an rgb_raw.csv backup exists
(VSLAM-LAB's pre-sampling frame list), masks are generated from it, so a downsampled rgb.csv still gets complete
coverage.

VSLAM-LAB (Capabilities/mask2former.py) resolves dataset/sequence targets into --sequence-path folders, calls this
script through the `inference` pixi task and wires the masks into experiments ('segmentation: mask2former').
This fork's upstream (facebookresearch/Mask2Former, detectron2) code is not used: the model runs through the
transformers port, with its checkpoint cached in weights/.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

REPO_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = REPO_ROOT / "weights"  # HuggingFace cache_dir for the checkpoint
DEFAULT_MODEL_ID = "facebook/mask2former-swin-large-coco-panoptic"
MASK_FOLDER_BASE = "mask2former"
COMPLETE_MARKER = ".mask2former_complete"

# COCO-panoptic "thing" classes treated as movable/dynamic in a SLAM scene.
DYNAMIC_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
}

SCRIPT_LABEL = "\033[95m[vslamlab_mask2former.py]\033[0m "


def print_info(msg: str) -> None:
    print(f"{SCRIPT_LABEL}{msg}", flush=True)


def print_warning(msg: str) -> None:
    print(f"{SCRIPT_LABEL}\033[93m[warning]\033[0m {msg}", flush=True)


def raw_path(csv_path: Path) -> Path:
    """<name>_raw<suffix>: VSLAM-LAB's pre-edit backup of a sequence csv (full frame list before sampling)."""
    return csv_path.with_name(f"{csv_path.stem}_raw{csv_path.suffix}")


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """(header, rows) of a VSLAM-LAB sequence csv, rows sorted by the timestamp column."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    rows.sort(key=lambda row: int(row[0]) if row[0].lstrip("-").isdigit() else float(row[0]))
    return header, rows


def rgb_stream_indices(header: list[str]) -> list[int]:
    """Stream indices i for every path_rgb_<i> column in an rgb.csv header (mono: [0],
    stereo: [0, 1], ...)."""
    indices = []
    for col in header:
        suffix = col.removeprefix("path_rgb_")
        if col.startswith("path_rgb_") and suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def load_model(model_id: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=WEIGHTS_DIR)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, cache_dir=WEIGHTS_DIR).to(device).eval()
    return processor, model


def mask_frame(processor, model, device: torch.device, image_path: Path) -> np.ndarray:
    """1 = static, 0 = dynamic, uint8, same (H, W) as the source frame."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    segmentation = result["segmentation"].cpu().numpy()
    mask = np.ones(segmentation.shape, dtype=np.uint8)
    for segment in result["segments_info"]:
        label = model.config.id2label[segment["label_id"]]
        if label in DYNAMIC_CLASSES:
            mask[segmentation == segment["id"]] = 0
    return mask


def mask_sequence(
    seq_path: Path, model_loader, device: torch.device, *,
    mask_folder_base: str = MASK_FOLDER_BASE, overwrite: bool = False,
) -> None:
    rgb_csv = seq_path / "rgb.csv"
    rgb_raw = raw_path(rgb_csv)
    # Prefer the pre-sampling backup when one exists: it holds the full frame list, so masks cover every frame
    # even after rgb.csv was downsampled.
    source_csv = rgb_raw if rgb_raw.exists() else rgb_csv
    if not source_csv.exists():
        print_warning(f"Skipping {seq_path} - missing rgb.csv")
        return

    header, rows = read_csv_rows(source_csv)
    streams = rgb_stream_indices(header)
    if not streams:
        print_warning(f"Skipping {seq_path} - rgb.csv has no 'path_rgb_<i>' columns")
        return

    for stream in streams:
        mask_dir = seq_path / f"{mask_folder_base}_{stream}"
        marker = mask_dir / COMPLETE_MARKER
        if marker.exists() and not overwrite:
            print_info(f"Skipping {seq_path} rgb_{stream} - {mask_dir.name} already exists (use --overwrite to recompute)")
            continue

        if mask_dir.exists():
            shutil.rmtree(mask_dir)
        mask_dir.mkdir(parents=True)

        processor, model = model_loader()
        path_idx = header.index(f"path_rgb_{stream}")
        image_paths = [seq_path / row[path_idx] for row in rows]
        for image_path in tqdm(image_paths, desc=f"{seq_path.parent.name}/{seq_path.name} rgb_{stream}"):
            mask = mask_frame(processor, model, device, image_path)
            Image.fromarray(mask, mode="L").save(mask_dir / image_path.name)

        marker.touch()
        print_info(f"{seq_path} - wrote {len(image_paths)} masks to {mask_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-frame static(1)/dynamic(0) semantic masks with Mask2Former."
    )
    parser.add_argument("--sequence-path", nargs="+", type=Path, default=[], dest="sequence_paths",
                        help="VSLAM-LAB sequence folders (each with rgb.csv)")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="HuggingFace Mask2Former checkpoint id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-folder-base", default=MASK_FOLDER_BASE, dest="mask_folder_base",
                        help="Mask folder prefix; each rgb stream i writes to <base>_<i> (default: mask2former)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute masks even if they already exist")
    parser.add_argument("--prefetch", action="store_true",
                        help="Download/cache the checkpoint into weights/ and exit, no sequences needed "
                             "(the `install` pixi task)")
    args = parser.parse_args()

    if args.prefetch:
        load_model(args.model_id, torch.device("cpu"))
        print_info(f"Checkpoint ready: {args.model_id} in {WEIGHTS_DIR}")
        return
    if not args.sequence_paths:
        parser.error("--sequence-path is required (unless --prefetch)")

    device = torch.device(args.device)
    model = {}

    def model_loader():  # lazy: sequences whose masks are complete do not pay the model load
        if not model:
            model["pm"] = load_model(args.model_id, device)
        return model["pm"]

    for seq_path in args.sequence_paths:
        mask_sequence(seq_path.resolve(), model_loader, device,
                      mask_folder_base=args.mask_folder_base, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
