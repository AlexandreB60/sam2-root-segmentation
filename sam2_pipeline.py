#!/usr/bin/env python3
"""
Full SAM 2 pipeline for soil root segmentation.
Covers data preparation, fine-tuning, evaluation and visualisation.

Usage :
    python sam2_pipeline.py prepare          # prepare the data
    python sam2_pipeline.py train            # run the fine-tuning
    python sam2_pipeline.py eval             # evaluate the fine-tuned model
    python sam2_pipeline.py viz              # generate side-by-side images
    python sam2_pipeline.py all              # run everything
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from glob import glob

import cv2
import matplotlib
import numpy as np
import torch
import yaml
from PIL import Image

matplotlib.use("Agg")  # no interactive display on headless machines
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize

from config import (
    CONFIG_PATH,
    DEVICE,
    FINETUNED_CHECKPOINT,
    INFERENCE_CONFIG,
    LOG_DIR,
    MASKS_DIR,
    PREPARED_DATA_DIR,
    PRETRAINED_CHECKPOINT,
    SAM2_DIR,
    SPLIT_SEED,
    TEST_FRAC,
    TILES_DIR,
    VAL_OF_TEMP,
)

sys.path.insert(0, SAM2_DIR)
# =========================================================
#  DATA PREPARATION
# =========================================================


def prepare_data(
    tiles_dir=TILES_DIR, masks_dir=MASKS_DIR, output_dir=PREPARED_DATA_DIR
):
    """
    Convert the raw images and masks to theVOSDataset (DAVIS-like) layout.
    Applies the 70/15/15 split defined in config.py.

    Resulting structure:
        prepared_data/
            images/image_000001/00000.jpg
            masks/image_000001/0/00000.png
            train_list.txt  (557 images)
            val_list.txt    (119 images)
            test_list.txt   (120 images)
    """
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    print(f"  Source images  : {tiles_dir}")
    print(f"  Source masks : {masks_dir}")
    print(f"  output         : {output_dir}")

    images_base = os.path.join(output_dir, "images")
    masks_base = os.path.join(output_dir, "masks")
    os.makedirs(images_base, exist_ok=True)
    os.makedirs(masks_base, exist_ok=True)

    # 70/15/15 split; see config.py for the note on how it differs from sklearn
    all_images = sorted(glob(os.path.join(tiles_dir, "*")))
    n = len(all_images)

    rng = np.random.RandomState(SPLIT_SEED)
    n_temp = int(np.ceil(n * TEST_FRAC))
    ind = rng.permutation(n)
    temp_imgs = [all_images[i] for i in ind[:n_temp]]
    train_imgs = [all_images[i] for i in ind[n_temp:]]

    rng2 = np.random.RandomState(SPLIT_SEED)
    n_test = int(np.ceil(len(temp_imgs) * VAL_OF_TEMP))
    ind2 = rng2.permutation(len(temp_imgs))
    test_imgs = [temp_imgs[i] for i in ind2[:n_test]]
    val_imgs = [temp_imgs[i] for i in ind2[n_test:]]

    print(
        f"\n  Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}"
    )

    splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
    name_map = {}
    counter = 1

    for img_path in all_images:
        video_name = f"image_{counter:06d}"
        name_map[img_path] = video_name
        counter += 1

        img_dir = os.path.join(images_base, video_name)
        mask_dir = os.path.join(masks_base, video_name, "0")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        shutil.copy2(img_path, os.path.join(img_dir, "00000.jpg"))

        mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        src_mask = os.path.join(masks_dir, mask_name)
        if os.path.exists(src_mask):
            # Grayscale is required by SAM 2's vos_dataset.py
            Image.open(src_mask).convert("L").save(os.path.join(mask_dir, "00000.png"))
        else:
            print(f"  [WARN] missing mask : {mask_name}")

        if counter % 100 == 1:
            print(f"  processed {counter - 1}/{n}...")

    for split_name, img_list in splits.items():
        list_path = os.path.join(output_dir, f"{split_name}_list.txt")
        with open(list_path, "w") as f:
            for img_path in sorted(img_list):
                f.write(name_map[img_path] + "\n")
        print(f"  {split_name}_list.txt : {len(img_list)} images")

    print(f"\n{n} pairs prepared in {output_dir}/")


# =========================================================
#  FINE-TNING (via the SAM2 training framework)
# =========================================================


def _update_config(config_path, images_dir, masks_dir, train_list, log_dir):
    """Update the YAML with the real paths and return the new file path"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["dataset"]["img_folder"] = os.path.abspath(images_dir)
    config["dataset"]["gt_folder"] = os.path.abspath(masks_dir)
    config["dataset"]["file_list_txt"] = os.path.abspath(train_list)

    if "launcher" not in config or config["launcher"] is None:
        config["launcher"] = {}
    config["launcher"]["experiment_log_dir"] = os.path.abspath(log_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = config_path.replace(".yaml", f"_updated_{ts}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return out_path


def _to_hydra_name(config_path):
    """Convert an absolute path to one relative to the sam2 package (Hydra requirement)."""
    sam2_pkg_dir = os.path.join(SAM2_DIR, "sam2")
    abs_cfg = os.path.abspath(config_path)
    abs_pkg = os.path.abspath(sam2_pkg_dir)
    if abs_cfg.startswith(abs_pkg + os.sep) or abs_cfg.startswith(abs_pkg + "/"):
        return os.path.relpath(abs_cfg, abs_pkg).replace("\\", "/")
    return config_path


def train(
    prepared_data_dir=PREPARED_DATA_DIR,
    config_path=CONFIG_PATH,
    num_gpus=1,
    log_dir=LOG_DIR,
):
    """
    Run the SAM 2 fine-tuning through the official framework (sam2/training/train.py)
    The YAML config is updated automatically with absolute paths
    """
    print("\n" + "=" * 60)
    print(" SAM2 FINE-TUNING")
    print("=" * 60)

    images_dir = os.path.join(prepared_data_dir, "images")
    masks_dir = os.path.join(prepared_data_dir, "masks")
    train_list = os.path.join(prepared_data_dir, "train_list.txt")

    for path in [images_dir, masks_dir, train_list, PRETRAINED_CHECKPOINT]:
        if not os.path.exists(path):
            print(f"[ERR] missing file : {path}")
            return 1

    updated_config = _update_config(
        config_path, images_dir, masks_dir, train_list, log_dir
    )
    hydra_name = _to_hydra_name(updated_config)
    training_script = os.path.join(SAM2_DIR, "training", "train.py")

    cmd = [
        sys.executable,
        training_script,
        "-c",
        hydra_name,
        "--use-cluster",
        "0",
        "--num-gpus",
        str(num_gpus),
    ]
    print(f"[CMD] {' '.join(cmd)}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = SAM2_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, cwd=SAM2_DIR, env=env)

    if result.returncode == 0:
        print(f"\n[OK] Fine-tuning done — checkpoints in {log_dir}/checkpoints/")
    else:
        print(f"\n[ERR] Fine-tuning failed (code {result.returncode})")
    return result.returncode


# =========================================================
#  EVALUATION
# =========================================================


def _load_sam2(checkpoint, config=INFERENCE_CONFIG):
    """Load the SAM 2 model from a fine-tuned checkpoint"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    hydra_cfg = _to_hydra_name(config) if os.path.isabs(config) else config
    model = build_sam2(hydra_cfg, checkpoint, device=DEVICE)
    return SAM2ImagePredictor(model)


def _predict(predictor, img_rgb, gt_mask, method="center_point"):
    """Predict a mask from a point or bounding-box prompt

    center_point and bbox are derived from the ground-truth mask (oracle protocol).
    image_center uses no ground_truth information and serve as controls for how much the model relies on the prompt."""
    predictor.set_image(img_rgb)
    h, w = gt_mask.shape
    y, x = np.where(gt_mask > 0.5)

    point_coords, point_labels, box = None, None, None

    if method == "bbox" and len(x) > 0:
        box = np.array([[x.min(), y.min(), x.max(), y.max()]])
    elif method == "image_center":
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
    else:
        # center_point, or bbox on an empty mask
        if len(x) > 0:
            point_coords = np.array([[int(np.mean(x)), int(np.mean(y))]])
        else:
            point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
    best = np.argmax(scores)
    return masks[best].astype(np.float32), float(scores[best])


def _dice(pred, gt, eps=1e-7):
    p, g = (pred > 0.5).astype(np.float32), (gt > 0.5).astype(np.float32)
    return (2 * np.sum(p * g) + eps) / (np.sum(p) + np.sum(g) + eps)


def _iou(pred, gt, eps=1e-7):
    p, g = (pred > 0.5).astype(np.float32), (gt > 0.5).astype(np.float32)
    return (np.sum(p * g) + eps) / (np.sum(np.maximum(p, g)) + eps)


def _precision_recall(pred, gt, eps=1e-7):
    p, g = (pred > 0.5).astype(np.float32), (gt > 0.5).astype(np.float32)
    tp = np.sum(p * g)
    return (tp + eps) / (np.sum(p) + eps), (tp + eps) / (np.sum(g) + eps)


def _cl_dice(pred, gt, eps=1e-7):
    """Centerline Dice — topological accuracy measured on the skeletons"""
    p = (pred > 0.5).astype(np.float32)
    g = (gt > 0.5).astype(np.float32)
    if p.sum() == 0 or g.sum() == 0:
        return 0.0 if p.sum() != g.sum() else 1.0
    skel_p = skeletonize(p > 0).astype(np.float32)
    skel_g = skeletonize(g > 0).astype(np.float32)
    t_prec = (skel_p * g).sum() / (skel_p.sum() + eps)
    t_sens = (skel_g * p).sum() / (skel_g.sum() + eps)
    return 2 * (t_prec * t_sens) / (t_prec + t_sens + eps)


def _nsd(pred, gt, tolerance=1.0, eps=1e-7):
    """Normalized Surface Distance — boundary accuracy within a tolerance"""
    p = pred > 0.5
    g = gt > 0.5
    if not p.any() and not g.any():
        return 1.0
    surf_p = p ^ binary_erosion(p)
    surf_g = g ^ binary_erosion(g)
    if not surf_p.any() or not surf_g.any():
        return 0.0
    dist_g = distance_transform_edt(~surf_g)
    dist_p = distance_transform_edt(~surf_p)
    num = (surf_p & (dist_g <= tolerance)).sum() + (
        surf_g & (dist_p <= tolerance)
    ).sum()
    den = surf_p.sum() + surf_g.sum()
    return float(num) / (float(den) + eps)


def evaluate(
    checkpoint=FINETUNED_CHECKPOINT,
    prepared_data_dir=PREPARED_DATA_DIR,
    num_samples=None,
    prompt_method="center_point",
    split="val",
):
    """
    Evaluate the fine-tuned model on the requested split.
    The defaults split is 'val'. SAM 2 was fine-tuned on train_list.txt only,
    with no model selection on the validation set (the training run reports
    steps: {'train': 27850, 'val':0}), so that split stayed entirely unseen
    and its metrics are comparable to test metrics. See the README

    Prints mean DICE, clDice, NSD, IoU, precision and recall
    """
    print("\n" + "=" * 60)
    print("EVALUATION OF THE FINE-TUNED MODEL")
    print("=" * 60)
    print(f"  Checkpoint    : {checkpoint}")
    print(f"  Prompt method : {prompt_method}")
    print(f"  Split         : {split}")

    predictor = _load_sam2(checkpoint)
    list_path = os.path.join(prepared_data_dir, f"{split}_list.txt")
    with open(list_path) as f:
        names = [l.strip() for l in f if l.strip()]
    if num_samples:
        names = names[:num_samples]
    print(f"  Images to evaluate : {len(names)}\n")

    results = {
        "dice": [],
        "cldice": [],
        "nsd": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "confidence": [],
    }
    errors = []

    for idx, name in enumerate(names, 1):
        try:
            img_path = os.path.join(prepared_data_dir, "images", name, "00000.jpg")
            mask_path = os.path.join(prepared_data_dir, "masks", name, "0", "00000.png")
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            gt_mask = (np.array(Image.open(mask_path).convert("L")) > 0).astype(
                np.float32
            )

            pred, conf = _predict(predictor, img_rgb, gt_mask, prompt_method)
            prec, rec = _precision_recall(pred, gt_mask)

            results["dice"].append(_dice(pred, gt_mask))
            results["cldice"].append(_cl_dice(pred, gt_mask))
            results["nsd"].append(_nsd(pred, gt_mask))
            results["iou"].append(_iou(pred, gt_mask))
            results["precision"].append(prec)
            results["recall"].append(rec)
            results["confidence"].append(conf)

            if idx % 10 == 0 or idx == 1:
                print(
                    f"  [{idx:3d}/{len(names)}] {name} | "
                    f"Dice: {results['dice'][-1]:.4f} | clDice: {results['cldice'][-1]:.4f} | NSD: {results['nsd'][-1]:.4f}"
                )
        except Exception as e:
            errors.append((name, str(e)))
            print(f"  [ERR] {name}: {e}")

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    for metric, values in results.items():
        if values:
            print(
                f"  {metric.upper():12s} : {np.mean(values):.4f} ± {np.std(values):.4f}  (median {np.median(values):.4f})"
            )
    print(f"  evaluated : {len(names) - len(errors)}/{len(names)}")
    print(f"{'=' * 60}")
    return results


# =========================================================
#  VISUALISATION
# =========================================================


def _overlay(img_bgr, mask, color_bgr, alpha=0.45):
    out = img_bgr.copy().astype(np.float32)
    for c, v in enumerate(color_bgr):
        out[:, :, c] = np.where(
            mask > 0.5, out[:, :, c] * (1 - alpha) + v * alpha, out[:, :, c]
        )
    return out.astype(np.uint8)


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def visualize(
    checkpoint=FINETUNED_CHECKPOINT,
    prepared_data_dir=PREPARED_DATA_DIR,
    num_samples=10,
    out_dir="eval_viz",
):
    """
    Build 3-column panels:  original image | ground truth (green) | prediction (orange).
    PNG files are written to out_dir/.
    """
    print("\n" + "=" * 60)
    print("VISUALISATION")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)
    predictor = _load_sam2(checkpoint)
    val_list = os.path.join(prepared_data_dir, "val_list.txt")
    with open(val_list) as f:
        names = [l.strip() for l in f if l.strip()][:num_samples]

    for name in names:
        try:
            img_path = os.path.join(prepared_data_dir, "images", name, "00000.jpg")
            mask_path = os.path.join(prepared_data_dir, "masks", name, "0", "00000.png")
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            img_bgr = img_rgb[:, :, ::-1].copy()
            gt = (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.float32)

            pred, _ = _predict(predictor, img_rgb, gt)
            dice = _dice(pred, gt)
            iou = _iou(pred, gt)

            panel_orig = _label(img_bgr.copy(), "Original image ")
            panel_gt = _label(
                _overlay(img_bgr, gt, (0, 255, 0)), "Ground truth (green)"
            )
            panel_pred = _label(
                _overlay(img_bgr, pred, (0, 140, 255)),
                f"Prediction (orange) Dice={dice:.3f} IoU={iou:.3f}",
            )

            h = min(panel_orig.shape[0], 512)

            def resize(im):
                return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))

            composite = np.hstack(
                [resize(panel_orig), resize(panel_gt), resize(panel_pred)]
            )

            out_path = os.path.join(out_dir, f"{name}.png")
            cv2.imwrite(out_path, composite)
            print(f"  [OK] {name} | Dice={dice:.3f} IoU={iou:.3f} → {out_path}")
        except Exception as e:
            print(f"  [ERR] {name}: {e}")

    print(f"\nImages saved in {out_dir}/")


def plot_histogram(
    checkpoint=FINETUNED_CHECKPOINT,
    prepared_data_dir=PREPARED_DATA_DIR,
    metric="dice",
    split="val",
    n_bins=30,
    out_dir="eval_viz",
    prompt_method="center_point",
    nsd_tolerance=1.0,
):
    """
    Compute the chosen metric over every image of the split and plot its distribution histogram (mean and median)

    metric : 'dice', 'cldice' or 'nsd'
    split  : 'val' or 'test'
    """
    print("\n" + "=" * 60)
    print(f"HISTOGRAMME {metric.upper()} — split {split}")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)
    predictor = _load_sam2(checkpoint)
    list_path = os.path.join(prepared_data_dir, f"{split}_list.txt")
    with open(list_path) as f:
        names = [l.strip() for l in f if l.strip()]

    metric_fns = {
        "dice": _dice,
        "cldice": _cl_dice,
        "nsd": lambda p, g: _nsd(p, g, tolerance=nsd_tolerance),
    }
    if metric not in metric_fns:
        raise ValueError("metric must be 'dice', 'cldice' or 'nsd'")
    fn = metric_fns[metric]

    scores = []
    print(f"  Calcul over {len(names)} images...\n")
    for idx, name in enumerate(names, 1):
        try:
            img_path = os.path.join(prepared_data_dir, "images", name, "00000.jpg")
            mask_path = os.path.join(prepared_data_dir, "masks", name, "0", "00000.png")
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            gt = (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.float32)
            pred, _ = _predict(predictor, img_rgb, gt, prompt_method)
            scores.append(fn(pred, gt))
            if idx % 20 == 0:
                print(f"  {idx}/{len(names)}...")
        except Exception as e:
            print(f"  [ERR] {name}: {e}")

    scores = np.array(scores)
    mean_s, med_s, std_s = np.mean(scores), np.median(scores), np.std(scores)
    print(
        f"\n  {metric.upper()} — mean={mean_s:.4f}  median={med_s:.4f}  std={std_s:.4f}"
    )

    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=n_bins, edgecolor="black", alpha=0.7)
    plt.axvline(
        mean_s,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"mean = {mean_s:.3f}",
    )
    plt.axvline(
        med_s,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"median = {med_s:.3f}",
    )
    plt.xlabel(f"Indice {metric.upper()}")
    plt.ylabel("Number of images")
    plt.title(
        f"{metric.upper()} distribution — fine-tuned SAM2 ({len(scores)} {split}images)"
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.5)

    fig_path = os.path.join(out_dir, f"{split}_{metric}_histogram.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Histogram saved : {fig_path}")
    return scores


# =========================================================
#  MAIN
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM2 pipeline for soil root segmentation"
    )
    parser.add_argument(
        "mode",
        choices=["prepare", "train", "eval", "viz", "hist", "all"],
        help="step to run (hist = metric distribution histogram)",
    )
    parser.add_argument("--checkpoint", default=FINETUNED_CHECKPOINT)
    parser.add_argument("--prepared_data_dir", default=PREPARED_DATA_DIR)
    parser.add_argument("--tiles_dir", default=TILES_DIR)
    parser.add_argument("--masks_dir", default=MASKS_DIR)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--prompt_method",
        choices=["center_point", "bbox", "image_center"],
        default="center_point",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--out_dir", default="eval_viz")
    parser.add_argument(
        "--metric",
        choices=["dice", "cldice", "nsd"],
        default="dice",
        help="metric used for the histogram",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Split used for evaluation and the histogram",
    )
    args = parser.parse_args()

    if args.mode in ("prepare", "all"):
        prepare_data(args.tiles_dir, args.masks_dir, args.prepared_data_dir)

    if args.mode in ("train", "all"):
        code = train(args.prepared_data_dir, num_gpus=args.num_gpus)
        if code != 0:
            sys.exit(code)

    if args.mode in ("eval", "all"):
        evaluate(
            args.checkpoint,
            args.prepared_data_dir,
            args.num_samples,
            args.prompt_method,
            args.split,
        )

    if args.mode in ("viz", "all"):
        n = args.num_samples if args.num_samples else 10
        visualize(args.checkpoint, args.prepared_data_dir, n, args.out_dir)

    if args.mode == "hist":
        plot_histogram(
            args.checkpoint,
            args.prepared_data_dir,
            metric=args.metric,
            split=args.split,
            out_dir=args.out_dir,
            prompt_method=args.prompt_method,
        )
