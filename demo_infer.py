import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from luna16_data import SUPPORTED_IMAGE_SUFFIXES, _slice_index

MODEL_REGISTRY = {
    "baseline": {
        "module": "model_baseline",
        "class_name": "PVTFormerBaseline",
        "checkpoint": "checkpoint_baseline.pth",
        "use_25d": False,
    },
    "attention_gate": {
        "module": "model3",
        "class_name": "PVTFormer",
        "checkpoint": "checkpoint_attention_gate.pth",
        "use_25d": True,
    },
    "voxel_attention": {
        "module": "model2",
        "class_name": "PVTFormer_Voxel",
        "checkpoint": "checkpoint_voxel_attention.pth",
        "use_25d": True,
    },
    "coordinate_attention": {
        "module": "model4",
        "class_name": "PVTFormer",
        "checkpoint": "checkpoint_coordinate_attention.pth",
        "use_25d": True,
    },
}

REPO_ROOT = Path(__file__).resolve().parent
LEGACY_ROOT = Path(os.environ.get("LEGACY_PVTFORMER_ROOT", REPO_ROOT.parent / "PVTFormer-main"))
LEGACY_CHECKPOINTS = {
    "baseline": LEGACY_ROOT / "files0/checkpoint.pth",
    "attention_gate": LEGACY_ROOT / "files/checkpoint_model3.pth",
    "voxel_attention": LEGACY_ROOT / "files/checkpoint_model2.pth",
    "coordinate_attention": LEGACY_ROOT / "files/checkpoint_model4.pth",
}


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def seeding(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _find_matching_mask(mask_dir, image_path):
    candidates = [mask_dir / f"{image_path.stem}{suffix}" for suffix in sorted(SUPPORTED_IMAGE_SUFFIXES)]
    return next((path for path in candidates if path.exists()), None)


def _largest_positive_slice(image_paths, mask_dir):
    """Return the image index whose mask has the largest positive area."""
    best_index = None
    best_area = 0
    for index, image_path in enumerate(image_paths):
        mask_path = _find_matching_mask(mask_dir, image_path)
        if mask_path is None:
            continue
        mask = read_gray(mask_path)
        area = int(np.count_nonzero(mask > 127))
        if area > best_area:
            best_index = index
            best_area = area
    if best_index is None:
        raise ValueError(f"No positive nodule mask was found under: {mask_dir}")
    return best_index


def collect_case_slices(case_dir, slice_index=None, auto_positive=False):
    case_dir = Path(case_dir)
    image_paths = sorted(
        [
            path
            for path in (case_dir / "images").iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ],
        key=_slice_index,
    )
    if not image_paths:
        raise FileNotFoundError(f"No PNG/JPG slices found under: {case_dir / 'images'}")

    mask_dir = case_dir / "masks" / "nodule"
    if slice_index is None and auto_positive:
        slice_index = _largest_positive_slice(image_paths, mask_dir)
    elif slice_index is None:
        slice_index = len(image_paths) // 2
    if slice_index < 0 or slice_index >= len(image_paths):
        raise IndexError(f"slice_index={slice_index} is outside 0..{len(image_paths) - 1}")

    prev_index = max(slice_index - 1, 0)
    next_index = min(slice_index + 1, len(image_paths) - 1)
    center_path = image_paths[slice_index]

    mask_path = _find_matching_mask(mask_dir, center_path)
    return image_paths[prev_index], center_path, image_paths[next_index], mask_path


def build_input(prev_path, center_path, next_path, size, use_25d=True):
    slices = []
    for path in (prev_path, center_path, next_path):
        image = read_gray(path)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        slices.append(image)

    if use_25d:
        image_np = np.stack(slices, axis=0)
        image_np = np.expand_dims(image_np, axis=1).astype(np.float32) / 255.0
    else:
        image_np = cv2.cvtColor(slices[1], cv2.COLOR_GRAY2RGB)
        image_np = np.transpose(image_np, (2, 0, 1)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np).unsqueeze(0)
    center_bgr = cv2.cvtColor(slices[1], cv2.COLOR_GRAY2BGR)
    return tensor, center_bgr


def load_mask(mask_path, size):
    if mask_path is None:
        return None

    mask = read_gray(mask_path)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8) * 255
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def colorize_probability(probability):
    prob_uint8 = np.clip(probability * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(prob_uint8, cv2.COLORMAP_TURBO)


def make_overlay(center_bgr, pred_mask, alpha=0.45):
    overlay = center_bgr.copy()
    red_mask = np.zeros_like(center_bgr)
    red_mask[:, :, 2] = pred_mask
    return cv2.addWeighted(overlay, 1.0, red_mask, alpha, 0)


def binary_metrics(pred_mask, gt_mask):
    prediction = pred_mask > 127
    target = gt_mask[:, :, 0] > 127
    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    denominator_dice = 2 * true_positive + false_positive + false_negative
    denominator_iou = true_positive + false_positive + false_negative
    return {
        "dice_f1": 1.0 if denominator_dice == 0 else 2 * true_positive / denominator_dice,
        "iou_jaccard": 1.0 if denominator_iou == 0 else true_positive / denominator_iou,
        "recall": 1.0 if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative),
        "precision": 1.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive),
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
    }


def _label_panel(image, label):
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(panel, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def save_demo_outputs(output_dir, stem, center_bgr, probability, pred_mask, gt_mask=None):
    output_dir = Path(output_dir)
    create_dir(output_dir)

    pred_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    probability_bgr = colorize_probability(probability)
    overlay = make_overlay(center_bgr, pred_mask)

    cv2.imwrite(str(output_dir / f"{stem}_input.png"), center_bgr)
    cv2.imwrite(str(output_dir / f"{stem}_probability.png"), probability_bgr)
    cv2.imwrite(str(output_dir / f"{stem}_prediction.png"), pred_bgr)
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay)

    line = np.ones((center_bgr.shape[0], 6, 3), dtype=np.uint8) * 255
    input_panel = _label_panel(center_bgr, "Input CT")
    probability_panel = _label_panel(probability_bgr, "Probability")
    prediction_panel = _label_panel(pred_bgr, "Prediction")
    overlay_panel = _label_panel(overlay, "Overlay")
    if gt_mask is None:
        joint = np.concatenate(
            [input_panel, line, probability_panel, line, prediction_panel, line, overlay_panel], axis=1
        )
    else:
        ground_truth_panel = _label_panel(gt_mask, "Reference mask")
        joint = np.concatenate(
            [
                input_panel,
                line,
                ground_truth_panel,
                line,
                probability_panel,
                line,
                prediction_panel,
                line,
                overlay_panel,
            ],
            axis=1,
        )
    cv2.imwrite(str(output_dir / f"{stem}_joint.png"), joint)


def resolve_checkpoint(attention, checkpoint, weight_set="current"):
    if checkpoint:
        return Path(checkpoint)
    if weight_set == "legacy":
        return LEGACY_CHECKPOINTS[attention]
    return Path("checkpoints") / MODEL_REGISTRY[attention]["checkpoint"]


def load_model_factory(attention):
    info = MODEL_REGISTRY[attention]
    module = __import__(info["module"], fromlist=[info["class_name"]])
    return getattr(module, info["class_name"])


def run_demo(args):
    seeding(args.seed)
    size = (args.image_size, args.image_size)

    if args.case_dir:
        prev_path, center_path, next_path, auto_mask_path = collect_case_slices(
            args.case_dir, args.slice_index, auto_positive=args.auto_positive
        )
        mask_path = Path(args.mask) if args.mask else auto_mask_path
        stem = f"{Path(args.case_dir).name}_{Path(center_path).stem}"
    else:
        if args.center is None:
            raise ValueError("Provide either --case-dir or --center.")
        center_path = Path(args.center)
        prev_path = Path(args.prev) if args.prev else center_path
        next_path = Path(args.next) if args.next else center_path
        mask_path = Path(args.mask) if args.mask else None
        stem = center_path.stem

    checkpoint_path = resolve_checkpoint(args.attention, args.checkpoint, args.weight_set)
    if not checkpoint_path.exists() and not args.allow_random_weights:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the selected model first or pass --checkpoint. "
            "Use --allow-random-weights only for pipeline testing."
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model_factory(args.attention)().to(device)
    if checkpoint_path.exists():
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("WARNING: running with random weights. Output is not a trained model result.")
    model.eval()

    use_25d = MODEL_REGISTRY[args.attention]["use_25d"]
    image, center_bgr = build_input(prev_path, center_path, next_path, size, use_25d=use_25d)
    image = image.to(device)
    gt_mask = load_mask(mask_path, size)

    with torch.no_grad():
        logits = model(image)
        probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    pred_mask = (probability > args.threshold).astype(np.uint8) * 255
    output_dir = Path(args.output_dir) / args.attention
    save_demo_outputs(output_dir, stem, center_bgr, probability, pred_mask, gt_mask)

    summary = {
        "model": args.attention,
        "weight_set": args.weight_set,
        "checkpoint": str(checkpoint_path.resolve()),
        "threshold": args.threshold,
        "input_slices": {
            "previous": str(prev_path),
            "center": str(center_path),
            "next": str(next_path),
        },
        "reference_mask": str(mask_path) if mask_path else None,
        "metrics": binary_metrics(pred_mask, gt_mask) if gt_mask is not None else None,
    }
    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Attention: {args.attention}")
    print(f"Weight set: {args.weight_set}")
    print(f"Input slices: prev={prev_path}, center={center_path}, next={next_path}")
    print(f"Mask: {mask_path if mask_path else 'not provided'}")
    print(f"Output directory: {output_dir}")
    if summary["metrics"] is not None:
        metrics = summary["metrics"]
        print(
            "Slice metrics: "
            f"Dice={metrics['dice_f1']:.4f}, IoU={metrics['iou_jaccard']:.4f}, "
            f"Recall={metrics['recall']:.4f}, Precision={metrics['precision']:.4f}"
        )
    print(
        f"Saved: {stem}_input.png, {stem}_probability.png, {stem}_prediction.png, "
        f"{stem}_overlay.png, {stem}_joint.png, {stem}_summary.json"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single-case visual demo for a trained LUNA16 attention model.")
    parser.add_argument(
        "--attention",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="attention_gate",
        help="Which trained attention model to run.",
    )
    parser.add_argument(
        "--weight-set",
        choices=("current", "legacy"),
        default="current",
        help="Use current-project checkpoints or the recovered original-project checkpoints.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to checkpoints/<matching checkpoint>.pth.")
    parser.add_argument("--case-dir", default=None, help="Task03_lung case directory, e.g. data/Task03_lung/nodule_0.")
    parser.add_argument("--slice-index", type=int, default=None, help="Center slice index inside --case-dir/images. Defaults to middle slice.")
    parser.add_argument(
        "--auto-positive",
        action="store_true",
        help="When --slice-index is omitted, select the slice with the largest reference nodule mask.",
    )
    parser.add_argument("--prev", default=None, help="Previous slice JPG path. Optional when --center is used.")
    parser.add_argument("--center", default=None, help="Center slice JPG path. Required if --case-dir is not used.")
    parser.add_argument("--next", default=None, help="Next slice JPG path. Optional when --center is used.")
    parser.add_argument("--mask", default=None, help="Optional ground-truth mask JPG path for joint visualization.")
    parser.add_argument("--output-dir", default="demo_outputs", help="Directory for generated demo images.")
    parser.add_argument("--image-size", type=int, default=192, help="Inference image size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary prediction mask.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--allow-random-weights", action="store_true", help="Allow demo pipeline to run without a trained checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
