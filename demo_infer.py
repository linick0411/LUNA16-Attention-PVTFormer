import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


MODEL_REGISTRY = {
    "attention_gate": {
        "module": "model3",
        "class_name": "PVTFormer",
        "checkpoint": "checkpoint_attention_gate.pth",
    },
    "voxel_attention": {
        "module": "model2",
        "class_name": "PVTFormer_Voxel",
        "checkpoint": "checkpoint_voxel_attention.pth",
    },
    "coordinate_attention": {
        "module": "model4",
        "class_name": "PVTFormer",
        "checkpoint": "checkpoint_coordinate_attention.pth",
    },
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


def collect_case_slices(case_dir, slice_index=None):
    case_dir = Path(case_dir)
    image_paths = sorted((case_dir / "images").glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPG slices found under: {case_dir / 'images'}")

    if slice_index is None:
        slice_index = len(image_paths) // 2
    if slice_index < 0 or slice_index >= len(image_paths):
        raise IndexError(f"slice_index={slice_index} is outside 0..{len(image_paths) - 1}")

    prev_index = max(slice_index - 1, 0)
    next_index = min(slice_index + 1, len(image_paths) - 1)
    center_path = image_paths[slice_index]

    mask_path = case_dir / "masks" / "nodule" / center_path.name
    return image_paths[prev_index], center_path, image_paths[next_index], mask_path if mask_path.exists() else None


def build_input(prev_path, center_path, next_path, size):
    slices = []
    for path in (prev_path, center_path, next_path):
        image = read_gray(path)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        slices.append(image)

    image_np = np.stack(slices, axis=0)
    image_np = np.expand_dims(image_np, axis=1).astype(np.float32) / 255.0
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

    line = np.ones((center_bgr.shape[0], 10, 3), dtype=np.uint8) * 255
    if gt_mask is None:
        joint = np.concatenate([center_bgr, line, probability_bgr, line, pred_bgr, line, overlay], axis=1)
    else:
        joint = np.concatenate([center_bgr, line, gt_mask, line, probability_bgr, line, pred_bgr, line, overlay], axis=1)
    cv2.imwrite(str(output_dir / f"{stem}_joint.png"), joint)


def resolve_checkpoint(attention, checkpoint):
    if checkpoint:
        return Path(checkpoint)
    return Path("checkpoints") / MODEL_REGISTRY[attention]["checkpoint"]


def load_model_factory(attention):
    info = MODEL_REGISTRY[attention]
    module = __import__(info["module"], fromlist=[info["class_name"]])
    return getattr(module, info["class_name"])


def run_demo(args):
    seeding(args.seed)
    size = (args.image_size, args.image_size)

    if args.case_dir:
        prev_path, center_path, next_path, auto_mask_path = collect_case_slices(args.case_dir, args.slice_index)
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

    checkpoint_path = resolve_checkpoint(args.attention, args.checkpoint)
    if not checkpoint_path.exists() and not args.allow_random_weights:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the selected model first or pass --checkpoint. "
            "Use --allow-random-weights only for pipeline testing."
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model_factory(args.attention)().to(device)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("WARNING: running with random weights. Output is not a trained model result.")
    model.eval()

    image, center_bgr = build_input(prev_path, center_path, next_path, size)
    image = image.to(device)
    gt_mask = load_mask(mask_path, size)

    with torch.no_grad():
        logits = model(image)
        probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    pred_mask = (probability > args.threshold).astype(np.uint8) * 255
    output_dir = Path(args.output_dir) / args.attention
    save_demo_outputs(output_dir, stem, center_bgr, probability, pred_mask, gt_mask)

    print(f"Attention: {args.attention}")
    print(f"Input slices: prev={prev_path}, center={center_path}, next={next_path}")
    print(f"Mask: {mask_path if mask_path else 'not provided'}")
    print(f"Output directory: {output_dir}")
    print(f"Saved: {stem}_input.png, {stem}_probability.png, {stem}_prediction.png, {stem}_overlay.png, {stem}_joint.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single-case visual demo for a trained LUNA16 attention model.")
    parser.add_argument(
        "--attention",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="voxel_attention",
        help="Which trained attention model to run.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to checkpoints/<matching checkpoint>.pth.")
    parser.add_argument("--case-dir", default=None, help="Task03_lung case directory, e.g. data/Task03_lung/nodule_0.")
    parser.add_argument("--slice-index", type=int, default=None, help="Center slice index inside --case-dir/images. Defaults to middle slice.")
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
