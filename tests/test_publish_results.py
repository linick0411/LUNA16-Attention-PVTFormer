import csv
import hashlib
import json

from scripts.publish_results import MODELS, publish_results


HISTORY_FIELDS = ("epoch", "train_loss", "train_f1", "valid_loss", "valid_f1")


def _write_history(path, valid_f1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=HISTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 1,
                "train_loss": 0.5,
                "train_f1": 0.5,
                "valid_loss": 0.4,
                "valid_f1": valid_f1,
            }
        )


def test_publish_results_selects_strongest_run_and_hashes_weights(tmp_path):
    results_root = tmp_path / "results"
    logs_root = tmp_path / "logs"
    checkpoints_root = tmp_path / "checkpoints"
    output_root = tmp_path / "published"
    results_root.mkdir()
    checkpoints_root.mkdir()

    (results_root / "summary.csv").write_text("model,f1\nbaseline,0.5\n", encoding="utf-8")
    (results_root / "aggregation-audit.json").write_text("{}\n", encoding="utf-8")
    validation_path = results_root / "validation.json"
    validation_path.write_text(json.dumps({"root": "/private/path", "folder_count": 3}), encoding="utf-8")

    history_names = {
        "baseline": "history_pvtformer_baseline_2d_run.csv",
        "concat_25d": "history_concat_25d_control_run.csv",
        "attention_gate": "history_attention_gate_run.csv",
        "voxel_attention": "history_voxel_attention_run.csv",
        "coordinate_attention": "history_coordinate_attention_best.csv",
    }
    for model in MODELS:
        model_root = results_root / model
        model_root.mkdir()
        (model_root / "metrics.json").write_text(
            json.dumps({"model": model, "f1": 0.5, "checkpoint": f"/private/checkpoint_{model}.pth"}),
            encoding="utf-8",
        )
        _write_history(logs_root / history_names[model], 0.8 if model == "coordinate_attention" else 0.5)
        (checkpoints_root / f"checkpoint_{model}.pth").write_bytes(model.encode("utf-8"))

    _write_history(logs_root / "history_coordinate_attention_later.csv", 0.7)
    publish_results(results_root, logs_root, checkpoints_root, output_root, validation_path)

    with (output_root / "training-summary.csv").open(newline="", encoding="utf-8") as file_obj:
        summary = {row["model"]: row for row in csv.DictReader(file_obj)}
    assert summary["coordinate_attention"]["selected_history"] == "history_coordinate_attention_best.csv"
    assert summary["coordinate_attention"]["available_runs"] == "2"
    assert summary["coordinate_attention"]["best_valid_f1"] == "0.8"

    checkpoint_manifest = json.loads((output_root / "checkpoint-manifest.json").read_text(encoding="utf-8"))
    baseline = next(row for row in checkpoint_manifest["files"] if row["model"] == "baseline")
    assert baseline["sha256"] == hashlib.sha256(b"baseline").hexdigest()
    assert json.loads((output_root / "data-validation.json").read_text(encoding="utf-8"))["root"] == "data/Task03_lung_sphere-v1"
