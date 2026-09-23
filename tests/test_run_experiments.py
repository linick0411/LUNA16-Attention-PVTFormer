from scripts.run_experiments import build_commands, parse_models, should_skip_completed


def test_experiment_order_trains_then_evaluates_each_model():
    commands = build_commands("all", ["baseline", "attention_gate"])

    assert [(model, stage) for model, stage, _ in commands] == [
        ("baseline", "train"),
        ("baseline", "evaluate"),
        ("attention_gate", "train"),
        ("attention_gate", "evaluate"),
    ]


def test_parse_models_accepts_all():
    assert parse_models("all") == [
        "baseline",
        "concat_25d",
        "attention_gate",
        "voxel_attention",
        "coordinate_attention",
    ]


def test_resume_skips_only_completed_artifacts(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    results_dir = tmp_path / "results"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "checkpoint_baseline.pth").write_bytes(b"checkpoint")

    assert not should_skip_completed("baseline", "train", checkpoints_dir, results_dir)
    assert not should_skip_completed("baseline", "evaluate", checkpoints_dir, results_dir)

    metrics = results_dir / "baseline" / "metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("{}", encoding="utf-8")
    assert should_skip_completed("baseline", "evaluate", checkpoints_dir, results_dir)

    assert should_skip_completed("baseline", "train", checkpoints_dir, results_dir)
