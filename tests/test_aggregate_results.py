import json

from scripts.aggregate_results import METRICS, MODEL_ORDER, aggregate_results


def test_aggregate_results_writes_all_models_in_fixed_order(tmp_path):
    for index, model in enumerate(MODEL_ORDER):
        model_dir = tmp_path / model
        model_dir.mkdir()
        payload = {metric: 0.1 + index for metric in METRICS}
        (model_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    csv_path, markdown_path = aggregate_results(tmp_path)

    csv_text = csv_path.read_text(encoding="utf-8")
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert csv_text.index("baseline") < csv_text.index("concat_25d") < csv_text.index("attention_gate")
    assert "Corrected Reproduction Summary" in markdown_text
    assert all(model in markdown_text for model in MODEL_ORDER)
