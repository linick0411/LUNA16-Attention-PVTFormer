#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

case_dir="${1:-data/Task03_lung_sphere-v1/nodule_592}"
output_dir="${2:-demo_outputs/original-project}"

if [[ ! -d "$case_dir/images" ]]; then
  echo "Case directory not found: $case_dir" >&2
  exit 1
fi

.venv/bin/python demo_infer.py \
  --attention attention_gate \
  --weight-set legacy \
  --case-dir "$case_dir" \
  --auto-positive \
  --output-dir "$output_dir"

echo
echo "Demo completed. Open the *_joint.png file under:"
echo "$repo_root/$output_dir/attention_gate"
