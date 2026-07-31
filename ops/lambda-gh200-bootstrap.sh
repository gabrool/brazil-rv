#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 EXPECTED_GIT_SHA EXPECTED_BUNDLE_SHA256 EXPECTED_BOOTSTRAP_SHA256 INSTANCE_ID" >&2
    exit 2
fi

EXPECTED_GIT_SHA="$1"
EXPECTED_BUNDLE_SHA256="$2"
EXPECTED_BOOTSTRAP_SHA256="$3"
INSTANCE_ID="$4"

FILESYSTEM_ROOT=/lambda/nfs/brazil-rv-east3
WORKSPACE=/home/ubuntu/Brazil-RV
REPOSITORY=/home/ubuntu/Brazil-RV/quant/b3-quant
MODEL_RUNS="$FILESYSTEM_ROOT/quant-data/b3/processed/model_runs"
OPS_DIR="$MODEL_RUNS/_ops"
BUNDLE_PATH="/home/ubuntu/brazil-rv_${EXPECTED_GIT_SHA}.bundle"
SUCCESS_PATH="$OPS_DIR/bootstrap_gh200_${INSTANCE_ID}_success.json"

[[ "$EXPECTED_GIT_SHA" =~ ^[0-9a-fA-F]{40,64}$ ]]
[[ "$EXPECTED_BUNDLE_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]
[[ "$EXPECTED_BOOTSTRAP_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]
[[ "$INSTANCE_ID" =~ ^[A-Za-z0-9._-]+$ ]]
printf '%s  %s\n' "$EXPECTED_BOOTSTRAP_SHA256" "$(readlink -f "$0")" | sha256sum -c -
mountpoint -q "$FILESYSTEM_ROOT"

[[ "$(uname -m)" == aarch64 ]]
GPU_COUNT="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | awk 'NF { count++ } END { print count+0 }')"
[[ "$GPU_COUNT" == 1 ]]
mountpoint -q "$FILESYSTEM_ROOT"

cd "$FILESYSTEM_ROOT"
sha256sum -c SHA256SUMS.txt

mkdir -p "$MODEL_RUNS" "$OPS_DIR" "$WORKSPACE/quant"

DATA_LINK="$WORKSPACE/quant-data"
DATA_TARGET="$FILESYSTEM_ROOT/quant-data"
if [[ -L "$DATA_LINK" ]]; then
    [[ "$(readlink -f "$DATA_LINK")" == "$(readlink -f "$DATA_TARGET")" ]]
elif [[ -e "$DATA_LINK" ]]; then
    echo "Refusing to replace non-symlink path: $DATA_LINK" >&2
    exit 1
else
    ln -s "$DATA_TARGET" "$DATA_LINK"
fi

FEATURE_POINTER="$DATA_LINK/b3/processed/features/m1_features_v1_canonical_path.txt"
[[ -f "$FEATURE_POINTER" ]]
mapfile -t FEATURE_POINTER_LINES < "$FEATURE_POINTER"
[[ ${#FEATURE_POINTER_LINES[@]} -eq 1 ]]
FEATURE_STORE="${FEATURE_POINTER_LINES[0]}"
[[ -n "$FEATURE_STORE" ]]
[[ "$FEATURE_STORE" == "$WORKSPACE/quant-data/b3/processed/features/"* ]]
[[ -d "$FEATURE_STORE" ]]
realpath -e "$FEATURE_STORE" >/dev/null

validate_repository() {
    [[ -e "$REPOSITORY" && ! -L "$REPOSITORY" ]]
    [[ "$(git -C "$REPOSITORY" rev-parse --is-inside-work-tree)" == true ]]
    [[ "$(git -C "$REPOSITORY" symbolic-ref --short HEAD)" == main ]]
    [[ "$(git -C "$REPOSITORY" rev-parse HEAD)" == "$EXPECTED_GIT_SHA" ]]
    [[ -z "$(git -C "$REPOSITORY" status --porcelain)" ]]
}

if [[ -f "$SUCCESS_PATH" ]]; then
    validate_repository
    python3 - \
        "$SUCCESS_PATH" \
        "$INSTANCE_ID" \
        "$EXPECTED_GIT_SHA" \
        "$EXPECTED_BUNDLE_SHA256" \
        "$EXPECTED_BOOTSTRAP_SHA256" \
        "$MODEL_RUNS" <<'PY'
import datetime
import json
import pathlib
import sys

marker_path = pathlib.Path(sys.argv[1])
instance_id, git_sha, bundle_sha, bootstrap_sha = sys.argv[2:6]
model_runs = pathlib.Path(sys.argv[6]).resolve()
marker = json.loads(marker_path.read_text(encoding="utf-8"))
required = {
    "passed",
    "instance_id",
    "git_sha",
    "bundle_sha256",
    "bootstrap_sha256",
    "completed_at_utc",
    "sanity_report_path",
}
if set(marker) != required:
    raise SystemExit("existing bootstrap marker schema is invalid")
if marker["passed"] is not True:
    raise SystemExit("existing bootstrap marker did not pass")
expected = {
    "instance_id": instance_id,
    "git_sha": git_sha,
    "bundle_sha256": bundle_sha,
    "bootstrap_sha256": bootstrap_sha,
}
for key, value in expected.items():
    if marker[key] != value:
        raise SystemExit(f"existing bootstrap marker has wrong {key}")
completed = datetime.datetime.fromisoformat(
    str(marker["completed_at_utc"]).replace("Z", "+00:00")
)
if completed.tzinfo is None:
    raise SystemExit("existing bootstrap marker completion time is not UTC-aware")
report_path = pathlib.Path(marker["sanity_report_path"])
report = report_path.resolve(strict=True)
if model_runs not in report.parents:
    raise SystemExit("existing sanity report is outside model_runs")
payload = json.loads(report.read_text(encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit("existing GH200 sanity report did not pass")
PY
    echo "BRAZIL_RV_BOOTSTRAP_ALREADY_COMPLETE=$SUCCESS_PATH"
    exit 0
fi

BOOTSTRAP_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BOOTSTRAP_LOG="$OPS_DIR/bootstrap_gh200_${INSTANCE_ID}_${BOOTSTRAP_TIMESTAMP}.log"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1
echo "BRAZIL_RV_BOOTSTRAP_LOG=$BOOTSTRAP_LOG"

if [[ -e "$REPOSITORY" || -L "$REPOSITORY" ]]; then
    validate_repository
else
    [[ -f "$BUNDLE_PATH" ]]
    printf '%s  %s\n' "$EXPECTED_BUNDLE_SHA256" "$BUNDLE_PATH" | sha256sum -c -
    [[ "$(git bundle list-heads "$BUNDLE_PATH" refs/heads/main)" == "$EXPECTED_GIT_SHA refs/heads/main" ]]
    git clone --branch main "$BUNDLE_PATH" "$REPOSITORY"
    validate_repository
fi

if [[ -f "$BUNDLE_PATH" ]]; then
    printf '%s  %s\n' "$EXPECTED_BUNDLE_SHA256" "$BUNDLE_PATH" | sha256sum -c -
    git -C "$REPOSITORY" bundle verify "$BUNDLE_PATH"
    [[ "$(git bundle list-heads "$BUNDLE_PATH" refs/heads/main)" == "$EXPECTED_GIT_SHA refs/heads/main" ]]
    rm "$BUNDLE_PATH"
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
export BRAZIL_RV_ROOT="$WORKSPACE"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$HOME/.cache/uv"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"

cd "$REPOSITORY/research"
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen python -m compileall -q src tests
git diff --check
test -z "$(git status --porcelain)"

SANITY_MARKER="$(mktemp "$OPS_DIR/.sanity-start.XXXXXX")"
SUCCESS_TEMP=
cleanup_temporary_files() {
    rm -f "$SANITY_MARKER"
    if [[ -n "$SUCCESS_TEMP" ]]; then
        rm -f "$SUCCESS_TEMP"
    fi
}
trap cleanup_temporary_files EXIT

TORCH_LOGS="recompiles,graph_breaks" \
uv run --frozen python -m brazil_rv.modeling.sanity

SANITY_REPORTS=()
while IFS= read -r -d '' report; do
    SANITY_REPORTS+=("$report")
done < <(find "$MODEL_RUNS" -type f -name sanity_report.json -newer "$SANITY_MARKER" -print0)
[[ ${#SANITY_REPORTS[@]} -eq 1 ]]
SANITY_REPORT="${SANITY_REPORTS[0]}"

uv run --frozen python - "$SANITY_REPORT" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("passed") is not True:
    raise SystemExit("GH200 sanity report did not contain passed == true")
PY

COMPLETED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SUCCESS_TEMP="$(mktemp "$OPS_DIR/.bootstrap-success.XXXXXX")"
python3 - \
    "$SUCCESS_TEMP" \
    "$INSTANCE_ID" \
    "$EXPECTED_GIT_SHA" \
    "$EXPECTED_BUNDLE_SHA256" \
    "$EXPECTED_BOOTSTRAP_SHA256" \
    "$COMPLETED_AT_UTC" \
    "$SANITY_REPORT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "passed": True,
    "instance_id": sys.argv[2],
    "git_sha": sys.argv[3],
    "bundle_sha256": sys.argv[4],
    "bootstrap_sha256": sys.argv[5],
    "completed_at_utc": sys.argv[6],
    "sanity_report_path": sys.argv[7],
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
mv "$SUCCESS_TEMP" "$SUCCESS_PATH"
SUCCESS_TEMP=

echo "GH200 bootstrap succeeded; production training was not started."
echo "Sanity report: $SANITY_REPORT"
echo "Success marker: $SUCCESS_PATH"
