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
REPOSITORY="$WORKSPACE/quant/b3-quant"
OPS_DIR="$FILESYSTEM_ROOT/quant-data/b3/processed/model_runs/_ops"
BUNDLE_PATH="/home/ubuntu/brazil-rv_${EXPECTED_GIT_SHA}.bundle"
LOG_PATH="$OPS_DIR/bootstrap_gh200_${INSTANCE_ID}.log"
SUCCESS_PATH="$OPS_DIR/bootstrap_gh200_${INSTANCE_ID}_success.json"

[[ "$EXPECTED_GIT_SHA" =~ ^[0-9a-fA-F]{40,64}$ ]]
[[ "$EXPECTED_BUNDLE_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]
[[ "$EXPECTED_BOOTSTRAP_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]
[[ "$INSTANCE_ID" =~ ^[A-Za-z0-9._-]+$ ]]
printf '%s  %s\n' "$EXPECTED_BOOTSTRAP_SHA256" "$(readlink -f "$0")" | sha256sum -c -
mountpoint -q "$FILESYSTEM_ROOT"
[[ "$(uname -m)" == aarch64 ]]
[[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | awk 'NF { count++ } END { print count+0 }')" == 1 ]]

mkdir -p "$OPS_DIR" "$WORKSPACE/quant"
exec > >(tee -a "$LOG_PATH") 2>&1
echo "BRAZIL_RV_BOOTSTRAP_LOG=$LOG_PATH"

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

validate_repository() {
    [[ -d "$REPOSITORY/.git" && ! -L "$REPOSITORY" ]]
    [[ "$(git -C "$REPOSITORY" symbolic-ref --short HEAD)" == main ]]
    [[ "$(git -C "$REPOSITORY" rev-parse HEAD)" == "$EXPECTED_GIT_SHA" ]]
    [[ -z "$(git -C "$REPOSITORY" status --porcelain)" ]]
}

printf '%s  %s\n' "$EXPECTED_BUNDLE_SHA256" "$BUNDLE_PATH" | sha256sum -c -
[[ "$(git bundle list-heads "$BUNDLE_PATH" refs/heads/main)" == "$EXPECTED_GIT_SHA refs/heads/main" ]]
if [[ -e "$REPOSITORY" ]]; then
    validate_repository
else
    git clone --branch main "$BUNDLE_PATH" "$REPOSITORY"
    validate_repository
fi
git -C "$REPOSITORY" bundle verify "$BUNDLE_PATH"
rm "$BUNDLE_PATH"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
export BRAZIL_RV_ROOT="$WORKSPACE"
export UV_CACHE_DIR="$HOME/.cache/uv"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"

cd "$REPOSITORY/research"
uv sync --frozen
uv run --frozen python -c 'import brazil_rv, torch; print(torch.cuda.get_device_name(0))'
uv run --frozen python -m brazil_rv.modeling.train --help >/dev/null
uv run --frozen python -m compileall -q src

COMPLETED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SUCCESS_TEMP="$(mktemp "$OPS_DIR/.bootstrap-success.XXXXXX")"
trap 'rm -f "$SUCCESS_TEMP"' EXIT
python3 - "$SUCCESS_TEMP" "$INSTANCE_ID" "$EXPECTED_GIT_SHA" "$LOG_PATH" "$COMPLETED_AT_UTC" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "passed": True,
    "instance_id": sys.argv[2],
    "git_sha": sys.argv[3],
    "bootstrap_log": sys.argv[4],
    "completed_at_utc": sys.argv[5],
}, indent=2) + "\n", encoding="utf-8")
PY
mv "$SUCCESS_TEMP" "$SUCCESS_PATH"
trap - EXIT

echo "GH200 bootstrap succeeded; training was not started."
echo "Success marker: $SUCCESS_PATH"
