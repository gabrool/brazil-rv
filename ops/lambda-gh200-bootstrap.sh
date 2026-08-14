#!/usr/bin/env bash
set -euo pipefail

require_clean_main_checkout() {
    local repository="$1"
    if [[ ! -d "$repository/.git" || -L "$repository" || -L "$repository/.git" ]]; then
        echo "Existing checkout is not an ordinary Git repository: $repository" >&2
        return 1
    fi
    if [[ "$(git -C "$repository" symbolic-ref --short HEAD 2>/dev/null)" != main ]]; then
        echo "Existing checkout must be on main: $repository" >&2
        return 1
    fi
    if [[ -n "$(git -C "$repository" status --porcelain)" ]]; then
        echo "Existing checkout must be clean: $repository" >&2
        return 1
    fi
}

install_or_update_repository() {
    local repository="$1" bundle="$2" expected_sha="$3"
    local bundle_head current_sha fetched_sha
    bundle_head="$(git bundle list-heads "$bundle" refs/heads/main)" || return 1
    if [[ "$bundle_head" != "$expected_sha refs/heads/main" ]]; then
        echo "Bundle main does not match expected commit $expected_sha." >&2
        return 1
    fi

    if [[ -e "$repository" || -L "$repository" ]]; then
        require_clean_main_checkout "$repository" || return 1
        git -C "$repository" bundle verify "$bundle" >/dev/null || return 1
        git -C "$repository" fetch --no-tags "$bundle" refs/heads/main || return 1
        fetched_sha="$(git -C "$repository" rev-parse FETCH_HEAD)" || return 1
        if [[ "$fetched_sha" != "$expected_sha" ]]; then
            echo "Fetched bundle commit does not match $expected_sha." >&2
            return 1
        fi
        current_sha="$(git -C "$repository" rev-parse HEAD)" || return 1
        if ! git -C "$repository" merge-base --is-ancestor "$current_sha" "$expected_sha"; then
            echo "Existing main diverges from $expected_sha; refusing to overwrite it." >&2
            return 1
        fi
        git -C "$repository" merge --ff-only --no-edit "$expected_sha" || return 1
    else
        git clone --branch main "$bundle" "$repository" || return 1
        git -C "$repository" bundle verify "$bundle" >/dev/null || return 1
    fi

    require_clean_main_checkout "$repository" || return 1
    if [[ "$(git -C "$repository" rev-parse HEAD)" != "$expected_sha" ]]; then
        echo "Repository did not reach expected commit $expected_sha." >&2
        return 1
    fi
}

main() {
    if [[ $# -ne 4 ]]; then
        echo "usage: $0 EXPECTED_GIT_SHA EXPECTED_BUNDLE_SHA256 EXPECTED_BOOTSTRAP_SHA256 INSTANCE_ID" >&2
        exit 2
    fi

    local expected_git_sha="$1" expected_bundle_sha256="$2"
    local expected_bootstrap_sha256="$3" instance_id="$4"
    local filesystem_root=/lambda/nfs/brazil-rv-east3
    local workspace=/home/ubuntu/Brazil-RV
    local repository="$workspace/quant/b3-quant"
    local ops_dir="$filesystem_root/quant-data/b3/processed/model_runs/_ops"
    local bundle_path="/home/ubuntu/brazil-rv_${expected_git_sha}.bundle"
    local log_path="$ops_dir/bootstrap_gh200_${instance_id}.log"
    local success_path="$ops_dir/bootstrap_gh200_${instance_id}_success.json"

    [[ "$expected_git_sha" =~ ^[0-9a-fA-F]{40,64}$ ]]
    [[ "$expected_bundle_sha256" =~ ^[0-9a-fA-F]{64}$ ]]
    [[ "$expected_bootstrap_sha256" =~ ^[0-9a-fA-F]{64}$ ]]
    [[ "$instance_id" =~ ^[A-Za-z0-9._-]+$ ]]
    printf '%s  %s\n' "$expected_bootstrap_sha256" "$(readlink -f "$0")" | sha256sum -c -
    mountpoint -q "$filesystem_root"
    [[ "$(uname -m)" == aarch64 ]]
    [[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | awk 'NF { count++ } END { print count+0 }')" == 1 ]]

    mkdir -p "$ops_dir" "$workspace/quant"
    exec > >(tee -a "$log_path") 2>&1
    echo "BRAZIL_RV_BOOTSTRAP_LOG=$log_path"

    local data_link="$workspace/quant-data" data_target="$filesystem_root/quant-data"
    if [[ -L "$data_link" ]]; then
        [[ "$(readlink -f "$data_link")" == "$(readlink -f "$data_target")" ]]
    elif [[ -e "$data_link" ]]; then
        echo "Refusing to replace non-symlink path: $data_link" >&2
        exit 1
    else
        ln -s "$data_target" "$data_link"
    fi

    printf '%s  %s\n' "$expected_bundle_sha256" "$bundle_path" | sha256sum -c -
    install_or_update_repository "$repository" "$bundle_path" "$expected_git_sha"
    rm "$bundle_path"

    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    fi
    export BRAZIL_RV_ROOT="$workspace"
    export UV_CACHE_DIR="$HOME/.cache/uv"
    export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"

    cd "$repository/research"
    uv sync --frozen --no-default-groups
    uv run --frozen --no-default-groups python -c 'import brazil_rv, torch; print(torch.cuda.get_device_name(0))'
    uv run --frozen --no-default-groups python -m brazil_rv.modeling.train --help >/dev/null
    uv run --frozen --no-default-groups python -m compileall -q src

    local completed_at_utc success_temp
    completed_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    success_temp="$(mktemp "$ops_dir/.bootstrap-success.XXXXXX")"
    trap 'rm -f "$success_temp"' EXIT
    python3 - "$success_temp" "$instance_id" "$expected_git_sha" "$log_path" "$completed_at_utc" <<'PY'
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
    mv "$success_temp" "$success_path"
    trap - EXIT

    echo "GH200 bootstrap succeeded; training was not started."
    echo "Success marker: $success_path"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
