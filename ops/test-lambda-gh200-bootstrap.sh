#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lambda-gh200-bootstrap.sh"

temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
source_repository="$temporary/source"
target_repository="$temporary/target"
diverged_repository="$temporary/diverged"
bundle="$temporary/main.bundle"

git init -q -b main "$source_repository"
git -C "$source_repository" config user.name 'Bootstrap Test'
git -C "$source_repository" config user.email 'bootstrap-test@example.invalid'
printf 'A\n' >"$source_repository/value.txt"
git -C "$source_repository" add value.txt
git -C "$source_repository" commit -q -m A
commit_a="$(git -C "$source_repository" rev-parse HEAD)"

git clone -q "$source_repository" "$target_repository"
git clone -q "$source_repository" "$diverged_repository"
git -C "$target_repository" remote set-url origin "$temporary/no-origin"
git -C "$diverged_repository" remote set-url origin "$temporary/no-origin"

printf 'B\n' >"$source_repository/value.txt"
git -C "$source_repository" commit -q -am B
commit_b="$(git -C "$source_repository" rev-parse HEAD)"
git -C "$source_repository" bundle create "$bundle" refs/heads/main

install_or_update_repository "$target_repository" "$bundle" "$commit_b"
[[ "$(git -C "$target_repository" rev-parse HEAD)" == "$commit_b" ]]
[[ "$(git -C "$target_repository" symbolic-ref --short HEAD)" == main ]]
[[ -z "$(git -C "$target_repository" status --porcelain)" ]]

git -C "$diverged_repository" config user.name 'Bootstrap Test'
git -C "$diverged_repository" config user.email 'bootstrap-test@example.invalid'
printf 'C\n' >"$diverged_repository/diverged.txt"
git -C "$diverged_repository" add diverged.txt
git -C "$diverged_repository" commit -q -m C
commit_c="$(git -C "$diverged_repository" rev-parse HEAD)"
[[ "$commit_c" != "$commit_a" ]]
if install_or_update_repository "$diverged_repository" "$bundle" "$commit_b"; then
    echo 'Diverged checkout was overwritten.' >&2
    exit 1
fi
[[ "$(git -C "$diverged_repository" rev-parse HEAD)" == "$commit_c" ]]
[[ "$(git -C "$diverged_repository" symbolic-ref --short HEAD)" == main ]]
[[ -z "$(git -C "$diverged_repository" status --porcelain)" ]]

echo 'PASS verified bundle fast-forward and divergence refusal'
