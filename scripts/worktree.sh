#!/usr/bin/env bash
# Worktree lifecycle for parallel component development.
#
# Each MFT component gets its own git worktree so that ten agents can work
# concurrently without sharing a working directory. Worktrees live under
# .worktrees/ inside the repo, which keeps every file the agents touch inside
# the project directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT_DIR="$ROOT/.worktrees"
BASE="${BASE:-master}"

usage() {
	cat <<'EOF'
usage: worktree.sh add <name> [branch]     create worktree + branch from BASE
       worktree.sh remove <name>           remove worktree (keeps branch)
       worktree.sh list                    list worktrees
       worktree.sh prune                   prune stale worktree admin files

env:
  BASE   base branch for new worktrees (default: master)

examples:
  worktree.sh add kite-broker feat/kite-broker
  worktree.sh add risk-engine feat/risk-engine
EOF
}

cmd_add() {
	local name="${1:?name required}"
	local branch="${2:-feat/$name}"
	local path="$WT_DIR/$name"

	if [ -e "$path" ]; then
		echo "worktree already exists: $path" >&2
		return 1
	fi

	mkdir -p "$WT_DIR"

	if git show-ref --verify --quiet "refs/heads/$branch"; then
		echo "branch $branch already exists, checking it out" >&2
		git worktree add "$path" "$branch"
	else
		git worktree add -b "$branch" "$path" "$BASE"
	fi

	echo "worktree ready: $path (branch $branch, base $BASE)"
}

cmd_remove() {
	local name="${1:?name required}"
	local path="$WT_DIR/$name"

	if [ ! -d "$path" ]; then
		echo "no such worktree: $path" >&2
		return 1
	fi

	# --force discards uncommitted work in the worktree. The branch and its
	# commits are preserved, so pushed work is never lost.
	git worktree remove --force "$path"
	echo "removed $path"
}

cmd_list() {
	git worktree list
}

cmd_prune() {
	git worktree prune
	echo "pruned"
}

case "${1:-}" in
add) shift; cmd_add "$@" ;;
remove) shift; cmd_remove "$@" ;;
list) shift; cmd_list "$@" ;;
prune) shift; cmd_prune "$@" ;;
*) usage; exit 1 ;;
esac
