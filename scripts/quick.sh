#!/usr/bin/env bash
# The tests for what this branch changed, and nothing else: the loop inside a
# task (docs/TESTING.md, "Iterating"). scripts/quick.py maps each changed path
# to the tests that cover it; `scripts/quick.sh --dry-run` prints the plan.
#
#   scripts/quick.sh                  changes since the merge-base with main
#   scripts/quick.sh --base HEAD~1    changes since another commit
#   scripts/quick.sh --extended       the mapped files' extended tests too
#
# A worker finishes a task with `quick.sh --extended` and the lints; the
# essentials gate then runs once on each wave of merged branches
# (docs/TESTING.md, "Iterating"), and scripts/gate.sh --extended before each
# push.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
exec uv run python scripts/quick.py "$@"
