#!/usr/bin/env bash
# Step 4 cache tests runner.
# Usage:
#   bash run_step4_tests.sh          # run all step4 tests
#   bash run_step4_tests.sh -v       # verbose output
#   bash run_step4_tests.sh -k judge # run only judge tests
#
# Pass any extra pytest flags as arguments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 4 Cache Tests ==="
echo ""

# Test files in recommended execution order.
TEST_FILES=(
    "tests/cache/components/test_key_builder.py"
    "tests/cache/components/test_judge.py"
    "tests/cache/components/test_gate.py"
    "tests/cache/test_cache_storage.py"
    "tests/cache/test_orchestrator.py"
    "tests/cache/test_interceptor.py"
)

# Verify all test files exist.
for f in "${TEST_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Test file not found: $f"
        exit 1
    fi
done

echo "Running ${#TEST_FILES[@]} test files..."
echo ""

# Run with uv. Pass through any extra args (e.g. -v, -k, --tb=short).
uv run pytest "${TEST_FILES[@]}" "$@"

echo ""
echo "=== All Step 4 tests completed ==="
