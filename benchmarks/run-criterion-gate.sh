#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <cargo_package> <bench_name> <report_path> <baseline_path>" >&2
  exit 2
fi

package="$1"
bench_name="$2"
report_path="$3"
baseline_path="$4"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
baseline_checker="$script_dir/check-criterion-baseline.sh"

if [[ ! -x "$baseline_checker" ]]; then
  echo "baseline checker not found or not executable: $baseline_checker" >&2
  exit 2
fi

attempts="${YSCV_CRITERION_GATE_ATTEMPTS:-2}"
if ! [[ "$attempts" =~ ^[0-9]+$ ]] || [[ "$attempts" -lt 1 ]]; then
  echo "invalid YSCV_CRITERION_GATE_ATTEMPTS: '$attempts' (expected integer >= 1)" >&2
  exit 2
fi

mkdir -p "$(dirname "$report_path")"

last_error=""
for ((attempt = 1; attempt <= attempts; attempt++)); do
  echo "criterion gate warm-up ($package/$bench_name) attempt $attempt/$attempts"
  cargo bench -p "$package" --bench "$bench_name" -- --quick >/dev/null

  echo "criterion gate benchmark run ($package/$bench_name) attempt $attempt/$attempts"
  if ! cargo bench -p "$package" --bench "$bench_name" -- --quick 2>&1 | tee "$report_path"; then
    last_error="cargo bench command failed"
    continue
  fi

  if "$baseline_checker" "$report_path" "$baseline_path"; then
    echo "criterion gate passed on attempt $attempt/$attempts for $package/$bench_name"
    exit 0
  fi

  last_error="baseline check failed"
  if [[ "$attempt" -lt "$attempts" ]]; then
    echo "criterion baseline check failed; retrying..." >&2
  fi
done

echo "criterion gate failed for $package/$bench_name after $attempts attempt(s): $last_error" >&2
exit 1
