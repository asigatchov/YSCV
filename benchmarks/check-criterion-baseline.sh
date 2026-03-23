#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <criterion_report.txt> <baseline.txt>" >&2
  exit 2
fi

report_path="$1"
baseline_path="$2"

if [[ ! -f "$report_path" ]]; then
  echo "criterion report not found: $report_path" >&2
  exit 2
fi
if [[ ! -f "$baseline_path" ]]; then
  echo "baseline file not found: $baseline_path" >&2
  exit 2
fi

tmp_metrics="$(mktemp)"
trap 'rm -f "$tmp_metrics"' EXIT

awk '
function to_us(value, unit) {
  if (unit == "us" || unit == "µs") {
    return value + 0.0
  }
  if (unit == "ns") {
    return (value + 0.0) / 1000.0
  }
  if (unit == "ms") {
    return (value + 0.0) * 1000.0
  }
  if (unit == "s") {
    return (value + 0.0) * 1000000.0
  }
  return -1
}

{
  if ($0 ~ /^[^[:space:]]+\/[^[:space:]]+$/) {
    benchmark_name = $0
    next
  }

  if (benchmark_name != "" && index($0, "time:") > 0 && index($0, "[") > 0 && index($0, "]") > 0) {
    line = $0
    sub(/.*\[/, "", line)
    sub(/\].*/, "", line)
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)

    count = split(line, tokens, /[[:space:]]+/)
    if (count < 6) {
      printf("failed to parse benchmark time line for '%s': %s\n", benchmark_name, $0) > "/dev/stderr"
      exit 2
    }

    upper_us = to_us(tokens[5], tokens[6])
    if (upper_us < 0) {
      printf("unsupported benchmark unit '%s' for '%s'\n", tokens[6], benchmark_name) > "/dev/stderr"
      exit 2
    }
    printf("%s\t%.6f\n", benchmark_name, upper_us)
    benchmark_name = ""
  }
}
' "$report_path" > "$tmp_metrics"

if [[ ! -s "$tmp_metrics" ]]; then
  echo "no criterion benchmark metrics found in report: $report_path" >&2
  exit 2
fi

if [[ "$(wc -l < "$tmp_metrics" | tr -d ' ')" -eq 0 ]]; then
  echo "failed to parse criterion benchmark metrics from report: $report_path" >&2
  exit 2
fi

checks=0
failures=0

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line="$(printf '%s' "$line" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
  if [[ -z "$line" ]]; then
    continue
  fi

  if [[ "$line" != *"="* ]]; then
    echo "invalid baseline entry (missing '='): $line" >&2
    failures=$((failures + 1))
    continue
  fi

  key="${line%%=*}"
  value="${line#*=}"

  if [[ "$key" == *.max_upper_us ]]; then
    benchmark_name="${key%.max_upper_us}"
    actual="$(
      awk -F '\t' -v benchmark="$benchmark_name" '
        $1 == benchmark {
          print $2
          found = 1
          exit
        }
        END {
          if (!found) {
            exit 1
          }
        }
      ' "$tmp_metrics" 2>/dev/null || true
    )"
    if [[ -z "$actual" ]]; then
      echo "missing benchmark in report: $benchmark_name" >&2
      failures=$((failures + 1))
      continue
    fi

    checks=$((checks + 1))
    if awk -v actual="$actual" -v limit="$value" 'BEGIN { exit !(actual <= limit) }'; then
      echo "PASS $benchmark_name upper_us=$actual <= $value"
    else
      echo "FAIL $benchmark_name upper_us=$actual > $value" >&2
      failures=$((failures + 1))
    fi
    continue
  fi

  if [[ "$key" == *.min_upper_us ]]; then
    benchmark_name="${key%.min_upper_us}"
    actual="$(
      awk -F '\t' -v benchmark="$benchmark_name" '
        $1 == benchmark {
          print $2
          found = 1
          exit
        }
        END {
          if (!found) {
            exit 1
          }
        }
      ' "$tmp_metrics" 2>/dev/null || true
    )"
    if [[ -z "$actual" ]]; then
      echo "missing benchmark in report: $benchmark_name" >&2
      failures=$((failures + 1))
      continue
    fi

    checks=$((checks + 1))
    if awk -v actual="$actual" -v limit="$value" 'BEGIN { exit !(actual >= limit) }'; then
      echo "PASS $benchmark_name upper_us=$actual >= $value"
    else
      echo "FAIL $benchmark_name upper_us=$actual < $value" >&2
      failures=$((failures + 1))
    fi
    continue
  fi

  echo "unsupported baseline key: $key" >&2
  failures=$((failures + 1))
done < "$baseline_path"

if [[ "$checks" -eq 0 ]]; then
  echo "no baseline checks executed: $baseline_path" >&2
  exit 2
fi

if [[ "$failures" -ne 0 ]]; then
  echo "criterion baseline check failed: $failures failures" >&2
  exit 1
fi

echo "criterion baseline check passed: $checks checks"
