#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <output_tsv> <criterion_report_1.txt> [criterion_report_2.txt ...]" >&2
  exit 2
fi

output_path="$1"
shift

timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
github_sha="${GITHUB_SHA:-local}"
github_ref="${GITHUB_REF:-local}"
github_run_id="${GITHUB_RUN_ID:-local}"
github_run_attempt="${GITHUB_RUN_ATTEMPT:-local}"

tmp_metrics="$(mktemp)"
trap 'rm -f "$tmp_metrics"' EXIT

mkdir -p "$(dirname "$output_path")"
printf "timestamp_utc\tgithub_sha\tgithub_ref\tgithub_run_id\tgithub_run_attempt\treport\tbenchmark\tupper_us\n" > "$output_path"

total_metrics=0

for report_path in "$@"; do
  if [[ ! -f "$report_path" ]]; then
    echo "criterion report not found: $report_path" >&2
    exit 2
  fi

  : > "$tmp_metrics"
  awk -v ts="$timestamp_utc" \
      -v sha="$github_sha" \
      -v ref="$github_ref" \
      -v run_id="$github_run_id" \
      -v run_attempt="$github_run_attempt" \
      -v report_name="$(basename "$report_path")" '
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
      printf("failed to parse benchmark time line for '\''%s'\'': %s\n", benchmark_name, $0) > "/dev/stderr"
      exit 2
    }

    upper_us = to_us(tokens[5], tokens[6])
    if (upper_us < 0) {
      printf("unsupported benchmark unit '\''%s'\'' for '\''%s'\''\n", tokens[6], benchmark_name) > "/dev/stderr"
      exit 2
    }

    printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.6f\n", ts, sha, ref, run_id, run_attempt, report_name, benchmark_name, upper_us)
    benchmark_name = ""
  }
}
' "$report_path" > "$tmp_metrics"

  if [[ ! -s "$tmp_metrics" ]]; then
    echo "no criterion benchmark metrics found in report: $report_path" >&2
    exit 2
  fi

  cat "$tmp_metrics" >> "$output_path"
  metrics_count="$(wc -l < "$tmp_metrics" | tr -d ' ')"
  total_metrics=$((total_metrics + metrics_count))
done

if [[ "$total_metrics" -eq 0 ]]; then
  echo "no criterion benchmark metrics exported" >&2
  exit 2
fi

echo "exported $total_metrics criterion metrics to $output_path"
