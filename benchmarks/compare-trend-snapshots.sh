#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <micro|runtime> <reference_tsv> <current_tsv> <output_tsv>" >&2
  exit 2
fi

kind="$1"
reference_tsv="$2"
current_tsv="$3"
output_tsv="$4"

case "$kind" in
  micro|runtime) ;;
  *)
    echo "unsupported trend kind: $kind (expected: micro|runtime)" >&2
    exit 2
    ;;
esac

for path in "$reference_tsv" "$current_tsv"; do
  if [[ ! -f "$path" ]]; then
    echo "trend file not found: $path" >&2
    exit 2
  fi
done

threshold_var="YSCV_TREND_MAX_REGRESSION_PCT_MICRO"
if [[ "$kind" == "runtime" ]]; then
  threshold_var="YSCV_TREND_MAX_REGRESSION_PCT_RUNTIME"
fi
regression_threshold="${!threshold_var:-}"
if [[ -n "$regression_threshold" ]] && ! [[ "$regression_threshold" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "$threshold_var must be a non-negative number, got: $regression_threshold" >&2
  exit 2
fi

mkdir -p "$(dirname "$output_tsv")"

awk -F '\t' \
  -v kind="$kind" \
  -v output_path="$output_tsv" \
  -v regression_threshold_pct="$regression_threshold" '
function abs_value(x) {
  return x < 0 ? -x : x
}

function is_number(v) {
  return v ~ /^-?[0-9]+([.][0-9]+)?$/
}

function metric_direction(metric_name) {
  if (kind == "micro") {
    return "higher_worse"
  }

  if (metric_name ~ /^(mean_ms|p95_ms|max_ms|min_ms|mae|rmse|max_abs_error|fp|fn|idsw)$/) {
    return "higher_worse"
  }
  if (metric_name ~ /^(fps|precision|recall|f1|ap|mota|motp|matches|tp)$/) {
    return "lower_worse"
  }
  return "neutral"
}

function direction_label(direction) {
  if (direction == "higher_worse") {
    return "higher_is_worse"
  }
  if (direction == "lower_worse") {
    return "lower_is_worse"
  }
  return "neutral"
}

function status_for_delta(direction, delta) {
  if (abs_value(delta) <= 1e-12) {
    return "unchanged"
  }

  if (direction == "higher_worse") {
    return delta > 0 ? "regression" : "improvement"
  }
  if (direction == "lower_worse") {
    return delta < 0 ? "regression" : "improvement"
  }
  return delta > 0 ? "up" : "down"
}

function metric_key_micro(report_name, benchmark_name) {
  return report_name "\t" benchmark_name
}

function metric_key_runtime(source_name, profile_name, metric_name) {
  return source_name "\t" profile_name "\t" metric_name
}

function print_row(first, second, metric_name, direction, reference_value, current_value, delta_value, delta_pct, status) {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
    first, second, metric_name, direction, reference_value, current_value, delta_value, delta_pct, status >> output_path
}

function parse_header() {
  delete header_idx
  for (i = 1; i <= NF; i++) {
    header_idx[$i] = i
  }
}

BEGIN {
  OFS = "\t"
  if (kind == "micro") {
    print "report", "benchmark", "metric", "direction", "reference", "current", "delta", "delta_pct", "status" > output_path
  } else {
    print "source", "profile", "metric", "direction", "reference", "current", "delta", "delta_pct", "status" > output_path
  }
}

FNR == 1 {
  parse_header()
  next
}

NR == FNR {
  if (kind == "micro") {
    if (!("report" in header_idx) || !("benchmark" in header_idx) || !("upper_us" in header_idx)) {
      print "invalid micro trend header in reference file: " FILENAME > "/dev/stderr"
      exit 2
    }
    key = metric_key_micro($(header_idx["report"]), $(header_idx["benchmark"]))
    first = $(header_idx["report"])
    second = $(header_idx["benchmark"])
    metric_name = "upper_us"
    value_raw = $(header_idx["upper_us"])
  } else {
    if (!("source" in header_idx) || !("profile" in header_idx) || !("metric" in header_idx) || !("value" in header_idx)) {
      print "invalid runtime trend header in reference file: " FILENAME > "/dev/stderr"
      exit 2
    }
    value_raw = $(header_idx["value"])
    if (!is_number(value_raw)) {
      next
    }
    key = metric_key_runtime($(header_idx["source"]), $(header_idx["profile"]), $(header_idx["metric"]))
    first = $(header_idx["source"])
    second = $(header_idx["profile"])
    metric_name = $(header_idx["metric"])
  }

  if (!(key in ref_seen)) {
    ref_order[++ref_order_count] = key
  }
  ref_seen[key] = 1
  ref_first[key] = first
  ref_second[key] = second
  ref_metric[key] = metric_name
  ref_value[key] = value_raw + 0.0
  next
}

{
  if (kind == "micro") {
    if (!("report" in header_idx) || !("benchmark" in header_idx) || !("upper_us" in header_idx)) {
      print "invalid micro trend header in current file: " FILENAME > "/dev/stderr"
      exit 2
    }
    key = metric_key_micro($(header_idx["report"]), $(header_idx["benchmark"]))
    first = $(header_idx["report"])
    second = $(header_idx["benchmark"])
    metric_name = "upper_us"
    value_raw = $(header_idx["upper_us"])
  } else {
    if (!("source" in header_idx) || !("profile" in header_idx) || !("metric" in header_idx) || !("value" in header_idx)) {
      print "invalid runtime trend header in current file: " FILENAME > "/dev/stderr"
      exit 2
    }
    value_raw = $(header_idx["value"])
    if (!is_number(value_raw)) {
      next
    }
    key = metric_key_runtime($(header_idx["source"]), $(header_idx["profile"]), $(header_idx["metric"]))
    first = $(header_idx["source"])
    second = $(header_idx["profile"])
    metric_name = $(header_idx["metric"])
  }

  if (!(key in cur_seen)) {
    cur_order[++cur_order_count] = key
  }
  cur_seen[key] = 1
  cur_first[key] = first
  cur_second[key] = second
  cur_metric[key] = metric_name
  cur_value[key] = value_raw + 0.0
}

END {
  gate_enabled = regression_threshold_pct != ""
  gate_threshold = regression_threshold_pct + 0.0

  total_current = 0
  matched = 0
  regressions = 0
  improvements = 0
  unchanged = 0
  directional_changes = 0
  new_metrics = 0
  removed_metrics = 0
  gated_regressions = 0

  for (i = 1; i <= cur_order_count; i++) {
    key = cur_order[i]
    total_current++

    if (!(key in ref_seen)) {
      new_metrics++
      print_row(cur_first[key], cur_second[key], cur_metric[key], "n/a", "n/a", sprintf("%.6f", cur_value[key]), "n/a", "n/a", "new")
      continue
    }

    matched++
    reference_value = ref_value[key]
    current_value = cur_value[key]
    delta = current_value - reference_value
    direction = metric_direction(cur_metric[key])
    status = status_for_delta(direction, delta)

    if (direction == "higher_worse" || direction == "lower_worse") {
      directional_changes++
      if (status == "regression") {
        regressions++
      } else if (status == "improvement") {
        improvements++
      } else if (status == "unchanged") {
        unchanged++
      }
    }

    if (reference_value == 0) {
      delta_pct = "n/a"
    } else {
      delta_pct_value = (delta / reference_value) * 100.0
      delta_pct = sprintf("%.6f", delta_pct_value)
      if (gate_enabled && status == "regression" && abs_value(delta_pct_value) > gate_threshold) {
        gated_regressions++
      }
    }

    print_row(cur_first[key], cur_second[key], cur_metric[key], direction_label(direction), sprintf("%.6f", reference_value), sprintf("%.6f", current_value), sprintf("%.6f", delta), delta_pct, status)
  }

  for (i = 1; i <= ref_order_count; i++) {
    key = ref_order[i]
    if (!(key in cur_seen)) {
      removed_metrics++
      print_row(ref_first[key], ref_second[key], ref_metric[key], "n/a", sprintf("%.6f", ref_value[key]), "n/a", "n/a", "n/a", "removed")
    }
  }

  printf("trend diff summary (%s): current=%d matched=%d new=%d removed=%d directional=%d regressions=%d improvements=%d unchanged=%d\n", \
    kind, total_current, matched, new_metrics, removed_metrics, directional_changes, regressions, improvements, unchanged) > "/dev/stderr"

  if (gate_enabled) {
    printf("trend regression gate (%s): %s <= %.4f%% (matched directional regressions beyond threshold: %d)\n", \
      kind, (gated_regressions == 0 ? "PASS" : "FAIL"), gate_threshold, gated_regressions) > "/dev/stderr"
  }

  if (gate_enabled && gated_regressions > 0) {
    exit 1
  }
}
' "$reference_tsv" "$current_tsv"

echo "wrote trend diff report to $output_tsv"
