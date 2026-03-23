#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <output_tsv> <people_benchmark_report.txt> <face_benchmark_report.txt> <dataset_eval_output.txt>" >&2
  exit 2
fi

output_path="$1"
people_report="$2"
face_report="$3"
eval_output="$4"

for path in "$people_report" "$face_report" "$eval_output"; do
  if [[ ! -f "$path" ]]; then
    echo "input file not found: $path" >&2
    exit 2
  fi
done

timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
github_sha="${GITHUB_SHA:-local}"
github_ref="${GITHUB_REF:-local}"
github_run_id="${GITHUB_RUN_ID:-local}"
github_run_attempt="${GITHUB_RUN_ATTEMPT:-local}"

mkdir -p "$(dirname "$output_path")"
printf "timestamp_utc\tgithub_sha\tgithub_ref\tgithub_run_id\tgithub_run_attempt\tsource\tprofile\tmetric\tvalue\n" > "$output_path"

rows=0

append_row() {
  local source="$1"
  local profile="$2"
  local metric="$3"
  local value="$4"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$timestamp_utc" \
    "$github_sha" \
    "$github_ref" \
    "$github_run_id" \
    "$github_run_attempt" \
    "$source" \
    "$profile" \
    "$metric" \
    "$value" >> "$output_path"
  rows=$((rows + 1))
}

parse_benchmark_report() {
  local mode="$1"
  local report_path="$2"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == frames=* ]]; then
      append_row "deterministic_benchmark" "${mode}.pipeline" "frames" "${line#frames=}"
      continue
    fi

    if [[ "$line" == stage=* ]]; then
      local stage=""
      local runs=""
      local mean_ms=""
      local p95_ms=""
      local fps=""
      local token=""

      for token in $line; do
        case "$token" in
          stage=*) stage="${token#stage=}" ;;
          runs=*) runs="${token#runs=}" ;;
          mean_ms=*) mean_ms="${token#mean_ms=}" ;;
          p95_ms=*) p95_ms="${token#p95_ms=}" ;;
          fps=*) fps="${token#fps=}" ;;
        esac
      done

      if [[ -n "$stage" ]]; then
        local profile="${mode}.${stage}"
        [[ -n "$runs" ]] && append_row "deterministic_benchmark" "$profile" "runs" "$runs"
        [[ -n "$mean_ms" ]] && append_row "deterministic_benchmark" "$profile" "mean_ms" "$mean_ms"
        [[ -n "$p95_ms" ]] && append_row "deterministic_benchmark" "$profile" "p95_ms" "$p95_ms"
        [[ -n "$fps" ]] && append_row "deterministic_benchmark" "$profile" "fps" "$fps"
      fi
    fi
  done < "$report_path"
}

parse_eval_output() {
  local path="$1"
  local current_profile=""
  local token=""

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == detection_eval_voc* ]]; then
      current_profile="dataset_eval.detection_voc"
      for token in $line; do
        case "$token" in
          frames=*) append_row "dataset_eval" "$current_profile" "frames" "${token#frames=}" ;;
        esac
      done
      continue
    fi

    if [[ "$line" == detection_eval_kitti* ]]; then
      current_profile="dataset_eval.detection_kitti"
      for token in $line; do
        case "$token" in
          frames=*) append_row "dataset_eval" "$current_profile" "frames" "${token#frames=}" ;;
        esac
      done
      continue
    fi

    if [[ "$line" == detection_eval_openimages* ]]; then
      current_profile="dataset_eval.detection_openimages"
      for token in $line; do
        case "$token" in
          frames=*) append_row "dataset_eval" "$current_profile" "frames" "${token#frames=}" ;;
        esac
      done
      continue
    fi

    if [[ "$line" == detection_eval_yolo* ]]; then
      current_profile="dataset_eval.detection_yolo"
      for token in $line; do
        case "$token" in
          frames=*) append_row "dataset_eval" "$current_profile" "frames" "${token#frames=}" ;;
        esac
      done
      continue
    fi

    if [[ "$line" == detection_eval_coco* ]]; then
      current_profile="dataset_eval.detection_coco"
      for token in $line; do
        case "$token" in
          frames=*) append_row "dataset_eval" "$current_profile" "frames" "${token#frames=}" ;;
        esac
      done
      continue
    fi

    if [[ "$line" == detection_eval_widerface* ]]; then
      current_profile="dataset_eval.detection_widerface"
      for token in $line; do
        case "$token" in
          frames=*) append_row "dataset_eval" "$current_profile" "frames" "${token#frames=}" ;;
        esac
      done
      continue
    fi

    if [[ "$line" == detection_eval* ]]; then
      current_profile="dataset_eval.detection"
      for token in $line; do
        case "$token" in
          frames=*) append_row "dataset_eval" "$current_profile" "frames" "${token#frames=}" ;;
        esac
      done
      continue
    fi

    if [[ "$line" == tracking_eval_mot* ]]; then
      current_profile="dataset_eval.tracking_mot"
      for token in $line; do
        case "$token" in
          frames=*) append_row "dataset_eval" "$current_profile" "frames" "${token#frames=}" ;;
        esac
      done
      continue
    fi

    if [[ "$line" == tracking_eval* ]]; then
      current_profile="dataset_eval.tracking"
      for token in $line; do
        case "$token" in
          frames=*) append_row "dataset_eval" "$current_profile" "frames" "${token#frames=}" ;;
        esac
      done
      continue
    fi

    if [[ "$line" == *"tp="* && "$current_profile" == dataset_eval.detection* ]]; then
      for token in $line; do
        case "$token" in
          tp=*|fp=*|fn=*|precision=*|recall=*|f1=*|ap=*)
            append_row "dataset_eval" "$current_profile" "${token%%=*}" "${token#*=}"
            ;;
        esac
      done
      continue
    fi

    if [[ "$line" == *"matches="* && "$current_profile" == dataset_eval.tracking* ]]; then
      for token in $line; do
        case "$token" in
          gt=*|matches=*|fp=*|fn=*|idsw=*|precision=*|recall=*|f1=*|mota=*|motp=*)
            append_row "dataset_eval" "$current_profile" "${token%%=*}" "${token#*=}"
            ;;
        esac
      done
      continue
    fi
  done < "$path"
}

parse_benchmark_report "people" "$people_report"
parse_benchmark_report "face" "$face_report"
parse_eval_output "$eval_output"

if [[ "$rows" -eq 0 ]]; then
  echo "no runtime trend metrics exported" >&2
  exit 2
fi

echo "exported $rows runtime trend metrics to $output_path"
