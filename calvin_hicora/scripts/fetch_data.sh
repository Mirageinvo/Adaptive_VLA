#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {debug|D} OUTPUT_DIRECTORY" >&2
  echo "Streams the official CALVIN zip over HTTP into an actions-only v3 cache." >&2
  echo "The archive is NOT saved to disk (Freiburg Accept-Ranges + sequential body)." >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
split="$1"
output_root="$2"

# The CALVIN host currently fails its TLS handshake, so the official HTTP URL
# is intentional. Re-test HTTPS before changing it.
base_url="http://calvin.cs.uni-freiburg.de/dataset"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
converter="${script_dir}/convert_from_zip_actions_only.py"

case "$split" in
  debug)
    archive="calvin_debug_dataset.zip"
    url="${base_url}/${archive}"
    converted_root="${output_root}/converted_debug"
    pipeline_flag=(--pipeline-validation)
    # actions.npy + indices for debug is tiny; keep a small safety margin.
    required_gb=2
    ;;
  D)
    archive="task_D_D.zip"
    url="${base_url}/${archive}"
    converted_root="${output_root}/converted_task_D_D"
    pipeline_flag=()
    # Peak disk is the actions cache (~0.1 GiB) + indices, not the 177 GiB zip.
    required_gb=5
    ;;
  *)
    usage
    ;;
esac

mkdir -p "$output_root" "$converted_root"
checksum_path="${output_root}/sha256sum.txt"

available_kb="$(df -Pk "$converted_root" | awk 'NR==2 {print $4}')"
required_kb=$((required_gb * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "ERROR: ${required_gb} GiB free space required for the actions-only cache; df reports $((available_kb / 1024 / 1024)) GiB." >&2
  exit 1
fi

curl --fail --location --output "$checksum_path" "${base_url}/sha256sum.txt"
[[ -s "$checksum_path" ]] || {
  echo "ERROR: downloaded checksum file is empty" >&2
  exit 1
}

expected_sha256="$(
  awk -v name="$archive" '
    NF >= 2 {
      file = $NF
      sub(/^\*/, "", file)
      if (file == name) {
        print tolower($1)
        exit
      }
    }
  ' "$checksum_path"
)"
if [[ -z "$expected_sha256" ]]; then
  echo "ERROR: ${archive} is absent from official sha256sum.txt" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
echo "Streaming ${url} -> ${converted_root} (zip NOT saved on disk)"
echo "Official sha256=${expected_sha256}"
echo "Tip: run inside tmux: tmux new -s calvin_download"

python3 "$converter" \
  --source-url "$url" \
  --output-root "$converted_root" \
  --expected-sha256 "$expected_sha256" \
  --checksum-url "${base_url}/sha256sum.txt" \
  --overwrite \
  "${pipeline_flag[@]}"

echo "Converted actions-only cache: ${converted_root}"
echo "data_manifest.json records source.url + sha256; archive was never stored."
echo "For old mirrors, verify corrected language annotations and scene_info.npy against CALVIN upstream."
