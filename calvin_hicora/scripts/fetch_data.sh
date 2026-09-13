#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {debug|D} OUTPUT_DIRECTORY" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
split="$1"
output_root="$2"

# The CALVIN host currently fails its TLS handshake, so the official HTTP URL
# is intentional. Re-test HTTPS before changing it.
base_url="http://calvin.cs.uni-freiburg.de/dataset"

case "$split" in
  debug)
    archive="calvin_debug_dataset.zip"
    url="${base_url}/${archive}"
    expected_dir="calvin_debug_dataset"
    ;;
  D)
    archive="task_D_D.zip"
    url="${base_url}/${archive}"
    expected_dir="task_D_D"
    ;;
  *)
    usage
    ;;
esac

mkdir -p "$output_root"
archive_path="${output_root}/${archive}"
checksum_path="${output_root}/sha256sum.txt"

# For D, 500 GiB is required on this filesystem because it must hold both the
# archive and its extracted tree. If extraction targets another filesystem,
# this check does not validate that target's capacity.
required_gb=5
if [[ "$split" == "D" ]]; then
  required_gb=500
fi
available_kb="$(df -Pk "$output_root" | awk 'NR==2 {print $4}')"
required_kb=$((required_gb * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "ERROR: ${required_gb} GiB free space required; df reports $((available_kb / 1024 / 1024)) GiB." >&2
  exit 1
fi

curl --fail --location --continue-at - --output "$archive_path" "$url"
curl --fail --location --output "$checksum_path" \
  "${base_url}/sha256sum.txt"
[[ -s "$checksum_path" ]] || {
  echo "ERROR: downloaded checksum file is empty" >&2
  exit 1
}

expected_line="$(
  awk -v name="$archive" '
    NF >= 2 {
      file = $NF
      sub(/^\*/, "", file)
      if (file == name) {
        print $1 "  " name
        exit
      }
    }
  ' "$checksum_path"
)"
if [[ -z "$expected_line" ]]; then
  echo "ERROR: ${archive} is absent from official sha256sum.txt" >&2
  exit 1
fi
(
  cd "$output_root"
  printf '%s\n' "$expected_line" | sha256sum --check -
)

if [[ ! -d "${output_root}/${expected_dir}" ]]; then
  command -v unzip >/dev/null 2>&1 || {
    echo "ERROR: unzip is required to extract ${archive}" >&2
    exit 1
  }
  unzip -q "$archive_path" -d "$output_root"
fi

echo "Downloaded and verified: ${output_root}/${expected_dir}"
echo "Keep the archive or record its SHA256 in data_manifest.json."
echo "For old mirrors, verify corrected language annotations and scene_info.npy against CALVIN upstream."
if [[ "$split" == "debug" ]]; then
  echo "Next:"
  echo "  python calvin_hicora/scripts/convert_calvin.py --dataset-root '${output_root}/${expected_dir}' --output-root data/calvin_converted/debug --pipeline-validation --source-archive '${archive_path}'"
else
  echo "Next:"
  echo "  python calvin_hicora/scripts/convert_calvin.py --dataset-root '${output_root}/${expected_dir}' --output-root '${output_root}/converted_task_D_D' --source-archive '${archive_path}'"
fi
