#!/usr/bin/env bash
# Stream Freiburg CALVIN zip -> RGB+meta tree without saving the archive.
#
# Official archives are ZIP (not tar). Depth/tactile live inside episode_*.npz,
# so this wraps stream_extract_rgb_zip.py (HTTP Range stream + per-frame strip).
#
# Usage:
#   ./stream_full_dataset.sh [task_D_D|task_ABC_D|task_ABCD_D] [OUTPUT_ROOT]
set -euo pipefail

ARCHIVE_KEY="${1:-task_D_D}"
OUTPUT_ROOT="${2:-${CALVIN_OUT:-/datasets/askhabaliev_gs/calvin_full}}"

base_url="http://calvin.cs.uni-freiburg.de/dataset"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
extractor="${script_dir}/stream_extract_rgb_zip.py"

case "${CALVIN_ARCHIVE:-$ARCHIVE_KEY}" in
  task_D_D|D|D_D) archive="task_D_D.zip" ;;
  task_ABC_D|ABC|ABC_D) archive="task_ABC_D.zip" ;;
  task_ABCD_D|ABCD|ABCD_D) archive="task_ABCD_D.zip" ;;
  *)
    archive="${CALVIN_ARCHIVE:-$ARCHIVE_KEY}"
    [[ "$archive" == *.zip ]] || archive="${archive}.zip"
    ;;
esac

url="${base_url}/${archive}"
dest="${OUTPUT_ROOT%/}/${archive%.zip}"
mkdir -p "$dest" "${OUTPUT_ROOT}"

echo "=== CALVIN stream extract ==="
echo "archive : $archive"
echo "url     : $url"
echo "dest    : $dest"
echo "NOTE    : zip never saved; depth/tactile/rgb_tactile stripped from episode npz"
df -h "$OUTPUT_ROOT" || df -h / || true

checksum_path="${OUTPUT_ROOT%/}/sha256sum.txt"
curl --fail --location --silent --show-error --output "$checksum_path" "${base_url}/sha256sum.txt"
[[ -s "$checksum_path" ]] || { echo "ERROR: empty sha256sum.txt" >&2; exit 1; }

expected_sha256="$(
  awk -v name="$archive" '
    NF >= 2 {
      file = $NF
      sub(/^\*/, "", file)
      if (file == name) { print tolower($1); exit }
    }
  ' "$checksum_path"
)"
if [[ -z "$expected_sha256" ]]; then
  echo "ERROR: ${archive} absent from official sha256sum.txt" >&2
  exit 1
fi
echo "official sha256: $expected_sha256"

export PYTHONUNBUFFERED=1
extra=()
if [[ "${SKIP_DISK_CHECK:-0}" == "1" ]]; then
  extra+=(--skip-disk-preflight)
fi

py_cmd=(
  python -u calvin_hicora/scripts/stream_extract_rgb_zip.py
  --source-url "$url"
  --output-root "$dest"
  --archive-name "$archive"
  --expected-sha256 "$expected_sha256"
  --checksum-url "${base_url}/sha256sum.txt"
  --log-every 10000
)
if ((${#extra[@]})); then
  py_cmd+=("${extra[@]}")
fi

if command -v docker >/dev/null 2>&1; then
  image=""
  for c in avla_codec_s0_askhabaliev avla_codec_s1_askhabaliev avla_hicora_askhabaliev_gs; do
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$c"; then
      image="$(docker inspect "$c" --format '{{.Config.Image}}')"
      break
    fi
  done
  if [[ -n "$image" ]]; then
    echo "using docker image: $image (bind /datasets + home)"
    exec docker run --rm --network host \
      -v /home/askhabaliev_gs:/home/askhabaliev_gs \
      -v /datasets/askhabaliev_gs:/datasets/askhabaliev_gs \
      -e HOME=/home/askhabaliev_gs \
      -e PYTHONUNBUFFERED=1 \
      -e PYTHONPATH=/home/askhabaliev_gs/Adaptive_VLA-1/calvin_hicora/scripts \
      -w /home/askhabaliev_gs/Adaptive_VLA-1 \
      "$image" \
      "${py_cmd[@]}"
  fi
fi

cd "$repo_root"
exec python3 -u "$extractor" \
  --source-url "$url" \
  --output-root "$dest" \
  --archive-name "$archive" \
  --expected-sha256 "$expected_sha256" \
  --checksum-url "${base_url}/sha256sum.txt" \
  --log-every 10000 \
  "${extra[@]}"
