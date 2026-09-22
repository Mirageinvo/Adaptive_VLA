#!/usr/bin/env bash
# Wait until aicenter GPUs are exclusive (<2000 MiB used per GPU, no foreign
# compute apps), then run the honest 10-step SmolVLM2-2.2B FSDP memory probe.
# Never launches production VLA train — probe only.
set -euo pipefail

HOST_REPO="${HOST_REPO:-/home/askhabaliev_gs/Adaptive_VLA-1}"
LOG_DIR="${LOG_DIR:-${HOST_REPO}/logs}"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/vla_exclusive_watcher.log}"
PROBE_SCRIPT="${PROBE_SCRIPT:-${HOST_REPO}/calvin_hicora/scripts/run_aicenter_vla_2b_memory_probe.sh}"
MAX_USED_MIB_PER_GPU="${MAX_USED_MIB_PER_GPU:-2000}"
POLL_SEC="${POLL_SEC:-60}"
STABLE_ROUNDS="${STABLE_ROUNDS:-2}"
STEPS="${STEPS:-10}"

mkdir -p "${LOG_DIR}"
exec >>"${WATCH_LOG}" 2>&1

echo "=== exclusive watcher start $(date -Is) ==="
echo "max_used_mib_per_gpu=${MAX_USED_MIB_PER_GPU} poll=${POLL_SEC}s stable_rounds=${STABLE_ROUNDS}"

exclusive_ok() {
  python3 - <<PY
import subprocess, sys

max_used = float("${MAX_USED_MIB_PER_GPU}")
smi = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()
used = []
for line in smi:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        continue
    used.append((int(parts[0]), float(parts[1])))
if not used:
    print("FAIL: no gpu rows")
    sys.exit(2)
for idx, mib in used:
    if mib >= max_used:
        print(f"WAIT: GPU{idx} used={mib:.0f} MiB >= {max_used:.0f}")
        sys.exit(1)

apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
     "--format=csv,noheader"],
    text=True,
).strip()
foreign = []
for line in apps.splitlines():
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        continue
    pid, name = parts[0], parts[1]
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        cmd = name
    # Allow only empty / no apps. Any compute app is foreign while we wait.
    foreign.append({"pid": pid, "name": name, "cmd": cmd[:120]})
if foreign:
    print("WAIT: foreign compute apps:", foreign)
    sys.exit(1)
print("EXCLUSIVE:", ", ".join(f"GPU{i}={m:.0f}MiB" for i, m in used))
sys.exit(0)
PY
}

stable=0
while true; do
  if exclusive_ok; then
    stable=$((stable + 1))
    echo "$(date -Is) exclusive_ok round=${stable}/${STABLE_ROUNDS}"
    if [[ "${stable}" -ge "${STABLE_ROUNDS}" ]]; then
      echo "$(date -Is) launching 2.2B exclusive probe steps=${STEPS}"
      STEPS="${STEPS}" bash "${PROBE_SCRIPT}"
      rc=$?
      echo "$(date -Is) probe_exit=${rc}"
      exit "${rc}"
    fi
  else
    stable=0
  fi
  sleep "${POLL_SEC}"
done
