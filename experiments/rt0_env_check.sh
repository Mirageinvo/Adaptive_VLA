#!/bin/bash
# RoboTwin, этап 0: проверка окружения нового кластера (ничего не ставит).
#   bash experiments/rt0_env_check.sh [каталог данных]
# Печатает: GPU и драйвер, возможности NVIDIA в контейнере (для Vulkan нужен
# graphics), ICD Vulkan и vulkaninfo, python/conda, ffmpeg, место на диске.
# Отчёт: reports/rt0/env_check.txt (тот же вывод).
DATA="${1:-.}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p reports/rt0
exec > >(tee reports/rt0/env_check.txt) 2>&1
sec () { echo; echo "=== $1"; }
ok=1
sec "дата и машина"; date; hostname; uname -r
sec "GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version \
    --format=csv
  nvidia-smi | grep -i "cuda version" || true
else echo "nvidia-smi не найден"; ok=0; fi
sec "возможности NVIDIA в контейнере"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<не задано>}"
case "${NVIDIA_DRIVER_CAPABILITIES:-}" in
  *graphics*|all) echo "  graphics есть";;
  *) echo "  ВНИМАНИЕ: нет graphics — Vulkan (SAPIEN) в контейнере не заработает;"
     echo "  контейнер нужно поднять с -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics"
     ok=0;;
esac
[ -f /.dockerenv ] && echo "  внутри docker" || echo "  не docker (или не определено)"
sec "Vulkan"
found=0
for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
  [ -d "$d" ] && ls -l "$d" && found=1
done
[ $found -eq 0 ] && echo "  каталогов ICD нет"
ls /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json \
  2>/dev/null || { echo "  ВНИМАНИЕ: nvidia_icd.json не найден"; ok=0; }
if command -v vulkaninfo >/dev/null; then
  vulkaninfo --summary 2>&1 | grep -E "deviceName|driverName|apiVersion|ERROR" \
    | head -20
  vulkaninfo --summary 2>&1 | grep -qi "nvidia" \
    || { echo "  ВНИМАНИЕ: vulkaninfo не видит NVIDIA"; ok=0; }
else echo "  vulkaninfo не установлен (пакет vulkan-tools)"; ok=0; fi
ldconfig -p 2>/dev/null | grep -E "libvulkan|libGLX_nvidia|libEGL_nvidia" \
  | head || true
sec "python / conda"
command -v python3 && python3 --version
command -v python3.10 && python3.10 --version || echo "  python3.10 нет в PATH"
command -v conda && conda env list 2>/dev/null | head -20 || echo "  conda нет"
sec "ffmpeg"; ffmpeg -version 2>/dev/null | head -1 || echo "  ffmpeg нет"
sec "диск ($DATA)"; df -h "$DATA"
echo
if [ $ok -eq 1 ]; then echo "ИТОГ: явных препятствий для SAPIEN/Vulkan не найдено"
else echo "ИТОГ: есть препятствия (см. ВНИМАНИЕ выше)"; fi
echo "отчёт: reports/rt0/env_check.txt"
