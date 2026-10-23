#!/bin/bash
# RoboTwin, этап 0: проверка окружения нового кластера (ничего не ставит).
#   bash experiments/rt0_env_check.sh [каталог данных] [каталог RoboTwin]
# Проверяет: GPU и драйвер; в контейнере — NVIDIA_DRIVER_CAPABILITIES с
# graphics (требование RoboTwin для Vulkan в Docker); ICD Vulkan и
# vulkaninfo; python3.10, conda, ffmpeg; место на диске; SHA RoboTwin и его
# подмодулей (XPolicyLab) — только чтение. Каждая команда проверяется на
# наличие и на код возврата; любая неисправность -> ИТОГ с препятствиями и
# код 3. Отчёт: reports/rt0/env_check.txt.
DATA="${1:-.}"
RT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p reports/rt0
exec > >(tee reports/rt0/env_check.txt) 2>&1
sec () { echo; echo "=== $1"; }
PROB=()
SOFT=()
bad () { echo "  ВНИМАНИЕ: $1"; PROB+=("$1"); }
soft () { echo "  ВНИМАНИЕ (не решающее): $1"; SOFT+=("$1"); }
have () { command -v "$1" >/dev/null 2>&1; }

sec "дата и машина"; date; hostname; uname -r

sec "GPU"
if ! have nvidia-smi; then bad "nvidia-smi не найден"
else
  out="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version \
        --format=csv 2>&1)"; rc=$?
  echo "$out"
  [ $rc -ne 0 ] && bad "nvidia-smi завершился с кодом $rc"
  nvidia-smi 2>/dev/null | grep -i "cuda version" || true
fi

sec "контейнер и возможности NVIDIA"
IN_CT=0
if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || \
   grep -qaE "docker|containerd|kubepods|lxc" /proc/1/cgroup 2>/dev/null; then
  IN_CT=1
fi
echo "контейнер: $([ $IN_CT -eq 1 ] && echo да || echo нет)"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<не задано>}"
if [ $IN_CT -eq 1 ]; then
  case "${NVIDIA_DRIVER_CAPABILITIES:-}" in
    *graphics*|all) echo "  graphics есть";;
    *) bad "в контейнере нет graphics: поднять с -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics";;
  esac
else
  echo "  не контейнер: требование graphics не применяется"
fi

sec "Vulkan"
icd=""
# VK_ICD_FILENAMES / VK_DRIVER_FILES (без root, напр. в домашнем каталоге)
# имеют приоритет над системными каталогами
for f in $(echo "${VK_ICD_FILENAMES:-} ${VK_DRIVER_FILES:-}" | tr ':' ' ') \
         /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
  if [ -f "$f" ]; then
    echo "ICD: $f"; grep -o '"library_path"[^,}]*' "$f" || true
    icd="$f"; break
  fi
done
[ -z "$icd" ] && bad "ICD NVIDIA не найден (ни VK_ICD_FILENAMES, ни системные каталоги)"
ls /usr/lib/x86_64-linux-gnu 2>/dev/null | grep -q "libnvidia-glvkspirv" \
  || bad "нет libnvidia-glvkspirv: Vulkan-часть драйвера не проброшена в контейнер"
if ! have vulkaninfo; then bad "vulkaninfo не установлен (пакет vulkan-tools)"
else
  out="$(vulkaninfo --summary 2>&1)"; rc=$?
  echo "$out" | grep -E "deviceName|driverName|apiVersion|ERROR" | head -20
  if echo "$out" | grep -q "Found no drivers\|ERROR_INCOMPATIBLE_DRIVER"; then
    bad "загрузчик Vulkan не нашёл рабочего драйвера (ICD)"
  elif [ $rc -ne 0 ] && echo "$out" | grep -q "VK_KHR_display"; then
    # в безголовом контейнере vulkaninfo падает на перечислении мониторов,
    # не дойдя до списка устройств: решает rt0_render_check.py
    soft "vulkaninfo упал (код $rc) на VK_KHR_display без монитора; решающая проверка — rt0_render_check.py"
  elif [ $rc -ne 0 ]; then
    bad "vulkaninfo завершился с кодом $rc"
  elif ! echo "$out" | grep -qi "nvidia"; then
    bad "vulkaninfo не видит NVIDIA"
  fi
fi
ldconfig -p 2>/dev/null | grep -E "libvulkan|libGLX_nvidia|libEGL_nvidia" | head || true

sec "python / conda / ffmpeg"
have python3 && python3 --version
if have python3.10; then python3.10 --version
else echo "  python3.10 нет в PATH (нужен в окружении RoboTwin, напр. conda)"; fi
if have conda; then conda env list 2>/dev/null | head -20
else echo "  conda нет в PATH"; fi
if ! have ffmpeg; then bad "ffmpeg не найден"
else
  out="$(ffmpeg -version 2>&1)"; rc=$?
  echo "$out" | head -1
  [ $rc -ne 0 ] && bad "ffmpeg -version завершился с кодом $rc"
fi

sec "диск ($DATA)"
if ! df -h "$DATA"; then bad "df для $DATA не выполнился"; fi

if [ -n "$RT" ]; then
  sec "версия RoboTwin ($RT), только чтение"
  if [ -d "$RT/.git" ] || [ -f "$RT/.git" ]; then
    echo "RoboTwin HEAD: $(git -C "$RT" rev-parse HEAD 2>&1)"
    echo "изменения в рабочей копии: $(git -C "$RT" status --porcelain 2>/dev/null | wc -l)"
    echo "подмодули:"; git -C "$RT" submodule status 2>&1 || bad "git submodule status не выполнился"
    [ -d "$RT/XPolicyLab" ] || bad "в $RT нет XPolicyLab (клонировать с --recurse-submodules)"
  else bad "$RT не git-репозиторий RoboTwin"; fi
fi

echo
if [ ${#SOFT[@]} -gt 0 ]; then
  echo "не решающие замечания ${#SOFT[@]}:"; printf '  - %s\n' "${SOFT[@]}"
fi
if [ ${#PROB[@]} -eq 0 ]; then
  echo "ИТОГ: явных препятствий не найдено"; CODE=0
else
  echo "ИТОГ: препятствий ${#PROB[@]}:"; printf '  - %s\n' "${PROB[@]}"; CODE=3
fi
echo "отчёт: reports/rt0/env_check.txt"
exit $CODE
