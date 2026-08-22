#!/usr/bin/env bash
# Установка симулятора LIBERO в контейнер, где УЖЕ есть рабочий torch.
#
# Собрано по граблям, на которые ушло несколько часов при первой установке
# (21 авг 2026). Порядок и флаги существенны, произвольно менять нельзя.
#
# ГЛАВНОЕ ПРАВИЛО: НЕ ставить из ~/LIBERO/requirements.txt. Там numpy==1.22.4,
# transformers==4.21.1, opencv-python==4.6, robosuite==1.4.0, gym==0.25.2 —
# всё это снесёт рабочие версии. По robosuite и gym требования LIBERO и
# ActionCodec расходятся; держим версии ActionCodec, потому что его код
# запускает оценку.
#
# ВТОРОЕ ПРАВИЛО: numpy проверять после каждого шага. Установка opencv и
# robosuite тянет numpy за собой и дважды меняла его версию сама. numpy 2.2.6
# с torch 2.4.1 совместим — проверено прогоном, а не рассуждением о версиях.
# Откатывать «на всякий случай» не надо: однажды я предложил понижение,
# которое сломало бы рабочее окружение, а измерение показало, что всё цело.
#
# ТРЕТЬЕ: `pip install -e ~/LIBERO` молча НЕ работает. В ~/LIBERO/libero/
# нет __init__.py, это namespace-пакет, и editable-finder его не подхватывает.
# Путь добавляется через PYTHONPATH. Импорт — `from libero.libero import
# benchmark`, пакет вложен дважды.
#
# Запуск:  bash experiments/setup_libero.sh
set -eu

say() { printf '\n=== %s\n' "$*"; }
numpy_now() { python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "нет"; }

NP0=$(numpy_now)
say "numpy до установки: $NP0"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

say "1/5 базовые пакеты С зависимостями"
# Версии зафиксированы: robosuite 1.4.1 и gym 0.22.0 — те, на которых
# работает ActionCodec. LIBERO просит другие, и мы это игнорируем осознанно.
pip install --user mujoco==3.2.3 robosuite==1.4.1 gym==0.22.0 \
                   libero2gym==0.1.0 bddl==1.0.1
say "numpy после шага 1: $(numpy_now)"

say "2/5 пакеты БЕЗ зависимостей"
# --no-deps обязателен: robomimic и hydra-core тянут за собой старые numpy и
# протобуф, которые ломают torch и transformers.
pip install --user --no-deps future easydict robomimic==0.2.0 hydra-core==1.2.0
say "numpy после шага 2: $(numpy_now)"

say "3/5 opencv — ТОЛЬКО headless"
# Графическая сборка падает на отсутствующей libgthread-2.0.so.0, а пятая
# версия конфликтует с lerobot.
pip install --user opencv-python-headless==4.12.0.88
say "numpy после шага 3: $(numpy_now)"

say "4/5 LIBERO клоном (НЕ pip install -e)"
if [ ! -d "$HOME/LIBERO" ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$HOME/LIBERO"
else
  echo "  $HOME/LIBERO уже есть, пропускаю клон"
fi

say "5/5 проверка"
NP1=$(numpy_now)
echo "numpy: $NP0 -> $NP1"
export PYTHONPATH="$HOME/LIBERO"
export MUJOCO_GL=egl
python3 - <<'PY'
import numpy, torch
print("numpy", numpy.__version__, "| torch", torch.__version__)
from libero.libero import benchmark
print("libero.libero импортируется (пакет вложен дважды — так и надо)")
import robosuite, gym, mujoco
print("robosuite", robosuite.__version__, "| gym", gym.__version__,
      "| mujoco", mujoco.__version__)
import cv2
print("opencv", cv2.__version__)
PY

cat <<'TXT'

=== Готово. Запускать всё с двумя переменными:
      export PYTHONPATH=$HOME/LIBERO
      export MUJOCO_GL=egl

    Исключения при освобождении EGL-контекста в деструкторе БЕЗОБИДНЫ:
    они возникают после успешного рендера и помечены "Exception ignored".

    Дальше — самопроверка зонда, она не требует ни GPU, ни симулятора:
      python3 experiments/k5c_drift_probe.py --selftest
TXT
