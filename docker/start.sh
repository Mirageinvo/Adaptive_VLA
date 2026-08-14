#!/bin/bash
set -e

ARCH=$(uname -m)
CONTAINER_NAME=avla_${USER}
IMAGE=${ARCH}/avla:latest

# Датасеты robomimic крупные, а квота в home на кластерах обычно мала.
# Если задать AVLA_DATA, каталог примонтируется внутрь как /workspace/avla/data.
DATA_MOUNT=()
if [ -n "${AVLA_DATA}" ]; then
    mkdir -p "${AVLA_DATA}"
    DATA_MOUNT=(-v "$(readlink -f "${AVLA_DATA}")":/workspace/avla/data)
    echo "data: ${AVLA_DATA} -> /workspace/avla/data"
fi

# Монтирования задаются ТОЛЬКО при создании контейнера. Если он уже есть,
# новый AVLA_DATA не подхватится — нужно docker rm -f и пересоздать.
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "  (монтирования у него прежние; чтобы сменить — docker rm -f ${CONTAINER_NAME})"
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Starting existing container ${CONTAINER_NAME}"
    echo "  (монтирования у него прежние; чтобы сменить — docker rm -f ${CONTAINER_NAME})"
    docker start ${CONTAINER_NAME}
else
    echo "Creating new container ${CONTAINER_NAME}"
    docker run -it -d \
        --name ${CONTAINER_NAME} \
        --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        -e MUJOCO_GL=egl \
        --ipc host \
        -v "$(pwd)/..":/workspace/avla \
        "${DATA_MOUNT[@]}" \
        -w /workspace/avla \
        ${IMAGE}
fi
