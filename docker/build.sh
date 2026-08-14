#!/bin/bash
set -e

ARCH=$(uname -m)

docker build .. \
    -f Dockerfile.${ARCH} \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t ${ARCH}/avla:latest
