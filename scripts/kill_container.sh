#!/usr/bin/env bash
set -euo pipefail

removed=false
for CONTAINER in practice3-sim practice3-sim-cpu; do
  if docker inspect "$CONTAINER" &>/dev/null; then
    docker rm -f "$CONTAINER"
    echo "Контейнер '$CONTAINER' удалён."
    removed=true
  fi
done

if ! $removed; then
  echo "Контейнеры practice3-sim / practice3-sim-cpu не найдены."
fi
