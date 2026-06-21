#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/workspace"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$PROJECT_DIR/environment"
REBUILD=false
RESTART=false
FORCE_CPU=false
FORCE_GPU=false

for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=true ;;
    --restart) RESTART=true ;;
    --cpu) FORCE_CPU=true ;;
    --gpu) FORCE_GPU=true ;;
    *) echo "Неизвестный флаг: $arg"; exit 1 ;;
  esac
done

if $FORCE_CPU && $FORCE_GPU; then
  echo "Ошибка: нельзя одновременно указать --cpu и --gpu."
  exit 1
fi

NVIDIA_OK=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  NVIDIA_OK=true
fi

if $FORCE_GPU; then
  if ! $NVIDIA_OK; then
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  Запрошен --gpu, но nvidia-smi недоступен на хосте.      ║"
    echo "║  Нужны драйвер NVIDIA и Container Toolkit:               ║"
    echo "║  https://docs.nvidia.com/datacenter/cloud-native/        ║"
    echo "║  container-toolkit/install-guide.html                    ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    exit 1
  fi
  USE_GPU=true
elif $FORCE_CPU; then
  USE_GPU=false
elif $NVIDIA_OK; then
  USE_GPU=true
else
  USE_GPU=false
fi

if $USE_GPU; then
  IMAGE="practice3-vnc"
  DOCKERFILE="Dockerfile"
  CONTAINER="practice3-sim"
  echo "==> Режим GPU: образ ${IMAGE}, контейнер ${CONTAINER}"
  echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  IMAGE="practice3-vnc-cpu"
  DOCKERFILE="Dockerfile.cpu"
  CONTAINER="practice3-sim-cpu"
  echo "==> Режим без GPU (CPU-образ)."
  echo "==> Обучение с GPU: Google Colab или машина с NVIDIA + ./scripts/run_container.sh --gpu"
fi

container_running() { docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; }
container_exists()  { docker inspect "$CONTAINER" &>/dev/null; }
image_exists()      { docker image inspect "$IMAGE" &>/dev/null; }

HOSTNAME=$(hostname -f 2>/dev/null || hostname)
USER_NAME=$(whoami)

print_connect_info() {
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  Practice_3 контейнер готов!"
  echo "════════════════════════════════════════════════════════════"
  echo ""
  echo "  Docker Desktop (macOS / Windows):"
  echo "    noVNC:       http://localhost:6080/vnc.html"
  echo "    TensorBoard: http://localhost:6006"
  echo "    VNC-клиент:  localhost:5900"
  echo ""
  echo "  Linux / удалённая ВМ:"
  echo "    noVNC:       http://${HOSTNAME}:6080/vnc.html"
  echo "    TensorBoard: http://${HOSTNAME}:6006"
  echo ""
  echo "  SSH-туннель:"
  echo "    ssh -L 6080:localhost:6080 -L 6006:localhost:6006 ${USER_NAME}@${HOSTNAME}"
  echo "════════════════════════════════════════════════════════════"
  echo ""
}

if $REBUILD; then
  echo "==> Останавливаю контейнер ${CONTAINER}..."
  docker rm -f "$CONTAINER" 2>/dev/null || true
  echo "==> Пересобираю образ..."
  docker build \
    --build-arg HTTP_PROXY= \
    --build-arg HTTPS_PROXY= \
    --build-arg http_proxy= \
    --build-arg https_proxy= \
    -t "$IMAGE" -f "$ENV_DIR/$DOCKERFILE" "$PROJECT_DIR"
elif $RESTART; then
  echo "==> Перезапускаю контейнер..."
  docker rm -f "$CONTAINER" 2>/dev/null || true
fi

if ! image_exists; then
  echo "==> Образ не найден, собираю ${IMAGE}..."
  docker build \
    --build-arg HTTP_PROXY= \
    --build-arg HTTPS_PROXY= \
    --build-arg http_proxy= \
    --build-arg https_proxy= \
    -t "$IMAGE" -f "$ENV_DIR/$DOCKERFILE" "$PROJECT_DIR"
fi

if container_running; then
  echo "==> Контейнер уже запущен, подключаюсь..."
  print_connect_info
  exec docker exec -it -w "$WORKDIR" "$CONTAINER" bash
fi

if container_exists; then
  echo "==> Запускаю остановленный контейнер..."
  docker start "$CONTAINER"
  sleep 2
  print_connect_info
  exec docker exec -it -w "$WORKDIR" "$CONTAINER" bash
fi

echo "==> Запускаю новый контейнер ${CONTAINER}..."
if [ "$(uname -s)" = "Linux" ]; then
  NETWORK_ARGS="--network host"
  PORT_ARGS=""
else
  NETWORK_ARGS=""
  PORT_ARGS="-p 5900:5900 -p 6080:6080 -p 6006:6006"
fi

DATASET_DIR="$PROJECT_DIR/dataset"
mkdir -p "$DATASET_DIR"

if $USE_GPU; then
  docker run -d \
    --name "$CONTAINER" \
    --ipc=host \
    $PORT_ARGS \
    $NETWORK_ARGS \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v "$PROJECT_DIR":"$WORKDIR" \
    -v "$DATASET_DIR":"$WORKDIR/dataset" \
    "$IMAGE"
else
  docker run -d \
    --name "$CONTAINER" \
    --ipc=host \
    $PORT_ARGS \
    $NETWORK_ARGS \
    -v "$PROJECT_DIR":"$WORKDIR" \
    -v "$DATASET_DIR":"$WORKDIR/dataset" \
    "$IMAGE"
fi

sleep 2
print_connect_info
echo "==> Подключаюсь..."
exec docker exec -it -w "$WORKDIR" "$CONTAINER" bash
