# Dockerfile — GPU-образ с CUDA, PyTorch, MuJoCo, BC-зависимости
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── Core packages ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    xvfb \
    x11vnc \
    x11-xserver-utils \
    fluxbox \
    novnc \
    python3-websockify \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libosmesa6-dev \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    libegl1-mesa-dev \
    wget \
    curl \
    net-tools \
    supervisor \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch (CUDA 12) ────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

# ── MuJoCo + JAX ─────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    mujoco \
    numpy \
    "jax[cuda12]"

# ── BC-специфичные зависимости ────────────────────────────────
RUN pip3 install --no-cache-dir \
    lerobot \
    einops \
    timm \
    matplotlib \
    tensorboard \
    imageio[ffmpeg] \
    tqdm

# ── Environment variables ──────────────────────────────────────
ENV DISPLAY=:99
ENV MUJOCO_GL=egl
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# ── Supervisor config ──────────────────────────────────────────
COPY environment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

WORKDIR /workspace

EXPOSE 5900 6080 6006

ENTRYPOINT []
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
