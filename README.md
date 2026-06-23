# Practice 3 — Визуомоторный Behavior Cloning (Franka Panda, PandaPickCube)

Стартовый репозиторий третьей практики курса Physical AI. Вы пройдёте полный
пайплайн **Behavior Cloning (BC)**: от сбора видео/action-демонстраций опорной
политикой до обучения визуомоторной модели (изображение → действие) и её
rollout-оценки.

Робот — **Franka Emika Panda** (модель из [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
Apache 2.0). Задача — pick-and-place: поднять куб и доставить его в целевую зону.
Наблюдение BC-модели — RGB-кадр **84×84** с камеры на гриппере.

## Требования

- Docker (Desktop на macOS/Windows или Engine на Linux).
- Для обучения BC желательна NVIDIA GPU; без неё — Google Colab (см. ниже).
- ~3 ГБ под Docker-образ, плюс место под датасеты (`dataset/`) и логи (`logs/`).

## Быстрый старт

```bash
git clone <URL этого репозитория>
cd RoboticsCourse-Practice_3
./scripts/run_container.sh          # соберёт образ и откроет shell в /workspace
```

Скрипт сам определяет железо: при наличии `nvidia-smi` собирает GPU-образ, иначе —
CPU-образ. Флаги: `--cpu`, `--gpu`, `--rebuild`, `--restart`.

Внутри контейнера доступны:
- **noVNC** (графический рабочий стол с MuJoCo): http://localhost:6080/vnc.html
- **TensorBoard** (графики обучения): http://localhost:6006

Остановить/удалить контейнер: `./scripts/kill_container.sh`.

## Потоки (выберите по железу)

| Поток | Симуляция (teleop, сбор, инференс) | Обучение BC |
|---|---|---|
| **GPU** (NVIDIA) | в контейнере | в контейнере (`train_bc.py`) |
| **CPU** (Intel/AMD, Apple Silicon) | в контейнере | в Google Colab (ноутбук) или готовые чекпоинты через `download_artifacts.py` |

На CPU сбор больших датасетов и обучение ResNet-энкодера медленные — используйте
готовые артефакты или Colab.

## Пайплайн и команды по урокам

```
checkpoints/rl_expert.pt                  # опорная политика (privileged state)
        │
        ▼
teleop.py / collect_data.py               # Уроки 3–4: сбор демонстраций
        │
        ▼
dataset/{train_1k, train_10k, eval}/      # .npz: obs (T,84,84,3), actions (T,8), dones, success
        │
        ▼
train_bc.py  ──►  logs/<exp>/             # Урок 5: обучение BC, TensorBoard + checkpoints/{best,last}.pt
        │
        ▼
inference.py  ──►  Success Rate           # Урок 6: rollout-оценка
```

```bash
# Урок 3 — демонстрационный эпизод от опорной политики
python3 collect_data.py --checkpoint checkpoints/rl_expert.pt --num_episodes 1 \
    --save_dir dataset/demo --only_success --seed 7

# Урок 4 — автоматический сбор датасетов
python3 collect_data.py --checkpoint checkpoints/rl_expert.pt --num_episodes 1000 \
    --save_dir dataset/train_1k --only_success --seed 42
python3 collect_data.py --checkpoint checkpoints/rl_expert.pt --num_episodes 200 \
    --save_dir dataset/eval --only_success --seed 100

# Урок 5 — обучение BC (чекпоинты → logs/bc_1k/checkpoints/, логи → logs/bc_1k/)
python3 train_bc.py --train_dir dataset/train_1k --eval_dir dataset/eval \
    --exp_name bc_1k --epochs 100 --batch_size 64 --lr 1e-4
tensorboard --logdir logs/ --bind_all --port 6006

# Урок 6 — rollout-оценка
python3 inference.py --checkpoint logs/bc_1k/checkpoints/best.pt --model bc --episodes 50 --seed 999
python3 inference.py --checkpoint checkpoints/rl_expert.pt --model rl --episodes 50 --seed 999
```

Готовые датасеты и чекпоинты (для CPU-потока):

```bash
python3 scripts/download_artifacts.py --datasets train_1k train_10k eval --checkpoints bc_1k bc_10k
```

## Структура проекта

| Путь | Назначение |
|---|---|
| `env.py` | Среда PandaPickCube (MuJoCo): reset/step, рендер камеры гриппера, privileged state |
| `model.py` | `BCPolicy` (ResNet-энкодер + MLP, image→action) и `RLPolicy` (MLP, state→action) |
| `teleop.py` | Ручное телеуправление с клавиатуры (сбор демонстраций) |
| `collect_data.py` | Автоматический сбор датасета опорной политикой |
| `train_bc.py` | Обучение BC-модели, логи в TensorBoard, чекпоинты |
| `inference.py` | Rollout-оценка BC/RL-моделей, Success Rate |
| `assets/` | MuJoCo-модель Panda (Menagerie) + сцена (стол, куб, целевая зона) |
| `checkpoints/rl_expert.pt` | Опорная политика по privileged state (источник демонстраций) |
| `dataset/`, `logs/` | Собранные данные и логи/чекпоинты обучения (не версионируются) |
| `environment/` | Dockerfile (GPU), Dockerfile.cpu, supervisord (Xvfb+VNC+noVNC) |
| `scripts/` | `run_container.sh`, `kill_container.sh`, `download_artifacts.py`, `distill_expert.py` |

## Опорная политика (`checkpoints/rl_expert.pt`)

Опорная политика работает по **privileged state** (точные координаты куба и цели,
состояние суставов) и служит источником качественных демонстраций. BC-модель
такого доступа не имеет — она учится только по изображению с камеры.

> Подготовка курса: `rl_expert.pt` получен дистилляцией детерминированного
> скриптового контроллера (`expert_scripted.py`, IK по privileged state) в `RLPolicy`
> через `scripts/distill_expert.py`. Студентам этот шаг проходить не нужно —
> чекпоинт уже в репозитории.

## Лицензия моделей

Меши и kinematics Franka Panda — из MuJoCo Menagerie (Apache License 2.0).
