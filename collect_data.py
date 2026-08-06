"""Автоматический сбор датасета опорной RL-политикой.

Форматы (флаг --format):
    npz     — простой формат: один эпизод = один .npz (Урок 3, демо-эпизод);
    lerobot — индустриальный LeRobotDataset v3.0 (Parquet + MP4), Урок 4.

Запуск:
    # демо-эпизод (.npz) — Урок 3
    python3 collect_data.py --checkpoint checkpoints/rl_expert.pt --num_episodes 1 --save_dir dataset/demo --only_success
    # масштабный сбор в LeRobot — Урок 4
    python3 collect_data.py --checkpoint checkpoints/rl_expert.pt --num_episodes 1000 --save_dir dataset/train_1k --only_success --seed 42 --format lerobot
"""
import argparse
import os
import time

import numpy as np
import torch

from env import PandaPickCubeEnv
from model import RLPolicy

# Параметры LeRobot-датасета
REPO_ID = "local/practice3"
FPS = 20


class LeRobotWriter:
    """Запись эпизодов в индустриальный формат LeRobotDataset v3.0 (Урок 4).

    Студент реализует три метода: создание датасета со схемой признаков
    (`features`), покадровое добавление эпизода и финализацию.
    """

    def __init__(self, save_dir: str):
        # TODO (Урок 4, шаг 1–2): опишите схему `features` и создайте датасет.
        #   features = {
        #       "observation.state": {"dtype": "float32", "shape": (8,), "names": [...]},
        #       "action":            {"dtype": "float32", "shape": (8,), "names": [...]},
        #       "observation.images.front": {"dtype": "video", "shape": (84, 84, 3),
        #                                    "names": ["height", "width", "channels"]},
        #   }
        #   from lerobot.datasets.lerobot_dataset import LeRobotDataset
        #   self.dataset = LeRobotDataset.create(repo_id=REPO_ID, fps=FPS, root=save_dir,
        #                                        robot_type="panda", features=features, use_videos=True)
        #   ВАЖНО: папка save_dir не должна существовать (create требует пустого пути).
        raise NotImplementedError("Реализуйте LeRobotWriter.__init__ (см. Урок 4).")

    def add_episode(self, obs_arr, state_arr, act_arr):
        # TODO (Урок 4, шаг 3): на каждом шаге t вызовите self.dataset.add_frame({...})
        #   c ключами observation.state, action, observation.images.front и "task";
        #   после всех кадров эпизода — self.dataset.save_episode().
        raise NotImplementedError("Реализуйте LeRobotWriter.add_episode (см. Урок 4).")

    def finalize(self):
        # TODO (Урок 4, шаг 3): self.dataset.finalize() — сброс метаданных на диск.
        raise NotImplementedError("Реализуйте LeRobotWriter.finalize (см. Урок 4).")


def parse_args():
    parser = argparse.ArgumentParser(description="Автоматический сбор данных")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--only_success", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["npz", "lerobot"], default="npz",
                        help="npz (Урок 3) или lerobot (Урок 4)")
    return parser.parse_args()


def load_rl_policy(checkpoint_path: str, device: str) -> RLPolicy:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy = RLPolicy(state_dim=ck.get("state_dim", 29), action_dim=ck.get("action_dim", 8),
                      hidden_dims=ck.get("hidden_dims", (512, 256, 128)))
    policy.load_state_dict(ck["model_state_dict"])
    policy.to(device); policy.eval()
    return policy


def collect_episode(env, policy, device, rng_seed):
    """Один эпизод опорной политики. Возвращает obs (T,84,84,3), state (T,8),
    actions (T,8), dones (T,), success."""
    obs = env.reset(seed=rng_seed)
    state = env.get_privileged_state()
    obs_list, state_list, action_list = [obs], [state[:8].copy()], []

    for _ in range(env.episode_length):
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy(state_tensor).squeeze(0).cpu().numpy()
        obs, success, done = env.step(action)
        state = env.get_privileged_state()
        action_list.append(action)
        obs_list.append(obs)
        state_list.append(state[:8].copy())
        if done:
            break

    T = len(action_list)
    obs_arr = np.stack(obs_list[:T]).astype(np.uint8)
    state_arr = np.stack(state_list[:T]).astype(np.float32)
    act_arr = np.stack(action_list).astype(np.float32)
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0
    return obs_arr, state_arr, act_arr, dones, int(success)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[collect_data] Checkpoint: {args.checkpoint} | format: {args.format} | seed: {args.seed} | device: {device}")
    policy = load_rl_policy(args.checkpoint, device)
    env = PandaPickCubeEnv()

    writer = None
    if args.format == "lerobot":
        writer = LeRobotWriter(args.save_dir)
    else:
        os.makedirs(args.save_dir, exist_ok=True)

    saved, attempts = 0, 0
    total_lengths = []
    t_start = time.time()
    rng = np.random.RandomState(args.seed)

    while saved < args.num_episodes:
        attempts += 1
        obs_arr, state_arr, act_arr, dones, success = collect_episode(env, policy, device, rng.randint(0, 2**31))
        if args.only_success and not success:
            continue

        if args.format == "lerobot":
            writer.add_episode(obs_arr, state_arr, act_arr)
        else:
            filename = os.path.join(args.save_dir, f"episode_{saved:04d}.npz")
            np.savez(filename, obs=obs_arr, actions=act_arr, dones=dones, success=success)

        saved += 1
        total_lengths.append(obs_arr.shape[0])
        if saved % 100 == 0 or saved == args.num_episodes:
            sr = saved / attempts * 100 if attempts else 0
            print(f"[collect_data] Episode {saved}/{args.num_episodes} saved | "
                  f"attempts: {attempts} | SR: {sr:.1f}% | avg_len: {np.mean(total_lengths):.0f}")

    if args.format == "lerobot":
        writer.finalize()

    print(f"[collect_data] Done. Saved {saved} episodes ({args.format}) to {args.save_dir}/ "
          f"за {time.time() - t_start:.0f}s")
    env.close()


if __name__ == "__main__":
    main()
