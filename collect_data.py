"""Автоматический сбор датасета опорной RL-политикой.

Форматы (флаг --format):
    npz     — один эпизод = один .npz (Урок 3, демо-эпизод);
    lerobot — LeRobotDataset v3.0 (Parquet + MP4), Урок 4.

Запуск:
    # демо-эпизод (.npz) — Урок 3
    python3 collect_data.py --checkpoint checkpoints/rl_expert.pt --num_episodes 1 --save_dir dataset/demo --only_success
    # масштабный сбор в LeRobot — Урок 4
    python3 collect_data.py --checkpoint checkpoints/rl_expert.pt --num_episodes 1000 --save_dir dataset/train_1k --only_success --seed 42 --format lerobot
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from env import PandaPickCubeEnv
from model import RLPolicy

# Параметры LeRobot-датасета
REPO_ID = "local/practice3"
FPS = 20


class LeRobotWriter:
    """Запись эпизодов в LeRobotDataset v3.0 (Parquet + MP4).

    В Уроке 4 заполните схему ``features`` и словарь одного кадра ``frame``.
    """

    def __init__(self, save_dir: str):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(save_dir)
        if root.exists():
            raise FileExistsError(
                f"Путь датасета уже существует: {root}. "
                "Укажите новый путь или удалите незавершённый датасет."
            )

        # TODO (Урок 4, часть 1): опишите схему observation.state, action
        # и observation.images.front. Точные требования приведены в уроке.
        features = ...
        if features is Ellipsis:
            raise NotImplementedError(
                "Заполните TODO features в LeRobotWriter (см. Урок 4)."
            )

        self.dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            fps=FPS,
            root=root,
            robot_type="panda",
            features=features,
            use_videos=True,
            image_writer_threads=4,
            vcodec="h264",
        )

    def add_episode(self, obs_arr, state_arr, act_arr):
        lengths = {len(obs_arr), len(state_arr), len(act_arr)}
        if len(lengths) != 1:
            raise ValueError(
                "Длины observation, state и action должны совпадать: "
                f"{len(obs_arr)}, {len(state_arr)}, {len(act_arr)}"
            )

        for obs, state, action in zip(obs_arr, state_arr, act_arr, strict=True):
            # TODO (Урок 4, часть 2): соберите словарь frame с состоянием,
            # действием, изображением и текстовым описанием задачи.
            frame = ...
            if frame is Ellipsis:
                raise NotImplementedError(
                    "Заполните TODO frame в LeRobotWriter (см. Урок 4)."
                )
            self.dataset.add_frame(frame)

        self.dataset.save_episode()

    def finalize(self):
        self.dataset.finalize()


def parse_args():
    parser = argparse.ArgumentParser(description="Автоматический сбор данных")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--only_success", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format",
        choices=["npz", "lerobot"],
        default="npz",
        help="npz (Урок 3) или lerobot (Урок 4)",
    )
    return parser.parse_args()


def load_rl_policy(checkpoint_path: str, device: str) -> RLPolicy:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy = RLPolicy(
        state_dim=ck.get("state_dim", 29),
        action_dim=ck.get("action_dim", 8),
        hidden_dims=ck.get("hidden_dims", (512, 256, 128)),
    )
    policy.load_state_dict(ck["model_state_dict"])
    policy.to(device)
    policy.eval()
    return policy


def collect_episode(env, policy, device, rng_seed):
    """Собрать один эпизод опорной политики."""
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

    length = len(action_list)
    obs_arr = np.stack(obs_list[:length]).astype(np.uint8)
    state_arr = np.stack(state_list[:length]).astype(np.float32)
    act_arr = np.stack(action_list).astype(np.float32)
    dones = np.zeros(length, dtype=np.float32)
    dones[-1] = 1.0
    return obs_arr, state_arr, act_arr, dones, int(success)


def save_npz_episode(save_dir, episode_index, obs_arr, act_arr, dones, success):
    filename = os.path.join(save_dir, f"episode_{episode_index:04d}.npz")
    np.savez(
        filename,
        obs=obs_arr,
        actions=act_arr,
        dones=dones,
        success=int(success),
    )
    return filename


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"[collect_data] Checkpoint: {args.checkpoint} | format: {args.format} | "
        f"seed: {args.seed} | device: {device}"
    )
    policy = load_rl_policy(args.checkpoint, device)
    env = PandaPickCubeEnv()

    writer = None
    if args.format == "lerobot":
        writer = LeRobotWriter(args.save_dir)
    else:
        os.makedirs(args.save_dir, exist_ok=True)

    saved, attempts = 0, 0
    total_lengths = []
    started_at = time.time()
    rng = np.random.RandomState(args.seed)

    try:
        while saved < args.num_episodes:
            attempts += 1
            obs_arr, state_arr, act_arr, dones, success = collect_episode(
                env, policy, device, rng.randint(0, 2**31)
            )
            if args.only_success and not success:
                continue

            if writer is not None:
                writer.add_episode(obs_arr, state_arr, act_arr)
            else:
                save_npz_episode(
                    args.save_dir, saved, obs_arr, act_arr, dones, success
                )

            saved += 1
            total_lengths.append(obs_arr.shape[0])
            if saved % 100 == 0 or saved == args.num_episodes:
                success_rate = saved / attempts * 100 if attempts else 0
                print(
                    f"[collect_data] Episode {saved}/{args.num_episodes} saved | "
                    f"attempts: {attempts} | SR: {success_rate:.1f}% | "
                    f"avg_len: {np.mean(total_lengths):.0f}"
                )
    finally:
        if writer is not None:
            writer.finalize()
        env.close()

    print(
        f"[collect_data] Done. Saved {saved} episodes ({args.format}) to "
        f"{args.save_dir}/ за {time.time() - started_at:.0f}s"
    )


if __name__ == "__main__":
    main()
