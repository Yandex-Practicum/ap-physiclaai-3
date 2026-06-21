"""Автоматический сбор датасета с помощью опорной RL-политики.

Запуск:
    python3 collect_data.py --checkpoint checkpoints/rl_expert.pt --num_episodes 1000 --save_dir dataset/train_1k --only_success --seed 42
    python3 collect_data.py --checkpoint checkpoints/rl_expert.pt --num_episodes 200 --save_dir dataset/eval --only_success --seed 100
"""

import argparse
import os
import time

import numpy as np
import torch

from env import PandaPickCubeEnv
from model import RLPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Автоматический сбор данных")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Путь к чекпоинту опорной RL-политики")
    parser.add_argument("--num_episodes", type=int, required=True,
                        help="Количество успешных эпизодов для сбора")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Папка для сохранения эпизодов")
    parser.add_argument("--only_success", action="store_true",
                        help="Сохранять только успешные эпизоды")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def load_rl_policy(checkpoint_path: str, device: str) -> RLPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    state_dim = checkpoint.get("state_dim", 29)
    action_dim = checkpoint.get("action_dim", 8)
    hidden_dims = checkpoint.get("hidden_dims", (512, 256, 128))

    policy = RLPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()
    return policy


def collect_episode(env, policy, device, rng_seed):
    obs = env.reset(seed=rng_seed)
    state = env.get_privileged_state()

    obs_list = [obs]
    action_list = []

    for _ in range(env.episode_length):
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy(state_tensor).squeeze(0).cpu().numpy()

        obs, success, done = env.step(action)
        state = env.get_privileged_state()

        action_list.append(action)
        obs_list.append(obs)

        if done:
            break

    T = len(action_list)
    obs_arr = np.stack(obs_list[:T]).astype(np.uint8)
    act_arr = np.stack(action_list).astype(np.float32)
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0

    return obs_arr, act_arr, dones, int(success)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[collect_data] Loaded checkpoint: {args.checkpoint}")
    print(f"[collect_data] Target: {args.num_episodes} successful episodes")
    print(f"[collect_data] Seed: {args.seed}")
    print(f"[collect_data] Device: {device}")

    policy = load_rl_policy(args.checkpoint, device)
    env = PandaPickCubeEnv()

    os.makedirs(args.save_dir, exist_ok=True)

    saved = 0
    attempts = 0
    total_lengths = []
    t_start = time.time()
    rng = np.random.RandomState(args.seed)

    while saved < args.num_episodes:
        ep_seed = rng.randint(0, 2**31)
        attempts += 1

        obs_arr, act_arr, dones, success = collect_episode(env, policy, device, ep_seed)

        if args.only_success and not success:
            continue

        filename = os.path.join(args.save_dir, f"episode_{saved:04d}.npz")
        np.savez(filename, obs=obs_arr, actions=act_arr, dones=dones, success=success)

        saved += 1
        total_lengths.append(obs_arr.shape[0])

        if saved % 100 == 0 or saved == args.num_episodes:
            elapsed = time.time() - t_start
            sr = saved / attempts * 100 if attempts > 0 else 0
            avg_len = np.mean(total_lengths) if total_lengths else 0
            print(f"[collect_data] Episode {saved}/{args.num_episodes} saved | "
                  f"attempts: {attempts} | SR: {sr:.1f}% | avg_len: {avg_len:.0f}")

    elapsed = time.time() - t_start
    sr = saved / attempts * 100
    print(f"[collect_data] Done. Saved {saved} episodes to {args.save_dir}/")
    print(f"[collect_data] Total time: {elapsed:.0f}s | Attempts: {attempts} | SR: {sr:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
