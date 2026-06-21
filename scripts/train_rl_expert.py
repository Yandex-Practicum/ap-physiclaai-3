"""Обучение RL-эксперта для сбора данных (запускается при подготовке курса, не студентами).

Запуск:
    python3 scripts/train_rl_expert.py --episodes 50000 --save checkpoints/rl_expert.pt

Обучает MLP-политику на privileged state методом простого policy gradient.
После обучения сохраняет чекпоинт в формате, совместимом с collect_data.py и inference.py.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from env import PandaPickCubeEnv
from model import RLPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--save", type=str, default="checkpoints/rl_expert.pt")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = RLPolicy(state_dim=29, action_dim=8).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    env = PandaPickCubeEnv()

    print(f"Training RL expert on {device}...")
    print(f"Episodes: {args.episodes}")

    best_sr = 0.0
    for episode in range(args.episodes):
        seed = np.random.randint(0, 2**31)
        env.reset(seed=seed)
        state = env.get_privileged_state()

        log_probs = []
        rewards = []

        for _ in range(env.episode_length):
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            action = policy(state_t).squeeze(0)
            noise = torch.randn_like(action) * 0.1
            noisy_action = torch.clamp(action + noise, -1, 1)

            log_prob = -0.5 * ((noisy_action - action) / 0.1) ** 2
            log_probs.append(log_prob.sum())

            _, success, done = env.step(noisy_action.detach().cpu().numpy())
            state = env.get_privileged_state()
            rewards.append(1.0 if success else 0.0)

            if done:
                break

        R = sum(rewards)
        policy_loss = -R * sum(log_probs)
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if (episode + 1) % 1000 == 0:
            sr = evaluate(env, policy, device, n_episodes=50)
            print(f"Episode {episode + 1}/{args.episodes} | SR: {sr:.0%}")
            if sr > best_sr:
                best_sr = sr
                save_checkpoint(policy, args.save)
                print(f"  New best: {sr:.0%} → saved to {args.save}")

    sr = evaluate(env, policy, device, n_episodes=100)
    print(f"\nFinal SR: {sr:.0%}")
    save_checkpoint(policy, args.save)
    env.close()


def evaluate(env, policy, device, n_episodes=50):
    policy.eval()
    successes = 0
    for i in range(n_episodes):
        env.reset(seed=10000 + i)
        state = env.get_privileged_state()
        for _ in range(env.episode_length):
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(state_t).squeeze(0).cpu().numpy()
            _, success, done = env.step(action)
            state = env.get_privileged_state()
            if done:
                successes += int(success)
                break
    policy.train()
    return successes / n_episodes


def save_checkpoint(policy, path):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "state_dim": 29,
        "action_dim": 8,
        "hidden_dims": (512, 256, 128),
    }, path)


if __name__ == "__main__":
    main()
