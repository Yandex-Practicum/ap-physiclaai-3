"""Дистилляция скриптового эксперта в RLPolicy (MLP по privileged state).

Запускается при подготовке курса (не студентами). Собирает пары
(privileged_state, action) от детерминированного скриптового контроллера и
обучает MLP-политику регрессией (MSE). Результат сохраняется в формате,
совместимом с collect_data.py и inference.py.

Запуск (внутри контейнера):
    python3 scripts/distill_expert.py --episodes 250 --epochs 300 --save checkpoints/rl_expert.pt
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import PandaPickCubeEnv
from model import RLPolicy
from expert_scripted import ScriptedExpert


def collect(env, expert, n_episodes, seed0=0, noise=0.15):
    """Сбор (state, действие_эксперта). Исполняем действие с шумом (DART):
    робот посещает возмущённые состояния, но метка — чистое действие эксперта.
    Это учит будущую политику возвращаться на траекторию (борьба с накоплением ошибок)."""
    env._get_obs = lambda: np.zeros((84, 84, 3), np.uint8)  # рендер не нужен
    States, Actions = [], []
    succ = 0
    rng = np.random.RandomState(0)
    for i in range(n_episodes):
        env.reset(seed=seed0 + i); expert.reset()
        s = False
        for _ in range(env.episode_length):
            state = env.get_privileged_state()
            a = expert.act()
            States.append(state); Actions.append(a)
            a_exec = a.copy()
            a_exec[:7] = np.clip(a[:7] + rng.randn(7) * noise, -1, 1)  # шум только в суставы
            _, s, done = env.step(a_exec)
            if done: break
        succ += int(s)
    print(f"[collect] {n_episodes} эп., SR эксперта(с шумом)={succ/n_episodes:.0%}, пар={len(States)}")
    return np.array(States, np.float32), np.array(Actions, np.float32)


def dagger_collect(env, expert, policy, device, n_episodes, seed0=20000):
    """Прогон ученика; на каждом шаге метим состояние действием эксперта."""
    env._get_obs = lambda: np.zeros((84, 84, 3), np.uint8)
    States, Actions = [], []
    policy.eval()
    for i in range(n_episodes):
        env.reset(seed=seed0 + i); expert.reset()
        st = env.get_privileged_state()
        for _ in range(env.episode_length):
            a_expert = expert.act()                  # «правильное» действие
            States.append(st); Actions.append(a_expert)
            with torch.no_grad():                    # но идём по действию ученика
                a_student = policy(torch.from_numpy(st).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            _, s, done = env.step(a_student)
            st = env.get_privileged_state()
            if done: break
    policy.train()
    print(f"[dagger] собрано {len(States)} корректирующих пар")
    return np.array(States, np.float32), np.array(Actions, np.float32)


def train(policy, opt, lossf, Xt, Yt, epochs, batch, tag=""):
    N = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(N, device=Xt.device); tot = 0.0
        for b in range(0, N, batch):
            idx = perm[b:b + batch]
            opt.zero_grad(); loss = lossf(policy(Xt[idx]), Yt[idx])
            loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if (ep + 1) % 50 == 0:
            print(f"  {tag} epoch {ep+1}/{epochs} mse={tot/N:.5f}")


def evaluate(env, policy, device, n=40, seed0=5000):
    env._get_obs = lambda: np.zeros((84, 84, 3), np.uint8)
    policy.eval(); succ = 0
    for i in range(n):
        env.reset(seed=seed0 + i)
        st = env.get_privileged_state()
        s = False
        for _ in range(env.episode_length):
            with torch.no_grad():
                a = policy(torch.from_numpy(st).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            _, s, done = env.step(a)
            st = env.get_privileged_state()
            if done: break
        succ += int(s)
    policy.train()
    return succ / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=250)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save", type=str, default="checkpoints/rl_expert.pt")
    p.add_argument("--dagger", type=int, default=3, help="число итераций DAgger")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = PandaPickCubeEnv()
    expert = ScriptedExpert(env)

    X, Y = collect(env, expert, args.episodes)
    Xt = torch.from_numpy(X).to(device); Yt = torch.from_numpy(Y).to(device)

    policy = RLPolicy(state_dim=29, action_dim=8).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    lossf = nn.MSELoss()

    train(policy, opt, lossf, Xt, Yt, args.epochs, args.batch, tag="BC")
    sr = evaluate(env, policy, device)
    print(f"[distill] SR после BC = {sr:.0%}")

    for it in range(args.dagger):
        Xd, Yd = dagger_collect(env, expert, policy, device, n_episodes=120, seed0=20000 + it * 1000)
        Xt = torch.cat([Xt, torch.from_numpy(Xd).to(device)])
        Yt = torch.cat([Yt, torch.from_numpy(Yd).to(device)])
        train(policy, opt, lossf, Xt, Yt, args.epochs // 2, args.batch, tag=f"DAgger{it+1}")
        sr = evaluate(env, policy, device)
        print(f"[distill] SR после DAgger {it+1} = {sr:.0%} (всего пар={len(Xt)})")

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "state_dim": 29, "action_dim": 8, "hidden_dims": (512, 256, 128),
    }, args.save)
    print(f"[distill] сохранено → {args.save}")
    env.close()


if __name__ == "__main__":
    main()
