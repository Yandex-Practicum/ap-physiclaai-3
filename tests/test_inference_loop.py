"""Unit-тест для цикла closed-loop инференса run_episode_bc (Урок 6).

Проверяет, что студент корректно реализовал цикл: эпизод доходит до успеха,
возвращает (success, steps), на таймауте возвращает (False, episode_length),
и инференс идёт под torch.no_grad(). Тест не зависит от MuJoCo — использует
фиктивные среду и модель с тем же API, что у реального проекта.
"""
import numpy as np
import torch
import torch.nn as nn

from inference import run_episode_bc


class DummyEnv:
    """Имитация PandaPickCubeEnv: reset(seed)->obs, step(action)->(obs, success, done)."""
    episode_length = 10

    def __init__(self, succeed_at=5):
        self.succeed_at = succeed_at
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return np.zeros((84, 84, 3), dtype=np.uint8)

    def step(self, action):
        assert np.asarray(action).shape == (8,), "action должен быть вектором из 8 чисел"
        self.t += 1
        obs = np.zeros((84, 84, 3), dtype=np.uint8)
        success = self.t >= self.succeed_at
        done = success or self.t >= self.episode_length
        return obs, success, done


class GradCheckPolicy(nn.Module):
    """Модель нужной формы; запоминает, был ли отключён градиент в forward."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(84 * 84 * 3, 8)
        self.grad_was_enabled = None

    def forward(self, obs):
        self.grad_was_enabled = torch.is_grad_enabled()
        b = obs.shape[0]
        return self.lin(obs.reshape(b, -1).float())


def test_returns_success_and_steps():
    env = DummyEnv(succeed_at=5)
    policy = GradCheckPolicy()
    success, steps = run_episode_bc(env, policy, device="cpu", seed=0)
    assert bool(success) is True
    assert steps == 5


def test_timeout_returns_false():
    env = DummyEnv(succeed_at=999)  # успех недостижим
    policy = GradCheckPolicy()
    success, steps = run_episode_bc(env, policy, device="cpu", seed=0)
    assert bool(success) is False
    assert steps == env.episode_length


def test_uses_no_grad():
    env = DummyEnv(succeed_at=3)
    policy = GradCheckPolicy()
    run_episode_bc(env, policy, device="cpu", seed=0)
    assert policy.grad_was_enabled is False, \
        "инференс должен выполняться под torch.no_grad()"
