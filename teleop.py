"""Ручное телеуправление Franka Panda с записью эпизодов.

Запуск:
    python3 teleop.py --save_dir dataset/manual
    python3 teleop.py --demo
"""

import argparse
import os
import signal
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from env import PandaPickCubeEnv, NUM_ARM_JOINTS

signal.signal(signal.SIGINT, lambda *_: sys.exit(0))


def parse_args():
    parser = argparse.ArgumentParser(description="Телеуправление PandaPickCube")
    parser.add_argument("--save_dir", type=str, default="dataset/manual",
                        help="Папка для сохранения эпизодов")
    parser.add_argument("--demo", action="store_true",
                        help="Демо-режим: случайные действия, без сохранения")
    return parser.parse_args()


class TeleopController:
    """Обработка клавиатурного ввода для управления гриппером."""

    def __init__(self):
        self.action = np.zeros(8, dtype=np.float32)
        self.gripper_open = True
        self.save_requested = False
        self.cancel_requested = False
        self.quit_requested = False
        self._step = 0.3

    def key_callback(self, key):
        self.action[:] = 0.0

        if key == 265:   # Up
            self.action[0] = self._step
        elif key == 264:  # Down
            self.action[0] = -self._step
        elif key == 263:  # Left
            self.action[1] = self._step
        elif key == 262:  # Right
            self.action[1] = -self._step
        elif key == 266:  # PageUp
            self.action[2] = self._step
        elif key == 267:  # PageDown
            self.action[2] = -self._step
        elif key == 32:   # Space — toggle gripper
            self.gripper_open = not self.gripper_open
        elif key == 257:  # Enter — save
            self.save_requested = True
        elif key == 256:  # Escape — cancel
            self.cancel_requested = True

        self.action[NUM_ARM_JOINTS] = 1.0 if self.gripper_open else -1.0


def save_episode(save_dir, obs_list, action_list, success):
    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir) if f.endswith(".npz")]
    idx = len(existing)
    filename = os.path.join(save_dir, f"episode_{idx:04d}.npz")

    obs_arr = np.stack(obs_list).astype(np.uint8)
    act_arr = np.stack(action_list).astype(np.float32)
    dones = np.zeros(len(obs_list), dtype=np.float32)
    dones[-1] = 1.0

    np.savez(filename, obs=obs_arr, actions=act_arr, dones=dones, success=int(success))
    print(f"Эпизод сохранён: {filename} ({len(obs_list)} шагов, success={success})")


def run_demo(env):
    """Демо-режим: случайные действия в MuJoCo viewer."""
    print("Демо-режим: робот выполняет случайные движения.")
    print("Закройте окно для выхода.")

    mj_data = mujoco.MjData(env.model)
    mj_data.qpos[:] = env.data.qpos[:]
    mj_data.qvel[:] = env.data.qvel[:]
    mujoco.mj_forward(env.model, mj_data)

    with mujoco.viewer.launch_passive(env.model, mj_data) as viewer:
        env.reset(seed=42)
        mj_data.qpos[:] = env.data.qpos[:]
        mj_data.qvel[:] = env.data.qvel[:]

        step = 0
        while viewer.is_running():
            action = np.random.uniform(-0.3, 0.3, size=8).astype(np.float32)
            action[7] = 1.0

            obs, success, done = env.step(action)

            mj_data.qpos[:] = env.data.qpos[:]
            mj_data.qvel[:] = env.data.qvel[:]
            mujoco.mj_forward(env.model, mj_data)
            viewer.sync()
            time.sleep(0.05)

            step += 1
            if done:
                env.reset()
                mj_data.qpos[:] = env.data.qpos[:]
                mj_data.qvel[:] = env.data.qvel[:]
                step = 0


def run_teleop(env, save_dir):
    """Интерактивный режим: управление с клавиатуры."""
    controller = TeleopController()

    print("Телеуправление Franka Panda")
    print("  Стрелки ← → ↑ ↓   — перемещение в плоскости")
    print("  PageUp / PageDown   — вверх/вниз")
    print("  Пробел              — открыть/закрыть гриппер")
    print("  Enter               — сохранить эпизод как успешный")
    print("  Esc                 — отменить текущую попытку")
    print()

    mj_data = mujoco.MjData(env.model)

    with mujoco.viewer.launch_passive(env.model, mj_data) as viewer:
        while viewer.is_running():
            obs = env.reset()
            mj_data.qpos[:] = env.data.qpos[:]
            mj_data.qvel[:] = env.data.qvel[:]
            mujoco.mj_forward(env.model, mj_data)
            viewer.sync()

            obs_list = [obs]
            action_list = []
            controller.save_requested = False
            controller.cancel_requested = False

            print("Новый эпизод. Управляйте роботом...")

            while viewer.is_running():
                action = controller.action.copy()
                action_list.append(action)

                obs, success, done = env.step(action)
                obs_list.append(obs)

                mj_data.qpos[:] = env.data.qpos[:]
                mj_data.qvel[:] = env.data.qvel[:]
                mujoco.mj_forward(env.model, mj_data)
                viewer.sync()
                time.sleep(0.05)

                if controller.save_requested:
                    save_episode(save_dir, obs_list[:-1], action_list, success=True)
                    break

                if controller.cancel_requested:
                    print("Попытка отменена.")
                    break

                if done:
                    if success:
                        save_episode(save_dir, obs_list[:-1], action_list, success=True)
                    else:
                        print("Таймаут. Попробуйте снова.")
                    break


def main():
    args = parse_args()
    env = PandaPickCubeEnv()
    env.reset(seed=0)

    try:
        if args.demo:
            run_demo(env)
        else:
            run_teleop(env, args.save_dir)
    finally:
        env.close()


if __name__ == "__main__":
    main()
