"""Детерминированный скриптовый эксперт pick-and-place по privileged state.

Использует точные позиции куба и цели (из MuJoCo) и operational-space IK,
чтобы надёжно решать задачу. Поведение этого контроллера затем дистиллируется
в RLPolicy (см. scripts/distill_expert.py) и сохраняется как rl_expert.pt.
"""

import numpy as np
import mujoco


class ScriptedExpert:
    ARM = 7
    GRASP_L = 0.103   # смещение грасп-точки вдоль локальной +z кисти (к кончикам пальцев)
    ABOVE = 0.10      # высота зависания над кубом
    HOLD_Z = 0.58     # высота подъёма/переноса

    def __init__(self, env):
        self.env = env
        self.m = env.model
        self.d = env.data
        self.hand_id = env._hand_body_id
        self.cube_id = env._cube_body_id
        self.target_id = env._target_body_id
        self._finger_adr = self.m.jnt_qposadr[
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.grasp_timer = 0
        # целевая ориентация гриппера = ориентация в home-позе (смотрит вертикально вниз)
        self._R_target = self.d.xmat[self.hand_id].reshape(3, 3).copy()

    def _grasp_point(self):
        """Мировая позиция точки между кончиками пальцев."""
        R = self.d.xmat[self.hand_id].reshape(3, 3)
        return self.d.xpos[self.hand_id] + R @ np.array([0.0, 0.0, self.GRASP_L])

    def _ik_delta(self, gp_target, gain=3.0, damp=0.12, rot_gain=0.6):
        """6-DOF DLS IK: ведёт грасп-точку к gp_target, удерживая ориентацию гриппера."""
        R = self.d.xmat[self.hand_id].reshape(3, 3)
        point = self.d.xpos[self.hand_id] + R @ np.array([0.0, 0.0, self.GRASP_L])
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jacp, jacr, point, self.hand_id)
        J = np.vstack([jacp[:, :self.ARM], jacr[:, :self.ARM]])  # (6,7)
        pos_err = gp_target - point
        # ошибка ориентации как вектор (R_target @ R_cur^T -> ось*угол)
        Rerr = self._R_target @ R.T
        rot_err = 0.5 * np.array([Rerr[2, 1] - Rerr[1, 2],
                                  Rerr[0, 2] - Rerr[2, 0],
                                  Rerr[1, 0] - Rerr[0, 1]])
        err = np.concatenate([gain * pos_err, rot_gain * rot_err])
        dq = J.T @ np.linalg.solve(J @ J.T + (damp ** 2) * np.eye(6), err)
        q_des = self.d.qpos[:self.ARM] + dq
        delta = (q_des - self.d.ctrl[:self.ARM]) / 0.05
        return np.clip(delta, -1.0, 1.0)

    def act(self):
        cube = self.d.xpos[self.cube_id].copy()
        target = self.m.body_pos[self.target_id].copy()
        gp = self._grasp_point()
        fq = self.d.qpos[self._finger_adr]  # раскрытие пальца: ~0.04 открыт, ~0 закрыт
        a = np.zeros(8, dtype=np.float32)
        grip_open, grip_closed = 1.0, -1.0

        xy_err = np.linalg.norm(gp[:2] - cube[:2])
        # захват: гриппер зажат на кубе (не открыт ~0.04 и не закрыт впустую ~0) и куб рядом
        holding = (0.008 < fq < 0.035) and (np.linalg.norm(gp - cube) < 0.05)

        if holding:
            # удержание: поднять → перенести → опустить к цели
            if cube[2] < self.HOLD_Z - 0.10:
                tgt = np.array([cube[0], cube[1], self.HOLD_Z])
            elif np.linalg.norm(gp[:2] - target[:2]) > 0.03:
                tgt = np.array([target[0], target[1], self.HOLD_Z])
            else:
                tgt = np.array([target[0], target[1], target[2]])
            a[:7] = self._ik_delta(tgt); a[7] = grip_closed
        else:
            # (пере)захват
            if fq < 0.012 and (gp[2] - cube[2]) < 0.03:
                # гриппер закрыт впустую у куба — переоткрыть и отойти вверх
                tgt = np.array([cube[0], cube[1], cube[2] + self.ABOVE])
                a[:7] = self._ik_delta(tgt); a[7] = grip_open
            elif xy_err > 0.015 or (gp[2] - cube[2]) > self.ABOVE + 0.03:
                # не выровнен по XY или слишком высоко — зависнуть над кубом
                tgt = np.array([cube[0], cube[1], cube[2] + self.ABOVE])
                a[:7] = self._ik_delta(tgt); a[7] = grip_open
            elif (gp[2] - cube[2]) > 0.012:
                # выровнен — спуск к центру куба (XY продолжает трекаться)
                tgt = np.array([cube[0], cube[1], cube[2]])
                a[:7] = self._ik_delta(tgt, gain=2.0); a[7] = grip_open
            else:
                # на уровне куба и выровнен — закрыть гриппер
                tgt = np.array([cube[0], cube[1], cube[2]])
                a[:7] = self._ik_delta(tgt, gain=1.5); a[7] = grip_closed

        return a


def rollout(env, expert, seed, render=False):
    env.reset(seed=seed)
    expert.reset()
    success = False
    for t in range(env.episode_length):
        a = expert.act()
        obs, success, done = env.step(a)
        if done:
            break
    return success, t + 1


if __name__ == "__main__":
    import argparse
    from env import PandaPickCubeEnv
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()
    env = PandaPickCubeEnv()
    expert = ScriptedExpert(env)
    succ = 0
    lens = []
    for i in range(args.n):
        s, L = rollout(env, expert, seed=1000 + i)
        succ += int(s)
        lens.append(L)
    print(f"SR={succ}/{args.n}={succ/args.n:.0%}  mean_len={np.mean(lens):.0f}")
    env.close()
