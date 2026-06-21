"""PandaPickCube environment: MuJoCo Menagerie Franka Panda + куб + камера на гриппере."""

import os
from typing import Optional

import mujoco
import mujoco.renderer
import numpy as np

SCENE_XML = os.path.join(os.path.dirname(__file__), "assets", "scene.xml")

NUM_ARM_JOINTS = 7
NUM_FINGERS = 2
ACTION_DIM = NUM_ARM_JOINTS + 1  # 7 joints + 1 gripper command
OBS_SIZE = 84
EPISODE_LENGTH = 300
SUCCESS_DIST = 0.05


class PandaPickCubeEnv:
    """Franka Panda pick-and-place environment.

    Observation: RGB image from gripper camera (84, 84, 3), uint8.
    Action: 8-dim vector — 7 joint position deltas + 1 gripper command.
    Privileged state: joint positions, joint velocities, cube pose, target pos, gripper pos.
    """

    def __init__(self, render_mode: Optional[str] = None):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.renderer.Renderer(self.model, height=OBS_SIZE, width=OBS_SIZE)

        self.dt = self.model.opt.timestep
        self.episode_length = EPISODE_LENGTH
        self.action_dim = ACTION_DIM
        self._step_count = 0

        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_cam")

        self._cube_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._cube_qpos_adr = self.model.jnt_qposadr[self._cube_jnt_id]

        self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        elif not hasattr(self, "_rng"):
            self._rng = np.random.RandomState()

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        cube_x = 0.4 + self._rng.uniform(-0.1, 0.1)
        cube_y = self._rng.uniform(-0.15, 0.15)
        cube_z = 0.425
        adr = self._cube_qpos_adr
        self.data.qpos[adr:adr + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]

        target_x = 0.4 + self._rng.uniform(-0.1, 0.1)
        target_y = self._rng.uniform(-0.15, 0.15)
        target_z = 0.55
        self.model.body_pos[self._target_body_id] = [target_x, target_y, target_z]

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        joint_deltas = action[:NUM_ARM_JOINTS] * 0.05
        gripper_cmd = (action[NUM_ARM_JOINTS] + 1.0) / 2.0 * 255.0

        current_ctrl = self.data.ctrl[:NUM_ARM_JOINTS].copy()
        target_ctrl = current_ctrl + joint_deltas

        for i in range(NUM_ARM_JOINTS):
            lo = self.model.actuator_ctrlrange[i, 0]
            hi = self.model.actuator_ctrlrange[i, 1]
            target_ctrl[i] = np.clip(target_ctrl[i], lo, hi)

        self.data.ctrl[:NUM_ARM_JOINTS] = target_ctrl
        self.data.ctrl[NUM_ARM_JOINTS] = gripper_cmd

        n_substeps = max(1, int(0.05 / self.dt))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        success = self._check_success()
        done = success or self._step_count >= self.episode_length
        return obs, success, done

    def _get_obs(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self._cam_id)
        return self.renderer.render().copy()

    def get_privileged_state(self) -> np.ndarray:
        joint_pos = self.data.qpos[:NUM_ARM_JOINTS + NUM_FINGERS].copy()
        joint_vel = self.data.qvel[:NUM_ARM_JOINTS].copy()
        cube_pos = self.data.xpos[self._cube_body_id].copy()
        adr = self._cube_qpos_adr
        cube_quat = self.data.qpos[adr + 3:adr + 7].copy()
        target_pos = self.model.body_pos[self._target_body_id].copy()
        gripper_pos = self.data.xpos[self._hand_body_id].copy()

        return np.concatenate([
            joint_pos,        # 9
            joint_vel,        # 7
            cube_pos,         # 3
            cube_quat,        # 4
            target_pos,       # 3
            gripper_pos,      # 3
        ]).astype(np.float32)  # 29

    def _check_success(self) -> bool:
        cube_pos = self.data.xpos[self._cube_body_id]
        target_pos = self.model.body_pos[self._target_body_id]
        dist = np.linalg.norm(cube_pos - target_pos)
        return bool(dist < SUCCESS_DIST and cube_pos[2] > 0.45)

    @property
    def privileged_state_dim(self) -> int:
        return 29

    def close(self):
        self.renderer.close()
