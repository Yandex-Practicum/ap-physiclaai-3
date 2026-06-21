"""PandaPickCube environment: MuJoCo-based pick-and-place с камерой на гриппере."""

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
SUCCESS_HEIGHT = 0.55
SUCCESS_DIST = 0.05

HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])


class PandaPickCubeEnv:
    """Franka Panda pick-and-place environment.

    Observation: RGB image from gripper camera (84, 84, 3), uint8.
    Action: 8-dim vector — 7 joint position deltas + 1 gripper command.
    Privileged state: concatenation of joint positions, joint velocities,
                      cube position, cube quaternion, target position, gripper state.
    """

    def __init__(self, render_mode: Optional[str] = None):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.renderer.Renderer(self.model, height=OBS_SIZE, width=OBS_SIZE)
        self.render_mode = render_mode

        self._viewer_renderer = None
        if render_mode == "human":
            self._viewer_renderer = mujoco.renderer.Renderer(
                self.model, height=480, width=640
            )

        self.dt = self.model.opt.timestep
        self.episode_length = EPISODE_LENGTH
        self.action_dim = ACTION_DIM
        self._step_count = 0
        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_cam")

        self._cube_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        ]

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        elif not hasattr(self, "_rng"):
            self._rng = np.random.RandomState()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:NUM_ARM_JOINTS + NUM_FINGERS] = HOME_QPOS

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
        gripper_cmd = (action[NUM_ARM_JOINTS] + 1.0) / 2.0 * 0.04

        current_qpos = self.data.qpos[:NUM_ARM_JOINTS].copy()
        target_qpos = current_qpos + joint_deltas

        for i in range(NUM_ARM_JOINTS):
            jnt_id = i
            lo = self.model.jnt_range[jnt_id, 0]
            hi = self.model.jnt_range[jnt_id, 1]
            target_qpos[i] = np.clip(target_qpos[i], lo, hi)

        self.data.ctrl[:NUM_ARM_JOINTS] = target_qpos
        self.data.ctrl[NUM_ARM_JOINTS] = gripper_cmd
        self.data.ctrl[NUM_ARM_JOINTS + 1] = gripper_cmd

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
        cube_quat = self._get_cube_quat()
        target_pos = self.model.body_pos[self._target_body_id].copy()
        gripper_pos = self.data.xpos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        ].copy()

        return np.concatenate([
            joint_pos,        # 9
            joint_vel,        # 7
            cube_pos,         # 3
            cube_quat,        # 4
            target_pos,       # 3
            gripper_pos,      # 3
        ]).astype(np.float32)  # 29

    def _get_cube_quat(self) -> np.ndarray:
        adr = self._cube_qpos_adr
        return self.data.qpos[adr + 3:adr + 7].copy()

    def _check_success(self) -> bool:
        cube_pos = self.data.xpos[self._cube_body_id]
        target_pos = self.model.body_pos[self._target_body_id]
        dist = np.linalg.norm(cube_pos - target_pos)
        return bool(dist < SUCCESS_DIST and cube_pos[2] > 0.45)

    def render_viewer(self) -> Optional[np.ndarray]:
        if self._viewer_renderer is None:
            return None
        self._viewer_renderer.update_scene(self.data)
        return self._viewer_renderer.render().copy()

    @property
    def privileged_state_dim(self) -> int:
        return 29

    def close(self):
        self.renderer.close()
        if self._viewer_renderer is not None:
            self._viewer_renderer.close()
