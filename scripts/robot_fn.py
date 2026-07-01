import logging
import threading
import time
from functools import cached_property
from typing import Any
import socket

import numpy as np
import torch
from anyio.functools import lru_cache_items
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.utils import Robot

from sympy.codegen.ast import continue_
from torch.nn.functional import grid_sample
from examples.hansrobot.config_hansrobot import hangsrobotConfig
from examples.hansrobot.RobotTrajectoryController import *
from typing import List

logger = logging.getLogger(__name__)

cps_8892_right = socket.socket()
cps_8892_right.connect(('192.168.1.20', 8892))
cps_8892_left = socket.socket()
cps_8892_left.connect(('192.168.1.30', 8892))


class hansrobot(Robot):
    config_class = hangsrobotConfig
    name = "hansrobot"

    def __init__(self, config: hangsrobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        self.right_robot = RobotTrajectoryController(ip="192.168.1.20",
                                                     port=10003,
                                                     servo_time=0.03,
                                                     lookahead_time=0.4)

        self.left_robot = RobotTrajectoryController(ip="192.168.1.30",
                                                    port=10003,
                                                    servo_time=0.03,
                                                    lookahead_time=0.4)
        self.sym = False
        self.left_last_gripper = 0
        self.right_last_gripper = 0
        # self.left_gripper_state = "open"  # "open", "closed", "moving"
        # self.right_gripper_state = "open"

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {"observation.state": List[float]}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            f"observation.images.{cam}": (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        # for cam in self.cameras:
        #     self.cameras[cam].connect()

        self.left_robot.connect(0)
        self.right_robot.connect(1)
        self.left_robot.gripper_speed(400, 0)
        self.right_robot.gripper_speed(400, 1)

        return None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None  # 25% torque when overloaded

    def setup_motors(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any]:

        # Read arm position
        start = time.perf_counter()

        obs_dict = {}

        # 1. 获取 6 个关节角度
        right_joint_angles = self.right_robot.get_joint(1)  # [a1, a2, a3, a4, a5, a6]
        left_joint_angles = self.left_robot.get_joint(0)
        # 2. 获取夹爪位置
        right_gripper_pos = self.right_robot.gripper_read(1, 1)
        left_gripper_pos = self.left_robot.gripper_read(0, 1)
        right_gripper_value = right_gripper_pos
        left_gripper_value = left_gripper_pos

        # 3. 构建 7 维状态向量：[joint1.pos, ..., joint6.pos, gripper.pos]
        left_state_vector = [
                                float(angle) for angle in left_joint_angles  # 6 个关节
                            ] + [left_gripper_value]

        right_state_vector = [
                                 float(angle) for angle in right_joint_angles  # 6 个关节
                             ] + [right_gripper_value]  # +1 个夹爪
        state_vector = left_state_vector + right_state_vector

        # 4. 将 7 维 float 数组赋给 observation.state
        obs_dict["observation.state"] = state_vector
        right_effort = self.right_robot.get_tcp_force(1)
        left_effort = self.left_robot.get_tcp_force(0)
        obs_dict['effort'] = left_effort + right_effort

        # Capture images from cameras
        # for cam_key, cam in self.cameras.items():
        #     start = time.perf_counter()
        #     # 构造目标键名：observation.images.{cam_key}_image
        #     obs_key = f"observation.images.{cam_key}"
        #     obs_dict[obs_key] = cam.async_read()
        # --- 保存图像（不修改原代码，仅追加）---
        import os
        from pathlib import Path

        SAVE_DIR = Path("./saved_images")
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        # 假设你已经执行了：
        # for cam_key, cam in self.cameras.items():
        #     obs_key = f"observation.images.{cam_key}"
        #     obs_dict[obs_key] = cam.async_read()  # ← 原代码，不修改！

        # ✅ 在这行后面追加保存逻辑：
        for cam_key, cam in self.cameras.items():
            obs_key = f"observation.images.{cam_key}"
            image = cam.async_read()  # ← 原代码，不修改！
            obs_dict[obs_key] = cam.async_read()  # ← 原代码，不修改！
        #
        #     # ==========================================
        #     # ✅ 追加：保存图像到本地文件夹（按 cam_key 分类）
        #     # ==========================================
        #     if image is None:
        #         # print(f"[WARNING] {cam_key}: 图像为 None，跳过保存")
        #         continue
        #
        #     # 创建以 cam_key 命名的子目录
        #     cam_save_dir = SAVE_DIR / cam_key
        #     cam_save_dir.mkdir(parents=True, exist_ok=True)
        #
        #     # 生成时间戳文件名：cam_key_YYYYMMDD_HHMMSS.png
        #     timestamp = time.strftime("%Y%m%d_%H%M%S")
        #     filename = f"{cam_key}_{timestamp}.png"
        #     filepath = cam_save_dir / filename
        #
        #     # 保存图像（支持 PIL 和 numpy）
        #     try:
        #         if hasattr(image, 'save'):  # PIL Image
        #             image.save(filepath)
        #         else:  # numpy array (HWC)
        #             import numpy as np
        #             from PIL import Image
        #             img_pil = Image.fromarray(image.astype(np.uint8))
        #             img_pil.save(filepath)
        #         # print(f"[INFO] 保存图像: {filepath}")
        #     except Exception as e:
        #         print(f"[ERROR] 保存 {cam_key} 图像失败: {e}")

        # print('obs_dict',obs_dict)
        # print('obs_dict', obs_dict)
        return obs_dict

    def send_action(self, action: list) -> dict[str, Any]:

        #print('hansrobot:',action)
        left_joint = action[:6]
        left_gripper = int(action[6])

        # print('left_gripper', left_gripper)

        self.left_robot.start_servo(0)
        self.left_robot.send_trajectory_point(left_joint, cps_8892_left)
        # 夹爪控制
        #self.left_robot.griperControl(1, left_gripper,0)

        if abs(left_gripper - self.left_last_gripper) > 20:
            self.left_robot.griperControl(1, left_gripper, 0)
            self.left_last_gripper = left_gripper



        right_joint = action[7:13]
        right_gripper = int(action[13])
        #print('right_gripper', right_gripper)
        self.right_robot.start_servo(1)
        self.right_robot.send_trajectory_point(right_joint, cps_8892_right)
        #self.right_robot.griperControl(1, right_gripper, 1)

        if abs(right_gripper - self.right_last_gripper) > 20:
            self.right_robot.griperControl(1, right_gripper, 1)
            self.right_last_gripper = right_gripper

        goal_pos_dict = {}

        # 返回格式化后的动作字典
        return goal_pos_dict


    def send_action_safe(self, action: list) -> dict[str, Any]:
        """
        发送动作命令：机械臂安全判断
        action: [left_joint(6) + left_gripper(1) + right_joint(6) + right_gripper(1)] = 共 14 个值
        - 前 6 个：左机械臂关节角度（度）
        - 第 7 个：左夹爪
        - 第 8-13 个：右机械臂关节角度（度）
        - 第 14 个：右夹爪
        """
        JOINT_THRESHOLD = 20.0

        left_joint = action[:6]
        left_gripper = int(action[6])
        right_joint = action[7:13]
        right_gripper = int(action[13])

        left_safe = True
        if hasattr(self, 'left_robot_history'):
            for i in range(6):
                delta = abs(left_joint[i] - self.left_robot_history[i])
                if delta >= JOINT_THRESHOLD:
                    left_safe = False
                    print(f"[SAFETY] Left arm joint {i} delta={delta:.2f} exceeds threshold {JOINT_THRESHOLD}")
                    break

        if left_safe:
            self.left_robot.start_servo(0)
            self.left_robot.send_trajectory_point(left_joint, cps_8892_left)
            self.left_robot_history = list(left_joint)

        right_safe = True
        if hasattr(self, 'right_robot_history'):
            for i in range(6):
                delta = abs(right_joint[i] - self.right_robot_history[i])
                if delta >= JOINT_THRESHOLD:
                    right_safe = False
                    print(f"[SAFETY] Right arm joint {i} delta={delta:.2f} exceeds threshold {JOINT_THRESHOLD}")
                    break

        if right_safe:
            self.right_robot.start_servo(1)
            self.right_robot.send_trajectory_point(right_joint, cps_8892_right)
            self.right_robot_history = list(right_joint)

        if abs(left_gripper - self.left_last_gripper) > 20:
            self.left_robot.griperControl(1, left_gripper, 0)
            self.left_last_gripper = left_gripper

        if abs(right_gripper - self.right_last_gripper) > 20:
            self.right_robot.griperControl(1, right_gripper, 1)
            self.right_last_gripper = right_gripper

        return {}

    def disconnect(self):
        for cam in self.cameras.values():
            cam.disconnect()
        is_connected = False

    def move(self, left_joint, right_joint):
        self.left_robot.set_override(0, 0.5)
        self.right_robot.set_override(1, 0.5)
        self.left_robot.move_to_joint_position(left_joint, 0)
        self.right_robot.move_to_joint_position(right_joint, 1)
        self.left_robot.griperControl(1, 0, 0)
        self.right_robot.griperControl(1, 0, 1)


if __name__ == '__main__':
    our = hansrobot(hansrobot.config_class())
    # action = {'1': 1}
    our.connect()
    # idx = 0
    # while idx < 10:
    #     action1 = [171.646, -30.41, 48.252, 28.213, 82.968, -30.934, 1100.0, 7.237, 29.552, -49.834, -21.884, -79.167, -59.941, 1100.0]
    #     our.send_action(action1)
    #     idx += 1
    # print('obs:', our.get_observation())
    left_joint = [164.70703, -30.7578125, 55.32031, 36.982422, 92.15625, 72.361328]
    right_joint = [17.402344, 6.064453, -68.64258, -27.597656,
                   -83.32031, -19.59961]
    our.move(left_joint, right_joint)

    our.disconnect()