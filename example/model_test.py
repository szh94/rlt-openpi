import os
import time

# =============================
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["TORCHINDUCTOR_AUTOTUNE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import aloha_policy
from examples.hansrobot.hansrobot import *

obs_example = aloha_policy.make_aloha_example()
# print(obs_example)

# config = _config.get_config("pi05—jax")
# checkpoint_dir = "/home/zhike/model/openpi-jax/1294-60000"

config = _config.get_config("pi05-hans-pytorch")
checkpoint_dir = "/home/zhike/model/openpi_pytorch/30000"
# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print('load model success')

robot = hansrobot(hangsrobotConfig())
robot.connect()

def convert_image_hwc_to_chw(img):
    # img: [H, W, 3] -> [3, H, W]
    return np.transpose(img, (2, 0, 1))

def get_obs():
    raw_observation = robot.get_observation()
    # print("raw_observation", raw_observation)
    # # [H, W, 3] -> [3, H, W]
    head_img = convert_image_hwc_to_chw(raw_observation['observation.images.head_image'])
    left_img = convert_image_hwc_to_chw(raw_observation['observation.images.left_wrist_image'])
    right_img = convert_image_hwc_to_chw(raw_observation['observation.images.right_wrist_image'])
    obs = {
        'images': {
            "cam_high": head_img,
            "cam_left_wrist": left_img,
            "cam_right_wrist": right_img,
        },
        "state": raw_observation['observation.state'],
        "prompt": "place phone",
    }
    return obs
try:
    obs = get_obs()
    while True:
        # print(obs)
        actions = policy.infer(obs)["actions"]
        for action in actions:
            robot.send_action_safe(action)
finally:
    robot.disconnect()