import os
# =============================
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["TORCHINDUCTOR_AUTOTUNE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import time

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import aloha_policy
import numpy as np

config = _config.get_config("pi05_jax_full")
# checkpoint_dir = "/home/zhike/model/openpi-jax/full/1414-60000"
checkpoint_dir = "/home/zhike/model/openpi-jax/full/160-3w/models (1)/models/pretrained_model"


# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

obs = aloha_policy.make_aloha_example()
obs['state'] = np.array([164.707, -30.758, 55.32, 36.983, 92.156, 72.361, 0.0, 17.403, 6.064, -68.643, -27.598, -83.321, -19.6, 0.0])

print(f"[DEBUG] obs.state = {np.asarray(obs['state'])}")

print("[DEBUG] obs keys and value shapes:")
for k, v in obs.items():
    # 嵌套字典
    if isinstance(v, dict):
        shapes = ", ".join(
            f"{ik}: {getattr(iv, 'shape', type(iv).__name__)}"
            for ik, iv in v.items()
        )
        print(f" {k}:dict{{ {shapes} }}")
    elif hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', 'N/A')}")
    else:
        print(f"  {k}: type={type(v).__name__}, value={v}")

for i in range(1):
  t0 = time.monotonic()
  action = policy.infer(obs)
  t1 = time.monotonic()
  t_infer = (t1 - t0) * 1000
  print(f"step {i}: infer time={t_infer:.1f}ms")
  print(f"action[0]: {[f'{x:.2f}' for x in action['actions'][0][:14]]}")
