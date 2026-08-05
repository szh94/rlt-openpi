import os
# =============================
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["TORCHINDUCTOR_AUTOTUNE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import aloha_policy

config = _config.get_config("pi05—jax—full")
checkpoint_dir = "/home/zhike/model/openpi-jax/full/sim/models (2)/models/pretrained_model"

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

obs = aloha_policy.make_aloha_example()

print("[DEBUG] obs keys and value shapes:")
for k, v in obs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', 'N/A')}")
    else:
        print(f"  {k}: type={type(v).__name__}, value={v}")

action = policy.infer(obs)
print(action)