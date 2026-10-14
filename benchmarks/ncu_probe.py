#!/usr/bin/env python3
"""Minimal standalone probe for Nsight Compute: one forward pass, clean exit.

ncu only flushes its .ncu-rep report when the profiled process tree exits
on its own with a normal exit code -- socket_test_optimized_AR.py's server
runs serve_forever() and never exits by itself, so killing it externally
(even gracefully) truncates the report before ncu can write it. This script
reuses the same model-construction and observation-conversion code but
skips the websocket server entirely: run exactly one forward pass on each
rank, then exit normally so ncu's target application reports a clean 0.
"""

import os

if "TRITON_CACHE_DIR" in os.environ:
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        os.environ["TRITON_CACHE_DIR"], f"rank{os.environ.get('LOCAL_RANK', '0')}"
    )

import deepspeed  # noqa: F401  (see socket_test_optimized_AR.py for why this must import first)
import asyncio
import datetime
import logging

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from eval_utils.policy_server import PolicyServerConfig
from socket_test_optimized_AR import ARDroidRoboarenaPolicy, WebsocketPolicyServer
from test_client_AR import _make_zero_observation

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("NCU_PROBE_MODEL_PATH", "./checkpoints/DreamZero-DROID")

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("ip",))
logger.info(f"Rank {rank}/{world_size} using device {device}")

signal_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=600))

policy = GrootSimPolicy(
    embodiment_tag=EmbodimentTag("oxe_droid"),
    model_path=MODEL_PATH,
    device="cuda",
    device_mesh=device_mesh,
)

if rank == 0:
    wrapper_policy = ARDroidRoboarenaPolicy(groot_policy=policy, signal_group=signal_group, output_dir=None)
    server_config = PolicyServerConfig(
        image_resolution=(180, 320),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )
    obs = _make_zero_observation(server_config, prompt="pick up the object")
    logger.info("Running single profiled forward pass on rank 0...")
    actions = wrapper_policy.infer(obs)
    torch.cuda.synchronize()
    logger.info(f"Done. Action shape: {actions.shape}")

    # Cleanly shut down rank 1's worker loop (signal 1 == shutdown, matches
    # WebsocketPolicyServer._worker_loop's handling).
    signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
    signal_tensor.fill_(1)
    dist.broadcast(signal_tensor, src=0, group=signal_group)
else:
    worker_server = WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=0, metadata={}, output_dir=None, signal_group=signal_group
    )
    asyncio.run(worker_server._worker_loop())

dist.barrier()
logger.info(f"Rank {rank} exiting cleanly.")
