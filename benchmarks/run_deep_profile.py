#!/usr/bin/env python3
"""Deep-profiling probe for X-WAM, bypassing the broker/server/client ZMQ
layer entirely and calling XWAMRunner.generate() directly on DROID-
substituted multi-view input -- same pattern used for RynnVLA-002/Fast-WAM/
AVDC/GR00T N1.5 this session.

PROFILE_MODE=nsys (default): 1 warmup + XWAM_NUM_STEPS steady-state calls,
  each wrapped in torch.profiler, Chrome traces written to
  $DEEP_OUTDIR/torch_profile/step{i}.json.
PROFILE_MODE=ncu: 1 warmup + 1 steady-state call only, no torch.profiler.

Both modes bracket only the steady-state call(s) in
cudaProfilerStart()/Stop() so nsys/ncu with --capture-range=cudaProfilerApi
(or --profile-from-start off) exclude model load, torch.compile warmup,
and the first (often-slower) call -- same convention as every other deep
profile this session.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.xwam_runner import XWAMRunner  # noqa: E402

DROID_DIR = REPO_ROOT.parent / "dreamzero" / "debug_image"
EXP_PATH = Path(os.environ.get("XWAM_EXP_PATH", str(REPO_ROOT / "checkpoints" / "robocasa_sft")))
STEPS_CKPT = os.environ["XWAM_STEPS_CKPT"]
WAN_CHECKPOINT_DIR = os.environ.get(
    "XWAM_WAN_CHECKPOINT_DIR",
    "/scratch/sws0/user/karya/wam/Motus/pretrained_models/Wan2.2-TI2V-5B",
)
PROFILE_MODE = os.environ.get("PROFILE_MODE", "nsys")
OUTDIR = Path(os.environ["DEEP_OUTDIR"])


def load_frame(path: Path, resize_hw: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = resize_hw
    return cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)


def build_rgb(video_hw: tuple[int, int]) -> torch.Tensor:
    """[B=1, V=3, C=3, H, W], float in [-1, 1], bf16 -- matches the model's
    own (pred + 1) * 127.5 inverse-normalization in XWAMRunner.generate()."""
    views = [
        load_frame(DROID_DIR / "exterior_image_1_left.mp4", video_hw),  # robot0_agentview_left
        load_frame(DROID_DIR / "exterior_image_2_left.mp4", video_hw),  # robot0_agentview_right
        load_frame(DROID_DIR / "wrist_image_left.mp4", video_hw),       # robot0_eye_in_hand
    ]
    arr = np.stack(views, axis=0).astype(np.float32) / 127.5 - 1.0  # [V, H, W, C]
    arr = arr.transpose(0, 3, 1, 2)  # [V, C, H, W]
    rgb = torch.from_numpy(arr).unsqueeze(0)  # [1, V, C, H, W]
    return rgb


@torch.no_grad()
def main() -> None:
    config = OmegaConf.load(EXP_PATH / "config.yaml")
    config.wan_checkpoint_dir = WAN_CHECKPOINT_DIR
    # action_num is derived at runtime by policy_server.py (not stored in
    # the checkpoint's own config.yaml) -- mirror that exact derivation.
    config.action_num = config.dataset.frame_skip // config.dataset.action_skip
    video_hw = tuple(config.dataset.video_size)  # e.g. [256, 320]

    print(f"Loading XWAMRunner (wan_checkpoint_dir={WAN_CHECKPOINT_DIR})...")
    t_load0 = time.perf_counter()
    model = XWAMRunner(config=config).cuda().bfloat16()
    ckpt_path = EXP_PATH / "checkpoints" / STEPS_CKPT / "checkpoint" / "mp_rank_00_model_states.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["module"])
    model.eval()
    model.model = torch.compile(model.model)
    print(f"Model load: {time.perf_counter() - t_load0:.2f}s")

    rgb = build_rgb(video_hw).cuda().bfloat16()
    proprio = torch.zeros((1, config.proprio_dim), dtype=torch.bfloat16, device="cuda")
    prompt = ["pick up the object"]

    n_steady = 1 if PROFILE_MODE == "ncu" else int(os.environ.get("XWAM_NUM_STEPS", "5"))

    # --- warmup (includes the torch.compile tax): NOT inside
    # cudaProfilerStart/Stop, so nsys/ncu exclude it entirely. ---
    with nvtx.range("warmup_step0"):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, xt_actions, xt_proprios, _ = model.generate(
            rgb=rgb, proprio=proprio, prompt=prompt, seeds=[0],
            early_stop=True, cfg=0.0, run_depth=False,
        )
        torch.cuda.synchronize()
        print(f"warmup: {time.perf_counter() - t0:.2f}s, "
              f"xt_actions shape {tuple(xt_actions.shape)}")

    cudart = torch.cuda.cudart()
    torch_profile_dir = OUTDIR / "torch_profile"
    torch_profile_dir.mkdir(parents=True, exist_ok=True)

    cudart.cudaProfilerStart()
    for i in range(n_steady):
        torch.cuda.synchronize()
        if PROFILE_MODE == "ncu":
            with nvtx.range(f"steady_step_{i}"):
                t0 = time.perf_counter()
                _, xt_actions, xt_proprios, _ = model.generate(
                    rgb=rgb, proprio=proprio, prompt=prompt, seeds=[i],
                    early_stop=True, cfg=0.0, run_depth=False,
                )
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        else:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                with nvtx.range(f"steady_step_{i}"):
                    t0 = time.perf_counter()
                    _, xt_actions, xt_proprios, _ = model.generate(
                        rgb=rgb, proprio=proprio, prompt=prompt, seeds=[i],
                        early_stop=True, cfg=0.0, run_depth=False,
                    )
                    torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            prof.export_chrome_trace(str(torch_profile_dir / f"step{i}.json"))
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
        print(f"steady step {i}: {dt:.3f}s, xt_actions shape {tuple(xt_actions.shape)}")
    cudart.cudaProfilerStop()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak GPU memory allocated: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
