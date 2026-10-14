import os
import time
import json
import pickle
import zmq
import tyro
import imageio
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

import robocasa
import robosuite
from robosuite.controllers import load_composite_controller_config

ee_to_cam = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.05],
        [0.0, 0.0, -1.0, -0.097],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Robocasa -> pretrain: eef axes rotated +90 deg around z.
# R_pretrain = R_robocasa @ EEF_AXES_XFORM
# R_robocasa = R_pretrain @ EEF_AXES_XFORM.T
EEF_AXES_XFORM = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

TASK_MAX_STEPS = {
    # Pick and place tasks
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    # Door tasks
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    # Drawer tasks
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    # Stove tasks
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    # Sink tasks
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    # Coffee tasks
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    # Microwave tasks
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}

camera_names = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def render_obs(env, camera_names, base2world, camera_height=256, camera_width=256):
    rgbs = []
    for cam_name in camera_names:
        rgb = env.sim.render(
            height=camera_height, width=camera_width, camera_name=cam_name, depth=False, segmentation=False
        )
        rgb = rgb[::-1].copy()
        rgbs.append(rgb)

    rgbs = np.stack(rgbs, axis=0)
    rgbs_view = rgbs.transpose(1, 0, 2, 3).reshape(camera_height, -1, 3)
    rgbs_norm = rgbs.astype(np.float32) / 127.5 - 1.0

    controller = env.robots[0].composite_controller
    eef_pos, eef_mat = (
        controller.part_controllers[controller.arms[0]].ref_pos,
        controller.part_controllers[controller.arms[0]].ref_ori_mat,
    )
    eef2world = np.eye(4)
    eef2world[:3, :3] = eef_mat
    eef2world[:3, 3] = eef_pos

    eef2base = np.linalg.inv(base2world) @ eef2world
    eef2base_pos = eef2base[:3, 3]
    rot_mat = eef2base[:3, :3] @ EEF_AXES_XFORM  # robocasa -> pretrain
    rot_quat = R.from_matrix(rot_mat).as_quat(canonical=True)[..., [3, 0, 1, 2]]  # xyzw -> wxyz

    gripper_openness = (
        controller.part_controllers[list(controller.grippers.keys())[0]].joint_pos[0:1]
        / controller.part_controllers[list(controller.grippers.keys())[0]].actuator_max[0]
    )

    zero_padding = np.zeros(8)
    eef_states = np.concatenate([eef2base_pos, rot_quat, gripper_openness, zero_padding])

    return rgbs_view, rgbs_norm, eef_states


def create_env(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=256,
    camera_heights=256,
    seed=None,
    render_onscreen=False,
    # robocasa-related configs
    obj_instance_split="B",
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,  # (not render_onscreen),
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env


@dataclass
class Args:
    action_length: int = 32
    save_root_dir: str = "./eval_results/robocasa/"
    env_global_rank: int = 0
    """Global rank of this client across all machines and environments"""
    world_size: int = 1
    """Total number of environment clients across all machines (WORLD_SIZE * num_envs_per_machine)"""
    num_evals_per_worker: int = 5
    server_addr: str = "localhost"
    server_port: int = 10086
    """Broker frontend port (must match policy_broker.py --frontend_port)"""
    cfg: float = 0.0


def main(args: Args):
    env_name_list = list(TASK_MAX_STEPS.keys())
    env_name = env_name_list[args.env_global_rank % len(env_name_list)]

    global_rank = args.env_global_rank
    world_size = args.world_size

    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(f"tcp://{args.server_addr}:{args.server_port}")

    info = {}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(args.num_evals_per_worker)):
        env = create_env(
            env_name=env_name,
            render_onscreen=False,
            seed=global_rank * args.num_evals_per_worker + rollout_i,  # set seed=None to run unseeded
        )
        env.reset()

        controller = env.robots[0].composite_controller
        base_pos, base_mat = (
            controller.part_controllers[controller.arms[0]].origin_pos,
            controller.part_controllers[controller.arms[0]].origin_ori,
        )
        base2world = np.eye(4)
        base2world[:3, :3] = base_mat
        base2world[:3, 3] = base_pos

        # run rollouts with random actions and save video
        num_steps = TASK_MAX_STEPS[env_name]

        video_array = []

        print(f"Rollout {rollout_i} / {args.num_evals_per_worker} started: {env_name} - {env.get_ep_meta()['lang']}")

        step_i = 0
        success = False
        while step_i < num_steps:
            _, rgbs, eef_states = render_obs(env, camera_names, base2world)

            data_batch = {
                "env_rank": global_rank,
                "rollout_id": rollout_i,
                "step_id": step_i,
                "video": rgbs.copy(),
                "proprios": eef_states.copy(),
                "prompt": [env.get_ep_meta()["lang"]],
                "cfg": args.cfg,
            }

            socket.send(pickle.dumps(data_batch))
            result = pickle.loads(socket.recv())  # shape: [Ta, Da]
            action = result["actions"]

            action = action[: args.action_length]
            pad_action = np.zeros(env.action_spec[0].shape)
            for ai in range(action.shape[0]):
                pad_action[:7] = action[ai]

                if step_i % 4 == 0:
                    video_img, _, _ = render_obs(env, camera_names, base2world)
                    video_array.append(video_img)

                env.step(pad_action)
                step_i += 1

                if env._check_success():
                    success = True
                    num_success_rollouts += 1
                    break

                if step_i >= num_steps:
                    success = False
                    break

            if success:
                break

        env.close()

        os.makedirs(f"{args.save_root_dir}/{env_name}", exist_ok=True)
        video_path = (
            f"{args.save_root_dir}/{env_name}/{global_rank}_{rollout_i}_{'success' if success else 'failure'}.mp4"
        )
        imageio.mimsave(video_path, video_array, fps=10)
        print(f"Saved video to {video_path}")

    info[env_name] = {
        "num_success_rollouts": num_success_rollouts,
        "num_rollouts": args.num_evals_per_worker,
        "success_rate": num_success_rollouts / args.num_evals_per_worker,
    }

    print(info)
    with open(os.path.join(args.save_root_dir, f"eval_results_{global_rank}.json"), "w") as f:
        json.dump(info, f, indent=4)

    while True:
        end_files = [e for e in os.listdir(args.save_root_dir) if e.endswith(".json")]
        if len(end_files) >= world_size:
            break
        print(
            f"[Rank {global_rank}] Waiting for all end files... ({len(list(end_files))}/{world_size}) files present. Sleeping for 30 seconds."
        )
        time.sleep(30)


if __name__ == "__main__":
    main(tyro.cli(Args))
