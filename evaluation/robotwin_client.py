import sys
import os

CURR_ROOT = os.getcwd()
ROBOTWIN_ROOT = "./third_party/RoboTwin"
os.chdir(ROBOTWIN_ROOT)

sys.path.append("./script")
sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

print(sys.path)

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path

import yaml
from datetime import datetime
import importlib
from typing import Optional
from dataclasses import dataclass

import tyro
import imageio

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


@dataclass
class Args:
    task_name: str = "adjust_bottle"
    policy_name: str = "X-WAM"
    task_config: str = "demo_randomized"
    instruction_type: str = "unseen"
    seed: int = 0
    num_evals_per_worker: int = 10
    save_root_dir: str = "./eval_results/robotwin/"
    action_length: int = 32
    server_addr: str = "localhost"
    server_port: int = 10086
    cfg: float = 0.0


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(args: Args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = args.task_name
    task_config = args.task_config
    ckpt_setting = None
    policy_name = args.policy_name
    instruction_type = args.instruction_type

    usr_args = {
        "task_name": task_name,
        "task_config": task_config,
        "ckpt_setting": ckpt_setting,
        "policy_name": policy_name,
        "instruction_type": instruction_type,
        "seed": args.seed,
        "action_length": args.action_length,
        "server_addr": args.server_addr,
        "server_port": args.server_port,
        "cfg": args.cfg,
    }

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    task_args["task_name"] = task_name
    task_args["task_config"] = task_config
    task_args["ckpt_setting"] = ckpt_setting

    embodiment_type = task_args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(etype):
        robot_file = _embodiment_types[etype]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(CURR_ROOT) / args.save_root_dir / task_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(task_args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(task_args["domain_randomization"]["random_background"]))
    if task_args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(task_args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(task_args["domain_randomization"]["random_light"]))
    if task_args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(task_args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(task_args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(task_args["domain_randomization"]["random_head_camera_dis"]))
    print("\033[94mHead Camera Config:\033[0m " + str(task_args["camera"]["head_camera_type"]) + ", " +
          str(task_args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(task_args["camera"]["wrist_camera_type"]) + ", " +
          str(task_args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(task_name)
    task_args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(task_args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(task_args["right_embodiment_config"]["arm_joints_name"][1])

    seed = args.seed
    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = args.num_evals_per_worker
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(
        task_name, TASK_ENV, task_args, usr_args, model, st_seed,
        test_num=test_num, save_dir=save_dir,
        instruction_type=instruction_type,
    )
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, "_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")


def eval_policy(task_name, TASK_ENV, task_args, usr_args, model, st_seed,
                test_num=100, save_dir=None, instruction_type=None):
    print(f"\033[34mTask Name: {task_args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {task_args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = task_args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    clear_cache_freq = task_args["clear_cache_freq"]

    task_args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                task_args["render_freq"] = render_freq
                continue
            except Exception:
                TASK_ENV.close_env()
                now_seed += 1
                task_args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            task_args["render_freq"] = render_freq
            continue

        task_args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        succ = False
        all_video_chunks = []
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            video_chunk = eval_func(
                TASK_ENV, model, observation,
                env_rank=0, rollout_id=now_id,
                step_id=TASK_ENV.take_action_cnt, usr_args=usr_args,
            )
            if video_chunk is not None:
                all_video_chunks.append(video_chunk)
            if TASK_ENV.eval_success:
                succ = True
                break

        if save_dir is not None and all_video_chunks:
            suffix = "success" if succ else "failure"
            video_path = save_dir / f"episode_{now_id:04d}_{suffix}.mp4"
            frames = np.concatenate(all_video_chunks, axis=0)
            imageio.mimwrite(
                str(video_path), frames, fps=8,
                codec='libx264',
                output_params=['-crf', '30', '-preset', 'fast', '-pix_fmt', 'yuv420p'],
            )

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{task_args['policy_name']}\033[0m | "
            f"\033[92m{task_args['task_config']}\033[0m | \033[91m{task_args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
            f"\033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    main(tyro.cli(Args))
