# X-WAM Evaluation Guidelines

## Installation

Please clone the whole repository with submodules:

```bash
git clone --recurse-submodules https://github.com/sharinka0715/X-WAM.git
cd X-WAM
```

If you have already cloned without submodules:

```bash
git submodule update --init --recursive
```

### Base Environment

Follow the main [README](../README.md) to install the base environment (PyTorch, FlashAttention, etc.).

### RoboCasa

Refer to `third_party/robocasa/README.md` for installation.

### RoboTwin 2.0

Refer to `third_party/RoboTwin/README.md` for installation. You can ignore the `torch` / `huggingface_hub` version requirements in `third_party/RoboTwin/script/requirements.txt`.

### Fix NumPy versions

If you install all evaluation packages in one environment, you should make sure that NumPy version is compatible (we use `numpy==1.23.5` in our experiments).

## Download Checkpoints

Download the checkpoints from Hugging Face:

```bash
hf download sharinka0715/X-WAM-checkpoints --local-dir checkpoints
```

You also need the Wan2.2-TI2V-5B base weights. Specify the path via `--wan_checkpoint_dir` when launching the policy server.

## Evaluation

The evaluation system uses a broker-server-client architecture:
- **Policy Broker**: middleware that dispatches inference requests from clients to servers
- **Policy Server**: loads the model and performs inference
- **Client**: runs the simulation environment and sends observations to the broker

### Step 1: Start the Policy Broker

```bash
python evaluation/policy_broker.py \
    --frontend_port 10086 \
    --backend_port 10087
```

### Step 2: Start the Policy Server(s)

Launch one or more policy servers (each on a separate GPU):

```bash
# RoboCasa
CUDA_VISIBLE_DEVICES=0 python evaluation/policy_server.py \
    --exp_path checkpoints/robocasa_sft \
    --wan_checkpoint_dir /path/to/wan22_5b \
    --broker_port 10087 \
    --denoise_steps 50 \
    --action_denoise_steps 10

# RoboTwin 2.0
CUDA_VISIBLE_DEVICES=0 python evaluation/policy_server.py \
    --exp_path checkpoints/robotwin_sft \
    --wan_checkpoint_dir /path/to/wan22_5b \
    --broker_port 10087 \
    --denoise_steps 50 \
    --action_denoise_steps 10
```

You can launch multiple servers on different GPUs to parallelize inference:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluation/policy_server.py --exp_path checkpoints/robocasa_sft --wan_checkpoint_dir /path/to/wan22_5b --broker_port 10087 &
CUDA_VISIBLE_DEVICES=2 python evaluation/policy_server.py --exp_path checkpoints/robocasa_sft --wan_checkpoint_dir /path/to/wan22_5b --broker_port 10087 &
```

### Step 3: Start the Evaluation Client(s)

#### RoboCasa

Each client evaluates one task. There are 24 tasks in total, indexed 0–23. Launch one client per task:

```bash
python evaluation/robocasa_client.py \
    --env_global_rank 0 \
    --world_size 24 \
    --num_evals_per_worker 5 \
    --server_port 10086 \
    --save_root_dir ./eval_results/robocasa/
```

To evaluate all 24 tasks in parallel:

```bash
for i in $(seq 0 23); do
    python evaluation/robocasa_client.py \
        --env_global_rank $i \
        --world_size 24 \
        --num_evals_per_worker 100 \
        --server_port 10086 \
        --save_root_dir ./eval_results/robocasa/ &
done
wait
```

#### RoboTwin 2.0

If you are using your own RoboTwin installation (not the submodule), modify `ROBOTWIN_ROOT` at the top of `evaluation/robotwin_client.py`:

```python
ROBOTWIN_ROOT = "/path/to/your/RoboTwin"
```

Then launch evaluation for each task:

```bash
python evaluation/robotwin_client.py \
    --task_name adjust_bottle \
    --task_config demo_randomized \
    --num_evals_per_worker 10 \
    --server_port 10086 \
    --save_root_dir ./eval_results/robotwin/
```

To evaluate all 50 tasks:

```bash
TASKS=(adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle shake_bottle_horiz stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch)

for task in "${TASKS[@]}"; do
    python evaluation/robotwin_client.py \
        --task_name $task \
        --task_config demo_randomized \
        --num_evals_per_worker 100 \
        --server_port 10086 \
        --save_root_dir ./eval_results/robotwin/ &
done
wait
```

## Results

Evaluation results (success rates and rollout videos) are saved to the `--save_root_dir` directory.
