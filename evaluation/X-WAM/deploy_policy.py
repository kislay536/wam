# import packages and module here
import zmq
import pickle
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation

# Base frame is redefined by rotating the frame +90 deg around its z axis.
# For coordinates expressed in the frame, this is a change of basis by Rz(-90 deg).
BASE_COORD_XFORM = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# New eef axes are defined as x' = z, z' = x, and y' = -y to keep a right-handed frame.
# This matrix maps coordinates from the new eef frame to the old eef frame.
EEF_AXES_XFORM = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def compute_future_poses(initial_proprio: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """
    Given initial pose (xyz+wxyz+gripper) and a sequence of [T,7] deltas (xyz+axisangle+gripper),
    compute absolute future poses in global frame for each step.
    Args:
        initial_proprio: np.ndarray of shape [8], (xyz + quat, wxyz order, gripper)
        actions: np.ndarray of shape [T, 7], (delta_xyz + delta_axisangle + delta_gripper, all global frame, all relative-to-previous)
    Returns:
        poses: np.ndarray of shape [T, 8], (xyz + quat in wxyz order, global frame, gripper)
    """
    T = actions.shape[0]
    poses = np.zeros((T, 8), dtype=np.float64)
    # Start with initial pose
    pos = initial_proprio[:3].astype(np.float64).copy()
    quat_wxyz = initial_proprio[3:7].astype(np.float64)
    rot = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])  # wxyz -> xyzw for scipy
    gripper = initial_proprio[7].astype(np.float64).copy()

    for i in range(T):
        d_pos = actions[i, :3]
        d_axisangle = actions[i, 3:6]
        # Accumulate position
        pos = pos + d_pos
        # Update orientation: compose dR after current quat (i.e., q_new = dR * q_prev)
        dR = Rotation.from_rotvec(d_axisangle)
        rot = dR * rot

        poses[i, :3] = pos
        poses[i, 3:7] = rot.as_quat()[[3, 0, 1, 2]]

        # Accumulate gripper
        gripper = gripper + actions[i, 6]
        poses[i, 7] = gripper
    return poses


def compute_seed(env_rank, rollout_id, step_id):
    """Deterministically derive a uint32 seed from (env_rank, rollout_id, step_id)."""
    key = f"{env_rank}_{rollout_id}_{step_id}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def encode_obs(observation):  # Post-Process Observation
    obs = observation
    # ...
    return obs


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    context = zmq.Context()
    if usr_args.get("direct", False):
        socket = context.socket(zmq.REQ)
    else:
        socket = context.socket(zmq.DEALER)
    socket.connect(f"tcp://{usr_args['server_addr']}:{usr_args['server_port']}")

    return socket


def eval(TASK_ENV, model: zmq.Context, observation, env_rank=0, rollout_id=0, step_id=0, usr_args={}):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    camera_ids = ["head_camera", "left_camera", "right_camera"]
    # Post-Process Observation
    rgbs = []
    for camera_id in camera_ids:
        rgbs.append(observation["observation"][camera_id]["rgb"])

    rgbs = np.stack(rgbs, axis=0)  # [V, H, W, 3]
    rgbs = rgbs.astype(np.float32) / 127.5 - 1.0

    # Read bimanual end-effector poses and actions
    left_endpose = observation["endpose"]["left_endpose"]
    left_gripper = observation["endpose"]["left_gripper"]
    right_endpose = observation["endpose"]["right_endpose"]
    right_gripper = observation["endpose"]["right_gripper"]

    # Left arm: extract xyz and quaternion
    left_xyz = left_endpose[:3] @ BASE_COORD_XFORM.T
    left_quat = np.array(left_endpose[3:])  # wxyz
    left_rot_mat = Rotation.from_quat(left_quat[[1, 2, 3, 0]]).as_matrix()
    left_rot_mat = BASE_COORD_XFORM @ left_rot_mat @ EEF_AXES_XFORM
    left_quat = Rotation.from_matrix(left_rot_mat).as_quat(canonical=True)[..., [3, 0, 1, 2]]  # [B,4] xyzw -> wxyz
    left_arm = np.concatenate([left_xyz, left_quat, [left_gripper]])

    # Right arm: extract xyz and quaternion
    right_xyz = right_endpose[:3] @ BASE_COORD_XFORM.T
    right_quat = np.array(right_endpose[3:])  # wxyz
    right_rot_mat = Rotation.from_quat(right_quat[[1, 2, 3, 0]]).as_matrix()
    right_rot_mat = BASE_COORD_XFORM @ right_rot_mat @ EEF_AXES_XFORM
    right_quat = Rotation.from_matrix(right_rot_mat).as_quat(canonical=True)[..., [3, 0, 1, 2]]  # [B,4] xyzw -> wxyz
    right_arm = np.concatenate([right_xyz, right_quat, [right_gripper]])

    robot_states = np.concatenate([left_arm, right_arm])

    instruction = TASK_ENV.get_instruction()

    data_batch = {
        "env_rank": env_rank,
        "rollout_id": rollout_id,
        "step_id": step_id,
        "video": rgbs,
        # "depths": depths,
        "proprios": robot_states,
        "prompt": [instruction],
        "cfg": usr_args["cfg"],
    }

    model.send(pickle.dumps(data_batch))
    result = pickle.loads(model.recv())  # shape: [Ta, Da]

    actions = result["actions"]

    # delta action control: accumulate deltas into absolute poses, then inverse-transform
    left_actions = compute_future_poses(left_arm, actions[:, 0:7])
    left_gripper = left_actions[:, 7:8]
    left_xyz = left_actions[:, :3] @ BASE_COORD_XFORM
    left_quat = left_actions[:, 3:7]
    left_mat = Rotation.from_quat(left_quat[..., [1, 2, 3, 0]]).as_matrix()
    left_mat = BASE_COORD_XFORM.T @ left_mat @ EEF_AXES_XFORM.T
    left_quat = Rotation.from_matrix(left_mat).as_quat(canonical=True)[..., [3, 0, 1, 2]]

    right_actions = compute_future_poses(right_arm, actions[:, 7:14])
    right_gripper = right_actions[:, 7:8]
    right_xyz = right_actions[:, :3] @ BASE_COORD_XFORM
    right_quat = right_actions[:, 3:7]
    right_mat = Rotation.from_quat(right_quat[..., [1, 2, 3, 0]]).as_matrix()
    right_mat = BASE_COORD_XFORM.T @ right_mat @ EEF_AXES_XFORM.T
    right_quat = Rotation.from_matrix(right_mat).as_quat(canonical=True)[..., [3, 0, 1, 2]]

    actions = np.concatenate([left_xyz, left_quat, left_gripper, right_xyz, right_quat, right_gripper], axis=1)
    action_length = min(actions.shape[0], usr_args["action_length"])

    gt_videos = []
    for ai in range(action_length):  # Execute each step of the action
        action = actions[ai]
        TASK_ENV.take_action(action, action_type="ee")
        if (ai + 1) % 4 == 0:
            observation = TASK_ENV.get_obs()
            rgbs = []
            for camera_id in camera_ids:
                rgb = observation["observation"][camera_id]["rgb"]
                rgbs.append(rgb)
            rgbs = np.stack(rgbs, axis=0)  # [V, H, W, 3]
            gt_videos.append(rgbs)

    gt_videos = np.stack(gt_videos, axis=0)  # [T, V, H, W, 3]
    gt_videos = gt_videos.transpose(0, 2, 1, 3, 4)  # [T, V, H, W, 3] -> [T, H, V, W, 3]
    gt_videos = gt_videos.reshape(gt_videos.shape[0], gt_videos.shape[1], -1, 3)  # [T, H, V, W, 3] -> [T, H, V*W, 3]

    return gt_videos


def reset_model(model):
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass
