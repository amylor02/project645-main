import numpy as np
import torch
from torch.utils.data import Dataset
import pymotion.rotations.dual_quat as dquat
from pymotion.ops.skeleton import to_root_dual_quat
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import pymotion.rotations.quat as quat
import pymotion.rotations.ortho6d_torch as ortho6d
import pymotion.rotations.ortho6d as ortho6d_np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.special import expit

USE_GLOBAL_HEIGHT = True  # root keeps an explicit height channel
# Legacy switch retained for compatibility with older exports.
USE_CANONICAL_XZ_POSITIONS = False
ROOT_CHANNELS_ARE_GLOBAL_POSITIONS = False
DOWNSAMPLE_FACTOR = 8
WINDOW_STEP_SIZE = 8
LMA_FRAME_STRIDE = 4
SYNTHETIC_CONTACT_JOINT_COUNT = 1
SYNTHETIC_CONTACT_CHANNELS = 4
SYNTHETIC_CONTACT_PAD_CHANNELS = 5
LMA_KEYS = (
    "BODY",
    "EFFORT_WEIGHT_STRONG",
    "EFFORT_TIME_SUDDEN",
    "EFFORT_FLOW_BOUND",
    "SHAPE",
    "SPACE",
)
ALL_LMA_KEYS = LMA_KEYS


def _moving_average_windows(sequence, centers, window_size):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"Expected [T, C] sequence, got shape {sequence.shape}")

    window_size = max(int(window_size), 1)
    half_window = window_size // 2
    outputs = []
    for center in centers:
        start = max(int(center) - half_window, 0)
        end = min(start + window_size, sequence.shape[0])
        start = max(end - window_size, 0)
        outputs.append(sequence[start:end].mean(axis=0, dtype=np.float32))

    if not outputs:
        return np.zeros((0, sequence.shape[1]), dtype=np.float32)
    return np.stack(outputs, axis=0).astype(np.float32)


def _compute_rough_root_trajectory(
    root_features, stride=DOWNSAMPLE_FACTOR, avg_window=3
):
    root_features = np.asarray(root_features, dtype=np.float32)
    if root_features.ndim != 2 or root_features.shape[1] != 9:
        raise ValueError(
            f"Expected root trajectory source with shape [T, 9], got {root_features.shape}"
        )

    if root_features.shape[0] == 0:
        return np.zeros((0, 9), dtype=np.float32)

    centers = np.arange(0, root_features.shape[0], max(int(stride), 1), dtype=np.int64)
    return _moving_average_windows(root_features, centers, avg_window)


def _root_position_channels(global_pos, origin=None):
    root_pos = global_pos[:, :3].clone()
    if origin is not None:
        root_pos[:, 0] = root_pos[:, 0] - origin[0]
        root_pos[:, 2] = root_pos[:, 2] - origin[2]
    return root_pos


def _root_position_features(global_pos, origin=None):
    root_pos = _root_position_channels(global_pos, origin=origin)
    if not USE_GLOBAL_HEIGHT:
        return root_pos
    return torch.cat([root_pos, global_pos[:, 1:2]], dim=1)


def _root_velocity_height_features(global_pos):
    root_pos = global_pos[:, :3].clone()
    root_motion = torch.zeros_like(root_pos)
    if root_pos.shape[0] > 1:
        root_motion[1:, 0] = root_pos[1:, 0] - root_pos[:-1, 0]
        root_motion[1:, 2] = root_pos[1:, 2] - root_pos[:-1, 2]
    root_motion[:, 1] = root_pos[:, 1]
    return root_motion


def _build_contact_joint_features(contact_binary, device):
    if not isinstance(contact_binary, torch.Tensor):
        contact_binary = torch.tensor(contact_binary, dtype=torch.float32, device=device)
    else:
        contact_binary = contact_binary.to(device=device, dtype=torch.float32)

    if contact_binary.dim() == 1:
        contact_binary = contact_binary.unsqueeze(-1)

    joint = torch.zeros(
        (contact_binary.shape[0], SYNTHETIC_CONTACT_CHANNELS + SYNTHETIC_CONTACT_PAD_CHANNELS),
        dtype=torch.float32,
        device=device,
    )
    joint[:, : min(SYNTHETIC_CONTACT_CHANNELS, contact_binary.shape[1])] = contact_binary[
        :, :SYNTHETIC_CONTACT_CHANNELS
    ]
    return joint


def split_motion_joints(frame_major_motion, synthetic_joint_count=SYNTHETIC_CONTACT_JOINT_COUNT):
    if synthetic_joint_count <= 0:
        return frame_major_motion, None
    if frame_major_motion.shape[-2] < synthetic_joint_count:
        raise ValueError(
            f"Expected at least {synthetic_joint_count} joints, got {frame_major_motion.shape}"
        )
    return (
        frame_major_motion[..., :-synthetic_joint_count, :],
        frame_major_motion[..., -synthetic_joint_count:, :],
    )


ROOT_TRANSLATION_ANCHOR_FRAME = 2


def _clamp_root_anchor_frame(frame_count, anchor_frame=None):
    if frame_count <= 0:
        return 0
    if anchor_frame is None:
        anchor_frame = ROOT_TRANSLATION_ANCHOR_FRAME
    return int(np.clip(int(anchor_frame), 0, frame_count - 1))


def _resolve_root_anchor_np(initial_root_pos, frame_count, anchor_frame=None):
    anchor_frame = _clamp_root_anchor_frame(frame_count, anchor_frame)
    initial_root_pos = np.asarray(initial_root_pos, dtype=np.float32)
    if initial_root_pos.ndim == 1:
        return initial_root_pos.reshape(3), anchor_frame
    if initial_root_pos.ndim == 2 and initial_root_pos.shape[-1] == 3:
        return initial_root_pos[anchor_frame].reshape(3), anchor_frame
    raise ValueError(
        f"Expected root anchor position [3] or root trajectory [F,3], got {initial_root_pos.shape}"
    )


def _resolve_root_anchor_torch(initial_root_pos, frame_count, anchor_frame=None):
    anchor_frame = _clamp_root_anchor_frame(frame_count, anchor_frame)
    if initial_root_pos.dim() == 1:
        return initial_root_pos.reshape(3), anchor_frame
    if initial_root_pos.dim() == 2 and initial_root_pos.shape[-1] == 3:
        if initial_root_pos.shape[0] == frame_count:
            return initial_root_pos[anchor_frame], anchor_frame
        return initial_root_pos, anchor_frame
    if initial_root_pos.dim() == 3 and initial_root_pos.shape[-1] == 3:
        return initial_root_pos[:, anchor_frame, :], anchor_frame
    raise ValueError(
        f"Expected root anchor tensor [3], [B,3], [F,3], or [B,F,3], got {tuple(initial_root_pos.shape)}"
    )


def integrate_root_translation_np(root_joint_features, initial_root_pos, anchor_frame=None):
    root_joint_features = np.asarray(root_joint_features, dtype=np.float32)
    initial_root_pos, anchor_frame = _resolve_root_anchor_np(
        initial_root_pos,
        root_joint_features.shape[0],
        anchor_frame=anchor_frame,
    )
    positions = np.zeros((root_joint_features.shape[0], 3), dtype=np.float32)
    cum_x = np.cumsum(root_joint_features[:, 6], axis=0)
    cum_z = np.cumsum(root_joint_features[:, 8], axis=0)
    anchor_x = cum_x[anchor_frame] if cum_x.size > 0 else 0.0
    anchor_z = cum_z[anchor_frame] if cum_z.size > 0 else 0.0
    positions[:, 0] = initial_root_pos[0] + (cum_x - anchor_x)
    positions[:, 1] = root_joint_features[:, 7]
    positions[:, 2] = initial_root_pos[2] + (cum_z - anchor_z)
    return positions


def integrate_root_translation_torch(root_joint_features, initial_root_pos, anchor_frame=None):
    initial_root_pos = initial_root_pos.to(device=root_joint_features.device, dtype=root_joint_features.dtype)
    frame_dim = root_joint_features.dim() - 2
    initial_root_pos, anchor_frame = _resolve_root_anchor_torch(
        initial_root_pos,
        root_joint_features.shape[frame_dim],
        anchor_frame=anchor_frame,
    )
    positions = torch.zeros(
        root_joint_features.shape[:-1] + (3,),
        dtype=root_joint_features.dtype,
        device=root_joint_features.device,
    )
    cum_x = torch.cumsum(root_joint_features[..., 6], dim=frame_dim)
    cum_z = torch.cumsum(root_joint_features[..., 8], dim=frame_dim)
    anchor_x = cum_x.select(dim=frame_dim, index=anchor_frame).unsqueeze(frame_dim)
    anchor_z = cum_z.select(dim=frame_dim, index=anchor_frame).unsqueeze(frame_dim)
    positions[..., 0] = initial_root_pos[..., 0:1] + (cum_x - anchor_x)
    positions[..., 1] = root_joint_features[..., 7]
    positions[..., 2] = initial_root_pos[..., 2:3] + (cum_z - anchor_z)
    return positions


def _resample_1d(values, target_len):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    target_len = max(int(target_len), 1)
    if values.size == target_len:
        return values.astype(np.float32)
    if values.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if values.size == 1:
        return np.full(target_len, float(values[0]), dtype=np.float32)

    src = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(dst, src, values).astype(np.float32)


def _coerce_lma_dict(lma_features, target_len=None):
    if lma_features is None:
        return None

    if isinstance(lma_features, dict):
        coerced = {}
        for key in LMA_KEYS:
            if key not in lma_features:
                continue
            values = np.asarray(lma_features[key], dtype=np.float32).reshape(-1)
            coerced[key] = (
                _resample_1d(values, target_len) if target_len is not None else values
            )
        return coerced or None

    values = np.asarray(lma_features, dtype=np.float32)
    if values.ndim == 1:
        if values.size % len(LMA_KEYS) != 0:
            return None
        values = values.reshape(-1, len(LMA_KEYS))
    if values.ndim != 2 or values.shape[0] == 0:
        return None

    if values.shape[1] < len(LMA_KEYS):
        padded = np.zeros((values.shape[0], len(LMA_KEYS)), dtype=np.float32)
        padded[:, : values.shape[1]] = values
        values = padded

    coerced = {}
    for index, key in enumerate(LMA_KEYS):
        column = values[:, index].astype(np.float32)
        coerced[key] = (
            _resample_1d(column, target_len) if target_len is not None else column
        )
    return coerced


class TrainMotionData(Dataset):
    def __init__(self, param, scale, device):
        self.motions = []
        self.norm_motions = []
        self.means_motions = []
        self.var_motions = []
        self.param = param
        self.scale = scale
        self.device = device
        self.rots = []
        self.global_pos = []
        # self.downsample_factor = self.param["stride_encoder_conv"] ** 3
        self.downsample_factor = DOWNSAMPLE_FACTOR

    def add_motion(
        self,
        offsets,
        global_pos,
        rotations,
        parents,
        pos=None,
        og_rots=None,
        end_sites=None,
        end_sites_parents=None,
        lma_features=None,
        style_id=None,
        style_label=None,
    ):
        """
        Parameters:
        -----------
        offsets: np.array of shape (n_joints, 3)
        global_pos: np.array of shape (n_frames, 3)
        rotations: np.array of shape (n_frames, n_joints, 4) (quaternions)
        parents: np.array of shape (n_joints)

        Returns:
        --------
        self.motions:
            offsets: tensor of shape (n_joints, 3)
            dqs: tensor of shape (windows_size, n_joints * 8) (dual quaternions)
            displacement: tensor of shape (windows_size, 3)
        """
        frames = rotations.shape[0]
        pad_size = 0
        if frames < self.param["window_size"]:
            pad_size = self.param["window_size"] - frames
            last_global_pos = global_pos[-1]
            last_rotation = rotations[-1]
            last_pos = pos[-1]
            global_pos = np.concatenate(
                [global_pos] + [last_global_pos[np.newaxis, :]] * pad_size, axis=0
            )
            rotations = np.concatenate(
                [rotations] + [last_rotation[np.newaxis, :, :]] * pad_size, axis=0
            )
            if og_rots is not None:
                og_rots = np.concatenate(
                    [og_rots] + [og_rots[-1][np.newaxis, :, :]] * pad_size, axis=0
                )
            pos = np.concatenate(
                [pos] + [last_pos[np.newaxis, :, :]] * pad_size, axis=0
            )
            frames = self.param["window_size"]

        elif frames % self.param["window_size"] != 0:
            pad_size = self.param["window_size"] - frames % self.param["window_size"]
            last_global_pos = global_pos[-1]
            last_rotation = rotations[-1]
            last_pos = pos[-1]
            global_pos = np.concatenate(
                [global_pos] + [last_global_pos[np.newaxis, :]] * pad_size, axis=0
            )
            rotations = np.concatenate(
                [rotations] + [last_rotation[np.newaxis, :, :]] * pad_size, axis=0
            )
            if og_rots is not None:
                og_rots = np.concatenate(
                    [og_rots] + [og_rots[-1][np.newaxis, :, :]] * pad_size, axis=0
                )
            pos = np.concatenate(
                [pos] + [last_pos[np.newaxis, :, :]] * pad_size, axis=0
            )
            frames += pad_size

        assert frames >= self.param["window_size"]

        # because human has tpose at initial frame
        rotations[0] = rotations[1]
        global_pos[0] = global_pos[1]
        pos[0] = pos[1]

        self.rots = rotations.copy()

        # create dual quaternions
        fake_global_pos = np.zeros((frames, 3))
        dqs = to_root_dual_quat(rotations, fake_global_pos, parents, offsets)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = ortho6d.from_dual_quat(
            dqs
        )  # <-------------------------------------------------------------
        dqs = torch.flatten(dqs, 1, 2)
        offsets = torch.from_numpy(offsets).type(torch.float32).to(self.device)
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        self.global_pos = global_pos.clone()
        displacement = _root_velocity_height_features(global_pos)

        dqs[:, 6:9] = displacement

        if og_rots is not None:
            rotations = og_rots

        rots = torch.from_numpy(rotations).type(torch.float32).to(self.device)
        self.rots = ortho6d.from_quat(rots[:, 0]).reshape(-1, 6)
        self.rots = torch.cat([self.rots, displacement], dim=-1)
        root_traj_avg_window = int(self.param.get("rough_root_avg_window", 3))
        rough_root_traj = _compute_rough_root_trajectory(
            self.rots.detach().cpu().numpy(),
            stride=self.downsample_factor,
            avg_window=root_traj_avg_window,
        )
        rough_root_traj = (
            torch.from_numpy(rough_root_traj).type(torch.float32).to(self.device)
        )
        style_id_tensor = None
        if style_id is not None:
            style_id_tensor = torch.tensor(
                int(style_id), dtype=torch.long, device=self.device
            )
        rots = rots.flatten(start_dim=1, end_dim=2).permute(1, 0).unsqueeze(0)
        # change global_pos to (1, 3, frames)
        global_pos = global_pos.permute(1, 0).unsqueeze(0)
        rot_order = np.tile(
            ["y", "x", "z"], (rotations.shape[0], rotations.shape[1], 1)
        )

        rots_euler = quat.to_euler(rotations, rot_order)

        tags = compute_tags(
            pos,
            downsample_factor=self.downsample_factor,
            rots=rots_euler[:, 0],
            is_deg=False,
            quats=rotations,
            padded_frames=pad_size,
            skeleton_height=self.param["skeleton_height"],
            head_height=self.param["head_height"],
            head_idx=self.param["head_idx"],
            feet_idx=self.param["feet_idxs"],
            not_dog=self.param["not_dog"],
            feet_contact_threshold=self.param["feet_contact_threshold"],
            window_size=self.param["window_size"],
            shoulder_idx=self.param.get("shoulder_idxs", [5, 9]),
            lma_features=lma_features,
            parents=parents,
            visualize=self.param.get("visualize_on_load", False),
            training=True,
            sparse_joints=self.param.get("sparse_joints"),
        )

        for key in tags:
            if not isinstance(tags[key], torch.Tensor):
                tags[key] = torch.tensor(tags[key]).float().to(self.device)

        contact_joint = _build_contact_joint_features(
            tags["foot_contact_binary"],
            device=self.device,
        )
        dqs = torch.cat([dqs, contact_joint], dim=1)

        for start in range(0, frames, self.param["window_step"]):
            end = start + self.param["window_size"]

            tags_step = WINDOW_STEP_SIZE
            waypoints_step = 8

            if end < frames:
                _dqs_window = dqs[start:end]
                displacement_window = displacement[start:end]
                motion = {
                    "offsets": offsets,
                    "denorm_offsets": offsets.clone(),
                    "dqs": _dqs_window,
                    "displacement": displacement_window,
                    "foot_positions": pos[
                        start:end,
                        list(
                            dict.fromkeys(
                                int(index) for index in self.param.get("feet_idxs", [])
                            )
                        ),
                        :,
                    ],
                    "end_sites": end_sites,
                    "end_sites_parents": np.array(end_sites_parents),
                    "tags": {
                        "velo_foot": tags["velo_foot"][start:end],
                        "yaw_sin": tags["yaw_sin"][start:end],
                        "yaw_cos": tags["yaw_cos"][start:end],
                        "ctrl_forward_alignment": tags["ctrl_forward_alignment"][
                            start:end
                        ],
                        "ctrl_lateral_alignment": tags["ctrl_lateral_alignment"][
                            start:end
                        ],
                        "ctrl_velocity": tags["ctrl_velocity"][start:end],
                        "ctrl_acceleration": tags["ctrl_acceleration"][start:end],
                        "ctrl_yaw_rate": tags["ctrl_yaw_rate"][start:end],
                        "ctrl_yaw_accel": tags["ctrl_yaw_accel"][start:end],
                        "ctrl_height": tags["ctrl_height"][start:end],
                        "ctrl_head_height": tags["ctrl_head_height"][start:end],
                        "ctrl_roll": tags["ctrl_roll"][start:end],
                        "ctrl_foot_labels": tags["ctrl_foot_labels"][start:end],
                        "ctrl_vertical_velocity": tags["ctrl_vertical_velocity"][
                            start:end
                        ],
                        "smooth_root_pos": tags["smooth_root_pos"][start:end],
                        "foot_contact_probs": tags["foot_contact_probs"][start:end],
                        "foot_contact_binary": tags["foot_contact_binary"][start:end],
                        "support_contact": tags["support_contact"][start:end],
                        "foot_contact_latent": tags["foot_contact_latent"][
                            start
                            // self.downsample_factor : end
                            // self.downsample_factor
                        ],
                        "support_contact_latent": tags["support_contact_latent"][
                            start
                            // self.downsample_factor : end
                            // self.downsample_factor
                        ],
                        "rough_root_traj": rough_root_traj[
                            start
                            // self.downsample_factor : end
                            // self.downsample_factor
                        ],
                        **(
                            {"style_id": style_id_tensor.clone()}
                            if style_id_tensor is not None
                            else {}
                        ),
                        **{
                            k: tags[k][
                                start // LMA_FRAME_STRIDE : end // LMA_FRAME_STRIDE
                            ]
                            for k in LMA_KEYS
                            if k in tags
                        },
                    },
                    "global_pos": self.global_pos[start:end],
                    "rots": self.rots[start:end],
                    "style_label": style_label if style_label is not None else "",
                }
                self.motions.append(motion)

        # Means
        motion_mean = {
            "offsets": torch.mean(offsets, dim=0).to(self.device),
            "dqs": torch.mean(dqs, dim=0).to(self.device),
            "displacement": torch.mean(displacement, dim=0).to(self.device),
            "yaw_sin": torch.mean(tags["yaw_sin"], dim=0).to(self.device),
            "yaw_cos": torch.mean(tags["yaw_cos"], dim=0).to(self.device),
            "ctrl_forward_alignment": torch.mean(
                tags["ctrl_forward_alignment"], dim=0
            ).to(self.device),
            "ctrl_lateral_alignment": torch.mean(
                tags["ctrl_lateral_alignment"], dim=0
            ).to(self.device),
            "ctrl_velocity": torch.mean(tags["ctrl_velocity"], dim=0).to(self.device),
            "ctrl_acceleration": torch.mean(tags["ctrl_acceleration"], dim=0).to(
                self.device
            ),
            "ctrl_yaw_rate": torch.mean(tags["ctrl_yaw_rate"], dim=0).to(self.device),
            "ctrl_yaw_accel": torch.mean(tags["ctrl_yaw_accel"], dim=0).to(self.device),
            "ctrl_height": torch.mean(tags["ctrl_height"], dim=0).to(self.device),
            "ctrl_head_height": torch.mean(tags["ctrl_head_height"], dim=0).to(
                self.device
            ),
            "ctrl_roll": torch.mean(tags["ctrl_roll"], dim=0).to(self.device),
            "ctrl_vertical_velocity": torch.mean(
                tags["ctrl_vertical_velocity"], dim=0
            ).to(self.device),
            "rots": torch.mean(self.rots, dim=0).to(self.device),
            "smooth_root_pos": torch.mean(tags["smooth_root_pos"], dim=0).to(
                self.device
            ),
            "rough_root_traj": torch.mean(rough_root_traj, dim=0).to(self.device),
        }
        for _lma_k in ALL_LMA_KEYS:
            if _lma_k in tags:
                motion_mean[_lma_k] = torch.mean(tags[_lma_k], dim=0).to(self.device)

        self.means_motions.append(motion_mean)
        # Stds
        motion_var = {
            "offsets": torch.var(offsets, dim=0).to(self.device),
            "dqs": torch.var(dqs, dim=0).to(self.device),
            "displacement": torch.var(displacement, dim=0).to(self.device),
            "yaw_sin": torch.var(tags["yaw_sin"], dim=0).to(self.device),
            "yaw_cos": torch.var(tags["yaw_cos"], dim=0).to(self.device),
            "ctrl_forward_alignment": torch.var(
                tags["ctrl_forward_alignment"], dim=0
            ).to(self.device),
            "ctrl_lateral_alignment": torch.var(
                tags["ctrl_lateral_alignment"], dim=0
            ).to(self.device),
            "ctrl_velocity": torch.var(tags["ctrl_velocity"], dim=0).to(self.device),
            "ctrl_acceleration": torch.var(tags["ctrl_acceleration"], dim=0).to(
                self.device
            ),
            "ctrl_yaw_rate": torch.var(tags["ctrl_yaw_rate"], dim=0).to(self.device),
            "ctrl_yaw_accel": torch.var(tags["ctrl_yaw_accel"], dim=0).to(self.device),
            "ctrl_height": torch.var(tags["ctrl_height"], dim=0).to(self.device),
            "ctrl_head_height": torch.var(tags["ctrl_head_height"], dim=0).to(
                self.device
            ),
            "ctrl_roll": torch.var(tags["ctrl_roll"], dim=0).to(self.device),
            "ctrl_vertical_velocity": torch.var(
                tags["ctrl_vertical_velocity"], dim=0
            ).to(self.device),
            "rots": torch.var(self.rots, dim=0).to(self.device),
            "smooth_root_pos": torch.var(tags["smooth_root_pos"], dim=0).to(
                self.device
            ),
            "rough_root_traj": torch.var(rough_root_traj, dim=0).to(self.device),
        }
        for _lma_k in ALL_LMA_KEYS:
            if _lma_k in tags:
                motion_var[_lma_k] = torch.var(tags[_lma_k], dim=0).to(self.device)
        self.var_motions.append(motion_var)

    def normalize(self):
        """
        Normalize motions by means and stds

        Returns:
        --------
        self.norm_motions:
            offsets: tensor of shape (n_joints, 3)
            dqs: tensor of shape (windows_size, n_joints * 8) (dual quaternions)
        """
        offsets_means = torch.stack([m["offsets"] for m in self.means_motions], dim=0)
        offsets_vars = torch.stack([s["offsets"] for s in self.var_motions], dim=0)
        dqs_means = torch.stack([m["dqs"] for m in self.means_motions], dim=0)
        dqs_vars = torch.stack([s["dqs"] for s in self.var_motions], dim=0)

        rots_means = torch.stack([m["rots"] for m in self.means_motions], dim=0)
        rots_vars = torch.stack([s["rots"] for s in self.var_motions], dim=0)

        displacement_means = torch.stack(
            [m["displacement"] for m in self.means_motions], dim=0
        )
        displacement_vars = torch.stack(
            [s["displacement"] for s in self.var_motions], dim=0
        )

        self.means = {
            "offsets": torch.mean(offsets_means, dim=0).to(self.device),
            "dqs": torch.mean(dqs_means, dim=0).to(self.device),
            "displacement": torch.mean(displacement_means, dim=0).to(self.device),
            "yaw_sin": torch.mean(
                torch.stack([m["yaw_sin"] for m in self.means_motions], dim=0), dim=0
            ).to(self.device),
            "yaw_cos": torch.mean(
                torch.stack([m["yaw_cos"] for m in self.means_motions], dim=0), dim=0
            ).to(self.device),
            "ctrl_forward_alignment": torch.mean(
                torch.stack(
                    [m["ctrl_forward_alignment"] for m in self.means_motions], dim=0
                ),
                dim=0,
            ).to(self.device),
            "ctrl_lateral_alignment": torch.mean(
                torch.stack(
                    [m["ctrl_lateral_alignment"] for m in self.means_motions], dim=0
                ),
                dim=0,
            ).to(self.device),
            "ctrl_velocity": torch.mean(
                torch.stack([m["ctrl_velocity"] for m in self.means_motions], dim=0),
                dim=0,
            ).to(self.device),
            "ctrl_acceleration": torch.mean(
                torch.stack(
                    [m["ctrl_acceleration"] for m in self.means_motions], dim=0
                ),
                dim=0,
            ).to(self.device),
            "ctrl_yaw_rate": torch.mean(
                torch.stack([m["ctrl_yaw_rate"] for m in self.means_motions], dim=0),
                dim=0,
            ).to(self.device),
            "ctrl_yaw_accel": torch.mean(
                torch.stack([m["ctrl_yaw_accel"] for m in self.means_motions], dim=0),
                dim=0,
            ).to(self.device),
            "ctrl_height": torch.mean(
                torch.stack([m["ctrl_height"] for m in self.means_motions], dim=0),
                dim=0,
            ).to(self.device),
            "ctrl_head_height": torch.mean(
                torch.stack([m["ctrl_head_height"] for m in self.means_motions], dim=0),
                dim=0,
            ).to(self.device),
            "ctrl_roll": torch.mean(
                torch.stack([m["ctrl_roll"] for m in self.means_motions], dim=0), dim=0
            ).to(self.device),
            "ctrl_vertical_velocity": torch.mean(
                torch.stack(
                    [m["ctrl_vertical_velocity"] for m in self.means_motions], dim=0
                ),
                dim=0,
            ).to(self.device),
            "rots": torch.mean(rots_means, dim=0).to(self.device),
            "smooth_root_pos": torch.mean(
                torch.stack([m["smooth_root_pos"] for m in self.means_motions], dim=0),
                dim=0,
            ).to(self.device),
            "rough_root_traj": torch.mean(
                torch.stack([m["rough_root_traj"] for m in self.means_motions], dim=0),
                dim=0,
            ).to(self.device),
        }
        for _lma_k in ALL_LMA_KEYS:
            _lma_vals = [m[_lma_k] for m in self.means_motions if _lma_k in m]
            if _lma_vals:
                self.means[_lma_k] = torch.mean(
                    torch.stack(_lma_vals, dim=0), dim=0
                ).to(self.device)

        # Source: https://stats.stackexchange.com/a/26647
        self.stds = {
            "offsets": torch.sqrt(torch.mean(offsets_vars, dim=0)).to(self.device),
            "dqs": torch.sqrt(torch.mean(dqs_vars, dim=0)).to(self.device),
            "displacement": torch.sqrt(torch.mean(displacement_vars, dim=0)).to(
                self.device
            ),
            "yaw_sin": torch.sqrt(
                torch.mean(
                    torch.stack([s["yaw_sin"] for s in self.var_motions], dim=0), dim=0
                )
            ).to(self.device),
            "yaw_cos": torch.sqrt(
                torch.mean(
                    torch.stack([s["yaw_cos"] for s in self.var_motions], dim=0), dim=0
                )
            ).to(self.device),
            "ctrl_forward_alignment": torch.sqrt(
                torch.mean(
                    torch.stack(
                        [s["ctrl_forward_alignment"] for s in self.var_motions], dim=0
                    ),
                    dim=0,
                )
            ).to(self.device),
            "ctrl_lateral_alignment": torch.sqrt(
                torch.mean(
                    torch.stack(
                        [s["ctrl_lateral_alignment"] for s in self.var_motions], dim=0
                    ),
                    dim=0,
                )
            ).to(self.device),
            "ctrl_velocity": torch.sqrt(
                torch.mean(
                    torch.stack([s["ctrl_velocity"] for s in self.var_motions], dim=0),
                    dim=0,
                )
            ).to(self.device),
            "ctrl_acceleration": torch.sqrt(
                torch.mean(
                    torch.stack(
                        [s["ctrl_acceleration"] for s in self.var_motions], dim=0
                    ),
                    dim=0,
                )
            ).to(self.device),
            "ctrl_yaw_rate": torch.sqrt(
                torch.mean(
                    torch.stack([s["ctrl_yaw_rate"] for s in self.var_motions], dim=0),
                    dim=0,
                )
            ).to(self.device),
            "ctrl_yaw_accel": torch.sqrt(
                torch.mean(
                    torch.stack([s["ctrl_yaw_accel"] for s in self.var_motions], dim=0),
                    dim=0,
                )
            ).to(self.device),
            "ctrl_height": torch.sqrt(
                torch.mean(
                    torch.stack([s["ctrl_height"] for s in self.var_motions], dim=0),
                    dim=0,
                )
            ).to(self.device),
            "ctrl_head_height": torch.sqrt(
                torch.mean(
                    torch.stack(
                        [s["ctrl_head_height"] for s in self.var_motions], dim=0
                    ),
                    dim=0,
                )
            ).to(self.device),
            "ctrl_roll": torch.sqrt(
                torch.mean(
                    torch.stack([s["ctrl_roll"] for s in self.var_motions], dim=0),
                    dim=0,
                )
            ).to(self.device),
            "ctrl_vertical_velocity": torch.sqrt(
                torch.mean(
                    torch.stack(
                        [s["ctrl_vertical_velocity"] for s in self.var_motions], dim=0
                    ),
                    dim=0,
                )
            ).to(self.device),
            "rots": torch.sqrt(torch.mean(rots_vars, dim=0)).to(self.device),
            "smooth_root_pos": torch.sqrt(
                torch.mean(
                    torch.stack(
                        [s["smooth_root_pos"] for s in self.var_motions], dim=0
                    ),
                    dim=0,
                )
            ).to(self.device),
            "rough_root_traj": torch.sqrt(
                torch.mean(
                    torch.stack(
                        [s["rough_root_traj"] for s in self.var_motions], dim=0
                    ),
                    dim=0,
                )
            ).to(self.device),
        }
        for _lma_k in ALL_LMA_KEYS:
            _lma_vars = [s[_lma_k] for s in self.var_motions if _lma_k in s]
            if _lma_vars:
                self.stds[_lma_k] = torch.sqrt(
                    torch.mean(torch.stack(_lma_vars, dim=0), dim=0)
                ).to(self.device)

        # --- Stabilization: i think near zero stds causing instability during training---
        eps = 1e-8
        for k, v in list(self.stds.items()):
            if torch.is_tensor(v):
                v_fixed = v.clone()
                mask = v_fixed.abs() < eps
                if mask.any():
                    # keep normalization well-defined; equivalent to "no scaling" for that channel
                    v_fixed[mask] = 1.0
                self.stds[k] = v_fixed

        for key in self.stds:
            if (
                torch.count_nonzero(self.stds[key]) != torch.numel(self.stds[key])
                or (self.stds[key] < 1e-6).any()
            ):
                print(f"WARNING: {key} stds are zero")
                self.stds[key][self.stds[key] < 1e-6] = 1

        # Normalized
        for motion in self.motions:
            norm_motion = {
                "offsets": (motion["offsets"] - self.means["offsets"])
                / self.stds["offsets"],
                "dqs": (motion["dqs"] - self.means["dqs"]) / self.stds["dqs"],
                "displacement": (motion["displacement"] - self.means["displacement"])
                / self.stds["displacement"],
                "tags": {
                    "velo_foot": motion["tags"]["velo_foot"],
                    "yaw_sin": (motion["tags"]["yaw_sin"] - self.means["yaw_sin"])
                    / self.stds["yaw_sin"],
                    "yaw_cos": (motion["tags"]["yaw_cos"] - self.means["yaw_cos"])
                    / self.stds["yaw_cos"],
                    "ctrl_forward_alignment": (
                        motion["tags"]["ctrl_forward_alignment"]
                        - self.means["ctrl_forward_alignment"]
                    )
                    / self.stds["ctrl_forward_alignment"],
                    "ctrl_lateral_alignment": (
                        motion["tags"]["ctrl_lateral_alignment"]
                        - self.means["ctrl_lateral_alignment"]
                    )
                    / self.stds["ctrl_lateral_alignment"],
                    "ctrl_velocity": (
                        motion["tags"]["ctrl_velocity"] - self.means["ctrl_velocity"]
                    )
                    / self.stds["ctrl_velocity"],
                    "ctrl_acceleration": (
                        motion["tags"]["ctrl_acceleration"]
                        - self.means["ctrl_acceleration"]
                    )
                    / self.stds["ctrl_acceleration"],
                    "ctrl_yaw_accel": (
                        motion["tags"]["ctrl_yaw_accel"] - self.means["ctrl_yaw_accel"]
                    )
                    / self.stds["ctrl_yaw_accel"],
                    "ctrl_yaw_rate": (
                        motion["tags"]["ctrl_yaw_rate"] - self.means["ctrl_yaw_rate"]
                    )
                    / self.stds["ctrl_yaw_rate"],
                    "ctrl_height": (
                        motion["tags"]["ctrl_height"] - self.means["ctrl_height"]
                    )
                    / self.stds["ctrl_height"],
                    "ctrl_head_height": (
                        motion["tags"]["ctrl_head_height"]
                        - self.means["ctrl_head_height"]
                    )
                    / self.stds["ctrl_head_height"],
                    "ctrl_roll": (motion["tags"]["ctrl_roll"] - self.means["ctrl_roll"])
                    / self.stds["ctrl_roll"],
                    "ctrl_foot_labels": (motion["tags"]["ctrl_foot_labels"]),
                    "ctrl_vertical_velocity": (
                        motion["tags"]["ctrl_vertical_velocity"]
                        - self.means["ctrl_vertical_velocity"]
                    )
                    / self.stds["ctrl_vertical_velocity"],
                    "smooth_root_pos": (
                        motion["tags"]["smooth_root_pos"]
                        - self.means["smooth_root_pos"]
                    )
                    / self.stds["smooth_root_pos"],
                    "foot_contact_probs": motion["tags"]["foot_contact_probs"],
                    "foot_contact_binary": motion["tags"]["foot_contact_binary"],
                    "support_contact": motion["tags"]["support_contact"],
                    "foot_contact_latent": motion["tags"]["foot_contact_latent"],
                    "support_contact_latent": motion["tags"]["support_contact_latent"],
                    "rough_root_traj": (
                        motion["tags"]["rough_root_traj"]
                        - self.means["rough_root_traj"]
                    )
                    / self.stds["rough_root_traj"],
                    **(
                        {"style_id": motion["tags"]["style_id"]}
                        if "style_id" in motion["tags"]
                        else {}
                    ),
                    **{
                        k: (motion["tags"][k] - self.means[k]) / self.stds[k]
                        for k in ALL_LMA_KEYS
                        if k in motion["tags"] and k in self.means and k in self.stds
                    },
                },
                "rots": (motion["rots"] - self.means["rots"]) / self.stds["rots"],
                "style_label": motion.get("style_label") or "",
            }

            # print(len(motion["tags"]), motion["tags"]["BODY"].shape)

            self.norm_motions.append(norm_motion)

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, index):
        return (self.motions[index], self.norm_motions[index])
        # ========== [FIX] get_len / __len__ helpers for compatibility ==========

    def get_len(self):
        """
        Return number of motions / windows stored in this dataset.
        This is defensive: try common internal attributes used across the codebase.
        """
        for attr in (
            "motions",
            "motions_list",
            "data",
            "items",
            "motions_data",
            "_motions",
        ):
            if hasattr(self, attr):
                try:
                    return len(getattr(self, attr))
                except Exception:
                    continue
        if hasattr(self, "num_motions"):
            try:
                return int(self.num_motions)
            except Exception:
                pass
        # Fallback to __len__ if implemented, otherwise 0
        try:
            return int(self.__len__())
        except Exception:
            return 0

    def get_item(self, index):
        """
        Return the raw (denormalized) motion dict at index for debugging/inspection.
        Matches what train code expects when iterating samples before DataLoader collate.
        """
        return self.motions[index]

    def __len__(self):
        # Prefer an explicit backing container where possible
        return self.get_len()

    # ========== [END FIX] ==========


class TestMotionData:
    def __init__(self, param, scale, device):
        self.norm_motions = []
        self.bvhs = []
        self.filenames = []
        self.param = param
        self.scale = scale
        self.device = device
        self.downsample_factor = DOWNSAMPLE_FACTOR

    def set_means_stds(self, means, stds):
        self.means = means
        self.stds = stds

    def add_motion(
        self,
        offsets,
        global_pos,
        rotations,
        parents,
        bvh,
        filename,
        pos=None,
        og_rots=None,
        end_sites=None,
        end_sites_parents=None,
        lma_features=None,
        style_id=None,
        style_label=None,
    ):
        """
        Parameters:
        -----------
        offsets: np.array of shape (n_joints, 3)
        global_pos: np.array of shape (n_frames, 3)
        rotations: np.array of shape (n_frames, n_joints, 4) (quaternions)
        parents: np.array of shape (n_joints)

        Returns:
        --------
        self.norm_motions:
            offsets: tensor of shape (n_joints, 3)
            dqs: tensor of shape (windows_size, n_joints * 8) (dual quaternions)
            displacement: tensor of shape (windows_size, 3)
        """
        frames = rotations.shape[0]
        pad_size = 0
        PAD_TO = 64  # self.param["window_size"]
        if frames % PAD_TO != 0:
            pad_size = PAD_TO - frames % PAD_TO
            last_global_pos = global_pos[-1]
            last_rotation = rotations[-1]
            last_pos = pos[-1]
            global_pos = np.concatenate(
                [global_pos] + [last_global_pos[np.newaxis, :]] * pad_size, axis=0
            )
            rotations = np.concatenate(
                [rotations] + [last_rotation[np.newaxis, :, :]] * pad_size, axis=0
            )
            if og_rots is not None:
                og_rots = np.concatenate(
                    [og_rots] + [og_rots[-1][np.newaxis, :, :]] * pad_size, axis=0
                )
            pos = np.concatenate(
                [pos] + [last_pos[np.newaxis, :, :]] * pad_size, axis=0
            )
            frames += pad_size

        # because human has tpose at initial frame
        rotations[0] = rotations[1]
        global_pos[0] = global_pos[1]
        pos[0] = pos[1]

        # assert frames >= self.param["window_size"]
        # create dual quaternions
        fake_global_pos = np.zeros((frames, 3))
        dqs = to_root_dual_quat(rotations, fake_global_pos, parents, offsets)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = ortho6d.from_dual_quat(dqs)
        dqs = torch.flatten(dqs, 1, 2)

        offsets = torch.from_numpy(offsets).type(torch.float32).to(self.device)
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        self.global_pos = global_pos
        self.pos_all_joints = pos
        displacement = _root_velocity_height_features(global_pos)

        if og_rots is not None:
            rotations = og_rots

        dqs[:, 6:9] = displacement
        rots = torch.from_numpy(rotations).type(torch.float32).to(self.device)
        self.rots = ortho6d.from_quat(rots[:, 0]).reshape(-1, 6)
        self.rots = torch.cat([self.rots, displacement], dim=-1)
        root_traj_avg_window = int(self.param.get("rough_root_avg_window", 3))
        rough_root_traj = _compute_rough_root_trajectory(
            self.rots.detach().cpu().numpy(),
            stride=self.downsample_factor,
            avg_window=root_traj_avg_window,
        )
        rough_root_traj = (
            torch.from_numpy(rough_root_traj).type(torch.float32).to(self.device)
        )
        style_id_tensor = None
        if style_id is not None:
            style_id_tensor = torch.tensor(
                int(style_id), dtype=torch.long, device=self.device
            )

        # self.plot_dqs_first9(dqs)

        rots = rots.flatten(start_dim=1, end_dim=2).permute(1, 0).unsqueeze(0)
        # change global_pos to (1, 3, frames)
        rot_order = np.tile(
            ["y", "x", "z"], (rotations.shape[0], rotations.shape[1], 1)
        )
        rots_euler = quat.to_euler(rotations, rot_order)
        tags = compute_tags(
            pos,
            downsample_factor=self.downsample_factor,
            rots=rots_euler[:, 0],
            is_deg=False,
            quats=rotations,
            padded_frames=pad_size,
            skeleton_height=self.param["skeleton_height"],
            head_height=self.param["head_height"],
            head_idx=self.param["head_idx"],
            feet_idx=self.param["feet_idxs"],
            not_dog=self.param["not_dog"],
            feet_contact_threshold=self.param["feet_contact_threshold"],
            window_size=self.param["window_size"],
            shoulder_idx=self.param.get("shoulder_idxs", [5, 9]),
            lma_features=lma_features,
            parents=parents,
            visualize=self.param.get("visualize_on_load", False),
            sparse_joints=self.param.get("sparse_joints"),
        )

        for key in tags:
            if not isinstance(tags[key], torch.Tensor):
                tags[key] = torch.tensor(tags[key]).float().to(self.device)

        contact_joint = _build_contact_joint_features(
            tags["foot_contact_binary"],
            device=self.device,
        )
        dqs = torch.cat([dqs, contact_joint], dim=1)

        motion = {
            "offsets": offsets,
            "denorm_offsets": offsets.clone(),
            "dqs": dqs,
            "displacement": displacement,
            "foot_positions": pos[
                :,
                list(
                    dict.fromkeys(
                        int(index) for index in self.param.get("feet_idxs", [])
                    )
                ),
                :,
            ],
            "end_sites": end_sites,
            "end_sites_parents": np.array(end_sites_parents),
            "tags": {
                "velo_foot": tags["velo_foot"],
                "yaw_sin": tags["yaw_sin"],
                "yaw_cos": tags["yaw_cos"],
                "ctrl_forward_alignment": tags["ctrl_forward_alignment"],
                "ctrl_lateral_alignment": tags["ctrl_lateral_alignment"],
                "ctrl_velocity": tags["ctrl_velocity"],
                "ctrl_acceleration": tags["ctrl_acceleration"],
                "ctrl_yaw_rate": tags["ctrl_yaw_rate"],
                "ctrl_yaw_accel": tags["ctrl_yaw_accel"],
                "ctrl_height": tags["ctrl_height"],
                "ctrl_head_height": tags["ctrl_head_height"],
                "ctrl_roll": tags["ctrl_roll"],
                "ctrl_foot_labels": tags["ctrl_foot_labels"],
                "ctrl_vertical_velocity": tags["ctrl_vertical_velocity"],
                "smooth_root_pos": tags["smooth_root_pos"],
                "foot_contact_probs": tags["foot_contact_probs"],
                "foot_contact_binary": tags["foot_contact_binary"],
                "support_contact": tags["support_contact"],
                "foot_contact_latent": tags["foot_contact_latent"],
                "support_contact_latent": tags["support_contact_latent"],
                "rough_root_traj": rough_root_traj,
                **(
                    {"style_id": style_id_tensor.clone()}
                    if style_id_tensor is not None
                    else {}
                ),
                **{k: tags[k] for k in ALL_LMA_KEYS if k in tags},
            },
            "global_pos": global_pos,
            "rots": self.rots,
            "style_label": style_label if style_label is not None else "",
        }

        self.norm_motions.append(motion)
        self.bvhs.append(bvh)
        self.filenames.append(filename)

    def normalize(self):
        # Normalize
        assert self.means is not None
        assert self.stds is not None

        for motion in self.norm_motions:
            motion["offsets"] = (motion["offsets"] - self.means["offsets"]) / self.stds[
                "offsets"
            ]

            motion["dqs"] = (motion["dqs"] - self.means["dqs"]) / self.stds["dqs"]
            motion["displacement"] = (
                motion["displacement"] - self.means["displacement"]
            ) / self.stds["displacement"]
            motion["rots"] = (motion["rots"] - self.means["rots"]) / self.stds["rots"]
            if "tags" in motion:
                tags = motion["tags"]
                tags["yaw_sin"] = (tags["yaw_sin"] - self.means["yaw_sin"]) / self.stds[
                    "yaw_sin"
                ]
                tags["yaw_cos"] = (tags["yaw_cos"] - self.means["yaw_cos"]) / self.stds[
                    "yaw_cos"
                ]
                tags["ctrl_forward_alignment"] = (
                    tags["ctrl_forward_alignment"]
                    - self.means["ctrl_forward_alignment"]
                ) / self.stds["ctrl_forward_alignment"]
                tags["ctrl_lateral_alignment"] = (
                    tags["ctrl_lateral_alignment"]
                    - self.means["ctrl_lateral_alignment"]
                ) / self.stds["ctrl_lateral_alignment"]
                tags["ctrl_velocity"] = (
                    tags["ctrl_velocity"] - self.means["ctrl_velocity"]
                ) / self.stds["ctrl_velocity"]
                tags["ctrl_acceleration"] = (
                    tags["ctrl_acceleration"] - self.means["ctrl_acceleration"]
                ) / self.stds["ctrl_acceleration"]
                tags["ctrl_yaw_rate"] = (
                    tags["ctrl_yaw_rate"] - self.means["ctrl_yaw_rate"]
                ) / self.stds["ctrl_yaw_rate"]
                tags["ctrl_yaw_accel"] = (
                    tags["ctrl_yaw_accel"] - self.means["ctrl_yaw_accel"]
                ) / self.stds["ctrl_yaw_accel"]
                tags["ctrl_height"] = (
                    tags["ctrl_height"] - self.means["ctrl_height"]
                ) / self.stds["ctrl_height"]
                tags["ctrl_head_height"] = (
                    tags["ctrl_head_height"] - self.means["ctrl_head_height"]
                ) / self.stds["ctrl_head_height"]
                tags["ctrl_foot_labels"] = tags["ctrl_foot_labels"]
                tags["ctrl_roll"] = (
                    tags["ctrl_roll"] - self.means["ctrl_roll"]
                ) / self.stds["ctrl_roll"]
                tags["ctrl_vertical_velocity"] = (
                    tags["ctrl_vertical_velocity"]
                    - self.means["ctrl_vertical_velocity"]
                ) / self.stds["ctrl_vertical_velocity"]
                tags["smooth_root_pos"] = (
                    tags["smooth_root_pos"] - self.means["smooth_root_pos"]
                ) / self.stds["smooth_root_pos"]
                tags["foot_contact_probs"] = tags["foot_contact_probs"]
                tags["foot_contact_binary"] = tags["foot_contact_binary"]
                tags["support_contact"] = tags["support_contact"]
                tags["foot_contact_latent"] = tags["foot_contact_latent"]
                tags["support_contact_latent"] = tags["support_contact_latent"]
                rough_root_mean = self.means.get("rough_root_traj", self.means["rots"])
                rough_root_std = self.stds.get("rough_root_traj", self.stds["rots"])
                tags["rough_root_traj"] = (
                    tags["rough_root_traj"] - rough_root_mean
                ) / rough_root_std
                if "style_id" in tags:
                    tags["style_id"] = tags["style_id"].to(
                        self.device, dtype=torch.long
                    )
                for _lma_k in ALL_LMA_KEYS:
                    if _lma_k in tags and _lma_k in self.means and _lma_k in self.stds:
                        tags[_lma_k] = (tags[_lma_k] - self.means[_lma_k]) / self.stds[
                            _lma_k
                        ]

    def set_tags(self, tags):
        self.norm_motions[0]["tags"] = tags

    def get_bvh(self, index):
        return self.bvhs[index], self.filenames[index]

    def get_len(self):
        return len(self.norm_motions)

    def get_item(self, index):
        return self.norm_motions[index]

    def plot_dqs_first9(self, index=0, show=True, save_path=None):
        """Plot the first 9 channels of the stored `dqs` for a test motion.

        - First 6 channels (ortho6d rotation components) are plotted together.
        - Next 3 channels (velocity-height) are plotted together.

        Args:
            index (int): index of motion in `self.norm_motions`.
            show (bool): whether to call `plt.show()`.
            save_path (str|None): if provided, save the figure to this path.
        """
        motion = self.get_item(index)
        if "dqs" not in motion:
            raise ValueError("motion at index does not contain 'dqs'")

        dqs = motion["dqs"]
        # ensure cpu numpy array and shape [T, C]
        if isinstance(dqs, torch.Tensor):
            data = dqs.detach().cpu().numpy()
        else:
            data = np.array(dqs)

        if data.ndim == 1:
            data = data[:, np.newaxis]

        C = data.shape[1]
        if C < 9:
            raise ValueError(f"dqs has {C} channels, need at least 9 to plot first 9")

        t = np.arange(data.shape[0])

        # Plot first 6 channels (ortho6d)
        plt.figure(figsize=(10, 3))
        for c in range(6):
            plt.plot(t, data[:, c], label=f"chan_{c}")
        plt.title(f"dqs first 6 channels (index={index})")
        plt.xlabel("frame")
        plt.ylabel("value")
        plt.legend(loc="upper right", ncol=3)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path + "_dqs_0-5.png")
        if show:
            plt.show()

        # Plot next 3 channels (velocity-height)
        plt.figure(figsize=(8, 3))
        for c in range(6, 9):
            plt.plot(t, data[:, c], label=f"chan_{c}")
        plt.title(f"dqs channels 6-8 (index={index})")
        plt.xlabel("frame")
        plt.ylabel("value")
        plt.legend(loc="upper right")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path + "_dqs_6-8.png")
        if show:
            plt.show()

    def get_pos(self):
        return self.pos_all_joints


class RunMotionData:
    def __init__(self, param, device):
        self.param = param
        self.device = device

    def set_means_stds(self, means, stds):
        self.means = means
        self.stds = stds

    def set_offsets(self, offsets):
        """
        Parameters:
        -----------
        offsets: np.array of shape (n_joints, 3)
        """
        offsets = torch.from_numpy(offsets).type(torch.float32).to(self.device)
        self.motion = {
            "offsets": offsets,
            "denorm_offsets": offsets.clone(),
        }

    def set_motion_from_bvh(self, offsets, global_pos, rotations, parents):
        frames = rotations.shape[0]
        assert frames >= self.param["window_size"]
        # create dual quaternions
        fake_global_pos = np.zeros((frames, 3))
        dqs = to_root_dual_quat(rotations, fake_global_pos, np.array(parents), offsets)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = torch.flatten(dqs, 1, 2)
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        displacement = _root_velocity_height_features(global_pos)
        dqs[:, 6:9] = displacement
        contact_joint = torch.zeros(
            (dqs.shape[0], SYNTHETIC_CONTACT_CHANNELS + SYNTHETIC_CONTACT_PAD_CHANNELS),
            dtype=dqs.dtype,
            device=self.device,
        )

        self.motion["dqs"] = torch.cat([dqs, contact_joint], dim=1)
        self.motion["displacement"] = displacement

    def set_motion(self, positions, rotations):
        frames = rotations.shape[0]
        assert frames >= self.param["window_size"]
        global_pos = positions[:, 0, :]
        # create dual quaternions
        dqs = dquat.from_rotation_translation(rotations, positions)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = torch.flatten(dqs, 1, 2)
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        displacement = _root_velocity_height_features(global_pos)
        dqs[:, 6:9] = displacement
        contact_joint = torch.zeros(
            (dqs.shape[0], SYNTHETIC_CONTACT_CHANNELS + SYNTHETIC_CONTACT_PAD_CHANNELS),
            dtype=dqs.dtype,
            device=self.device,
        )

        self.motion["dqs"] = torch.cat([dqs, contact_joint], dim=1)
        self.motion["displacement"] = displacement

    def normalize_offsets(self):
        # Normalize
        assert self.means is not None
        assert self.stds is not None
        self.motion["offsets"] = (
            self.motion["offsets"] - self.means["offsets"]
        ) / self.stds["offsets"]

    def normalize_motion(self):
        # Normalize
        assert self.means is not None
        assert self.stds is not None
        self.motion["dqs"] = (self.motion["dqs"] - self.means["dqs"]) / self.stds["dqs"]
        self.motion["displacement"] = (
            self.motion["displacement"] - self.means["displacement"]
        ) / self.stds["displacement"]

    def get_item(self):
        return self.motion


def delay_embedding(data, d, tau):
    N = len(data)
    indices = np.arange(d) * tau + np.arange(N - (d - 1) * tau)[:, None]
    embedded_data = data[indices]
    return embedded_data


def calc_velocity_from_pos(pos, downsample_factor, num_groups):
    x_values = pos[:, 0, 0].copy()  # ROOT X-axis movement data
    z_values = pos[:, 0, 2].copy()  # ROOT Z-axis movement data
    step_size = WINDOW_STEP_SIZE  # Calculate every 8 frames
    window_size = downsample_factor  # Look ahead this many frames

    # Calculate number of windows with step size 8
    num_windows = num_groups

    vx_per_window = np.zeros(num_windows)
    vz_per_window = np.zeros(num_windows)

    # Calculate velocities using sliding window
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, pos.shape[0] - 1)

        # Calculate displacement over window
        vx_per_window[i] = x_values[end_idx] - x_values[start_idx]
        vz_per_window[i] = z_values[end_idx] - z_values[start_idx]

    velocity_per_window = np.sqrt(vx_per_window**2 + vz_per_window**2 + 1e-8) * 2
    return velocity_per_window


def calc_velocity_from_pos_torch(pos, downsample_factor, num_groups):
    x_values = pos[..., 0, 0]  # ROOT X-axis movement data
    z_values = pos[..., 0, 2]  # ROOT Z-axis movement data
    step_size = WINDOW_STEP_SIZE  # Calculate every 8 frames
    window_size = downsample_factor  # Look ahead this many frames

    # Calculate number of windows with step size 8
    num_windows = num_groups

    # Calculate velocities using sliding window
    indices = torch.arange(num_windows) * step_size

    # Get start and end indices for each window
    start_indices = indices
    end_indices = torch.min(
        end_indices, torch.tensor(pos.shape[0] - 1, device=pos.device)
    )

    # Calculate displacements
    vx_per_window = x_values[..., end_indices] - x_values[..., start_indices]
    vz_per_window = z_values[..., end_indices] - z_values[..., start_indices]

    velocity_per_window = torch.sqrt(vx_per_window**2 + vz_per_window**2 + 1e-8) * 2
    return velocity_per_window


def construct_signal(peaks, valleys, num_points, inactive):
    """
    Construct a signal from given peaks and valleys.

    Args:
        peaks (list): Indices of peak positions.
        valleys (list): Indices of valley positions.
        num_points (int): Total number of points in the signal.

    Returns:
        signal (np.array): Constructed signal.
    """
    # Initialize the signal array
    signal = np.zeros(num_points)
    # Ensure peaks and valleys are sorted
    points = np.concatenate((peaks, valleys))
    points = sorted(points)
    # Iterate through the points to construct the signal
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        x = np.linspace(0, 1, end - start)  # Normalize to 0 to 1
        if points[i] in inactive:
            signal[start:end] = 0
        elif points[i] in peaks:
            # If current point is a peak
            signal[start:end] = 0.5 * (
                1 + np.cos(np.pi * x)
            )  # Descend to valley (sine wave)
        elif points[i] in valleys:
            # If current point is a valley
            signal[start:end] = 0.5 * (
                1 - np.cos(np.pi * x)
            )  # Ascend to peak (sine wave)

    return signal


def compute_foot_labels(displacements, idxes=[8, 12, 15, 18], threshold=0.05):
    """
    Compute binary foot labels based on foot positions and velocities.
    If the foot is moving, label it as 1, otherwise 0.
    """

    num_frames = displacements.shape[0]
    num_joints = len(idxes)
    labels = np.zeros((num_frames, num_joints))
    disp_magnitudes = np.zeros((num_frames, num_joints))

    for i, j in enumerate(idxes):
        # Compute global position for the foot joint: add root's position
        diff = np.diff(displacements[:, j, :], axis=0)
        # Use only the x and z components to compute the displacement magnitude
        # joint_disp_xz = global_foot[:, [0, 2]]
        disp_magnitudes[1:, i] = np.linalg.norm(diff, axis=1)
        # Threshold may be adjusted based on your data scale
        labels[1:, i] = disp_magnitudes[1:, i] > threshold

    return labels, disp_magnitudes


def compute_foot_labels2(
    displacements,
    idxes=[8, 12, 15, 18],
    threshold=0.05,
    temp=0.02,
    smooth_sigma=1,
    adapt_mode="none",
    adapt_k=1.5,
):
    """
    Compute per-foot contact probabilities from joint positions.

    Args:
        displacements: np.ndarray shape (F, J, 3) (joint world positions per frame)
        idxes: list of joint indices to consider as feet
        threshold: legacy fixed threshold (used only if adapt_mode == 'fixed')
        temp: temperature for sigmoid mapping (smaller -> sharper probability)
        smooth_sigma: gaussian smoothing sigma applied to per-joint magnitudes
        adapt_mode: 'mad' | 'percentile' | 'fixed' - how to compute adaptive threshold
        adapt_k: multiplier (for MAD) or percentile (0..1) when using percentile mode

    Returns:
        probs: np.ndarray shape (F, n_feet) of contact probabilities in [0,1]
        disp_magnitudes: np.ndarray shape (F, n_feet) of displacement magnitudes (smoothed)
    """
    num_frames = displacements.shape[0]
    num_joints = len(idxes)
    disp_magnitudes = np.zeros((num_frames, num_joints), dtype=np.float32)

    # compute per-frame displacement magnitudes (use x,z components)
    for i, j in enumerate(idxes):
        # differences between consecutive frames for joint j
        disp_between = displacements[1:, j, :] - displacements[:-1, j, :]
        mags = np.linalg.norm(disp_between, axis=1)
        disp_magnitudes[1:, i] = mags

    # smooth magnitudes to reduce noise
    if smooth_sigma is not None and smooth_sigma > 0:
        for i in range(num_joints):
            disp_magnitudes[:, i] = gaussian_filter1d(
                disp_magnitudes[:, i], sigma=smooth_sigma
            )

    # compute adaptive threshold per-joint
    if adapt_mode == "mad":
        med = np.median(disp_magnitudes, axis=0)
        mad = np.median(np.abs(disp_magnitudes - med[None, :]), axis=0)
        thr = med + adapt_k * mad
    elif adapt_mode == "percentile":
        p = float(adapt_k) if adapt_k <= 1.0 else adapt_k / 100.0
        thr = np.percentile(disp_magnitudes, p * 100.0, axis=0)
    else:  # fixed threshold
        thr = np.full((num_joints,), float(threshold), dtype=disp_magnitudes.dtype)

    # prevent tiny thresholds
    thr = np.maximum(thr, 1e-6)

    # map magnitude -> contact probability: contact when mag < thr
    # logits = (thr - mag) / temp  -> p = sigmoid(logits)
    denom = float(temp) if temp > 0.0 else 1e-6
    logits = (thr[None, :] - disp_magnitudes) / denom

    # probs = 1.0 / (1.0 + np.exp(-logits))
    probs = expit(logits)  # more stable numerically

    # optional: remove isolated spikes (if a one-frame positive surrounded by negatives)
    # convert to binary mask for isolation detection then restore probabilities for non-isolated frames
    binary = (probs > 0.5).astype(np.float32)
    for i in range(num_joints):
        for t in range(1, num_frames - 1):
            if binary[t, i] == 1 and binary[t - 1, i] == 0 and binary[t + 1, i] == 0:
                probs[t, i] = 0.0

    return probs, disp_magnitudes


def _fill_short_contact_gaps(binary_mask, max_gap=2):
    if max_gap <= 0:
        return binary_mask

    binary_mask = np.asarray(binary_mask, dtype=np.float32).copy()
    num_frames, num_feet = binary_mask.shape
    for foot_index in range(num_feet):
        active = np.flatnonzero(binary_mask[:, foot_index] > 0.5)
        if active.size < 2:
            continue
        for left, right in zip(active[:-1], active[1:]):
            gap = int(right - left - 1)
            if 0 < gap <= max_gap:
                binary_mask[left + 1 : right, foot_index] = 1.0
    return binary_mask.reshape(num_frames, num_feet)


def _remove_short_contact_runs(binary_mask, min_run_length=2):
    if min_run_length <= 1:
        return binary_mask

    binary_mask = np.asarray(binary_mask, dtype=np.float32).copy()
    num_frames, num_feet = binary_mask.shape
    for foot_index in range(num_feet):
        start = None
        for frame_index in range(num_frames + 1):
            is_active = (
                frame_index < num_frames and binary_mask[frame_index, foot_index] > 0.5
            )
            if is_active and start is None:
                start = frame_index
            elif not is_active and start is not None:
                if frame_index - start < min_run_length:
                    binary_mask[start:frame_index, foot_index] = 0.0
                start = None
    return binary_mask.reshape(num_frames, num_feet)


def _downsample_temporal_features(sequence, stride=DOWNSAMPLE_FACTOR, avg_window=3):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim == 1:
        sequence = sequence[:, None]
    if sequence.ndim != 2:
        raise ValueError(f"Expected [T, C] temporal features, got {sequence.shape}")
    if sequence.shape[0] == 0:
        return np.zeros((0, sequence.shape[1]), dtype=np.float32)

    centers = np.arange(0, sequence.shape[0], max(int(stride), 1), dtype=np.int64)
    return _moving_average_windows(sequence, centers, avg_window)


def compute_contact_signals(
    positions,
    idxes=[8, 12, 15, 18],
    frame_rate=30.0,
    smooth_sigma=1.25,
    ground_percentile=2.0,
    height_percentile=35.0,
    speed_percentile=40.0,
    height_margin=0.02,
    min_speed_threshold=0.03,
    height_temp=0.015,
    speed_temp=0.08,
    on_threshold=0.6,
    off_threshold=0.4,
    max_gap=2,
    min_contact_frames=2,
):
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3:
        raise ValueError(f"Expected [T, J, 3] positions, got {positions.shape}")

    foot_positions = positions[:, idxes, :]
    num_frames, num_feet, _ = foot_positions.shape
    if num_frames == 0:
        zeros = np.zeros((0, num_feet), dtype=np.float32)
        return {
            "contact_probs": zeros,
            "contact_binary": zeros,
            "support_contact": np.zeros((0, 1), dtype=np.float32),
            "foot_speed": zeros,
            "foot_height": zeros,
        }

    dt = 1.0 / max(float(frame_rate), 1e-6)
    foot_height = foot_positions[:, :, 1]
    foot_speed = (
        np.linalg.norm(
            np.diff(
                foot_positions[:, :, [0, 2]],
                axis=0,
                prepend=foot_positions[0:1, :, [0, 2]],
            ),
            axis=-1,
        )
        / dt
    )

    if smooth_sigma is not None and smooth_sigma > 0:
        foot_height = gaussian_filter1d(foot_height, sigma=smooth_sigma, axis=0)
        foot_speed = gaussian_filter1d(foot_speed, sigma=smooth_sigma, axis=0)

    ground_height = np.percentile(foot_height.reshape(-1), ground_percentile)
    relative_height = np.maximum(foot_height - ground_height, 0.0)

    contact_probs = np.zeros((num_frames, num_feet), dtype=np.float32)
    contact_binary = np.zeros((num_frames, num_feet), dtype=np.float32)
    for foot_index in range(num_feet):
        height_threshold = np.percentile(
            relative_height[:, foot_index], height_percentile
        ) + float(height_margin)
        speed_threshold = max(
            np.percentile(foot_speed[:, foot_index], speed_percentile),
            float(min_speed_threshold),
        )

        height_score = expit(
            (height_threshold - relative_height[:, foot_index])
            / max(float(height_temp), 1e-6)
        )
        speed_score = expit(
            (speed_threshold - foot_speed[:, foot_index]) / max(float(speed_temp), 1e-6)
        )
        foot_prob = np.sqrt(height_score * speed_score).astype(np.float32)
        contact_probs[:, foot_index] = foot_prob

        is_contact = False
        for frame_index, probability in enumerate(foot_prob):
            if is_contact:
                is_contact = probability >= float(off_threshold)
            else:
                is_contact = probability >= float(on_threshold)
            contact_binary[frame_index, foot_index] = 1.0 if is_contact else 0.0

    contact_binary = _fill_short_contact_gaps(contact_binary, max_gap=max_gap)
    contact_binary = _remove_short_contact_runs(
        contact_binary, min_run_length=min_contact_frames
    )

    contact_probs = np.where(
        contact_binary > 0.5,
        np.maximum(contact_probs, 0.5 + 0.5 * contact_probs),
        np.minimum(contact_probs, 0.5 * contact_probs),
    ).astype(np.float32)
    support_contact = np.max(contact_probs, axis=1, keepdims=True).astype(np.float32)

    return {
        "contact_probs": contact_probs,
        "contact_binary": contact_binary.astype(np.float32),
        "support_contact": support_contact,
        "foot_speed": foot_speed.astype(np.float32),
        "foot_height": relative_height.astype(np.float32),
    }


def compute_tags(
    pos,
    head_idx=2,
    head_height=1.0,
    downsample_factor=8,
    is_human=False,
    skeleton_height=0.90,
    rots=None,
    is_deg=True,
    scalar=1.0,
    feet_idx=[8, 12, 15, 18],
    quats=None,
    padded_frames=0,
    shoulder_idx=[5, 9],
    not_dog=True,
    feet_contact_threshold=0.02,
    window_size=None,
    lma_features=None,
    parents=None,
    visualize=False,
    training=False,
    sparse_joints=None,
):

    num_frames = pos.shape[0]
    num_groups = (
        num_frames // WINDOW_STEP_SIZE
    )  # each latent vector (encoder embedding) corresponds to 8 frames
    time_step = 1 / 30.0

    if is_human:
        pos[0:5] = np.copy(pos[5:10])

    root_pos = np.copy(pos[:, 0, :])

    #     # actually using spine for dog
    lying_down_threshold = 0.35
    head_height = pos[:, head_idx, 1].copy()  # - pos[:,0,1] #- head_height
    # head_height = np.where(head_height > lying_down_threshold, 1.0, 0.0)

    root_yaw = np.unwrap(rots[:, 0])  # Unwrap to avoid jumps

    #####################################################################################
    #####################################################################################

    pos_l_shoulder = pos[:, shoulder_idx[0]]
    pos_r_shoulder = pos[:, shoulder_idx[1]]

    # Extract X and Z coordinates for both shoulders
    l_shoulder_x = pos_l_shoulder[:, 0]
    l_shoulder_z = pos_l_shoulder[:, 2]
    r_shoulder_x = pos_r_shoulder[:, 0]
    r_shoulder_z = pos_r_shoulder[:, 2]

    # Compute line segment vectors in XZ plane
    dx = r_shoulder_x - l_shoulder_x
    dz = r_shoulder_z - l_shoulder_z

    # Compute perpendicular vectors (rotate by 90 degrees in XZ plane)
    perp_dx = -dz
    perp_dz = dx

    #####
    root_x = pos[:, 0, 0]
    root_z = pos[:, 0, 2]
    # Compute midpoint of shoulders
    mid_shoulder_x = (l_shoulder_x + r_shoulder_x) / 2
    mid_shoulder_z = (l_shoulder_z + r_shoulder_z) / 2

    # Compute vector from root to midpoint of shoulders
    vec_x = mid_shoulder_x - root_x
    vec_z = mid_shoulder_z - root_z

    # Optionally normalize the vector
    norm = np.sqrt(vec_x**2 + vec_z**2)
    vec_x_norm = vec_x / (norm + 1e-8)
    vec_z_norm = vec_z / (norm + 1e-8)
    #####

    # Normalize perpendicular vectors for consistent arrow length
    norm = np.sqrt(perp_dx**2 + perp_dz**2)

    # perp_angles = np.arctan2(perp_dx_norm,perp_dz_norm )  # Angle in radians
    # perp_angles = np.arctan2(perp_dz_norm, perp_dx_norm)  # Angle in radians
    perp_angles = np.arctan2(vec_x_norm, vec_z_norm)  # Angle in radians

    # Optionally unwrap to avoid discontinuities
    perp_angles_unwrapped = np.unwrap(perp_angles)
    sin_perp_angles = np.sin(perp_angles_unwrapped)
    cos_perp_angles = np.cos(perp_angles_unwrapped)

    zigma = 5

    # sigma = 20
    sigma = zigma
    sin_perp_angles = gaussian_filter1d(sin_perp_angles, sigma=sigma)
    cos_perp_angles = gaussian_filter1d(cos_perp_angles, sigma=sigma)

    shoulders_yaw_quaternions = quat.from_angle_axis(
        np.expand_dims(perp_angles_unwrapped, axis=-1),
        axis=np.tile(np.array([0, 1, 0]), (len(perp_angles_unwrapped), 1)),
    )  # Rotation around the vertical axis (Y-axis)
    shoulders_yaw_quaternions = quat.normalize(
        quat.unroll(shoulders_yaw_quaternions, axis=0)
    )
    # shoulders_yaw_quaternions = quat.normalize(shoulders_yaw_quaternions)
    shoulders_yaw_quaternions = gaussian_filter1d(
        shoulders_yaw_quaternions, sigma=30, axis=0
    )

    extra = 0  # 5
    extra_sincos = 0
    if not_dog:
        perp_angles_unwrapped = root_yaw  #
    # perp_angles_unwrapped = perp_angles_unwrapped + np.deg2rad(90)
    # perp_angles_unwrapped+=np.pi/2
    # perp_angles_unwrapped = gaussian_filter1d(perp_angles_unwrapped, sigma=25)

    # sigma = 25
    sigma = zigma
    sin_perp_angles = np.sin(
        gaussian_filter1d(perp_angles_unwrapped, sigma=sigma + extra_sincos)
    )
    cos_perp_angles = np.cos(
        gaussian_filter1d(perp_angles_unwrapped, sigma=sigma + extra_sincos)
    )

    contact_signals = compute_contact_signals(
        pos,
        feet_idx,
        frame_rate=1.0 / max(time_step, 1e-6),
        min_speed_threshold=max(float(feet_contact_threshold), 1e-4),
    )
    foot_labels = contact_signals["contact_probs"]
    support_contact = contact_signals["support_contact"]
    legacy_binary_foot_labels, disp_magnitudes_old = compute_foot_labels(
        pos, feet_idx, threshold=feet_contact_threshold
    )
    foot_contact_latent = _downsample_temporal_features(
        foot_labels,
        stride=downsample_factor,
        avg_window=3,
    )
    support_contact_latent = _downsample_temporal_features(
        support_contact,
        stride=downsample_factor,
        avg_window=3,
    )

    root_displacements = np.diff(pos[:, 0, [0, 2]], axis=0, prepend=pos[0:1, 0, [0, 2]])

    # sigma = 10
    sigma = zigma
    root_displacements[:, 0] = gaussian_filter1d(root_displacements[:, 0], sigma=sigma)
    root_displacements[:, 1] = gaussian_filter1d(root_displacements[:, 1], sigma=sigma)
    root_displacements = root_displacements * scalar

    forward_direction = np.stack([sin_perp_angles, cos_perp_angles], axis=-1)
    forward_direction_norm = np.linalg.norm(forward_direction, axis=1, keepdims=True)
    forward_direction = forward_direction / np.clip(forward_direction_norm, 1e-8, None)

    motion_speed = np.linalg.norm(root_displacements, axis=1, keepdims=True)
    motion_speed_scalar = motion_speed[:, 0]
    motion_eps = max(float(np.percentile(motion_speed_scalar, 25)) * 0.5, 1e-6)
    moving_mask = motion_speed_scalar > motion_eps
    right_direction = np.stack(
        [forward_direction[:, 1], -forward_direction[:, 0]], axis=-1
    )
    ctrl_forward_alignment = (
        np.sum(root_displacements * forward_direction, axis=1) / time_step
    )
    ctrl_lateral_alignment = (
        np.sum(root_displacements * right_direction, axis=1) / time_step
    )
    ctrl_forward_alignment[~moving_mask] = 0.0
    ctrl_lateral_alignment[~moving_mask] = 0.0

    ctrl_velocity = motion_speed_scalar / (1 / 30.0)  # * scalar

    ctrl_acceleration = np.diff(ctrl_velocity[1:], prepend=ctrl_velocity[0]) / time_step
    ctrl_acceleration = np.insert(ctrl_acceleration, 0, ctrl_acceleration[0])

    # sigma = 30
    sigma = zigma
    ctrl_yaw_rate = (
        np.diff(
            gaussian_filter1d(perp_angles_unwrapped, sigma=sigma + extra),
            prepend=perp_angles_unwrapped[0],
        )
        / time_step
    )
    ctrl_yaw_rate[0] = ctrl_yaw_rate[1]

    ctrl_yaw_accel = np.diff(ctrl_yaw_rate, prepend=ctrl_yaw_rate[0]) / time_step

    ctrl_height = root_pos[:, 1] / skeleton_height

    # sigma = 8
    sigma = zigma
    ctrl_height = gaussian_filter1d(ctrl_height, sigma=sigma)
    # ctrl_height[ctrl_height >= 1.1] = 1.6

    ctrl_vertical_velocity = np.diff(ctrl_height, prepend=ctrl_height[0]) / time_step
    ctrl_vertical_velocity[0] = ctrl_vertical_velocity[1]  # Fix first frame
    # ctrl_height[ctrl_height <= 0.5] = 0

    # ctrl_vertical_velocity[ctrl_height <= 1.1 ] = 0
    # ctrl_vertical_velocity[ctrl_vertical_velocity > 0.2 ] = 1
    # ctrl_vertical_velocity[ctrl_vertical_velocity < -0.2 ] = -1

    left_shoulder = pos[:, shoulder_idx[0], :]  # [frames, 3]
    right_shoulder = pos[:, shoulder_idx[1], :]  # [frames, 3]
    shoulder_vector = left_shoulder - right_shoulder  # [frames, 3]
    # Compute the roll angle (angle between shoulder vector and horizontal plane)
    roll_angles = np.arctan2(
        shoulder_vector[:, 1], np.linalg.norm(shoulder_vector[:, [0, 2]], axis=1)
    )  # [frames]

    # sigma = 30
    sigma = zigma
    roll_angles = gaussian_filter1d(roll_angles, sigma=sigma)
    ctrl_roll_angles = np.round(roll_angles)

    # sin_perp_angles = gaussian_filter1d(sin_perp_angles, sigma=10)
    # cos_perp_angles = gaussian_filter1d(cos_perp_angles, sigma=10)

    smooth_root_pos = gaussian_filter1d(root_pos, sigma=1, axis=0)
    smooth_root_pos[:, 0] = smooth_root_pos[:, 0] * scalar
    smooth_root_pos[:, 1] /= skeleton_height
    smooth_root_pos[:, 2] = smooth_root_pos[:, 2] * scalar

    # plt.figure(figsize=(10, 6))
    # plt.plot(ctrl_velocity)
    # plt.plot(ctrl_velocity_)
    # plt.plot(ctrl_velocity__)
    # plt.plot(ctrl_acceleration)
    # plt.plot(ctrl_height)
    # plt.plot(ctrl_vertical_velocity)
    # plt.plot(ctrl_yaw_rate)
    # plt.plot(gaussian_filter1d(root_yaw,sigma=30))
    # plt.plot(root_yaw)
    # plt.plot(ctrl_yaw_accel)
    # plt.plot(head_height)
    # plt.plot(pos[:,head_idx,1])
    # plt.plot(roll_angles)
    # plt.grid(True)
    # plt.plot(sin_perp_angles)
    # plt.plot(cos_perp_angles)
    # plt.plot(disp_magnitudes)
    # plt.plot(disp_magnitudes_old)
    # plt.plot(smooth_root_pos)
    # plt.plot(foot_labels[:,-2:])
    # plt.plot(pos[:,8])
    # plt.plot(ctrl_velocity)
    # plt.plot(root_pos)
    # plt.plot(perp_angles_unwrapped)
    #######################
    # ax=plt.gca()
    # ax.plot(ctrl_height, color="C0", linewidth=1.25)
    # X axis: frames
    # ax.set_xlim(0, num_frames - 1)
    # Y axis: data-driven with small padding to avoid clipping
    # ymin = 0
    # ymax = 3
    # pad = (ymax - ymin) * 0.1
    # if pad == 0:
    #     pad = 0.5
    # ax.set_ylim(ymin - pad, ymax + pad)
    #######################

    # plt.show()

    #####################################################################################
    #####################################################################################
    tags = {
        "velo_foot": np.expand_dims(disp_magnitudes_old, axis=1),
        "yaw_sin": sin_perp_angles,
        "yaw_cos": cos_perp_angles,
        "ctrl_forward_alignment": ctrl_forward_alignment,
        "ctrl_lateral_alignment": ctrl_lateral_alignment,
        "ctrl_velocity": ctrl_velocity,
        "ctrl_acceleration": ctrl_acceleration,
        "ctrl_yaw_rate": ctrl_yaw_rate,
        "ctrl_yaw_accel": ctrl_yaw_accel,
        "ctrl_height": ctrl_height,
        "ctrl_head_height": head_height,
        "ctrl_roll": ctrl_roll_angles,
        "ctrl_foot_labels": (
            foot_labels[:, -2:] if foot_labels.shape[1] > 2 else foot_labels
        ),
        "ctrl_vertical_velocity": ctrl_vertical_velocity,
        "smooth_root_pos": smooth_root_pos,
        "binary_foot_labels": contact_signals["contact_binary"],
        "legacy_binary_foot_labels": legacy_binary_foot_labels,
        "foot_contact_probs": foot_labels,
        "foot_contact_binary": contact_signals["contact_binary"],
        "foot_contact_joint_indices": np.asarray(
            feet_idx[: contact_signals["contact_binary"].shape[1]],
            dtype=np.int64,
        ),
        "foot_contact_latent": foot_contact_latent,
        "support_contact": support_contact,
        "support_contact_latent": support_contact_latent,
    }

    lma_target_len = max((num_frames + LMA_FRAME_STRIDE - 1) // LMA_FRAME_STRIDE, 1)
    existing_lma = _coerce_lma_dict(lma_features, target_len=lma_target_len)
    if existing_lma is not None:
        for key in LMA_KEYS:
            if key in existing_lma:
                tags[key] = existing_lma[key].astype(np.float32)

    if visualize:
        try:
            from visualizer import visualize_motion_and_tags

            visualize_motion_and_tags(pos, tags, parents=parents)
        except Exception as _viz_err:
            print(f"[visualizer] Could not launch visualizer: {_viz_err}")

    return tags


def plot_lma_and_ctrl(
    tags, ctrl_keys=None, figsize=(10, None), title=None, savepath=None
):
    """
    Plot LMA channels and control signals each in their own subplot on a single page.

    Args:
        tags (dict): dictionary containing tag arrays (numpy or torch tensors).
        ctrl_keys (list or None): list of control keys to plot (default: ["ctrl_velocity","ctrl_forward_alignment","ctrl_lateral_alignment","ctrl_height"]).
        figsize (tuple): optional figure size; height is auto-scaled if None.
        title (str): optional suptitle for the figure.
        savepath (str): if provided, save the figure to this path.
    """
    import torch as _torch

    lma_keys = list(LMA_KEYS)
    present_lma = [k for k in lma_keys if k in tags]
    if ctrl_keys is None:
        ctrl_keys = [
            "ctrl_velocity",
            "ctrl_forward_alignment",
            "ctrl_lateral_alignment",
            "ctrl_height",
        ]
    present_ctrl = [k for k in ctrl_keys if k in tags]

    nrows = len(present_lma) + len(present_ctrl)
    if nrows == 0:
        print("No LMA or control signals found in tags to plot.")
        return

    height = 2 * nrows
    if figsize[1] is None:
        fig = plt.figure(figsize=(figsize[0], height))
        axes = fig.subplots(nrows, 1, sharex=True)
    else:
        fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)

    if nrows == 1:
        axes = [axes]

    idx = 0

    def _to_np(x):
        if isinstance(x, _torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    for k in present_lma:
        y = _to_np(tags[k])
        axes[idx].plot(y, color="C0")
        axes[idx].set_ylabel(k)
        axes[idx].grid(True)
        idx += 1

    for k in present_ctrl:
        y = _to_np(tags[k])
        axes[idx].plot(np.atleast_1d(y), color="C1")
        axes[idx].set_ylabel(k)
        axes[idx].grid(True)
        idx += 1

    axes[-1].set_xlabel("Frame")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_forward_motion_vector(
    tags,
    forward_key="ctrl_forward_alignment",
    side_key="ctrl_lateral_alignment",
    figsize=(10, 5),
    title=None,
    savepath=None,
):
    """Plot the facing-local alignment controls.

    `ctrl_forward_alignment` is positive when moving in the facing direction and
    negative when moving backward.

    `ctrl_lateral_alignment` is positive when moving to the character's right and
    negative when moving to the character's left.

    The pair is computed as forward/right unit basis vectors dotted with the
    unnormalized XZ velocity, so the sign captures alignment and the norm
    captures speed.
    """
    import torch as _torch

    missing = [key for key in (forward_key, side_key) if key not in tags]
    if missing:
        raise KeyError(f"Missing forward-motion vector keys: {missing}")

    forward = tags[forward_key]
    side = tags[side_key]
    if isinstance(forward, _torch.Tensor):
        forward = forward.detach().cpu().numpy()
    else:
        forward = np.asarray(forward)
    if isinstance(side, _torch.Tensor):
        side = side.detach().cpu().numpy()
    else:
        side = np.asarray(side)

    forward = np.squeeze(forward)
    side = np.squeeze(side)
    magnitude = np.sqrt(forward**2 + side**2)
    x = np.arange(forward.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].plot(x, forward, color="C2", linewidth=1.5, label="forward")
    axes[0].plot(x, side, color="C4", linewidth=1.5, label="lateral")
    axes[0].axhline(0.0, color="0.5", linewidth=0.8, linestyle="-")
    axes[0].set_ylabel("component")
    axes[0].set_title(title or "Forward/Lateral Alignment")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(x, magnitude, color="C1", linewidth=1.5)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("magnitude")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_forward_motion_alignment(tags, figsize=(10, 5), title=None, savepath=None):
    """Backward-compatible wrapper for the forward-local motion vector plot."""
    return plot_forward_motion_vector(
        tags, figsize=figsize, title=title, savepath=savepath
    )
