import torch
import torch.nn as nn
import pymotion.rotations.ortho6d_torch as ortho6d


class IK_NET(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()

        self.param = param
        self.parents = [None if parent is None else int(parent) for parent in parents]
        self.device = device
        self.motion_channels = 9
        self.hidden_size = 128
        self.sparse_joint_ids = [
            int(joint_id) for joint_id in self.param["sparse_joints"]
        ]
        self.head_idx = int(self.param.get("head_idx", -1))

        self.limb_configs = self._build_limb_configs()
        if not self.limb_configs:
            raise ValueError(
                "IK_NET could not derive any limb chains from sparse_joints={}".format(
                    self.sparse_joint_ids
                )
            )

        self.target_joint_ids = [
            config["effector_joint_id"] for config in self.limb_configs
        ]
        self.limb_networks = nn.ModuleDict()
        for config in self.limb_configs:
            input_size = (
                len(config["offset_joint_ids"]) * 3
                + len(config["input_joint_ids"]) * self.motion_channels
                + self.motion_channels
            )
            output_size = len(config["output_joint_ids"]) * self.motion_channels
            self.limb_networks[config["name"]] = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, output_size),
            ).to(device)

    def _expand_joint_channels(self, joint_ids):
        return [
            channel
            for joint_id in joint_ids
            for channel in range(
                joint_id * self.motion_channels,
                (joint_id + 1) * self.motion_channels,
            )
        ]

    def _chain_to_root(self, joint_id):
        chain = []
        current = int(joint_id)
        visited = set()
        while current is not None and current not in visited:
            chain.append(current)
            visited.add(current)
            parent = self.parents[current]
            current = None if parent is None else int(parent)
        return chain

    def _build_limb_configs(self):
        configs = []
        seen_effectors = set()
        for joint_id in self.sparse_joint_ids[1:]:
            if joint_id == self.head_idx or joint_id in seen_effectors:
                continue
            seen_effectors.add(joint_id)
            chain_to_root = self._chain_to_root(joint_id)
            non_root_chain = [
                chain_joint for chain_joint in chain_to_root if chain_joint != 0
            ]
            output_joint_ids = list(reversed(non_root_chain[:3]))
            if not output_joint_ids:
                continue
            configs.append(
                {
                    "name": "effector_{}".format(joint_id),
                    "effector_joint_id": joint_id,
                    "sparse_joint_position": self.sparse_joint_ids.index(joint_id),
                    "input_joint_ids": chain_to_root,
                    "offset_joint_ids": output_joint_ids,
                    "output_joint_ids": output_joint_ids,
                    "output_channel_indices": self._expand_joint_channels(
                        output_joint_ids
                    ),
                    "input_channel_indices": self._expand_joint_channels(chain_to_root),
                }
            )
        return configs

    def _gather_motion_channels(self, motion, channel_indices, frame=None):
        if frame is None:
            return motion[:, channel_indices, :]
        return motion[:, channel_indices, frame]

    def _normalize_output_channels(
        self, prediction, channel_indices, mean_dqs, std_dqs
    ):
        mean = mean_dqs[channel_indices].to(prediction.device, dtype=prediction.dtype)
        std = std_dqs[channel_indices].to(prediction.device, dtype=prediction.dtype)
        std = std.clamp_min(1e-8)

        if prediction.dim() == 3:
            denormalized = prediction * std.view(1, -1, 1) + mean.view(1, -1, 1)
            batch_size, _, frame_count = denormalized.shape
            joint_major = denormalized.permute(0, 2, 1).reshape(
                batch_size,
                frame_count,
                -1,
                self.motion_channels,
            )
            joint_major = ortho6d.normalize(joint_major)
            renormalized = (
                joint_major.reshape(batch_size, frame_count, -1).permute(0, 2, 1)
                - mean.view(1, -1, 1)
            ) / std.view(1, -1, 1)
            return renormalized

        denormalized = prediction * std.view(1, -1) + mean.view(1, -1)
        joint_major = denormalized.reshape(-1, self.motion_channels)
        joint_major = ortho6d.normalize(joint_major)
        return (
            joint_major.reshape(prediction.shape[0], -1) - mean.view(1, -1)
        ) / std.view(1, -1)

    def forward(
        self,
        decoder_output,
        sparse_input,
        mean_dqs,
        std_dqs,
        offsets,
        frame,
    ):
        sparse_joint_count = len(self.sparse_joint_ids)
        sparse_joint_features = sparse_input[
            :, : sparse_joint_count * self.motion_channels, :
        ].reshape(
            sparse_input.shape[0],
            sparse_joint_count,
            self.motion_channels,
            sparse_input.shape[-1],
        )

        refined_output = decoder_output.clone()
        for config in self.limb_configs:
            limb_prediction = self.forward_limb(
                self.limb_networks[config["name"]],
                sparse_joint_features,
                offsets,
                refined_output,
                config,
                std_dqs,
                mean_dqs,
                frame,
            )
            if frame is None:
                refined_output[:, config["output_channel_indices"], :] = limb_prediction
            else:
                refined_output[:, config["output_channel_indices"], frame] = (
                    limb_prediction.squeeze(-1)
                )
        return refined_output

    def forward_limb(
        self,
        sequential,
        sparse_joint_features,
        offsets,
        decoder_output,
        config,
        std_dqs,
        mean_dqs,
        frame,
    ):
        effector_features = sparse_joint_features[
            :, config["sparse_joint_position"], :, :
        ]
        if frame is None:
            frame_count = decoder_output.shape[-1]
            offset_features = offsets[:, config["offset_joint_ids"], :].flatten(
                start_dim=1,
                end_dim=2,
            )
            offset_features = offset_features.unsqueeze(-1).expand(-1, -1, frame_count)
            decoder_features = self._gather_motion_channels(
                decoder_output,
                config["input_channel_indices"],
            )
            prediction = sequential(
                torch.cat(
                    (offset_features, decoder_features, effector_features),
                    dim=1,
                ).permute(0, 2, 1)
            ).permute(0, 2, 1)
            return self._normalize_output_channels(
                prediction,
                config["output_channel_indices"],
                mean_dqs,
                std_dqs,
            )

        offset_features = offsets[:, config["offset_joint_ids"], :].flatten(
            start_dim=1,
            end_dim=2,
        )
        decoder_features = self._gather_motion_channels(
            decoder_output,
            config["input_channel_indices"],
            frame=frame,
        )
        prediction = sequential(
            torch.cat(
                (
                    offset_features,
                    decoder_features,
                    effector_features[:, :, frame],
                ),
                dim=1,
            )
        )
        prediction = self._normalize_output_channels(
            prediction,
            config["output_channel_indices"],
            mean_dqs,
            std_dqs,
        )
        return prediction.unsqueeze(-1)
