import importlib
import torch
import torch.nn as nn
from loss import MSE_DQ
from motion_data import (
    ROOT_CHANNELS_ARE_GLOBAL_POSITIONS,
    integrate_root_translation_torch,
    split_motion_joints,
)
# from torch.optim.lr_scheduler import LambdaLR
from pymotion.ops.skeleton import from_root_dual_quat_torch, translation_each_joint_torch, compute_global_pos_torch
from pymotion.rotations.dual_quat_torch import to_rotation_translation
import pymotion.rotations.ortho6d_torch as ortho6d
import numpy as np
import matplotlib.pyplot as plt


def _unique_feet_indices(indices):
    return list(dict.fromkeys(int(index) for index in indices))


def _load_autoencoder_classes(module_name: str):
    module = importlib.import_module(module_name)
    return module.Autoencoder, module.StaticEncoder

class Generator_Model(nn.Module):
    def __init__(self, device, param, parents, train_data, is_vae = False, is_vq_vae= False) -> None:
        super().__init__()

        self.device = device
        self.param = param
        self.parents = parents
        self.synthetic_contact_joint_count = int(
            param.get("synthetic_contact_joint_count", 0)
        )
        self.motion_parents = list(parents) + [0] * self.synthetic_contact_joint_count
        self.data = train_data
        self.is_vae = bool(param.get("use_vae", False) or is_vae)
        self.is_vq_vae = bool(is_vq_vae) and not self.is_vae
        self.training_stage = param.get("training_stage", "rnn") # "rnn" or "vq_vae"
        self.enable_root_loss = False

        autoencoder_module_name = param.get(
            "autoencoder_module",
            "autoencoder_no_enc_9_groups_no_rootbranch",
        )
        Autoencoder, StaticEncoder = _load_autoencoder_classes(autoencoder_module_name)
        self.autoencoder_module_name = autoencoder_module_name

        self.static_encoder = StaticEncoder(param, parents, device).to(device)
        self.autoencoder = Autoencoder(
            param,
            self.motion_parents,
            device,
            is_vae=self.is_vae,
            is_vq_vae=self.is_vq_vae,
        ).to(device)
        self.supports_continuous_latent = getattr(self.autoencoder, "supports_continuous_latent", False)

        parameters = list(self.static_encoder.parameters()) + list(
            self.autoencoder.parameters()
        )
        self.parameters = parameters
        self.named_params = list(self.static_encoder.named_parameters()) + list(self.autoencoder.named_parameters())

        # Print number parameters
        dec_params = 0
        for parameter in parameters:
            dec_params += parameter.numel()
        print("# parameters generator:", dec_params)

        param["learning_rate"] = 1e-3
        param["warmup_steps"] = 1000 
        
        self.optimizer = torch.optim.AdamW(self.parameters, param["learning_rate"])            
        self.loss = MSE_DQ(param, parents, device).to(device)
        train_data.losses.append(self.loss)
        self.prior_ctrl_grad_norm = 0.0
        self.prior_lma_grad_norm = 0.0
        self.prior_root_loss = 0.0
        self.current_kl_beta = 1.0
        self.optimization_step = 0
        self.posterior_mean = None
        self.posterior_logvar = None

    def forward(self):
       
        # Execute Static Encoder to obtain the offsets
        # input offsets has shape (1, n_joints, 3)

        self.ae_offsets = self.static_encoder(self.data.offsets)
        self.autoencoder.set_training_stage(self.training_stage)
        # Execute Autoencoder to obtain the motion        
        if self.is_vq_vae or self.supports_continuous_latent:
            self.res_decoder, self.vq_dict, self.output_logits, \
            self.target_indices,self.rots_waypoints, self.phi, self.ctrl_signals_predicted, self.prior_root_prediction = self.autoencoder(
                self.data.sparse_motion,
                self.ae_offsets,
                self.data.mean_dqs,
                self.data.std_dqs,
                self.data.denorm_offsets,
                mean_root=self.data.mean_root,
                std_root=self.data.std_root,
                mean_sin_cos = [self.data.mean_sin,self.data.mean_cos],
                std_sin_cos = [self.data.std_sin, self.data.std_cos],
                tags = self.data.tags,
            )
            prior = getattr(self.autoencoder, "codebook_predictor2", None)
            self.ctrl_prediction_loss = getattr(
                prior,
                "last_ctrl_prediction_loss",
                torch.tensor(0.0, device=self.device),
            )
            get_variational_stats = getattr(self.autoencoder, "get_variational_stats", None)
            if callable(get_variational_stats):
                self.posterior_mean, self.posterior_logvar, _ = get_variational_stats()
            else:
                self.posterior_mean = None
                self.posterior_logvar = None
           
            self.enc_output, self.lstm_output = self.autoencoder.get_latents()

            dqs = self.res_decoder.permute(0,2,1)
            batch,frames,dims = dqs.shape
            dqs = dqs.flatten(0,1)
            dqs = dqs * self.data.std_dqs + self.data.mean_dqs
            dqs = dqs.reshape(batch,frames,dims//9,9) # reshape to (batch, frames, n_joints, 8)
            skeletal_dqs, _ = split_motion_joints(
                dqs,
                synthetic_joint_count=self.synthetic_contact_joint_count,
            )
            root_positions = integrate_root_translation_torch(
                skeletal_dqs[:, :, 0, :],
                self.data.global_pos,
            )
            dqs = ortho6d.to_dual_quat(skeletal_dqs)
            rots, self.pos = to_rotation_translation(dqs)

            pos_accum, glob_pos_zeroed = self.compute_pos_accum_and_global_zeroed(
                pos_t=self.pos,
                glob_pos_pred=self.data.global_pos,
            )
            self.pos_accum = pos_accum
            self.global_pos_zeroed = glob_pos_zeroed
            
            glob_pos ,glob_rots = compute_global_pos_torch(
                rots,
                root_positions,
                self.data.denorm_offsets,
                self.parents,
                end_sites=self.data.end_sites,
                end_sites_parents=self.data.end_sites_parents,
            )

            unique_feet = _unique_feet_indices(self.param.get("feet_idxs", []))
            self.predicted_foot_positions = (
                glob_pos[..., unique_feet, :] if unique_feet else None
            )
            diffs = torch.diff(glob_pos[..., self.param["feet_idxs"], :], dim=1)
            norms = torch.norm(diffs, dim = -1)            
            self.velo = norms

            # self.plot_pos_accum_and_global_zeroed(batch_idx=0, show=True)

        else:
            return getattr(self, "res_decoder", None)

        return self.res_decoder

    @staticmethod
    def _grad_norm_from_parameters(parameters):
        total_sq = 0.0
        has_grad = False
        for parameter in parameters:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            total_sq += grad.pow(2).sum().item()
            has_grad = True
        return total_sq ** 0.5 if has_grad else 0.0

    def _update_prior_gradient_diagnostics(self):
        prior = getattr(self.autoencoder, "codebook_predictor2", None)
        if prior is None:
            self.prior_ctrl_grad_norm = 0.0
            self.prior_lma_grad_norm = 0.0
            return

        self.prior_ctrl_grad_norm = self._grad_norm_from_parameters(
            prior.ctrl_encoder.parameters(),
        )

        lma_parameters = list(prior.lma_raw_encoder.parameters())
        lma_encoded_proj = getattr(prior, "lma_encoded_proj", None)
        if lma_encoded_proj is not None:
            lma_parameters.extend(lma_encoded_proj.parameters())

        self.prior_lma_grad_norm = self._grad_norm_from_parameters(lma_parameters)

    def _get_vae_kl_beta(self):
        if not self.is_vae:
            return 1.0

        target_beta = float(self.param.get("vae_kl_beta", 1e-3))
        start_beta = float(self.param.get("vae_kl_beta_start", 0.0))
        warmup_steps = max(1, int(self.param.get("vae_kl_warmup_steps", 10000)))
        progress = min(1.0, float(self.optimization_step) / float(warmup_steps))
        return start_beta + (target_beta - start_beta) * progress

    
    def optimize_parameters_vq_vae(self):
        self.optimizer.zero_grad()
         
        loss, KLD, logits_loss, velo_loss, rots_incr_loss, vel_loss, acc_loss, spectral_loss, pos_xz_loss, prior_root_loss = self.loss.forward_generator_vq_vae(
            self.res_decoder,
            self.data.motion,
            self.vq_dict,
            self.output_logits, # for rnn
            self.target_indices, # for rnn
            velo = self.velo,
            target_velo = self.data.tags["velo_foot"],
            means=self.data.mean_dqs,
            stds =self.data.std_dqs,
            yaw_target = self.data.rots,
            yaw_pred = self.rots_waypoints,
            phi = self.phi,
            enable_root_loss=self.enable_root_loss,
            pos_accum=getattr(self, 'pos_accum', None),
            global_pos_zeroed=getattr(self, 'global_pos_zeroed', None),
            predicted_feet=getattr(self, 'predicted_foot_positions', None),
            target_foot_positions=getattr(self.data, 'foot_positions', None),
            foot_contact_binary=self.data.tags.get("foot_contact_binary"),
            root_velocity_weight=self.param.get("root_velocity_loss_weight", 0.0),
            pos_xz_weight=self.param.get("root_position_xz_loss_weight", 0.0),
            foot_sliding_weight=self.param.get(
                "contact_aware_foot_sliding_loss_weight", 0.0
            ),
            prior_root_pred=getattr(self, 'prior_root_prediction', None),
            prior_root_weight=self.param.get("prior_root_loss_weight", 1.0),
        )

        self.prior_root_loss = prior_root_loss.detach().item() if torch.is_tensor(prior_root_loss) else 0.0

        self.current_kl_beta = self._get_vae_kl_beta()
        latent_loss_weight = self.current_kl_beta if self.is_vae else 1.0
        total_loss = loss + latent_loss_weight * KLD + logits_loss * 0 #+ vq_recon_loss#+ KLD_timing#+ KLD_timing # + KLD_h # + KLD_t + predicted_timing_loss
       
        if self.training_stage == "rnn" and self.output_logits is not None:

            ctrl_prediction_loss = getattr(self, "ctrl_prediction_loss", None)
            if ctrl_prediction_loss is None:
                ctrl_prediction_loss = torch.tensor(0.0, device=logits_loss.device)

            loss = logits_loss + ctrl_prediction_loss + prior_root_loss
            total_loss = loss #+ rots_incr_loss

            for name, param in self.named_params:
                if 'codebook_predictor2' in name: #or 'tag_enc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        total_loss.backward()
        if self.training_stage == "rnn" and self.output_logits is not None:
            self._update_prior_gradient_diagnostics()
        else:
            self.prior_ctrl_grad_norm = 0.0
            self.prior_lma_grad_norm = 0.0
        torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=1.0)
        self.optimizer.step()
        self.optimization_step += 1

        return loss, KLD, velo_loss, rots_incr_loss, vel_loss, acc_loss, spectral_loss, pos_xz_loss, prior_root_loss

    def compute_pos_accum_and_global_zeroed(self, pos_t=None, glob_pos_pred=None):
        """Compute differentiable accumulated root position and zeroed global positions.

        Args:
            pos_t: torch.Tensor of shape [B, T, J, 3] or [T, J, 3]
            glob_pos_pred: predicted global positions tensor from compute_global_pos_torch
        Sets:
            self.pos_accum: [B, T, 3]
            self.global_pos_zeroed: [B, T, 3]
        Returns:
            pos_accum, global_pos_zeroed
        """
        if pos_t is None:
            pos_t = getattr(self, 'pos', None)

        if pos_t is None:
            raise ValueError('No position tensor available for accumulation')

        if pos_t.dim() == 4:
            root_pos = pos_t[:, :, 0, :]
        elif pos_t.dim() == 3:
            root_pos = pos_t.unsqueeze(0)[:, :, 0, :]
        else:
            root_pos = pos_t

        if ROOT_CHANNELS_ARE_GLOBAL_POSITIONS:
            pos_accum = root_pos.clone()
        else:
            abs_x = torch.cumsum(root_pos[..., 0], dim=1)
            abs_z = torch.cumsum(root_pos[..., 2], dim=1)
            pos_accum = torch.stack([abs_x, root_pos[..., 1], abs_z], dim=-1)
        self.pos_accum = pos_accum

        # build zeroed global pos
        gp_pred = glob_pos_pred if glob_pos_pred is not None else locals().get('glob_pos', None)
        try:
            if gp_pred is None:
                raise ValueError
            if gp_pred.dim() == 4:
                gp_root = gp_pred[:, :, 0, :]   # [B, T, J, 3] -> [B, T, 3]
            elif gp_pred.dim() == 3:
                gp_root = gp_pred               # already [B, T, 3]
            else:
                gp_root = gp_pred
            global_pos_zeroed = gp_root - gp_root[:, :1, :]
        except Exception:
            global_pos_zeroed = pos_accum - pos_accum[:, :1, :]

        self.global_pos_zeroed = global_pos_zeroed
        return pos_accum, global_pos_zeroed

    def plot_pos_accum_and_global_zeroed(self, batch_idx: int = 0, show: bool = True):
        """Simple plot for `self.pos_accum` and `self.global_pos_zeroed` for a given batch index.

        Args:
            batch_idx: index in batch dimension to plot
            show: whether to call `plt.show()`
        Returns:
            (fig, axes)
        """
        if not hasattr(self, 'pos_accum') or not hasattr(self, 'global_pos_zeroed'):
            raise RuntimeError('pos_accum or global_pos_zeroed not computed yet')

        pa = self.pos_accum[batch_idx].cpu().detach().numpy()  # [T, 3]
        gp = self.global_pos_zeroed[batch_idx].cpu().detach().numpy()  # [T, 3]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(pa)
        axes[0].set_title('pos_accum (X,Y,Z)')
        axes[0].legend(['X', 'Y', 'Z'])
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(gp)
        axes[1].set_title('global_pos_zeroed (X,Y,Z)')
        axes[1].legend(['X', 'Y', 'Z'])
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes