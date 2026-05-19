import random
import torch
from torch.utils.data.dataloader import DataLoader
from motion_data import TestMotionData, TrainMotionData, ROOT_CHANNELS_ARE_GLOBAL_POSITIONS, USE_CANONICAL_XZ_POSITIONS, compute_phase
import pymotion.rotations.quat as quat
import pymotion.rotations.dual_quat as dquat
import pymotion.rotations.ortho6d as ortho6d
from pymotion.ops.skeleton import from_root_dual_quat, translation_each_joint,from_root_dual_quat_to_root
from pymotion.io.bvh import BVH
from pymotion.rotations.dual_quat import to_rotation_translation
import numpy as np
from train_data import Train_Data
from generator_architecture import Generator_Model
from ik_architecture import IK_Model
from scipy.interpolate import interp1d
import time
import argparse
import os
import eval_metrics
scale = 1
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

# Train Modes
GENERATOR = 1
IK = 2

incr_rots = True

human_param = {
    "batch_size": 64,
    "epochs": 1000,
    "kernel_size_temporal_dim": 7,
    "neighbor_distance": 0,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        4,  # left foot
        8,  # right foot
        #11, # chest
        13,  # head
        17,  # left hand
        21,  # right hand
        #22,  #dummy
        
    ],
    "window_size": 64,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -11,
    "ema_updates": 74,#36,#205,
    "codebook_size": 512,
    "ema_decay": 0.9,
    "skeleton_height": 0.96,
    "head_idx": 13,
    "head_height": 1.6,
    "feet_idxs": [3,7,3,7],
    "feet_contact_threshold": 0.008,
    "not_dog": True,
    "root_branch_dim": 64,
    "gru_window": 8,
    "gru_layers": 2,
    "gru_hidden_dim": 512,
    "input_proj": -1,
}


human_param2 = {
    "batch_size": 128,
    "epochs": 1000,
    "kernel_size_temporal_dim": 7,
    "neighbor_distance": 0,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        4,  # left foot
        8,  # right foot
        #11, # chest
        13,  # head
        17,  # left hand
        21,  # right hand
        #22,  #dummy
        
    ],
    "window_size": 128,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -11,
    "ema_updates": 205,
    "codebook_size": 512,
    "ema_decay": 0.9,
    "skeleton_height": 0.84,
    "head_idx": 13,
    "head_height": 1.42,
    "feet_idxs": [3,7,3,7],
    "not_dog": True,
}

human_param_xsens = {
    "batch_size": 256,
    "epochs": 1000,
    "kernel_size_temporal_dim": 7,
    "neighbor_distance": 2,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        4,  # left foot
        8,  # right foot
        #11, # chest
        13,  # head
        17,  # left hand
        21,  # right hand
        #22,  #dummy
        
    ],
    "window_size": 64,
    "window_step": 16,
    "seed": 2222,
    "extra_joint": -1 #11 
}

#dog params <------- reordered

dog_param_no_tail = {
    "batch_size": 4,
    "epochs": 10000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 2,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        15,  # head
        18,  # left hand
        4,  # right hand
        8,  # left foot
        12,  # right foot
    ],
    "window_size": 64,
    "window_step": 4,
    "seed": 2222,
    "extra_joint": -1,
}



#dog params <-------
dog_param = {
    "batch_size": 16,
    "epochs": 500,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        4,  # head
        8,  # left hand
        12,  # right hand
        15,  # left foot
        18,  # right foot
        20,  # tail
    ],
    "window_size": 64,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 201,
    "codebook_size": 512,
    "ema_decay": 0.9,
    "skeleton_height": 0.9,
    "head_idx": 2,
    "head_height": 1.0,
    "feet_idxs": [8,12,15,18],
    "feet_contact_threshold": 0.07,
    "not_dog": False,
    "root_branch_dim": 32,
    "gru_window": 4,
    "gru_layers": 2,
    "gru_hidden_dim": 256,
    "gru_vq_dim": 64,
    "input_proj": -1,
}


alligator_param = {
    "batch_size": 32,
    "epochs": 4000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        7,  # head
        10,  # left hand
        13,  # right hand
        18,  # left foot
        21,  # right foot
        24,  # tail
    ],
    "window_size": 32,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 35,
    "codebook_size": 64,
    "ema_decay": 0.5,
}


ostrich_param_ = {
    "batch_size": 32,
    "epochs": 4000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 0,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,
        5,
        12,
        13,
        #16,
        19,
        20,
        #26,
        26,
        28,
        #35,
        33,
        35,
        #43,
        #50,
        #51,
        #52,
        40,
        42,
    ],
    "window_size": 32,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 35,
    "codebook_size": 32,
    "ema_decay": 0.5,
}



# ostrich_param = {
#     "batch_size": 32,
#     "epochs": 4000,
#     "kernel_size_temporal_dim": 15,
#     "neighbor_distance": 0,
#     "stride_encoder_conv": 2,
#     "learning_rate": 1e-4,
#     "lambda_root": 10,
#     "lambda_ee": 10 / scale,
#     "lambda_ee_reg": 1 / scale,
#     "sparse_joints": [
#         0,
#         5,
#         12, #
#         13,
#         15, #
#         16,
#         22, #
#         23,
#         25,
#         26,
#         32,
#         34,
#         35,
#         40,
#         42,
#         43,
#         48, #
#         50,
#         51,
#         52,
#         53,
#     ],
#     "window_size": 32,
#     "window_step": 8,
#     "seed": 2222,
#     "extra_joint": -1,
#     "ema_updates": 35,
#     "codebook_size": 32,
#     "ema_decay": 0.5,
#     "skeleton_height": 0.69,
#     "head_idx": 53,
#     "head_height": 1.12,
#     "root_branch_dim": 8,
#     "gru_window": 1,
#     "gru_layers": 1,
#     "gru_hidden_dim": 32,
# }

cat_param = {
    "batch_size": 16,
    "epochs": 2000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        5, # tail
        12, #r toe
        18, #l toe
        27, #r finger
        33, #l finger
        43, # head
    ],
    "window_size": 64,
    "window_step": 16,
    "seed": 2222,
    "extra_joint": -1,
}


ostrich_param = {
    "batch_size": 32,
    "epochs": 300,
    "kernel_size_temporal_dim": 3,
    "neighbor_distance": 0,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,
        5,
        12, #
        13,
        15, #
        16,
        22, #
        23,
        25,
        26,
        32,
        34,
        35,
        40,
        42,
        43,
        48, #
        50,
        51,
        52,
        53,
    ],
    "window_size": 128,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 75,
    "codebook_size": 16,
    "ema_decay": 0.5,
    "skeleton_height": 0.69,
    "head_idx": 53,
    "head_height": 1.12,
    "feet_idxs": [20,10,20,10],
    "feet_contact_threshold": 0.04,
    "not_dog": True,
    "root_branch_dim": 24,
    "gru_window": 1,
    "gru_layers": 1,
    "gru_hidden_dim": 16,
    "gru_vq_dim": 16,
    "input_proj": 32,
}


param = dog_param
rm_flag= False
frame_step = 1
param2 = human_param

assert param["kernel_size_temporal_dim"] % 2 == 1


def main(args):
    # Set seed
    torch.manual_seed(param["seed"])
    random.seed(param["seed"])
    np.random.seed(param["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    # Prepare Data
    train_eval_dir = args.data_path
    # check if train and eval directories exist
    train_dir = os.path.join(train_eval_dir, "train")
    if not os.path.exists(train_dir):
        raise ValueError("train directory does not exist")
    train_files = os.listdir(train_dir)
    eval_dir = os.path.join(train_eval_dir, "eval")
    if not os.path.exists(eval_dir):
        raise ValueError("eval directory does not exist")
    eval_files = os.listdir(eval_dir)
    train_dataset = TrainMotionData(param, scale, device)
    eval_dataset = TestMotionData(param, scale, device)
    reference_parents = None  # used to make sure all bvh have the same structure

    rots_missing_frames_list = []
    ground_truth_rots_list = []
    phase_list = []
    velocities_list = []
    phase_chunks = []
    velocity_chunks = []
    # Train Files
    for filename in train_files:
        if filename[-4:] == ".bvh":
            
            bvh_from_disk = get_bvh_from_disk(train_dir, filename)
            rots, pos, parents, offsets, _, og_rots = get_info_from_bvh(
                bvh_from_disk, get_missing_frames= False
            )
            #rots[0,0,:] = rots_missing_frames[0,0,:]
            ########################
            # create chunks of 256 frames each (for root predicotr)
            # list_og_rots = rots_missing_frames[0]
            # list_missing = rots_missing_frames[1]
            # num_frames = list_missing[0].shape[0]
            # chunk_size = 256
            # for i in range(frame_step):
            #     rots_missing_frames = list_missing[i]
            #     true_rots = list_og_rots[i]
            #     for start in range(0, num_frames, chunk_size):
            #         end = start + chunk_size
            #         current_chunk = torch.tensor(rots_missing_frames[start:end]).float()
                    
            #         # Check if current chunk is smaller than chunk_size and pad if necessary
            #         if current_chunk.shape[0] < chunk_size:
            #             padding = torch.zeros((chunk_size - current_chunk.shape[0], current_chunk.shape[1]))
            #             padding_gr_tr = torch.zeros((chunk_size - current_chunk.shape[0], ground_truth_chunk.shape[1]))
            #             current_chunk = torch.cat((current_chunk, padding), dim=0)
            #             ground_truth_chunk = torch.tensor(true_rots[start:end, 0, :]).float()  # actual root rotations
            #             ground_truth_chunk = torch.cat((ground_truth_chunk, padding_gr_tr.clone()), dim=0)
            #         else:
            #             ground_truth_chunk = torch.tensor(true_rots[start:end, 0, :]).float()

            #         rots_missing_frames_list.append(current_chunk)
            #         ground_truth_rots_list.append(ground_truth_chunk)
            #########################
            
            if reference_parents is None:
                reference_parents = parents.copy()
            assert (
                reference_parents == parents
            )  # make sure all bvh have the same structure
            # Train Dataset
            #pos_all_joints = translation_each_joint(rots, pos[:,0,:], parents, offsets)
            pos_all_joints = bvh_from_disk.compute_global_pos()

            train_dataset.add_motion(
                offsets,
                pos[:, 0, :],  # only global position
                rots,
                parents,
                pos_all_joints,
                og_rots=og_rots,
                end_sites=bvh_from_disk.data["end_sites"],
                end_sites_parents=bvh_from_disk.data["end_sites_parents"],
            )
            chunk_size = 512
            num_frames = pos_all_joints.shape[0]

            # Initialize lists to store results

    #         # Process pos_all_joints in chunks of 512 frames
    #         for start_index in range(0, num_frames, chunk_size):
    #             end_index = min(start_index + chunk_size, num_frames)  # Ensure we do not exceed array bounds
    #             pos_chunk = pos_all_joints[start_index:end_index]  # Shape: [frames_in_chunk, joints, 3]
                
    #             # Compute phase and velocity for the current chunk
    #             phase, velocity_xz = compute_phase(pos_chunk, indices_only=False)
    #             # import matplotlib.pyplot as plt
    #             # plt.figure(figsize=(12, 6))
    #             # # Plot Part 1
    #             # plt.subplot(2, 1, 1)
    #             # plt.plot(phase[0], label='Real Phase Part 1', color='blue')
    #             # plt.legend()
                
    #             # plt.tight_layout()
    #             # plt.show()

    #             # Pad phase and velocity if necessary
    #             if phase.shape[1] < chunk_size:
    #                 phase = np.pad(phase, ((0, 0), (0, chunk_size - phase.shape[1])), 'constant')  # Pad along frames dimension
    #             if len(velocity_xz) < chunk_size:
    #                 velocity_xz = np.pad(velocity_xz, (0, chunk_size - len(velocity_xz)), 'constant')  # Pad with zeros

    #             phase_chunks.append(phase)  # Shape: [2, frames]
    #             velocity_chunks.append(velocity_xz.reshape(1, -1))  # Shape: [1, frames]


    # print(len(rots_missing_frames_list), len(ground_truth_rots_list))
    ##########################################################
    ################ root rotation predictor #################
    train_root_predictor = False
    train_phase_predictor = False
    
                
    ##########################################################
    
    
    # Once all train files are added, compute the means and stds and normalize
    train_dataset.normalize()
    eval_dataset.set_means_stds(train_dataset.means, train_dataset.stds)
    # Eval Files
    for filename in eval_files:
        if filename[-4:] == ".bvh":
            rots, pos, parents, offsets, bvh, og_rots = get_info_from_bvh(
                get_bvh_from_disk(eval_dir, filename)
            )
            assert (
                reference_parents == parents
            )  # make sure all bvh have the same structure
            #pos_all_joints = translation_each_joint(rots, pos[:,0,:], parents, offsets)
            pos_all_joints = bvh.compute_global_pos()
            # Eval Dataset
            
            eval_dataset.add_motion(
                offsets,
                pos[:, 0, :],  # only global position
                rots,
                parents,
                bvh,
                filename,
                pos_all_joints,
                og_rots = og_rots,
                end_sites=bvh.data["end_sites"],
                end_sites_parents=bvh.data["end_sites_parents"],
            )
    # Once all eval files are added, normalize
    eval_dataset.normalize()

    train_dataloader = DataLoader(train_dataset, param["batch_size"], shuffle=True)
    # Create Models
    train_data = Train_Data(device, param)
    generator_model = Generator_Model(device, param, reference_parents, train_data, is_vq_vae=True).to(
        device
    )
    if args.train_mode & IK != 0:
        ik_model = IK_Model(device, param, reference_parents, train_data).to(device)
    train_data.set_means(train_dataset.means["dqs"])
    train_data.set_stds(train_dataset.stds["dqs"])
    train_data.set_root_means_stds(train_dataset.means["rots"],train_dataset.stds["rots"])
    train_data.set_sin_cos_means_stds(train_dataset.means["yaw_sin"], train_dataset.means["yaw_cos"]
                                      ,train_dataset.stds["yaw_cos"], train_dataset.stds["yaw_cos"])
    print(args)
    # Load Models
    print(args.train_mode & IK)
    _, generator_path, ik_path = get_model_paths(args.name, train_eval_dir)
    #if args.train_mode & GENERATOR == 0 or (args.load and args.train_mode & IK != 0):
    if args.train_mode & GENERATOR == 0 or (args.load):
        print("loading pretrained model")
        # Generator is always needed with IK, load it if not training it
        load_model(generator_model, generator_path, train_data, device)
    if args.train_mode & IK != 0 and args.load:
        load_model(ik_model, ik_path, train_data, device)

    if (args.train_mode & GENERATOR == 0 or args.train_mode & IK == 0) and args.load:
        # Check previous best evaluation loss
        results, disp_8, yaw_diff = evaluate_generator(generator_model, train_data, eval_dataset)
        if args.train_mode & IK != 0:
            results_ik = evaluate_ik(ik_model, results, train_data, eval_dataset)
            results = results_ik
        mpjpe, mpeepe = eval_save_result(
            results,
            train_dataset.means,
            train_dataset.stds,
            eval_dir,
            device,
            save=False,
            disp_8=disp_8,
            yaw_diff=yaw_diff
        )
        best_evaluation = mpjpe + mpeepe
    else:
        best_evaluation = float("inf")
    # Training Loop
    start_time = time.time()

    for epoch in range(param["epochs"]):
        avg_train_loss = 0.0
        avg_vq_loss = 0.0
        avg_perplexity = 0.0
        avg_ce_loss = 0.0

        for step, (denorm_motion, norm_motion) in enumerate(train_dataloader):
            # Forward
            train_data.set_offsets(norm_motion["offsets"], denorm_motion["offsets"])
            train_data.set_rot_order(bvh.data["rot_order"])
            train_data.set_end_sites(denorm_motion["end_sites"],denorm_motion["end_sites_parents"])
            train_data.set_motions(
                norm_motion["dqs"],
                norm_motion["displacement"],
                norm_motion["disp_8"],
                norm_motion["tags"]["sin_diff"],
                norm_motion["tags"]["cos_diff"], 
                norm_motion["loss_weights"],
            )
            train_data.set_phase(denorm_motion["phase"])
            train_data.set_phase_per_8_frames(denorm_motion["phase_per_8_frames"])      
            train_data.set_velocity_per_8_frames(denorm_motion["velocity_per_8_frames"])
            train_data.set_tags(norm_motion["tags"])
            train_data.set_rots(norm_motion["rots"])
            train_data.set_global_pos(denorm_motion["global_pos"])
            train_data.set_energy(denorm_motion["energy_feet"].clone().detach().to(device, dtype=torch.float32))
            
            if args.train_mode & GENERATOR != 0:
                generator_model.train()
            if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                res_decoder, disp_8, yaw_diff = generator_model.forward()
            if args.train_mode & IK != 0:
                ik_model.train()
                ik_model.forward(res_decoder)
            # Loss
            loss = 0.0
            kld_loss = 0.0
            if args.train_mode & GENERATOR != 0:

                loss_generator, vq_loss, perplexity, cross_entropy_loss = generator_model.optimize_parameters_vq_vae()
                loss = loss_generator.item()
                vq_loss = vq_loss.item()
                perplexity = perplexity.item()
                cross_entropy_loss = cross_entropy_loss.item()

            if args.train_mode & IK != 0:
                loss_ik = ik_model.optimize_parameters(res_decoder)
                loss += loss_ik.item()
                
            avg_train_loss += loss
            avg_vq_loss += vq_loss
            avg_perplexity += perplexity  
            avg_ce_loss += cross_entropy_loss  
            # Evaluate & Print
            if step == len(train_dataloader) - 1:
                if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                    results, disp_8, yaw_diff = evaluate_generator(
                        generator_model, train_data, eval_dataset
                    )
                    
                    if args.train_mode & IK != 0:
                        results_ik = evaluate_ik(
                            ik_model,
                            results,
                            train_data,
                            eval_dataset,
                        )
                        results = results_ik
                    if(epoch% 50 == 0 ):
                        save = True
                    else:
                        save = False
                    mpjpe, mpeepe = eval_save_result(
                        results,
                        train_dataset.means,
                        train_dataset.stds,
                        eval_dir,
                        device,
                        save=save,
                        disp_8=disp_8,
                        yaw_diff= yaw_diff
                    )
                    evaluation_loss = mpjpe + mpeepe
                # If best, save model
                was_best = False
                if evaluation_loss < best_evaluation:
                    save_model(
                        generator_model if args.train_mode & GENERATOR != 0 else None,
                        ik_model if args.train_mode & IK != 0 else None,
                        train_dataset,
                        args.name,
                        train_eval_dir,
                    )
                    best_evaluation = evaluation_loss
                    was_best = True
                    
                elif epoch % 10 == 0 and epoch!= 0: 
                    save_model_shared(
                        generator_model if args.train_mode & GENERATOR != 0 else None,
                        ik_model if args.train_mode & IK != 0 else None,
                        train_dataset,
                        args.name,
                        train_eval_dir,
                    )

                # Print
                avg_train_loss /= len(train_dataloader)
                avg_vq_loss /= len(train_dataloader)
                avg_perplexity /= len(train_dataloader)
                avg_ce_loss /= len(train_dataloader)

                if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                    # print(
                    #     "Epoch: {} - Train Loss: {:.4f} - KLD: {:.4f} - KLD_S: {:.4f} - mu,var: {:.4f},{:.4f} - Eval Loss: {:.4f} - MPJPE: {:.4f} - MPEEPE: {:.4f}".format(
                    #         epoch, avg_train_loss, avg_kld_loss, avg_kld_loss_s,mu_s,var_s,evaluation_loss, mpjpe, mpeepe
                    #     )
                    #         + ("*" if was_best else "")
                    # )
                                        print(
                        "Epoch: {} - Train Loss: {:.4f} - KLD: {:.14f} - velo_loss: {:.4f} - CE: {:.4f} - Eval Loss: {:.4f}".format(
                            epoch, avg_train_loss,avg_vq_loss, avg_perplexity, avg_ce_loss, evaluation_loss,
                        )
                            + ("*" if was_best else "")
                    )

    end_time = time.time()
    print("Training Time:", end_time - start_time)

    # Load Best Model -> Save and/or Evaluate
    if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
        load_model(generator_model, generator_path, train_data, device)
        results, disp_8, yaw_diff = evaluate_generator(generator_model, train_data, eval_dataset)
        if args.train_mode & IK != 0:
            load_model(ik_model, ik_path, train_data, device)
            results_ik = evaluate_ik(ik_model, results, train_data, eval_dataset)
            results = results_ik

        mpjpe, mpeepe = eval_save_result(
            results, train_dataset.means, train_dataset.stds, eval_dir, device, disp_8=disp_8, yaw_diff= yaw_diff
        )
        evaluation_loss = mpjpe + mpeepe

    print("Evaluate Loss: {}".format(evaluation_loss))
    if args.train_mode & (GENERATOR | IK) != 0:
        print("Mean Per Joint Position Error: {}".format(mpjpe))
        print("Mean End Effector Position Error: {}".format(mpeepe))


def eval_save_result(results, train_means, train_stds, eval_dir, device, save=True, disp_8 = None, yaw_diff=None):
    # Save Result
    array_mpjpe = np.empty((len(results),))
    array_mpeepe = np.empty((len(results),))
    for step, (res, bvh, filename) in enumerate(results):
        if save:
            eval_path, eval_filename = result_to_bvh(
                res, train_means, train_stds, bvh, filename, disp_8 = disp_8[step], yaw_diff=yaw_diff[step]
            )
            # Evaluate Positional Error
            mpjpe, mpeepe = eval_metrics.eval_pos_error(
                get_bvh_from_disk(eval_dir, filename),
                get_bvh_from_disk(eval_path, eval_filename, True),#<---------
                device,
            )
        else:
            result_to_bvh(res, train_means, train_stds, bvh, None, save=False)
            # Evaluate Positional Error
            mpjpe, mpeepe = eval_metrics.eval_pos_error(
                get_bvh_from_disk(eval_dir, filename),
                bvh,
                device,
            )

        array_mpjpe[step] = mpjpe
        array_mpeepe[step] = mpeepe

    return np.mean(array_mpjpe), np.mean(array_mpeepe)


# def load_model(model, model_path, train_data, device):
#     model_name = os.path.basename(model_path)[: -len(".pt")]
#     assert model_name == "generator" or model_name == "ik"
#     if model_name == "generator":
#         data_path = model_path[: -len("generator.pt")] + "data.pt"
#         checkpoint = torch.load(model_path, map_location=device)
#         model.load_state_dict(checkpoint["model_state_dict"])
#     elif model_name == "ik":
#         data_path = model_path[: -len("ik.pt")] + "data.pt"
#         checkpoint = torch.load(model_path, map_location=device)
#         model.load_state_dict(checkpoint["model_state_dict"])
#     data = torch.load(data_path, map_location=device)
#     means = data["means"]
#     stds = data["stds"]
#     train_data.set_means(means["dqs"])
#     train_data.set_stds(stds["dqs"])
#     return means, stds

def load_model(model, model_path, train_data, device, ignore_transform_net = True):
    model_name = os.path.basename(model_path)[: -len(".pt")]
    assert model_name == "generator" or model_name == "ik"
    if model_name == "generator":
        data_path = model_path[: -len("generator.pt")] + "data.pt"
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        
        # dont load transform net
        # if ignore_transform_net:
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.codebook_predictor')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.disp')}
            # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.codebook_predictor2.rot')}
            # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.codebook_predictor2.yaw')}
            # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.root_branch')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.ema')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.num_embeddings')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.logits')}
            # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.num_quantizers')}
            # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.input_proj')}
            # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.forward_dir_proj')}
            # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.root_branch')}
            # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.root_upsample')}
            # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.ctrl_and_input_proj')}
            # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.final_cnn_layer')}
        #state_dict = {k: v for k, v in state_dict.items() if (not 'timing_predictor' in k)}
        #state_dict = {k: v for k, v in state_dict.items() if (not 'timing_decoder' in k)}
        model.load_state_dict(state_dict, strict=False)
        
    elif model_name == "ik":
        data_path = model_path[: -len("ik.pt")] + "data.pt"
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    data = torch.load(data_path, map_location=device)
    means = data["means"]
    stds = data["stds"]
    train_data.set_means(means["dqs"])
    train_data.set_stds(stds["dqs"])
    train_data.set_root_means_stds(means["rots"],stds["rots"])
    train_data.set_sin_cos_means_stds(means["yaw_sin"], means["yaw_cos"], stds["yaw_sin"], stds["yaw_cos"])
    return means, stds

def get_model_paths(name, train_eval_dir):
    model_name = (
        "model_" + name + "_" + os.path.basename(os.path.normpath(train_eval_dir))
    )
    model_dir = os.path.join("models", model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data_path = os.path.join(model_dir, "data.pt")
    generator_path = os.path.join(model_dir, "generator.pt")
    ik_path = os.path.join(model_dir, "ik.pt")
    return data_path, generator_path, ik_path

def get_model_paths_shared(name, train_eval_dir):
    model_name = (
        "model_" + name + "_" + os.path.basename(os.path.normpath(train_eval_dir))
    )
    model_dir = os.path.join("models", model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data_path = os.path.join(model_dir, "best_root/data.pt")
    generator_path = os.path.join(model_dir, "best_root/generator.pt")
    ik_path = os.path.join(model_dir, "ik.pt")
    return data_path, generator_path, ik_path


def save_model(
    generator_model,
    ik_model,
    train_dataset,
    name,
    train_eval_dir,
):
    data_path, generator_path, ik_path = get_model_paths(name, train_eval_dir)

    if train_dataset is not None:
        torch.save(
            {
                "means": train_dataset.means,
                "stds": train_dataset.stds,
            },
            data_path,
        )
    if generator_model is not None:
        torch.save(
            {
                "model_state_dict": generator_model.state_dict(),
            },
            generator_path,
        )
    if ik_model is not None:
        torch.save(
            {
                "model_state_dict": ik_model.state_dict(),
            },
            ik_path,
        )

def save_model_shared(
    generator_model,
    ik_model,
    train_dataset,
    name,
    train_eval_dir,
):
    data_path, generator_path, ik_path = get_model_paths_shared(name, train_eval_dir)
    #generator_path = os.path.join(generator_path,"best_root")

    if train_dataset is not None:
        torch.save(
            {
                "means": train_dataset.means,
                "stds": train_dataset.stds,
            },
            data_path,
        )
    if generator_model is not None:
        torch.save(
            {
                "model_state_dict": generator_model.state_dict(),
            },
            generator_path,
        )
    if ik_model is not None:
        torch.save(
            {
                "model_state_dict": ik_model.state_dict(),
            },
            ik_path,
        )



def get_bvh_from_disk(path, filename, remove=True):
    remove=rm_flag
    path = os.path.join(path, filename)
    bvh = BVH()
    bvh.load(path)
    #remove lower body joints
    if remove:
        bvh.remove_joints([19,20])
    return bvh


def get_info_from_bvh(bvh, incremental_rots = False, get_missing_frames = False, get_phase = False):
    rot_roder = np.tile(bvh.data["rot_order"][0], (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1))  # made a change here
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_roder),
        axis=0,
    )
    rots = quat.normalize(rots)  # make sure all quaternions are unit quaternions
    og_rots = rots.copy()
    
    # rots = quat.normalize(rots)
    pos = bvh.data["positions"]
    parents = bvh.data["parents"]
    parents[0] = 0  # BVH sets root as None
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)  # force to zero offset for root joint
    
    if(incremental_rots):
        rots = quat.compute_incremental_quaternions(og_rots)
        #rots = quat.compute_incremental_intermittent_quaternions(rots)
        
    if(get_missing_frames):
        list_og_rots = []
        list_missing_frames = []
        foot_1_idx=15
        foot_2_idx=18
        #og_rots= quat.compute_incremental_quaternions(og_rots)
        
        for i in range(frame_step):            
            #rots_missing_frames = quat.compute_filtered_quaternions(og_rots[i:,:,:],pos[i:,0,:])#
            rots_missing_frames = quat.compute_incremental_intermittent_quaternions(og_rots[i:,:,:], pos[i:,0,:])
            phase, _  = compute_phase(pos[i:],joint_id=[0])
            phase = phase.squeeze()
            rots_missing_frames = np.concatenate([rots_missing_frames,phase[:,np.newaxis]],axis=1)
            #rots_missing_frames =  rots_filtered_frames
            #rots_missing_frames = quat.compute_incremental_intermittent_quaternions(og_rots[i:,:,:], pos[i:,0,:])
            #rots_missing_frames[:,4:] = rots_filtered_frames[:,4:].copy()
            og_quats = og_rots[i:,:,:]#og_quats = quat.compute_incremental_quaternions(og_rots[i:,:,:])
            displacement = pos[i+1:, 0, :] - pos[i:-1, 0, :]
            displacement = np.vstack((np.zeros((1, 3)), displacement))
            true_values = np.zeros((og_quats.shape[0], og_quats.shape[1], 7)) # [frames, joints, quat(4)+displacement(3)+4+4(feet)]
            #true_values[:,0,:] = np.concatenate((og_quats[:,0,:], og_quats[:,foot_1_idx,:], og_quats[:,foot_2_idx,:], displacement), axis=-1) # quaternion + displacement [w,x,y,z] + [x,y,z]
            true_values[:,0,:] = np.concatenate((og_quats[:,0,:], displacement), axis=-1)
            list_og_rots.append(true_values)    # actual rotations and displacements
            list_missing_frames.append(rots_missing_frames)
        
        return rots, pos, parents, offsets, bvh, (list_og_rots, list_missing_frames)
    
    if(get_phase):
        list_og_rots = []
        list_missing_frames = []
        foot_1_idx=15
        foot_2_idx=18
        #og_rots= quat.compute_incremental_quaternions(og_rots)
        
      
        
        return rots, pos, parents, offsets, bvh, (list_og_rots, list_missing_frames)

    return rots, pos, parents, offsets, bvh, og_rots


def evaluate_generator(generator_model, train_data, dataset, sparse_motions=None):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    generator_model.eval()
    results = []
    yaw_diffs = []
    disps = []
    with torch.no_grad():
        for index in range(dataset.get_len()):
            norm_motion = dataset.get_item(index)
            train_data.set_offsets(
                norm_motion["offsets"].unsqueeze(0),
                norm_motion["denorm_offsets"].unsqueeze(0),
            )
            train_data.set_end_sites(
                torch.tensor(norm_motion["end_sites"], dtype=torch.float32).to('cuda').unsqueeze(0),
                torch.tensor(norm_motion["end_sites_parents"], dtype=torch.float32).to('cuda').unsqueeze(0),
            )
            train_data.set_motions(
                norm_motion["dqs"].unsqueeze(0),
                norm_motion["displacement"].unsqueeze(0),
                norm_motion["disp_8"].unsqueeze(0),
                norm_motion["tags"]["sin_diff"].unsqueeze(0),
                norm_motion["tags"]["cos_diff"].unsqueeze(0),  
                norm_motion["loss_weights"].unsqueeze(0),
            )
            train_data.set_rots(
                norm_motion["rots"].unsqueeze(0),
            )
            train_data.set_phase(
                torch.tensor(norm_motion["phase"], dtype=torch.float32).to('cuda').unsqueeze(0)
            )
            train_data.set_phase_per_8_frames(
                torch.tensor(norm_motion["phase_per_8_frames"], dtype=torch.float32).to('cuda').unsqueeze(0)
            )
            train_data.set_velocity_per_8_frames(  
                torch.tensor(norm_motion["velocity_per_8_frames"], dtype=torch.float32).to('cuda').unsqueeze(0)
            )


            tags_tensor_dict = {
            key: value.clone().detach().unsqueeze(0) #torch.tensor(value, dtype=torch.float32).to('cuda').unsqueeze(0)
            for key, value in norm_motion["tags"].items()
            }
            train_data.set_tags(
               tags_tensor_dict
            )
            train_data.set_energy(
               norm_motion["energy_feet"].clone().detach()#torch.tensor(norm_motion["energy_feet"],dtype=torch.float32).to('cuda').unsqueeze(0)
            )
        
            if sparse_motions is not None:
                train_data.set_sparse_motion(sparse_motions[index])
            bvh, filename = dataset.get_bvh(index)
            train_data.set_rot_order(bvh.data["rot_order"])
            train_data.set_global_pos(torch.tensor(bvh.data["positions"][:,0]).to('cuda').unsqueeze(0))  
            #train_data.set_rots(bvh.data["rotations"])
            res, disp_8, yaw_diff = generator_model.forward()
            results.append((res, bvh, filename))
            disps.append(disp_8)
            yaw_diffs.append(yaw_diff)
            
            
    return results, disps, yaw_diffs


def evaluate_ik(ik_model, results_decoder, train_data, dataset):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    ik_model.eval()
    results = []
    with torch.no_grad():
        for index in range(dataset.get_len()):
            norm_motion = dataset.get_item(index)
            train_data.set_offsets(
                norm_motion["offsets"].unsqueeze(0),
                norm_motion["denorm_offsets"].unsqueeze(0),
            )
            train_data.set_motions(
                norm_motion["dqs"].unsqueeze(0),
                norm_motion["displacement"].unsqueeze(0),
                norm_motion["disp_8"].unsqueeze(0),
                norm_motion["tags"]["sin_diff"].unsqueeze(0),
                norm_motion["tags"]["cos_diff"].unsqueeze(0),   
                norm_motion["loss_weights"].unsqueeze(0),
            )
            res = ik_model.forward(results_decoder[index][0])
            bvh, filename = dataset.get_bvh(index)
            results.append((res, bvh, filename))
    return results


def run_set_data(train_data, dataset):
    with torch.no_grad():
        norm_motion = dataset.get_item()
        train_data.set_offsets(
            norm_motion["offsets"].unsqueeze(0),
            norm_motion["denorm_offsets"].unsqueeze(0),
        )
        train_data.set_motions(
            norm_motion["dqs"].unsqueeze(0),
            norm_motion["displacement"].unsqueeze(0),
            norm_motion["disp_8"].unsqueeze(0),
            norm_motion["tags"]["sin_diff"].unsqueeze(0),
            norm_motion["tags"]["cos_diff"].unsqueeze(0),  
            norm_motion["loss_weights"].unsqueeze(0),
        )


def run_generator(model):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    model.eval()
    with torch.no_grad():
        res_decoder = model.forward()
    return res_decoder


def run_ik(model, res_decoder, frame=None):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    model.eval()
    with torch.no_grad():
        res = model.forward(res_decoder, frame)
    return res

def upsample_rots_via_matrices(rots_6d, scale_factor=8):
    """
    Upsample 6D rotation representation by converting to rotation matrices,
    interpolating each matrix element along time, re-orthonormalizing via SVD,
    and converting back to ortho6d.

    Accepts numpy arrays of shape (T,6) or (B,T,6) and returns same batch shape
    with time dimension multiplied by scale_factor.
    """
    import numpy as _np
    from scipy.interpolate import interp1d as _interp1d

    if rots_6d is None:
        return rots_6d

    single = False
    if rots_6d.ndim == 2:
        # (T,6) -> (1,T,6)
        rots_6d = rots_6d[_np.newaxis, ...]
        single = True

    B, T, C = rots_6d.shape
    assert C == 6, "expected ortho6d 6D input"
    out_T = T * scale_factor
    t_orig = _np.arange(T)
    t_new = _np.linspace(0, T - 1, out_T)

    out = _np.zeros((B, out_T, 6), dtype=rots_6d.dtype)
    rots_6d = rots_6d.reshape(B,T,3,2)
    for b in range(B):
        # convert to matrices (T,3,3)
        mats = ortho6d.to_matrix(rots_6d[b])
        # interpolate each matrix element over time
        mats_interp = _np.zeros((out_T, 3, 3), dtype=mats.dtype)
        for i in range(3):
            for j in range(3):
                f = _interp1d(t_orig, mats[:, i, j], kind="linear", fill_value="extrapolate")
                mats_interp[:, i, j] = f(t_new)

        # re-orthonormalize each interpolated matrix and convert back to ortho6d
        for t in range(out_T):
            M = mats_interp[t]
            # SVD-based orthonormalization
            U, _, Vt = _np.linalg.svd(M)
            R = U.dot(Vt)
            # ensure right-handed
            if _np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U.dot(Vt)
            out[b, t] = ortho6d.from_matrix(R).reshape(6)  # assume ortho6d.from_matrix accepts (3,3) -> (6,)
    if single:
        return out[0]
    return out

def rotate_root_numpy(root, phi):
    """
    Rotate `root` (ortho6d 6D + 3D position) by yaw angle phi (radians) around Y axis.

    Args:
        root: np.ndarray shape (T,9) or (B,T,9). layout [...,:6]=ortho6d, [...,6:9]=position.
        phi: scalar angle in radians (or broadcastable array for each frame).
    Returns:
        rotated: same shape as root
    """
    arr = np.asarray(root)
    single = False
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # -> (1,T,9)
        single = True
    B, T, C = arr.shape
    assert C == 9

    N = B * T
    rot6_flat = arr[..., :6].reshape(N, 3, 2)   # (N,3,2)
    pos_flat = arr[..., 6:9].reshape(N, 3)     # (N,3)

    # to rotation matrices (N,3,3)
    mats = ortho6d.to_matrix(rot6_flat)

    # build yaw rotation matrix R (3x3) and broadcast to (N,3,3)
    c = np.cos(phi)
    s = np.sin(phi)
    R = np.array([[c, 0.0, s],
                  [0.0, 1.0, 0.0],
                  [-s, 0.0, c]], dtype=mats.dtype)
    Rb = np.repeat(R[np.newaxis, ...], N, axis=0)

    # left-multiply: new = R @ old
    new_mats = np.einsum("bij,bjk->bik", Rb, mats)

    # back to ortho6d 6-vector and rotate positions
    new_rot6 = ortho6d.from_matrix(new_mats).reshape(B, T, 6)
    new_pos = (Rb @ pos_flat[..., None]).squeeze(-1).reshape(B, T, 3)

    out = np.concatenate([new_rot6, new_pos], axis=-1)  # (B,T,9)
    return out[0] if single else out

def result_to_bvh(res, means, stds, bvh, filename, save=True, initial_frame = None, feet_idx = None, copy_init_frame = False,
                  disp_8 = None, yaw_diff=None, initial_sin_cos=None, initial_ortho=None):
    
    res = res.permute(0, 2, 1)
    res = res.flatten(0, 1)
    res = res.cpu().detach().numpy()
    frames = res.shape[0]
    pos = res[:,6:10]
    #rots_6d = res[:frames//8, :6].copy()
    root = res[:, :9].copy()
    root = root * stds["rots"].cpu().numpy() + means["rots"].cpu().numpy()

    rots_6d = root[:,:6]
    root_pos = root[:,6:9]

    #if(initial_ortho is not None):
    #    rots_6d[0] = initial_ortho[0].copy()
    #rots_6d = ortho6d.compute_cumulative(rots_6d)
    # rots_6d = upsample_rots_via_matrices(rots_6d, scale_factor=8)

    # get dqs and displacement
    dqs = res
    dqs = dqs * stds["dqs"].cpu().numpy() + means["dqs"].cpu().numpy()
    # dqs[:,:9] = rotate_root_numpy(dqs[:,:9], np.pi/2)
    dqs = dqs.reshape(dqs.shape[0], dqs.shape[1]//9, 9) # frames, n_joints, 9
    # dqs[:,0,:6] = rots_6d.copy() #
    # dqs[:,0,6:9] = root_pos.copy() #
    # denormalize
    pred_positions = np.copy(dqs[:,0,6:9])
    dqs = ortho6d.to_dual_quat(dqs)
    
    # get rotations and translations from dual quatenions
    dqs = dqs.reshape(dqs.shape[0], -1, 8)
    #dqs = dquat.unroll(dqs, axis=0)
    #dqs = dquat.canonicalize_(dqs)


    # import matplotlib.pyplot as plt
    # plt.plot(pred_positions)
    # plt.show()
    import matplotlib.pyplot as plt
    
    if not ROOT_CHANNELS_ARE_GLOBAL_POSITIONS and not USE_CANONICAL_XZ_POSITIONS:
        pred_positions[np.abs(pred_positions) < 0.005] = 0
        pred_positions[0, [0, 2]] = bvh.data["positions"][1, 0, [0, 2]] * 0
        pred_positions[:, 0] = np.cumsum(pred_positions[:, 0], axis=0)
        pred_positions[:, 2] = np.cumsum(pred_positions[:, 2], axis=0)
    
    from scipy.ndimage import gaussian_filter1d
    def plot_trajectory(ax, positions, n=None, smooth_sigma=3, cmap='viridis', step_arrow=40):
        """
        positions: (F, J, 3) or (F, 3) - will use root joint at index 0 if 3D-per-joint
        """
        if positions.ndim == 3:
            root = positions[:, 0, :]  # (F,3)
        else:
            root = positions
        if n is None:
            n = root.shape[0]
        root = root[:n]

        # smooth X,Z optionally
        x = gaussian_filter1d(root[:, 0], sigma=smooth_sigma)
        z = gaussian_filter1d(root[:, 2], sigma=smooth_sigma)

        # line + color by time
        t = np.arange(len(x))
        points = np.stack([x, z], axis=1)
        ax.plot(x, z, color='0.2', lw=1.5, alpha=0.6)
        sc = ax.scatter(x, z, c=t, cmap=cmap, s=12, lw=0, alpha=0.9)
        cbar = plt.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label('frame')

        # start / end markers
        ax.scatter(x[0], z[0], color='green', s=80, marker='*', label='start')
        ax.scatter(x[-1], z[-1], color='red', s=60, marker='X', label='end')

        # # arrows showing motion direction (subsample)
        # if len(x) > step_arrow:
        #     diffs = np.stack([np.diff(x), np.diff(z)], axis=1)
        #     norms = np.linalg.norm(diffs, axis=1, keepdims=True).clip(min=1e-6)
        #     dirs = diffs / norms
        #     idxs = np.arange(0, len(dirs), step_arrow)
        #     ax.quiver(x[idxs], z[idxs], dirs[idxs,0], dirs[idxs,1],
        #               angles='xy', scale_units='xy', scale=0.5, color='tab:blue', width=0.0008, alpha=0.9)

        ax.set_xlabel('X (world)')
        ax.set_ylabel('Z (world)')
        ax.set_title('Root trajectory (XZ plane)')
        # ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_aspect('equal', adjustable='box')
        return ax

    # fig, ax = plt.subplots(figsize=(6,6))
    # plot_trajectory(ax, pred_positions, n=1800, smooth_sigma=1, step_arrow=64)
    # plt.tight_layout()
    # plt.show()
    # from scipy.ndimage import gaussian_filter1d
    # pred_positions[:,1] = gaussian_filter1d(pred_positions[:,1],sigma=10)
    
    # pred_positions[:,2] = np.cumsum(pred_positions[:,2],axis=0)
    # pred_positions = pred_positions + bvh.data["positions"][0,0,:3]#######################
   

    zeros = np.zeros((pos.shape[0], pos.shape[1]))
    #trans[0,0] = bvh.data["positions"][0,0,:3]
    #pos = np.concatenate([pos, zeros], axis=-1) # because displacement has 4 items (x,y,z increments and actual y)
    pos = pos * stds["displacement"].cpu().numpy() + means["displacement"].cpu().numpy()
    pos[0,:3] = bvh.data["positions"][0,0,:3]
    #pos = np.cumsum(pos, axis=0)
    #pos[:,0] = np.cumsum(pos[:,0],axis=0)
    #pos[:,2] = np.cumsum(pos[:,2],axis=0)
    pos = pred_positions
    trans, rots = from_root_dual_quat(dqs, np.array(bvh.data["parents"]))
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    # plt.plot(rots[:,0])
    # plt.show()
    trans, _ = from_root_dual_quat_to_root(dqs, np.array(bvh.data["parents"]))
    _, trans = to_rotation_translation(dqs)
    #trans = translation_each_joint(rots, pos[:,:-1], bvh.data["parents"], bvh.data["offsets"])
    #trans = bvh.compute_global_pos()
    # plot trans 
    diff = np.diff(trans[:,[8,12,15,18]], axis=0)
    velo = np.linalg.norm(diff[:], axis=-1)


    if(disp_8 is not None and False):
        
        step = 8
        disp_8 = disp_8.flatten(0, 1)
        disp_8 = disp_8.cpu().detach().numpy()
        disp_8 = disp_8 * stds["disp_8"].cpu().numpy() + means["disp_8"].cpu().numpy()
        disp_8[0] = bvh.data["positions"][0,0,:3]
        pos_8 = np.cumsum(disp_8, axis=0)
        
        # Original timestamps (one sample every 8 frames)
        M = pos_8.shape[0]
        t_original = np.arange(0, M * step, step)  # [0, 8, 16, ..., (M-1)*8]

        # New timestamps: one sample per frame
        t_new = np.arange(0, M * step)  # [0, 1, 2, ..., (M*8 - 1)]

        # Interpolate for each axis
        pos_full = np.zeros((M * step, 3))
        for j in range(3):
            interp_func = interp1d(t_original, pos_8[:, j], kind='linear', fill_value="extrapolate")
            pos_full[:, j] = interp_func(t_new)
            pos = pos_full

    #print(bvh.data["positions"][:,0])
    #phase = compute_phase(pos)

    
    # if(initial_frame is not None):
    #     min_len=min(rots.shape[0], initial_frame.shape[0])
    #     rots[:min_len,0,:] = initial_frame[:min_len,:]
        #rots[0,0,:] = initial_frame
        #rots[:,0,:] = initial_frame[:rots.shape[0], :]
    #initial_rotation = quat.from_euler(bvh.data["rotations"][0,0,:], rot_roder[0,0,:])
    #rots[0,0,:] = initial_rotation
    if(copy_init_frame):
        min_len=min(rots.shape[0], initial_frame.shape[0])
        rots[:min_len,0,:] = initial_frame[:min_len, :]
        #rots[:,0,0] = yaw + rots[0,0,0]
        #rots[0,0,:] = initial_frame[0, :]
    
    #rots = quat.compute_cumulative_quaternions(rots, feet_idx)
    # quaternions to euler

    rot_roder = np.tile(bvh.data["rot_order"][0], (rots.shape[0], rots.shape[1], 1))
    #rotations = np.degrees(quat.to_euler(rots, order=rot_roder))
    rotations = bvh.to_degrees(quat.to_euler(rots, order=rot_roder))
    rot_roder = np.tile(bvh.data["rot_order"][0], (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1))  # made a change here
    
    # og_quats = quat.unroll(quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_roder),axis=0)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 1, 1)
    # plt.plot(rots[:,0])
    # plt.subplot(2, 1, 2)
    # plt.plot(og_quats[:,0])
    # plt.show()
    if(yaw_diff is not None and False):
        rotations[:,0,0] = np.degrees(yaw) #+ rotations[0,0,0]
    bvh.data["rotations"] = rotations

    # positions
        
    from scipy.interpolate import UnivariateSpline

    # Define smoothing factor (adjust as needed, higher = smoother)
    smoothing_factor = 1
    offset_distance = 1.0  # 1 meter in front of the character

    positions = bvh.data["positions"][: rotations.shape[0]]
    num_frames = positions.shape[0]

    # Get the original root positions - we'll need these for calculations
    original_root_pos = positions[:, 0, :].copy()

    # Calculate the offset points for each frame (1 meter in front of character)
    # Get y-rotation angles (assuming these represent the character's facing direction)
    y_rotations = rotations[:, 0, 0]  # Y-axis rotation for root joint
    y_rotations_rad = np.radians(y_rotations[:positions.shape[0]])  # Convert to radians
 
    """# Calculate offset positions in XZ plane (1 meter in front based on orientation)
    offset_x = original_root_pos[:, 0] + offset_distance * np.sin(y_rotations_rad)
    offset_z = original_root_pos[:, 2] + offset_distance * np.cos(y_rotations_rad)
    offset_y = original_root_pos[:, 1].copy()  # Y position unchanged

    # Now sample and smooth these offset positions
    sample_indices = np.append(np.arange(0, num_frames, 32), num_frames-1)

    # For X dimension
    samples_x = offset_x[sample_indices]
    spline_x = UnivariateSpline(sample_indices, samples_x, s=smoothing_factor)
    smooth_offset_x = spline_x(np.arange(num_frames))

    # For Z dimension
    samples_z = offset_z[sample_indices]
    spline_z = UnivariateSpline(sample_indices, samples_z, s=smoothing_factor)
    smooth_offset_z = spline_z(np.arange(num_frames))

    # For Y dimension (height)
    samples_y = offset_y[sample_indices]
    spline_y = UnivariateSpline(sample_indices, samples_y, s=smoothing_factor)
    smooth_offset_y = spline_y(np.arange(num_frames))"""

    # Apply the smoothed offset positions
    #positions[:, 0, 0] = smooth_offset_x
    #positions[:, 0, 1] = smooth_offset_y
    #positions[:, 0, 2] = smooth_offset_z
 
    #positions[:,0] = pos[:positions.shape[0],:positions.shape[2]]
    # from scipy.ndimage import gaussian_filter1d
    # for i in range (positions.shape[2]):
    #     positions[:,0,i] = gaussian_filter1d(positions[:,0,i],sigma=2)
    og_positions = np.copy(bvh.data["positions"])
    bvh.data["positions"] = positions
    bvh.data["positions"][:,0] = pos[:positions.shape[0]]
    
    # copy og
    # bvh.data["positions"] = og_positions[:rotations.shape[0]]
    # save
    bvh.data["parents"][0] = None  # BVH sets root as None
    path = None
       
    
    if save:
        path = "data"
        filename = "eval_" + filename
        bvh.save(os.path.join(path, filename))
    return path, filename


# from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d

# def compute_phase(pos):
#     # Assuming x_values, z_values, and y_values are 1D arrays representing the movement over time
#     x_values = pos[:,0,0]  # X-axis movement data
#     z_values = pos[:,0,2]  # Z-axis movement data
#     y_values = pos[:,0,1]  # Y-axis movement data
#     epsilon = 0.01  # Stationary threshold

#     # Step 2: Compute xz plane velocity
#     vx = np.diff(x_values, prepend=x_values[0])
#     vz = np.diff(z_values, prepend=z_values[0])
#     velocity_xz = np.sqrt(vx**2 + vz**2)

#     # Step 3: Discard low velocities
#     velocity_threshold = 0.01  # Set your threshold here
#     active_indices = np.where(velocity_xz > velocity_threshold)[0]
#     inactive_indices = np.where(velocity_xz < velocity_threshold)[0]
#     # Step 4: Extract active y values
#     active_y_values = y_values[active_indices]

#     # Step 5: Smooth the active y values for better peak detection
#     smoothed_y = gaussian_filter1d(y_values, sigma=5)
    
#     # Step 7: Locate peaks and valleys
#     peaks, _ = find_peaks(smoothed_y,)
#     valleys, _ = find_peaks(-smoothed_y,)

#     starting_point = min(peaks[0],valleys[0])
    
#     def construct_signal(peaks, valleys, num_points):
#         """
#         """
#         # Initialize the signal array
#         signal = np.zeros(num_points)
#         # Ensure peaks and valleys are sorted
#         points = np.concatenate((peaks,valleys))
#         points = sorted(points)
#         # Iterate through the points to construct the signal
#         for i in range(len(points) - 1):
#             start = points[i]
#             end = points[i + 1]
#             x = np.linspace(0, 1, end - start)  # Normalize to 0 to 1
#             if points[i] in peaks:
#                 # If current point is a peak
#                 signal[start:end] = 0.5 * (1 + np.cos(np.pi * x))  # Descend to valley (sine wave)
#             elif points[i] in valleys:
#                 # If current point is a valley
#                 signal[start:end] = 0.5 * (1 - np.cos(np.pi * x))  # Ascend to peak (sine wave)
        
#         return signal

    
#     num_points = len(smoothed_y)
#     phase = construct_signal(peaks, valleys, num_points)
#     return phase
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Motion Upsampling Network")
    parser.add_argument(
        "data_path",
        type=str,
        help="path to data directory containing one or multiple .bvh for training, last .bvh is used as test data",
    )
    parser.add_argument(
        "name",
        type=str,
        help="name of the experiment, used to save the model and the logs",
    )
    parser.add_argument(
        "train_mode",
        type=str.lower,
        choices=["generator", "ik", "all"],
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="load the model(s) from a checkpoint",
    )
    args = parser.parse_args()
    if args.train_mode == "generator":
        args.train_mode = GENERATOR
    elif args.train_mode == "ik":
        args.train_mode = IK
    elif args.train_mode == "all":
        args.train_mode = GENERATOR | IK
    main(args)