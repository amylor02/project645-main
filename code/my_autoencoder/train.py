import random
import torch
from torch.utils.data.dataloader import DataLoader
from motion_data import TestMotionData, TrainMotionData
import pymotion.rotations.quat as quat
from pymotion.ops.skeleton import from_root_dual_quat, translation_each_joint
from pymotion.io.bvh import BVH
import numpy as np
from train_data import Train_Data
from generator_architecture import Generator_Model
from ik_architecture import IK_Model
import time
import argparse
import os
import eval_metrics
scale = 1
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

# Train Modes
GENERATOR = 1
IK = 2

human_param = {
    "batch_size": 256,
    "epochs": 1000,
    "kernel_size_temporal_dim": 15,
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
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1 
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
    "batch_size": 256,
    "epochs": 4000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 0,
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
    "window_size": 512,
    "window_step": 32,
    "seed": 2222,
    "extra_joint": -1,
    "feet_idx": [
        8,
        12,
        15,
        18,
    ]
}

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

param = dog_param
rm_flag= False

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
    # Train Files
    for filename in train_files:
        if filename[-4:] == ".bvh":
            rots, pos, parents, offsets, _ = get_info_from_bvh(
                get_bvh_from_disk(train_dir, filename)
            )
            if reference_parents is None:
                reference_parents = parents.copy()
            assert (
                reference_parents == parents
            )  # make sure all bvh have the same structure
            # Train Dataset7
            pos_all_joints = translation_each_joint(rots, pos[:,0,:], parents, offsets)
            train_dataset.add_motion(
                offsets,
                pos[:, 0, :],  # only global position
                rots,
                parents,
                pos_all_joints,
            )
    # Once all train files are added, compute the means and stds and normalize
    train_dataset.normalize()
    eval_dataset.set_means_stds(train_dataset.means, train_dataset.stds)
    # Eval Files
    for filename in eval_files:
        if filename[-4:] == ".bvh":
            rots, pos, parents, offsets, bvh = get_info_from_bvh(
                get_bvh_from_disk(eval_dir, filename)
            )
            assert (
                reference_parents == parents
            )  # make sure all bvh have the same structure
            # Eval Dataset
            pos_all_joints = translation_each_joint(rots, pos[:,0,:], parents, offsets)
            eval_dataset.add_motion(
                offsets,
                pos[:, 0, :],  # only global position
                rots,
                parents,
                bvh,
                filename,
                pos_all_joints,
            )
    # Once all eval files are added, normalize
    eval_dataset.normalize()

    train_dataloader = DataLoader(train_dataset, param["batch_size"], shuffle=True)
    print(len(train_dataset),len(train_dataloader))
    # Create Models
    train_data = Train_Data(device, param)
    generator_model = Generator_Model(device, param, reference_parents, train_data).to(
        device
    )
    if args.train_mode & IK != 0:
        ik_model = IK_Model(device, param, reference_parents, train_data).to(device)
    train_data.set_means(train_dataset.means["dqs"])
    train_data.set_stds(train_dataset.stds["dqs"])

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
        results = evaluate_generator(generator_model, train_data, eval_dataset)
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
        )
        best_evaluation = mpjpe + mpeepe
    else:
        best_evaluation = float("inf")
    # Training Loop
        start_time = time.time()
     
    for epoch in range(param["epochs"]):
        avg_train_loss = 0.0
        avg_feet_loss = 0.0
        avg_energy_loss = 0.0
        for step, (denorm_motion, norm_motion) in enumerate(train_dataloader):
            # Forward
            train_data.set_offsets(norm_motion["offsets"], denorm_motion["offsets"])
            train_data.set_motions(
                norm_motion["dqs"],
                norm_motion["displacement"],
            )

            train_data.set_tags(denorm_motion["tags"])
            train_data.set_energy(denorm_motion["energy_feet"].clone().detach().to(device, dtype=torch.float32))
            train_data.set_global_pos(denorm_motion["global_pos"])
            if args.train_mode & GENERATOR != 0:
                generator_model.train()
            if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                res_decoder = generator_model.forward()
            if args.train_mode & IK != 0:
                ik_model.train()
                ik_model.forward(res_decoder)
            # Loss
            loss = 0.0
            recon_feet_loss = 0.0
            energy_loss = 0.0
            if args.train_mode & GENERATOR != 0:
                loss_generator, recon_feet_loss, energy_loss = generator_model.optimize_parameters()
                loss = loss_generator.item()
                recon_feet_loss = recon_feet_loss.item()
                energy_loss = energy_loss.item()
            if args.train_mode & IK != 0:
                loss_ik = ik_model.optimize_parameters(res_decoder)
                loss += loss_ik.item()
            avg_train_loss += loss
            avg_feet_loss += recon_feet_loss
            avg_energy_loss += energy_loss
            # Evaluate & Print
            if step == len(train_dataloader) - 1:
                if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                    results = evaluate_generator(
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
                    mpjpe, mpeepe = eval_save_result(
                        results,
                        train_dataset.means,
                        train_dataset.stds,
                        eval_dir,
                        device,
                        save=False,
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
                avg_feet_loss /= len(train_dataloader)
                avg_energy_loss /= len(train_dataloader)
                if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                    print(
                        "Epoch: {} - Train Loss: {:.4f} - Eval Loss: {:.4f} - MPJPE: {:.4f} - MPEEPE: {:.4f} - F: {:.4f} - E: {:.4f}".format(
                            epoch, avg_train_loss, evaluation_loss, mpjpe, mpeepe, avg_feet_loss, avg_energy_loss
                        )
                            + ("*" if was_best else "")
                    )

    end_time = time.time()
    print("Training Time:", end_time - start_time)

    # Load Best Model -> Save and/or Evaluate
    if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
        load_model(generator_model, generator_path, train_data, device)
        results = evaluate_generator(generator_model, train_data, eval_dataset)
        if args.train_mode & IK != 0:
            load_model(ik_model, ik_path, train_data, device)
            results_ik = evaluate_ik(ik_model, results, train_data, eval_dataset)
            results = results_ik

        mpjpe, mpeepe = eval_save_result(
            results, train_dataset.means, train_dataset.stds, eval_dir, device
        )
        evaluation_loss = mpjpe + mpeepe

    print("Evaluate Loss: {}".format(evaluation_loss))
    if args.train_mode & (GENERATOR | IK) != 0:
        print("Mean Per Joint Position Error: {}".format(mpjpe))
        print("Mean End Effector Position Error: {}".format(mpeepe))


def eval_save_result(results, train_means, train_stds, eval_dir, device, save=True):
    # Save Result
    array_mpjpe = np.empty((len(results),))
    array_mpeepe = np.empty((len(results),))
    for step, (res, bvh, filename) in enumerate(results):
        if save:
            eval_path, eval_filename = result_to_bvh(
                res, train_means, train_stds, bvh, filename
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
        if ignore_transform_net:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.transform_net.')}
            
        #state_dict = {k: v for k, v in state_dict.items() if (not 'autoregressive' in k)}
            
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

initial = None
def get_info_from_bvh(bvh, incr_rots = True):
    rot_roder = np.tile(bvh.data["rot_order"][0], (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1))  # made a change here
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_roder),
        axis=0,
    )
    rots = quat.normalize(rots)  # make sure all quaternions are unit quaternions
    if(incr_rots and False):
        rots = quat.compute_incremental_quaternions(rots)    
    # rots = rox
    # print(rots[1])
    # print("------")
    # global initial 
    # initial = rox
    pos = bvh.data["positions"]
    parents = bvh.data["parents"]
    parents[0] = 0  # BVH sets root as None
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)  # force to zero offset for root joint
    return rots, pos, parents, offsets, bvh


def evaluate_generator(generator_model, train_data, dataset, sparse_motions=None):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    generator_model.eval()
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
            )

            tags_tensor_dict = {
            key: torch.tensor(value, dtype=torch.float32).to('cuda').unsqueeze(0)
            for key, value in norm_motion["tags"].items()
            }
            train_data.set_tags(
               tags_tensor_dict
            )
            train_data.set_global_pos(
                norm_motion["global_pos"]
            )
            train_data.set_energy(
               torch.tensor(norm_motion["energy_feet"],dtype=torch.float32).to('cuda').unsqueeze(0)
            )
            if sparse_motions is not None:
                train_data.set_sparse_motion(sparse_motions[index])
            res = generator_model.forward()
            bvh, filename = dataset.get_bvh(index)
            results.append((res, bvh, filename))
    return results


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


def result_to_bvh(res, means, stds, bvh, filename, save=True, initial_frame = None):
 
    res = res.permute(0, 2, 1)
    res = res.flatten(0, 1)
    displacements = res[:,4:7].clone()
    res = res.cpu().detach().numpy()
    displacements = displacements.cpu().detach().numpy()
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(displacements[:, 0], label='Displacement X')
    # plt.plot(displacements[:, 1], label='Displacement Y')
    # plt.plot(displacements[:, 2], label='Displacement Z')
    # plt.xlabel('Frame')
    # plt.ylabel('Displacement')
    # plt.title('Displacements (X, Y, Z)')
    # plt.legend()
    # plt.show()
    displacements[np.abs(displacements) < 0.5] = 0

    # get dqs and displacement
    dqs = res
    # denormalize
    dqs = dqs * stds["dqs"].cpu().numpy() + means["dqs"].cpu().numpy()
    # get rotations and translations from dual quatenions
    dqs = dqs.reshape(dqs.shape[0], -1, 8)
    pos, rots = from_root_dual_quat(dqs, np.array(bvh.data["parents"]))
   
    pos[:,0] = displacements
    pos = pos / 100
    # global initial
    # rots = initial
    bvh_positions = bvh.data["positions"][0,0]
    pos[0,0] = bvh_positions
    
    pos = np.cumsum(pos, axis=0)
    
    if(initial_frame is not None):
        min_len=min(rots.shape[0], initial_frame.shape[0])
        rots[:min_len,0,:] = initial_frame[:min_len,:] 
        if(min_len == 1 and False):
            rots = quat.compute_cumulative_quaternions(rots)
    
    #rots = quat.compute_cumulative_quaternions(rots)
  
    # quaternions to euler

    rot_roder = np.tile(bvh.data["rot_order"][0], (rots.shape[0], rots.shape[1], 1))
    rotations = bvh.to_degrees(quat.to_euler(rots, order=rot_roder))
    positions = bvh.data["positions"][: rotations.shape[0]]
    # if(initial_frame is not None):
    #     initial_rots = initial_frame
    #     min_len=min(initial_rots.shape[0], rotations.shape[0])
    #     print(initial_rots.shape, rot_roder.shape)
    #     rot_roder = np.tile(bvh.data["rot_order"][0], (initial_rots.shape[0], initial_rots.shape[1], 1))
    #     initial_rots = bvh.to_degrees(quat.to_euler(initial_rots, order=rot_roder))
    #     rotations[:min_len,0,0] = initial_rots[:min_len,0,0]

    # print(pos[:10,0])
    # print(positions[:10,0])

    bvh.data["rotations"] = rotations
    

     
    bvh.data["positions"] = positions
    # save
    bvh.data["parents"][0] = None  # BVH sets root as None
    path = None
    if save:
        path = "data"
        filename = "eval_" + filename
        bvh.save(os.path.join(path, filename))
    return path, filename


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
