import os
import numpy as np
import torch
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import translation_each_joint
import pymotion.rotations.quat as quat
from motion_data import compute_tags
import train_vq_vae
from generator_architecture import Generator_Model
from train_data import Train_Data
from motion_data import TestMotionData
import pymotion.rotations.ortho6d_torch as ortho6d
from pymotion.ops.skeleton import to_root_dual_quat

def load_bvh_and_compute_tags(bvh_path, param=None, is_human=False):
    """
    Load a BVH file and compute motion tags.
    
    Args:
        bvh_path: str, path to BVH file
        param: dict, parameters for skeleton processing (optional)
        is_human: bool, whether skeleton is human or not
    
    Returns:
        tags: dict, computed motion tags
        bvh: BVH object
        pos_all_joints: np.ndarray, global joint positions
    """
    if not os.path.exists(bvh_path):
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")
    
    # Load BVH file
    bvh = BVH()
    bvh.load(bvh_path)
    
    # Extract rotations and positions
    rot_order = np.tile(bvh.data["rot_order"][0], (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1))
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_order),
        axis=0,
    )
    rots = quat.normalize(rots)
    
    
    pos = bvh.data["positions"]
    parents = bvh.data["parents"]
    parents[0] = 0  # BVH sets root as None
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)  # force to zero offset for root joint
    
    # Compute global joint positions
    pos_all_joints = translation_each_joint(rots, pos[:, 0, :], parents, offsets)
    
    # Set default parameters if not provided
    if param is None:
        param = {
            "head_idx": 4 if is_human else 2,
            "head_height": 1.0,
            "skeleton_height": 0.90,
            "feet_idxs": [15, 18] if is_human else [8, 12, 15, 18],
            "not_dog": is_human,
            "feet_contact_threshold": 0.02
        }
    
    # Convert rotations to euler for tags computation
    rots_euler = bvh.data["rotations"][:, 0]  # root y rotation
    
    # Compute motion tags
    tags = compute_tags(
        pos_all_joints,
        head_idx=param["head_idx"],
        head_height=param["head_height"],
        downsample_factor=8,
        is_human=is_human,
        skeleton_height=param["skeleton_height"],
        rots=np.deg2rad(rots_euler),
        is_deg=False,
        scalar=1.0,
        feet_idx=param["feet_idxs"],
        quats=rots,
        padded_frames=0,
        window_size=param.get("window_size", 128),
        shoulder_idx=[5, 9] if is_human else [5, 9],
        not_dog=param["not_dog"],
        feet_contact_threshold=param["feet_contact_threshold"],
        sparse_joints=param.get("sparse_joints"),
    )
    
    return tags, bvh, pos_all_joints


# def compute_motion_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
#     """
#     Compute FID between real and generated motion feature distributions.
    
#     Args:
#         real_features: [N_real, feature_dim] - features extracted from real motions
#         generated_features: [N_gen, feature_dim] - features from generated motions
    
#     Returns:
#         fid_score: float
#     """
#     from scipy import linalg
    
#     # Compute mean and covariance
#     mu_real = np.mean(real_features, axis=0)
#     mu_gen = np.mean(generated_features, axis=0)
    
#     sigma_real = np.cov(real_features, rowvar=False)
#     sigma_gen = np.cov(generated_features, rowvar=False)
    
#     # Compute squared difference of means
#     diff = mu_real - mu_gen
    
#     # Compute sqrt of product of covariances
#     covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)
    
#     # Handle numerical errors
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
    
#     # FID formula
#     fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2*covmean)
    
#     return float(fid)

def compute_motion_fid(real_features: np.ndarray, generated_features: np.ndarray, eps=1e-6) -> float:
    from scipy import linalg

    mu_real = np.mean(real_features, axis=0)
    mu_gen  = np.mean(generated_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen  = np.cov(generated_features, rowvar=False)

    # Add eps for numerical stability
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_gen  += np.eye(sigma_gen.shape[0]) * eps

    diff = mu_real - mu_gen

    # Compute sqrt of product
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)

    # Handle numerical issues
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_gen + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fid)


def compute_fid_between_datasets(real_dir, generated_dir, model_path, param, device='cuda'):
    """
    Compute FID between two datasets (real and generated) using the latent space 
    of a pre-trained VQ-VAE generator model.
    
    Args:
        real_dir: str, directory containing real BVH files
        generated_dir: str, directory containing generated BVH files
        model_path: str, path to the folder containing 'generator.pt'
        param: dict, parameters for the model/data
        device: str, 'cuda' or 'cpu'
        
    Returns:
        fid: float, the computed FID score
    """
    print(f"Computing FID between:\n  Real: {real_dir}\n  Gen:  {generated_dir}")
    
    # 1. Initialize Model and Data structures
    # We need parents to initialize the Generator_Model. Load from first real file.
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.bvh')]
    if not real_files:
        raise FileNotFoundError(f"No BVH files found in {real_dir}")
    
    # Load info from first file to get parents
    # Using train_vq_vae helper as in load_latents.py
    first_file = real_files[0]
    fname = os.path.basename(first_file)
    dname = os.path.dirname(first_file)
    
    # get_info_from_bvh returns: rots, pos, parents, offsets, bvh, og_rots (ignored)
    _, _, parents, _, _, _ = train_vq_vae.get_info_from_bvh(
        train_vq_vae.get_bvh_from_disk(dname, fname, remove=False), 
        incremental_rots=False, 
        get_missing_frames=False
    )
    
    # Initialize Train_Data and Generator_Model
    train_data = Train_Data(device, param)
    generator_model = Generator_Model(device, param, parents, train_data, is_vq_vae=True).to(device)
    generator_model.eval()
    
    # Load model weights
    gen_pt_path = os.path.join(model_path, "generator.pt")
    if not os.path.exists(gen_pt_path):
        raise FileNotFoundError(f"Generator model not found at {gen_pt_path}")
        
    means, stds = train_vq_vae.load_model(generator_model, gen_pt_path, train_data, device)
    
    # 2. Helper function to extract latents from a directory
    def get_latents_from_dir(directory):
        # Setup dataset
        dataset = TestMotionData(param, train_vq_vae.scale, device)
        dataset.set_means_stds(means, stds)
        
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.bvh')]
        print(f"Processing {len(files)} files in {directory}...")
        
        for f in files:
            fn = os.path.basename(f)
            dn = os.path.dirname(f)
            
            # Load BVH data
            rots, pos, parents_local, offsets, bvh, _ = train_vq_vae.get_info_from_bvh(
                train_vq_vae.get_bvh_from_disk(dn, fn, remove=False), 
                incremental_rots=False, 
                get_missing_frames=False
            )
            
            # Preprocessing for VAE (incremental quats)
            rots = quat.compute_incremental_quaternions(rots)
            
            # Compute global positions for normalization/tags
            pos_all_joints = translation_each_joint(rots, pos[:,0,:], parents_local, offsets)
            
            dataset.add_motion(offsets, pos[:,0,:], rots, parents_local, bvh, f, pos_all_joints)
            
        # Normalize all added motions
        dataset.normalize()
        
        latents_list = []
        
        with torch.no_grad():
            for i in range(dataset.get_len()):
                norm_motion = dataset.get_item(i)
                
                # Populate train_data for the model
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
                    norm_motion["loss_weights"],
                )
                train_data.set_phase(
                    torch.tensor(norm_motion["phase"], dtype=torch.float32).to(device).unsqueeze(0)
                )
                train_data.set_phase_per_8_frames(
                    torch.tensor(norm_motion["phase_per_8_frames"], dtype=torch.float32).to(device).unsqueeze(0)
                )
                train_data.set_velocity_per_8_frames(
                    torch.tensor(norm_motion["velocity_per_8_frames"], dtype=torch.float32).to(device).unsqueeze(0)
                )   
                train_data.set_tags(norm_motion["tags"])
                
                # Run Encoder
                # We assume the first return value is the latent representation
                encoder_out = generator_model.autoencoder.encoder(train_data.sparse_motion)
                
                if isinstance(encoder_out, tuple):
                    latent = encoder_out[0]
                else:
                    latent = encoder_out
                    
                print(latent.min(), latent.max())
                
                # Reshape latent: [Batch, Channels, Frames] -> [Frames, Channels]
                # Assuming batch size is 1
                if latent.dim() == 3:
                    latent = latent.permute(0, 2, 1).squeeze(0) # [Frames, Channels]
                
                latents_list.append(latent.cpu().numpy())
                
        if len(latents_list) > 0:
            return np.concatenate(latents_list, axis=0)
        else:
            return np.array([])

    # 3. Extract features and compute FID
    real_latents = get_latents_from_dir(real_dir)
    gen_latents = get_latents_from_dir(generated_dir)
    
    real_min = np.min(real_latents, axis=0, keepdims=True)  # [1, feature_dim]
    real_max = np.max(real_latents, axis=0, keepdims=True)  # [1, feature_dim]
    
    # Avoid division by zero for constant features
    real_range = real_max - real_min
    real_range = np.clip(real_range, 1e-8, None)
    
    # Normalize BOTH real and generated to [0, 1] using real statistics
    # real_latents = (real_latents - real_min) / real_range
    # gen_latents = (gen_latents - real_min) / real_range
    
    
    print(real_latents.mean(), real_latents.max())
    if len(real_latents) == 0 or len(gen_latents) == 0:
        print("Error: Could not extract latents from one or both directories.")
        return float('nan')
        
    print(f"Real latents shape: {real_latents.shape}")
    print(f"Gen latents shape:  {gen_latents.shape}")
    
    fid_score = compute_motion_fid(real_latents, gen_latents)
    print(f"Computed FID: {fid_score}")
    
    return fid_score


def compare_motion_tags(bvh_path1, bvh_path2, param1=None, param2=None, is_human1=False, is_human2=False):
    """
    Load two BVH files and compute tags for comparison.
    
    Args:
        bvh_path1: str, path to first BVH file
        bvh_path2: str, path to second BVH file  
        param1: dict, parameters for first skeleton
        param2: dict, parameters for second skeleton
        is_human1: bool, whether first skeleton is human
        is_human2: bool, whether second skeleton is human
    
    Returns:
        tags1: dict, tags for first motion
        tags2: dict, tags for second motion
        bvh1: BVH object for first file
        bvh2: BVH object for second file
    """
    print(f"Loading BVH file 1: {bvh_path1}")
    tags1, bvh1, pos1 = load_bvh_and_compute_tags(bvh_path1, param1, is_human1)
    
    print(f"Loading BVH file 2: {bvh_path2}")
    tags2, bvh2, pos2 = load_bvh_and_compute_tags(bvh_path2, param2, is_human2)
    
    print(f"Motion 1: {pos1.shape[0]} frames, {pos1.shape[1]} joints")
    print(f"Motion 2: {pos2.shape[0]} frames, {pos2.shape[1]} joints")
    
    # Print available tag keys for both motions
    print(f"Tags in motion 1: {list(tags1.keys())}")
    print(f"Tags in motion 2: {list(tags2.keys())}")
    
    return tags1, tags2, bvh1, bvh2


def compute_step_timing_difference(binary_labels1, binary_labels2):
    """
    For each step start (0→1 transition) in binary_labels1, find the closest 
    step start in binary_labels2 and measure the frame distance.
    
    Args:
        binary_labels1: np.ndarray shape (F1, num_feet) - binary foot contact labels for motion 1
        binary_labels2: np.ndarray shape (F2, num_feet) - binary foot contact labels for motion 2
    
    Returns:
        dict with:
            - 'avg_distance': float, average frame distance across all feet and all steps
            - 'per_foot_distances': list of float, average distance per foot
            - 'all_distances': list of int, all individual frame distances
            - 'step_starts_1': list of lists, detected step starts per foot in motion 1
            - 'step_starts_2': list of lists, detected step starts per foot in motion 2
    """
    num_feet = binary_labels1.shape[1]
    all_distances = []
    per_foot_distances = []
    step_starts_1_all = []
    step_starts_2_all = []
    for foot_idx in range(num_feet):
        # Detect step starts (0→1 transitions) for this foot
        foot1 = binary_labels1[:, foot_idx]
        foot2 = binary_labels2[:, foot_idx]
        
        # Find transitions from 0 to 1
        step_starts_1 = []
        for i in range(1, len(foot1)):
            if foot1[i] == 1 and foot1[i-1] == 0:
                step_starts_1.append(i)
        
        step_starts_2 = []
        for i in range(1, len(foot2)):
            if foot2[i] == 1 and foot2[i-1] == 0:
                step_starts_2.append(i)
        
        step_starts_1_all.append(step_starts_1)
        step_starts_2_all.append(step_starts_2)
        
        # For each step start in motion 1, find closest in motion 2
        foot_distances = []
        for start1 in step_starts_1:
            if len(step_starts_2) == 0:
                continue
            
            # Find closest step start in motion 2
            distances_to_all = [abs(start1 - start2) for start2 in step_starts_2]
            min_distance = min(distances_to_all)
            foot_distances.append(min_distance)
            all_distances.append(min_distance)
        
        if len(foot_distances) > 0:
            per_foot_distances.append(np.mean(foot_distances))
        else:
            per_foot_distances.append(np.nan)
        
        print(f"Foot {foot_idx}: {len(step_starts_1)} steps in motion 1, {len(step_starts_2)} steps in motion 2")
    
    if len(all_distances) > 0:
        avg_distance = np.mean(all_distances)
        std = np.std(all_distances)
    else:
        avg_distance = np.nan
    
    return {
        'avg_distance': avg_distance,
        'std': std,
        'per_foot_distances': per_foot_distances,
        'all_distances': all_distances,
        'step_starts_1': step_starts_1_all,
        'step_starts_2': step_starts_2_all,
        'num_steps_1': sum(len(s) for s in step_starts_1_all),
        'num_steps_2': sum(len(s) for s in step_starts_2_all)
    }


def compute_step_timing_difference_radians(binary_labels1, binary_labels2, fps=30):
    """
    For each step start (0→1 transition) in binary_labels1, find the closest 
    step start in binary_labels2 and measure the phase difference in radians.
    
    This computes a per-foot gait phase and measures timing differences as angular offsets.
    
    Args:
        binary_labels1: np.ndarray shape (F1, num_feet) - binary foot contact labels for motion 1
        binary_labels2: np.ndarray shape (F2, num_feet) - binary foot contact labels for motion 2
        fps: float, frames per second (default 30)
    
    Returns:
        dict with:
            - 'avg_phase_diff_rad': float, average phase difference in radians
            - 'avg_phase_diff_deg': float, average phase difference in degrees
            - 'std_phase_diff_rad': float, std deviation in radians
            - 'per_foot_phase_diffs': list of float, average phase diff per foot (radians)
            - 'all_phase_diffs': list of float, all individual phase differences (radians)
            - 'avg_frame_distance': float, average frame distance
            - 'step_periods_1': list, average step period per foot in motion 1 (frames)
            - 'step_periods_2': list, average step period per foot in motion 2 (frames)
    """
    num_feet = binary_labels1.shape[1]
    all_phase_diffs = []
    per_foot_phase_diffs = []
    all_frame_distances = []
    step_periods_1 = []
    step_periods_2 = []
    
    for foot_idx in range(num_feet):
        # Detect step starts (0→1 transitions) for this foot
        foot1 = binary_labels1[:, foot_idx]
        foot2 = binary_labels2[:, foot_idx]
        
        # Find transitions from 0 to 1
        step_starts_1 = []
        for i in range(1, len(foot1)):
            if foot1[i] == 1 and foot1[i-1] == 0:
                step_starts_1.append(i)
        
        step_starts_2 = []
        for i in range(1, len(foot2)):
            if foot2[i] == 1 and foot2[i-1] == 0:
                step_starts_2.append(i)
        
        if len(step_starts_1) < 2 or len(step_starts_2) < 2:
            print(f"Foot {foot_idx}: Insufficient steps for phase analysis")
            per_foot_phase_diffs.append(np.nan)
            step_periods_1.append(np.nan)
            step_periods_2.append(np.nan)
            continue
        
        # Calculate average step period (stride cycle) for each motion
        periods_1 = np.diff(step_starts_1)
        periods_2 = np.diff(step_starts_2)
        avg_period_1 = np.mean(periods_1)
        avg_period_2 = np.mean(periods_2)
        
        step_periods_1.append(avg_period_1)
        step_periods_2.append(avg_period_2)
        
        # For each step in motion 1, find closest in motion 2 and compute phase difference
        foot_phase_diffs = []
        for start1 in step_starts_1:
            if len(step_starts_2) == 0:
                continue
            
            # Find closest step start in motion 2
            distances_to_all = [abs(start1 - start2) for start2 in step_starts_2]
            min_distance = min(distances_to_all)
            closest_idx = np.argmin(distances_to_all)
            closest_start2 = step_starts_2[closest_idx]
            
            all_frame_distances.append(min_distance)
            
            # Convert frame distance to phase difference (radians)
            # Use the average of both periods to normalize
            avg_period = (avg_period_1 + avg_period_2) / 2.0
            
            # Phase difference = (frame_distance / period) * 2π
            # Normalize to [-π, π]
            phase_diff = (min_distance / avg_period) * (2 * np.pi)
            
            # Wrap to [-π, π]
            phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
            
            foot_phase_diffs.append(abs(phase_diff))  # Take absolute value
            all_phase_diffs.append(abs(phase_diff))
        
        if len(foot_phase_diffs) > 0:
            per_foot_phase_diffs.append(np.mean(foot_phase_diffs))
        else:
            per_foot_phase_diffs.append(np.nan)
        
        print(f"Foot {foot_idx}: {len(step_starts_1)} steps in motion 1, "
              f"{len(step_starts_2)} steps in motion 2, "
              f"avg period: {avg_period_1:.1f} frames (motion 1), {avg_period_2:.1f} frames (motion 2)")
    
    if len(all_phase_diffs) > 0:
        avg_phase_diff_rad = np.mean(all_phase_diffs)
        std_phase_diff_rad = np.std(all_phase_diffs)
        avg_phase_diff_deg = np.degrees(avg_phase_diff_rad)
    else:
        avg_phase_diff_rad = np.nan
        std_phase_diff_rad = np.nan
        avg_phase_diff_deg = np.nan
    
    avg_frame_distance = np.mean(all_frame_distances) if len(all_frame_distances) > 0 else np.nan
    
    return {
        'avg_phase_diff_rad': avg_phase_diff_rad,
        'avg_phase_diff_deg': avg_phase_diff_deg,
        'std_phase_diff_rad': std_phase_diff_rad,
        'std_phase_diff_deg': np.degrees(std_phase_diff_rad) if not np.isnan(std_phase_diff_rad) else np.nan,
        'per_foot_phase_diffs': per_foot_phase_diffs,
        'all_phase_diffs': all_phase_diffs,
        'avg_frame_distance': avg_frame_distance,
        'step_periods_1': step_periods_1,
        'step_periods_2': step_periods_2,
        'num_steps_1': sum(len([i for i in range(1, len(binary_labels1[:, f])) 
                                 if binary_labels1[i, f] == 1 and binary_labels1[i-1, f] == 0]) 
                          for f in range(num_feet)),
        'num_steps_2': sum(len([i for i in range(1, len(binary_labels2[:, f])) 
                                 if binary_labels2[i, f] == 1 and binary_labels2[i-1, f] == 0]) 
                          for f in range(num_feet))
    }



def timing(bvh_path1, bvh_path2, param1, param2):
    try:
        # Load and compute tags for both motions
        tags1, tags2, bvh1, bvh2 = compare_motion_tags(
            bvh_path1, bvh_path2,
            param1, param2,
            is_human1=True, is_human2=False
        )
        
        # Example: compare velocity tags
        if "velocity" in tags1 and "velocity" in tags2:
            print(f"Motion 1 velocity range: {np.min(tags1['velocity'])} - {np.max(tags1['velocity'])}")
            print(f"Motion 2 velocity range: {np.min(tags2['velocity'])} - {np.max(tags2['velocity'])}")
        
        results = compute_step_timing_difference(tags1["binary_foot_labels"][:,0:1], tags2["binary_foot_labels"][:,0:1])
        print(f"\nTotal steps in motion 1: {results['num_steps_1']}")
        print(f"Total steps in motion 2: {results['num_steps_2']}")
        print(f"\nAverage frame distance between step starts: {results['avg_distance']:.2f} frames, std: {results['std']:.2f} ")
        print(results['per_foot_distances'])
        
        print("===================================================")
        
        results_phase = compute_step_timing_difference_radians(
            tags1["binary_foot_labels"][:, 0:1], 
            tags2["binary_foot_labels"][:, 0:1]
        )
        print(f"\nAverage phase difference: {results_phase['avg_phase_diff_rad']:.3f} rad "
              f"({results_phase['avg_phase_diff_deg']:.1f}°)")
        print(f"Std deviation: {results_phase['std_phase_diff_rad']:.3f} rad "
              f"({results_phase['std_phase_diff_deg']:.1f}°)")
        
        
        import matplotlib.pyplot as plt
        print(tags1["binary_foot_labels"].shape)
        plt.plot(tags1['binary_foot_labels'][:,0])
        plt.plot(tags2['binary_foot_labels'][:,0])
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update the bvh_path1 and bvh_path2 variables with valid file paths")
        
        
def compute_jitter_from_bvh(bvh_path, param=None, is_human=False, fps=30.0):
    """
    Compute jitter (squared jerk) for a single BVH file.
    
    Jitter = 10^2 * m/s^3 (squared magnitude of jerk)
    Jerk = third derivative of position (m/s^3)
    
    Args:
        bvh_path: str, path to BVH file
        param: dict, parameters for skeleton processing
        is_human: bool, whether skeleton is human
        fps: float, frames per second (default 30.0)
    
    Returns:
        dict with:
            - 'avg_jitter': float, average jitter across all joints and all frames
            - 'per_joint_jitter': np.ndarray, average jitter per joint
            - 'jerk_magnitude': np.ndarray, jerk magnitude over time [frames, joints]
            - 'jitter_over_time': np.ndarray, jitter over time [frames, joints]
    """
    if not os.path.exists(bvh_path):
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")
    
    # Load BVH file
    bvh = BVH()
    bvh.load(bvh_path)
    
    # Extract rotations and positions
    rot_order = np.tile(bvh.data["rot_order"][0], 
                       (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1))
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_order),
        axis=0,
    )
    rots = quat.normalize(rots)
    
    pos = bvh.data["positions"]
    parents = bvh.data["parents"]
    parents[0] = 0
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)
    
    # Compute global joint positions: [frames, joints, 3]
    pos_all_joints = translation_each_joint(rots, pos[:, 0, :], parents, offsets)
    
    num_frames = pos_all_joints.shape[0]
    num_joints = pos_all_joints.shape[1]
    
    # Time step between frames
    dt = 1.0 / fps
    
    # First derivative: velocity [frames-1, joints, 3]
    velocity = np.diff(pos_all_joints, axis=0) / dt
    
    # Second derivative: acceleration [frames-2, joints, 3]
    acceleration = np.diff(velocity, axis=0) / dt
    
    # Third derivative: jerk [frames-3, joints, 3]
    jerk = np.diff(acceleration, axis=0) / dt
    
    # Compute jerk magnitude per joint per frame: [frames-3, joints]
    jerk_magnitude = np.linalg.norm(jerk, axis=2)
    
    # Compute jitter: squared jerk magnitude * 10^2
    # Jitter = (jerk_magnitude)^2 * 100
    # jitter_over_time = (jerk_magnitude ** 2) * 100.0
    jitter_over_time = jerk_magnitude / 100
    
    # Average jitter per joint across all frames
    per_joint_jitter = np.mean(jitter_over_time, axis=0)
    
    # Average jitter across all joints and all frames
    avg_jitter = np.mean(jitter_over_time)
    
    return {
        'avg_jitter': avg_jitter,
        'per_joint_jitter': per_joint_jitter,
        'jerk_magnitude': jerk_magnitude,
        'jitter_over_time': jitter_over_time,
        'num_frames': num_frames,
        'num_joints': num_joints
    }
    

def compute_jitter_for_directory(directory_path, param=None, is_human=False, fps=30.0, 
                                 visualize=False, save_results=False):
    """
    Compute average jitter for all BVH files in a directory.
    
    Args:
        directory_path: str, path to directory containing BVH files
        param: dict, parameters for skeleton processing
        is_human: bool, whether skeletons are human
        fps: float, frames per second
        visualize: bool, whether to create visualization plots
        save_results: bool, whether to save results to CSV
    
    Returns:
        dict with:
            - 'overall_avg_jitter': float, average jitter across all files
            - 'per_file_jitter': dict, {filename: jitter_value}
            - 'per_file_results': dict, {filename: full_results_dict}
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Find all BVH files
    bvh_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".bvh"):
                bvh_files.append(os.path.join(root, file))
    
    if not bvh_files:
        print(f"No BVH files found in {directory_path}")
        return None
    
    print(f"Found {len(bvh_files)} BVH files")
    print("Computing jitter for each file...\n")
    
    per_file_jitter = {}
    per_file_results = {}
    all_jitter_values = []
    
    for i, bvh_path in enumerate(bvh_files):
        filename = os.path.basename(bvh_path)
        try:
            print(f"[{i+1}/{len(bvh_files)}] Processing: {filename}")
            
            results = compute_jitter_from_bvh(bvh_path, param, is_human, fps)
            
            per_file_jitter[filename] = results['avg_jitter']
            per_file_results[filename] = results
            all_jitter_values.append(results['avg_jitter'])
            
            print(f"  Avg jitter: {results['avg_jitter']:.4f} (10^2 m/s^3)")
            print(f"  Frames: {results['num_frames']}, Joints: {results['num_joints']}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    if not all_jitter_values:
        print("No files processed successfully")
        return None
    
    # Compute overall statistics
    overall_avg_jitter = np.mean(all_jitter_values)
    overall_std_jitter = np.std(all_jitter_values)
    overall_min_jitter = np.min(all_jitter_values)
    overall_max_jitter = np.max(all_jitter_values)
    
    print("\n" + "="*60)
    print("JITTER ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(all_jitter_values)}")
    print(f"Overall average jitter: {overall_avg_jitter:.4f} (10^2 m/s^3)")
    print(f"Standard deviation: {overall_std_jitter:.4f}")
    print(f"Min jitter: {overall_min_jitter:.4f} (file: {min(per_file_jitter, key=per_file_jitter.get)})")
    print(f"Max jitter: {overall_max_jitter:.4f} (file: {max(per_file_jitter, key=per_file_jitter.get)})")
    print("="*60)
    
    # Save results to CSV
    if save_results:
        output_csv = os.path.join(directory_path, "jitter_analysis.csv")
        with open(output_csv, 'w') as f:
            f.write("filename,avg_jitter,num_frames,num_joints\n")
            for filename, jitter in per_file_jitter.items():
                results = per_file_results[filename]
                f.write(f"{filename},{jitter:.6f},{results['num_frames']},{results['num_joints']}\n")
        print(f"\nResults saved to: {output_csv}")
    
    # Visualization
    if visualize:
        import matplotlib.pyplot as plt
        
        # Plot 1: Histogram of jitter values
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(all_jitter_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.axvline(overall_avg_jitter, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {overall_avg_jitter:.4f}')
        plt.xlabel('Jitter (10^2 m/s^3)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Average Jitter Across Files', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Sorted bar chart of jitter per file (top 20)
        plt.subplot(2, 2, 2)
        sorted_files = sorted(per_file_jitter.items(), key=lambda x: x[1], reverse=True)[:20]
        filenames = [os.path.splitext(f[0])[0][:20] for f in sorted_files]  # Truncate names
        jitter_vals = [f[1] for f in sorted_files]
        
        plt.barh(range(len(filenames)), jitter_vals, color='coral', edgecolor='black')
        plt.yticks(range(len(filenames)), filenames, fontsize=8)
        plt.xlabel('Jitter (10^2 m/s^3)', fontsize=12)
        plt.title('Top 20 Files by Jitter', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Per-joint jitter for a sample file (use file with median jitter)
        plt.subplot(2, 2, 3)
        median_idx = np.argsort(all_jitter_values)[len(all_jitter_values)//2]
        sample_filename = list(per_file_jitter.keys())[median_idx]
        sample_results = per_file_results[sample_filename]
        
        joint_indices = np.arange(sample_results['num_joints'])
        plt.bar(joint_indices, sample_results['per_joint_jitter'], 
               color='lightgreen', edgecolor='black', alpha=0.7)
        plt.xlabel('Joint Index', fontsize=12)
        plt.ylabel('Jitter (10^2 m/s^3)', fontsize=12)
        plt.title(f'Per-Joint Jitter (Sample: {os.path.splitext(sample_filename)[0]})', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Jitter over time for sample file
        plt.subplot(2, 2, 4)
        jitter_time = sample_results['jitter_over_time']
        avg_jitter_per_frame = np.mean(jitter_time, axis=1)
        
        plt.plot(avg_jitter_per_frame, linewidth=1.5, color='purple', alpha=0.7)
        plt.axhline(sample_results['avg_jitter'], color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {sample_results['avg_jitter']:.4f}")
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('Jitter (10^2 m/s^3)', fontsize=12)
        plt.title(f'Jitter Over Time (Sample: {os.path.splitext(sample_filename)[0]})', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_fig = os.path.join(directory_path, "jitter_analysis.png")
        plt.savefig(output_fig, dpi=150)
        print(f"Visualization saved to: {output_fig}")
        
        plt.show()
    
    return {
        'overall_avg_jitter': overall_avg_jitter,
        'overall_std_jitter': overall_std_jitter,
        'overall_min_jitter': overall_min_jitter,
        'overall_max_jitter': overall_max_jitter,
        'per_file_jitter': per_file_jitter,
        'per_file_results': per_file_results
    }
        

def compute_foot_skating_from_bvh(bvh_path, param=None, is_human=False, fps=30.0, 
                                  height_thresh=0.2, vel_thresh=0.02): # 5cm height, 2cm/frame velocity
    """
    Compute foot skating metrics for a single BVH file.
    
    Skating is defined as: Foot is close to ground (height < height_thresh) 
    BUT moving fast (velocity > vel_thresh).
    
    Args:
        bvh_path: str, path to BVH file
        param: dict, parameters containing 'feet_idxs'
        is_human: bool, whether skeleton is human
        fps: float, frames per second
        height_thresh: float, max height to be considered "on ground" (meters)
        vel_thresh: float, max velocity to be considered "planted" (meters/frame)
    
    Returns:
        dict with:
            - 'skating_ratio': float, % of frames where skating occurs
            - 'avg_skating_dist': float, average distance slid per skating frame
            - 'per_foot_skating': list, skating ratio per foot
    """
    if not os.path.exists(bvh_path):
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")
    
    # Load BVH
    bvh = BVH()
    bvh.load(bvh_path)
    
    # Extract rotations and positions
    rot_order = np.tile(bvh.data["rot_order"][0], 
                       (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1))
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_order),
        axis=0,
    )
    rots = quat.normalize(rots)
    
    pos = bvh.data["positions"]
    parents = bvh.data["parents"]
    parents[0] = 0
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)
    
    # Compute global joint positions: [frames, joints, 3]
    pos_all_joints = translation_each_joint(rots, pos[:, 0, :], parents, offsets)
    
    # Get feet indices
    if param is None:
        feet_idx = [15, 18] if is_human else [8, 12, 15, 18]
    else:
        feet_idx = param["feet_idxs"][:2]
        
    if(is_human):
        pos_all_joints = pos_all_joints[3:]
    total_skating_frames = 0
    total_frames_analyzed = 0
    total_skating_dist = 0.0
    per_foot_skating = []
    
    root_velo_xz = np.linalg.norm(np.diff(pos_all_joints[:,0,[0,2]], axis=0, prepend=pos_all_joints[0:1,0,[0,2]]), axis=1)
    dist = np.sum(root_velo_xz)
    print("Total root distance:", dist)
    
    # Calculate ground height (assume min y of all feet over all time is ground)
    # Or assume y=0 is ground. Let's use y=0 as standard for normalized data.
    ground_y = 0.0
    import matplotlib.pyplot as plt
    print(pos_all_joints[:,feet_idx].shape)
    plt.plot(pos_all_joints[:,feet_idx][:,:,1])
    plt.show()
    for foot_id in feet_idx:
        foot_pos = pos_all_joints[:, foot_id, :] # [frames, 3]
        
        # 1. Calculate Height
        foot_height = foot_pos[:, 1] - ground_y
        
        # 2. Calculate Velocity (displacement per frame)
        # We only care about horizontal sliding (XZ plane)
        foot_vel_xz = np.linalg.norm(np.diff(foot_pos[:, [0, 2]], axis=0, prepend=foot_pos[0:1, [0, 2]]), axis=1)
        # 3. Detect Skating
        # Condition: Foot is on ground AND Foot is moving
        is_contact = foot_height < height_thresh
        is_moving = foot_vel_xz > vel_thresh
        
        is_skating = np.logical_and(is_contact, is_moving)
        
        skating_frames = np.sum(is_skating)
        num_frames = len(foot_pos)
        plt.plot(foot_vel_xz)
        # Calculate metrics
        ratio = skating_frames / num_frames if num_frames > 0 else 0
        per_foot_skating.append(ratio)
        
        # Accumulate distance slid during skating frames
        skating_dist = np.sum(foot_vel_xz[is_skating])
        
        total_skating_frames += skating_frames
        total_frames_analyzed += num_frames
        total_skating_dist += skating_dist
    plt.show()
    avg_skating_ratio = total_skating_frames / total_frames_analyzed if total_frames_analyzed > 0 else 0
    avg_skating_dist = total_skating_dist / total_skating_frames if total_skating_frames > 0 else 0
    
    return {
        'skating_ratio': avg_skating_ratio, # Percentage of time feet slide
        'avg_skating_dist': avg_skating_dist, # Avg distance slid per slide event
        'per_foot_skating': per_foot_skating,
        'total_skating_dist': total_skating_dist
    }
    

def main():
    """
    Example usage - modify these paths to your BVH files
    """
    # Parameters for different skeleton types
    human_param = train_vq_vae.human_param
    dog_param = train_vq_vae.dog_param
    ostrich_param = train_vq_vae.ostrich_param
    
    timing = False
    jitter = False
    skating = False
    fid = True
    
    if(timing):
        bvh_path1 = "quant/phase/with/long_walk_2.bvh"
        bvh_path2 = "quant/phase/without/dog.bvh"
        
  
        
        timing(bvh_path1, bvh_path2 ,ostrich_param, dog_param)
    
    if(jitter):
        dir='./quant/jitter/ostrich'
        param=ostrich_param
        compute_jitter_for_directory(dir,param)
    
    if(skating):
        # # for dog
        # bvh_path_skating = "quant/skating/dog_dataset.bvh"
        # param = dog_param
        # skating = compute_foot_skating_from_bvh(bvh_path_skating, param, is_human=False,height_thresh=0.15, vel_thresh=0.02)
        
        # # for human
        # bvh_path_skating = "quant/skating/human.bvh"
        # param = human_param
        # skating = compute_foot_skating_from_bvh(bvh_path_skating, param, is_human=True,height_thresh=0.1, vel_thresh=0.012)
        
        # for ostrich
        bvh_path_skating = "quant/skating/ostrich.bvh"
        param = ostrich_param
        skating = compute_foot_skating_from_bvh(bvh_path_skating, param, is_human=False,height_thresh=0.1, vel_thresh=0.02)
    
        print(skating["skating_ratio"])
        print(skating["avg_skating_dist"])
        print(skating["total_skating_dist"])
    
    ############ FID ###############
    if(fid):
        # real_dataset = "./dog_data_ds_aligned/train/"
        # gen_dataset = "./dog_data_ds_aligned/train/"
        real_dataset = "./quant/fid/ostrich/real_ostrich"
        # gen_dataset = "./quant/fid/real_dog/"
        gen_dataset = "./quant/fid/ostrich/gen_ostrich_autoreg"
        model = "./models/model_test_ostrich_3/best_root/"
        compute_fid_between_datasets(real_dir=real_dataset, generated_dir=gen_dataset, model_path=model, param=ostrich_param)
        
if __name__ == "__main__":
    main()