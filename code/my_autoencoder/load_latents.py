import os
import torch
import numpy as np
# from autoencoder import Autoencoder
from generator_architecture import Generator_Model
from train_data import Train_Data
from sklearn.cluster import KMeans
from motion_data import TestMotionData
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import train_vq_vae
import train
from pymotion.ops.skeleton import translation_each_joint
import pymotion.rotations.quat as quat
#from sklearn_extra.cluster import KMedoids

IS_VAE = True

bvh_files = []

if(IS_VAE):
    eval_dataset = TestMotionData(train_vq_vae.param, train_vq_vae.scale, 'cuda')
else:
    eval_dataset = TestMotionData(train_vq_vae.param, train_vq_vae.scale, 'cuda')

initial_frame = None
def load_bvh_files(database_path):

    for root, _, files in os.walk(database_path):
        for file in files:
            if file.endswith(".bvh"):
                file_path = os.path.join(root, file)
                bvh_files.append(file_path)
    return bvh_files

def process_bvh_file(file_path):
    global initial_frame

    filename = os.path.basename(file_path)
    dir = file_path[: -len(filename)]

    if(IS_VAE):
        rots, pos, parents, offsets, bvh,_ = train_vq_vae.get_info_from_bvh(
            train_vq_vae.get_bvh_from_disk(dir, filename), incremental_rots=False, get_missing_frames=False
        )
    else:
        rots, pos, parents, offsets, bvh,_ = train.get_info_from_bvh(
            train.get_bvh_from_disk(dir, filename)
        )

    if(IS_VAE):
        if(initial_frame is None):
            initial_frame = rots[:,0,:]
        rots = quat.compute_incremental_quaternions(rots)

    return offsets, pos, rots, parents, bvh

def process_database(generator_model, database_path, device, parents, means, stds, bvh_files = None):
    # Initialize a list to store latent vectors
    latent_vectors_list = []
    sampled_vectors_list = []
    # Load all BVH files from the database directory
    if bvh_files is None:
        bvh_files = load_bvh_files(database_path)

    # Create an eval dataset
    eval_dataset.set_means_stds(means, stds)
    filename=[ ]
    bvhs= []
    # Iterate over all BVH files

    for file_path in bvh_files:
        print(f"Processing file: {file_path}")
       
        filename.append(os.path.basename(file_path))
        # Load and preprocess the BVH file
        offsets, positions, rotations, parents, bvh = process_bvh_file(file_path)
        bvhs.append(bvh)
        # Add motion to the eval dataset
        pos_all_joints = translation_each_joint(rotations, positions[:,0,:], parents, offsets)
        eval_dataset.add_motion(
            offsets,
            positions[:, 0, :],  # only global position
            rotations,
            parents,
            bvh,
            file_path,
            pos_all_joints
        )

    # Normalize the eval dataset
    eval_dataset.normalize()

    # with torch.no_grad():
    #     results = train_vq_vae.evaluate_generator(generator_model, train_data, eval_dataset)
    # Process each motion in the eval dataset
    num_codebook_vectors = generator_model.autoencoder.encoder.num_embeddings
    print("Calculating all latent vectors...")
    all_indices = []
    for index in range(eval_dataset.get_len()):
        norm_motion = eval_dataset.get_item(index)
        train_data.set_offsets(
            norm_motion["offsets"].unsqueeze(0),
            norm_motion["denorm_offsets"].unsqueeze(0),
        )
        train_data.set_motions(
            norm_motion["dqs"].unsqueeze(0),
            norm_motion["displacement"].unsqueeze(0),
            # norm_motion["disp_8"].unsqueeze(0),
            # norm_motion["tags"]["sin_diff"].unsqueeze(0),
            # norm_motion["tags"]["cos_diff"].unsqueeze(0), 
            # norm_motion["loss_weights"],
        )
        # train_data.set_phase(
        #     torch.tensor(norm_motion["phase"], dtype=torch.float32).to(device).unsqueeze(0)
        # )
        # train_data.set_phase_per_8_frames(
        #     torch.tensor(norm_motion["phase_per_8_frames"], dtype=torch.float32).to(device).unsqueeze(0)
        #     )
        # train_data.set_velocity_per_8_frames(
        #     torch.tensor(norm_motion["velocity_per_8_frames"], dtype=torch.float32).to(device).unsqueeze(0)
        #     )   
        # tags_tensor_dict = {
        #     key: torch.tensor(value, dtype=torch.float32).to('cuda').unsqueeze(0)
        #     for key, value in norm_motion["tags"].items()
        #     }
        train_data.set_tags(
            norm_motion["tags"]#tags_tensor_dict
        )

        # train_vq_vae.result_to_bvh(
        # results[index][0], means, stds, bvhs[index], filename[index], save=True, initial_frame=None, feet_idx= None
        # )
        # Pass the data through the encoder
        with torch.no_grad():
            if(IS_VAE):
                _ ,embedding_indices, _, embedding_indices_all = generator_model.autoencoder.encoder(
                    train_data.sparse_motion,
                )

                all_indices.append(embedding_indices_all)

            else:
                latent_vectors, recon_feet = generator_model.autoencoder.encoder(train_data.sparse_motion, tags=train_data.tags)
                # tags = train_data.tags
                # tags = torch.cat([tags["velocity"], tags["acceleration"], tags["ang_velocity"], tags["height"]], dim=-1)
                tags = generator_model.autoencoder.encoder.tags_vector
                print(train_data.sparse_motion.shape)
            
            # find unique indices
    
    
    all_indices_cat = torch.cat(all_indices,dim=1)
    unique_indices_c0 = torch.unique(all_indices_cat[:,:,0])
   # unique_indices_c1 = torch.unique(all_indices_cat[:,:,1])
    #unique_indices_c2 = torch.unique(all_indices_cat[:,:,2])
    
    print(unique_indices_c0.shape[0],"/",num_codebook_vectors," or ",unique_indices_c0.shape[0]/num_codebook_vectors * 100, "%")
    #print(unique_indices_c1.shape[0],"/",num_codebook_vectors," or ",unique_indices_c1.shape[0]/num_codebook_vectors * 100, "%")
    #print(unique_indices_c2.shape[0],"/",num_codebook_vectors," or ",unique_indices_c2.shape[0]/num_codebook_vectors * 100, "%")
    B_all, T_all, L_all = all_indices_cat.shape  # [B, total_frames, num_levels]
    total_tokens = (B_all * T_all)
    print(f"Collected indices shape: {all_indices_cat.shape}, codebook size: {num_codebook_vectors}")

    usage_stats = {}
    # per-level stats
    for lvl in range(L_all):
        idxs = all_indices_cat[:, :, lvl].reshape(-1).cpu()
        counts = torch.bincount(idxs, minlength=num_codebook_vectors).float()
        probs = counts / counts.sum().clamp_min(1.0)
        nonzero = int((probs > 0).sum().item())
        # entropy over non-zero probs
        nz_probs = probs[probs > 0]
        entropy = -(nz_probs * nz_probs.log()).sum().item() if nz_probs.numel() > 0 else 0.0
        perplexity = float(np.exp(entropy))
        usage_stats[f"level_{lvl}"] = {
            "nonzero": nonzero,
            "nonzero_frac": nonzero / num_codebook_vectors,
            "perplexity": perplexity,
        }
        print(f"Level {lvl}: used {nonzero}/{num_codebook_vectors} ({nonzero/num_codebook_vectors*100:.2f}%), perplexity={perplexity:.2f}")

    # overall across all levels combined
    flat_all = all_indices_cat.reshape(-1).cpu()
    counts_all = torch.bincount(flat_all, minlength=num_codebook_vectors).float()
    probs_all = counts_all / counts_all.sum().clamp_min(1.0)
    nz_all = int((probs_all > 0).sum().item())
    nz_probs_all = probs_all[probs_all > 0]
    entropy_all = -(nz_probs_all * nz_probs_all.log()).sum().item() if nz_probs_all.numel() > 0 else 0.0
    perplexity_all = float(np.exp(entropy_all))
    print(f"Overall: used {nz_all}/{num_codebook_vectors} ({nz_all/num_codebook_vectors*100:.2f}%), perplexity={perplexity_all:.2f}")
   
   
    k = min(20, counts_all.numel())
    # topk with largest=False gives smallest counts
    smallest_vals, smallest_idxs = torch.topk(counts_all, k=k, largest=False)
    smallest_vals = smallest_vals.long().cpu().numpy()
    smallest_idxs = smallest_idxs.cpu().numpy()
    print("Lowest hit-count codebook entries (index : count):")
    for idx, val in zip(smallest_idxs, smallest_vals):
        print(f"  {int(idx):4d} : {int(val):6d}")
   

def load_parents(database_path):
    bvh_files = load_bvh_files(database_path)
    if not bvh_files:
        raise FileNotFoundError("No BVH files found in the database directory.")
    _, _, _, parents, _ = process_bvh_file(bvh_files[0])
    return parents, bvh_files

def process_custom_latent(custom_latent):
    ae_offsets = generator_model.static_encoder(train_data.offsets)
    custom_latent = torch.tensor(custom_latent).unsqueeze(0).to(device)
    output = generator_model.autoencoder.decoder(custom_latent.permute(0,2,1), ae_offsets, train_data.mean_dqs, train_data.std_dqs, train_data.denorm_offsets, parents)
    bvh,filename = eval_dataset.get_bvh(0)
    results = [(output,bvh,filename)]
    if(IS_VAE):
        train_vq_vae.result_to_bvh(
            results[0][0], means, stds, bvh, os.path.basename(filename), save=True, initial_frame=initial_frame, feet_idx= None
            )
    else:
        train.result_to_bvh(
            results[0][0], means, stds, bvh, os.path.basename(filename), save=True
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process database and extract latent vectors")
    parser.add_argument("model_path", type=str, help="Path to the generator model")
    parser.add_argument("database_path", type=str, help="Path to the database directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")

    args = parser.parse_args()

    # Load the generator model
    device = torch.device(args.device)

    if(IS_VAE):
        param = train_vq_vae.param  # Replace with actual parameters
    else:
        param = train.param

    # Load parents from the first BVH file in the database
    parents, bvh_files = load_parents(args.database_path)

    train_data = Train_Data(device, param)
    generator_model = Generator_Model(device, param, parents, train_data, is_vq_vae=IS_VAE).to(device)
    generator_model.eval()
    # Load the model state
    generator_model_path = os.path.join(args.model_path, "generator.pt")
    # checkpoint = torch.load(generator_model_path, map_location=device)
    # generator_model.load_state_dict(checkpoint["model_state_dict"])
    
    if(IS_VAE):
        means, stds = train_vq_vae.load_model(
            generator_model, generator_model_path, train_data, device
        )
    else:
        means, stds = train.load_model(
            generator_model, generator_model_path, train_data, device
        )

    # Process the database and extract latent vectors
    process_database(generator_model, args.database_path, device, parents, means, stds, bvh_files=bvh_files)

    """concatenated_latent_space, latent_vectors, concatenated_sampled_space = process_database(generator_model, args.database_path, device, parents, means, stds, bvh_files=bvh_files)
    concatenated_latent_space = np.array(concatenated_latent_space.permute(0,2,1).squeeze(0).cpu())
    concatenated_sampled_space = np.array(concatenated_sampled_space.permute(0,2,1).squeeze(0).cpu())

     # Bind each vector in the first item of latent_vectors to the closest cluster center
    first_latent_vector = latent_vectors[0]  # [batch, channels, frames]
    batch_size, channels, frames = first_latent_vector.shape
    
    first_latent_vector = first_latent_vector.permute(0,2,1).squeeze(0).cpu().numpy()  # [batch * frames, channels]
    #print(first_latent_vector)
    load_codebook = False
    if(load_codebook == False):
        #####
        import cupy as cp
        from cupy.linalg import norm
        print("computing random indices")
        num_samples = 2048
        num_frames = concatenated_latent_space.shape[0]
        num_samples = min(num_samples, num_frames)
        random_indices = np.random.choice(num_frames, num_samples, replace=False)
        print("gathering samples")
        # Extract the randomly sampled frames
        sampled_latent_space = concatenated_latent_space[random_indices]
        #data_gpu = cp.array(sampled_latent_space)

        # Compute pairwise distances on the GPU
        #print("computing distance matrix")
        #distance_matrix_gpu = cp.linalg.norm(data_gpu[:, None] - data_gpu, axis=2)
        #distance_matrix = cp.asnumpy(distance_matrix_gpu)
        #####
        #print("computing kmedoids")
        #num_clusters = 100
        # Initialize KMedoids with precomputed distances
        #kmeans = KMedoids(n_clusters=num_clusters, random_state=0, method='pam', metric='precomputed')
        #kmeans.fit(distance_matrix)

        print("--clusters--")
        #print(np.array(kmeans.cluster_centers_).shape)

    
        # Compute distances to cluster centers
        #medoid_indices = kmeans.medoid_indices_
        #cluster_centers = concatenated_latent_space[medoid_indices]
        #cluster_centers = sampled_latent_space
        cluster_centers = concatenated_sampled_space

        np.save("cluster_centers_test.npy",cluster_centers)

    else:
        cluster_centers = np.load("cluster_centers_test.npy")

    distances = np.linalg.norm(first_latent_vector[:, np.newaxis] - cluster_centers, axis=2)
    # Find the closest cluster center for each frame vector
    closest_clusters = np.argmin(distances, axis=1)
    #print(closest_clusters)
    print(closest_clusters)
    replaced_latent_vector = cluster_centers[closest_clusters]
    print(replaced_latent_vector.shape)
    # Print the closest clusters for the first latent vector
    #print("Closest clusters for the first latent vector:")
    #print(closest_clusters)

    # Print the shape of the concatenated latent space
    print(f"Concatenated latent space shape: {concatenated_latent_space.shape}")


    # Fit t-SNE on the combined data
    tsne = TSNE(n_components=2, random_state=0)
    pca = PCA(n_components=2, random_state=0)

    #concatenated_latent_space_2d = pca.fit_transform(concatenated_latent_space)
    num_codebook_vectors = 20
    indices = np.linspace(0, concatenated_latent_space.shape[0] - 1, num_codebook_vectors, dtype=int)
    codebook = concatenated_latent_space[indices]
    print("codebook size", cluster_centers.shape)
    #replaced_latent_vector = np.transpose(replaced_latent_vector, (0, 2, 1)).reshape(frames, channels)
    print(first_latent_vector.shape, replaced_latent_vector.shape, np.array(cluster_centers).shape)
    combined_data = np.vstack([
        first_latent_vector,  # Original latent vectors
        replaced_latent_vector,  # Replaced latent vectors
        np.array(cluster_centers)  # Codebook vectors
    ])
    print(combined_data.shape)

    # Perform PCA on the combined latent space
    pca = PCA(n_components=2, random_state=0)
    combined_latent_space_2d = pca.fit_transform(combined_data)

    # Split the transformed space back into original, replaced, and codebook vectors
    latent_space_2d = combined_latent_space_2d[:frames]
    replaced_latent_space_2d = combined_latent_space_2d[frames:2*frames]
    codebook_2d = combined_latent_space_2d[2 * frames:]

    # Plot the original and replaced latent vectors
    plt.figure(figsize=(10, 7))
    plt.scatter(latent_space_2d[:, 0], latent_space_2d[:, 1], c=closest_clusters, cmap='viridis', s=30, label='Original Latent Vectors')
    plt.scatter(replaced_latent_space_2d[:, 0], replaced_latent_space_2d[:, 1], c=closest_clusters, cmap='cool', s=30, label='Replaced Latent Vectors', marker='x')
    plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], color='red', s=30, label='Codebook Vectors', marker='*')
    # for i, txt in enumerate(range(latent_space_2d.shape[0])):
    #     plt.annotate(txt, (latent_space_2d[i, 0], latent_space_2d[i, 1]), fontsize=8)
    plt.colorbar(label='Codebook ID')
    plt.title("PCA of Concatenated Latent Space with Codebook Assignments")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(loc='upper right')
    plt.show()

    process_custom_latent(replaced_latent_vector[:,:-7]) # without tags"""