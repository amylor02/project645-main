# MAI645 Motion Prediction Project

This repository contains the motion-representation comparison project built around auto-conditioned LSTM baselines for three motion formats:

- positional,
- Euler angles,
- quaternion.

It also contains the quantitative evaluation code, report notebook, and the pretrained checkpoints used in the saved quantitative results under `synthesis/quant`.

The main dataset styles already present in the repo are:

- `indian`
- `martial`
- `salsa`

## 1. Environment setup

Create the conda environment from the repo root:

```bash
conda env create -f mai645.yml
conda activate mai645
```

Notes:

- run commands from the repository root,
- the positional scripts still assume CUDA explicitly, so use a CUDA-enabled PyTorch environment for the full baseline workflow,
- the Euler and quaternion scripts are more tolerant of CPU fallback, but the intended setup is still GPU training.

After cloning, run the three preprocessing scripts in section 6.1 first to create `train_data_pos/`, `train_data_euler/`, and `train_data_quad/` before trying to train or synthesize the direct baselines.

## 2. What is already included

The repository already contains:

- source BVH files in `train_data_bvh/`,
- pretrained checkpoints in `weights/`,
- synthesis outputs and quantitative results in `synthesis/`.

The generated representation folders are not the thing to assume after a fresh clone. Create them first by running all three scripts in section 6.1:

- `python .\code\generate_training_pos_data.py ...`
- `python .\code\generate_training_euler_data.py ...`
- `python .\code\generate_training_quad_data.py ...`

Even if you only want to synthesize from the shipped direct-baseline checkpoints, those preprocessing outputs are still needed because the synthesis scripts draw their seed clips from the generated representation datasets.

## 3. Pretrained models used in the quantitative tests

The quantitative evaluator in `code/evaluate_quantitative_metrics.py` uses these checkpoint folders by default. If you pass a folder to a synthesis script, it automatically picks the latest `.weight` file in that folder.

| Representation       | Style   | Checkpoint folder                | Checkpoint used in `synthesis/quant/summary.json` |
| -------------------- | ------- | -------------------------------- | ------------------------------------------------- |
| positional           | indian  | `weights/pos/indian`             | `weights/pos/indian/0096000.weight`               |
| positional           | martial | `weights/pos/martial`            | `weights/pos/martial/0096000.weight`              |
| positional           | salsa   | `weights/pos/salsa`              | `weights/pos/salsa/0096000.weight`                |
| euler                | indian  | `weights/euler/indian`           | `weights/euler/indian/0096000.weight`             |
| euler                | martial | `weights/euler/martial`          | `weights/euler/martial/0099500.weight`            |
| euler                | salsa   | `weights/euler/salsa`            | `weights/euler/salsa/0096000.weight`              |
| quaternion no-FK MSE | indian  | `weights/quad_no_fk_mse/indian`  | `weights/quad_no_fk_mse/indian/0096000.weight`    |
| quaternion no-FK MSE | martial | `weights/quad_no_fk_mse/martial` | `weights/quad_no_fk_mse/martial/0096000.weight`   |
| quaternion no-FK MSE | salsa   | `weights/quad_no_fk_mse/salsa`   | `weights/quad_no_fk_mse/salsa/0096000.weight`     |

## 4. Synthesize motion from the shipped checkpoints

The examples below use the same checkpoint folders that the quantitative evaluator uses. You can switch styles by changing `martial` to `indian` or `salsa` consistently in both the data folder and checkpoint folder.

### 4.1 Positional baseline

```bash
python .\code\synthesize_pos_motion.py --dances_folder .\train_data_pos\martial\ --read_weight_path .\weights\pos\martial\ --write_bvh_motion_folder .\synthesis\pos\martial\ --in_frame 171 --out_frame 171 --batch_size 5 --initial_seq_len 20 --generate_frames_number 400
```

### 4.2 Euler baseline

```bash
python .\code\synthesize_euler_motion.py --dances_folder .\train_data_euler\martial\ --read_weight_path .\weights\euler\martial\ --write_bvh_motion_folder .\synthesis\euler\martial\ --in_frame 132 --out_frame 132 --batch_size 5 --initial_seq_len 20 --generate_frames_number 400
```

There is also a spelling-wrapper command if you want the deliverable name:

```bash
python .\code\synthise_euler_motion.py --dances_folder .\train_data_euler\martial\ --read_weight_path .\weights\euler\martial\ --write_bvh_motion_folder .\synthesis\euler\martial\ --in_frame 132 --out_frame 132 --batch_size 5 --initial_seq_len 20 --generate_frames_number 400
```

### 4.3 Quaternion baseline used by the quantitative runs

```bash
python .\code\synthesize_quad_motion.py --dances_folder .\train_data_quad\martial\ --read_weight_path .\weights\quad_no_fk_mse\martial\ --write_bvh_motion_folder .\synthesis\quad_no_fk_mse\martial\ --in_frame 175 --out_frame 175 --batch_size 5 --initial_seq_len 20 --generate_frames_number 400
```

### 4.4 Output behavior

- each synthesis script writes BVH files into the folder given by `--write_bvh_motion_folder`,
- if `--read_weight_path` points to a folder, the script loads the lexicographically latest `.weight` file in that folder,
- `--initial_seq_len` is the real seed length,
- `--generate_frames_number` is the number of future frames generated after the seed.

## 5. Rerun the quantitative evaluation with the same pretrained models

From the repo root:

```bash
python .\code\evaluate_quantitative_metrics.py --output_root .\synthesis\quant
```

That command uses the same default experiment folders listed above unless you override them in code.

## 6. Retraining from scratch

Use the same style names already present in the repo: `indian`, `martial`, or `salsa`.

### 6.1 Preprocess BVH into representation data

Run these three commands first after cloning. The direct training and direct synthesis commands later in this README assume these generated folders already exist.

Positional:

```bash
python .\code\generate_training_pos_data.py --src_bvh_folder .\train_data_bvh\martial\ --output_npy_folder .\train_data_pos\martial\ --output_bvh_folder .\reconstructed_bvh_data_pos\martial\
```

Euler:

```bash
python .\code\generate_training_euler_data.py --src_bvh_folder .\train_data_bvh\martial\ --output_npy_folder .\train_data_euler\martial\ --output_bvh_folder .\reconstructed_bvh_data_euler\martial\
```

Quaternion:

```bash
python .\code\generate_training_quad_data.py --src_bvh_folder .\train_data_bvh\martial\ --output_npy_folder .\train_data_quad\martial\ --output_bvh_folder .\reconstructed_bvh_data_quad\martial\
```

### 6.2 Train positional

```bash
python .\code\pytorch_train_pos_aclstm.py --dances_folder .\train_data_pos\martial\ --write_weight_folder .\weights\pos\martial\ --write_bvh_motion_folder .\previews\pos\martial\ --in_frame 171 --out_frame 171 --batch_size 64 --total_iterations 10000
```

### 6.3 Train Euler

```bash
python .\code\pytorch_train_euler_aclstm.py --dances_folder .\train_data_euler\martial\ --write_weight_folder .\weights\euler\martial\ --write_bvh_motion_folder .\previews\euler\martial\ --in_frame 132 --out_frame 132 --batch_size 64 --total_iterations 10000
```

Optional Euler controls:

- `--recenter_root`
- `--recenter_y`
- `--augment_yaw_range_degrees 45`
- `--normalization_stats_path <stats.npz>` if training on normalized Euler data

### 6.4 Train quaternion the same way the quantitative tests expect

To match the quantitative setup, train the no-FK quaternion variant with the sign-aligned quaternion MSE option and save it under `weights/quad_no_fk_mse/<style>`:

```bash
python .\code\pytorch_train_quad_aclstm.py --dances_folder .\train_data_quad\martial\ --write_weight_folder .\weights\quad_no_fk_mse\martial\ --write_bvh_motion_folder .\previews\quad_no_fk_mse\martial\ --in_frame 175 --out_frame 175 --batch_size 64 --total_iterations 10000 --use_quaternion_mse_loss
```

Equivalent direct call:

```bash
python .\code\pytorch_train_quad_aclstm_no_fk.py --dances_folder .\train_data_quad\martial\ --write_weight_folder .\weights\quad_no_fk_mse\martial\ --write_bvh_motion_folder .\previews\quad_no_fk_mse\martial\ --in_frame 175 --out_frame 175 --batch_size 64 --total_iterations 10000 --use_quaternion_mse_loss
```

Important note:

- `pytorch_train_quad_aclstm.py` now delegates to `pytorch_train_quad_aclstm_no_fk.py` when run as a script,
- legacy `--fk_loss_weight` is ignored by that compatibility wrapper,
- if you want augmentation in the no-FK trainer, add `--augment_yaw_range_degrees` and/or `--augment_translation_range`.

### 6.5 Resume training from an existing checkpoint

All three training scripts accept `--read_weight_path` so you can continue from a saved `.weight` file.

Example:

```bash
python .\code\pytorch_train_euler_aclstm.py --dances_folder .\train_data_euler\martial\ --write_weight_folder .\weights\euler\martial\ --write_bvh_motion_folder .\previews\euler\martial\ --read_weight_path .\weights\euler\martial\0099500.weight --in_frame 132 --out_frame 132 --batch_size 64 --total_iterations 2000
```

## 7. Optional normalization workflow

Normalization support exists, but it is most practical for Euler.

Example Euler normalization:

```bash
python .\code\normalize_representation_data.py --representation euler --src_folder .\train_data_euler\martial\ --output_folder .\train_data_euler_normalized\martial\ --stats_path .\train_data_euler_normalized\euler_zscore_stats.npz
```

Then pass the stats file to both training and synthesis:

```bash
python .\code\pytorch_train_euler_aclstm.py --dances_folder .\train_data_euler_normalized\martial\ --write_weight_folder .\weights\euler_norm\martial\ --write_bvh_motion_folder .\previews\euler_norm\martial\ --in_frame 132 --out_frame 132 --batch_size 64 --total_iterations 10000 --normalization_stats_path .\train_data_euler_normalized\euler_zscore_stats.npz
```

```bash
python .\code\synthesize_euler_motion.py --dances_folder .\train_data_euler_normalized\martial\ --read_weight_path .\weights\euler_norm\martial\ --write_bvh_motion_folder .\synthesis\euler_norm\martial\ --in_frame 132 --out_frame 132 --batch_size 5 --initial_seq_len 20 --generate_frames_number 400 --normalization_stats_path .\train_data_euler_normalized\euler_zscore_stats.npz
```

## 8. Latent Autoencoder And Latent LSTM

The latent pipeline is separate from the positional, Euler, and quaternion `.npy` baselines.
It uses:

- a `train/` and `eval/` BVH dataset root,
- a `my_autoencoder` model folder under `models/`,
- an exported latent dataset folder with `latent_dataset_metadata.json`,
- an optional latent LSTM checkpoint folder under `weights/`.

### 8.1 Shipped latent assets in this repo

The maintained latent example path in this repo is the retargeted VAE workflow:

- autoencoder model folder: `models/model_martial_ret_vae_my_autoencoder_martial_ret`
- BVH dataset root: `tmp/my_autoencoder_martial_ret`
- exported latent dataset: `tmp/my_autoencoder_martial_ret_latents`
- standard latent LSTM checkpoints: `weights/latent_ret/martial`
- `LSTMCell` latent LSTM checkpoints: `weights/latent_ret_lstmcells/martial`

### 8.2 Run the shipped autoencoder by itself

To inspect pure autoencoder reconstruction quality without the latent LSTM, reconstruct eval clips directly through the pretrained generator:

```bash
python .\code\my_autoencoder\decode_latent_dataset.py --mode model_recon --data_path .\tmp\my_autoencoder_martial_ret\ --model_path .\models\model_martial_ret_vae_my_autoencoder_martial_ret\ --write_bvh_motion_folder .\previews\my_autoencoder_model_recon_ret\ --split eval --max_files 3
```

If you want to regenerate the latent export used by the latent LSTM, run:

```bash
python .\code\my_autoencoder\export_latent_dataset.py --data_path .\tmp\my_autoencoder_martial_ret\ --model_path .\models\model_martial_ret_vae_my_autoencoder_martial_ret\ --write_latent_folder .\tmp\my_autoencoder_martial_ret_latents\ --split all
```

That command writes both the latent `.npy` files and the `latent_dataset_metadata.json` manifest that ties the latent export back to the exact autoencoder checkpoint and source dataset.

### 8.3 Run the shipped latent LSTM

The synthesis examples below intentionally use the `train` split.

Standard stacked LSTM checkpoint:

```bash
python .\code\synthesize_latent_motion.py --latent_folder .\tmp\my_autoencoder_martial_ret_latents\ --read_weight_path .\weights\latent_ret\martial\ --write_latent_motion_folder .\synthesis\latent_ret\latents\ --write_bvh_motion_folder .\synthesis\latent_ret\bvh\ --split train --num_samples 3 --initial_seq_len 2 --generate_latent_steps 200
```

`LSTMCell`-style checkpoint:

```bash
python .\code\synthesize_latent_motion.py --latent_folder .\tmp\my_autoencoder_martial_ret_latents\ --read_weight_path .\weights\latent_ret_lstmcells\martial\ --write_latent_motion_folder .\synthesis\latent_ret_lstmcells\latents\ --write_bvh_motion_folder .\synthesis\latent_ret_lstmcells\bvh\ --split train --num_samples 3 --initial_seq_len 2 --generate_latent_steps 200
```

Notes:

- `--read_weight_path` can point to either a single checkpoint file or a checkpoint folder,
- both `--initial_seq_len` and `--generate_latent_steps` are measured in latent time steps, not raw BVH frames,
- the decode step is automatic and uses the autoencoder metadata stored in the latent manifest,
- if you retrain or replace the autoencoder, re-export the latent dataset before reusing an old latent LSTM.

### 8.4 Retrain the autoencoder from raw BVH files

To build a fresh `train/` and `eval/` split from the raw martial BVHs:

```bash
python .\code\my_autoencoder\prepare_bvh_train_eval_split.py --src_bvh_folder .\train_data_bvh\martial\ --output_dataset_folder .\tmp\my_autoencoder_martial\ --validation_fraction 0.2 --split_seed 1234
```

To train a new VAE-backed autoencoder on that split:

```bash
python .\code\my_autoencoder\train_mai645_vae.py .\tmp\my_autoencoder_martial\ martial_latent
```

That creates a model folder under `models/model_<name>_<dataset_basename>/` containing at least `generator.pt` and `data.pt`.

If you want to retrain the same retargeted autoencoder configuration used by the shipped latent assets, use the existing retargeted dataset root and the retargeted skeleton settings:

```bash
python .\code\my_autoencoder\train_mai645_vae.py .\tmp\my_autoencoder_martial_ret\ martial_ret_vae --sparse_joints 0 4 8 13 17 21 --feet_idxs 4 8 --head_idx 13 --shoulder_idxs 15 19 --skeleton_height 0.68 --feet_contact_threshold 0.02
```

Optional notes:

- add `--train_mode all` if you also want the legacy IK stage trained,
- add `--load` to continue training an existing model folder,
- add `--smoke_test` only for quick wiring checks, not report-quality training.

### 8.5 Retrain the latent LSTM on exported latents

After training or choosing an autoencoder model, export the latent dataset:

```bash
python .\code\my_autoencoder\export_latent_dataset.py --data_path .\tmp\my_autoencoder_martial_ret\ --model_path .\models\model_martial_ret_vae_my_autoencoder_martial_ret\ --write_latent_folder .\tmp\my_autoencoder_martial_ret_latents\ --split all
```

Then train the standard latent LSTM:

```bash
python .\code\pytorch_train_latent_lstm.py --latent_folder .\tmp\my_autoencoder_martial_ret_latents\ --write_weight_folder .\weights\latent_ret\martial\ --seq_len 32 --batch_size 64 --hidden_size 512 --total_iterations 10000
```

Or train the `LSTMCell` variant:

```bash
python .\code\pytorch_train_latent_lstm.py --latent_folder .\tmp\my_autoencoder_martial_ret_latents\ --write_weight_folder .\weights\latent_ret_lstmcells\martial\ --seq_len 32 --batch_size 64 --hidden_size 512 --total_iterations 10000 --model_type lstm_cells
```

Optional latent-LSTM controls:

- add `--groundtruth_num 5 --condition_num 5` if you want the same 5-ground-truth / 5-feedback conditioning schedule used by the acLSTM baselines,
- add `--read_weight_path <checkpoint>` to resume training,
- change `--window_stride` if you want fewer overlapping latent windows.

### 8.6 End-to-end latent retraining order

If you are rebuilding the latent pipeline from scratch, the safe order is:

1. prepare the `train/` and `eval/` BVH split,
2. train the autoencoder,
3. export the latent dataset,
4. train the latent LSTM,
5. synthesize and decode with `synthesize_latent_motion.py`.

## 9. Useful folders

- `code/`: preprocessing, training, synthesis, and evaluation scripts
- `models/`: `my_autoencoder` generator and optional IK checkpoints
- `tmp/`: latent datasets, `train/eval` BVH splits, and intermediate exports
- `weights/`: pretrained and retrained checkpoints
- `synthesis/`: generated BVH outputs and quantitative evaluation exports

## 10. Viewing BVH output

You can inspect generated BVH files in tools such as MotionBuilder, Maya, Blender add-ons, or a lightweight online viewer such as:

http://lo-th.github.io/olympe/BVH_player.html
