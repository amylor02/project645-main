# Adding this normalization pipeline to see if training is more stable
# and if results are better.

import argparse
import json
import os
from os import listdir

import numpy as np

import read_bvh


def get_files_with_extension(folder_path, extension):
    file_names = []
    for file_name in sorted(listdir(folder_path)):
        if len(file_name) > len(extension) and file_name.endswith(extension):
            file_names.append(file_name)
    return file_names


def wrap_angles_degrees(angle_values):
    return ((angle_values + 180.0) % 360.0) - 180.0


def prepare_pos_motion(motion):
    return np.array(motion, dtype=np.float64, copy=True)


def prepare_euler_motion(motion):
    prepared_motion = np.array(motion, dtype=np.float64, copy=True)
    prepared_motion[:, 3:] = wrap_angles_degrees(prepared_motion[:, 3:])
    return prepared_motion


def prepare_quaternion_motion(motion):
    prepared_motion = np.array(motion, dtype=np.float64, copy=True)
    quaternion_offset = 3
    quaternion_width = 4
    quaternion_count = int(
        (prepared_motion.shape[1] - quaternion_offset) / quaternion_width
    )

    for frame_index in range(prepared_motion.shape[0]):
        for quaternion_index in range(quaternion_count):
            start_index = quaternion_offset + quaternion_index * quaternion_width
            end_index = start_index + quaternion_width
            prepared_motion[frame_index, start_index:end_index] = (
                read_bvh.normalize_quaternion(
                    prepared_motion[frame_index, start_index:end_index]
                )
            )

    return prepared_motion


def prepare_motion_for_representation(motion, representation):
    if representation == "pos":
        return prepare_pos_motion(motion)
    if representation == "euler":
        return prepare_euler_motion(motion)
    if representation == "quat":
        return prepare_quaternion_motion(motion)
    raise ValueError("Unsupported representation: {}".format(representation))


def compute_zscore_stats(prepared_motions, epsilon):
    stacked_frames = np.concatenate(prepared_motions, axis=0)
    channel_mean = np.mean(stacked_frames, axis=0)
    channel_std = np.std(stacked_frames, axis=0)
    zero_std_mask = channel_std < epsilon
    safe_std = np.where(zero_std_mask, 1.0, channel_std)
    return channel_mean, channel_std, safe_std, zero_std_mask


def normalize_motion(prepared_motion, channel_mean, safe_std):
    return (prepared_motion - channel_mean) / safe_std


def save_stats(
    stats_path,
    representation,
    method,
    epsilon,
    channel_mean,
    channel_std,
    safe_std,
    zero_std_mask,
    file_count,
    total_frames,
):
    np.savez(
        stats_path,
        representation=representation,
        method=method,
        epsilon=np.array([epsilon], dtype=np.float64),
        channel_mean=channel_mean,
        channel_std=channel_std,
        safe_std=safe_std,
        zero_std_mask=zero_std_mask.astype(np.uint8),
    )

    summary_path = os.path.splitext(stats_path)[0] + ".json"
    summary = {
        "representation": representation,
        "method": method,
        "epsilon": epsilon,
        "file_count": file_count,
        "total_frames": total_frames,
        "feature_count": int(channel_mean.shape[0]),
        "zero_std_feature_count": int(np.sum(zero_std_mask)),
    }
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        type=str,
        choices=["pos", "euler", "quat"],
        required=True,
        help="Motion representation stored in the NumPy files",
    )
    parser.add_argument(
        "--src_folder",
        type=str,
        required=True,
        help="Folder containing representation-specific .npy motion files",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder where normalized .npy motion files will be written",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="",
        help="Optional path for normalization statistics (.npz). Defaults inside the output folder",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["none", "zscore"],
        default="zscore",
        help="Normalization method to apply",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Minimum standard deviation threshold to avoid division by zero",
    )
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    input_file_names = get_files_with_extension(args.src_folder, ".npy")
    if len(input_file_names) == 0:
        raise ValueError("No .npy files found in {}".format(args.src_folder))

    print(
        "Preparing {} {} files from {}".format(
            len(input_file_names), args.representation, args.src_folder
        )
    )

    prepared_motions = []
    total_frames = 0
    for index, file_name in enumerate(input_file_names, start=1):
        print("[{}/{}] Reading {}".format(index, len(input_file_names), file_name))
        motion = np.load(os.path.join(args.src_folder, file_name))
        prepared_motion = prepare_motion_for_representation(motion, args.representation)
        prepared_motions.append(prepared_motion)
        total_frames += prepared_motion.shape[0]

    if args.method == "zscore":
        channel_mean, channel_std, safe_std, zero_std_mask = compute_zscore_stats(
            prepared_motions, args.epsilon
        )
    else:
        feature_count = prepared_motions[0].shape[1]
        channel_mean = np.zeros(feature_count, dtype=np.float64)
        channel_std = np.ones(feature_count, dtype=np.float64)
        safe_std = np.ones(feature_count, dtype=np.float64)
        zero_std_mask = np.zeros(feature_count, dtype=bool)

    if args.stats_path == "":
        stats_path = os.path.join(
            args.output_folder,
            "{}_{}_stats.npz".format(args.representation, args.method),
        )
    else:
        stats_path = args.stats_path

    for index, file_name in enumerate(input_file_names, start=1):
        print("[{}/{}] Writing {}".format(index, len(input_file_names), file_name))
        prepared_motion = prepared_motions[index - 1]
        normalized_motion = normalize_motion(prepared_motion, channel_mean, safe_std)
        np.save(os.path.join(args.output_folder, file_name), normalized_motion)

    save_stats(
        stats_path,
        args.representation,
        args.method,
        args.epsilon,
        channel_mean,
        channel_std,
        safe_std,
        zero_std_mask,
        len(input_file_names),
        total_frames,
    )

    print("Saved normalization statistics to {}".format(stats_path))


if __name__ == "__main__":
    main()
