import argparse
import os
import random
import shutil

from mai645_latent_utils import ensure_directory


def get_bvh_files(src_bvh_folder):
    return sorted(
        [
            file_name
            for file_name in os.listdir(src_bvh_folder)
            if file_name.endswith(".bvh")
        ]
    )


def copy_file(src_path, dst_path, copy_mode):
    if copy_mode == "hardlink":
        if os.path.exists(dst_path):
            os.remove(dst_path)
        os.link(src_path, dst_path)
        return
    shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_bvh_folder", type=str, required=True)
    parser.add_argument("--output_dataset_folder", type=str, required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--copy_mode", type=str, default="copy", choices=["copy", "hardlink"])
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args()

    bvh_files = get_bvh_files(args.src_bvh_folder)
    if len(bvh_files) == 0:
        raise ValueError("No .bvh files found under {}".format(args.src_bvh_folder))

    if args.max_files > 0:
        bvh_files = bvh_files[: args.max_files]

    file_indices = list(range(len(bvh_files)))
    random.Random(args.split_seed).shuffle(file_indices)

    validation_count = int(round(len(file_indices) * args.validation_fraction))
    validation_count = max(1, validation_count) if len(file_indices) > 1 else 0
    validation_count = min(validation_count, max(0, len(file_indices) - 1))
    validation_indices = set(file_indices[:validation_count])

    train_dir = os.path.join(args.output_dataset_folder, "train")
    eval_dir = os.path.join(args.output_dataset_folder, "eval")
    ensure_directory(train_dir)
    ensure_directory(eval_dir)

    for file_index, file_name in enumerate(bvh_files):
        destination_dir = eval_dir if file_index in validation_indices else train_dir
        src_path = os.path.join(args.src_bvh_folder, file_name)
        dst_path = os.path.join(destination_dir, file_name)
        copy_file(src_path, dst_path, args.copy_mode)

    print("Prepared dataset split at {}".format(args.output_dataset_folder))
    print("Train files:", len(bvh_files) - len(validation_indices))
    print("Eval files:", len(validation_indices))


if __name__ == "__main__":
    main()