import read_bvh
import numpy as np
from os import listdir
import os
import argparse


def get_files_with_extension(folder_path, extension):
    file_names = []
    for file_name in sorted(listdir(folder_path)):
        if len(file_name) > len(extension) and file_name.endswith(extension):
            file_names.append(file_name)
    return file_names


def generate_pos_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    if os.path.exists(tar_traindata_folder) == False:
        os.makedirs(tar_traindata_folder)
    bvh_dances_names = get_files_with_extension(src_bvh_folder, ".bvh")
    print("Encoding {} BVH files from {}".format(len(bvh_dances_names), src_bvh_folder))
    for index, bvh_dance_name in enumerate(bvh_dances_names, start=1):
        print(
            "[{}/{}] Encoding {}".format(index, len(bvh_dances_names), bvh_dance_name)
        )
        dance = read_bvh.get_train_data(os.path.join(src_bvh_folder, bvh_dance_name))
        np.save(os.path.join(tar_traindata_folder, bvh_dance_name + ".npy"), dance)


def generate_pos_bvh_from_traindata(src_train_folder, tar_bvh_folder):
    if os.path.exists(tar_bvh_folder) == False:
        os.makedirs(tar_bvh_folder)
    dances_names = get_files_with_extension(src_train_folder, ".npy")
    print(
        "Reconstructing {} BVH files from {}".format(
            len(dances_names), src_train_folder
        )
    )
    for index, dance_name in enumerate(dances_names, start=1):
        print("[{}/{}] Reconstructing {}".format(index, len(dances_names), dance_name))
        dance = np.load(os.path.join(src_train_folder, dance_name))
        dance2 = []
        for i in range(int(dance.shape[0])):
            dance2 = dance2 + [dance[i]]

        read_bvh.write_traindata_to_bvh(
            os.path.join(tar_bvh_folder, dance_name + ".bvh"), np.array(dance2)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_bvh_folder",
        type=str,
        default="train_data_bvh/martial/",
        help="Source folder containing BVH files",
    )
    parser.add_argument(
        "--output_npy_folder",
        type=str,
        default="train_data_pos/martial/",
        help="Target folder for positional NumPy files",
    )
    parser.add_argument(
        "--output_bvh_folder",
        type=str,
        default="reconstructed_bvh_data_pos/martial/",
        help="Target folder for reconstructed BVH files",
    )
    parser.add_argument(
        "--skip_reconstruction",
        action="store_true",
        help="Only write NumPy training data and skip reconstruction back to BVH",
    )
    args = parser.parse_args()

    generate_pos_traindata_from_bvh(args.src_bvh_folder, args.output_npy_folder)

    if args.skip_reconstruction:
        print("Skipping BVH reconstruction")
        return

    generate_pos_bvh_from_traindata(args.output_npy_folder, args.output_bvh_folder)


if __name__ == "__main__":
    main()
