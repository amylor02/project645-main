import argparse
import os

import numpy as np
import torch

from mai645_latent_utils import build_test_dataset
from mai645_latent_utils import create_runtime
from mai645_latent_utils import encode_dataset_item
from mai645_latent_utils import ensure_directory
from mai645_latent_utils import save_json
from mai645_latent_utils import summarize_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--write_latent_folder", type=str, required=True)
    parser.add_argument("--split", type=str, default="all", choices=["train", "eval", "all"])
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--use_posterior_sample", action="store_true")
    args = parser.parse_args()

    ensure_directory(args.write_latent_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime = create_runtime(args.data_path, args.model_path, device)
    dataset, file_records = build_test_dataset(runtime, split=args.split, max_files=args.max_files)

    export_mode = "posterior_sample" if args.use_posterior_sample else "posterior_mean"
    manifest = summarize_runtime(runtime)
    manifest.update(
        {
            "data_path": os.path.abspath(args.data_path),
            "split": args.split,
            "latent_export_mode": export_mode,
            "files": [],
        }
    )

    for index, file_record in enumerate(file_records):
        split_output_dir = os.path.join(args.write_latent_folder, file_record.split_name)
        ensure_directory(split_output_dir)

        encoded_item = encode_dataset_item(
            runtime,
            dataset,
            index,
            use_mean=args.use_posterior_sample is False,
        )
        latent_file_name = file_record.filename + ".npy"
        latent_path = os.path.join(split_output_dir, latent_file_name)
        np.save(latent_path, encoded_item["latent"].astype(np.float32))

        manifest["files"].append(
            {
                "source_split": file_record.split_name,
                "source_file_name": file_record.filename,
                "source_file_path": os.path.abspath(file_record.file_path),
                "source_relative_path": file_record.source_relative_path.replace("\\", "/"),
                "latent_file_name": latent_file_name,
                "latent_file_path": os.path.abspath(latent_path),
                "latent_relative_path": os.path.join(file_record.split_name, latent_file_name).replace("\\", "/"),
                "original_frame_count": encoded_item["original_frame_count"],
                "prepared_frame_count": encoded_item["prepared_frame_count"],
                "latent_length": encoded_item["latent_length"],
                "latent_width": encoded_item["latent_width"],
            }
        )
        print(
            "Exported {} -> latent shape {}".format(
                file_record.source_relative_path,
                encoded_item["latent"].shape,
            )
        )

    save_json(
        os.path.join(args.write_latent_folder, "latent_dataset_metadata.json"),
        manifest,
    )


if __name__ == "__main__":
    main()