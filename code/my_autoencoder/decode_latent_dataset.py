import argparse
import os

import numpy as np
import torch

from mai645_latent_utils import build_test_dataset
from mai645_latent_utils import collect_latent_file_records
from mai645_latent_utils import create_runtime
from mai645_latent_utils import decode_dataset_item_to_bvh
from mai645_latent_utils import ensure_directory
from mai645_latent_utils import load_json
from mai645_latent_utils import reconstruct_dataset_item_to_bvh


def _build_dataset_indices(file_entries):
    indices = {}
    for index, file_entry in enumerate(file_entries):
        indices[file_entry["source_relative_path"]] = index
    return indices


def _build_entries_from_manifest(manifest, split, max_files):
    selected_entries = []
    for file_entry in manifest.get("files", []):
        if split != "all" and file_entry["source_split"] != split:
            continue
        selected_entries.append(file_entry)
        if max_files > 0 and len(selected_entries) >= max_files:
            break
    return selected_entries


def _build_entries_without_manifest(latent_folder, data_path, split, max_files):
    file_records = collect_latent_file_records(
        latent_folder, split=split, max_files=max_files
    )
    if len(file_records) == 0:
        raise FileNotFoundError(
            "No latent .npy files were found under {}. Run export_latent_dataset.py first or point --latent_folder at an existing latent export.".format(
                os.path.abspath(latent_folder)
            )
        )

    selected_entries = []
    for file_record in file_records:
        source_file_path = os.path.join(
            data_path, file_record.split_name, file_record.source_file_name
        )
        if os.path.exists(source_file_path) is False:
            raise FileNotFoundError(
                "Could not match latent file {} to source BVH {}".format(
                    file_record.file_path,
                    source_file_path,
                )
            )
        selected_entries.append(
            {
                "source_split": file_record.split_name,
                "source_file_name": file_record.source_file_name,
                "source_file_path": os.path.abspath(source_file_path),
                "source_relative_path": file_record.source_relative_path.replace(
                    "\\", "/"
                ),
                "latent_file_path": os.path.abspath(file_record.file_path),
                "latent_relative_path": file_record.latent_relative_path.replace(
                    "\\", "/"
                ),
            }
        )
    return selected_entries


def _resolve_runtime_inputs(args):
    manifest_path = args.manifest_path
    if manifest_path == "" and args.latent_folder != "":
        manifest_path = os.path.join(args.latent_folder, "latent_dataset_metadata.json")

    manifest = None
    if manifest_path != "" and os.path.exists(manifest_path):
        manifest = load_json(manifest_path)

    data_path = args.data_path
    model_path = args.model_path
    if manifest is not None:
        if data_path == "":
            data_path = manifest["data_path"]
        if model_path == "":
            model_path = manifest["generator_path"]

    return manifest_path, manifest, data_path, model_path


def _decode_from_latent_files(args, runtime, manifest, manifest_path, data_path):
    if args.latent_folder == "":
        raise ValueError("--latent_folder is required when --mode latent")

    if manifest is not None:
        selected_entries = _build_entries_from_manifest(
            manifest, args.split, args.max_files
        )
    else:
        if data_path == "" or args.model_path == "":
            raise FileNotFoundError(
                "Could not find latent manifest at {}. Run export_latent_dataset.py first, or rerun decode_latent_dataset.py with both --data_path and --model_path so it can reconstruct the file mapping without the manifest.".format(
                    os.path.abspath(manifest_path)
                )
            )
        selected_entries = _build_entries_without_manifest(
            args.latent_folder,
            data_path,
            args.split,
            args.max_files,
        )

    if len(selected_entries) == 0:
        raise ValueError("No latent entries matched the requested split")

    dataset_cache = {}
    dataset_index_cache = {}
    for split_name in sorted({entry["source_split"] for entry in selected_entries}):
        dataset, file_records = build_test_dataset(runtime, split=split_name)
        dataset_cache[split_name] = dataset
        dataset_index_cache[split_name] = {
            file_record.source_relative_path.replace("\\", "/"): index
            for index, file_record in enumerate(file_records)
        }

    for file_entry in selected_entries:
        split_name = file_entry["source_split"]
        dataset = dataset_cache[split_name]
        source_key = file_entry["source_relative_path"]
        dataset_index = dataset_index_cache[split_name][source_key]

        latent_path = file_entry.get("latent_file_path")
        if latent_path is None or os.path.exists(latent_path) is False:
            latent_path = os.path.join(
                args.latent_folder, file_entry["latent_relative_path"]
            )
        latent = np.load(latent_path).astype(np.float32)

        output_base = os.path.splitext(file_entry["source_file_name"])[0]
        output_filename = "{}_{}_decoded.bvh".format(split_name, output_base)
        output_path = decode_dataset_item_to_bvh(
            runtime,
            dataset,
            dataset_index,
            latent,
            output_filename,
            args.write_bvh_motion_folder,
        )
        print(
            "Decoded {} -> {}".format(
                source_key,
                output_path,
            )
        )


def _reconstruct_from_model(args, runtime, data_path, model_path):
    if data_path == "" or model_path == "":
        raise FileNotFoundError(
            "--mode model_recon requires --data_path and --model_path, or a manifest that provides both values."
        )

    dataset, file_records = build_test_dataset(
        runtime,
        split=args.split,
        max_files=args.max_files,
    )

    for dataset_index, file_record in enumerate(file_records):
        output_base = os.path.splitext(file_record.filename)[0]
        output_filename = "{}_{}_model_recon.bvh".format(
            file_record.split_name,
            output_base,
        )
        output_path = reconstruct_dataset_item_to_bvh(
            runtime,
            dataset,
            dataset_index,
            output_filename,
            args.write_bvh_motion_folder,
        )
        print(
            "Reconstructed {} -> {}".format(
                file_record.source_relative_path.replace("\\", "/"),
                output_path,
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="latent", choices=["latent", "model_recon"]
    )
    parser.add_argument("--latent_folder", type=str, default="")
    parser.add_argument("--write_bvh_motion_folder", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--use_ik", action="store_true")
    parser.add_argument(
        "--split", type=str, default="all", choices=["train", "eval", "all"]
    )
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args()

    ensure_directory(args.write_bvh_motion_folder)

    manifest_path, manifest, data_path, model_path = _resolve_runtime_inputs(args)

    if data_path == "" or model_path == "":
        if args.mode == "latent":
            raise FileNotFoundError(
                "decode_latent_dataset.py needs to resolve both --data_path and --model_path from either explicit arguments or the latent manifest."
            )
        raise FileNotFoundError(
            "--mode model_recon requires --data_path and --model_path, or a manifest that provides both values."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = create_runtime(
        data_path,
        model_path,
        device,
        load_ik=args.use_ik,
    )

    if args.mode == "latent":
        _decode_from_latent_files(args, runtime, manifest, manifest_path, data_path)
        return

    _reconstruct_from_model(args, runtime, data_path, model_path)


if __name__ == "__main__":
    main()
