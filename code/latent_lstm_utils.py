import os
import random
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MY_AUTOENCODER_DIR = os.path.join(CURRENT_DIR, "my_autoencoder")
if MY_AUTOENCODER_DIR not in sys.path:
    sys.path.insert(0, MY_AUTOENCODER_DIR)


from mai645_latent_utils import build_test_dataset
from mai645_latent_utils import collect_latent_file_records
from mai645_latent_utils import create_runtime
from mai645_latent_utils import decode_dataset_item_to_bvh
from mai645_latent_utils import ensure_directory
from mai645_latent_utils import load_json
from mai645_latent_utils import summarize_runtime

DEFAULT_MANIFEST_NAME = "latent_dataset_metadata.json"
CHECKPOINT_EXTENSIONS = (".pt", ".weight")
DEFAULT_MODEL_TYPE = "lstm"
LATENT_MODEL_TYPES = ("lstm", "lstm_cells")


def get_condition_lst(condition_num, groundtruth_num, seq_len):
    if condition_num <= 0 or groundtruth_num <= 0:
        return np.ones(seq_len, dtype=np.int32)

    cycle_length = condition_num + groundtruth_num
    condition_lst = np.ones(seq_len, dtype=np.int32)
    for step_index in range(seq_len):
        if step_index % cycle_length >= groundtruth_num:
            condition_lst[step_index] = 0
    return condition_lst


@dataclass
class LatentContext:
    latent_folder: str
    manifest_path: str
    manifest: dict[str, Any] | None
    data_path: str
    model_path: str


class LatentSequenceLSTM(nn.Module):
    def __init__(
        self,
        latent_width,
        hidden_size=512,
        num_layers=3,
        dropout=0.0,
    ):
        super(LatentSequenceLSTM, self).__init__()
        self.latent_width = int(latent_width)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.model_type = "lstm"

        self.lstm = nn.LSTM(
            input_size=self.latent_width,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.decoder = nn.Linear(self.hidden_size, self.latent_width)

    def forward(self, input_seq, hidden_state=None):
        lstm_output, hidden_state = self.lstm(input_seq, hidden_state)
        predicted_seq = self.decoder(lstm_output)
        return predicted_seq, hidden_state

    def forward_step(self, input_frame, hidden_state=None):
        if input_frame.ndim != 2:
            raise ValueError(
                "Expected latent frame with shape [B, C], got {}".format(
                    tuple(input_frame.shape)
                )
            )

        lstm_output, hidden_state = self.lstm(input_frame.unsqueeze(1), hidden_state)
        output_frame = self.decoder(lstm_output[:, 0, :])
        return output_frame, hidden_state

    def forward_conditioned(
        self,
        input_seq,
        condition_num,
        groundtruth_num,
        hidden_state=None,
    ):
        if input_seq.ndim != 3:
            raise ValueError(
                "Expected latent sequence with shape [B, T, C], got {}".format(
                    tuple(input_seq.shape)
                )
            )

        batch_size = input_seq.size(0)
        output_frames = []
        output_frame = torch.zeros(
            batch_size,
            self.latent_width,
            device=input_seq.device,
            dtype=input_seq.dtype,
        )
        condition_lst = get_condition_lst(
            condition_num,
            groundtruth_num,
            input_seq.size(1),
        )

        for step_index in range(input_seq.size(1)):
            if condition_lst[step_index] == 1:
                input_frame = input_seq[:, step_index, :]
            else:
                input_frame = output_frame

            output_frame, hidden_state = self.forward_step(input_frame, hidden_state)
            output_frames.append(output_frame)

        return torch.stack(output_frames, dim=1), hidden_state

    def generate(self, seed_seq, generate_steps):
        if seed_seq.ndim != 3:
            raise ValueError(
                "Expected seed sequence with shape [B, T, C], got {}".format(
                    tuple(seed_seq.shape)
                )
            )
        if seed_seq.size(1) == 0:
            raise ValueError("Seed sequence must contain at least one latent step")

        predicted_seed, hidden_state = self.forward(seed_seq)
        if generate_steps <= 0:
            return seed_seq.clone()

        generated_steps = []
        next_input = predicted_seed[:, -1:, :]
        for _ in range(generate_steps):
            generated_steps.append(next_input)
            predicted_step, hidden_state = self.forward(next_input, hidden_state)
            next_input = predicted_step[:, -1:, :]

        return torch.cat((seed_seq, torch.cat(generated_steps, dim=1)), dim=1)


class LatentSequenceLSTMCells(nn.Module):
    def __init__(
        self,
        latent_width,
        hidden_size=512,
        num_layers=3,
        dropout=0.0,
    ):
        super(LatentSequenceLSTMCells, self).__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.latent_width = int(latent_width)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.model_type = "lstm_cells"

        self.lstm_cells = nn.ModuleList()
        layer_input_size = self.latent_width
        for _ in range(self.num_layers):
            self.lstm_cells.append(nn.LSTMCell(layer_input_size, self.hidden_size))
            layer_input_size = self.hidden_size

        self.dropout_layer = nn.Dropout(self.dropout)
        self.decoder = nn.Linear(self.hidden_size, self.latent_width)

    def init_hidden(self, batch_size, device):
        hidden_states = []
        cell_states = []
        for _ in range(self.num_layers):
            hidden_states.append(
                torch.zeros(batch_size, self.hidden_size, device=device)
            )
            cell_states.append(torch.zeros(batch_size, self.hidden_size, device=device))
        return hidden_states, cell_states

    def forward_lstm(self, input_frame, hidden_states, cell_states):
        next_hidden_states = []
        next_cell_states = []
        layer_input = input_frame

        for layer_index, lstm_cell in enumerate(self.lstm_cells):
            next_hidden, next_cell = lstm_cell(
                layer_input,
                (hidden_states[layer_index], cell_states[layer_index]),
            )
            next_hidden_states.append(next_hidden)
            next_cell_states.append(next_cell)

            if layer_index + 1 < self.num_layers and self.dropout > 0.0:
                layer_input = self.dropout_layer(next_hidden)
            else:
                layer_input = next_hidden

        output_frame = self.decoder(layer_input)
        return output_frame, next_hidden_states, next_cell_states

    def forward(self, input_seq, hidden_state=None):
        if input_seq.ndim != 3:
            raise ValueError(
                "Expected latent sequence with shape [B, T, C], got {}".format(
                    tuple(input_seq.shape)
                )
            )

        batch_size = input_seq.size(0)
        device = input_seq.device
        if hidden_state is None:
            hidden_states, cell_states = self.init_hidden(batch_size, device)
        else:
            hidden_states, cell_states = hidden_state

        output_frames = []
        for step_index in range(input_seq.size(1)):
            output_frame, hidden_states, cell_states = self.forward_lstm(
                input_seq[:, step_index, :],
                hidden_states,
                cell_states,
            )
            output_frames.append(output_frame)

        return torch.stack(output_frames, dim=1), (hidden_states, cell_states)

    def forward_conditioned(
        self,
        input_seq,
        condition_num,
        groundtruth_num,
        hidden_state=None,
    ):
        if input_seq.ndim != 3:
            raise ValueError(
                "Expected latent sequence with shape [B, T, C], got {}".format(
                    tuple(input_seq.shape)
                )
            )

        batch_size = input_seq.size(0)
        device = input_seq.device
        if hidden_state is None:
            hidden_states, cell_states = self.init_hidden(batch_size, device)
        else:
            hidden_states, cell_states = hidden_state

        output_frames = []
        output_frame = torch.zeros(
            batch_size,
            self.latent_width,
            device=device,
            dtype=input_seq.dtype,
        )
        condition_lst = get_condition_lst(
            condition_num,
            groundtruth_num,
            input_seq.size(1),
        )

        for step_index in range(input_seq.size(1)):
            if condition_lst[step_index] == 1:
                input_frame = input_seq[:, step_index, :]
            else:
                input_frame = output_frame

            output_frame, hidden_states, cell_states = self.forward_lstm(
                input_frame,
                hidden_states,
                cell_states,
            )
            output_frames.append(output_frame)

        return torch.stack(output_frames, dim=1), (hidden_states, cell_states)

    def generate(self, seed_seq, generate_steps):
        if seed_seq.ndim != 3:
            raise ValueError(
                "Expected seed sequence with shape [B, T, C], got {}".format(
                    tuple(seed_seq.shape)
                )
            )
        if seed_seq.size(1) == 0:
            raise ValueError("Seed sequence must contain at least one latent step")

        predicted_seed, hidden_state = self.forward(seed_seq)
        if generate_steps <= 0:
            return seed_seq.clone()

        hidden_states, cell_states = hidden_state
        generated_steps = []
        next_input = predicted_seed[:, -1, :]
        for _ in range(generate_steps):
            output_frame, hidden_states, cell_states = self.forward_lstm(
                next_input,
                hidden_states,
                cell_states,
            )
            generated_steps.append(output_frame.unsqueeze(1))
            next_input = output_frame

        return torch.cat((seed_seq, torch.cat(generated_steps, dim=1)), dim=1)


def validate_model_type(model_type):
    normalized_model_type = str(model_type).lower()
    if normalized_model_type not in LATENT_MODEL_TYPES:
        raise ValueError(
            "Unsupported latent model_type {}. Expected one of {}".format(
                model_type,
                ", ".join(LATENT_MODEL_TYPES),
            )
        )
    return normalized_model_type


def resolve_model_type(train_config=None, requested_model_type=""):
    checkpoint_model_type = DEFAULT_MODEL_TYPE
    if train_config is not None:
        checkpoint_model_type = validate_model_type(
            train_config.get("model_type", DEFAULT_MODEL_TYPE)
        )

    if requested_model_type == "":
        return checkpoint_model_type

    normalized_requested_model_type = validate_model_type(requested_model_type)
    if normalized_requested_model_type != checkpoint_model_type:
        raise ValueError(
            "Requested model_type {} does not match checkpoint model_type {}".format(
                normalized_requested_model_type,
                checkpoint_model_type,
            )
        )
    return checkpoint_model_type


def build_latent_sequence_model(
    model_type,
    latent_width,
    hidden_size=512,
    num_layers=3,
    dropout=0.0,
):
    normalized_model_type = validate_model_type(model_type)
    if normalized_model_type == "lstm":
        return LatentSequenceLSTM(
            latent_width=latent_width,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    return LatentSequenceLSTMCells(
        latent_width=latent_width,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


class LatentWindowDataset(Dataset):
    def __init__(self, latent_entries, sample_length, window_stride=1):
        if sample_length < 2:
            raise ValueError("sample_length must be at least 2")

        self.latent_entries = list(latent_entries)
        self.sample_length = int(sample_length)
        self.window_stride = max(int(window_stride), 1)
        self.latent_sequences = []
        self.sample_locations = []
        self.latent_width = None

        for entry_index, entry in enumerate(self.latent_entries):
            latent = np.load(entry["latent_file_path"]).astype(np.float32)
            if latent.ndim != 2:
                raise ValueError(
                    "Expected latent array [T, C] in {}, got {}".format(
                        entry["latent_file_path"],
                        latent.shape,
                    )
                )

            if self.latent_width is None:
                self.latent_width = int(latent.shape[1])
            elif self.latent_width != int(latent.shape[1]):
                raise ValueError(
                    "Latent width mismatch: expected {}, found {} in {}".format(
                        self.latent_width,
                        latent.shape[1],
                        entry["latent_file_path"],
                    )
                )

            self.latent_sequences.append(latent)
            if latent.shape[0] < self.sample_length:
                continue

            last_start_index = latent.shape[0] - self.sample_length
            start_indices = list(range(0, last_start_index + 1, self.window_stride))
            if start_indices[-1] != last_start_index:
                start_indices.append(last_start_index)

            for start_index in start_indices:
                self.sample_locations.append((entry_index, start_index))

        if self.latent_width is None:
            raise ValueError("No latent files were loaded")
        if len(self.sample_locations) == 0:
            raise ValueError(
                "No valid latent windows found for sample_length {}".format(
                    self.sample_length
                )
            )

    def __len__(self):
        return len(self.sample_locations)

    def __getitem__(self, index):
        entry_index, start_index = self.sample_locations[index]
        latent_window = self.latent_sequences[entry_index][
            start_index : start_index + self.sample_length
        ]
        return torch.from_numpy(np.array(latent_window, dtype=np.float32, copy=True))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_manifest_path(latent_folder="", manifest_path=""):
    if manifest_path != "":
        return os.path.abspath(manifest_path)
    if latent_folder == "":
        return ""

    candidate_path = os.path.join(os.path.abspath(latent_folder), DEFAULT_MANIFEST_NAME)
    if os.path.exists(candidate_path):
        return candidate_path
    return ""


def load_latent_context(
    latent_folder="",
    manifest_path="",
    data_path="",
    model_path="",
):
    resolved_manifest_path = resolve_manifest_path(latent_folder, manifest_path)
    normalized_latent_folder = (
        os.path.abspath(latent_folder) if latent_folder != "" else ""
    )
    if normalized_latent_folder == "" and resolved_manifest_path != "":
        normalized_latent_folder = os.path.dirname(resolved_manifest_path)

    manifest = None
    if resolved_manifest_path != "" and os.path.exists(resolved_manifest_path):
        manifest = load_json(resolved_manifest_path)

    resolved_data_path = data_path
    resolved_model_path = model_path
    if manifest is not None:
        if resolved_data_path == "":
            resolved_data_path = manifest.get("data_path", "")
        if resolved_model_path == "":
            resolved_model_path = manifest.get("generator_path", "")

    if resolved_data_path != "":
        resolved_data_path = os.path.abspath(resolved_data_path)
    if resolved_model_path != "":
        resolved_model_path = os.path.abspath(resolved_model_path)

    return LatentContext(
        latent_folder=normalized_latent_folder,
        manifest_path=resolved_manifest_path,
        manifest=manifest,
        data_path=resolved_data_path,
        model_path=resolved_model_path,
    )


def _normalize_entry(file_entry, latent_folder=""):
    source_split = file_entry.get("source_split", "")
    source_file_name = file_entry.get("source_file_name", "")
    source_relative_path = file_entry.get("source_relative_path", "")
    latent_file_name = file_entry.get("latent_file_name", "")
    latent_relative_path = file_entry.get("latent_relative_path", "")
    latent_file_path = file_entry.get("latent_file_path", "")

    if source_relative_path == "" and source_split != "" and source_file_name != "":
        source_relative_path = os.path.join(source_split, source_file_name)
    if source_file_name == "" and source_relative_path != "":
        source_file_name = os.path.basename(source_relative_path)
    if latent_file_name == "" and latent_relative_path != "":
        latent_file_name = os.path.basename(latent_relative_path)
    if latent_file_name == "" and latent_file_path != "":
        latent_file_name = os.path.basename(latent_file_path)
    if latent_relative_path == "" and latent_file_name != "" and source_split != "":
        latent_relative_path = os.path.join(source_split, latent_file_name)
    if latent_file_path == "" and latent_relative_path != "" and latent_folder != "":
        latent_file_path = os.path.join(latent_folder, latent_relative_path)

    normalized_entry = dict(file_entry)
    normalized_entry["source_split"] = source_split
    normalized_entry["source_file_name"] = source_file_name
    normalized_entry["source_relative_path"] = source_relative_path.replace("\\", "/")
    normalized_entry["latent_file_name"] = latent_file_name
    normalized_entry["latent_relative_path"] = latent_relative_path.replace("\\", "/")
    normalized_entry["latent_file_path"] = os.path.abspath(latent_file_path)
    return normalized_entry


def load_latent_entries(latent_context, split="all", max_files=0):
    if latent_context.manifest is not None:
        selected_entries = []
        for file_entry in latent_context.manifest.get("files", []):
            if split != "all" and file_entry.get("source_split") != split:
                continue
            selected_entries.append(
                _normalize_entry(file_entry, latent_context.latent_folder)
            )
            if max_files > 0 and len(selected_entries) >= max_files:
                break
        return selected_entries

    if latent_context.latent_folder == "":
        raise FileNotFoundError(
            "A latent folder is required when latent_dataset_metadata.json is unavailable"
        )

    file_records = collect_latent_file_records(
        latent_context.latent_folder,
        split=split,
        max_files=max_files,
    )
    selected_entries = []
    for file_record in file_records:
        selected_entries.append(
            {
                "source_split": file_record.split_name,
                "source_file_name": file_record.source_file_name,
                "source_relative_path": file_record.source_relative_path.replace(
                    "\\", "/"
                ),
                "latent_file_name": file_record.filename,
                "latent_relative_path": file_record.latent_relative_path.replace(
                    "\\", "/"
                ),
                "latent_file_path": os.path.abspath(file_record.file_path),
            }
        )
    return selected_entries


def infer_latent_width(latent_context, latent_entries):
    if latent_context.manifest is not None:
        manifest_width = latent_context.manifest.get("expected_latent_width")
        if manifest_width is not None:
            return int(manifest_width)

    if len(latent_entries) == 0:
        raise ValueError("No latent entries were available to infer latent width")

    latent = np.load(latent_entries[0]["latent_file_path"])
    if latent.ndim != 2:
        raise ValueError(
            "Expected latent array [T, C] in {}, got {}".format(
                latent_entries[0]["latent_file_path"],
                latent.shape,
            )
        )
    return int(latent.shape[1])


def resolve_checkpoint_path(read_weight_path):
    normalized_path = os.path.abspath(read_weight_path)
    if os.path.isfile(normalized_path):
        return normalized_path

    if os.path.isdir(normalized_path):
        checkpoint_files = []
        for file_name in sorted(os.listdir(normalized_path)):
            if file_name.endswith(CHECKPOINT_EXTENSIONS):
                checkpoint_files.append(file_name)
        if len(checkpoint_files) == 0:
            raise FileNotFoundError(
                "No checkpoint files were found under {}".format(normalized_path)
            )
        return os.path.join(normalized_path, checkpoint_files[-1])

    raise FileNotFoundError(
        "Checkpoint path does not exist: {}".format(normalized_path)
    )


def load_lstm_checkpoint(read_weight_path, device):
    resolved_path = resolve_checkpoint_path(read_weight_path)
    checkpoint_payload = torch.load(resolved_path, map_location=device)
    if (
        isinstance(checkpoint_payload, dict)
        and "model_state_dict" in checkpoint_payload
    ):
        return resolved_path, checkpoint_payload
    if isinstance(checkpoint_payload, dict):
        return resolved_path, {"model_state_dict": checkpoint_payload}
    raise ValueError("Unsupported checkpoint format in {}".format(resolved_path))


def build_lstm_checkpoint_payload(
    model,
    optimizer,
    iteration,
    train_config,
    latent_context,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
        "train_config": dict(train_config),
        "latent_context": {
            "latent_folder": latent_context.latent_folder,
            "manifest_path": latent_context.manifest_path,
            "data_path": latent_context.data_path,
            "model_path": latent_context.model_path,
        },
    }
    if latent_context.manifest is not None:
        payload["latent_manifest"] = {
            "expected_latent_width": latent_context.manifest.get(
                "expected_latent_width"
            ),
            "downsampling_factor": latent_context.manifest.get("downsampling_factor"),
            "autoencoder_module": latent_context.manifest.get("autoencoder_module"),
            "use_vae": latent_context.manifest.get("use_vae"),
        }
    return payload


def create_autoencoder_runtime(latent_context, device, load_ik=False):
    if latent_context.data_path == "" or latent_context.model_path == "":
        raise FileNotFoundError(
            "Autoencoder decode requires --data_path and --model_path, or a latent manifest that contains both values"
        )
    return create_runtime(
        latent_context.data_path,
        latent_context.model_path,
        device,
        load_ik=load_ik,
    )


def build_runtime_dataset_caches(runtime, split_names):
    dataset_cache = {}
    dataset_index_cache = {}
    for split_name in sorted(set(split_names)):
        dataset, file_records = build_test_dataset(runtime, split=split_name)
        dataset_cache[split_name] = dataset
        dataset_index_cache[split_name] = {
            file_record.source_relative_path.replace("\\", "/"): index
            for index, file_record in enumerate(file_records)
        }
    return dataset_cache, dataset_index_cache


def decode_latent_sequence_for_entry(
    runtime,
    dataset_cache,
    dataset_index_cache,
    latent_entry,
    latent_sequence,
    output_filename,
    output_dir,
):
    source_split = latent_entry["source_split"]
    source_relative_path = latent_entry["source_relative_path"].replace("\\", "/")
    dataset = dataset_cache[source_split]
    if source_relative_path not in dataset_index_cache[source_split]:
        raise KeyError(
            "Could not resolve dataset index for {}".format(source_relative_path)
        )
    dataset_index = dataset_index_cache[source_split][source_relative_path]
    return decode_dataset_item_to_bvh(
        runtime,
        dataset,
        dataset_index,
        latent_sequence,
        output_filename,
        output_dir,
    )


def choose_seed_entries(latent_entries, seed_file="", num_samples=1, random_seed=1234):
    if len(latent_entries) == 0:
        raise ValueError("No latent entries matched the requested split")

    if seed_file != "":
        normalized_seed = seed_file.replace("\\", "/")
        matched_entries = []
        for entry in latent_entries:
            source_name = entry["source_file_name"]
            latent_name = entry["latent_file_name"]
            candidate_names = {
                entry["source_relative_path"],
                entry["latent_relative_path"],
                source_name,
                latent_name,
                os.path.splitext(source_name)[0],
                os.path.splitext(latent_name)[0],
            }
            if normalized_seed in candidate_names:
                matched_entries.append(entry)

        if len(matched_entries) == 0:
            raise FileNotFoundError(
                "Could not find a latent seed matching {}".format(seed_file)
            )
        return matched_entries[:1]

    rng = random.Random(random_seed)
    sample_count = min(int(num_samples), len(latent_entries))
    return rng.sample(latent_entries, sample_count)


def build_generation_metadata(
    latent_entry,
    checkpoint_path,
    initial_seq_len,
    generate_latent_steps,
    seed_start_index,
    runtime=None,
):
    metadata = {
        "source_split": latent_entry["source_split"],
        "source_file_name": latent_entry["source_file_name"],
        "source_relative_path": latent_entry["source_relative_path"],
        "latent_file_path": latent_entry["latent_file_path"],
        "checkpoint_path": checkpoint_path,
        "initial_seq_len": int(initial_seq_len),
        "generate_latent_steps": int(generate_latent_steps),
        "seed_start_index": int(seed_start_index),
    }
    if runtime is not None:
        metadata["runtime_summary"] = summarize_runtime(runtime)
    return metadata
