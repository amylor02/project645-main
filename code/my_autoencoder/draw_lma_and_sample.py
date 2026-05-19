import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from motion_data import LMA_FRAME_STRIDE, LMA_KEYS
from sample_latent_diffusion import (
    _coerce_lma_array,
    _resample_columns,
    infer_annotation_split,
    list_split_bvhs,
    resolve_source_split,
    run_sampling,
)
from train_latent_diffusion import load_lma_annotation, seed_all
from train_vq_vae import get_bvh_from_disk, get_info_from_bvh
from visualizer import visualize_motion_lma_comparison


class LMAFreehandEditor:
    def __init__(
        self,
        source_name: str,
        values: np.ndarray,
        channel_names,
        value_min=None,
        value_max=None,
    ):
        self.source_name = source_name
        self.channel_names = list(channel_names)
        self.original = np.asarray(values, dtype=np.float32).copy()
        self.edited = self.original.copy()
        self.num_steps = self.edited.shape[0]
        self.accepted = False
        self.last_channel_index = None
        self._drawing = False
        self._drawing_channel = None
        self._last_step = None
        self._last_value = None
        self.value_min = 0.0 if value_min is None else float(value_min)
        self.value_max = 1.0 if value_max is None else float(value_max)

        self.fig, self.axes = plt.subplots(
            len(self.channel_names),
            1,
            figsize=(14, max(8, 2 * len(self.channel_names))),
            sharex=True,
        )
        if not isinstance(self.axes, np.ndarray):
            self.axes = np.array([self.axes])

        self.original_lines = []
        self.edited_lines = []
        self.value_limits = []
        x = np.arange(self.num_steps)
        for channel_index, axis in enumerate(self.axes):
            self.value_limits.append((self.value_min, self.value_max))

            (original_line,) = axis.plot(
                x, self.original[:, channel_index], "--", color="0.55", linewidth=1.0
            )
            (edited_line,) = axis.plot(
                x, self.edited[:, channel_index], color="C0", linewidth=2.0
            )
            self.original_lines.append(original_line)
            self.edited_lines.append(edited_line)
            axis.set_ylabel(self.channel_names[channel_index])
            axis.set_ylim(self.value_min, self.value_max)
            axis.set_yticks([self.value_min, self.value_max])
            axis.grid(True, alpha=0.3)

        self.axes[-1].set_xlabel(
            f"LMA timestep (1 step = {LMA_FRAME_STRIDE} motion frames)"
        )
        self.status = self.fig.text(
            0.01,
            0.01,
            "Left-drag to draw. Enter = sample/save. r = reset channel. R = reset all. q = quit.",
        )
        self.fig.suptitle(f"Draw LMA Conditions: {source_name}")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        self.cid_press = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cid_release = self.fig.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cid_motion = self.fig.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def set_status(self, text: str):
        self.status.set_text(text)
        self.fig.canvas.draw_idle()

    def axis_to_channel(self, axis):
        for index, candidate in enumerate(self.axes):
            if candidate is axis:
                return index
        return None

    def clamp_value(self, channel_index: int, value: float) -> float:
        lo, hi = self.value_limits[channel_index]
        return float(np.clip(value, lo, hi))

    def clamp_step(self, x_value: float) -> int:
        if x_value is None:
            return 0
        return int(np.clip(int(round(float(x_value))), 0, self.num_steps - 1))

    def draw_segment(self, channel_index: int, step: int, value: float):
        value = self.clamp_value(channel_index, value)
        if (
            self._last_step is None
            or self._last_value is None
            or self._last_step == step
        ):
            self.edited[step, channel_index] = value
        else:
            start_step = self._last_step
            end_step = step
            start_value = self._last_value
            if end_step < start_step:
                start_step, end_step = end_step, start_step
                start_value, value = value, start_value
            interp_steps = np.arange(start_step, end_step + 1)
            interp_values = np.linspace(
                start_value, value, len(interp_steps), dtype=np.float32
            )
            self.edited[interp_steps, channel_index] = interp_values

        self.edited_lines[channel_index].set_ydata(self.edited[:, channel_index])
        self.fig.canvas.draw_idle()
        self._last_step = step
        self._last_value = value

    def reset_channel(self, channel_index: int):
        self.edited[:, channel_index] = self.original[:, channel_index]
        self.edited_lines[channel_index].set_ydata(self.edited[:, channel_index])
        self.fig.canvas.draw_idle()
        self.set_status(f"Reset channel: {self.channel_names[channel_index]}")

    def reset_all(self):
        self.edited[:, :] = self.original
        for channel_index, line in enumerate(self.edited_lines):
            line.set_ydata(self.edited[:, channel_index])
        self.fig.canvas.draw_idle()
        self.set_status("Reset all LMA channels")

    def on_press(self, event):
        channel_index = self.axis_to_channel(event.inaxes)
        if channel_index is None:
            return
        self.last_channel_index = channel_index
        if event.button == 3:
            self.reset_channel(channel_index)
            return
        if event.button != 1:
            return

        self._drawing = True
        self._drawing_channel = channel_index
        self._last_step = None
        self._last_value = None
        step = self.clamp_step(event.xdata)
        value = self.clamp_value(
            channel_index, event.ydata if event.ydata is not None else 0.0
        )
        self.draw_segment(channel_index, step, value)
        self.set_status(f"Drawing channel: {self.channel_names[channel_index]}")

    def on_motion(self, event):
        channel_index = self.axis_to_channel(event.inaxes)
        if channel_index is not None:
            self.last_channel_index = channel_index
        if not self._drawing or channel_index != self._drawing_channel:
            return
        step = self.clamp_step(event.xdata)
        value = self.clamp_value(
            channel_index,
            event.ydata if event.ydata is not None else self._last_value or 0.0,
        )
        self.draw_segment(channel_index, step, value)

    def on_release(self, event):
        self._drawing = False
        self._drawing_channel = None
        self._last_step = None
        self._last_value = None

    def on_key(self, event):
        if event.key in {"enter", "return"}:
            self.accepted = True
            self.set_status("Accepting edited LMA and closing editor")
            plt.close(self.fig)
            return
        if event.key in {"escape", "q"}:
            self.accepted = False
            self.set_status("Closing without sampling")
            plt.close(self.fig)
            return
        if event.key == "R":
            self.reset_all()
            return
        if event.key == "r":
            if self.last_channel_index is not None:
                self.reset_channel(self.last_channel_index)
            return

    def show(self):
        plt.show(block=True)
        return self.accepted, self.edited.copy()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw per-channel LMA signals and optionally sample with them"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Dataset root containing train/, eval/, and annotations/",
    )
    parser.add_argument(
        "--vae-model-path",
        required=True,
        help="Path to pretrained VAE generator.pt or its containing directory",
    )
    parser.add_argument(
        "--diffusion-model-path",
        required=True,
        help="Path to latent_diffusion_prior.pt or latent_diffusion_prior_best.pt",
    )
    parser.add_argument(
        "--source-split",
        choices=["auto", "train", "eval"],
        default="auto",
        help="Dataset split used when --bvh-path is not provided. Use auto to infer from --bvh-path.",
    )
    parser.add_argument(
        "--clip-index", type=int, default=0, help="Clip index inside the selected split"
    )
    parser.add_argument(
        "--eval-index",
        dest="clip_index",
        type=int,
        help="Backward-compatible alias for --clip-index",
    )
    parser.add_argument(
        "--bvh-path",
        default=None,
        help="Optional direct BVH path. If provided, it overrides split/index selection",
    )
    parser.add_argument(
        "--mode", choices=["full", "lma_only", "traj_only", "uncond"], default="full"
    )
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--lma-cfg-scale", type=float, default=None)
    parser.add_argument("--traj-cfg-scale", type=float, default=None)
    parser.add_argument("--style-cfg-scale", type=float, default=None)
    parser.add_argument(
        "--full-mode-lma-prefix",
        action="store_true",
        help="When mode=full, use lma_only guidance for the first denoising steps and switch to full guidance for the remainder.",
    )
    parser.add_argument(
        "--full-mode-lma-prefix-fraction",
        type=float,
        default=0.5,
        help="Fraction of denoising steps that use lma_only before switching back to full when --full-mode-lma-prefix is enabled.",
    )
    parser.add_argument(
        "--full-mode-lma-suffix",
        action="store_true",
        help="When mode=full, use full guidance first and switch to lma_only guidance for the final denoising steps.",
    )
    parser.add_argument(
        "--full-mode-lma-suffix-fraction",
        type=float,
        default=0.5,
        help="Fraction of final denoising steps that use lma_only when --full-mode-lma-suffix is enabled.",
    )
    parser.add_argument("--style-label", default=None)
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--chunk-len", type=int, default=None)
    parser.add_argument("--overlap-len", type=int, default=None)
    parser.add_argument("--halo-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--latent-target", choices=["mean", "sample"], default="mean")
    parser.add_argument(
        "--source-latent-edit",
        action="store_true",
        help="Initialize sampling from a partially noised source latent instead of pure noise",
    )
    parser.add_argument(
        "--edit-strength",
        type=float,
        default=None,
        help="Source-latent edit strength in [0, 1]. Lower stays closer to the source clip.",
    )
    parser.add_argument(
        "--source-latent-noise-timestep",
        type=int,
        default=None,
        help="Advanced override for the source-latent forward-noise timestep",
    )
    parser.add_argument("--frame-len", type=int, default=None)
    parser.add_argument("--root-override-blend", type=float, default=1.0)
    parser.add_argument("--disable-predicted-root-override", action="store_true")
    parser.add_argument("--disable-foot-grounding", action="store_true")
    parser.add_argument("--foot-grounding-strength", type=float, default=1.0)
    parser.add_argument("--save-bvh", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--edited-lma-output",
        default=None,
        help="Optional path for the edited LMA CSV. Defaults next to the sample outputs",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Only save the edited LMA, do not run sampling",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Do not open the motion-and-LMA comparison visualizer after sampling",
    )
    parser.add_argument(
        "--comparison-mp4-path",
        default=None,
        help="Optional MP4 path for saving the synchronized comparison video at 30fps",
    )
    parser.add_argument(
        "--value-min",
        type=float,
        default=None,
        help="Optional fixed minimum draw value for all channels",
    )
    parser.add_argument(
        "--value-max",
        type=float,
        default=None,
        help="Optional fixed maximum draw value for all channels",
    )
    parser.add_argument("--seed", type=int, default=2222)
    return parser.parse_args()


def resolve_source_clip(args):
    if args.bvh_path is not None:
        bvh_file = Path(args.bvh_path)
        if not bvh_file.exists():
            raise FileNotFoundError(f"BVH file does not exist: {bvh_file}")
        annotation_split = infer_annotation_split(
            args.data_path, args.source_split, bvh_file
        )
        return bvh_file, annotation_split

    resolved_split = resolve_source_split(args.source_split)
    split_files = list_split_bvhs(args.data_path, resolved_split)
    if not split_files:
        raise RuntimeError(
            f"No BVH files found under {Path(args.data_path) / resolved_split}"
        )
    if args.clip_index < 0 or args.clip_index >= len(split_files):
        raise IndexError(
            f"clip-index must be in [0, {len(split_files) - 1}], got {args.clip_index}"
        )
    bvh_file = Path(args.data_path) / resolved_split / split_files[args.clip_index]
    return bvh_file, resolved_split


def load_source_lma(args):
    bvh_file, annotation_split = resolve_source_clip(args)
    bvh = get_bvh_from_disk(str(bvh_file.parent), bvh_file.name)
    _, pos, _, _, _, _ = get_info_from_bvh(bvh, get_missing_frames=False)
    num_frames = int(pos.shape[0])
    target_len = max((num_frames + LMA_FRAME_STRIDE - 1) // LMA_FRAME_STRIDE, 1)
    raw_lma = load_lma_annotation(Path(args.data_path), annotation_split, bvh_file.name)
    if raw_lma is None:
        raw_lma = np.zeros((target_len, len(LMA_KEYS)), dtype=np.float32)
    raw_lma = _coerce_lma_array(raw_lma)
    raw_lma = _resample_columns(raw_lma, target_len)
    return bvh_file, annotation_split, raw_lma


def save_edited_lma(output_path: Path, values: np.ndarray):
    dataframe = pd.DataFrame(values, columns=list(LMA_KEYS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def main():
    args = parse_args()
    seed_all(args.seed)

    bvh_file, annotation_split, raw_lma = load_source_lma(args)
    print(f"Loaded source clip: {bvh_file}")
    print(f"Annotation split: {annotation_split}")
    print(f"LMA shape: {raw_lma.shape}")

    editor = LMAFreehandEditor(
        source_name=bvh_file.name,
        values=raw_lma,
        channel_names=LMA_KEYS,
        value_min=args.value_min,
        value_max=args.value_max,
    )
    accepted, edited_lma = editor.show()
    if not accepted:
        print("Editor closed without sampling")
        return

    default_output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(args.diffusion_model_path).resolve().parent
    )
    edited_output_path = (
        Path(args.edited_lma_output)
        if args.edited_lma_output is not None
        else default_output_dir / f"edited_lma_{bvh_file.stem}.csv"
    )
    save_edited_lma(edited_output_path, edited_lma)
    print(f"Saved edited LMA to {edited_output_path}")

    if args.no_sample:
        return

    print("Running sampler with the edited LMA override")
    sample_result = run_sampling(args, lma_override_array=edited_lma)

    if args.no_comparison:
        return

    comparison_title = f"Original vs Edited Sample: {sample_result['source_filename']}"
    if args.comparison_mp4_path is not None:
        print(f"Saving comparison MP4 to {args.comparison_mp4_path}")

    print("Opening original-vs-edited comparison visualizer")
    visualize_motion_lma_comparison(
        sample_result["source_joint_positions"],
        sample_result["sampled_joint_positions"],
        raw_lma,
        edited_lma,
        parents=sample_result.get("parents"),
        fps=30,
        title=comparison_title,
        save_path=args.comparison_mp4_path,
        block=True,
    )


if __name__ == "__main__":
    main()
