"""
Interactive motion + LMA signal visualizer.

Layout
------
Left  : 3D skeleton player (live bone rendering)
Right : Stacked per-signal strips — LMA group on top, ctrl group below
Bottom: Frame slider + Play/Pause button

Usage
-----
    from visualizer import visualize_motion_and_tags
    visualize_motion_and_tags(pos, tags, parents=parents)

Called automatically from compute_tags() when visualize=True.
"""

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ---------------------------------------------------------------------------
# Signal metadata
# ---------------------------------------------------------------------------

_LMA_KEYS = [
    "BODY",
    "EFFORT_WEIGHT_STRONG",
    "EFFORT_TIME_SUDDEN",
    "EFFORT_FLOW_BOUND",
    "SHAPE",
    "SPACE",
]

_LMA_LABELS = {
    "BODY": "Body",
    "EFFORT_WEIGHT_STRONG": "Effort: Weight",
    "EFFORT_TIME_SUDDEN": "Effort: Time",
    "EFFORT_FLOW_BOUND": "Effort: Flow",
    "SHAPE": "Shape",
    "SPACE": "Space",
}

_CTRL_KEYS = [
    "ctrl_velocity",
    "ctrl_forward_alignment",
    "ctrl_lateral_alignment",
    "ctrl_acceleration",
    "ctrl_height",
    "ctrl_vertical_velocity",
    "ctrl_yaw_rate",
    "yaw_sin",
    "yaw_cos",
    "ctrl_head_height",
]

_CTRL_LABELS = {
    "ctrl_velocity": "Velocity",
    "ctrl_forward_alignment": "Fwd Align",
    "ctrl_lateral_alignment": "Lat Align",
    "ctrl_acceleration": "Acceleration",
    "ctrl_height": "Height",
    "ctrl_vertical_velocity": "Vert. Velocity",
    "ctrl_yaw_rate": "Yaw Rate",
    "yaw_sin": "Yaw sin",
    "yaw_cos": "Yaw cos",
    "ctrl_head_height": "Head Height",
}

_CONTACT_LABELS = [
    "L Ankle Contact",
    "L Foot Contact",
    "R Ankle Contact",
    "R Foot Contact",
]

# LMA gets blue tones, ctrl gets amber/green tones
_LMA_COLORS = [
    "#2176ae",
    "#3a8fca",
    "#5ba3d9",
    "#1a5f8a",
    "#4e9ec4",
    "#7ab8d9",
]
_CONTACT_COLORS = ["#2f7d32", "#4ba64f", "#7fbf3f", "#1f5e2f"]
_CTRL_COLORS = [
    "#c9511f",
    "#e07b39",
    "#d4943e",
    "#7a9e3b",
    "#5b8a2d",
    "#b85c22",
    "#a04c1a",
    "#6b4c9a",
]

_LMA_BG = "#f0f5fb"
_CONTACT_BG = "#eef7ea"
_CTRL_BG = "#fdf4ec"

_JOINT_DEFAULT_COLOR = "#66aaff"
_JOINT_PLANTED_COLOR = "#ff4d5a"
_COMPARE_ORIGINAL_COLOR = "#3a8fca"
_COMPARE_EDITED_COLOR = "#c62b1d"
_COMPARE_ORIGINAL_BONE = "#9cc7ee"
_COMPARE_EDITED_BONE = "#fe655d"
_STATUS_COLOR = "#ccddff"
_STATUS_ERROR_COLOR = "#ff9aa6"
_AXIS_TEXT_COLOR = "#ffffff"
_SIGNAL_LABEL_SIZE = 8.0
_SIGNAL_TICK_SIZE = 6.5
_THREED_LABEL_SIZE = 7.5
_THREED_TICK_SIZE = 6.0
_THREED_TITLE_SIZE = 9.5
_DEFAULT_VIEW_ELEV = 18
_DEFAULT_VIEW_AZIM = -55
_CAMERA_AZIM_STEP = 8
_CAMERA_ELEV_STEP = 5
_COMPARISON_SIGNAL_GAP = 0.010
_EXPORT_DPI = 140
_EXPORT_PRIMARY_PROFILE = {
    "label": "h264",
    "codec": "h264",
    "bitrate": 1800,
    "extra_args": [
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ],
}
_EXPORT_FALLBACK_PROFILE = {
    "label": "mpeg4",
    "codec": "mpeg4",
    "bitrate": 1800,
    "extra_args": [
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_np(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32)


def _slugify_filename(text):
    text = "" if text is None else str(text)
    cleaned = []
    for char in text:
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        elif char in (" ", "."):
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or "motion_visualizer"


def _default_mp4_name(title):
    return f"{_slugify_filename(title)}.mp4"


def _normalize_mp4_output_path(output_path, title):
    path = Path(output_path).expanduser()
    if path.exists() and path.is_dir():
        return path / _default_mp4_name(title)
    if path.suffix.lower() == ".mp4":
        return path
    if path.suffix:
        return path.with_suffix(".mp4")
    return path / _default_mp4_name(title)


def _choose_mp4_save_path(initial_path, title):
    initial_path = _normalize_mp4_output_path(initial_path, title)
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen_path = filedialog.asksaveasfilename(
            title="Save MP4",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
            initialdir=str(initial_path.parent),
            initialfile=initial_path.name,
        )
        root.destroy()
        if not chosen_path:
            return None
        return _normalize_mp4_output_path(chosen_path, title)
    except Exception:
        return initial_path


def _create_ffmpeg_writer(fps, profile):
    return animation.FFMpegWriter(
        fps=fps,
        codec=profile["codec"],
        bitrate=profile["bitrate"],
        extra_args=list(profile["extra_args"]),
    )


def _save_with_ffmpeg_profiles(movie, output_path, fps, dpi):
    last_error = None
    for profile in (_EXPORT_PRIMARY_PROFILE, _EXPORT_FALLBACK_PROFILE):
        writer = _create_ffmpeg_writer(fps, profile)
        try:
            movie.save(str(output_path), writer=writer, dpi=dpi)
            return profile["label"]
        except Exception as exc:
            last_error = exc
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass

    raise RuntimeError(
        "MP4 export failed for all codec profiles. " f"Last error: {last_error}"
    ) from last_error


def _resample_to(arr, target_len):
    """Linearly resample a 1-D array to target_len samples."""
    arr = _to_np(arr).ravel()
    n = len(arr)
    if n == target_len:
        return arr
    if n == 0:
        return np.zeros(target_len, dtype=np.float32)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst, src, arr).astype(np.float32)


def _get_signal_axis_range(group, *arrays):
    if group == "lma":
        return 0.0, 1.0

    finite_arrays = [
        np.asarray(array, dtype=np.float32).reshape(-1) for array in arrays
    ]
    finite_arrays = [
        array[np.isfinite(array)] for array in finite_arrays if array.size > 0
    ]
    if not finite_arrays:
        return 0.0, 1.0

    lo = float(min(np.min(array) for array in finite_arrays))
    hi = float(max(np.max(array) for array in finite_arrays))
    pad = max((hi - lo) * 0.12, 0.05)
    return lo - pad, hi + pad


def _get_signal_tick_labels(group, lo, hi):
    if group == "lma":
        return [0.0, 1.0], ["0.00", "1.00"]
    return [lo, hi], [f"{lo:.2f}", f"{hi:.2f}"]


def _style_signal_axis(ax, label, show_xlabel):
    ax.set_ylabel(
        label,
        fontsize=_SIGNAL_LABEL_SIZE,
        rotation=0,
        labelpad=68,
        ha="right",
        va="center",
        color=_AXIS_TEXT_COLOR,
    )
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis="x", labelsize=_SIGNAL_TICK_SIZE, colors=_AXIS_TEXT_COLOR)
    ax.tick_params(axis="y", labelsize=_SIGNAL_TICK_SIZE, colors=_AXIS_TEXT_COLOR)
    if show_xlabel:
        ax.set_xlabel("Frame", fontsize=_SIGNAL_LABEL_SIZE, color=_AXIS_TEXT_COLOR)
    else:
        ax.set_xticklabels([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#d0d0d0")


def _style_3d_axis(ax, title):
    ax.set_xlabel("X", fontsize=_THREED_LABEL_SIZE, labelpad=-3, color=_AXIS_TEXT_COLOR)
    ax.set_ylabel("Z", fontsize=_THREED_LABEL_SIZE, labelpad=-3, color=_AXIS_TEXT_COLOR)
    ax.set_zlabel(
        "Y (up)",
        fontsize=_THREED_LABEL_SIZE,
        labelpad=-3,
        color=_AXIS_TEXT_COLOR,
    )
    ax.set_title(title, fontsize=_THREED_TITLE_SIZE, pad=5, color=_AXIS_TEXT_COLOR)
    ax.tick_params(labelsize=_THREED_TICK_SIZE, colors=_AXIS_TEXT_COLOR)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.label.set_color(_AXIS_TEXT_COLOR)
        except Exception:
            pass
        try:
            axis.set_tick_params(colors=_AXIS_TEXT_COLOR)
        except Exception:
            pass


def _resample_motion_to(pos, target_len):
    pos = _to_np(pos)
    if pos.ndim != 3:
        raise ValueError(f"Expected [F, J, 3] motion array, got shape {pos.shape}")

    frame_count, joint_count, dims = pos.shape
    if dims != 3:
        raise ValueError(f"Expected 3 motion coordinates, got shape {pos.shape}")
    if frame_count == target_len:
        return pos
    if frame_count == 0:
        return np.zeros((target_len, joint_count, 3), dtype=np.float32)
    if frame_count == 1:
        return np.repeat(pos, target_len, axis=0).astype(np.float32)

    src = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    flat = pos.reshape(frame_count, -1)
    resampled = np.stack(
        [np.interp(dst, src, flat[:, column]) for column in range(flat.shape[1])],
        axis=1,
    )
    return resampled.reshape(target_len, joint_count, 3).astype(np.float32)


def _coerce_lma_signal_map(values):
    if values is None:
        return {}

    if isinstance(values, dict):
        signals = {}
        for key in _LMA_KEYS:
            if key in values:
                signals[key] = _to_np(values[key]).reshape(-1)
        return signals

    array = _to_np(values)
    if array.ndim == 1:
        if array.size % len(_LMA_KEYS) != 0:
            raise ValueError(
                f"1D LMA array length must be divisible by {len(_LMA_KEYS)}, got {array.size}"
            )
        array = array.reshape(-1, len(_LMA_KEYS))
    if array.ndim != 2:
        raise ValueError(f"Expected [T, C] LMA array, got shape {array.shape}")

    if array.shape[1] < len(_LMA_KEYS):
        padded = np.zeros((array.shape[0], len(_LMA_KEYS)), dtype=np.float32)
        padded[:, : array.shape[1]] = array
        array = padded
    elif array.shape[1] > len(_LMA_KEYS):
        array = array[:, : len(_LMA_KEYS)]

    return {
        key: array[:, index].astype(np.float32) for index, key in enumerate(_LMA_KEYS)
    }


def _build_lma_comparison_rows(original_lma, edited_lma, num_frames):
    original_map = _coerce_lma_signal_map(original_lma)
    edited_map = _coerce_lma_signal_map(edited_lma)
    rows = []

    for key in _LMA_KEYS:
        if key not in original_map and key not in edited_map:
            continue
        original = _resample_to(
            original_map.get(key, np.zeros(0, dtype=np.float32)), num_frames
        )
        edited = _resample_to(
            edited_map.get(key, np.zeros(0, dtype=np.float32)), num_frames
        )
        rows.append((key, _LMA_LABELS.get(key, key), original, edited))

    return rows


def _build_planted_joint_mask(tags, num_frames, num_joints):
    contact_binary = tags.get("foot_contact_binary") if isinstance(tags, dict) else None
    joint_indices = (
        tags.get("foot_contact_joint_indices") if isinstance(tags, dict) else None
    )
    if contact_binary is None or joint_indices is None:
        return None

    contact_binary = _to_np(contact_binary)
    if contact_binary.ndim == 1:
        contact_binary = contact_binary[:, None]

    joint_indices = np.asarray(joint_indices, dtype=np.int64).ravel()
    if contact_binary.size == 0 or joint_indices.size == 0:
        return None

    mask = np.zeros((num_frames, num_joints), dtype=bool)
    num_channels = min(contact_binary.shape[1], joint_indices.size)
    for channel_index in range(num_channels):
        joint_index = int(joint_indices[channel_index])
        if 0 <= joint_index < num_joints:
            planted = _resample_to(contact_binary[:, channel_index], num_frames) >= 0.5
            mask[:, joint_index] = planted

    return mask


def _build_signal_rows(tags, num_frames):
    """
    Return ordered list of (key, group, label, color, data_np).
    All data_np are resampled to num_frames.
    LMA signals first, contact signals second, ctrl signals last.
    """
    rows = []
    for i, key in enumerate(_LMA_KEYS):
        if key in tags:
            data = _resample_to(tags[key], num_frames)
            color = _LMA_COLORS[i % len(_LMA_COLORS)]
            rows.append((key, "lma", _LMA_LABELS.get(key, key), color, data))

    if "foot_contact_binary" in tags:
        contact_data = _to_np(tags["foot_contact_binary"])
        if contact_data.ndim == 1:
            contact_data = contact_data[:, None]
        for index in range(min(contact_data.shape[1], len(_CONTACT_LABELS))):
            data = _resample_to(contact_data[:, index], num_frames)
            color = _CONTACT_COLORS[index % len(_CONTACT_COLORS)]
            rows.append(
                (
                    f"foot_contact_binary_{index}",
                    "contact",
                    _CONTACT_LABELS[index],
                    color,
                    data,
                )
            )

    for i, key in enumerate(_CTRL_KEYS):
        if key in tags:
            data = _resample_to(tags[key], num_frames)
            color = _CTRL_COLORS[i % len(_CTRL_COLORS)]
            rows.append((key, "ctrl", _CTRL_LABELS.get(key, key), color, data))

    return rows


# ---------------------------------------------------------------------------
# Main visualizer class
# ---------------------------------------------------------------------------


class MotionVisualizer:
    """
    Interactive 3D skeleton + signal strip player.

    Parameters
    ----------
    pos     : np.ndarray [frames, n_joints, 3]  global joint positions (Y-up)
    tags    : dict  output of compute_tags() — LMA + ctrl signals
    parents : list/array  parent index per joint (root → 0); optional
    fps     : playback speed
    title   : window title string
    """

    def __init__(self, pos, tags, parents=None, fps=30, title=None):
        self.pos = _to_np(pos)  # [F, J, 3]
        self.tags = tags
        self.parents = None if parents is None else list(int(p) for p in parents)
        self.fps = fps
        self.title = title or "Motion & LMA Visualizer"
        self.num_frames = self.pos.shape[0]

        self._frame = 0
        self._playing = False
        self._timer = None

        self._signal_rows = _build_signal_rows(tags, self.num_frames)
        self._joint_default_rgba = np.array(
            matplotlib.colors.to_rgba(_JOINT_DEFAULT_COLOR),
            dtype=np.float32,
        )
        self._joint_planted_rgba = np.array(
            matplotlib.colors.to_rgba(_JOINT_PLANTED_COLOR),
            dtype=np.float32,
        )
        self._planted_joint_mask = _build_planted_joint_mask(
            tags,
            self.num_frames,
            self.pos.shape[1],
        )
        self._build_figure()

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self):
        n_sig = max(len(self._signal_rows), 1)

        # Dynamic figure height: each signal row ~0.75 in, plus margins
        fig_h = max(8.0, n_sig * 0.78 + 1.8)
        self.fig = plt.figure(figsize=(18, fig_h))
        self.fig.suptitle(self.title, fontsize=10, fontweight="bold")
        self.fig.patch.set_facecolor("#1e1e2e")

        # --- Layout constants (figure fraction) ---
        LEFT_3D = 0.03
        W_3D = 0.40
        LEFT_SIG = 0.46
        W_SIG = 0.52
        TOP_MAIN = 0.93
        BOTTOM_MAIN = 0.09  # leave room for slider
        H_MAIN = TOP_MAIN - BOTTOM_MAIN

        # 3D skeleton axis (full height)
        self.ax3d = self.fig.add_axes(
            [LEFT_3D, BOTTOM_MAIN, W_3D, H_MAIN],
            projection="3d",
        )
        self._setup_3d_ax()

        # Signal strip axes (stacked top→bottom)
        self._sig_axes = []
        self._cursor_lines = []

        gap = 0.004  # thin gap between strips
        strip_h = (H_MAIN - gap * n_sig) / n_sig

        for i, (key, group, label, color, data) in enumerate(self._signal_rows):
            bottom = TOP_MAIN - (i + 1) * strip_h - i * gap
            ax = self.fig.add_axes([LEFT_SIG, bottom, W_SIG, strip_h])

            if group == "lma":
                bg = _LMA_BG
            elif group == "contact":
                bg = _CONTACT_BG
            else:
                bg = _CTRL_BG
            ax.set_facecolor(bg)
            ax.patch.set_alpha(0.85)

            ax.plot(data, color=color, linewidth=0.85, alpha=0.92)

            lo, hi = _get_signal_axis_range(group, data)
            yticks, yticklabels = _get_signal_tick_labels(group, lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlim(0, self.num_frames - 1)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            _style_signal_axis(ax, label, show_xlabel=i == n_sig - 1)

            if i > 0 and self._signal_rows[i - 1][1] != group:
                ax.spines["top"].set_color("#888")
                ax.spines["top"].set_linewidth(1.5)

            # Cursor vertical line
            vline = ax.axvline(x=0, color="#ff3355", linewidth=1.0, alpha=0.80)
            self._cursor_lines.append(vline)
            self._sig_axes.append(ax)

        # --- Slider ---
        self.ax_slider = self.fig.add_axes([0.12, 0.025, 0.72, 0.022])
        self.ax_slider.set_facecolor("#303050")
        self.slider = Slider(
            self.ax_slider,
            "Frame",
            valmin=0,
            valmax=self.num_frames - 1,
            valinit=0,
            valstep=1,
            color="#6688cc",
        )
        self.slider.label.set_fontsize(8)
        self.slider.valtext.set_fontsize(8)
        self.slider.on_changed(self._on_slider_change)

        # --- Play / Pause button ---
        self.ax_btn = self.fig.add_axes([0.01, 0.012, 0.085, 0.042])
        self.btn_play = Button(
            self.ax_btn, "▶  Play", color="#404060", hovercolor="#5566aa"
        )
        self.btn_play.label.set_fontsize(8)
        self.btn_play.on_clicked(self._on_play_toggle)

        self.ax_save_btn = self.fig.add_axes([0.86, 0.012, 0.11, 0.042])
        self.btn_save = Button(
            self.ax_save_btn,
            "Save MP4",
            color="#405040",
            hovercolor="#4f7c5f",
        )
        self.btn_save.label.set_fontsize(8)
        self.btn_save.on_clicked(self._on_save_mp4)

        self._status_text = self.fig.text(
            0.98,
            0.055,
            "Arrow keys or WASD rotate camera. Home resets view.",
            ha="right",
            va="center",
            fontsize=7,
            color=_STATUS_COLOR,
        )

        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Draw first frame
        self._update_frame(0)

    # ------------------------------------------------------------------
    # 3D skeleton setup
    # ------------------------------------------------------------------

    def _setup_3d_ax(self):
        ax = self.ax3d
        pos = self.pos  # [F, J, 3]

        # Compute display limits from full trajectory
        all_x = pos[:, :, 0].ravel()
        all_y = pos[:, :, 1].ravel()
        all_z = pos[:, :, 2].ravel()
        cx, cy, cz = all_x.mean(), all_y.mean(), all_z.mean()
        half = (
            max(
                (all_x.max() - all_x.min()),
                (all_y.max() - all_y.min()),
                (all_z.max() - all_z.min()),
                0.5,
            )
            * 0.6
        )

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cz - half, cz + half)  # Z → plot Y axis (depth)
        ax.set_zlim(cy - half, cy + half)  # Y → plot Z axis (height)

        _style_3d_axis(ax, "Skeleton Player")
        ax.view_init(elev=_DEFAULT_VIEW_ELEV, azim=_DEFAULT_VIEW_AZIM)

        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass  # older matplotlib

        ax.set_facecolor("#12121c")
        self.fig.patch.set_facecolor("#1e1e2e")

        # Faint root trajectory
        root_traj = pos[:, 0, :]
        ax.plot(
            root_traj[:, 0],
            root_traj[:, 2],
            root_traj[:, 1],
            color="#444466",
            linewidth=0.5,
            alpha=0.5,
            linestyle="--",
        )

        # Initial joint positions
        p0 = pos[0]
        self._joint_scatter = ax.scatter(
            p0[:, 0],
            p0[:, 2],
            p0[:, 1],
            c=self._joint_colors_for_frame(0),
            s=10,
            depthshade=True,
            zorder=4,
        )

        # Bone lines: one Line3D per bone (joint 1..J-1)
        self._bone_lines = []
        if self.parents is not None:
            n_joints = p0.shape[0]
            for j in range(1, n_joints):
                p = self.parents[j]
                if p == j:
                    (line,) = ax.plot([], [], [], color="#8899bb", linewidth=1.2)
                else:
                    (line,) = ax.plot(
                        [p0[j, 0], p0[p, 0]],
                        [p0[j, 2], p0[p, 2]],
                        [p0[j, 1], p0[p, 1]],
                        color="#8899bb",
                        linewidth=1.2,
                    )
                self._bone_lines.append(line)

        # Frame label
        self._frame_text = ax.text2D(
            0.02,
            0.97,
            f"Frame: 0 / {self.num_frames - 1}",
            transform=ax.transAxes,
            fontsize=8,
            color=_AXIS_TEXT_COLOR,
        )

    def _joint_colors_for_frame(self, frame):
        colors = np.tile(self._joint_default_rgba, (self.pos.shape[1], 1))
        if (
            self._planted_joint_mask is not None
            and 0 <= frame < self._planted_joint_mask.shape[0]
        ):
            colors[self._planted_joint_mask[frame]] = self._joint_planted_rgba
        return colors

    def _set_joint_colors(self, colors):
        self._joint_scatter.set_color(colors)
        self._joint_scatter.set_facecolor(colors)
        self._joint_scatter.set_edgecolor(colors)
        for attr in ("_facecolor3d", "_edgecolor3d", "_facecolors", "_edgecolors"):
            if hasattr(self._joint_scatter, attr):
                setattr(self._joint_scatter, attr, colors)

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def _update_frame(self, frame):
        frame = int(np.clip(frame, 0, self.num_frames - 1))
        self._frame = frame

        # --- 3D skeleton ---
        pf = self.pos[frame]
        xf, yf, zf = pf[:, 0], pf[:, 1], pf[:, 2]

        # Scatter: [X, Z(depth), Y(up)]
        self._joint_scatter._offsets3d = (xf, zf, yf)
        self._set_joint_colors(self._joint_colors_for_frame(frame))

        if self.parents is not None:
            for i, line in enumerate(self._bone_lines):
                j = i + 1
                p = self.parents[j]
                if p == j:
                    line.set_data_3d([], [], [])
                else:
                    line.set_data_3d(
                        [xf[j], xf[p]],
                        [zf[j], zf[p]],
                        [yf[j], yf[p]],
                    )

        self._frame_text.set_text(f"Frame: {frame} / {self.num_frames - 1}")

        for vline in self._cursor_lines:
            vline.set_xdata([frame, frame])

        if abs(self.slider.val - frame) > 0.5:
            self.slider.eventson = False
            self.slider.set_val(frame)
            self.slider.eventson = True

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------

    def _on_slider_change(self, val):
        self._update_frame(int(val))

    def _on_play_toggle(self, _event):
        if self._playing:
            self._stop()
        else:
            self._start()

    def _start(self):
        self._playing = True
        self.btn_play.label.set_text("⏸  Pause")
        interval_ms = max(1, int(1000 / self.fps))
        self._timer = self.fig.canvas.new_timer(interval=interval_ms)
        self._timer.add_callback(self._advance_frame)
        self._timer.start()

    def _stop(self):
        self._playing = False
        self.btn_play.label.set_text("▶  Play")
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _advance_frame(self):
        self._update_frame((self._frame + 1) % self.num_frames)

    def _on_close(self, _event):
        self._stop()

    def _rotate_camera(self, delta_azim=0, delta_elev=0, reset=False):
        if reset:
            elev = _DEFAULT_VIEW_ELEV
            azim = _DEFAULT_VIEW_AZIM
        else:
            elev = self.ax3d.elev + delta_elev
            azim = self.ax3d.azim + delta_azim
        self.ax3d.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def _on_key_press(self, event):
        key = (event.key or "").lower()
        if key in {"left", "a"}:
            self._rotate_camera(delta_azim=-_CAMERA_AZIM_STEP)
        elif key in {"right", "d"}:
            self._rotate_camera(delta_azim=_CAMERA_AZIM_STEP)
        elif key in {"up", "w"}:
            self._rotate_camera(delta_elev=_CAMERA_ELEV_STEP)
        elif key in {"down", "s"}:
            self._rotate_camera(delta_elev=-_CAMERA_ELEV_STEP)
        elif key == "home":
            self._rotate_camera(reset=True)

    def _set_status(self, message, is_error=False):
        if not hasattr(self, "_status_text"):
            return
        self._status_text.set_text(message)
        self._status_text.set_color(_STATUS_ERROR_COLOR if is_error else _STATUS_COLOR)
        self.fig.canvas.draw_idle()

    def _default_export_path(self):
        return Path.cwd() / "videos"

    def _collect_export_visibility(self):
        return {
            self.ax_slider: self.ax_slider.get_visible(),
            self.ax_btn: self.ax_btn.get_visible(),
            self.ax_save_btn: self.ax_save_btn.get_visible(),
        }

    def _toggle_export_ui(self, is_visible):
        self.ax_slider.set_visible(is_visible)
        self.ax_btn.set_visible(is_visible)
        self.ax_save_btn.set_visible(is_visible)

    def _on_save_mp4(self, _event):
        output_path = _choose_mp4_save_path(self._default_export_path(), self.title)
        if output_path is None:
            self._set_status("Save cancelled")
            return
        self._stop()
        self._set_status("Saving MP4...")
        try:
            saved_path = self.save_mp4(output_path)
        except Exception as exc:
            self._set_status(f"Save failed: {exc}", is_error=True)
            return
        else:
            profile_label = getattr(self, "_last_export_profile", "mp4")
            self._set_status(f"Saved {saved_path} ({profile_label})")

    def save_mp4(self, output_path, fps=None, dpi=_EXPORT_DPI):
        fps = self.fps if fps is None else int(fps)
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "Saving MP4 requires an ffmpeg installation visible to Matplotlib."
            )
        output_path = _normalize_mp4_output_path(output_path, self.title)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        visibility = self._collect_export_visibility()
        try:
            self._toggle_export_ui(False)
            self.fig.canvas.draw_idle()
            movie = animation.FuncAnimation(
                self.fig,
                lambda frame: self._update_frame(frame),
                frames=self.num_frames,
                interval=max(1, int(1000 / fps)),
                blit=False,
            )
            self._last_export_profile = _save_with_ffmpeg_profiles(
                movie,
                output_path,
                fps,
                dpi,
            )
        finally:
            for axis, is_visible in visibility.items():
                axis.set_visible(is_visible)
            self._update_frame(self._frame)

        return output_path

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def show(self, block=True):
        plt.show(block=block)


class MotionLMAComparisonVisualizer:
    def __init__(
        self,
        original_pos,
        edited_pos,
        original_lma,
        edited_lma,
        parents=None,
        fps=30,
        title=None,
    ):
        target_frames = max(_to_np(original_pos).shape[0], _to_np(edited_pos).shape[0])
        self.original_pos = _resample_motion_to(original_pos, target_frames)
        self.edited_pos = _resample_motion_to(edited_pos, target_frames)
        self.parents = None if parents is None else list(int(p) for p in parents)
        self.fps = int(fps)
        self.title = title or "Original vs Edited Motion and LMA"
        self.num_frames = target_frames
        self._frame = 0
        self._playing = False
        self._timer = None
        self._signal_rows = _build_lma_comparison_rows(
            original_lma,
            edited_lma,
            self.num_frames,
        )
        self._build_figure()

    def _build_figure(self):
        n_sig = max(len(self._signal_rows), 1)
        fig_h = max(8.0, n_sig * 0.85 + 1.8)
        self.fig = plt.figure(figsize=(18, fig_h))
        self.fig.suptitle(self.title, fontsize=10, fontweight="bold")
        self.fig.patch.set_facecolor("#1e1e2e")

        left = 0.03
        width_3d = 0.45
        left_sig = 0.51
        width_sig = 0.46
        top = 0.93
        bottom = 0.09
        height = top - bottom

        self.ax3d = self.fig.add_axes([left, bottom, width_3d, height], projection="3d")
        self._setup_3d_ax()

        self._sig_axes = []
        self._cursor_lines = []
        self._signal_lines = []
        gap = _COMPARISON_SIGNAL_GAP
        strip_h = (height - gap * n_sig) / n_sig

        for index, (key, label, original, edited) in enumerate(self._signal_rows):
            axis_bottom = top - (index + 1) * strip_h - index * gap
            ax = self.fig.add_axes([left_sig, axis_bottom, width_sig, strip_h])
            ax.set_facecolor(_LMA_BG)
            ax.patch.set_alpha(0.85)

            (original_line,) = ax.plot(
                original,
                color=_COMPARE_ORIGINAL_COLOR,
                linewidth=1.2,
                alpha=0.95,
                label="Original",
            )
            (edited_line,) = ax.plot(
                edited,
                color=_COMPARE_EDITED_COLOR,
                linewidth=1.4,
                alpha=0.95,
                label="Edited",
            )

            lo, hi = _get_signal_axis_range("lma", original, edited)
            yticks, yticklabels = _get_signal_tick_labels("lma", lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlim(0, self.num_frames - 1)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            _style_signal_axis(ax, label, show_xlabel=index == n_sig - 1)
            cursor = ax.axvline(x=0, color="#ff3355", linewidth=1.0, alpha=0.8)
            if index == 0:
                ax.legend(loc="upper right", fontsize=6, frameon=False)
            self._sig_axes.append(ax)
            self._cursor_lines.append(cursor)
            self._signal_lines.append((original_line, edited_line))

        self.ax_slider = self.fig.add_axes([0.12, 0.025, 0.72, 0.022])
        self.ax_slider.set_facecolor("#303050")
        self.slider = Slider(
            self.ax_slider,
            "Frame",
            valmin=0,
            valmax=self.num_frames - 1,
            valinit=0,
            valstep=1,
            color="#6688cc",
        )
        self.slider.label.set_fontsize(8)
        self.slider.valtext.set_fontsize(8)
        self.slider.on_changed(self._on_slider_change)

        self.ax_btn = self.fig.add_axes([0.01, 0.012, 0.085, 0.042])
        self.btn_play = Button(
            self.ax_btn, "▶  Play", color="#404060", hovercolor="#5566aa"
        )
        self.btn_play.label.set_fontsize(8)
        self.btn_play.on_clicked(self._on_play_toggle)

        self.ax_save_btn = self.fig.add_axes([0.86, 0.012, 0.11, 0.042])
        self.btn_save = Button(
            self.ax_save_btn,
            "Save MP4",
            color="#405040",
            hovercolor="#4f7c5f",
        )
        self.btn_save.label.set_fontsize(8)
        self.btn_save.on_clicked(self._on_save_mp4)

        self._status_text = self.fig.text(
            0.98,
            0.055,
            "Arrow keys or WASD rotate camera. Home resets view.",
            ha="right",
            va="center",
            fontsize=7,
            color=_STATUS_COLOR,
        )

        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._update_frame(0)

    def _setup_3d_ax(self):
        ax = self.ax3d
        all_pos = np.concatenate(
            [self.original_pos.reshape(-1, 3), self.edited_pos.reshape(-1, 3)], axis=0
        )
        all_x = all_pos[:, 0]
        all_y = all_pos[:, 1]
        all_z = all_pos[:, 2]
        cx, cy, cz = all_x.mean(), all_y.mean(), all_z.mean()
        half = (
            max(
                all_x.max() - all_x.min(),
                all_y.max() - all_y.min(),
                all_z.max() - all_z.min(),
                0.5,
            )
            * 0.6
        )

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cz - half, cz + half)
        ax.set_zlim(cy - half, cy + half)
        _style_3d_axis(ax, "Motion Comparison")
        ax.view_init(elev=_DEFAULT_VIEW_ELEV, azim=_DEFAULT_VIEW_AZIM)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
        ax.set_facecolor("#12121c")

        original_root = self.original_pos[:, 0, :]
        edited_root = self.edited_pos[:, 0, :]
        ax.plot(
            original_root[:, 0],
            original_root[:, 2],
            original_root[:, 1],
            color=_COMPARE_ORIGINAL_COLOR,
            linewidth=0.8,
            alpha=0.45,
            linestyle="--",
        )
        ax.plot(
            edited_root[:, 0],
            edited_root[:, 2],
            edited_root[:, 1],
            color=_COMPARE_EDITED_COLOR,
            linewidth=0.8,
            alpha=0.45,
            linestyle="--",
        )

        original_frame = self.original_pos[0]
        edited_frame = self.edited_pos[0]
        self._original_scatter = ax.scatter(
            original_frame[:, 0],
            original_frame[:, 2],
            original_frame[:, 1],
            c=_COMPARE_ORIGINAL_COLOR,
            s=10,
            depthshade=True,
            zorder=4,
        )
        self._edited_scatter = ax.scatter(
            edited_frame[:, 0],
            edited_frame[:, 2],
            edited_frame[:, 1],
            c=_COMPARE_EDITED_COLOR,
            s=10,
            depthshade=True,
            zorder=4,
        )

        self._original_bone_lines = []
        self._edited_bone_lines = []
        if self.parents is not None:
            joint_count = original_frame.shape[0]
            for joint_index in range(1, joint_count):
                parent_index = self.parents[joint_index]
                if parent_index == joint_index:
                    (original_line,) = ax.plot(
                        [], [], [], color=_COMPARE_ORIGINAL_BONE, linewidth=1.0
                    )
                    (edited_line,) = ax.plot(
                        [], [], [], color=_COMPARE_EDITED_BONE, linewidth=1.0
                    )
                else:
                    (original_line,) = ax.plot(
                        [
                            original_frame[joint_index, 0],
                            original_frame[parent_index, 0],
                        ],
                        [
                            original_frame[joint_index, 2],
                            original_frame[parent_index, 2],
                        ],
                        [
                            original_frame[joint_index, 1],
                            original_frame[parent_index, 1],
                        ],
                        color=_COMPARE_ORIGINAL_BONE,
                        linewidth=1.0,
                    )
                    (edited_line,) = ax.plot(
                        [edited_frame[joint_index, 0], edited_frame[parent_index, 0]],
                        [edited_frame[joint_index, 2], edited_frame[parent_index, 2]],
                        [edited_frame[joint_index, 1], edited_frame[parent_index, 1]],
                        color=_COMPARE_EDITED_BONE,
                        linewidth=1.0,
                    )
                self._original_bone_lines.append(original_line)
                self._edited_bone_lines.append(edited_line)

        self._frame_text = ax.text2D(
            0.02,
            0.97,
            f"Frame: 0 / {self.num_frames - 1}",
            transform=ax.transAxes,
            fontsize=8,
            color=_AXIS_TEXT_COLOR,
        )
        ax.text2D(
            0.02,
            0.92,
            "Original",
            transform=ax.transAxes,
            fontsize=8,
            color=_AXIS_TEXT_COLOR,
        )
        ax.text2D(
            0.16,
            0.92,
            "Edited",
            transform=ax.transAxes,
            fontsize=8,
            color=_AXIS_TEXT_COLOR,
        )

    def _set_scatter_offsets(self, scatter, frame_positions):
        scatter._offsets3d = (
            frame_positions[:, 0],
            frame_positions[:, 2],
            frame_positions[:, 1],
        )

    def _update_bones(self, frame_positions, bone_lines):
        if self.parents is None:
            return
        x = frame_positions[:, 0]
        y = frame_positions[:, 1]
        z = frame_positions[:, 2]
        for line_index, line in enumerate(bone_lines):
            joint_index = line_index + 1
            parent_index = self.parents[joint_index]
            if parent_index == joint_index:
                line.set_data_3d([], [], [])
            else:
                line.set_data_3d(
                    [x[joint_index], x[parent_index]],
                    [z[joint_index], z[parent_index]],
                    [y[joint_index], y[parent_index]],
                )

    def _update_frame(self, frame):
        frame = int(np.clip(frame, 0, self.num_frames - 1))
        self._frame = frame
        original_frame = self.original_pos[frame]
        edited_frame = self.edited_pos[frame]
        self._set_scatter_offsets(self._original_scatter, original_frame)
        self._set_scatter_offsets(self._edited_scatter, edited_frame)
        self._update_bones(original_frame, self._original_bone_lines)
        self._update_bones(edited_frame, self._edited_bone_lines)
        self._frame_text.set_text(f"Frame: {frame} / {self.num_frames - 1}")

        for cursor in self._cursor_lines:
            cursor.set_xdata([frame, frame])

        if abs(self.slider.val - frame) > 0.5:
            self.slider.eventson = False
            self.slider.set_val(frame)
            self.slider.eventson = True

        self.fig.canvas.draw_idle()

    def _on_slider_change(self, val):
        self._update_frame(int(val))

    def _on_play_toggle(self, _event):
        if self._playing:
            self._stop()
        else:
            self._start()

    def _start(self):
        self._playing = True
        self.btn_play.label.set_text("⏸  Pause")
        interval_ms = max(1, int(1000 / self.fps))
        self._timer = self.fig.canvas.new_timer(interval=interval_ms)
        self._timer.add_callback(self._advance_frame)
        self._timer.start()

    def _stop(self):
        self._playing = False
        self.btn_play.label.set_text("▶  Play")
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _advance_frame(self):
        self._update_frame((self._frame + 1) % self.num_frames)

    def _on_close(self, _event):
        self._stop()

    def _rotate_camera(self, delta_azim=0, delta_elev=0, reset=False):
        if reset:
            elev = _DEFAULT_VIEW_ELEV
            azim = _DEFAULT_VIEW_AZIM
        else:
            elev = self.ax3d.elev + delta_elev
            azim = self.ax3d.azim + delta_azim
        self.ax3d.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def _on_key_press(self, event):
        key = (event.key or "").lower()
        if key in {"left", "a"}:
            self._rotate_camera(delta_azim=-_CAMERA_AZIM_STEP)
        elif key in {"right", "d"}:
            self._rotate_camera(delta_azim=_CAMERA_AZIM_STEP)
        elif key in {"up", "w"}:
            self._rotate_camera(delta_elev=_CAMERA_ELEV_STEP)
        elif key in {"down", "s"}:
            self._rotate_camera(delta_elev=-_CAMERA_ELEV_STEP)
        elif key == "home":
            self._rotate_camera(reset=True)

    def _set_status(self, message, is_error=False):
        self._status_text.set_text(message)
        self._status_text.set_color(_STATUS_ERROR_COLOR if is_error else _STATUS_COLOR)
        self.fig.canvas.draw_idle()

    def _default_export_path(self):
        return Path.cwd() / "videos"

    def _collect_export_visibility(self):
        return {
            self.ax_slider: self.ax_slider.get_visible(),
            self.ax_btn: self.ax_btn.get_visible(),
            self.ax_save_btn: self.ax_save_btn.get_visible(),
        }

    def _toggle_export_ui(self, is_visible):
        self.ax_slider.set_visible(is_visible)
        self.ax_btn.set_visible(is_visible)
        self.ax_save_btn.set_visible(is_visible)

    def _on_save_mp4(self, _event):
        output_path = _choose_mp4_save_path(self._default_export_path(), self.title)
        if output_path is None:
            self._set_status("Save cancelled")
            return
        self._stop()
        self._set_status("Saving MP4...")
        try:
            saved_path = self.save_mp4(output_path)
        except Exception as exc:
            self._set_status(f"Save failed: {exc}", is_error=True)
            return
        else:
            profile_label = getattr(self, "_last_export_profile", "mp4")
            self._set_status(f"Saved {saved_path} ({profile_label})")

    def save_mp4(self, output_path, fps=None, dpi=_EXPORT_DPI):
        fps = self.fps if fps is None else int(fps)
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "Saving MP4 requires an ffmpeg installation visible to Matplotlib."
            )
        output_path = _normalize_mp4_output_path(output_path, self.title)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        visibility = self._collect_export_visibility()
        try:
            self._toggle_export_ui(False)
            self.fig.canvas.draw_idle()
            movie = animation.FuncAnimation(
                self.fig,
                lambda frame: self._update_frame(frame),
                frames=self.num_frames,
                interval=max(1, int(1000 / fps)),
                blit=False,
            )
            self._last_export_profile = _save_with_ffmpeg_profiles(
                movie,
                output_path,
                fps,
                dpi,
            )
        finally:
            for axis, is_visible in visibility.items():
                axis.set_visible(is_visible)
            self._update_frame(self._frame)

        return output_path

    def show(self, block=True):
        plt.show(block=block)


# ---------------------------------------------------------------------------
# Convenience function (called from compute_tags)
# ---------------------------------------------------------------------------


def visualize_motion_and_tags(pos, tags, parents=None, fps=30, title=None, block=True):
    """
    Convenience wrapper.  Creates a MotionVisualizer and shows it.

    Parameters
    ----------
    pos     : np.ndarray [frames, n_joints, 3]
    tags    : dict returned by compute_tags()
    parents : optional list/array of parent joint indices
    fps     : playback frame rate (default 30)
    title   : optional window title
    block   : if True (default) blocks until the window is closed
    """
    viz = MotionVisualizer(pos, tags, parents=parents, fps=fps, title=title)
    viz.show(block=block)


def visualize_motion_lma_comparison(
    original_pos,
    edited_pos,
    original_lma,
    edited_lma,
    parents=None,
    fps=30,
    title=None,
    block=True,
    save_path=None,
):
    viz = MotionLMAComparisonVisualizer(
        original_pos,
        edited_pos,
        original_lma,
        edited_lma,
        parents=parents,
        fps=fps,
        title=title,
    )
    if save_path is not None:
        viz.save_mp4(save_path, fps=fps)
    viz.show(block=block)
    return viz
