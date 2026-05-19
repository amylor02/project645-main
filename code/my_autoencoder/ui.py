import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, X, Y, StringVar, Tk
from tkinter import messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRAW_SCRIPT_RELATIVE = Path("src_dif_mdm_v2") / "draw_lma_and_sample.py"
LISTING_DATA_ROOT_RELATIVE = Path("data") / "DanceDB"
COMMAND_DATA_PATH_RELATIVE = Path("data") / "DanceDB"
VAE_MODEL_PATH_RELATIVE = Path("models") / "model_v2_DanceDB"
DIFFUSION_MODEL_PATH_RELATIVE = (
    Path("models")
    / "diffusion_latent_mdm_human_v2_hq_per_DanceDB"
    / "latent_diffusion_prior_best.pt"
)
MODE = "full"
SAVE_BVH = True
USE_FULL_MODE_LMA_SUFFIX = True
FULL_MODE_LMA_SUFFIX_FRACTION = 0.85
TRAJ_CFG_SCALE = 1.5
LMA_CFG_SCALE = 4.0
SAMPLE_STEPS = 300


def list_available_bvh_files():
    data_root = PROJECT_ROOT / LISTING_DATA_ROOT_RELATIVE
    entries = []
    for split_name in ("train", "eval"):
        split_dir = data_root / split_name
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.rglob("*.bvh")):
            entries.append(
                {
                    "split": split_name,
                    "name": path.relative_to(split_dir).as_posix(),
                    "absolute_path": path,
                    "project_relative_path": path.relative_to(PROJECT_ROOT),
                }
            )
    return entries


def build_command(selected_bvh_path: Path):
    command = [
        sys.executable,
        str(DRAW_SCRIPT_RELATIVE),
        "--data-path",
        str(COMMAND_DATA_PATH_RELATIVE),
        "--vae-model-path",
        str(VAE_MODEL_PATH_RELATIVE),
        "--diffusion-model-path",
        str(DIFFUSION_MODEL_PATH_RELATIVE),
        "--bvh-path",
        str(selected_bvh_path),
        "--mode",
        MODE,
        "--traj-cfg-scale",
        str(TRAJ_CFG_SCALE),
        "--lma-cfg-scale",
        str(LMA_CFG_SCALE),
        "--sample-steps",
        str(SAMPLE_STEPS),
    ]
    if USE_FULL_MODE_LMA_SUFFIX:
        command.append("--full-mode-lma-suffix")
        command.extend(
            [
                "--full-mode-lma-suffix-fraction",
                str(FULL_MODE_LMA_SUFFIX_FRACTION),
            ]
        )
    if SAVE_BVH:
        command.append("--save-bvh")
    return command


def format_command(command_parts):
    return subprocess.list2cmdline(command_parts)


class SampleLauncherUI:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("DanceDB LMA Sample Launcher")
        self.root.geometry("1100x720")

        self.items = list_available_bvh_files()
        self.filtered_items = list(self.items)
        self.output_queue = queue.Queue()
        self.process = None

        self.filter_var = StringVar()
        self.selection_var = StringVar(value="No file selected")
        self.status_var = StringVar(value="Ready")

        self._build_layout()
        self._populate_tree()
        self.root.after(100, self._drain_output_queue)

    def _build_layout(self):
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        controls = ttk.Frame(main_frame)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Filter files:").grid(row=0, column=0, sticky="w")
        filter_entry = ttk.Entry(controls, textvariable=self.filter_var)
        filter_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        filter_entry.bind("<KeyRelease>", self._on_filter_changed)

        refresh_button = ttk.Button(controls, text="Refresh", command=self._refresh)
        refresh_button.grid(row=0, column=2, sticky="e")

        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            list_frame,
            columns=("path",),
            show="headings",
            selectmode="browse",
        )
        self.tree.heading("path", text="File")
        self.tree.column("path", width=960, stretch=True)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_selection_changed)
        self.tree.bind("<Double-1>", self._run_selected_command)

        tree_scrollbar = ttk.Scrollbar(
            list_frame, orient=VERTICAL, command=self.tree.yview
        )
        tree_scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

        selection_frame = ttk.LabelFrame(main_frame, text="Selection", padding=8)
        selection_frame.grid(row=2, column=0, sticky="ew", pady=(8, 8))
        selection_frame.columnconfigure(0, weight=1)
        ttk.Label(selection_frame, textvariable=self.selection_var).grid(
            row=0, column=0, sticky="w"
        )

        buttons = ttk.Frame(main_frame)
        buttons.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        buttons.columnconfigure(0, weight=1)

        button_bar = ttk.Frame(buttons)
        button_bar.pack(fill=X)
        self.run_button = ttk.Button(
            button_bar, text="Run sampler", command=self._run_selected_command
        )
        self.run_button.pack(side=LEFT)
        self.stop_button = ttk.Button(
            button_bar, text="Stop", command=self._stop_process, state="disabled"
        )
        self.stop_button.pack(side=LEFT, padx=(8, 0))
        ttk.Label(button_bar, textvariable=self.status_var).pack(side=RIGHT)

        output_frame = ttk.LabelFrame(main_frame, text="Process Output", padding=8)
        output_frame.grid(row=4, column=0, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        self.output_text = ttk.Treeview(
            output_frame,
            columns=("line",),
            show="tree",
            selectmode="none",
            height=14,
        )
        self.output_text.grid(row=0, column=0, sticky="nsew")
        output_scrollbar = ttk.Scrollbar(
            output_frame, orient=VERTICAL, command=self.output_text.yview
        )
        output_scrollbar.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=output_scrollbar.set)

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        for index, item in enumerate(self.filtered_items):
            self.tree.insert(
                "",
                END,
                iid=str(index),
                values=(item["name"],),
            )

        if self.filtered_items:
            first_item_id = self.tree.get_children()[0]
            self.tree.selection_set(first_item_id)
            self.tree.focus(first_item_id)
            self._update_selection_state(first_item_id)
        else:
            self.selection_var.set("No files found under data/DanceDB/train or eval")

    def _refresh(self):
        self.items = list_available_bvh_files()
        self._apply_filter()
        self.status_var.set(f"Loaded {len(self.items)} BVH files")

    def _on_filter_changed(self, _event=None):
        self._apply_filter()

    def _apply_filter(self):
        filter_text = self.filter_var.get().strip().lower()
        if not filter_text:
            self.filtered_items = list(self.items)
        else:
            self.filtered_items = [
                item
                for item in self.items
                if filter_text in item["split"].lower()
                or filter_text in item["name"].lower()
                or filter_text in item["project_relative_path"].as_posix().lower()
            ]
        self._populate_tree()

    def _on_selection_changed(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            return
        self._update_selection_state(selection[0])

    def _update_selection_state(self, item_id: str):
        selected_item = self.filtered_items[int(item_id)]
        relative_path = selected_item["project_relative_path"].as_posix()
        self.selection_var.set(f"Selected: {relative_path}")

    def _get_selected_item(self):
        selection = self.tree.selection()
        if not selection:
            return None
        return self.filtered_items[int(selection[0])]

    def _run_selected_command(self, _event=None):
        if self.process is not None:
            messagebox.showinfo(
                "Sampler running", "A sampler process is already running."
            )
            return

        selected_item = self._get_selected_item()
        if selected_item is None:
            messagebox.showwarning("No selection", "Select a BVH file first.")
            return

        command = build_command(selected_item["project_relative_path"])
        self._append_output(f"> {format_command(command)}")
        self.status_var.set("Running sampler...")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        worker = threading.Thread(
            target=self._run_process_worker, args=(command,), daemon=True
        )
        worker.start()

    def _run_process_worker(self, command):
        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.output_queue.put(line.rstrip())
            return_code = self.process.wait()
            self.output_queue.put(f"\nProcess finished with exit code {return_code}")
        except Exception as exc:
            self.output_queue.put(f"Failed to start process: {exc}")
        finally:
            self.process = None
            self.output_queue.put("__PROCESS_COMPLETE__")

    def _stop_process(self):
        if self.process is None:
            return
        self.process.terminate()
        self.status_var.set("Stopping sampler...")

    def _drain_output_queue(self):
        while not self.output_queue.empty():
            line = self.output_queue.get()
            if line == "__PROCESS_COMPLETE__":
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                if self.status_var.get().startswith("Stopping"):
                    self.status_var.set("Sampler stopped")
                else:
                    self.status_var.set("Ready")
                continue
            self._append_output(line)
        self.root.after(100, self._drain_output_queue)

    def _append_output(self, text: str):
        self.output_text.insert("", END, text=text)
        children = self.output_text.get_children()
        if children:
            self.output_text.see(children[-1])


def main():
    root = Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    SampleLauncherUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
