from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import os


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "kaggle_sync_config.json"
EXAMPLE_CONFIG_PATH = PROJECT_DIR / "kaggle_sync_config.example.json"
STAGE_DIR = PROJECT_DIR / ".kaggle_stage" / "kernel"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "kaggle_outputs"

SOURCE_FILES = [
    "prepare.py",
    "train.py",
    "program.md",
    "pyproject.toml",
    "results.tsv",
]


def _run(cmd: list[str], cwd: Path | None = None, capture_output: bool = False):
    print("+", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        check=True,
        capture_output=capture_output,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def _kaggle_cmd() -> list[str]:
    explicit = os.environ.get("KAGGLE_EXE")
    if explicit:
        return [explicit]

    scripts_dir = Path(sys.executable).resolve().parent / "Scripts"
    kaggle_exe = scripts_dir / "kaggle.exe"
    if kaggle_exe.exists():
        return [str(kaggle_exe)]

    return ["kaggle"]


def _load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CONFIG_PATH.name}. Copy {EXAMPLE_CONFIG_PATH.name} to "
            f"{CONFIG_PATH.name} and fill in your Kaggle details."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = ["kernel_id", "kernel_title", "dataset_source", "dataset_mount_slug"]
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise RuntimeError(f"Missing required config keys: {', '.join(missing)}")
    return data


def _check_kaggle_cli():
    try:
        result = _run(_kaggle_cmd() + ["--version"], capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI is not installed or not on PATH. Install it with "
            "`python -m pip install kaggle` and ensure `kaggle` works in your shell."
        ) from exc
    version_text = (result.stdout or result.stderr).strip()
    print(version_text)


def _dataset_mount_path(config):
    slug = config["dataset_mount_slug"].strip("/")
    subdir = config.get("dataset_subdir", "").strip("/")
    if subdir:
        return f"/kaggle/input/{slug}/{subdir}"
    return f"/kaggle/input/{slug}"


def _render_kernel_runner(config):
    dataset_root = _dataset_mount_path(config)
    weights_slug = config.get("weights_dataset_mount_slug", "").strip("/")
    weights_file = config.get("weights_file", "resnet18-f37072fd.pth").strip("/")
    torch_index_url = config.get(
        "p100_torch_index_url",
        "https://download.pytorch.org/whl/cu126",
    )
    torch_version = config.get("p100_torch_version", "2.7.1")
    torchvision_version = config.get("p100_torchvision_version", "0.22.1")
    prepare_src = (PROJECT_DIR / "prepare.py").read_text(encoding="utf-8")
    train_src = (PROJECT_DIR / "train.py").read_text(encoding="utf-8")
    prepare_literal = repr(prepare_src)
    train_literal = repr(train_src)
    return f'''import os
import sys
import subprocess
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout

DEFAULT_DATASET_ROOT = r"{dataset_root}"
DEFAULT_WEIGHTS_PATH = r"/kaggle/input/{weights_slug}/{weights_file}"
P100_TORCH_INDEX_URL = r"{torch_index_url}"
P100_TORCH_VERSION = r"{torch_version}"
P100_TORCHVISION_VERSION = r"{torchvision_version}"


def _discover_dataset_root():
    preferred = Path(DEFAULT_DATASET_ROOT)
    if (preferred / "splits" / "train.csv").exists():
        return preferred
    base = Path("/kaggle/input")
    for candidate in base.rglob("train.csv"):
        if candidate.name != "train.csv":
            continue
        if candidate.parent.name != "splits":
            continue
        root = candidate.parent.parent
        if (root / "images").exists():
            return root
    return preferred


def _discover_weights_path():
    preferred = Path(DEFAULT_WEIGHTS_PATH)
    if preferred.exists():
        return preferred
    base = Path("/kaggle/input")
    for candidate in base.rglob("{weights_file}"):
        if candidate.name == "{weights_file}":
            return candidate
    return preferred


def _gpu_name():
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        print(f"gpu_probe_error: {{type(exc).__name__}}")
        return ""
    return (probe.stdout or probe.stderr or "").strip()


def _ensure_p100_compatible_torch():
    gpu_name = _gpu_name()
    if gpu_name:
        print(f"gpu_name: {{gpu_name}}")
    if "P100" not in gpu_name:
        return

    print("Detected Tesla P100 runtime; installing a PyTorch build compatible with sm_60.")
    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-input",
        "--no-cache-dir",
        "--upgrade",
        "--index-url",
        P100_TORCH_INDEX_URL,
        f"torch=={{P100_TORCH_VERSION}}",
        f"torchvision=={{P100_TORCHVISION_VERSION}}",
    ]
    print("+", " ".join(install_cmd))
    subprocess.run(install_cmd, check=True)


os.environ.setdefault("INDIWASTE_ROOT", str(_discover_dataset_root()))
os.environ.setdefault("RESNET18_WEIGHTS_PATH", str(_discover_weights_path()))
os.environ.setdefault("REQUIRE_CUDA", "1")

PREPARE_SRC = {prepare_literal}
TRAIN_SRC = {train_literal}

Path("prepare.py").write_text(PREPARE_SRC, encoding="utf-8")
Path("train.py").write_text(TRAIN_SRC, encoding="utf-8")
sys.path.insert(0, os.getcwd())
_ensure_p100_compatible_torch()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


from train import main


if __name__ == "__main__":
    with open("run.log", "w", encoding="utf-8") as f:
        tee_out = Tee(sys.stdout, f)
        tee_err = Tee(sys.stderr, f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            raise SystemExit(main())
'''


def _render_kernel_metadata(config):
    dataset_sources = [config["dataset_source"]]
    weights_source = config.get("weights_dataset_source", "").strip()
    if weights_source:
        dataset_sources.append(weights_source)
    return {
        "id": config["kernel_id"],
        "title": config["kernel_title"],
        "code_file": "run_project3_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": str(bool(config.get("is_private", True))).lower(),
        "enable_gpu": str(bool(config.get("enable_gpu", True))).lower(),
        "enable_internet": str(bool(config.get("enable_internet", True))).lower(),
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def stage_kernel():
    config = _load_config()
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_FILES:
        src = PROJECT_DIR / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required source file: {src}")
        shutil.copy2(src, STAGE_DIR / filename)

    runner_path = STAGE_DIR / "run_project3_kaggle.py"
    runner_path.write_text(_render_kernel_runner(config), encoding="utf-8")

    metadata_path = STAGE_DIR / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(_render_kernel_metadata(config), indent=2),
        encoding="utf-8",
    )

    print(f"Staged Kaggle kernel at {STAGE_DIR}")
    print(f"Dataset mount path inside Kaggle: {_dataset_mount_path(config)}")


def push_kernel():
    _check_kaggle_cli()
    if not STAGE_DIR.exists():
        stage_kernel()
    _run(_kaggle_cmd() + ["kernels", "push", "-p", str(STAGE_DIR)])


def kernel_status():
    config = _load_config()
    _check_kaggle_cli()
    _run(_kaggle_cmd() + ["kernels", "status", config["kernel_id"]])


def download_output():
    config = _load_config()
    _check_kaggle_cli()
    output_dir = PROJECT_DIR / config.get("output_dir", "kaggle_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(_kaggle_cmd() + ["kernels", "output", config["kernel_id"], "-p", str(output_dir)])
    print(f"Downloaded outputs to {output_dir}")


def watch_and_download():
    config = _load_config()
    _check_kaggle_cli()
    poll_seconds = int(config.get("poll_seconds", 30))
    kernel_id = config["kernel_id"]

    while True:
        result = _run(
            _kaggle_cmd() + ["kernels", "status", kernel_id],
            capture_output=True,
        )
        text = (result.stdout or "") + (result.stderr or "")
        print(text.strip())
        lowered = text.lower()
        if "complete" in lowered:
            download_output()
            return
        if "error" in lowered or "failed" in lowered:
            raise RuntimeError("Kaggle kernel finished with an error state.")
        time.sleep(poll_seconds)


def prepare_config():
    if CONFIG_PATH.exists():
        print(f"{CONFIG_PATH.name} already exists.")
        return
    shutil.copy2(EXAMPLE_CONFIG_PATH, CONFIG_PATH)
    print(f"Created {CONFIG_PATH.name}. Fill it in before running push/status/output.")


def main():
    parser = argparse.ArgumentParser(description="Local Kaggle sync helper for Project 3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare-config", help="Copy the example config into a working config file.")
    subparsers.add_parser("stage-kernel", help="Build the Kaggle kernel staging directory.")
    subparsers.add_parser("push", help="Push the staged kernel to Kaggle.")
    subparsers.add_parser("status", help="Check Kaggle kernel status.")
    subparsers.add_parser("download-output", help="Download Kaggle kernel outputs.")
    subparsers.add_parser("watch", help="Poll Kaggle until the run completes, then download outputs.")

    args = parser.parse_args()

    if args.command == "prepare-config":
        prepare_config()
    elif args.command == "stage-kernel":
        stage_kernel()
    elif args.command == "push":
        push_kernel()
    elif args.command == "status":
        kernel_status()
    elif args.command == "download-output":
        download_output()
    elif args.command == "watch":
        watch_and_download()
    else:
        raise RuntimeError(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
