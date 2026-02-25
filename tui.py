#!/usr/bin/env python3
"""Lite TUI for launching the pet pipeline locally or on a remote server.

Collects all arguments interactively, then runs locally or SSHes in.

Requirements (local only):
    pip install rich InquirerPy
"""

import re
import subprocess
import sys
import time

from InquirerPy import inquirer
from rich.console import Console
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REMOTE_HOST = "root@172.234.228.57"
REMOTE_VENV = "~/venv/bin/activate"
REMOTE_CODE_DIR = "~/pet_pipeline"
REMOTE_PYTHON = "~/venv/bin/python"

LOCAL_CODE_DIR = str(__import__("pathlib").Path(__file__).parent)
LOCAL_PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Collect arguments
# ---------------------------------------------------------------------------

def validate_remote_path(path: str) -> bool:
    """Basic sanity check for a remote path string."""
    path = path.strip()
    if not path:
        return False
    if not (path.startswith("/") or path.startswith("~")):
        return False
    return True


BACK_SENTINEL = "__BACK__"


def _select_with_back(message: str, choices: list, allow_back: bool = True):
    """Run an inquirer select prompt with an optional '← Go back' choice."""
    opts = list(choices)
    if allow_back:
        opts.append({"name": "← Go back", "value": BACK_SENTINEL})
    return inquirer.select(message=message, choices=opts).execute()


def _text_with_back(message: str, allow_back: bool = True):
    """Run an inquirer text prompt. Empty input = go back."""
    hint = " (leave blank to go back)" if allow_back else ""
    result = inquirer.text(message=f"{message}{hint}").execute()
    if allow_back and not result.strip():
        return BACK_SENTINEL
    if not validate_remote_path(result):
        from rich.console import Console
        Console().print("[red]  Path must start with / or ~[/red]")
        return BACK_SENTINEL
    return result.strip()


def collect_args() -> dict:
    """Interactively collect pipeline arguments with go-back support."""
    args = {}
    step = 0

    while True:
        # --- Step 0: Run where? ---
        if step == 0:
            target = _select_with_back(
                "Run where?",
                [
                    {"name": f"Remote server ({REMOTE_HOST})", "value": "remote"},
                    {"name": "Local machine", "value": "local"},
                ],
                allow_back=False,  # first step, nowhere to go back
            )
            args["target"] = target
            step = 1
            continue

        # --- Step 1: Pipeline mode ---
        if step == 1:
            mode = _select_with_back(
                "Pipeline mode:",
                [
                    {"name": "Batch — cluster all photos from scratch", "value": "batch"},
                    {"name": "Incremental — add new photos to existing clusters", "value": "add"},
                ],
            )
            if mode == BACK_SENTINEL:
                step = 0; continue
            args["mode"] = mode
            step = 2
            continue

        # --- Step 2: Input directory ---
        if step == 2:
            location = "on remote server" if args["target"] == "remote" else "local"
            input_dir = _text_with_back(f"Input directory ({location}):")
            if input_dir == BACK_SENTINEL:
                step = 1; continue
            args["input"] = input_dir
            step = 3
            continue

        # --- Step 3: Species (batch) / State file (incremental) ---
        if step == 3:
            if args["mode"] == "batch":
                species = _select_with_back(
                    "Species:",
                    [
                        {"name": "Dog", "value": "dog"},
                        {"name": "Cat", "value": "cat"},
                        {"name": "Auto-detect", "value": "auto"},
                    ],
                )
                if species == BACK_SENTINEL:
                    step = 2; continue
                args["species"] = species
                args["state"] = None
            else:
                location = "on remote server" if args["target"] == "remote" else "local"
                state = _text_with_back(f"Path to existing state.npz ({location}):")
                if state == BACK_SENTINEL:
                    step = 2; continue
                args["state"] = state
                args["species"] = None
            step = 4
            continue

        # --- Step 4: Clustering algorithm (batch only) ---
        if step == 4:
            if args["mode"] == "batch":
                algorithm = _select_with_back(
                    "Clustering algorithm:",
                    [
                        {"name": "HDBSCAN (density-based, default)", "value": "hdbscan"},
                        {"name": "Agglomerative (threshold-based, stable under removals)", "value": "agglomerative"},
                        {"name": "Chinese Whispers (graph-based, stable under removals)", "value": "chinese_whispers"},
                    ],
                )
                if algorithm == BACK_SENTINEL:
                    step = 3; continue
                args["algorithm"] = algorithm
            else:
                args["algorithm"] = None
            step = 5
            continue

        # --- Step 5: Output directory ---
        if step == 5:
            location = "on remote server" if args["target"] == "remote" else "local"
            output_dir = _text_with_back(f"Output directory ({location}):")
            if output_dir == BACK_SENTINEL:
                step = 4; continue
            args["output"] = output_dir
            step = 6
            continue

        # --- Step 6: GPU ---
        if step == 6:
            gpu = _select_with_back(
                "Use GPU?",
                [
                    {"name": "Yes", "value": True},
                    {"name": "No", "value": False},
                ],
            )
            if gpu == BACK_SENTINEL:
                step = 5; continue
            args["gpu"] = gpu
            step = 7
            continue

        # --- Step 7: Verbose ---
        if step == 7:
            verbose = _select_with_back(
                "Verbose logging?",
                [
                    {"name": "Yes", "value": True},
                    {"name": "No", "value": False},
                ],
            )
            if verbose == BACK_SENTINEL:
                step = 6; continue
            args["verbose"] = verbose
            break  # all done

    return args


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

def build_command(args: dict) -> str:
    """Build the python command from collected arguments."""
    if args["target"] == "remote":
        python_bin = REMOTE_PYTHON
        code_dir = REMOTE_CODE_DIR
    else:
        python_bin = LOCAL_PYTHON
        code_dir = LOCAL_CODE_DIR

    parts = [python_bin, f"{code_dir}/run.py"]

    if args["verbose"]:
        parts.append("--verbose")

    if args["mode"] == "batch":
        parts.append("batch")
        parts.extend(["--input", args["input"]])
        parts.extend(["--output", args["output"]])
        if args["species"] != "auto":
            parts.extend(["--species", args["species"]])
        if args.get("algorithm") and args["algorithm"] != "hdbscan":
            parts.extend(["--algorithm", args["algorithm"]])
        if args["gpu"]:
            parts.append("--gpu")
    else:
        parts.append("add")
        parts.extend(["--input", args["input"]])
        parts.extend(["--state", args["state"]])
        parts.extend(["--output", args["output"]])
        if args["gpu"]:
            parts.append("--gpu")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Execution with streaming
# ---------------------------------------------------------------------------

STEP_RE = re.compile(r"^\[(\d+/\d+)\]")


def _stream_output(process: subprocess.Popen, console: Console):
    """Stream and colorize pipeline output line by line."""
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue

        # Colorize known patterns
        if STEP_RE.match(line):
            console.print(f"  [bold cyan]{line}[/bold cyan]")
        elif "ERROR" in line:
            console.print(f"  [bold red]{line}[/bold red]")
        elif "====" in line:
            console.print(f"  [bold green]{line}[/bold green]")
        elif "RESULTS" in line:
            console.print(f"  [bold green]{line}[/bold green]")
        elif line.strip().startswith("Processing images:"):
            console.print(f"  [dim]{line}[/dim]", highlight=False)
        elif line.strip().startswith("Faces detected:") or line.strip().startswith("Bodies detected:"):
            console.print(f"  [yellow]{line}[/yellow]")
        elif line.strip().startswith(("Total images:", "Clusters:", "Clustered:",
                                      "Unclustered:", "Time:", "Output:", "Assigned existing:")):
            console.print(f"  [bold]{line}[/bold]")
        else:
            console.print(f"  {line}")


def run_remote(cmd: str, console: Console) -> int:
    """SSH into the server, run the command, and stream output."""
    remote_cmd = f"source {REMOTE_VENV} && cd {REMOTE_CODE_DIR} && PYTHONUNBUFFERED=1 {cmd}"
    ssh_cmd = ["ssh", REMOTE_HOST, remote_cmd]

    console.print()
    console.rule("[bold blue]Remote Execution")
    console.print(f"  [dim]$ {cmd}[/dim]\n")

    process = subprocess.Popen(
        ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    _stream_output(process, console)
    return process.wait()


def run_local(cmd: str, console: Console) -> int:
    """Run the command locally and stream output."""
    console.print()
    console.rule("[bold blue]Local Execution")
    console.print(f"  [dim]$ {cmd}[/dim]\n")

    env = {**__import__("os").environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env, cwd=LOCAL_CODE_DIR,
    )
    _stream_output(process, console)
    return process.wait()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    console = Console()

    console.print()
    console.print(Panel(
        "[bold]Pet Pipeline Launcher[/bold]\n"
        f"[dim]Remote: {REMOTE_HOST}  |  Local: {LOCAL_CODE_DIR}[/dim]",
        border_style="blue",
        padding=(1, 2),
    ))
    console.print()

    try:
        args = collect_args()
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        sys.exit(0)

    is_remote = args["target"] == "remote"
    cmd = build_command(args)

    console.print()
    target_label = f"remote ({REMOTE_HOST})" if is_remote else "local"
    console.print(Panel(
        f"[bold]{cmd}[/bold]",
        title=f"Command ({target_label})",
        border_style="yellow",
        padding=(1, 2),
    ))

    try:
        confirm = inquirer.confirm(
            message=f"Run this {target_label}?",
            default=True,
        ).execute()
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        sys.exit(0)

    if not confirm:
        console.print("[yellow]Aborted.[/yellow]")
        sys.exit(0)

    start = time.time()
    try:
        rc = run_remote(cmd, console) if is_remote else run_local(cmd, console)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)

    elapsed = time.time() - start

    console.print()
    if rc == 0:
        console.print(Panel(
            f"[bold green]Pipeline completed successfully[/bold green]\n"
            f"[dim]Elapsed: {elapsed:.1f}s[/dim]",
            border_style="green",
            padding=(1, 2),
        ))
    else:
        console.print(Panel(
            f"[bold red]Pipeline failed (exit code {rc})[/bold red]\n"
            f"[dim]Elapsed: {elapsed:.1f}s[/dim]",
            border_style="red",
            padding=(1, 2),
        ))

    sys.exit(rc)


if __name__ == "__main__":
    main()
