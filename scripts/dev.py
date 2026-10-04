#!/usr/bin/env python3
"""Supervise local services and terminate only processes started by this run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".dev" / "manager.json"


def identity(pid: int) -> str:
    """Read Linux process start time to guard against PID reuse."""
    return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]


def inspect(stop: bool) -> int:
    if not STATE.exists():
        print("Development supervisor is stopped.")
        return 0
    saved = json.loads(STATE.read_text())
    try:
        current = identity(saved["pid"])
    except FileNotFoundError:
        print("Development supervisor is stopped; stale state file retained.")
        return 0
    if current != saved["identity"]:
        raise RuntimeError("PID has been reused; refusing to signal an unrelated process")
    if stop:
        os.kill(saved["pid"], signal.SIGTERM)
        print("Stop requested for the development supervisor.")
    else:
        print(f"Development supervisor PID {saved['pid']} is running; logs: {STATE.parent}")
    return 0


def supervise() -> int:
    if STATE.exists():
        saved = json.loads(STATE.read_text())
        try:
            if identity(saved["pid"]) == saved["identity"]:
                raise RuntimeError("A development supervisor is already running")
        except FileNotFoundError:
            pass
    STATE.parent.mkdir(exist_ok=True)
    stopped = False
    children: list[subprocess.Popen] = []
    logs = []

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    STATE.write_text(json.dumps({"pid": os.getpid(), "identity": identity(os.getpid())}))
    try:
        for index, name in enumerate(("ingestion", "execution", "jobs", "inference")):
            environment = dict(os.environ, MFT_CONFIG=str(ROOT / "configs/config.yaml"), MFT_SERVICE_NAME=name,
                               MFT_METRICS_ADDR=f"127.0.0.1:{9090 + index}")
            if name == "inference":
                environment.setdefault("MFT_INFERENCE_MODEL", "heuristic")
                python = ROOT / "services/inference/.venv/bin/python"
                if not python.exists():
                    print("Inference environment is absent; run make inference-deps.")
                    continue
                command = [str(python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
                directory = ROOT / "services/inference"
            else:
                command = [str(ROOT / "bin" / name)]
                directory = ROOT
            handle = (STATE.parent / f"{name}.log").open("a")
            logs.append(handle)
            children.append(subprocess.Popen(command, cwd=directory, env=environment, stdout=handle, stderr=subprocess.STDOUT))
        print(f"Services started; logs: {STATE.parent}. Ctrl-C or make stop ends this run.", flush=True)
        while not stopped:
            failed = next((child for child in children if child.poll() is not None), None)
            if failed is not None:
                print(f"Service exited with code {failed.returncode}; stopping the group.", file=sys.stderr)
                return 1
            time.sleep(0.2)
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic() + 10
        for child in children:
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for handle in logs:
            handle.close()
        STATE.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stop", action="store_true")
    group.add_argument("--status", action="store_true")
    args = parser.parse_args()
    return inspect(args.stop) if args.stop or args.status else supervise()


if __name__ == "__main__":
    raise SystemExit(main())
