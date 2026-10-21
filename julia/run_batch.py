#!/usr/bin/env python3
"""Run the Julia experiments with a fixed number of concurrent processes."""

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JULIA = ROOT / "julia"
SCRIPT = JULIA / "run_experiments.jl"
LOGS = ROOT / "logs"


def run_mode(mode: str, time_limit: str, runs: str = "10", jobs: int = 6) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    pending = list(range(1, 21))
    running: list[tuple[int, subprocess.Popen, object]] = []
    while pending or running:
        while pending and len(running) < jobs:
            number = pending.pop(0)
            log_path = LOGS / f"{mode}-{number:02d}.log"
            handle = log_path.open("w")
            env = os.environ.copy()
            env["TIME_LIMIT"] = time_limit
            env["RUNS"] = runs
            proc = subprocess.Popen(
                ["julia", f"--project={JULIA}", str(SCRIPT), mode, str(number)],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running.append((number, proc, handle))
            print(f"START {mode} {number}", flush=True)
        still = []
        for number, proc, handle in running:
            code = proc.poll()
            if code is None:
                still.append((number, proc, handle))
            else:
                handle.close()
                print(f"DONE {mode} {number} exit={code}", flush=True)
        running = still
        if running:
            time.sleep(2)
    print(f"ALL DONE {mode}", flush=True)


if __name__ == "__main__":
    run_mode("t0", "1800")
    run_mode("full", "3600")
