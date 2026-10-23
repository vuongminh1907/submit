#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JULIA = ROOT / "julia"
SCRIPT = JULIA / "run_experiments.jl"
LOGS = ROOT / "logs"
EXPERIMENTS = (("t0", "1800"), ("full", "3600"))


def run_mode(mode: str, time_limit: str, runs: str = "10", seed: str = "2026") -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"TIME_LIMIT": time_limit, "RUNS": runs, "SEED": seed})
    for number in range(1, 21):
        log_path = LOGS / f"{mode}-{number:02d}.log"
        print(f"START {mode} setA-{number:02d}", flush=True)
        with log_path.open("w") as handle:
            code = subprocess.call(
                ["julia", f"--project={JULIA}", str(SCRIPT), mode, str(number)],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        print(f"DONE {mode} setA-{number:02d} exit={code}", flush=True)
        if code != 0:
            sys.exit(code)


if __name__ == "__main__":
    modes = sys.argv[1:] or [mode for mode, _ in EXPERIMENTS]
    for mode, time_limit in EXPERIMENTS:
        if mode in modes:
            run_mode(mode, time_limit)
