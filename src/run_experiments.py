#!/usr/bin/env python3
"""Command-line runner for Orange SetA experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

from exact_model import solve_exact
from heuristic import routes_to_json, solve_heuristic
from orange_data import load_instance
from validate import validate_routes


SUBMIT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = SUBMIT / "data" / "setA"


def machine_description() -> str:
    return (
        f"{platform.system()} {platform.release()}, "
        f"{platform.machine()}, Python {platform.python_version()}, "
        f"{os.cpu_count()} logical CPUs"
    )


def run_heuristic(args: argparse.Namespace) -> None:
    raw_rows: list[dict] = []
    solution_dir = Path(args.solutions)
    solution_dir.mkdir(parents=True, exist_ok=True)
    for number in args.instances:
        instance = load_instance(args.data, number)
        for run in range(1, args.runs + 1):
            result = solve_heuristic(
                instance,
                seed=args.seed + run - 1,
                max_iterations=args.iterations,
                candidate_count=args.candidates,
                time_limit=args.time_limit,
                periods_to_solve=args.periods,
            )
            validation = validate_routes(instance, result.routes)
            if not validation.feasible:
                raise RuntimeError("; ".join(validation.errors))
            solution_path = (
                solution_dir / f"{instance.name}-run{run:02d}-srpaths.json"
            )
            solution_path.write_text(
                json.dumps(routes_to_json(instance, result.routes), indent=2),
                encoding="utf-8",
            )
            row = {
                "instance": instance.name,
                "run": run,
                "vertices": instance.n,
                "links": instance.m,
                "demands": instance.num_demands,
                "periods": len(result.routes),
                "objective_mlu": validation.objective,
                "best_bound": "",
                "gap": "",
                "cpu_time_seconds": result.runtime,
                "bnb_nodes": "",
                "status": result.status,
                "feasible": validation.feasible,
                "budget_t1": validation.budget_used.get(1, ""),
                "configured_time_limit_seconds": args.time_limit,
                "seed": args.seed + run - 1,
                "max_iterations": args.iterations,
                "waypoint_candidates": args.candidates,
                "machine": machine_description(),
            }
            raw_rows.append(row)
            print(
                f"{instance.name} run {run}/{args.runs}: "
                f"MLU={validation.objective:.6f}, {result.runtime:.2f}s"
            )
    write_rows(Path(args.output), raw_rows)
    write_summary(Path(args.summary), raw_rows)


def run_exact(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    solution_dir = Path(args.solutions)
    solution_dir.mkdir(parents=True, exist_ok=True)
    for number in args.instances:
        instance = load_instance(args.data, number)
        for run in range(1, args.runs + 1):
            result = solve_exact(
                instance,
                num_periods=args.periods,
                time_limit=args.time_limit,
                mip_gap=args.gap,
            )
            feasible = result.routes is not None
            budget = ""
            objective = result.objective
            if feasible:
                validation = validate_routes(instance, result.routes or [])
                if not validation.feasible:
                    raise RuntimeError("; ".join(validation.errors))
                objective = validation.objective
                budget = validation.budget_used.get(1, "")
                solution_path = (
                    solution_dir
                    / (
                        f"{instance.name}-exact-{args.periods}period"
                        f"-run{run:02d}-srpaths.json"
                    )
                )
                solution_path.write_text(
                    json.dumps(
                        routes_to_json(instance, result.routes or []), indent=2
                    ),
                    encoding="utf-8",
                )
            rows.append(
                {
                    "instance": instance.name,
                    "run": run,
                    "vertices": instance.n,
                    "links": instance.m,
                    "demands": instance.num_demands,
                    "periods": args.periods,
                    "objective_mlu": objective if objective is not None else "",
                    "best_bound": (
                        result.dual_bound if result.dual_bound is not None else ""
                    ),
                    "gap": result.gap if result.gap is not None else "",
                    "cpu_time_seconds": result.runtime,
                    "total_time_seconds": result.total_runtime,
                    "bnb_nodes": (
                        result.node_count if result.node_count is not None else ""
                    ),
                    "status": result.status,
                    "feasible": feasible,
                    "budget_t1": budget,
                    "variables": result.num_variables,
                    "constraints": result.num_constraints,
                    "configured_time_limit_seconds": args.time_limit,
                    "machine": machine_description(),
                }
            )
            print(
                f"{instance.name} exact run {run}/{args.runs}: "
                f"{result.status}, objective={objective}, {result.runtime:.2f}s"
            )
    write_rows(Path(args.output), rows)
    write_summary(Path(args.summary), rows)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance"]), []).append(row)
    fields = [
        "instance",
        "runs",
        "vertices",
        "links",
        "demands",
        "mean_objective_mlu",
        "std_objective_mlu",
        "best_objective_mlu",
        "mean_cpu_time_seconds",
        "mean_gap",
        "mean_bnb_nodes",
        "status",
    ]
    summary_rows = []
    for name, group in grouped.items():
        objectives = [
            float(row["objective_mlu"])
            for row in group
            if row["objective_mlu"] != ""
        ]
        gaps = [float(row["gap"]) for row in group if row.get("gap", "") != ""]
        nodes = [
            float(row["bnb_nodes"])
            for row in group
            if row.get("bnb_nodes", "") != ""
        ]
        summary_rows.append(
            {
                "instance": name,
                "runs": len(group),
                "vertices": group[0]["vertices"],
                "links": group[0]["links"],
                "demands": group[0]["demands"],
                "mean_objective_mlu": (
                    statistics.mean(objectives) if objectives else ""
                ),
                "std_objective_mlu": (
                    statistics.stdev(objectives) if len(objectives) > 1 else 0.0
                ),
                "best_objective_mlu": min(objectives) if objectives else "",
                "mean_cpu_time_seconds": statistics.mean(
                    float(row["cpu_time_seconds"]) for row in group
                ),
                "mean_gap": statistics.mean(gaps) if gaps else "",
                "mean_bnb_nodes": statistics.mean(nodes) if nodes else "",
                "status": "|".join(sorted({str(row["status"]) for row in group})),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="SetA directory"
    )
    common.add_argument(
        "--instances", type=int, nargs="+", default=list(range(1, 21))
    )
    common.add_argument("--runs", type=int, default=10)
    common.add_argument("--periods", type=int, choices=(1, 2), default=2)
    common.add_argument("--time-limit", type=float, default=3600.0)
    common.add_argument(
        "--solutions", type=Path, default=SUBMIT / "solutions"
    )
    top = argparse.ArgumentParser(description=__doc__)
    sub = top.add_subparsers(dest="method", required=True)
    heuristic = sub.add_parser("heuristic", parents=[common])
    heuristic.add_argument("--seed", type=int, default=2026)
    heuristic.add_argument("--iterations", type=int, default=80)
    heuristic.add_argument("--candidates", type=int, default=20)
    heuristic.add_argument(
        "--output", type=Path, default=SUBMIT / "results" / "heuristic_raw.csv"
    )
    heuristic.add_argument(
        "--summary",
        type=Path,
        default=SUBMIT / "results" / "heuristic_summary.csv",
    )
    heuristic.set_defaults(function=run_heuristic)

    exact = sub.add_parser("exact", parents=[common])
    exact.add_argument("--gap", type=float, default=1e-4)
    exact.add_argument(
        "--output", type=Path, default=SUBMIT / "results" / "exact_raw.csv"
    )
    exact.add_argument(
        "--summary", type=Path, default=SUBMIT / "results" / "exact_summary.csv"
    )
    exact.set_defaults(function=run_exact)
    return top


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
