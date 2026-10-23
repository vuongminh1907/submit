#!/usr/bin/env python3
import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-6
COLUMNS = ["Inst.", "|V|", "|A|", "|D|", "Best", "Mean", "SD", "Solver (s)", "Gap", "Nodes", "Cert."]


def load_runs(results: Path, mode: str, number: int) -> list[dict[str, str]]:
    path = results / f"julia_{mode}_setA-{number:02d}.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> dict:
    objective = [float(r["objective"]) for r in rows]
    gap = [float(r["gap"]) for r in rows]
    return {
        "Inst.": "A" + rows[0]["instance"][-2:],
        "|V|": int(rows[0]["vertices"]),
        "|A|": int(rows[0]["links"]),
        "|D|": int(rows[0]["demands"]),
        "Best": min(objective),
        "Mean": statistics.mean(objective),
        "SD": statistics.stdev(objective) if len(objective) > 1 else 0.0,
        "Solver (s)": statistics.mean(float(r["cpu_time_seconds"]) for r in rows),
        "Gap": statistics.mean(gap),
        "Nodes": statistics.mean(float(r["bnb_nodes"]) for r in rows),
        "certified": sum(g <= TOLERANCE for g in gap),
        "runs": len(rows),
        "seeds": sorted(int(r["seed"]) for r in rows),
    }


def small(value: float, digits: int) -> str:
    if value < 1e-8:
        return "0"
    if value < 1e-3:
        mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{value:.4f}"


def time_text(value: float) -> str:
    return f"{value:.2f}" if value < 10 else f"{value:.1f}"


def formatted(row: dict) -> list[str]:
    return [
        row["Inst."], str(row["|V|"]), str(row["|A|"]), str(row["|D|"]),
        f"{row['Best']:.4f}", f"{row['Mean']:.4f}", small(row["SD"], 2), time_text(row["Solver (s)"]),
        small(row["Gap"], 3), f"{row['Nodes']:.1f}", f"{row['certified']}/{row['runs']}",
    ]


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(formatted(row) for row in rows)


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for row in rows:
        print("| " + " | ".join(formatted(row)) + " |")
    complete = sum(row["certified"] == row["runs"] for row in rows)
    print(f"\nCertified in all runs: {complete}/{len(rows)} instances")


def plot(path: Path, t0: list[dict], full: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    green, amber = "#117a47", "#b5530d"
    labels = [row["Inst."][1:] for row in t0]
    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    for offset, rows, alpha, hatch, name in (
        (-width / 2, t0, 1.0, None, "Period 0"),
        (width / 2, full, 0.55, "///", "Two periods"),
    ):
        for certified, color, suffix in ((True, green, "MLU-optimal"), (False, amber, "not certified")):
            idx = [i for i, row in enumerate(rows) if (row["certified"] == row["runs"]) == certified]
            ax.bar(
                [i + offset for i in idx], [rows[i]["Mean"] for i in idx], width,
                color=color, alpha=alpha, hatch=hatch, edgecolor="white", linewidth=0,
                label=f"{name}, {suffix}",
            )
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("SetA instance")
    ax.set_ylabel("Mean MLU")
    ax.set_ylim(0, 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Tables 1-2 and Figure 1 of the report from results/.")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--figure", type=Path, default=ROOT / "figures" / "mlu_compare.pdf")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    tables = {}
    for mode, title, output in (
        ("t0", "Table 1: period 0 (limit 1800 s)", "table1_t0.csv"),
        ("full", "Table 2: joint two-period model (limit 3600 s)", "table2_full.csv"),
    ):
        rows = [summarize(load_runs(args.results, mode, number)) for number in range(1, 21)]
        tables[mode] = rows
        write_table(args.results / output, rows)
        print_table(title, rows)
    if not args.no_figure:
        plot(args.figure, tables["t0"], tables["full"])
        print(f"\nFigure written to {args.figure}")


if __name__ == "__main__":
    main()
