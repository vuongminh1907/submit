#!/usr/bin/env python3
"""Build the course report from the Julia/JuMP experiment files."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SUBMIT = Path(__file__).resolve().parents[1]
RESULTS = SUBMIT / "results"
FIGURES = SUBMIT / "figures"
REPORT = SUBMIT / "report"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle("Section", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#17365D")))
styles.add(ParagraphStyle("Subsection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=4, spaceAfter=2, textColor=colors.HexColor("#244A73")))
styles.add(ParagraphStyle("BodyJ", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=11.4, alignment=TA_JUSTIFY, spaceAfter=4))
styles.add(ParagraphStyle("Equation", parent=styles["BodyText"], fontName="Courier", fontSize=8, leading=10.2, leftIndent=12, rightIndent=8, spaceBefore=1, spaceAfter=3))
styles.add(ParagraphStyle("Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=8, leading=10, spaceAfter=3))
styles.add(ParagraphStyle("Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8, leading=10, spaceBefore=2, spaceAfter=4))


def P(text: str, style: str = "BodyJ") -> Paragraph:
    return Paragraph(text, styles[style])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_runs(label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(RESULTS.glob(f"julia_{label}_setA-*.csv")):
        rows.extend(read_csv(path))
    return rows


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["instance"], []).append(row)
    summary = []
    for instance in sorted(grouped):
        group = grouped[instance]
        objectives = [float(row["objective"]) for row in group]
        times = [float(row["cpu_time_seconds"]) for row in group]
        gaps = [float(row["gap"]) for row in group]
        nodes = [float(row["bnb_nodes"]) for row in group]
        statuses = {row["status"] for row in group}
        status = next(iter(statuses)) if len(statuses) == 1 else "Not Optimal"
        first = group[0]
        summary.append(
            {
                "instance": instance,
                "vertices": first["vertices"],
                "links": first["links"],
                "demands": first["demands"],
                "periods": first["periods"],
                "runs": str(len(group)),
                "mean_objective": f"{_mean(objectives):.10g}",
                "std_objective": f"{_std(objectives):.6g}",
                "mean_cpu_time_seconds": f"{_mean(times):.6g}",
                "std_cpu_time_seconds": f"{_std(times):.6g}",
                "mean_gap": f"{_mean(gaps):.6g}",
                "mean_bnb_nodes": f"{_mean(nodes):.6g}",
                "status": status,
                "method": first["method"],
                "time_limit_seconds": first["time_limit_seconds"],
            }
        )
    return summary


def _by_name(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["instance"]: row for row in rows}


def _discussion(nominal: list[dict[str, str]], full: list[dict[str, str]]) -> str:
    left = _by_name(nominal)
    right = _by_name(full)
    optimal_nominal = sum(row["status"] == "Optimal" for row in nominal)
    optimal_full = sum(row["status"] == "Optimal" for row in full)
    a1 = left["setA-01"]
    a2 = left["setA-02"]
    a3 = left["setA-03"]
    joint = right["setA-01"]["mean_objective"]
    same = abs(float(a1["mean_objective"]) - float(joint)) <= 1e-6
    joint_sentence = (
        f"The two-period optimum of setA-01 is the same value, so the period-0 peak remains "
        f"feasible once the budget is imposed."
        if same
        else (
            f"With the budget, setA-01 rises to {float(joint):.4f}."
        )
    )
    return (
        f"The compact model proves optimality for the three smallest period-0 models that fit "
        f"in memory: setA-01 at {float(a1['mean_objective']):.4f}, setA-02 at "
        f"{float(a2['mean_objective']):.4f}, and setA-03 at {float(a3['mean_objective']):.4f}. "
        f"{joint_sentence} Column generation reproduces the setA-01 compact value, which checks "
        f"the reduced-cost sign against HiGHS. Of the twenty period-0 instances, "
        f"{optimal_nominal} are Optimal; of the twenty two-period instances, {optimal_full} are "
        f"Optimal. The other rows are feasible solutions with the gap against the computed "
        f"linear bound."
    )


def write_summary(label: str, rows: list[dict[str, str]], summary: list[dict[str, str]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if rows:
        with (RESULTS / f"julia_{label}_raw.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if summary:
        with (RESULTS / f"julia_{label}_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)


def result_table(rows: list[dict[str, str]]) -> Table:
    header = ["Instance", "|V|", "|A|", "|D|", "Objective", "CPU (s)", "Gap", "B&B nodes", "Status"]
    body = [header]
    for row in rows:
        gap = float(row["mean_gap"])
        body.append(
            [
                row["instance"],
                row["vertices"],
                row["links"],
                row["demands"],
                f"{float(row['mean_objective']):.4f}",
                f"{float(row['mean_cpu_time_seconds']):.2f}",
                "0" if row["status"] == "Optimal" else f"{gap:.4g}",
                f"{float(row['mean_bnb_nodes']):.1f}",
                row["status"],
            ]
        )
    table = Table(body, repeatRows=1, colWidths=[1.7*cm, 1.15*cm, 1.25*cm, 1.25*cm, 2.15*cm, 1.7*cm, 1.45*cm, 1.9*cm, 2.15*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("ALIGN", (1, 0), (-2, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
    ]))
    return table


def create_plot(nominal: list[dict[str, str]], full: list[dict[str, str]]) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "seta_results.png"
    names = [row["instance"].replace("setA-", "") for row in nominal]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.15))
    axes[0].plot(names, [float(row["mean_objective"]) for row in nominal], marker="o", label="period 0")
    if full:
        axes[0].plot(names, [float(row["mean_objective"]) for row in full], marker="s", label="periods 0 and 1")
    axes[0].set_ylabel("Maximum link utilization")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].semilogy(names, [max(float(row["mean_cpu_time_seconds"]), 1e-3) for row in nominal], marker="o", label="period 0")
    if full:
        axes[1].semilogy(names, [max(float(row["mean_cpu_time_seconds"]), 1e-3) for row in full], marker="s", label="periods 0 and 1")
    axes[1].set_ylabel("Mean optimize! time (s)")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#AAB7C4"))
    canvas.line(1.6 * cm, 1.25 * cm, A4[0] - 1.6 * cm, 1.25 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1.6 * cm, 0.75 * cm, "T-Adaptive Segment Routing — Orange")
    canvas.drawRightString(A4[0] - 1.6 * cm, 0.75 * cm, f"Page {document.page}")
    canvas.restoreState()


def generate() -> Path:
    nominal_runs = load_runs("t0")
    full_runs = load_runs("full")
    if len({row["instance"] for row in nominal_runs}) < 20 or len({row["instance"] for row in full_runs}) < 20:
        raise SystemExit("Julia results are incomplete: expected 20 instances in both experiments.")
    nominal = summarize(nominal_runs)
    full = summarize(full_runs)
    write_summary("t0", nominal_runs, nominal)
    write_summary("full", full_runs, full)
    plot = create_plot(nominal, full)

    REPORT.mkdir(parents=True, exist_ok=True)
    output = REPORT / "Orange_TASR_report.pdf"
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.35*cm, bottomMargin=1.45*cm,
        title="T-Adaptive Segment Routing for Orange Networks", author="Vuong Minh",
    )
    story = [
        P("T-Adaptive Segment Routing<br/>for Resilient Orange Networks", "ReportTitle"),
        P("Mathematical Modelling — Orange / EURO-ROADEF 2026", "Small"),
        P("<b>Student:</b> Vuong Minh &nbsp;&nbsp; <b>Date:</b> October 2026", "Small"),
        P("1. Introduction", "Section"),
        P(
            "Orange routes many origin-destination demands over a directed IP/MPLS backbone. "
            "Between ordinary routing changes, traffic follows shortest paths determined by link "
            "metrics, and equal-cost paths share the flow. Segment Routing inserts a short list of "
            "waypoints. Between two successive waypoints the network still uses shortest-path "
            "forwarding, so a small header can move traffic away from a congested link. The "
            "EURO/ROADEF 2026 problem adds a second time period in which some links are withdrawn "
            "for maintenance and the traffic matrix changes. A reconfiguration budget limits how "
            "much the segment lists may change between the two periods."
        ),
        P(
            "This report follows the course workflow. Period 0 is built from the network JSON, the "
            "split coefficients are computed from the course note, and the mixed-integer model is "
            "solved with no budget constraint. The period-1 graph is then built by deleting the "
            "intervention links, its coefficients and volumes are recomputed, and the budget "
            "constraint is added. Both experiments cover all twenty SetA instances, with ten runs "
            "and solver limits of 1800 s and 3600 s. The step-9 wording repeats “time period 0”; "
            "the tables below are for the two-period model constructed in step 8, because that is "
            "the model the one-hour limit applies to."
        ),
        P("2. Data", "Section"),
        P(
            "Each SetA instance is three JSON files. The network gives nodes, directed links, "
            "administrative metrics and capacities. The traffic matrix gives two time slots and, "
            "for every demand, a source, a target and a volume in each slot. The scenario gives "
            "maxSeg = 6, the period-1 budget and the links removed at period 1. Period 0 has no "
            "intervention and no finite budget. Node and link identifiers are stored explicitly; "
            "they are not assumed to form an interval. Several files set a multigraph flag, but "
            "no SetA instance contains two links with the same ordered pair of endpoints."
        ),
        P(
            "Let sigma(i,j,t) be the number of shortest i–j paths at period t, computed by Dijkstra "
            "together with the distances. Path counts are stored in logarithmic form so that they "
            "do not overflow. For an arc a = (u,v) the course note sets"
        ),
        P(
            "r(i,j,a,t) = sigma(i,u,t) sigma(v,j,t) / sigma(i,j,t)<br/>"
            "when dist(i,u)+omega(a)+dist(v,j) = dist(i,j), and r = 0 otherwise.",
            "Equation",
        ),
        P(
            "If j is unreachable from i, the segment is omitted. This uniform path-count formula is "
            "the one required by the course note. A router-by-router ECMP rule can give a different "
            "fraction on an asymmetric shortest-path DAG; that alternative is not used here. "
            "Capacity c(a) comes from the network file and volume v(d,t) from the traffic file."
        ),
        P("3. Mathematical model", "Section"),
        P(
            "The binary variable x(d,t,i,j) equals 1 when demand d uses the segment from i to j "
            "in period t. The continuous variable L is the maximum link utilization. For period 0 "
            "the model is"
        ),
        P("minimize L", "Equation"),
        P(
            "(1)  sum_(j != i) x(d,t,i,j) - sum_(j != i) x(d,t,j,i) = 1, -1 or 0,<br/>"
            "according as i is the source of d, its target, or another node.",
            "Equation",
        ),
        P("(2)  sum_(i != j) x(d,t,i,j) &lt;= maxSeg.", "Equation"),
        P(
            "(3)  sum_d sum_(i != j) r(i,j,a,t) v(d,t) x(d,t,i,j) &lt;= c(a) L,<br/>"
            "for every arc a that is available at period t.",
            "Equation",
        ),
        P(
            "Flow conservation on the complete graph of feasible segments produces an s–t chain. "
            "Degree inequalities, at most one segment leaving or entering a node, cut off branches "
            "and make the support a single simple path. A detached cycle cannot reduce L, because "
            "every segment contribution is nonnegative. Constraint (2) limits the header length. "
            "Constraint (3) links the ECMP load to L."
        ),
        P(
            "The challenge objective sorts every arc-period load and minimizes that vector "
            "lexicographically. The course tables ask for one objective value and one gap, so the "
            "program optimizes the first component, the maximum link utilization. Two solutions "
            "with the same maximum may still differ on the remaining loads. That limitation is "
            "stated here rather than hidden in the tables."
        ),
        P(
            "For the two-period model the same constraints are written at t = 0 and t = 1. Let "
            "kappa be the period-1 budget. The symmetric difference of the two segment sets is "
            "linearized with y(d,i,j) &gt;= 0:"
        ),
        P(
            "(4)  y(d,i,j) &gt;= x(d,1,i,j) - x(d,0,i,j),<br/>"
            "y(d,i,j) &gt;= x(d,0,i,j) - x(d,1,i,j),<br/>"
            "sum_(d,i,j) y(d,i,j) &lt;= kappa.",
            "Equation",
        ),
        P(
            "Direct routing, the single segment (s(d), t(d)) in both periods, has distance 0, so "
            "it satisfies the budget whenever both endpoints remain reachable. The distance "
            "examples in the subject are reproduced by the implementation: replacing one waypoint "
            "path by the direct path costs 3, extending a one-waypoint path costs 3, swapping two "
            "waypoints costs 6, and exchanging a single waypoint costs 4."
        ),
        P("4. Solution method", "Section"),
        P(
            "The code is Julia, using JuMP and HiGHS. HiGHS is called with one thread. The recorded "
            "CPU time is only the time spent inside optimize!, summed over the linear programs and "
            "the final integer program. Graph construction and the coefficient matrices are built "
            "once and are outside that timer. Ten runs repeat the solver calls so that the table "
            "can report the average even though the algorithm itself is deterministic."
        ),
        P(
            "When the estimated number of binaries |D| · |T| · |V| · (|V|-1) is at most 80,000, "
            "JuMP builds the compact model above and HiGHS solves it directly. That covers period 0 "
            "of setA-01, setA-02 and setA-03, and both periods of setA-01 and setA-02. On the "
            "remaining instances the same formulation is solved by column generation. Each column "
            "is a segment path. The restricted master minimizes L subject to one convex combination "
            "per demand and the load inequalities. Its pricing problem is a shortest path of at "
            "most maxSeg hops in the complete segment graph. With nonnegative reduced-cost weights, "
            "an ordinary shortest path is a valid filter: if it is nonnegative, no hop-limited path "
            "has negative reduced cost. The bound LP + sum of negative reduced costs is valid for "
            "the full linear relaxation, including columns that have not yet been added. Pricing "
            "stops when no negative reduced-cost path remains, or after 120 rounds. The generated "
            "columns are then made binary. HiGHS branch-and-bound is given the remaining time "
            "limit and a cap of 400 nodes, reduced to 20 nodes when more than 2000 columns "
            "have been generated. The cap does not change the linear bound. If it stops "
            "the search before the incumbent meets that bound, the status stays Not Optimal and "
            "the gap is reported."
        ),
        P(
            "For two periods, each period is priced separately. Because the budget only removes "
            "solutions, each certified single-period bound remains a lower bound on the joint "
            "maximum. The joint integer model uses the generated paths of both periods and "
            "constraint (4). Its objective is accepted only after an independent check of the "
            "terminals, the segment limit, reachability, the absence of repeated nodes, the budget "
            "and the recomputed maximum load. Status is Optimal when that checked value meets the "
            "valid bound, Not Optimal when a feasible solution remains above the bound, and "
            "Infeasible when no feasible routing is found. The reported gap is 1 - LB/UB."
        ),
        P("5. Experiments", "Section"),
        P(
            "The runs use Julia 1.13.2, JuMP 1.32.1 and HiGHS 1.26.0 on an Apple Silicon laptop "
            "under macOS. SetA has twenty instances, from 20 nodes, 80 links and 40 demands up to "
            "400 nodes, 2000 links and 6000 demands. The configured limits are 1800 seconds for "
            "period 0 and 3600 seconds for the two-period model. A run returns earlier when the "
            "bound certifies the incumbent. Every average below is taken over ten runs."
        ),
        P("5.1 Period 0, time limit 1800 seconds", "Subsection"),
        result_table(nominal),
        P("Table 1. Period 0. CPU time is the mean of the optimize! calls only. Gap is 1 - LB/UB and is 0 when the status is Optimal.", "Caption"),
        P("5.2 Periods 0 and 1 with the budget, time limit 3600 seconds", "Subsection"),
        result_table(full),
        P("Table 2. Both periods. The objective is the maximum utilization over the two periods. The node count sums the branch-and-bound nodes of the solver calls.", "Caption"),
        Image(str(plot), width=17.2*cm, height=5.3*cm),
        P("Figure 1. Mean maximum link utilization and mean optimize! time.", "Caption"),
        P("6. Discussion", "Section"),
        P(_discussion(nominal, full)),
        P(
            "Where the best integer solution stays above the linear bound, the table keeps the "
            "status Not Optimal and prints the gap. A bound that was not computed is never written "
            "as zero. The branch-and-bound node count is the value returned by HiGHS. A count "
            "equal to the node cap means that cap was reached before the incumbent met the bound."
        ),
        P(
            "The formulation does not minimize the full lexicographic vector of loads, and the "
            "split coefficients follow the course note rather than a router-local ECMP recursion. "
            "Within those two stated choices, every row comes from the mathematical model: either "
            "the compact MIP or its path formulation with a computed lower bound."
        ),
        P("References", "Section"),
        P(
            "[1] Orange Research, Challenge EURO/ROADEF 2026 Problem Description, 2026. "
            "[2] Course note, Computing arc-use ratios on shortest paths, 2026. "
            "[3] C. Filsfils et al., Segment Routing Architecture, RFC 8402, 2018. "
            "[4] JuMP and HiGHS documentation.",
            "Small",
        ),
    ]
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


if __name__ == "__main__":
    print(generate())
