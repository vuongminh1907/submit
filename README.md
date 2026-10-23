# Orange T-Adaptive Segment Routing: a Maximum-Link-Utilization approach

Code, experiment results and solutions for the course report
*The Orange T-Adaptive Segment Routing Problem: A Maximum-Link-Utilization Approach*
(Nguyen Trong Tuan, Vuong Dinh Minh, HUST).

- Report: [report/Official_Orange.pdf](report/Official_Orange.pdf)
- Slides: [report/Orange_T-ASR_slide.pdf](report/Orange_T-ASR_slide.pdf)

The implementation solves the scalar primary stage of the EURO/ROADEF 2026 T-ASR objective:
it minimizes the maximum link utilization (MLU) over all available links and all modeled
periods. MLU is the first component of the challenge's sorted lexicographic objective. Every
lower bound, gap and "MLU-optimal" label refers to this scalar objective, objective (3) of the report.

## Repository layout

```text
julia/
├── Project.toml, Manifest.toml   Julia 1.13.2, JuMP 1.32.1, HiGHS.jl 1.26.0
├── src/data.jl                   data model, period graphs G_t, split coefficients r, solution check
├── src/model.jl                  compact MIP, column generation, joint two-period master
├── run_experiments.jl            CLI: one or several instances, ten seeded runs each
├── run_batch.py                  all twenty Set A instances, one job at a time
└── test/runtests.jl              unit and consistency tests
scripts/make_tables.py            Tables 1-2 and Figure 1 of the report from results/
data/setA/                        the twenty Set A instances (net, tm, scenario JSON)
results/julia_{t0,full}_setA-XX.csv  one row per run (ten runs, seeds 2026-2035)
results/table1_t0.csv, table2_full.csv  Tables 1 and 2 of the report
solutions/                        segment paths (srpaths JSON) for every instance
figures/mlu_compare.pdf           Figure 1 of the report
report/                           submitted report, slides and LaTeX sources
```

## Correspondence with the report

### Data and model (report Section 2, slides 6-11)

| Report | Code |
|---|---|
| Sets and parameters $V, A, \omega, c, D, s, \tau, v, q_t, M, \kappa$ (Section 2.1) | `load_instance` in [julia/src/data.jl](julia/src/data.jl): explicit maps from JSON node and link identifiers; validation of positive metric and capacity |
| $A_t = A \setminus q_t$, $G_t = (V, A_t)$ | `build_period(inst, t)` sets `available` from the intervention links of period $t$ |
| $\ell^t_{ij}$, $\sigma^t_{ij}$ in logarithmic form | `_dijkstra` from every router; `logsigma` holds $\log\sigma$, combined by `_logaddexp` |
| Split coefficient $r^t_{ija}$, equation (1) | `build_period`: $r = \exp(\log\sigma_{iu} + \log\sigma_{vj} - \log\sigma_{ij})$ when $\ell_{iu} + \omega_a + \ell_{vj} = \ell_{ij}$; stored sparsely in `seg_arcs`, `seg_ratios` |
| Segment set $S_t$ (reachable ordered pairs $i \ne j$) | `reachable(g, i, j)` |
| Variables $x^{dt}_{ij}$, $L$, $y^d_{ij}$ (Section 2.2) | `solve_compact` in [julia/src/model.jl](julia/src/model.jl): `x[t, d, p]` binary, `L >= 0`, `y[d, s] >= 0` |
| Objective (3): $\min L$ | `@objective(model, Min, L)` in every master |
| Flow conservation (4) | `out_expr - in_expr == rhs` |
| Degree inequalities (5) | `out_expr <= 1{i != tau_d}`, `in_expr <= 1{i != s_d}` |
| Segment limit (6) | `sum(x[t, d, :]) <= maxseg` |
| Link utilization (7) | `loads[a] <= 0`, with `loads[a]` = $\sum r v x - c_a L$ |
| Absolute change (8) and budget (9) | `y >= second - first`, `y >= first - second`, `sum(y) <= budget`; only when two periods are modeled |
| Load and MLU (2)-(3) of a routing | `maximum_load`, `validate_solution` in `data.jl` |

### Solution method (report Section 3, slides 13-20)

| Report | Code |
|---|---|
| 3.1 Formulation selection: compact MIP if $\lvert D\rvert\lvert T\rvert\lvert V\rvert(\lvert V\rvert-1) \le 100\,000$ | `estimated_binaries`, `COMPACT_VARIABLE_LIMIT`, `solve_instance` |
| 3.2 Restricted master (10)-(11), initial direct segment | `solve_period_columns`: convexity rows (10), load rows (11), column `[s_d, tau_d]` per demand |
| 3.2 Pricing: bounded shortest path with at most $M$ segments, dynamic program on $M$ layers | `_best_path` (exact $M$-layer DP); `_dense_path` is used only when all weights are nonnegative and its path already has at most $M$ segments, which yields the same minimum |
| 3.2 Reduced cost from duals $\pi$, $\mu$ | `_unit_costs`, `_price`; the formula is checked against HiGHS `reduced_cost` on the first round |
| 3.2 Lagrangian bound $LB_t = z_{RMP} + \sum_d \min\{0, \bar c^*_{dt}\}$ | `lower = max(lower, lp_obj + correction)` |
| 3.2 Cycle removal before inserting a column | `_without_cycles` |
| 3.2 Binary restricted master for the incumbent | `set_binary` on all columns, then `optimize!` |
| 3.3 Separate column generation per period, 40% of the limit each | `solve_joint_columns`, `PERIOD_TIME_SHARE = 0.4` |
| 3.3 Initial bound $\max\{LB_0, LB_1\}$ | `lower = max(first.lower, second.lower, 0.0)` |
| 3.3 Pools enriched with waypoint paths and routes valid in both period graphs | `_enrich_pool`, `_add_to_valid_periods!`, `build_joint_master` |
| 3.3 Joint master: path selection per demand and period, load rows of both periods, (8)-(9) on columns | `JointMaster`, `add_joint_column!`: $\bar x$ of a segment is the sum of the $z$ of the paths that contain it |
| 3.3 Initial binary joint master warm-started from a repaired budget-feasible pair | `_fit_budget`, `_solve_joint_mip!`, `JOINT_WARMUP_SHARE` |
| 3.3 Joint pricing with load, selection and change-row duals; zero penalty for unseen segments | `_joint_pricing!`, `_joint_reduced_cost` |
| 3.3 $LB_{joint} = z^{LP}_{joint} + \sum_t\sum_d \min\{0, \bar c^*_{dt}\}$ | `lower = max(lower, lp_obj + correction)` in `_joint_pricing!` |
| 3.3 Safeguards: 40 rounds, 25 000 columns, stop after six rounds without LP improvement | `JOINT_PRICING_ROUNDS`, `JOINT_PRICING_COLUMNS`, `JOINT_PRICING_STALL` |
| 3.3 $LB = \max\{LB_0, LB_1, LB_{joint}\}$, final binary joint master | `solve_joint_columns` |
| 3.4 Remaining time: one-waypoint, random two-waypoint and random-cost paths; warm start; new seed | `_generate_columns!`, `_diversify_one_waypoint!`, `_diversify_two_waypoints!`, `_diversify_random_paths!`, `_improve_period!` |
| 3.4 Two periods: period-level diversification disabled; candidates added to every pool where feasible; joint MIP re-solved | `solve_period_columns(...; improve=false)`, improvement loop of `solve_joint_columns` |
| 3.4 Stop rules: gap closed, time limit, or no new path and an immediate integer solve | `certified(...)`, `elapsed >= remaining - 2`, `added == 0 && elapsed < 1` |
| 3.5 Independent check: terminals, reachability, segment count, repeated vertices, reconfiguration cost, recomputed loads | `validate_solution`, `route_ok`, `reconfiguration_cost`; `run_experiments.jl` also checks that the reported MLU equals the recomputed one |
| 3.5 Gap $\max\{0, 1 - LB/UB\}$; MLU-optimal if gap $\le 10^{-6}$ | `_gap`, `certified`, `CERTIFICATION_TOLERANCE = 1e-6` |

The compact MIP takes its lower bound from HiGHS `objective_bound`. HiGHS uses
`mip_rel_gap = 1e-6` and `mip_abs_gap = 0`, so its stopping rule is the same as the report's
certification tolerance. Bounds from column generation are Lagrangian bounds, which are valid
at every pricing round.

### Experimental protocol (report Section 4)

| Report | Code |
|---|---|
| Period 0 alone, no budget (Step 6) | mode `t0` |
| Periods 0 and 1 with the budget (Steps 7-9) | mode `full` |
| Limits 1800 s and 3600 s | defaults of `TIME_LIMIT` in `run_experiments.jl` and `run_batch.py` |
| Ten runs, seeds 2026-2035 | `RUNS=10`, `SEED=2026`; run $k$ uses seed $2025 + k$ |
| One solver thread, jobs run sequentially | `threads = 1` in `_configure!`; `run_batch.py` starts one Julia process at a time |
| Solver time counts only `optimize!` calls, summed over calls | `_timed_optimize!`; column `cpu_time_seconds` |
| Node counts summed over solver calls | column `bnb_nodes` |

## Requirements

- Julia 1.10 or newer (experiments used 1.13.2); packages are pinned in `julia/Manifest.toml`.
- Python 3 with `matplotlib` for `scripts/make_tables.py` only (`pip install -r requirements.txt`).

## Usage

All commands are run from the repository root.

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/run_experiments.jl selftest
```

The tests check equation (1) on the example of slide 7 (four shortest paths, three of them
through the arc, $r = 3/4$), the intervention graph, the reconfiguration-cost examples of slide 10,
the solution checker, and agreement between the compact MIP and column generation on setA-01
(period 0) and setA-02 (two periods), including the validity of $LB_{joint}$.

One instance, ten runs:

```bash
julia --project=julia julia/run_experiments.jl t0 1      # period 0, 1800 s per run
julia --project=julia julia/run_experiments.jl full 1    # two periods, 3600 s per run
```

Several instances, a shorter limit, or another dataset:

```bash
TIME_LIMIT=600 RUNS=1 julia --project=julia julia/run_experiments.jl t0 1 2 3
DATA_DIR=/abs/path/to/setA RESULTS_DIR=/tmp/out SOLUTIONS_DIR=/tmp/out \
  julia --project=julia julia/run_experiments.jl full 10
```

The complete protocol (both experiments, all instances, ten seeds) followed by the tables:

```bash
./run_all.sh
```

Tables 1-2 and Figure 1 from the stored results:

```bash
python3 scripts/make_tables.py
```

## Output

`results/julia_{t0,full}_setA-XX.csv` has one row per run:

| Column | Meaning |
|---|---|
| `objective` | checked MLU (UB) |
| `best_bound` | valid lower bound LB on the scalar MLU |
| `gap` | $\max\{0, 1 - LB/UB\}$ |
| `cpu_time_seconds` | time inside `optimize!`, summed over calls |
| `bnb_nodes` | branch-and-bound nodes, summed over calls |
| `status` | `Optimal` if gap $\le 10^{-6}$, otherwise `Not Optimal` |
| `method` | `compact-mip` or `column-generation` |

`solutions/setA-XX-julia-{t0,full}-srpaths.json` stores the best routing over the ten runs in
the challenge format: `{"srpaths": [{"d": demand, "t": period, "w": [waypoints]}]}`. Demands
routed on their direct segment have no entry. Per-seed routings are written as
`setA-XX-julia-{t0,full}-seedNNNN-srpaths.json`.

## Results

`scripts/make_tables.py` reproduces Tables 1 and 2 of the report from `results/`:

- Period 0: 12 of 20 instances are certified MLU-optimal in all ten runs.
- Two periods: 13 of 20 instances are certified in all ten runs; setA-03 is certified in 8 of 10.
- Every run of every instance produced a feasible, independently checked routing.

## Scope and limitations

- Only the scalar MLU (3) is optimized and certified, not the later ranks of the lexicographic objective.
- $r$ follows the course note's formula (1), which assigns equal weight to complete shortest paths.
  It can differ from router-local ECMP splitting on an asymmetric shortest-path DAG.
- The path-based integer master contains only generated columns. Its incumbents are feasible
  and its bounds are valid, but it is not a branch-and-price method.
- Solver time excludes graph construction, coefficient computation, pricing and validation,
  as required by the project instructions.
