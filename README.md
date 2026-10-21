# Orange T-Adaptive Segment Routing

Julia/JuMP/HiGHS implementation for the Orange project. The solver minimizes
maximum link utilization (MLU), using the split coefficients defined in
`calculating_r.pdf`.

## Repository layout

```text
julia/
├── Project.toml
├── src/data.jl              # input data and split coefficients
├── src/model.jl             # compact MIP and column generation
├── run_experiments.jl       # selected instances
└── run_batch.py             # all SetA instances
data/setA/                   # bundled SetA JSON instances
results/                     # experiment CSV files
solutions/                   # waypoint JSON solutions
src/make_report.py           # report generator
report/Orange_TASR_report.pdf
```

## Requirements

- Julia 1.10 or newer
- SetA is included under `data/setA`, with the `net`, `tm`, and `scenario`
  JSON files required for every instance.
- For a different dataset, set `DATA_DIR` to its absolute path.

## Setup

Run these commands from the repository root:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/run_experiments.jl selftest
```

## Run one instance

Period `t=0` only, one run of `setA-01`:

```bash
TIME_LIMIT=1800 RUNS=1 julia --project=julia julia/run_experiments.jl t0 1
```

Both periods with the reconfiguration budget:

```bash
TIME_LIMIT=3600 RUNS=1 julia --project=julia julia/run_experiments.jl full 1
```

Replace the final number with the desired instance:

```bash
# setA-02
TIME_LIMIT=1800 RUNS=1 julia --project=julia julia/run_experiments.jl t0 2

# setA-10
TIME_LIMIT=3600 RUNS=1 julia --project=julia julia/run_experiments.jl full 10
```

Multiple instances can be supplied in one command:

```bash
TIME_LIMIT=1800 RUNS=10 julia --project=julia julia/run_experiments.jl t0 1 2 3
```

To use data from another directory:

```bash
DATA_DIR=/absolute/path/to/setA TIME_LIMIT=1800 RUNS=1 \
  julia --project=julia julia/run_experiments.jl t0 1
```

Outputs are written to:

```text
results/julia_t0_setA-01.csv
results/julia_full_setA-01.csv
solutions/setA-01-julia-t0-srpaths.json
solutions/setA-01-julia-full-srpaths.json
```

## Run all SetA instances

```bash
python3 julia/run_batch.py
```

This runs `setA-01` through `setA-20` ten times: 1800 seconds for `t0` and
3600 seconds for `full`. Adjust the `jobs` value in `julia/run_batch.py` to
match available RAM and CPU resources.

Instances below 100,000 estimated binary variables use the compact MIP.
Larger instances use column generation and retain a valid lower bound.
CPU time in result files counts only calls to `optimize!`.
