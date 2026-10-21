#!/bin/sh
# Full SetA experiments, then the PDF report.
set -eu
cd "$(dirname "$0")"
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/run_experiments.jl selftest
python3 julia/run_batch.py
MPLCONFIGDIR=.matplotlib PYTHONPATH=src .venv/bin/python src/make_report.py
