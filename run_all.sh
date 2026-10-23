#!/bin/sh
set -eu
cd "$(dirname "$0")"
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/run_experiments.jl selftest
python3 julia/run_batch.py
MPLCONFIGDIR=.matplotlib python3 scripts/make_tables.py
