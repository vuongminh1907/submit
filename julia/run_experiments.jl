using JSON3
using JuMP
using HiGHS
using Random
using Test

include(joinpath(@__DIR__, "src", "data.jl"))
include(joinpath(@__DIR__, "src", "model.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA = normpath(get(ENV, "DATA_DIR", joinpath(ROOT, "data", "setA")))
const RESULTS = normpath(get(ENV, "RESULTS_DIR", joinpath(ROOT, "results")))
const SOLUTIONS = normpath(get(ENV, "SOLUTIONS_DIR", joinpath(ROOT, "solutions")))

function selftest()
    include(joinpath(@__DIR__, "test", "runtests.jl"))
end

function solution_json(inst::Instance, routes)
    entries = Dict{String,Any}[]
    for (t, period_routes) in enumerate(routes)
        for (d, route) in enumerate(period_routes)
            waypoints = [inst.node_ids[node] for node in route[2:end-1]]
            isempty(waypoints) && continue
            push!(entries, Dict("d" => d - 1, "t" => t - 1, "w" => waypoints))
        end
    end
    return Dict("srpaths" => entries)
end

function write_solution(path::AbstractString, inst::Instance, routes)
    open(path, "w") do io
        JSON3.pretty(io, solution_json(inst, routes))
    end
end

function run_instance(
    number::Int, periods::Int, time_limit::Float64, runs::Int, base_seed::Int,
)
    instance = load_instance(DATA, number)
    println("$(instance.name): building $periods period graph(s)")
    flush(stdout)
    graphs = PeriodGraph[build_period(instance, t) for t in 0:(periods - 1)]
    rows = String[]
    best = nothing
    label = periods == 1 ? "t0" : "full"
    for run in 1:runs
        seed = base_seed + run - 1
        result = solve_instance(instance, graphs, time_limit; random_seed=seed)
        if isfinite(result.objective)
            checked = validate_solution(instance, graphs, result.routes)
            abs(checked - result.objective) <= 1e-9 * max(1.0, checked) ||
                error("Reported MLU differs from the independently checked MLU.")
            write_solution(
                joinpath(SOLUTIONS, "$(instance.name)-julia-$label-seed$seed-srpaths.json"), instance, result.routes,
            )
        end
        if best === nothing || (isfinite(result.objective) && result.objective < best.objective)
            best = result
        end
        push!(rows, join([
            instance.name,
            run,
            periods,
            instance.n,
            length(instance.arcs),
            ndemands(instance),
            result.objective,
            result.bound,
            result.gap,
            result.optimize_time,
            result.nodes,
            result.status,
            result.method,
            time_limit,
            seed,
        ], ","))
        println(
            "$(instance.name) run $run/$runs: $(result.status), ",
            "MLU=$(result.objective), gap=$(result.gap), nodes=$(result.nodes), ",
            "optimize=$(round(result.optimize_time; digits=2))s, method=$(result.method), seed=$seed",
        )
        flush(stdout)
    end
    if best !== nothing && isfinite(best.objective)
        write_solution(joinpath(SOLUTIONS, "$(instance.name)-julia-$label-srpaths.json"), instance, best.routes)
    end
    return rows
end

function main(args)
    isempty(args) && error("Usage: run_experiments.jl selftest | {t0|full} instances...")
    args[1] == "selftest" && return selftest()
    periods = args[1] == "t0" ? 1 : args[1] == "full" ? 2 : error("Unknown mode $(args[1])")
    numbers = Int[parse(Int, arg) for arg in args[2:end]]
    time_limit = parse(Float64, get(ENV, "TIME_LIMIT", periods == 1 ? "1800" : "3600"))
    runs = parse(Int, get(ENV, "RUNS", "10"))
    base_seed = parse(Int, get(ENV, "SEED", "2026"))
    mkpath(RESULTS)
    mkpath(SOLUTIONS)
    label = periods == 1 ? "t0" : "full"
    for number in numbers
        rows = run_instance(number, periods, time_limit, runs, base_seed)
        path = joinpath(RESULTS, "julia_$(label)_setA-$(lpad(string(number), 2, '0')).csv")
        open(path, "w") do io
            println(io, "instance,run,periods,vertices,links,demands,objective,best_bound,gap,cpu_time_seconds,bnb_nodes,status,method,time_limit_seconds,seed")
            foreach(row -> println(io, row), rows)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
