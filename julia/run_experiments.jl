using JSON3
using JuMP
using HiGHS

include(joinpath(@__DIR__, "src", "data.jl"))
include(joinpath(@__DIR__, "src", "model.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA = normpath(get(ENV, "DATA_DIR", joinpath(ROOT, "data", "setA")))
const RESULTS = joinpath(ROOT, "results")
const SOLUTIONS = joinpath(ROOT, "solutions")

function selftest()
    @assert route_distance([1, 2, 3], [1, 3]) == 3
    @assert route_distance([1, 2, 3, 4], [1, 2, 4]) == 3
    @assert route_distance([1, 2, 3, 4], [1, 3, 2, 4]) == 6
    @assert route_distance([1, 2, 4], [1, 3, 4]) == 4

    mktempdir() do directory
        net = Dict(
            "directed" => true,
            "multigraph" => false,
            "nodes" => [Dict("id" => i, "name" => "v$i") for i in 0:3],
            "links" => [
                Dict("id" => 0, "from" => 0, "to" => 1, "metric" => 1, "capacity" => 10),
                Dict("id" => 1, "from" => 0, "to" => 2, "metric" => 1, "capacity" => 10),
                Dict("id" => 2, "from" => 1, "to" => 3, "metric" => 1, "capacity" => 10),
                Dict("id" => 3, "from" => 2, "to" => 3, "metric" => 1, "capacity" => 10),
            ],
        )
        tm = Dict("num_time_slots" => 1, "demands" => [Dict("s" => 0, "t" => 3, "v" => [10.0])])
        scenario = Dict("max_segments" => 4, "budget" => [], "interventions" => [])
        for (suffix, content) in ("net" => net, "tm" => tm, "scenario" => scenario)
            open(joinpath(directory, "setA-01-$suffix.json"), "w") do io
                JSON3.write(io, content)
            end
        end
        instance = load_instance(directory, 1)
        graph = build_period(instance, 0)
        ratios = path_ratios(graph, [1, 4])
        @assert all(isapprox.(ratios, [0.5, 0.5, 0.5, 0.5]; atol=1e-8))
    end
    println("Self tests passed.")
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

function run_instance(number::Int, periods::Int, time_limit::Float64, runs::Int)
    instance = load_instance(DATA, number)
    println("$(instance.name): building $periods period graph(s)")
    flush(stdout)
    graphs = PeriodGraph[build_period(instance, t) for t in 0:(periods - 1)]
    rows = String[]
    best = nothing
    for run in 1:runs
        result = solve_instance(instance, graphs, time_limit)
        isfinite(result.objective) && validate_solution(instance, graphs, result.routes)
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
        ], ","))
        println(
            "$(instance.name) run $run/$runs: $(result.status), ",
            "MLU=$(result.objective), gap=$(result.gap), nodes=$(result.nodes), ",
            "optimize=$(round(result.optimize_time; digits=2))s, method=$(result.method)",
        )
        flush(stdout)
    end
    if best !== nothing && isfinite(best.objective)
        label = periods == 1 ? "t0" : "full"
        open(joinpath(SOLUTIONS, "$(instance.name)-julia-$label-srpaths.json"), "w") do io
            JSON3.pretty(io, solution_json(instance, best.routes))
        end
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
    mkpath(RESULTS)
    mkpath(SOLUTIONS)
    label = periods == 1 ? "t0" : "full"
    for number in numbers
        rows = run_instance(number, periods, time_limit, runs)
        path = joinpath(RESULTS, "julia_$(label)_setA-$(lpad(string(number), 2, '0')).csv")
        open(path, "w") do io
            println(io, "instance,run,periods,vertices,links,demands,objective,best_bound,gap,cpu_time_seconds,bnb_nodes,status,method,time_limit_seconds")
            foreach(row -> println(io, row), rows)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
