if !isdefined(Main, :Instance)
    using JSON3
    using JuMP
    using HiGHS
    using Random
    using Test
    include(joinpath(@__DIR__, "..", "src", "data.jl"))
    include(joinpath(@__DIR__, "..", "src", "model.jl"))
end

const TEST_DATA = normpath(get(ENV, "DATA_DIR", joinpath(@__DIR__, "..", "..", "data", "setA")))

function write_instance(directory, nodes, links, demands; slots=1, maxseg=6, budget=Dict{String,Int}[], interventions=Dict{String,Any}[])
    net = Dict(
        "directed" => true,
        "multigraph" => false,
        "nodes" => [Dict("id" => i, "name" => "v$i") for i in nodes],
        "links" => [Dict("id" => k - 1, "from" => u, "to" => v, "metric" => w, "capacity" => c)
                    for (k, (u, v, w, c)) in enumerate(links)],
    )
    tm = Dict("num_time_slots" => slots, "demands" => [Dict("s" => s, "t" => t, "v" => v) for (s, t, v) in demands])
    scenario = Dict("max_segments" => maxseg, "budget" => budget, "interventions" => interventions)
    for (suffix, content) in ("net" => net, "tm" => tm, "scenario" => scenario)
        open(joinpath(directory, "setA-01-$suffix.json"), "w") do io
            JSON3.write(io, content)
        end
    end
    return load_instance(directory, 1)
end

arc_index(inst, u, v) = findfirst(a -> inst.node_ids[a.u] == u && inst.node_ids[a.v] == v, inst.arcs)

@testset "T-ASR implementation" begin
    @testset "split coefficients r, equation (1)" begin
        mktempdir() do directory
            inst = write_instance(directory, 0:3, [(0, 1, 1, 10), (0, 2, 1, 10), (1, 3, 1, 10), (2, 3, 1, 10)], [(0, 3, [10.0])])
            g = build_period(inst, 0)
            @test all(isapprox.(path_ratios(g, [1, 4]), 0.5; atol=1e-12))
            @test maximum_load(inst, g, [[1, 4]], 0) ≈ 0.5
        end
        mktempdir() do directory
            links = [
                (0, 1, 1, 10), (0, 2, 1, 10), (0, 3, 1, 10),
                (1, 4, 1, 10), (2, 4, 1, 10), (3, 4, 1, 10),
                (4, 5, 1, 10), (5, 9, 1, 10),
                (0, 6, 1, 10), (6, 7, 1, 10), (7, 8, 1, 10), (8, 9, 1, 10),
            ]
            inst = write_instance(directory, 0:9, links, [(0, 9, [1.0])])
            g = build_period(inst, 0)
            i, j = 1, 10
            @test exp(g.logsigma[i, j]) ≈ 4
            @test path_ratios(g, [i, j])[arc_index(inst, 4, 5)] ≈ 3 / 4
            @test path_ratios(g, [i, j])[arc_index(inst, 0, 6)] ≈ 1 / 4
        end
    end

    @testset "period graph after an intervention" begin
        mktempdir() do directory
            interventions = [Dict("t" => 1, "links" => [0])]
            inst = write_instance(
                directory, 0:3, [(0, 1, 1, 10), (0, 2, 1, 10), (1, 3, 1, 10), (2, 3, 1, 10)],
                [(0, 3, [10.0, 10.0])]; slots=2, budget=[Dict("t" => 1, "value" => 0)], interventions,
            )
            g0, g1 = build_period(inst, 0), build_period(inst, 1)
            @test all(g0.available)
            @test !g1.available[arc_index(inst, 0, 1)]
            @test path_ratios(g1, [1, 4])[arc_index(inst, 0, 2)] ≈ 1.0
            @test !reachable(g1, 1, 2)
        end
    end

    @testset "segment count and reconfiguration cost, constraints (6) and (8)-(9)" begin
        @test length([1, 5, 7, 4]) - 1 == 3
        @test route_distance([1, 2, 3], [1, 3]) == 3
        @test route_distance([1, 2, 3, 4], [1, 2, 4]) == 3
        @test route_distance([1, 2, 3, 4], [1, 3, 2, 4]) == 6
        @test route_distance([1, 2, 4], [1, 3, 4]) == 4
        @test reconfiguration_cost([[1, 2, 3], [1, 3]], [[1, 3], [1, 3]]) == 3
    end

    @testset "cycle removal before inserting a column" begin
        @test _without_cycles([1, 2, 3, 2, 4]) == [1, 2, 4]
        @test _without_cycles([1, 2, 3, 4, 2, 5, 3, 6]) == [1, 2, 5, 3, 6]
        @test _without_cycles([1, 4]) == [1, 4]
    end

    @testset "gap and MLU certificate" begin
        @test _gap(0.5, 1.0) ≈ 0.5
        @test _gap(1.2, 1.0) == 0.0
        @test certified(1.0 - 1e-7, 1.0)
        @test !certified(1.0 - 1e-5, 1.0)
        @test _status(0.9, 1.0, true) == "Not Optimal"
        @test _status(0.9, 1.0, false) == "Infeasible"
    end

    @testset "independent solution check" begin
        mktempdir() do directory
            inst = write_instance(
                directory, 0:3, [(0, 1, 1, 10), (0, 2, 1, 10), (1, 3, 1, 10), (2, 3, 1, 10), (1, 2, 1, 10), (2, 1, 1, 10)],
                [(0, 3, [10.0, 10.0])]; slots=2, maxseg=2, budget=[Dict("t" => 1, "value" => 3)],
            )
            graphs = [build_period(inst, 0), build_period(inst, 1)]
            @test validate_solution(inst, graphs, [[[1, 4]], [[1, 4]]]) ≈ 0.5
            @test validate_solution(inst, graphs, [[[1, 2, 4]], [[1, 4]]]) ≈ 1.0
            @test_throws ErrorException validate_solution(inst, graphs, [[[2, 4]], [[1, 4]]])
            @test_throws ErrorException validate_solution(inst, graphs, [[[1, 2, 3, 4]], [[1, 4]]])
            @test_throws ErrorException validate_solution(inst, graphs, [[[1, 2, 1, 4]], [[1, 4]]])
            @test_throws ErrorException validate_solution(inst, graphs, [[[1, 2, 4]], [[1, 3, 4]]])
        end
    end

    if isfile(joinpath(TEST_DATA, "setA-01-net.json"))
        @testset "compact MIP and column generation agree on setA-01, period 0" begin
            inst = load_instance(TEST_DATA, 1)
            graphs = [build_period(inst, 0)]
            compact = solve_compact(inst, graphs, 120.0, 2026)
            @test compact.status == "Optimal"
            @test isapprox(compact.objective, 0.9294; atol=5e-5)
            columns = solve_by_columns(inst, graphs, 60.0, 2026)
            @test columns.bound <= compact.objective + 1e-7
            @test columns.objective >= compact.objective - 1e-7
            @test validate_solution(inst, graphs, columns.routes) ≈ columns.objective
        end

        @testset "joint pricing gives a valid lower bound on setA-02, two periods" begin
            inst = load_instance(TEST_DATA, 2)
            graphs = [build_period(inst, 0), build_period(inst, 1)]
            compact = solve_compact(inst, graphs, 600.0, 2026)
            @test compact.status == "Optimal"
            @test isapprox(compact.objective, 0.9031; atol=5e-5)
            first = solve_period_columns(inst, graphs[1], 0, 20.0, 2026; improve=false)
            second = solve_period_columns(inst, graphs[2], 1, 20.0, 2027; improve=false)
            @test max(first.lower, second.lower) <= compact.objective + 1e-7
            master, start = build_joint_master(inst, graphs, first, second, 60.0, 2028)
            @test reconfiguration_cost(start[1], start[2]) <= inst.budget[1]
            joint_lower, _ = _joint_pricing!(master, inst, graphs, time() + 60.0)
            @test 0.0 <= joint_lower <= compact.objective + 1e-7
            joint = solve_joint_columns(inst, graphs, 60.0, 2026)
            @test joint.bound <= compact.objective + 1e-7
            @test joint.objective >= compact.objective - 1e-7
            @test validate_solution(inst, graphs, joint.routes) ≈ joint.objective
        end
    end
end
