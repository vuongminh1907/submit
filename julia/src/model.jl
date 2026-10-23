struct SolveResult
    objective::Float64
    bound::Float64
    gap::Float64
    optimize_time::Float64
    nodes::Int
    status::String
    routes::Vector{Vector{Vector{Int}}}
    method::String
end

struct PeriodSolve
    result::SolveResult
    columns::Vector{Vector{Vector{Int}}}
    routes::Vector{Vector{Int}}
    lower::Float64
end

const COMPACT_VARIABLE_LIMIT = 100_000
const CERTIFICATION_TOLERANCE = 1e-6
const REDUCED_COST_TOLERANCE = 1e-7
const PERIOD_PRICING_ROUNDS = 120
const PERIOD_TIME_SHARE = 0.4
const JOINT_WARMUP_SHARE = 0.1
const JOINT_PRICING_SHARE = 0.5
const JOINT_PRICING_ROUNDS = 40
const JOINT_PRICING_COLUMNS = 25_000
const JOINT_PRICING_STALL = 6
const ENRICHED_POOL_SIZE = 6

estimated_binaries(inst::Instance, periods::Int) = ndemands(inst) * periods * inst.n * max(inst.n - 1, 1)

function _gap(bound::Float64, objective::Float64)
    objective <= 1e-12 && return 0.0
    return max(0.0, 1.0 - bound / objective)
end

certified(bound::Float64, objective::Float64) = _gap(bound, objective) <= CERTIFICATION_TOLERANCE

function _status(bound::Float64, objective::Float64, feasible::Bool)
    feasible || return "Infeasible"
    return certified(bound, objective) ? "Optimal" : "Not Optimal"
end

function _node_count(model)
    try
        return Int(MOI.get(model, MOI.NodeCount()))
    catch
        return 0
    end
end

function _configure!(model, time_limit::Float64, random_seed::Int)
    set_silent(model)
    set_time_limit_sec(model, max(time_limit, 0.01))
    set_optimizer_attribute(model, "threads", 1)
    set_optimizer_attribute(model, "random_seed", random_seed)
    set_optimizer_attribute(model, "mip_rel_gap", CERTIFICATION_TOLERANCE)
    set_optimizer_attribute(model, "mip_abs_gap", 0.0)
    return model
end

function _exhaustive_mip!(model)
    set_optimizer_attribute(model, "mip_rel_gap", 0.0)
    set_optimizer_attribute(model, "mip_abs_gap", 0.0)
    set_optimizer_attribute(model, "presolve", "off")
    set_optimizer_attribute(model, "mip_heuristic_effort", 1.0)
    return model
end

function _timed_optimize!(model)
    started = time()
    optimize!(model)
    return time() - started
end

function solve_instance(
    inst::Instance, graphs::Vector{PeriodGraph}, time_limit::Float64;
    random_seed::Int=1,
)
    if estimated_binaries(inst, length(graphs)) <= COMPACT_VARIABLE_LIMIT
        return solve_compact(inst, graphs, time_limit, random_seed)
    end
    return solve_by_columns(inst, graphs, time_limit, random_seed)
end

function solve_compact(
    inst::Instance, graphs::Vector{PeriodGraph}, time_limit::Float64, random_seed::Int=1,
)
    model = _configure!(Model(HiGHS.Optimizer), time_limit, random_seed)
    periods = length(graphs)
    demands = ndemands(inst)
    n = inst.n
    pairs = Vector{Tuple{Int,Int}}[]
    position = Dict{Tuple{Int,Int},Int}[]
    for g in graphs
        current = Tuple{Int,Int}[]
        index = Dict{Tuple{Int,Int},Int}()
        for i in 1:n, j in 1:n
            if reachable(g, i, j)
                push!(current, (i, j))
                index[(i, j)] = length(current)
            end
        end
        push!(pairs, current)
        push!(position, index)
    end

    @variable(model, L >= 0)
    @variable(model, x[t=1:periods, d=1:demands, p=1:length(pairs[t])], Bin)
    @objective(model, Min, L)

    for t in 1:periods
        outgoing = [Int[] for _ in 1:n]
        incoming = [Int[] for _ in 1:n]
        for (p, (i, j)) in enumerate(pairs[t])
            push!(outgoing[i], p)
            push!(incoming[j], p)
        end
        for d in 1:demands
            source = inst.source[d]
            target = inst.target[d]
            for node in 1:n
                out_expr = AffExpr(0.0)
                for p in outgoing[node]
                    add_to_expression!(out_expr, x[t, d, p], 1.0)
                end
                in_expr = AffExpr(0.0)
                for p in incoming[node]
                    add_to_expression!(in_expr, x[t, d, p], 1.0)
                end
                rhs = node == source ? 1.0 : node == target ? -1.0 : 0.0
                @constraint(model, out_expr - in_expr == rhs)
                @constraint(model, out_expr <= (node == target ? 0.0 : 1.0))
                @constraint(model, in_expr <= (node == source ? 0.0 : 1.0))
            end
            @constraint(model, sum(x[t, d, p] for p in 1:length(pairs[t])) <= inst.maxseg)
        end

        loads = [AffExpr(0.0) for _ in inst.arcs]
        for (a, arc) in enumerate(inst.arcs)
            graphs[t].available[a] && add_to_expression!(loads[a], L, -arc.capacity)
        end
        for d in 1:demands
            volume = inst.volume[d, t]
            volume == 0 && continue
            for (p, (i, j)) in enumerate(pairs[t])
                seg_a = graphs[t].seg_arcs[i, j]
                seg_r = graphs[t].seg_ratios[i, j]
                for k in eachindex(seg_a)
                    add_to_expression!(loads[seg_a[k]], x[t, d, p], volume * seg_r[k])
                end
            end
        end
        for a in eachindex(inst.arcs)
            graphs[t].available[a] && @constraint(model, loads[a] <= 0)
        end
    end

    if periods == 2
        universe = union(keys(position[1]), keys(position[2]))
        @variable(model, y[d=1:demands, s in universe] >= 0)
        for d in 1:demands, segment in universe
            first = haskey(position[1], segment) ? x[1, d, position[1][segment]] : 0.0
            second = haskey(position[2], segment) ? x[2, d, position[2][segment]] : 0.0
            @constraint(model, y[d, segment] >= second - first)
            @constraint(model, y[d, segment] >= first - second)
        end
        @constraint(model, sum(y) <= inst.budget[1])
    end

    elapsed = _timed_optimize!(model)
    return _result_from_compact(inst, graphs, model, pairs, x, elapsed)
end

function _result_from_compact(inst, graphs, model, pairs, x, elapsed)
    nodes = _node_count(model)
    if termination_status(model) == MOI.INFEASIBLE || !has_values(model)
        return SolveResult(NaN, NaN, NaN, elapsed, nodes, "Infeasible", Vector{Vector{Int}}[], "compact-mip")
    end
    routes = Vector{Vector{Vector{Int}}}()
    for t in eachindex(graphs)
        period_routes = Vector{Vector{Int}}()
        for d in 1:ndemands(inst)
            successor = Dict{Int,Int}()
            for (p, (i, j)) in enumerate(pairs[t])
                value(x[t, d, p]) > 0.5 && (successor[i] = j)
            end
            route = Int[inst.source[d]]
            while route[end] != inst.target[d]
                haskey(successor, route[end]) || error("Compact incumbent does not contain an s-t path.")
                next_node = successor[route[end]]
                next_node in route && error("Compact incumbent contains a cycle.")
                push!(route, next_node)
            end
            push!(period_routes, route)
        end
        push!(routes, period_routes)
    end
    objective = validate_solution(inst, graphs, routes)
    bound = try
        min(Float64(objective_bound(model)), objective)
    catch
        0.0
    end
    status = _status(bound, objective, true)
    status == "Optimal" && (bound = objective)
    return SolveResult(objective, bound, _gap(bound, objective), elapsed, nodes, status, routes, "compact-mip")
end

function _best_path(unit::Matrix{Float64}, extra::Matrix{Float64}, source::Int, target::Int, maxseg::Int, volume::Float64)
    n = size(unit, 1)
    previous = fill(Inf, n)
    previous[source] = 0.0
    parents = Vector{Int}[]
    best = Inf
    best_k = 0
    for _ in 1:maxseg
        cost = fill(Inf, n)
        parent = zeros(Int, n)
        @inbounds for u in 1:n
            base = previous[u]
            isfinite(base) || continue
            for v in 1:n
                weight = unit[u, v]
                isfinite(weight) || continue
                candidate = base + volume * weight + extra[u, v]
                if candidate < cost[v]
                    cost[v] = candidate
                    parent[v] = u
                end
            end
        end
        push!(parents, parent)
        if cost[target] < best
            best = cost[target]
            best_k = length(parents)
        end
        previous = cost
    end
    isfinite(best) || return Int[], Inf
    nodes = Int[target]
    node = target
    for k in best_k:-1:1
        node = parents[k][node]
        node == 0 && return Int[], Inf
        push!(nodes, node)
    end
    reverse!(nodes)
    nodes[1] == source || return Int[], Inf
    return _without_cycles(nodes), best
end

function _without_cycles(nodes::Vector{Int})
    seen = Dict{Int,Int}()
    cleaned = Int[]
    for node in nodes
        if haskey(seen, node)
            for removed in cleaned[(seen[node] + 1):end]
                delete!(seen, removed)
            end
            resize!(cleaned, seen[node])
        else
            push!(cleaned, node)
            seen[node] = length(cleaned)
        end
    end
    return cleaned
end

function _dense_path(unit::Matrix{Float64}, source::Int, target::Int)
    n = size(unit, 1)
    dist = fill(Inf, n)
    parent = zeros(Int, n)
    dist[source] = 0.0
    used = falses(n)
    @inbounds for _ in 1:n
        u = 0
        best = Inf
        for v in 1:n
            if !used[v] && dist[v] < best
                best = dist[v]
                u = v
            end
        end
        (u == 0 || u == target) && break
        used[u] = true
        base = dist[u]
        for v in 1:n
            weight = unit[u, v]
            isfinite(weight) || continue
            candidate = base + weight
            if candidate < dist[v]
                dist[v] = candidate
                parent[v] = u
            end
        end
    end
    isfinite(dist[target]) || return Int[], Inf
    nodes = Int[target]
    node = target
    for _ in 1:n
        node == source && break
        node = parent[node]
        node == 0 && return Int[], Inf
        push!(nodes, node)
    end
    nodes[end] == source || return Int[], Inf
    reverse!(nodes)
    return _without_cycles(nodes), dist[target]
end

function _unit_costs(g::PeriodGraph, load_price::Vector{Float64})
    n = size(g.dist, 1)
    unit = fill(Inf, n, n)
    @inbounds for i in 1:n, j in 1:n
        i == j && continue
        arcs = g.seg_arcs[i, j]
        isempty(arcs) && continue
        ratios = g.seg_ratios[i, j]
        value = 0.0
        for k in eachindex(arcs)
            value += load_price[arcs[k]] * ratios[k]
        end
        unit[i, j] = value
    end
    return unit
end

_nonnegative(unit::Matrix{Float64}) = all(w -> !isfinite(w) || w >= -1e-9, unit)

function _price(
    unit::Matrix{Float64}, nonnegative::Bool, extra::Matrix{Float64}, has_extra::Bool,
    source::Int, target::Int, maxseg::Int, volume::Float64, convexity_dual::Float64,
)
    if nonnegative && !has_extra
        path, unit_cost = _dense_path(unit, source, target)
        isempty(path) && return Int[], 0.0
        reduced = volume * unit_cost - convexity_dual
        reduced >= -REDUCED_COST_TOLERANCE && return Int[], reduced
        length(path) - 1 <= maxseg && return path, reduced
    end
    path, cost = _best_path(unit, extra, source, target, maxseg, volume)
    isempty(path) && return Int[], 0.0
    return path, cost - convexity_dual
end

function _path_load_cost(volume::Float64, ratios::Vector{Float64}, load_price::Vector{Float64})
    value = 0.0
    @inbounds for a in eachindex(ratios)
        value += load_price[a] * ratios[a]
    end
    return volume * value
end

function solve_period_columns(
    inst::Instance, g::PeriodGraph, time_index::Int, time_limit::Float64,
    random_seed::Int=1; improve::Bool=true,
)
    started = time()
    deadline = started + time_limit
    demands = ndemands(inst)
    m = length(inst.arcs)
    column = time_index + 1
    model = _configure!(Model(HiGHS.Optimizer), time_limit, random_seed)
    @variable(model, L >= 0)
    @objective(model, Min, L)
    convexity = [@constraint(model, 0.0 == 1.0) for _ in 1:demands]
    load_rows = Vector{Any}(undef, m)
    for a in 1:m
        load_rows[a] = g.available[a] ?
            @constraint(model, -inst.arcs[a].capacity * L <= 0) : nothing
    end
    columns = [Vector{Int}[] for _ in 1:demands]
    variables = [VariableRef[] for _ in 1:demands]
    seen = [Set{Vector{Int}}() for _ in 1:demands]
    binary = Ref(false)

    function add_column!(d::Int, nodes::Vector{Int})
        nodes in seen[d] && return false
        push!(seen[d], copy(nodes))
        variable = @variable(model, lower_bound = 0.0)
        binary[] && set_binary(variable)
        push!(variables[d], variable)
        push!(columns[d], copy(nodes))
        set_normalized_coefficient(convexity[d], variable, 1.0)
        ratios = path_ratios(g, nodes)
        volume = inst.volume[d, column]
        for a in 1:m
            g.available[a] || continue
            coefficient = volume * ratios[a]
            coefficient == 0 && continue
            set_normalized_coefficient(load_rows[a], variable, coefficient)
        end
        return true
    end

    for d in 1:demands
        add_column!(d, [inst.source[d], inst.target[d]])
    end

    optimize_time = 0.0
    lp_certified = false
    lower = 0.0
    checked_costs = false
    pricing_rounds = 0
    no_extra = zeros(inst.n, inst.n)
    for _ in 1:PERIOD_PRICING_ROUNDS
        pricing_rounds += 1
        remaining = deadline - time()
        remaining <= 1 && break
        set_time_limit_sec(model, remaining)
        optimize_time += _timed_optimize!(model)
        termination_status(model) == MOI.OPTIMAL || break
        lp_obj = Float64(objective_value(model))
        load_dual = zeros(m)
        for a in 1:m
            g.available[a] && (load_dual[a] = dual(load_rows[a]))
        end
        convexity_dual = [dual(convexity[d]) for d in 1:demands]
        if !checked_costs
            for d in 1:demands, (p, variable) in enumerate(variables[d])
                ratios = path_ratios(g, columns[d][p])
                predicted = -_path_load_cost(inst.volume[d, column], ratios, load_dual) - convexity_dual[d]
                abs(predicted - reduced_cost(variable)) < 1e-4 ||
                    error("Reduced-cost formula does not match HiGHS.")
            end
            checked_costs = true
        end
        unit = _unit_costs(g, -load_dual)
        nonnegative = _nonnegative(unit)
        added = 0
        correction = 0.0
        for d in 1:demands
            path, reduced = _price(
                unit, nonnegative, no_extra, false, inst.source[d], inst.target[d],
                inst.maxseg, inst.volume[d, column], convexity_dual[d],
            )
            correction += min(reduced, 0.0)
            if reduced < -REDUCED_COST_TOLERANCE && !isempty(path)
                add_column!(d, path) ||
                    error("Reduced-cost formula disagrees with a column already in the pool.")
                added += 1
            end
        end
        lower = max(lower, lp_obj + correction)
        if added == 0
            lp_certified = true
            lower = max(lower, lp_obj)
            break
        end
    end
    println("t=$time_index pricing rounds=$pricing_rounds lp_certified=$lp_certified bound=$lower columns=$(sum(length, columns))")
    flush(stdout)

    for d in 1:demands, variable in variables[d]
        set_binary(variable)
    end
    binary[] = true
    direct_load = maximum_load(inst, g, [columns[d][1] for d in 1:demands], time_index)
    set_start_value(L, direct_load)
    for d in 1:demands, (p, variable) in enumerate(variables[d])
        set_start_value(variable, p == 1 ? 1.0 : 0.0)
    end
    set_time_limit_sec(model, max(deadline - time(), 0.01))
    optimize_time += _timed_optimize!(model)
    nodes = _node_count(model)
    routes = has_values(model) ? _selected_routes(columns, variables) : direct_routes(inst, g)
    objective = try
        maximum_load(inst, g, routes, time_index)
    catch
        NaN
    end

    if improve && isfinite(objective)
        objective, routes, extra_time, extra_nodes = _improve_period!(
            inst, g, time_index, model, L, add_column!, columns, variables,
            routes, objective, lower, deadline, random_seed,
        )
        optimize_time += extra_time
        nodes += extra_nodes
    end

    if !isfinite(objective)
        result = SolveResult(NaN, lower, NaN, optimize_time, nodes, "Infeasible", [routes], "column-generation")
        return PeriodSolve(result, columns, routes, lower)
    end
    bound = min(max(lower, 0.0), objective)
    status = _status(bound, objective, true)
    status == "Optimal" && (bound = objective)
    result = SolveResult(
        objective, bound, _gap(bound, objective), optimize_time, nodes, status, [routes], "column-generation",
    )
    return PeriodSolve(result, columns, routes, max(lower, 0.0))
end

function _selected_routes(columns, variables)
    return [columns[d][argmax([value(v) for v in variables[d]])] for d in eachindex(columns)]
end

function _generation_budget(remaining::Float64)
    return min(max(remaining * 0.25, 5.0), remaining - 1.0, 60.0)
end

function _generate_columns!(inst::Instance, generators, deadline::Float64, seed::Int)
    added = 0
    while time() < deadline
        batch = 0
        for (g, add!, waypoint_state) in generators
            count = _diversify_one_waypoint!(inst, g, add!, waypoint_state, seed; limit=500)
            seed += 1
            if count == 0
                count = _diversify_two_waypoints!(inst, g, add!, seed; limit=500)
                seed += 1
            end
            if count == 0
                count = _diversify_random_paths!(inst, g, add!, seed; limit=200)
                seed += 1
            end
            batch += count
        end
        added += batch
        batch == 0 && break
    end
    return added
end

function _improve_period!(
    inst, g, time_index, model, L, add_column!, columns, variables,
    routes, objective, lower, deadline, random_seed,
)
    demands = ndemands(inst)
    waypoint_state = ones(Int, demands)
    optimize_time = 0.0
    nodes = 0
    iteration = 0
    total_added = 0
    _exhaustive_mip!(model)
    while time() < deadline - 1.0
        certified(min(max(lower, 0.0), objective), objective) && break
        iteration += 1
        remaining = deadline - time()
        added = _generate_columns!(
            inst, [(g, add_column!, waypoint_state)],
            time() + _generation_budget(remaining), random_seed + iteration,
        )
        total_added += added
        remaining = deadline - time()
        remaining <= 1.0 && break
        println("t=$time_index improve round=$iteration added=$added columns=$(sum(length, columns)) remaining=$(round_seconds(remaining))s obj=$objective")
        flush(stdout)

        set_optimizer_attribute(model, "random_seed", random_seed + iteration)
        set_time_limit_sec(model, remaining)
        set_start_value(L, objective)
        for d in 1:demands, (p, variable) in enumerate(variables[d])
            set_start_value(variable, columns[d][p] == routes[d] ? 1.0 : 0.0)
        end
        elapsed = _timed_optimize!(model)
        optimize_time += elapsed
        nodes += _node_count(model)
        if has_values(model)
            candidate = _selected_routes(columns, variables)
            candidate_obj = try
                maximum_load(inst, g, candidate, time_index)
            catch
                NaN
            end
            if isfinite(candidate_obj) && candidate_obj < objective - 1e-12
                routes = candidate
                objective = candidate_obj
            end
        end
        elapsed >= remaining - 2.0 && break
        added == 0 && elapsed < 1.0 && break
    end
    println("t=$time_index improve done rounds=$iteration added=$total_added obj=$objective")
    flush(stdout)
    return objective, routes, optimize_time, nodes
end

round_seconds(value::Float64) = round(value; digits=1)

function _diversify_one_waypoint!(
    inst::Instance, g::PeriodGraph, add!, waypoint_state::Vector{Int}, seed::Int; limit::Int=200,
)
    added = 0
    rng = Random.MersenneTwister(seed)
    for d in randperm(rng, ndemands(inst))
        added >= limit && break
        source = inst.source[d]
        target = inst.target[d]
        candidates = [v for v in 1:inst.n if v != source && v != target]
        isempty(candidates) && continue
        start = waypoint_state[d]
        for offset in 0:(length(candidates) - 1)
            added >= limit && break
            idx = mod1(start + offset, length(candidates))
            waypoint = candidates[idx]
            waypoint_state[d] = mod1(idx + 1, length(candidates))
            reachable(g, source, waypoint) && reachable(g, waypoint, target) || continue
            path = [source, waypoint, target]
            length(path) - 1 <= inst.maxseg || continue
            add!(d, path) && (added += 1)
        end
    end
    return added
end

function _diversify_two_waypoints!(inst::Instance, g::PeriodGraph, add!, seed::Int; limit::Int=200)
    inst.maxseg < 3 && return 0
    demands = ndemands(inst)
    added = 0
    rng = Random.MersenneTwister(seed)
    attempts = 0
    while added < limit && attempts < limit * 20
        attempts += 1
        d = rand(rng, 1:demands)
        source = inst.source[d]
        target = inst.target[d]
        candidates = [v for v in 1:inst.n if v != source && v != target]
        length(candidates) < 2 && continue
        w1 = candidates[rand(rng, 1:length(candidates))]
        w2 = candidates[rand(rng, 1:length(candidates))]
        w1 == w2 && continue
        reachable(g, source, w1) && reachable(g, w1, w2) && reachable(g, w2, target) || continue
        add!(d, [source, w1, w2, target]) && (added += 1)
    end
    return added
end

function _diversify_random_paths!(inst::Instance, g::PeriodGraph, add!, seed::Int; limit::Int=200)
    demands = ndemands(inst)
    added = 0
    rng = Random.MersenneTwister(seed)
    n = inst.n
    extra = zeros(n, n)
    for _ in 1:limit
        d = rand(rng, 1:demands)
        unit = fill(Inf, n, n)
        @inbounds for i in 1:n, j in 1:n
            i == j && continue
            isempty(g.seg_arcs[i, j]) && continue
            unit[i, j] = rand(rng)
        end
        path, _ = _best_path(unit, extra, inst.source[d], inst.target[d], inst.maxseg, 1.0)
        isempty(path) && continue
        add!(d, path) && (added += 1)
    end
    return added
end

function solve_by_columns(
    inst::Instance, graphs::Vector{PeriodGraph}, time_limit::Float64, random_seed::Int=1,
)
    if length(graphs) == 1
        period = solve_period_columns(inst, graphs[1], 0, time_limit, random_seed; improve=true)
        result = period.result
        objective = validate_solution(inst, graphs, result.routes)
        bound = min(result.bound, objective)
        return SolveResult(
            objective, bound, _gap(bound, objective), result.optimize_time, result.nodes,
            _status(bound, objective, true), result.routes, result.method,
        )
    end
    return solve_joint_columns(inst, graphs, time_limit, random_seed)
end

mutable struct JointMaster
    model::Model
    L::VariableRef
    convexity::Matrix{ConstraintRef}
    load::Vector{Vector{Any}}
    budget::ConstraintRef
    columns::Matrix{Vector{Vector{Int}}}
    variables::Matrix{Vector{VariableRef}}
    index::Matrix{Dict{Vector{Int},Int}}
    changes::Vector{Dict{Tuple{Int,Int},Tuple{VariableRef,ConstraintRef,ConstraintRef}}}
    binary::Bool
end

function _joint_master(inst::Instance, graphs::Vector{PeriodGraph}, time_limit::Float64, random_seed::Int)
    model = _configure!(Model(HiGHS.Optimizer), time_limit, random_seed)
    demands = ndemands(inst)
    @variable(model, L >= 0)
    @objective(model, Min, L)
    convexity = Matrix{ConstraintRef}(undef, 2, demands)
    for t in 1:2, d in 1:demands
        convexity[t, d] = @constraint(model, 0.0 == 1.0)
    end
    load = [Any[g.available[a] ? @constraint(model, -arc.capacity * L <= 0) : nothing
                for (a, arc) in enumerate(inst.arcs)] for g in graphs]
    budget = @constraint(model, 0.0 <= Float64(inst.budget[1]))
    return JointMaster(
        model, L, convexity, load, budget,
        [Vector{Int}[] for _ in 1:2, _ in 1:demands],
        [VariableRef[] for _ in 1:2, _ in 1:demands],
        [Dict{Vector{Int},Int}() for _ in 1:2, _ in 1:demands],
        [Dict{Tuple{Int,Int},Tuple{VariableRef,ConstraintRef,ConstraintRef}}() for _ in 1:demands],
        true,
    )
end

function add_joint_column!(
    master::JointMaster, inst::Instance, graphs::Vector{PeriodGraph}, t::Int, d::Int, nodes::Vector{Int},
)
    haskey(master.index[t, d], nodes) && return false
    route_ok(inst, graphs[t], nodes, d) || return false
    model = master.model
    z = @variable(model, lower_bound = 0.0)
    master.binary && set_binary(z)
    push!(master.variables[t, d], z)
    push!(master.columns[t, d], copy(nodes))
    master.index[t, d][copy(nodes)] = length(master.columns[t, d])
    set_normalized_coefficient(master.convexity[t, d], z, 1.0)
    volume = inst.volume[d, t]
    ratios = path_ratios(graphs[t], nodes)
    for a in eachindex(inst.arcs)
        graphs[t].available[a] || continue
        coefficient = volume * ratios[a]
        coefficient == 0 && continue
        set_normalized_coefficient(master.load[t][a], z, coefficient)
    end
    direction = t == 1 ? 1.0 : -1.0
    for segment in segment_set(nodes)
        y, up, down = get!(master.changes[d], segment) do
            y = @variable(model, lower_bound = 0.0)
            up = @constraint(model, 1.0 * y >= 0.0)
            down = @constraint(model, 1.0 * y >= 0.0)
            set_normalized_coefficient(master.budget, y, 1.0)
            (y, up, down)
        end
        set_normalized_coefficient(up, z, direction)
        set_normalized_coefficient(down, z, -direction)
    end
    return true
end

function _add_to_valid_periods!(master, inst, graphs, t::Int, d::Int, nodes::Vector{Int})
    added = add_joint_column!(master, inst, graphs, t, d, nodes)
    other = 3 - t
    route_ok(inst, graphs[other], nodes, d) &&
        (added = add_joint_column!(master, inst, graphs, other, d, nodes) || added)
    return added
end

function _set_joint_start!(master::JointMaster, inst::Instance, graphs, routes, objective::Float64)
    set_start_value(master.L, objective)
    for t in 1:2, d in 1:ndemands(inst)
        chosen = get(master.index[t, d], routes[t][d], 0)
        for (p, z) in enumerate(master.variables[t, d])
            set_start_value(z, p == chosen ? 1.0 : 0.0)
        end
    end
    for d in 1:ndemands(inst)
        first = segment_set(routes[1][d])
        second = segment_set(routes[2][d])
        for (segment, (y, _, _)) in master.changes[d]
            set_start_value(y, abs(Float64(segment in second) - Float64(segment in first)))
        end
    end
end

function _solve_joint_mip!(master::JointMaster, inst, graphs, start_routes, start_objective, time_limit, seed)
    set_optimizer_attribute(master.model, "random_seed", seed)
    set_time_limit_sec(master.model, max(time_limit, 0.01))
    _set_joint_start!(master, inst, graphs, start_routes, start_objective)
    elapsed = _timed_optimize!(master.model)
    nodes = _node_count(master.model)
    has_values(master.model) || return nothing, Inf, elapsed, nodes
    routes = [_selected_routes(master.columns[t, :], master.variables[t, :]) for t in 1:2]
    objective = try
        validate_solution(inst, graphs, routes)
    catch
        Inf
    end
    return routes, objective, elapsed, nodes
end

function _joint_reduced_cost(master::JointMaster, inst, graphs, t, d, p, load_dual, convexity_dual, change_dual)
    nodes = master.columns[t, d][p]
    value = -_path_load_cost(inst.volume[d, t], path_ratios(graphs[t], nodes), load_dual[t]) - convexity_dual[t, d]
    direction = t == 1 ? 1.0 : -1.0
    for segment in segment_set(nodes)
        up, down = change_dual[d][segment]
        value -= direction * (up - down)
    end
    return value
end

function _joint_pricing!(master::JointMaster, inst::Instance, graphs::Vector{PeriodGraph}, deadline::Float64)
    demands = ndemands(inst)
    model = master.model
    restore = relax_integrality(model)
    master.binary = false
    optimize_time = 0.0
    lower = 0.0
    best_lp = Inf
    stall = 0
    added_total = 0
    rounds = 0
    lp_certified = false
    extra = zeros(inst.n, inst.n)
    while rounds < JOINT_PRICING_ROUNDS && time() < deadline - 1.0
        rounds += 1
        set_time_limit_sec(model, deadline - time())
        optimize_time += _timed_optimize!(model)
        termination_status(model) == MOI.OPTIMAL || break
        lp_obj = Float64(objective_value(model))
        load_dual = [[g.available[a] ? dual(master.load[t][a]) : 0.0 for a in eachindex(inst.arcs)]
                     for (t, g) in enumerate(graphs)]
        convexity_dual = [dual(master.convexity[t, d]) for t in 1:2, d in 1:demands]
        change_dual = [Dict(segment => (dual(up), dual(down)) for (segment, (_, up, down)) in master.changes[d])
                       for d in 1:demands]
        if rounds == 1
            for t in 1:2, d in 1:demands, (p, z) in enumerate(master.variables[t, d])
                predicted = _joint_reduced_cost(master, inst, graphs, t, d, p, load_dual, convexity_dual, change_dual)
                abs(predicted - reduced_cost(z)) < 1e-4 ||
                    error("Joint reduced-cost formula does not match HiGHS.")
            end
        end
        candidates = Tuple{Int,Int,Vector{Int}}[]
        correction = 0.0
        for t in 1:2
            unit = _unit_costs(graphs[t], -load_dual[t])
            nonnegative = _nonnegative(unit)
            direction = t == 1 ? 1.0 : -1.0
            for d in 1:demands
                touched = Tuple{Int,Int}[]
                for (segment, (up, down)) in change_dual[d]
                    penalty = -direction * (up - down)
                    abs(penalty) <= 1e-12 && continue
                    extra[segment[1], segment[2]] = penalty
                    push!(touched, segment)
                end
                path, reduced = _price(
                    unit, nonnegative, extra, !isempty(touched), inst.source[d], inst.target[d],
                    inst.maxseg, inst.volume[d, t], convexity_dual[t, d],
                )
                for (i, j) in touched
                    extra[i, j] = 0.0
                end
                correction += min(reduced, 0.0)
                reduced < -REDUCED_COST_TOLERANCE && !isempty(path) && push!(candidates, (t, d, path))
            end
        end
        lower = max(lower, lp_obj + correction)
        if correction >= -REDUCED_COST_TOLERANCE
            lp_certified = true
            lower = max(lower, lp_obj)
            break
        end
        added = 0
        for (t, d, path) in candidates
            added_total >= JOINT_PRICING_COLUMNS && break
            if add_joint_column!(master, inst, graphs, t, d, path)
                added += 1
                added_total += 1
            end
        end
        added == 0 && break
        if lp_obj < best_lp - 1e-9
            best_lp = lp_obj
            stall = 0
        else
            stall += 1
            stall >= JOINT_PRICING_STALL && break
        end
        added_total >= JOINT_PRICING_COLUMNS && break
    end
    restore()
    for t in 1:2, d in 1:demands, z in master.variables[t, d]
        is_binary(z) || set_binary(z)
    end
    master.binary = true
    println("joint pricing rounds=$rounds lp_certified=$lp_certified bound=$lower added=$added_total")
    flush(stdout)
    return max(lower, 0.0), optimize_time
end

function _fit_budget(inst, graphs, first_routes, second_routes)
    demands = ndemands(inst)
    left = [copy(first_routes[d]) for d in 1:demands]
    right = [copy(second_routes[d]) for d in 1:demands]
    for d in 1:demands
        route_ok(inst, graphs[1], left[d], d) || (left[d] = [inst.source[d], inst.target[d]])
        route_ok(inst, graphs[2], right[d], d) || (right[d] = [inst.source[d], inst.target[d]])
    end
    order = sort(1:demands, by = d -> route_distance(left[d], right[d]), rev = true)
    for d in order
        route_distance(left[d], right[d]) == 0 && break
        reconfiguration_cost(left, right) <= inst.budget[1] && break
        if route_ok(inst, graphs[2], left[d], d)
            right[d] = copy(left[d])
        elseif route_ok(inst, graphs[1], right[d], d)
            left[d] = copy(right[d])
        else
            left[d] = [inst.source[d], inst.target[d]]
            right[d] = [inst.source[d], inst.target[d]]
        end
    end
    return [left, right]
end

function _enrich_pool(inst::Instance, g::PeriodGraph, generated::Vector{Vector{Vector{Int}}})
    pool = [copy(paths) for paths in generated]
    for d in 1:ndemands(inst)
        direct = [inst.source[d], inst.target[d]]
        direct in pool[d] || pushfirst!(pool[d], direct)
        length(pool[d]) >= ENRICHED_POOL_SIZE && continue
        for waypoint in _waypoint_sample(inst.n, inst.source[d], inst.target[d])
            reachable(g, inst.source[d], waypoint) && reachable(g, waypoint, inst.target[d]) || continue
            path = [inst.source[d], waypoint, inst.target[d]]
            path in pool[d] && continue
            push!(pool[d], path)
            length(pool[d]) >= ENRICHED_POOL_SIZE && break
        end
    end
    return pool
end

function _waypoint_sample(n::Int, source::Int, target::Int)
    candidates = [v for v in 1:n if v != source && v != target]
    length(candidates) <= 12 && return candidates
    stride = max(1, length(candidates) ÷ 12)
    return candidates[1:stride:end]
end

function build_joint_master(
    inst::Instance, graphs::Vector{PeriodGraph}, first::PeriodSolve, second::PeriodSolve,
    time_limit::Float64, random_seed::Int,
)
    master = _joint_master(inst, graphs, time_limit, random_seed)
    pools = [_enrich_pool(inst, graphs[1], first.columns), _enrich_pool(inst, graphs[2], second.columns)]
    for t in 1:2, d in 1:ndemands(inst), path in pools[t][d]
        _add_to_valid_periods!(master, inst, graphs, t, d, path)
    end
    start = _fit_budget(inst, graphs, first.routes, second.routes)
    for t in 1:2, d in 1:ndemands(inst)
        add_joint_column!(master, inst, graphs, t, d, start[t][d])
    end
    return master, start
end

function solve_joint_columns(
    inst::Instance, graphs::Vector{PeriodGraph}, time_limit::Float64, random_seed::Int=1,
)
    started = time()
    deadline = started + time_limit
    first = solve_period_columns(inst, graphs[1], 0, PERIOD_TIME_SHARE * time_limit, random_seed; improve=false)
    second = solve_period_columns(inst, graphs[2], 1, PERIOD_TIME_SHARE * time_limit, random_seed + 1; improve=false)
    optimize_time = first.result.optimize_time + second.result.optimize_time
    nodes = first.result.nodes + second.result.nodes
    lower = max(first.lower, second.lower, 0.0)

    master, best_routes = build_joint_master(inst, graphs, first, second, max(deadline - time(), 0.01), random_seed + 2)
    best = validate_solution(inst, graphs, best_routes)

    function accept!(routes, objective)
        if routes !== nothing && objective < best - 1e-12
            best_routes = routes
            best = objective
        end
    end

    routes, objective, elapsed, count = _solve_joint_mip!(
        master, inst, graphs, best_routes, best, JOINT_WARMUP_SHARE * (deadline - time()), random_seed + 2,
    )
    optimize_time += elapsed
    nodes += count
    accept!(routes, objective)

    if !certified(min(lower, best), best)
        pricing_deadline = time() + JOINT_PRICING_SHARE * max(deadline - time(), 0.0)
        joint_lower, elapsed = _joint_pricing!(master, inst, graphs, pricing_deadline)
        optimize_time += elapsed
        lower = max(lower, joint_lower)
    end

    if !certified(min(lower, best), best) && deadline - time() > 1.0
        routes, objective, elapsed, count = _solve_joint_mip!(
            master, inst, graphs, best_routes, best, deadline - time(), random_seed + 3,
        )
        optimize_time += elapsed
        nodes += count
        accept!(routes, objective)
    end

    _exhaustive_mip!(master.model)
    waypoint_state = [ones(Int, ndemands(inst)) for _ in 1:2]
    generators = [
        (graphs[t], (d, path) -> _add_to_valid_periods!(master, inst, graphs, t, d, path), waypoint_state[t])
        for t in 1:2
    ]
    iteration = 0
    while time() < deadline - 1.0 && !certified(min(lower, best), best)
        iteration += 1
        remaining = deadline - time()
        added = _generate_columns!(inst, generators, time() + _generation_budget(remaining), random_seed + 10 * iteration)
        remaining = deadline - time()
        remaining <= 1.0 && break
        println("joint improve round=$iteration added=$added remaining=$(round_seconds(remaining))s obj=$best")
        flush(stdout)
        routes, objective, elapsed, count = _solve_joint_mip!(
            master, inst, graphs, best_routes, best, remaining, random_seed + 3 + iteration,
        )
        optimize_time += elapsed
        nodes += count
        accept!(routes, objective)
        elapsed >= remaining - 2.0 && break
        added == 0 && elapsed < 1.0 && break
    end

    objective = validate_solution(inst, graphs, best_routes)
    bound = min(lower, objective)
    status = _status(bound, objective, true)
    status == "Optimal" && (bound = objective)
    return SolveResult(
        objective, bound, _gap(bound, objective), optimize_time, nodes, status, best_routes, "column-generation",
    )
end
