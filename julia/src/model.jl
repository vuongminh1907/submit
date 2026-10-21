# Exact T-ASR models.
#
# The compact model is the formulation required by the subject:
#   (1) flow conservation, (2) segment limit, (3) load/MLU linking,
#   (4) reconfiguration budget. Degree inequalities are valid cuts that
#   make every selected support a single simple segment path.
#
# Large instances have O(|D||T||V|^2) binaries. For those instances the
# same formulation is solved through its equivalent path representation:
# column generation certifies the LP bound, then HiGHS branch-and-bound
# optimizes the generated integer master. The reported gap is
# 1 - (best valid lower bound) / (best feasible solution).

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

const COMPACT_VARIABLE_LIMIT = 100_000

function _gap(bound::Float64, objective::Float64)
    objective <= 1e-12 && return 0.0
    return max(0.0, 1.0 - bound / objective)
end

function _status(bound::Float64, objective::Float64, feasible::Bool)
    feasible || return "Infeasible"
    return _gap(bound, objective) <= 1e-6 ? "Optimal" : "Not Optimal"
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
    return model
end

function _set_node_cap!(model, node_cap::Int)
    no_node_cap = lowercase(strip(get(ENV, "NO_NODE_CAP", "0")))
    no_node_cap in ("1", "true", "yes", "on") && return
    set_optimizer_attribute(model, "mip_max_nodes", node_cap)
end

function solve_instance(
    inst::Instance, graphs::Vector{PeriodGraph}, time_limit::Float64;
    random_seed::Int=1,
)
    estimated = ndemands(inst) * length(graphs) * inst.n * max(inst.n - 1, 1)
    if estimated <= COMPACT_VARIABLE_LIMIT
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
                flow = AffExpr(0.0)
                for p in outgoing[node]
                    add_to_expression!(flow, x[t, d, p], 1.0)
                end
                for p in incoming[node]
                    add_to_expression!(flow, x[t, d, p], -1.0)
                end
                rhs = node == source ? 1.0 : node == target ? -1.0 : 0.0
                @constraint(model, flow == rhs)
                out_expr = AffExpr(0.0)
                for p in outgoing[node]
                    add_to_expression!(out_expr, x[t, d, p], 1.0)
                end
                @constraint(model, out_expr <= (node == target ? 0.0 : 1.0))
                in_expr = AffExpr(0.0)
                for p in incoming[node]
                    add_to_expression!(in_expr, x[t, d, p], 1.0)
                end
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

    started = time()
    optimize!(model)
    elapsed = time() - started
    return _result_from_compact(inst, graphs, model, pairs, x, L, elapsed)
end

function _result_from_compact(inst, graphs, model, pairs, x, L, elapsed)
    nodes = _node_count(model)
    termination = termination_status(model)
    if termination == MOI.INFEASIBLE || !has_values(model)
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
        Float64(objective_bound(model))
    catch
        0.0
    end
    bound = min(bound, objective)
    status = termination == MOI.OPTIMAL ? "Optimal" : _status(bound, objective, true)
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
                penalty = extra[u, v]
                isfinite(penalty) || continue
                candidate = base + volume * weight + penalty
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

function _unit_costs(g::PeriodGraph, dual_load::Vector{Float64})
    n = size(g.dist, 1)
    unit = fill(Inf, n, n)
    @inbounds for i in 1:n, j in 1:n
        i == j && continue
        arcs = g.seg_arcs[i, j]
        isempty(arcs) && continue
        ratios = g.seg_ratios[i, j]
        value = 0.0
        for k in eachindex(arcs)
            value += dual_load[arcs[k]] * ratios[k]
        end
        unit[i, j] = value
    end
    return unit
end

function _path_load_cost(volume::Float64, ratios::Vector{Float64}, dual_load::Vector{Float64})
    value = 0.0
    @inbounds for a in eachindex(ratios)
        value += dual_load[a] * ratios[a]
    end
    return volume * value
end

"""Solve one period, ignoring the inter-period budget."""
function solve_period_columns(
    inst::Instance, g::PeriodGraph, time_index::Int, time_limit::Float64,
    random_seed::Int=1,
)
    demands = ndemands(inst)
    m = length(inst.arcs)
    model = _configure!(Model(HiGHS.Optimizer), time_limit, random_seed)
    @variable(model, L >= 0)
    @objective(model, Min, L)
    convexity = Vector{ConstraintRef}(undef, demands)
    for d in 1:demands
        convexity[d] = @constraint(model, 0.0 == 1.0)
    end
    load_constraints = Vector{Any}(undef, m)
    for a in 1:m
        load_constraints[a] = g.available[a] ?
            @constraint(model, -inst.arcs[a].capacity * L <= 0) : nothing
    end
    columns = [Vector{Int}[] for _ in 1:demands]
    variables = [VariableRef[] for _ in 1:demands]
    seen = [Set{Vector{Int}}() for _ in 1:demands]

    function add_column!(d::Int, nodes::Vector{Int})
        nodes in seen[d] && return false
        push!(seen[d], copy(nodes))
        variable = @variable(model, lower_bound = 0.0)
        push!(variables[d], variable)
        push!(columns[d], nodes)
        set_normalized_coefficient(convexity[d], variable, 1.0)
        ratios = path_ratios(g, nodes)
        volume = inst.volume[d, time_index + 1]
        for a in 1:m
            g.available[a] || continue
            coefficient = volume * ratios[a]
            coefficient == 0 && continue
            set_normalized_coefficient(load_constraints[a], variable, coefficient)
        end
        return true
    end

    for d in 1:demands
        add_column!(d, [inst.source[d], inst.target[d]])
    end

    optimize_time = 0.0
    started = time()
    certified = false
    bound = 0.0
    # HiGHS reports reduced_cost(z) = -(dual_load · a_z) - dual(convexity).
    # Checked on setA-01 against both basic and nonbasic columns.
    sign = -1.0
    convexity_sign = -1.0
    checked_costs = false

    pricing_rounds = 0
    for _ in 1:120
        pricing_rounds += 1
        remaining = time_limit - (time() - started)
        remaining <= 1 && break
        set_time_limit_sec(model, remaining)
        call = time()
        optimize!(model)
        optimize_time += time() - call
        termination_status(model) == MOI.OPTIMAL || break
        lp_obj = Float64(objective_value(model))
        dual_load = zeros(m)
        for a in 1:m
            g.available[a] && (dual_load[a] = dual(load_constraints[a]))
        end
        alpha = [dual(convexity[d]) for d in 1:demands]
        if !checked_costs
            for d in 1:demands
                volume = inst.volume[d, time_index + 1]
                for (p, variable) in enumerate(variables[d])
                    ratios = path_ratios(g, columns[d][p])
                    base = _path_load_cost(volume, ratios, dual_load)
                    predicted = -base - alpha[d]
                    abs(predicted - reduced_cost(variable)) < 1e-4 ||
                        error("Reduced-cost formula does not match HiGHS.")
                end
            end
            checked_costs = true
        end
        unit = _unit_costs(g, sign .* dual_load)
        nonnegative = true
        @inbounds for weight in unit
            if isfinite(weight) && weight < -1e-9
                nonnegative = false
                break
            end
        end
        extra = zeros(inst.n, inst.n)
        added = 0
        reduced_sum = 0.0
        for d in 1:demands
            volume = inst.volume[d, time_index + 1]
            path = Int[]
            reduced = 0.0
            if nonnegative
                unlimited, unit_cost = _dense_path(unit, inst.source[d], inst.target[d])
                reduced = isempty(unlimited) ? 0.0 : volume * unit_cost + convexity_sign * alpha[d]
                if reduced >= -1e-7 || isempty(unlimited)
                    continue
                elseif length(unlimited) - 1 <= inst.maxseg
                    path = unlimited
                end
            end
            if isempty(path)
                path, cost = _best_path(unit, extra, inst.source[d], inst.target[d], inst.maxseg, volume)
                reduced = isempty(path) ? 0.0 : cost + convexity_sign * alpha[d]
            end
            reduced_sum += min(reduced, 0.0)
            if reduced < -1e-7
                add_column!(d, path) ||
                    error("Reduced-cost formula disagrees with a column already in the pool.")
                added += 1
            end
        end
        lp_bound = lp_obj + reduced_sum
        bound = max(bound, lp_bound)
        if added == 0
            certified = true
            bound = max(bound, lp_obj)
            break
        end
    end
    println("pricing rounds=$pricing_rounds certified=$certified bound=$bound columns=$(sum(length, columns))")
    flush(stdout)

    remaining = max(time_limit - (time() - started), 0.01)
    for d in 1:demands, variable in variables[d]
        set_binary(variable)
    end
    # The linear bound is already valid. A long branch-and-bound search can
    # improve the incumbent, but identical searches on all ten runs would
    # consume the whole limit without changing that bound. Cap the integer
    # search and keep the direct routing as a feasible start.
    direct_load = maximum_load(inst, g, [columns[d][1] for d in 1:demands], time_index)
    set_start_value(L, direct_load)
    for d in 1:demands, (p, variable) in enumerate(variables[d])
        set_start_value(variable, p == 1 ? 1.0 : 0.0)
    end
    set_time_limit_sec(model, remaining)
    # Large pools make every branch-and-bound node expensive. The linear bound
    # is already certified, so the node cap only limits the incumbent search.
    node_cap = sum(length, columns) > 2_000 ? 20 : 400
    _set_node_cap!(model, node_cap)
    call = time()
    optimize!(model)
    optimize_time += time() - call
    nodes = _node_count(model)
    routes = Vector{Vector{Int}}()
    if has_values(model)
        for d in 1:demands
            values = [value(variable) for variable in variables[d]]
            push!(routes, columns[d][argmax(values)])
        end
    else
        routes = direct_routes(inst, g)
    end
    objective = try
        maximum_load(inst, g, routes, time_index)
    catch
        NaN
    end
    if !isfinite(objective)
        return SolveResult(NaN, bound, NaN, optimize_time, nodes, "Infeasible", [routes], "column-generation"), columns
    end
    bound = min(max(bound, 0.0), objective)
    status = certified ? _status(bound, objective, true) : "Not Optimal"
    status == "Optimal" && (bound = objective)
    return SolveResult(
        objective, bound, _gap(bound, objective), optimize_time, nodes, status, [routes], "column-generation",
    ), columns
end

function solve_by_columns(
    inst::Instance, graphs::Vector{PeriodGraph}, time_limit::Float64, random_seed::Int=1,
)
    started = time()
    if length(graphs) == 1
        result, _ = solve_period_columns(inst, graphs[1], 0, time_limit, random_seed)
        objective = validate_solution(inst, graphs, result.routes)
        bound = min(result.bound, objective)
        return SolveResult(
            objective, bound, _gap(bound, objective), result.optimize_time, result.nodes,
            _status(bound, objective, true), result.routes, result.method,
        )
    end

    first, pool1 = solve_period_columns(inst, graphs[1], 0, 0.4 * time_limit, random_seed)
    second, pool2 = solve_period_columns(inst, graphs[2], 1, 0.4 * time_limit, random_seed + 1)
    # Each certified single-period bound remains valid after the budget is added.
    lower = max(first.bound, second.bound, 0.0)
    joint = _joint_mip_explicit(
        inst, graphs, [_enrich_pool(inst, graphs[1], pool1), _enrich_pool(inst, graphs[2], pool2)],
        time_limit - (time() - started), lower, random_seed + 2,
    )
    routes = _best_joint_routes(inst, graphs, joint.routes, first.routes[1], second.routes[1])
    objective = validate_solution(inst, graphs, routes)
    bound = min(lower, objective)
    status = _status(bound, objective, true)
    status == "Optimal" && (bound = objective)
    return SolveResult(
        objective, bound, _gap(bound, objective),
        first.optimize_time + second.optimize_time + joint.optimize_time,
        first.nodes + second.nodes + joint.nodes,
        status, routes, "column-generation",
    )
end

struct _JointMIP
    routes::Vector{Vector{Vector{Int}}}
    optimize_time::Float64
    nodes::Int
    bound_hint::Float64
end

function _best_joint_routes(inst, graphs, mip_routes, first_routes, second_routes)
    repaired = _fit_budget(inst, graphs, first_routes, second_routes)
    best = mip_routes
    best_load = try
        validate_solution(inst, graphs, mip_routes)
    catch
        Inf
    end
    repaired_load = try
        validate_solution(inst, graphs, repaired)
    catch
        Inf
    end
    if repaired_load < best_load
        return repaired
    end
    return best
end

function _route_ok(inst, g, route, d)
    route[1] == inst.source[d] && route[end] == inst.target[d] || return false
    length(route) - 1 <= inst.maxseg || return false
    length(unique(route)) == length(route) || return false
    return all(reachable(g, route[k], route[k + 1]) for k in 1:(length(route) - 1))
end

function _fit_budget(inst, graphs, first_routes, second_routes)
    demands = ndemands(inst)
    left = [copy(first_routes[d]) for d in 1:demands]
    right = [copy(second_routes[d]) for d in 1:demands]
    for d in 1:demands
        _route_ok(inst, graphs[1], left[d], d) || (left[d] = [inst.source[d], inst.target[d]])
        _route_ok(inst, graphs[2], right[d], d) || (right[d] = [inst.source[d], inst.target[d]])
    end
    order = sort(1:demands, by = d -> route_distance(left[d], right[d]), rev = true)
    for d in order
        route_distance(left[d], right[d]) == 0 && break
        used = sum(route_distance(left[k], right[k]) for k in 1:demands)
        used <= inst.budget[1] && break
        if _route_ok(inst, graphs[2], left[d], d)
            right[d] = copy(left[d])
        elseif _route_ok(inst, graphs[1], right[d], d)
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
        if length(pool[d]) < 6
            for waypoint in _waypoint_sample(inst.n, inst.source[d], inst.target[d])
                reachable(g, inst.source[d], waypoint) && reachable(g, waypoint, inst.target[d]) || continue
                path = [inst.source[d], waypoint, inst.target[d]]
                path in pool[d] && continue
                push!(pool[d], path)
                length(pool[d]) >= 6 && break
            end
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

function _joint_mip_explicit(inst, graphs, pools, time_limit, known_bound, random_seed::Int=1)
    model = _configure!(Model(HiGHS.Optimizer), max(time_limit, 0.01), random_seed)
    demands = ndemands(inst)
    @variable(model, L >= 0)
    @objective(model, Min, L)
    z = Matrix{Vector{VariableRef}}(undef, 2, demands)
    for t in 1:2, d in 1:demands
        z[t, d] = [@variable(model, binary=true) for _ in pools[t][d]]
        @constraint(model, sum(z[t, d]) == 1)
        # Prefer the direct segment path as a feasible warm start.
        set_start_value(z[t, d][1], 1.0)
        for p in 2:length(z[t, d])
            set_start_value(z[t, d][p], 0.0)
        end
    end
    sparse = [[Tuple{Vector{Int},Vector{Float64}}[] for _ in 1:demands] for _ in 1:2]
    for t in 1:2, d in 1:demands, nodes in pools[t][d]
        ratios = path_ratios(graphs[t], nodes)
        arcs = Int[]
        values = Float64[]
        for a in eachindex(ratios)
            ratios[a] == 0 && continue
            push!(arcs, a)
            push!(values, ratios[a])
        end
        push!(sparse[t][d], (arcs, values))
    end
    for t in 1:2
        expressions = [AffExpr(0.0) for _ in inst.arcs]
        for (a, arc) in enumerate(inst.arcs)
            graphs[t].available[a] && add_to_expression!(expressions[a], L, -arc.capacity)
        end
        for d in 1:demands
            volume = inst.volume[d, t]
            volume == 0 && continue
            for (p, (arcs, values)) in enumerate(sparse[t][d])
                for k in eachindex(arcs)
                    graphs[t].available[arcs[k]] || continue
                    add_to_expression!(expressions[arcs[k]], z[t, d][p], volume * values[k])
                end
            end
        end
        for a in eachindex(inst.arcs)
            graphs[t].available[a] && @constraint(model, expressions[a] <= 0)
        end
    end
    budget = AffExpr(0.0)
    for d in 1:demands
        segments = Set{Tuple{Int,Int}}()
        for t in 1:2, nodes in pools[t][d]
            union!(segments, segment_set(nodes))
        end
        for segment in segments
            y = @variable(model, lower_bound = 0.0)
            first = AffExpr(0.0)
            second = AffExpr(0.0)
            for (p, nodes) in enumerate(pools[1][d])
                segment in segment_set(nodes) && add_to_expression!(first, z[1, d][p], 1.0)
            end
            for (p, nodes) in enumerate(pools[2][d])
                segment in segment_set(nodes) && add_to_expression!(second, z[2, d][p], 1.0)
            end
            @constraint(model, y >= second - first)
            @constraint(model, y >= first - second)
            add_to_expression!(budget, y, 1.0)
        end
    end
    @constraint(model, budget <= inst.budget[1])
    for t in 1:2, d in 1:demands, (p, variable) in enumerate(z[t, d])
        set_start_value(variable, p == 1 ? 1.0 : 0.0)
    end
    column_count = sum(length(pool) for pool in pools for paths in pool)
    _set_node_cap!(model, column_count > 2_000 ? 20 : 400)
    started = time()
    optimize!(model)
    elapsed = time() - started
    routes = [direct_routes(inst, graphs[1]), direct_routes(inst, graphs[2])]
    if has_values(model) && termination_status(model) != MOI.INFEASIBLE
        for t in 1:2
            for d in 1:demands
                values = [value(variable) for variable in z[t, d]]
                routes[t][d] = pools[t][d][argmax(values)]
            end
        end
    end
    try
        validate_solution(inst, graphs, routes)
    catch
        routes = [direct_routes(inst, graphs[1]), direct_routes(inst, graphs[2])]
    end
    # The restricted master's dual bound is not valid for columns outside its
    # pool, so the only global lower bound retained is `known_bound`.
    return _JointMIP(routes, elapsed, _node_count(model), known_bound)
end
