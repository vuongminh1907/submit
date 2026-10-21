# Data, shortest-path counts, and split coefficients r(i,j,a,t).
# The coefficient formula is the one required by project/calculating_r.pdf.

struct Arc
    link_id::Int
    u::Int
    v::Int
    metric::Float64
    capacity::Float64
end

struct Instance
    name::String
    n::Int
    node_ids::Vector{Int}
    arcs::Vector{Arc}
    source::Vector{Int}
    target::Vector{Int}
    volume::Matrix{Float64}   # demand × time slot, columns are t = 0,1,...
    maxseg::Int
    budget::Dict{Int,Int}     # JSON time index => allowed changes
    interventions::Dict{Int,Set{Int}}
end

ndemands(inst::Instance) = length(inst.source)
nperiods(inst::Instance) = size(inst.volume, 2)

struct PeriodGraph
    available::BitVector
    dist::Matrix{Float64}
    logsigma::Matrix{Float64}
    seg_arcs::Matrix{Vector{Int}}
    seg_ratios::Matrix{Vector{Float64}}
end

function _heappush!(heap::Vector{Tuple{Float64,Int}}, item::Tuple{Float64,Int})
    push!(heap, item)
    i = length(heap)
    @inbounds while i > 1
        parent = i >> 1
        if heap[parent][1] <= heap[i][1]
            break
        end
        heap[parent], heap[i] = heap[i], heap[parent]
        i = parent
    end
    return heap
end

function _heappop!(heap::Vector{Tuple{Float64,Int}})
    item = heap[1]
    last = pop!(heap)
    if !isempty(heap)
        heap[1] = last
        i = 1
        n = length(heap)
        @inbounds while true
            left = i << 1
            left > n && break
            right = left + 1
            child = right <= n && heap[right][1] < heap[left][1] ? right : left
            heap[child][1] < heap[i][1] || break
            heap[child], heap[i] = heap[i], heap[child]
            i = child
        end
    end
    return item
end

function _logaddexp(a::Float64, b::Float64)
    a == -Inf && return b
    b == -Inf && return a
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

function _dijkstra(adj, source::Int, n::Int)
    dist = fill(Inf, n)
    logsigma = fill(-Inf, n)
    dist[source] = 0.0
    logsigma[source] = 0.0
    heap = Tuple{Float64,Int}[(0.0, source)]
    while !isempty(heap)
        du, u = _heappop!(heap)
        du > dist[u] && continue
        for (v, weight) in adj[u]
            alternative = du + weight
            if !isfinite(dist[v]) ||
               alternative < dist[v] - 1e-10 * max(1.0, abs(alternative), abs(dist[v]))
                dist[v] = alternative
                logsigma[v] = logsigma[u]
                _heappush!(heap, (alternative, v))
            elseif abs(alternative - dist[v]) <=
                   1e-10 * max(1.0, abs(alternative), abs(dist[v]))
                logsigma[v] = _logaddexp(logsigma[v], logsigma[u])
            end
        end
    end
    return dist, logsigma
end

function load_instance(directory::AbstractString, number::Integer)
    prefix = joinpath(directory, "setA-" * lpad(string(number), 2, '0'))
    net = JSON3.read(read(prefix * "-net.json", String))
    tm = JSON3.read(read(prefix * "-tm.json", String))
    scenario = JSON3.read(read(prefix * "-scenario.json", String))
    net.directed || error("The network must be directed.")

    node_ids = Int[Int(node.id) for node in net.nodes]
    length(unique(node_ids)) == length(node_ids) || error("Duplicate node id.")
    id_to_node = Dict(id => i for (i, id) in enumerate(node_ids))
    arcs = Arc[]
    seen_links = Set{Int}()
    seen_pairs = Set{Tuple{Int,Int}}()
    for link in net.links
        link_id = Int(link.id)
        link_id in seen_links && error("Duplicate link id $link_id.")
        push!(seen_links, link_id)
        u = id_to_node[Int(link.from)]
        v = id_to_node[Int(link.to)]
        (u, v) in seen_pairs &&
            error("Parallel arcs are not present in the supplied formulation.")
        push!(seen_pairs, (u, v))
        metric = Float64(link.metric)
        capacity = Float64(link.capacity)
        (metric > 0 && capacity > 0) || error("Metric and capacity must be positive.")
        push!(arcs, Arc(link_id, u, v, metric, capacity))
    end

    demands = collect(tm.demands)
    source = Int[id_to_node[Int(d.s)] for d in demands]
    target = Int[id_to_node[Int(d.t)] for d in demands]
    slots = Int(tm.num_time_slots)
    volume = Matrix{Float64}(undef, length(demands), slots)
    for (d, demand) in enumerate(demands)
        values = collect(demand.v)
        length(values) == slots || error("Demand volume length mismatch.")
        volume[d, :] .= Float64.(values)
    end
    any(source .== target) && error("A demand has equal source and target.")

    budget = Dict(Int(item.t) => Int(item.value) for item in scenario.budget)
    interventions = Dict(
        Int(item.t) => Set(Int(link) for link in item.links)
        for item in scenario.interventions
    )
    return Instance(
        "setA-" * lpad(string(number), 2, '0'),
        length(node_ids),
        node_ids,
        arcs,
        source,
        target,
        volume,
        Int(scenario.max_segments),
        budget,
        interventions,
    )
end

function build_period(inst::Instance, time_index::Int)
    down = time_index == 0 ? Set{Int}() : get(inst.interventions, time_index, Set{Int}())
    available = BitVector(arc.link_id ∉ down for arc in inst.arcs)
    n = inst.n
    m = length(inst.arcs)
    adj = [Tuple{Int,Float64}[] for _ in 1:n]
    for (a, arc) in enumerate(inst.arcs)
        available[a] && push!(adj[arc.u], (arc.v, arc.metric))
    end
    dist = Matrix{Float64}(undef, n, n)
    logsigma = Matrix{Float64}(undef, n, n)
    for source in 1:n
        row_dist, row_log = _dijkstra(adj, source, n)
        dist[source, :] .= row_dist
        logsigma[source, :] .= row_log
    end
    for i in 1:n
        dist[i, i] = 0.0
        logsigma[i, i] = 0.0
    end

    seg_arcs = [Int[] for _ in 1:n, _ in 1:n]
    seg_ratios = [Float64[] for _ in 1:n, _ in 1:n]
    @inbounds for i in 1:n
        for a in 1:m
            available[a] || continue
            arc = inst.arcs[a]
            left = dist[i, arc.u]
            isfinite(left) || continue
            left_log = logsigma[i, arc.u]
            for j in 1:n
                i == j && continue
                denominator = logsigma[i, j]
                denominator == -Inf && continue
                right = dist[arc.v, j]
                isfinite(right) || continue
                total = left + arc.metric + right
                shortest = dist[i, j]
                abs(total - shortest) <= 1e-8 * max(1.0, abs(shortest)) || continue
                right_log = logsigma[arc.v, j]
                ratio = exp(left_log + right_log - denominator)
                push!(seg_arcs[i, j], a)
                push!(seg_ratios[i, j], ratio)
            end
        end
    end
    return PeriodGraph(available, dist, logsigma, seg_arcs, seg_ratios)
end

reachable(g::PeriodGraph, i::Int, j::Int) =
    i != j && isfinite(g.dist[i, j]) && g.logsigma[i, j] > -Inf && !isempty(g.seg_arcs[i, j])

function add_segment_ratios!(destination::Vector{Float64}, g::PeriodGraph, i::Int, j::Int)
    reachable(g, i, j) || error("Unreachable segment $i -> $j.")
    arcs = g.seg_arcs[i, j]
    ratios = g.seg_ratios[i, j]
    @inbounds for k in eachindex(arcs)
        destination[arcs[k]] += ratios[k]
    end
    return destination
end

function path_ratios(g::PeriodGraph, nodes::Vector{Int})
    ratios = zeros(length(g.available))
    for k in 1:(length(nodes) - 1)
        add_segment_ratios!(ratios, g, nodes[k], nodes[k + 1])
    end
    return ratios
end

segment_set(nodes::Vector{Int}) = Set((nodes[k], nodes[k + 1]) for k in 1:(length(nodes) - 1))

route_distance(a::Vector{Int}, b::Vector{Int}) = length(symdiff(segment_set(a), segment_set(b)))

function maximum_load(inst::Instance, g::PeriodGraph, routes::Vector{Vector{Int}}, time_index::Int)
    flow = zeros(length(inst.arcs))
    column = time_index + 1
    for d in 1:ndemands(inst)
        ratios = path_ratios(g, routes[d])
        flow .+= inst.volume[d, column] .* ratios
    end
    peak = 0.0
    for a in eachindex(inst.arcs)
        g.available[a] || continue
        peak = max(peak, flow[a] / inst.arcs[a].capacity)
    end
    return peak
end

function direct_routes(inst::Instance, g::PeriodGraph)
    routes = Vector{Vector{Int}}(undef, ndemands(inst))
    for d in 1:ndemands(inst)
        s, t = inst.source[d], inst.target[d]
        reachable(g, s, t) || error("Demand $d is disconnected.")
        routes[d] = [s, t]
    end
    return routes
end

function validate_solution(
    inst::Instance,
    graphs::Vector{PeriodGraph},
    routes::Vector{Vector{Vector{Int}}},
)
    length(routes) == length(graphs) || error("Period count mismatch.")
    for (t, g) in enumerate(graphs)
        length(routes[t]) == ndemands(inst) || error("Demand count mismatch.")
        time_index = t - 1
        for d in 1:ndemands(inst)
            route = routes[t][d]
            (route[1] == inst.source[d] && route[end] == inst.target[d]) ||
                error("Demand $d has incorrect terminals at t=$time_index.")
            length(route) - 1 <= inst.maxseg || error("maxSeg violated.")
            length(unique(route)) == length(route) || error("Route contains a cycle.")
            for k in 1:(length(route) - 1)
                reachable(g, route[k], route[k + 1]) ||
                    error("Unreachable segment at t=$time_index.")
            end
        end
    end
    if length(routes) > 1
        used = sum(
            route_distance(routes[1][d], routes[2][d]) for d in 1:ndemands(inst)
        )
        used <= inst.budget[1] || error("Budget violated: $used > $(inst.budget[1]).")
    end
    return maximum(
        maximum_load(inst, graphs[t], routes[t], t - 1) for t in eachindex(graphs)
    )
end
