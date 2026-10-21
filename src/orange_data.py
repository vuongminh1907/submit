"""Data loading and shortest-path split coefficients for Orange T-ASR.

The implementation follows ``project/calculating_r.pdf`` exactly:

    r(i,j,a,t) = sigma(i,u,t) * sigma(v,j,t) / sigma(i,j,t)

when arc a=(u,v) belongs to a shortest i-j path, and zero otherwise.
JSON node and link identifiers are never assumed to be contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Arc:
    index: int
    link_id: int
    u: int
    v: int
    metric: float
    capacity: float


@dataclass
class OrangeInstance:
    name: str
    node_ids: list[int]
    id_to_node: dict[int, int]
    arcs: list[Arc]
    demand_source: np.ndarray
    demand_target: np.ndarray
    volumes: np.ndarray
    max_segments: int
    budget: dict[int, int]
    interventions: dict[int, set[int]]

    @property
    def n(self) -> int:
        return len(self.node_ids)

    @property
    def m(self) -> int:
        return len(self.arcs)

    @property
    def num_demands(self) -> int:
        return len(self.demand_source)

    @property
    def num_periods(self) -> int:
        return self.volumes.shape[1]


def load_instance(seta_dir: str | Path, instance: int | str) -> OrangeInstance:
    """Load one SetA instance from its net/tm/scenario JSON triplet."""
    number = f"{int(instance):02d}"
    base = Path(seta_dir)
    prefix = base / f"setA-{number}"
    with open(f"{prefix}-net.json", encoding="utf-8") as f:
        net = json.load(f)
    with open(f"{prefix}-tm.json", encoding="utf-8") as f:
        tm = json.load(f)
    with open(f"{prefix}-scenario.json", encoding="utf-8") as f:
        scenario = json.load(f)

    if not net.get("directed", False):
        raise ValueError("The Orange instances must be directed.")
    node_ids = [int(node["id"]) for node in net["nodes"]]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Duplicate node identifier.")
    id_to_node = {node_id: i for i, node_id in enumerate(node_ids)}

    arcs: list[Arc] = []
    seen_links: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    for index, link in enumerate(net["links"]):
        link_id = int(link["id"])
        if link_id in seen_links:
            raise ValueError(f"Duplicate link id {link_id}.")
        seen_links.add(link_id)
        u = id_to_node[int(link["from"])]
        v = id_to_node[int(link["to"])]
        # Supplied SetA files declare some graphs as multigraphs, but currently
        # contain no parallel ordered pair. Reject ambiguity if that changes.
        if (u, v) in seen_pairs:
            raise ValueError(
                "Parallel arcs need an explicit ECMP convention; "
                f"duplicate ordered pair {link['from']} -> {link['to']}."
            )
        seen_pairs.add((u, v))
        metric = float(link["metric"])
        capacity = float(link["capacity"])
        if metric <= 0 or capacity <= 0:
            raise ValueError("Metrics and capacities must be strictly positive.")
        arcs.append(Arc(index, link_id, u, v, metric, capacity))

    demands = tm["demands"]
    sources = np.array([id_to_node[int(d["s"])] for d in demands], dtype=int)
    targets = np.array([id_to_node[int(d["t"])] for d in demands], dtype=int)
    volumes = np.asarray([d["v"] for d in demands], dtype=float)
    periods = int(tm["num_time_slots"])
    if volumes.shape != (len(demands), periods):
        raise ValueError("Traffic matrix dimensions do not match num_time_slots.")

    budget = {int(item["t"]): int(item["value"]) for item in scenario["budget"]}
    interventions = {
        int(item["t"]): {int(link_id) for link_id in item["links"]}
        for item in scenario["interventions"]
    }
    return OrangeInstance(
        name=f"setA-{number}",
        node_ids=node_ids,
        id_to_node=id_to_node,
        arcs=arcs,
        demand_source=sources,
        demand_target=targets,
        volumes=volumes,
        max_segments=int(scenario["max_segments"]),
        budget=budget,
        interventions=interventions,
    )


class PeriodData:
    """Shortest-path information for one time period."""

    def __init__(self, instance: OrangeInstance, period: int):
        self.instance = instance
        self.period = period
        down = instance.interventions.get(period, set())
        self.available = np.array(
            [arc.link_id not in down for arc in instance.arcs], dtype=bool
        )
        self.adjacency: list[list[tuple[int, float]]] = [
            [] for _ in range(instance.n)
        ]
        for arc in instance.arcs:
            if self.available[arc.index]:
                self.adjacency[arc.u].append((arc.v, arc.metric))
        self.dist, self.sigma = self._all_pairs_shortest_data()
        self._ratio_cache: dict[tuple[int, int], np.ndarray] = {}

    def _dijkstra_count(self, source: int) -> tuple[np.ndarray, list[int]]:
        n = self.instance.n
        distance = np.full(n, np.inf)
        sigma = [0] * n
        distance[source] = 0.0
        sigma[source] = 1
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            dist_u, u = heapq.heappop(queue)
            if dist_u > distance[u]:
                continue
            for v, weight in self.adjacency[u]:
                alternative = dist_u + weight
                if not math.isfinite(distance[v]):
                    distance[v] = alternative
                    sigma[v] = sigma[u]
                    heapq.heappush(queue, (alternative, v))
                    continue
                tolerance = 1e-10 * max(
                    1.0, abs(alternative), abs(distance[v])
                )
                if alternative < distance[v] - tolerance:
                    distance[v] = alternative
                    sigma[v] = sigma[u]
                    heapq.heappush(queue, (alternative, v))
                elif abs(alternative - distance[v]) <= tolerance:
                    sigma[v] += sigma[u]
        return distance, sigma

    def _all_pairs_shortest_data(self) -> tuple[np.ndarray, list[list[int]]]:
        n = self.instance.n
        distances = np.empty((n, n), dtype=float)
        counts: list[list[int]] = []
        for source in range(n):
            distance, sigma = self._dijkstra_count(source)
            distances[source] = distance
            counts.append(sigma)
        return distances, counts

    def reachable(self, source: int, target: int) -> bool:
        return source != target and math.isfinite(self.dist[source, target])

    def arc_ratios(self, source: int, target: int) -> np.ndarray:
        """Return r(source,target,a,period) for every physical arc."""
        key = (source, target)
        cached = self._ratio_cache.get(key)
        if cached is not None:
            return cached
        result = np.zeros(self.instance.m, dtype=float)
        denominator = self.sigma[source][target]
        if source == target or denominator == 0:
            self._ratio_cache[key] = result
            return result
        shortest = self.dist[source, target]
        for arc in self.instance.arcs:
            if not self.available[arc.index]:
                continue
            left = self.dist[source, arc.u]
            right = self.dist[arc.v, target]
            if not (math.isfinite(left) and math.isfinite(right)):
                continue
            total = left + arc.metric + right
            tolerance = 1e-9 * max(1.0, abs(shortest), abs(total))
            if abs(total - shortest) <= tolerance:
                result[arc.index] = (
                    self.sigma[source][arc.u]
                    * self.sigma[arc.v][target]
                    / denominator
                )
        self._ratio_cache[key] = result
        return result

    def route_ratios(self, route: Iterable[int]) -> np.ndarray:
        """Sum split coefficients for consecutive nodes in a segment path."""
        nodes = list(route)
        result = np.zeros(self.instance.m, dtype=float)
        for source, target in zip(nodes, nodes[1:]):
            if not self.reachable(source, target):
                raise ValueError(
                    f"Unreachable segment {source}->{target} at t={self.period}."
                )
            result += self.arc_ratios(source, target)
        return result


def route_segments(route: Iterable[int]) -> set[tuple[int, int]]:
    nodes = list(route)
    return set(zip(nodes, nodes[1:]))


def route_distance(first: Iterable[int], second: Iterable[int]) -> int:
    """Symmetric-difference distance used by budget constraint (4)."""
    return len(route_segments(first) ^ route_segments(second))


def loads_for_routes(
    instance: OrangeInstance,
    period_data: PeriodData,
    routes: list[list[int]],
    period: int,
) -> np.ndarray:
    flow = np.zeros(instance.m, dtype=float)
    for demand, route in enumerate(routes):
        flow += instance.volumes[demand, period] * period_data.route_ratios(route)
    capacity = np.array([arc.capacity for arc in instance.arcs])
    load = flow / capacity
    load[~period_data.available] = 0.0
    return load
