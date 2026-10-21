"""Feasible local-search solver for large T-ASR instances.

The exact arc-flow MIP has O(|D||T||V|^2) binary variables.  This module
constructs valid segment paths directly, so it can process every SetA
instance while enforcing reachability, maxSeg, and inter-period budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time

import numpy as np

from orange_data import (
    OrangeInstance,
    PeriodData,
    route_distance,
    route_segments,
)


@dataclass
class HeuristicResult:
    routes: list[list[list[int]]]
    loads: list[np.ndarray]
    objective: float
    runtime: float
    iterations: int
    budget_used: dict[int, int]
    status: str = "FEASIBLE_HEURISTIC"


def _initial_routes(
    instance: OrangeInstance,
    periods: list[PeriodData],
) -> list[list[list[int]]]:
    all_routes: list[list[list[int]]] = []
    for t, data in enumerate(periods):
        routes: list[list[int]] = []
        for d, (source, target) in enumerate(
            zip(instance.demand_source, instance.demand_target)
        ):
            source, target = int(source), int(target)
            if data.reachable(source, target):
                routes.append([source, target])
                continue
            # A disconnected demand makes the instance infeasible for any
            # segment-routing path, because waypoints cannot reconnect SCCs.
            raise ValueError(
                f"Demand {d} ({source}->{target}) is disconnected at t={t}."
            )
        all_routes.append(routes)
    return all_routes


def _flow_and_contributions(
    instance: OrangeInstance,
    data: PeriodData,
    routes: list[list[int]],
    period: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    contributions: list[np.ndarray] = []
    flow = np.zeros(instance.m)
    for d, route in enumerate(routes):
        contribution = (
            instance.volumes[d, period] * data.route_ratios(route)
        )
        contributions.append(contribution)
        flow += contribution
    return flow, contributions


def _candidate_nodes(
    n: int,
    source: int,
    target: int,
    rng: random.Random,
    count: int,
) -> list[int]:
    choices = [v for v in range(n) if v not in (source, target)]
    if len(choices) <= count:
        return choices
    return rng.sample(choices, count)


def solve_heuristic(
    instance: OrangeInstance,
    seed: int = 1,
    max_iterations: int = 80,
    candidate_count: int = 20,
    time_limit: float = 120.0,
    periods_to_solve: int | None = None,
) -> HeuristicResult:
    """Solve periods sequentially with budget-aware one-waypoint local search."""
    started = time.perf_counter()
    rng = random.Random(seed)
    num_periods = periods_to_solve or instance.num_periods
    period_data = [PeriodData(instance, t) for t in range(num_periods)]
    routes = _initial_routes(instance, period_data)
    capacities = np.array([arc.capacity for arc in instance.arcs])
    all_loads: list[np.ndarray] = []
    budget_used: dict[int, int] = {}
    total_iterations = 0

    for period, data in enumerate(period_data):
        # Start t>0 from the preceding configuration whenever all its segments
        # remain reachable; this costs zero and makes the budget explicit.
        if period > 0:
            copied: list[list[int]] = []
            for d, previous in enumerate(routes[period - 1]):
                if all(data.reachable(i, j) for i, j in zip(previous, previous[1:])):
                    copied.append(previous.copy())
                else:
                    copied.append(
                        [int(instance.demand_source[d]), int(instance.demand_target[d])]
                    )
            routes[period] = copied

        flow, contributions = _flow_and_contributions(
            instance, data, routes[period], period
        )
        available = data.available
        load = np.zeros(instance.m)
        load[available] = flow[available] / capacities[available]
        current_budget = (
            sum(
                route_distance(routes[period - 1][d], routes[period][d])
                for d in range(instance.num_demands)
            )
            if period > 0
            else 0
        )
        budget_limit = instance.budget.get(period, 10**18)

        for _ in range(max_iterations):
            if time.perf_counter() - started >= time_limit:
                break
            total_iterations += 1
            bottleneck = int(np.argmax(load))
            current_objective = float(load[bottleneck])

            # Focus on demands that put the most traffic on the current
            # bottleneck, but randomize among the leading candidates.
            ranked = np.argsort(
                [-contribution[bottleneck] for contribution in contributions]
            )
            positive = [
                int(d)
                for d in ranked[: min(30, len(ranked))]
                if contributions[int(d)][bottleneck] > 1e-12
            ]
            if not positive:
                break
            demand = rng.choice(positive)
            source = int(instance.demand_source[demand])
            target = int(instance.demand_target[demand])

            candidate_routes = [[source, target]]
            for waypoint in _candidate_nodes(
                instance.n, source, target, rng, candidate_count
            ):
                if data.reachable(source, waypoint) and data.reachable(
                    waypoint, target
                ):
                    candidate_routes.append([source, waypoint, target])

            old_route = routes[period][demand]
            old_contribution = contributions[demand]
            best_route = old_route
            best_contribution = old_contribution
            best_load = load
            best_objective = current_objective
            best_budget = current_budget

            for candidate in candidate_routes:
                if candidate == old_route:
                    continue
                if len(candidate) - 1 > instance.max_segments:
                    continue
                proposed_budget = current_budget
                if period > 0:
                    previous = routes[period - 1][demand]
                    proposed_budget += (
                        route_distance(previous, candidate)
                        - route_distance(previous, old_route)
                    )
                    if proposed_budget > budget_limit:
                        continue
                candidate_contribution = (
                    instance.volumes[demand, period]
                    * data.route_ratios(candidate)
                )
                candidate_flow = flow - old_contribution + candidate_contribution
                candidate_load = np.zeros(instance.m)
                candidate_load[available] = (
                    candidate_flow[available] / capacities[available]
                )
                objective = float(np.max(candidate_load))
                # Strict MLU improvement. The tolerance avoids accepting
                # numerical noise and guarantees termination of accepted moves.
                if objective < best_objective - 1e-10:
                    best_route = candidate
                    best_contribution = candidate_contribution
                    best_load = candidate_load
                    best_objective = objective
                    best_budget = proposed_budget

            if best_route != old_route:
                flow += best_contribution - old_contribution
                routes[period][demand] = best_route
                contributions[demand] = best_contribution
                load = best_load
                current_budget = best_budget

        all_loads.append(load)
        if period > 0:
            budget_used[period] = current_budget

    objective = max(float(np.max(load)) for load in all_loads)
    return HeuristicResult(
        routes=routes,
        loads=all_loads,
        objective=objective,
        runtime=time.perf_counter() - started,
        iterations=total_iterations,
        budget_used=budget_used,
    )


def routes_to_json(
    instance: OrangeInstance,
    routes: list[list[list[int]]],
) -> dict:
    """Convert internal complete segment paths to challenge waypoint JSON."""
    output = []
    for period, period_routes in enumerate(routes):
        for demand, route in enumerate(period_routes):
            waypoints = [instance.node_ids[node] for node in route[1:-1]]
            if waypoints:
                output.append({"d": demand, "t": period, "w": waypoints})
    return {"srpaths": output}
