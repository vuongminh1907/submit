"""Exact binary arc-flow MIP for the Orange T-ASR formulation.

This is the model in constraints (1)--(4) of the subject, with maximum
link utilization (the first and dominant component of objective (5)).
Degree inequalities tighten the formulation and exclude branching.  A
secondary cycle-removal step extracts the simple source-target route;
cycles can never improve nonnegative link loads.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from orange_data import OrangeInstance, PeriodData, loads_for_routes


@dataclass
class ExactResult:
    routes: list[list[list[int]]] | None
    objective: float | None
    dual_bound: float | None
    gap: float | None
    runtime: float
    total_runtime: float
    node_count: int | None
    status: str
    message: str
    num_variables: int
    num_constraints: int
    loads: list[np.ndarray] | None


class _Rows:
    def __init__(self) -> None:
        self.row: list[int] = []
        self.col: list[int] = []
        self.value: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(
        self,
        terms: list[tuple[int, float]],
        lower: float = -np.inf,
        upper: float = np.inf,
    ) -> None:
        index = len(self.lower)
        for variable, coefficient in terms:
            if coefficient:
                self.row.append(index)
                self.col.append(variable)
                self.value.append(float(coefficient))
        self.lower.append(float(lower))
        self.upper.append(float(upper))


def solve_exact(
    instance: OrangeInstance,
    num_periods: int = 1,
    time_limit: float = 1800.0,
    mip_gap: float = 1e-4,
    presolve: bool = True,
) -> ExactResult:
    """Build and solve the exact model for the first ``num_periods``."""
    started = time.perf_counter()
    period_data = [PeriodData(instance, t) for t in range(num_periods)]

    reachable_pairs: list[list[tuple[int, int]]] = []
    for data in period_data:
        reachable_pairs.append(
            [
                (i, j)
                for i in range(instance.n)
                for j in range(instance.n)
                if data.reachable(i, j)
            ]
        )

    x_index: dict[tuple[int, int, int, int], int] = {}
    variable = 0
    for t in range(num_periods):
        for d in range(instance.num_demands):
            for i, j in reachable_pairs[t]:
                x_index[d, t, i, j] = variable
                variable += 1
    num_binary = variable

    # Continuous auxiliaries for |x(t)-x(t-1)|.
    z_index: dict[tuple[int, int, int, int], int] = {}
    if num_periods > 1:
        all_pairs = set().union(*map(set, reachable_pairs))
        for t in range(1, num_periods):
            for d in range(instance.num_demands):
                for i, j in all_pairs:
                    z_index[d, t, i, j] = variable
                    variable += 1

    max_load_index = variable
    variable += 1
    number_variables = variable
    rows = _Rows()

    # (1), plus in/out degree tightening and (2).
    for t in range(num_periods):
        pairs = reachable_pairs[t]
        outgoing: list[list[tuple[int, int]]] = [[] for _ in range(instance.n)]
        incoming: list[list[tuple[int, int]]] = [[] for _ in range(instance.n)]
        for i, j in pairs:
            outgoing[i].append((i, j))
            incoming[j].append((i, j))

        for d in range(instance.num_demands):
            source = int(instance.demand_source[d])
            target = int(instance.demand_target[d])
            for node in range(instance.n):
                flow_terms = [
                    (x_index[d, t, i, j], 1.0) for i, j in outgoing[node]
                ] + [
                    (x_index[d, t, i, j], -1.0) for i, j in incoming[node]
                ]
                rhs = 1.0 if node == source else -1.0 if node == target else 0.0
                rows.add(flow_terms, rhs, rhs)
                rows.add(
                    [(x_index[d, t, i, j], 1.0) for i, j in outgoing[node]],
                    upper=0.0 if node == target else 1.0,
                )
                rows.add(
                    [(x_index[d, t, i, j], 1.0) for i, j in incoming[node]],
                    upper=0.0 if node == source else 1.0,
                )
            rows.add(
                [(x_index[d, t, i, j], 1.0) for i, j in pairs],
                upper=float(instance.max_segments),
            )

    # (3): normalize by capacity so max_load directly represents MLU.
    for t, data in enumerate(period_data):
        for arc in instance.arcs:
            if not data.available[arc.index]:
                continue
            terms: list[tuple[int, float]] = [(max_load_index, -1.0)]
            for i, j in reachable_pairs[t]:
                ratio = data.arc_ratios(i, j)[arc.index]
                if ratio == 0:
                    continue
                for d in range(instance.num_demands):
                    coefficient = (
                        instance.volumes[d, t] * ratio / arc.capacity
                    )
                    if coefficient:
                        terms.append((x_index[d, t, i, j], coefficient))
            rows.add(terms, upper=0.0)

    # (4): standard absolute-value linearization.
    if num_periods > 1:
        all_pairs = set().union(*map(set, reachable_pairs))
        for t in range(1, num_periods):
            budget_terms: list[tuple[int, float]] = []
            for d in range(instance.num_demands):
                for i, j in all_pairs:
                    z = z_index[d, t, i, j]
                    previous = x_index.get((d, t - 1, i, j))
                    current = x_index.get((d, t, i, j))
                    first = [(z, -1.0)]
                    second = [(z, -1.0)]
                    if current is not None:
                        first.append((current, 1.0))
                        second.append((current, -1.0))
                    if previous is not None:
                        first.append((previous, -1.0))
                        second.append((previous, 1.0))
                    rows.add(first, upper=0.0)
                    rows.add(second, upper=0.0)
                    budget_terms.append((z, 1.0))
            rows.add(budget_terms, upper=float(instance.budget[t]))

    matrix = coo_matrix(
        (rows.value, (rows.row, rows.col)),
        shape=(len(rows.lower), number_variables),
    ).tocsr()
    objective = np.zeros(number_variables)
    objective[max_load_index] = 1.0
    lower_bounds = np.zeros(number_variables)
    upper_bounds = np.full(number_variables, np.inf)
    upper_bounds[:num_binary] = 1.0
    if z_index:
        upper_bounds[num_binary:max_load_index] = 1.0
    integrality = np.zeros(number_variables, dtype=np.uint8)
    integrality[:num_binary] = 1

    solve_started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            matrix, np.asarray(rows.lower), np.asarray(rows.upper)
        ),
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": float(mip_gap),
            "presolve": presolve,
            "disp": False,
        },
    )
    solve_elapsed = time.perf_counter() - solve_started
    total_elapsed = time.perf_counter() - started
    status_names = {
        0: "OPTIMAL",
        1: "TIME_LIMIT_OR_ITERATION_LIMIT",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "SOLVER_ERROR",
    }
    status = status_names.get(int(result.status), f"STATUS_{result.status}")

    routes = None
    loads = None
    objective_value = None
    if result.x is not None:
        routes = []
        for t in range(num_periods):
            period_routes: list[list[int]] = []
            for d in range(instance.num_demands):
                successors: dict[int, int] = {}
                for i, j in reachable_pairs[t]:
                    if result.x[x_index[d, t, i, j]] > 0.5:
                        successors[i] = j
                source = int(instance.demand_source[d])
                target = int(instance.demand_target[d])
                route = [source]
                visited = {source}
                while route[-1] != target:
                    if route[-1] not in successors:
                        raise RuntimeError("Solver incumbent does not contain an s-t path.")
                    next_node = successors[route[-1]]
                    if next_node in visited:
                        raise RuntimeError("Cycle encountered on extracted s-t path.")
                    route.append(next_node)
                    visited.add(next_node)
                period_routes.append(route)
            routes.append(period_routes)
        loads = [
            loads_for_routes(instance, period_data[t], routes[t], t)
            for t in range(num_periods)
        ]
        objective_value = max(float(np.max(item)) for item in loads)

    return ExactResult(
        routes=routes,
        objective=objective_value,
        dual_bound=(
            float(result.mip_dual_bound)
            if getattr(result, "mip_dual_bound", None) is not None
            else None
        ),
        gap=(
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        runtime=solve_elapsed,
        total_runtime=total_elapsed,
        node_count=(
            int(result.mip_node_count)
            if getattr(result, "mip_node_count", None) is not None
            else None
        ),
        status=status,
        message=str(result.message),
        num_variables=number_variables,
        num_constraints=len(rows.lower),
        loads=loads,
    )
