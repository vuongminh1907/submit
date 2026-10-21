"""Independent feasibility and objective checks for generated solutions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orange_data import (
    OrangeInstance,
    PeriodData,
    loads_for_routes,
    route_distance,
)


@dataclass
class ValidationResult:
    feasible: bool
    objective: float
    budget_used: dict[int, int]
    errors: list[str]


def validate_routes(
    instance: OrangeInstance,
    routes: list[list[list[int]]],
) -> ValidationResult:
    errors: list[str] = []
    budgets: dict[int, int] = {}
    loads: list[np.ndarray] = []
    if len(routes) > instance.num_periods:
        errors.append("More routing periods than traffic-matrix periods.")

    for period, period_routes in enumerate(routes):
        data = PeriodData(instance, period)
        if len(period_routes) != instance.num_demands:
            errors.append(
                f"t={period}: expected {instance.num_demands} routes, "
                f"received {len(period_routes)}."
            )
            continue
        for demand, route in enumerate(period_routes):
            source = int(instance.demand_source[demand])
            target = int(instance.demand_target[demand])
            if not route or route[0] != source or route[-1] != target:
                errors.append(f"d={demand},t={period}: wrong route terminals.")
                continue
            if len(route) - 1 > instance.max_segments:
                errors.append(f"d={demand},t={period}: maxSeg violated.")
            if len(route) != len(set(route)):
                errors.append(f"d={demand},t={period}: repeated node/cycle.")
            for i, j in zip(route, route[1:]):
                if not data.reachable(i, j):
                    errors.append(
                        f"d={demand},t={period}: unreachable segment {i}->{j}."
                    )
        try:
            loads.append(
                loads_for_routes(instance, data, period_routes, period)
            )
        except ValueError as exc:
            errors.append(str(exc))

        if period > 0:
            used = sum(
                route_distance(routes[period - 1][d], period_routes[d])
                for d in range(instance.num_demands)
            )
            budgets[period] = used
            if used > instance.budget[period]:
                errors.append(
                    f"t={period}: budget {used} > {instance.budget[period]}."
                )

    objective = max((float(np.max(load)) for load in loads), default=float("nan"))
    return ValidationResult(not errors, objective, budgets, errors)
