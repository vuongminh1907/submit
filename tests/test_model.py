from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from heuristic import solve_heuristic
from orange_data import PeriodData, load_instance, route_distance
from validate import validate_routes


ROOT = Path(__file__).resolve().parents[1]


class OrangeModelTests(unittest.TestCase):
    def test_route_distance_examples_from_subject(self) -> None:
        self.assertEqual(route_distance([0, 1, 2], [0, 2]), 3)
        self.assertEqual(route_distance([0, 1, 2, 3], [0, 1, 3]), 3)
        self.assertEqual(route_distance([0, 1, 2, 3], [0, 2, 1, 3]), 6)
        self.assertEqual(route_distance([0, 1, 3], [0, 2, 3]), 4)

    def test_split_coefficients_on_diamond(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            net = {
                "directed": True,
                "multigraph": False,
                "nodes": [{"id": i, "name": f"v{i}"} for i in range(4)],
                "links": [
                    {"id": 0, "from": 0, "to": 1, "metric": 1, "capacity": 10},
                    {"id": 1, "from": 0, "to": 2, "metric": 1, "capacity": 10},
                    {"id": 2, "from": 1, "to": 3, "metric": 1, "capacity": 10},
                    {"id": 3, "from": 2, "to": 3, "metric": 1, "capacity": 10},
                ],
            }
            tm = {
                "num_time_slots": 1,
                "demands": [{"s": 0, "t": 3, "v": [10]}],
            }
            scenario = {"max_segments": 4, "budget": [], "interventions": []}
            for suffix, content in (
                ("net", net),
                ("tm", tm),
                ("scenario", scenario),
            ):
                (base / f"setA-01-{suffix}.json").write_text(
                    json.dumps(content), encoding="utf-8"
                )
            instance = load_instance(base, 1)
            ratio = PeriodData(instance, 0).arc_ratios(0, 3)
            np.testing.assert_allclose(ratio, [0.5, 0.5, 0.5, 0.5])

    def test_heuristic_solution_is_independently_validated(self) -> None:
        instance = load_instance(ROOT / "data" / "setA", 1)
        result = solve_heuristic(
            instance,
            seed=7,
            max_iterations=5,
            candidate_count=5,
            time_limit=10,
        )
        validation = validate_routes(instance, result.routes)
        self.assertTrue(validation.feasible, validation.errors)
        self.assertLessEqual(validation.budget_used[1], instance.budget[1])
        self.assertAlmostEqual(validation.objective, result.objective, places=9)


if __name__ == "__main__":
    unittest.main()
