"""Regression checks for the centralized INF-02 timing configuration."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.timing import (
    CONTROL_HZ,
    CONTROL_PERIOD_STEPS,
    PHYSICS_HZ,
    SCENE_DT,
    SUBSTEPS,
)
from robots import g1_stand


class TimingConfigurationTests(unittest.TestCase):
    def test_physics_clock_matches_the_validated_scene(self):
        self.assertEqual(SCENE_DT, 0.005)
        self.assertEqual(SUBSTEPS, 1)
        self.assertEqual(PHYSICS_HZ, 200)
        self.assertEqual(1.0 / SCENE_DT, PHYSICS_HZ)

    def test_runtime_policy_rates_are_preserved(self):
        self.assertEqual(
            CONTROL_HZ,
            {"go2": 25, "drone": 200, "panda": 50, "g1": 50},
        )

    def test_policy_periods_are_derived_in_physics_ticks(self):
        self.assertEqual(
            CONTROL_PERIOD_STEPS,
            {"go2": 8, "drone": 1, "panda": 4, "g1": 4},
        )
        for robot, period in CONTROL_PERIOD_STEPS.items():
            self.assertEqual(PHYSICS_HZ // period, CONTROL_HZ[robot])
            self.assertAlmostEqual(SCENE_DT * period, 1.0 / CONTROL_HZ[robot])

    def test_g1_wrapper_uses_the_central_runtime_timing(self):
        self.assertEqual(g1_stand.SIM_DT, SCENE_DT)
        self.assertEqual(g1_stand.DECIMATION, CONTROL_PERIOD_STEPS["g1"])
        self.assertEqual(g1_stand.CONTROL_HZ, CONTROL_HZ["g1"])
        self.assertEqual(g1_stand.CONTROL_DT, SCENE_DT * CONTROL_PERIOD_STEPS["g1"])


if __name__ == "__main__":
    unittest.main()
