"""Regression checks for configurable ecosystem repository paths."""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ECOSYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECOSYSTEM_ROOT))

from core import paths  # noqa: E402
from robots import g1_stand  # noqa: E402


class EcosystemPathTests(unittest.TestCase):
    def test_defaults_are_derived_from_the_checkout_layout(self):
        workspace = paths.WORKSPACE_ROOT
        if (workspace / "team_lab").is_dir():
            expected_team_lab = workspace / "team_lab"
        elif (workspace / "experiments").is_dir() and (workspace / "team").is_dir():
            expected_team_lab = workspace
        else:
            expected_team_lab = workspace / "team_lab"

        self.assertEqual(paths.ECOSYSTEM_ROOT, ECOSYSTEM_ROOT)
        self.assertEqual(paths.TEAM_LAB_ROOT, expected_team_lab.resolve())
        self.assertEqual(paths.TEAM_LAB_EXPERIMENTS, paths.TEAM_LAB_ROOT / "experiments")
        self.assertEqual(paths.TEAM_LAB_LOGS, paths.TEAM_LAB_ROOT / "logs")
        self.assertEqual(
            g1_stand.CFG_DIR,
            paths.GENESIS_HUMANOID_ROOT / "logs" / "g1-stand-s2" / "20261111_213159",
        )
        self.assertEqual(
            g1_stand.URDF,
            paths.GENESIS_HUMANOID_ROOT / "assets" / "robot" / "unitree_g1"
            / "g1_29dof_simple_collision.urdf",
        )

    def test_workspace_and_repository_overrides_are_honored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            workspace = temp_root / "workspace"
            humanoid = temp_root / "custom-humanoid"
            world = temp_root / "custom-world"
            team_lab = temp_root / "custom-team-lab"
            env = {
                "GENESIS_WORKSPACE_ROOT": str(workspace),
                "GENESIS_HUMANOID_ROOT": str(humanoid),
                "GENESIS_WORLD_ROOT": str(world),
                "GENESIS_TEAM_LAB_ROOT": str(team_lab),
            }
            try:
                with patch.dict(os.environ, env):
                    importlib.reload(paths)
                    importlib.reload(g1_stand)
                    self.assertEqual(paths.WORKSPACE_ROOT, workspace.resolve())
                    self.assertEqual(paths.GENESIS_HUMANOID_ROOT, humanoid.resolve())
                    self.assertEqual(paths.GENESIS_WORLD_ROOT, world.resolve())
                    self.assertEqual(paths.TEAM_LAB_ROOT, team_lab.resolve())
                    self.assertEqual(g1_stand.CFG_DIR, paths.G1_STAND_RUN_DIR)
                    self.assertEqual(g1_stand.URDF, paths.G1_URDF)
            finally:
                importlib.reload(paths)
                importlib.reload(g1_stand)


if __name__ == "__main__":
    unittest.main()
