"""Filesystem locations shared by the ecosystem runtime.

The usual checkout layout has ``ecosystem``, ``team_lab``, ``Genesis-Humanoid``
and ``genesis-world`` as siblings under one workspace directory.  The ecosystem
can also be run from a combined worktree where the team_lab files are at the
workspace root.  Set ``GENESIS_WORKSPACE_ROOT`` when the sibling repositories
live elsewhere; the individual ``GENESIS_TEAM_LAB_ROOT``,
``GENESIS_HUMANOID_ROOT`` and ``GENESIS_WORLD_ROOT`` variables can override a
single repository path.

Relative environment-variable values are resolved from the process working
directory.  No path is tied to a particular user's home directory.
"""
from __future__ import annotations

import os
from pathlib import Path


ECOSYSTEM_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _discover_workspace_root() -> Path:
    """Find the common workspace when the four repositories are siblings."""
    candidates = (ECOSYSTEM_ROOT.parent, *ECOSYSTEM_ROOT.parent.parents)
    for candidate in candidates:
        if all((candidate / name).is_dir()
               for name in ("team_lab", "Genesis-Humanoid", "genesis-world")):
            return candidate
    # In an isolated/combined worktree, ecosystem is nested directly in the
    # team_lab checkout; its parent is the most useful project root available.
    return ECOSYSTEM_ROOT.parent


WORKSPACE_ROOT = _path_from_env("GENESIS_WORKSPACE_ROOT", _discover_workspace_root())


def _default_team_lab_root() -> Path:
    sibling = WORKSPACE_ROOT / "team_lab"
    if sibling.is_dir():
        return sibling
    if (WORKSPACE_ROOT / "experiments").is_dir() and (WORKSPACE_ROOT / "team").is_dir():
        return WORKSPACE_ROOT
    return sibling


TEAM_LAB_ROOT = _path_from_env("GENESIS_TEAM_LAB_ROOT", _default_team_lab_root())
GENESIS_WORLD_ROOT = _path_from_env(
    "GENESIS_WORLD_ROOT", WORKSPACE_ROOT / "genesis-world"
)
GENESIS_HUMANOID_ROOT = _path_from_env(
    "GENESIS_HUMANOID_ROOT", WORKSPACE_ROOT / "Genesis-Humanoid"
)

TEAM_LAB_EXPERIMENTS = TEAM_LAB_ROOT / "experiments"
TEAM_LAB_LOGS = TEAM_LAB_ROOT / "logs"
G1_STAND_RUN_DIR = (
    GENESIS_HUMANOID_ROOT / "logs" / "g1-stand-s2" / "20261111_213159"
)
G1_URDF = (
    GENESIS_HUMANOID_ROOT / "assets" / "robot" / "unitree_g1"
    / "g1_29dof_simple_collision.urdf"
)
