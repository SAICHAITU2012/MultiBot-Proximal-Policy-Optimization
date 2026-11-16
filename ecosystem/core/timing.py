"""Authoritative simulation and controller timing for the ecosystem world."""
from __future__ import annotations


SCENE_DT = 0.005
SUBSTEPS = 1
PHYSICS_HZ = round(1.0 / SCENE_DT)

# Runtime policy service rates. These may differ from the rates recorded for
# the upstream checkpoints; see docs/ASSET_INVENTORY.md and docs/M1_REPORT.md.
CONTROL_HZ = {
    "go2": 25,
    "drone": 200,
    "panda": 50,
    "g1": 50,
}


def _period_steps(control_hz: int) -> int:
    if control_hz <= 0 or PHYSICS_HZ % control_hz:
        raise ValueError(
            f"control rate {control_hz} Hz must be a positive integer divisor "
            f"of the {PHYSICS_HZ} Hz physics clock"
        )
    return PHYSICS_HZ // control_hz


CONTROL_PERIOD_STEPS = {
    name: _period_steps(rate) for name, rate in CONTROL_HZ.items()
}
