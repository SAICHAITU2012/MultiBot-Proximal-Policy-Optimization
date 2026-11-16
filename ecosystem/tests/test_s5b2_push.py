#!/usr/bin/env python3
"""Run the separate fixed-seed S5b.2 feasible push experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.s5b2_feasible_push import main


if __name__ == "__main__":
    raise SystemExit(main())
