#!/usr/bin/env python3
"""
machine_guard.py — thermal & load watchdog for training sessions on this MacBook (M4, 16 GB).

Why it exists:
  macOS protects the hardware by *throttling* before any damage can occur, but a
  training run that pins all cores for hours makes the machine unpleasant and slow.
  This guard gives us three things:
    1. A one-shot `check()` usable between RL iterations inside our training wrappers.
    2. A `wait_until_safe()` that PAUSES training (sleep) when load/thermal pressure
       crosses thresholds, instead of killing the run.
    3. A standalone `--watch` CSV logger so every session leaves an auditable
       health record (timestamp, cpu%, mem%, loadavg, throttle limit).

How it reads thermals (no sudo required):
  `pmset -g therm` prints CPU_Speed_Limit as a percentage. 100 = no throttling;
  <100 = macOS has already started throttling the CPU. We treat <95 as "back off".
  (Deep per-sensor temps need sudo powermetrics; we deliberately avoid that.)

Usage in training wrappers:
    from machine_guard import Guard
    guard = Guard(session_minutes=45, cpu_pause=92, throttle_floor=95)
    ...
    for it in range(n_iters):
        guard.wait_until_safe()          # sleeps/pauses if machine is stressed
        if guard.session_expired():      # hard session cap -> save & exit cleanly
            break
        ... train one iteration ...

CLI:
    python machine_guard.py --watch --interval 5 --csv logs/guard.csv
    python machine_guard.py --check
"""

import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:  # training env installs psutil; guard degrades gracefully
    psutil = None


def thermal_speed_limit() -> int:
    """CPU_Speed_Limit from `pmset -g therm` (100 = unthrottled). Returns 100 on failure."""
    try:
        out = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if "CPU_Speed_Limit" in line:
                return int(line.split("=")[1].strip())
    except Exception:
        pass
    return 100


def snapshot() -> dict:
    snap = {"time": datetime.now().isoformat(timespec="seconds")}
    if psutil is not None:
        snap["cpu_pct"] = psutil.cpu_percent(interval=0.3)
        snap["mem_pct"] = psutil.virtual_memory().percent
        load1, load5, load15 = psutil.getloadavg()
        ncpu = psutil.cpu_count(logical=True) or 1
        snap["load1_per_core"] = round(load1 / ncpu, 2)
    else:
        snap["cpu_pct"], snap["mem_pct"], snap["load1_per_core"] = -1, -1, -1
    snap["cpu_speed_limit"] = thermal_speed_limit()
    return snap


def is_stressed(snap: dict, cpu_pause: float, throttle_floor: int, mem_pause: float) -> tuple[bool, str]:
    if snap["cpu_speed_limit"] < throttle_floor:
        return True, f"thermal throttle active (CPU_Speed_Limit={snap['cpu_speed_limit']}%)"
    if snap["cpu_pct"] > cpu_pause:
        return True, f"cpu {snap['cpu_pct']}% > {cpu_pause}%"
    if psutil is not None and snap["mem_pct"] > mem_pause:
        return True, f"memory {snap['mem_pct']}% > {mem_pause}% (16 GB limit — risk of swap)"
    return False, "ok"


class Guard:
    """Use inside training loops: wait_until_safe() before each iteration."""

    def __init__(
        self,
        session_minutes: float = 45.0,
        cpu_pause: float = 92.0,
        throttle_floor: int = 95,
        mem_pause: float = 88.0,
        pause_seconds: float = 45.0,
        max_pause_total: float = 900.0,
        log_csv: str | None = None,
        verbose: bool = True,
    ):
        self.t_start = time.time()
        self.session_seconds = session_minutes * 60
        self.cpu_pause, self.throttle_floor, self.mem_pause = cpu_pause, throttle_floor, mem_pause
        self.pause_seconds, self.max_pause_total = pause_seconds, max_pause_total
        self.total_paused = 0.0
        self.n_pauses = 0
        self.log_csv, self.verbose = log_csv, verbose
        self._csv_file = None
        self._csv_writer = None
        if log_csv:
            Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(log_csv, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=["time", "cpu_pct", "mem_pct", "load1_per_core", "cpu_speed_limit", "event"]
            )
            self._csv_writer.writeheader()

    def _log(self, snap: dict, event: str):
        if self._csv_writer:
            row = dict(snap)
            row["event"] = event
            self._csv_writer.writerow(row)
            self._csv_file.flush()

    def session_expired(self) -> bool:
        """Hard wall-clock cap so sessions end cleanly with a checkpoint."""
        return (time.time() - self.t_start) >= self.session_seconds

    def wait_until_safe(self, max_extra_pause: float | None = None):
        """Pause training while the machine is stressed. Never waits more than
        max_extra_pause total in a single call (default max_pause_total)."""
        budget = max_extra_pause if max_extra_pause is not None else self.max_pause_total
        waited = 0.0
        while True:
            snap = snapshot()
            stressed, reason = is_stressed(snap, self.cpu_pause, self.throttle_floor, self.mem_pause)
            if not stressed or waited >= budget:
                if waited > 0:
                    self.total_paused += waited
                    self._log(snap, f"resumed_after_{waited:.0f}s_pause")
                    if self.verbose:
                        print(f"[guard] resumed after {waited:.0f}s of cooldown (was: {reason})")
                else:
                    self._log(snap, "ok")
                return snap
            if self.verbose and (self.n_pauses == 0 or waited == 0):
                print(f"[guard] pausing training: {reason}")
            self.n_pauses += 1
            self._log(snap, f"pause:{reason}")
            time.sleep(min(self.pause_seconds, budget - waited))
            waited += self.pause_seconds

    def close(self):
        if self._csv_file:
            snap = snapshot()
            self._log(snap, "session_end")
            self._csv_file.close()
        if self.verbose:
            print(
                f"[guard] session summary: {self.n_pauses} pauses, "
                f"{self.total_paused:.0f}s total cooldown"
            )


def main():
    ap = argparse.ArgumentParser(description="Thermal/load watchdog")
    ap.add_argument("--watch", action="store_true", help="log continuously")
    ap.add_argument("--check", action="store_true", help="one-shot status")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--csv", default="logs/guard.csv")
    args = ap.parse_args()

    if args.check or not args.watch:
        s = snapshot()
        stressed, reason = is_stressed(s, 92.0, 95, 88.0)
        print(s)
        print("STATUS:", "STRESSED — " + reason if stressed else "OK")
        return

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    print(f"[guard] watching every {args.interval}s -> {args.csv}  (Ctrl-C to stop)")
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "cpu_pct", "mem_pct", "load1_per_core", "cpu_speed_limit", "event"])
        w.writeheader()
        try:
            while True:
                s = snapshot()
                stressed, reason = is_stressed(s, 92.0, 95, 88.0)
                row = dict(s)
                row["event"] = f"{'pause:' + reason if stressed else 'ok'}"
                w.writerow(row)
                f.flush()
                if stressed:
                    print(f"[guard] {s['time']} STRESSED: {reason}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("[guard] stopped")


if __name__ == "__main__":
    main()
