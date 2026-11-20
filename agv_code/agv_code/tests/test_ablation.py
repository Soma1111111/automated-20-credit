"""
Regression / identity tests for the ablation.

    pytest tests -q                       (from agv_code/agv_code)

1. A, B, C must reproduce the committed results_hybrid/comparison.csv exactly - plan quantities only
   (makespan, tardiness, imbalance, waiting, windows, Algorithm 3 calls, messages); the compute times
   (cpu_*, wall_ms) are machine-dependent and are not compared.
2. B2 (one centralized server, lobbies, FCFS leg order) must produce exactly the plans of C (central +
   floor servers) on every seed: the server split is schedule-neutral.
3. A2 (paper rule, FCFS order) must produce exactly the plans of A: FCFS order is vacuous under the
   paper's rule, where every AGV starts at pi at time 0 and may only wait there.
Every run is checked with the independent agv_core.verify().
"""
import csv
import os
from functools import lru_cache

import pytest

from agv_ablation import FIELDS, same_plans, run_one
from agv_core import Params

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results_hybrid", "comparison.csv")
PRM = Params(eta=16.0, eta_p=16.0, delta=10.0, lam=1.0)
NJOBS, NAGVS = 100, 20
SEEDS = list(range(1, 31))

# plan quantities: everything a plan / schedule depends on, not the measured compute time
QUANTITIES = ["makespan", "fleet_back", "tardiness", "earliness", "imbalance", "wait_total", "wait_max",
              "alg3_calls", "max_windows_one_server", "messages", "payload_windows"]


@lru_cache(maxsize=None)
def run(cfg, seed):
    row, detail = run_one(cfg, seed, NJOBS, NAGVS, PRM)
    return row, detail


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def test_committed_csv_columns_are_covered():
    """The fields this test checks are the fields the committed CSV has (guards against renaming)."""
    with open(CSV, newline="") as fh:
        committed = csv.DictReader(fh).fieldnames
    assert set(QUANTITIES + ["config", "seed", "jobs", "agvs", "ok"]) <= set(committed)
    assert {"windows_total", "n_legs", "alg3_fallbacks"} <= set(FIELDS)


@pytest.mark.parametrize("cfg", ["A", "B", "C"])
def test_abc_reproduce_committed_results(cfg):
    """Configs A, B, C must be unchanged: same numbers as results_hybrid/comparison.csv."""
    with open(CSV, newline="") as fh:
        ref = [r for r in csv.DictReader(fh) if r["config"] == cfg]
    assert ref, f"no rows for config {cfg} in {CSV}"
    for r in ref:
        seed = int(r["seed"])
        row, detail = run(cfg, seed)
        assert row["ok"], f"{cfg} seed {seed}: verification failed {detail['res']['violations'][:3]}"
        assert row["jobs"] == int(r["jobs"]) and row["agvs"] == int(r["agvs"])
        for key in QUANTITIES:
            assert close(row[key], r[key]), \
                f"{cfg} seed {seed}: {key} changed ({row[key]!r} != committed {r[key]!r})"


@pytest.mark.parametrize("seed", SEEDS)
def test_b2_plans_equal_c(seed):
    """One server with FCFS legs (B2) == central + floor servers (C): exact window equality."""
    rB2, dB2 = run("B2", seed)
    rC, dC = run("C", seed)
    assert rB2["ok"] and rC["ok"], f"seed {seed}: verification failed"
    assert same_plans(dB2["res"]["plans"], dC["res"]["plans"], tol=1e-6), \
        f"seed {seed}: B2 and C disagree - the server split is not schedule-neutral"
    for key in ("makespan", "tardiness", "imbalance", "wait_total", "wait_max"):
        assert close(rB2[key], rC[key], tol=1e-9), f"seed {seed}: {key} differs between B2 and C"


@pytest.mark.parametrize("seed", SEEDS)
def test_a2_equals_a(seed):
    """Paper rule + FCFS order (A2) == paper rule + priority order (A)."""
    rA, dA = run("A", seed)
    rA2, dA2 = run("A2", seed)
    assert rA2["ok"], f"seed {seed}: A2 verification failed"
    assert same_plans(dA["res"]["plans"], dA2["res"]["plans"], tol=1e-6), \
        f"seed {seed}: A2 differs from A"
    assert close(rA["makespan"], rA2["makespan"], tol=1e-9)


@pytest.mark.parametrize("cfg", ["A", "B", "B2", "C"])
def test_every_plan_passes_verify(cfg):
    """Every plan of every configuration passes the independent verifier (no node/track conflict,
    no pause on an edge, every job served exactly once)."""
    for seed in (1, 2, 3):
        row, detail = run(cfg, seed)
        assert not detail["res"]["violations"], f"{cfg} seed {seed}: {detail['res']['violations'][:3]}"
        assert row["ok"] is True
