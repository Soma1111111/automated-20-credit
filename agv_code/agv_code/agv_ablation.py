"""
agv_ablation.py - separate the three effects that A / B / C mixes
=================================================================
agv_compare.py compares three configurations in which two changes happen at once.  This script adds the
missing baselines so that every step of the chain changes exactly ONE thing:

  A   centralized, paper's rule (an AGV waits only at the parking station pi)            [agv_core]
  B   centralized, AGVs may also wait in the lift lobbies                                 [agv_core]
  B2  same as B, but the per-floor legs are planned in the order the AGVs are released
      (earliest release first, ties -> AGV priority) instead of strictly by priority       [agv_central]
  C   hybrid: one central server + one floor server per floor, FCFS leg order            [agv_hybrid]
  A2  paper's rule + FCFS order; only meaningful if FCFS ordering exists under that rule - see below

Steps (each one changes a single factor):
  A  -> B   lobbies      : where an AGV may wait
  B  -> B2  ordering     : priority-ordered routes   ->   release(FCFS)-ordered legs
  B2 -> C   architecture : one global window store   ->   central + 3 floor stores

Why A2 collapses to A
---------------------
"FCFS leg order" needs a leg boundary to be observable: an AGV is served in the order it REACHES a
server.  Under the paper's rule the only place an AGV may wait is pi, and every AGV starts at pi at
time 0, so (a) the first leg of every AGV is released at time 0 and the FCFS key (release, priority)
is identical for the whole fleet -> the order is the priority order, and (b) a later leg would have to
start at a lift landing where waiting is forbidden (Algorithm 3 could only slide that leg back to pi,
i.e. across floors, which destroys the leg decomposition and assumption 5 "no pause on an edge").
So A2 is not a different configuration: it is exactly A, and this script runs it (split=False) only to
show that the plans are identical.  The ordering effect under the paper's rule is therefore zero by
construction, not an empirical finding.

Outputs (default: results_ablation/)
  comparison.csv   one row per (config, seed)
  summary.txt      mean / std per config, paired difference per step, and one sentence per effect
  fig5_ablation_steps.png   the four configurations side by side
  fig6_effect_per_step.png  mean paired change of each step

Run:
    python agv_ablation.py                       # seeds 1-30, 100 jobs, 20 AGVs
    python agv_ablation.py --seeds 1-5 --jobs 40
"""
import argparse
import csv
import os
import statistics as stats
import time

from agv_core import AGV, Log, Params, metrics, run_pipeline
from agv_central import run_unified
from agv_hybrid import run_hybrid
from agv_layouts import build_three_floors
from agv_multifloor_demo import make_jobs

CONFIGS = [("A", "Centralized (paper rule)"),
           ("B", "Centralized + lobbies"),
           ("B2", "Centralized + lobbies + FCFS legs"),
           ("C", "Hybrid: central + 3 floor servers")]
STEPS = [("A", "B", "lobbies"), ("B", "B2", "FCFS leg order"), ("B2", "C", "server split")]
COLORS = {"A": "#2a78d6", "B": "#eb6834", "B2": "#7a5af8", "C": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

METRICS = [  # key, label, unit, lower_is_better
    ("makespan", "Last delivery (makespan)", "s", True),
    ("tardiness", "Total tardiness", "s", True),
    ("imbalance", "Load imbalance (max - min usage)", "s", True),
    ("wait_total", "Total waiting (parking + lobbies)", "s", True),
    ("wait_max", "Longest wait of one AGV", "s", True),
    ("max_windows_one_server", "Time windows held by one server", "", True),
    ("messages", "Central <-> floor messages", "", True),
    ("cpu_ms", "Planning compute time, busiest server", "ms", True),
]
FIELDS = ["config", "seed", "jobs", "agvs", "ok", "makespan", "fleet_back", "tardiness", "earliness",
          "imbalance", "wait_total", "wait_max", "alg3_calls", "alg3_fallbacks", "cpu_total_ms", "cpu_ms",
          "wall_ms", "max_windows_one_server", "windows_total", "messages", "payload_windows", "n_legs"]
# fields of agv_compare.results_hybrid/comparison.csv, plus windows_total / n_legs / alg3_fallbacks


def make_layout(cfg):
    L = build_three_floors(lobbies=True)                 # the same graph for every configuration
    if cfg in ("B", "B2", "C"):                          # A and A2: the paper's rule, pi only
        L.exempt = {L.pi} | L.lobbies
    return L


def count_windows(L, plans):
    """All node windows (non-exempt) + all track windows of a plan set = what one server would hold."""
    return sum(sum(1 for n in pl.R if n not in L.exempt) + len(pl.R) - 1 for pl in plans.values())


def run_one(cfg, seed, n_jobs, n_agvs, prm):
    L = make_layout(cfg)
    jobs = make_jobs(L, prm, n_jobs, seed)
    agvs = [AGV(f"AGV-{k:02d}", k) for k in range(1, n_agvs + 1)]
    log = Log()
    tic = time.perf_counter()
    if cfg in ("A", "B"):
        res = run_pipeline(L, jobs, agvs, prm, log)
    elif cfg == "B2":
        res = run_unified(L, jobs, agvs, prm, log, order="fcfs", split=True)
    elif cfg == "A2":
        res = run_unified(L, jobs, agvs, prm, log, order="fcfs", split=False)
    elif cfg == "C":
        res = run_hybrid(L, jobs, agvs, prm, log, order="fcfs")
    else:
        raise ValueError(cfg)
    wall = time.perf_counter() - tic
    m = metrics(L, res["plans"], agvs, jobs, res["estimate"], res["calls"], prm)
    servers = res["servers"] if cfg == "C" else \
        {"central": dict(cpu=res["t_alg1"] + res["t_alg2"], windows=count_windows(L, res["plans"]))}
    row = dict(config=cfg, seed=seed, jobs=n_jobs, agvs=n_agvs, ok=not res["violations"],
               makespan=m["makespan_last_delivery"], fleet_back=m["fleet_back_at_parking"],
               tardiness=m["tardiness_final"], earliness=m["earliness_final"], imbalance=m["load_imbalance"],
               wait_total=sum(m["waiting_at_parking"].values()), wait_max=max(m["waiting_at_parking"].values()),
               alg3_calls=m["alg3_calls"], alg3_fallbacks=m["alg3_fallbacks"],
               cpu_total_ms=1000 * sum(s["cpu"] for s in servers.values()),
               cpu_ms=1000 * max(s["cpu"] for s in servers.values()), wall_ms=1000 * wall,
               max_windows_one_server=max(s["windows"] for s in servers.values()),
               windows_total=count_windows(L, res["plans"]),
               messages=res.get("messages", 0), payload_windows=res.get("payload", 0),
               n_legs=res.get("n_legs", 0))
    return row, dict(L=L, jobs=jobs, agvs=agvs, res=res, m=m, servers=servers)


def same_plans(p1, p2, tol=1e-6):
    return all(p1[k].R == p2[k].R and all(abs(x - y) <= tol for w1, w2 in zip(p1[k].P, p2[k].P)
                                          for x, y in zip(w1, w2)) for k in p1)


def parse_seeds(s):
    if "-" in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",")]


# ---------------------------------------------------------------------------------------- statistics
def fmt(v, unit=""):
    if unit == "ms":
        return f"{v:,.1f}"
    return f"{v:,.1f}" if abs(v) < 100 else f"{v:,.0f}"


def paired(rows, cfg_x, cfg_y, key):
    """Per-seed difference y - x (same seed = same job set => paired samples)."""
    xs = {r["seed"]: r[key] for r in rows if r["config"] == cfg_x}
    ys = {r["seed"]: r[key] for r in rows if r["config"] == cfg_y}
    return [ys[s] - xs[s] for s in sorted(xs)]


def better_worse_equal(diffs, lower_is_better=True):
    b = sum(1 for d in diffs if (d < -1e-9 if lower_is_better else d > 1e-9))
    w = sum(1 for d in diffs if (d > 1e-9 if lower_is_better else d < -1e-9))
    return b, sum(1 for d in diffs if abs(d) <= 1e-9), w


# ------------------------------------------------------------------------------------------- figures
def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_configs(rows, path, n_seeds):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.8))
    x = range(len(CONFIGS))
    for ax, (key, label, unit, _) in zip(axes.flat, METRICS):
        means = [stats.mean(r[key] for r in rows if r["config"] == c) for c, _ in CONFIGS]
        lo = [min(r[key] for r in rows if r["config"] == c) for c, _ in CONFIGS]
        hi = [max(r[key] for r in rows if r["config"] == c) for c, _ in CONFIGS]
        ax.bar(x, means, width=0.62, color=[COLORS[c] for c, _ in CONFIGS], edgecolor="white", linewidth=2)
        ax.errorbar(x, means, yerr=[[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=4)
        for xi, mv, h in zip(x, means, hi):
            ax.text(xi, h * 1.02 + 1e-9, fmt(mv), ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(list(x))
        ax.set_xticklabels([c for c, _ in CONFIGS], color=INK2)
        ax.set_title(f"{label} ({unit})".replace(" ()", ""), fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, max(hi) * 1.20 + 1e-9)
        _style(ax)
    fig.legend(handles=[Patch(color=COLORS[c], label=f"{c}  {n}") for c, n in CONFIGS],
               loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"Effect of each step: mean over {n_seeds} random job sets (bar), min-max (whisker); "
                 "A -> B lobbies, B -> B2 FCFS legs, B2 -> C server split", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_steps(rows, path, n_seeds):
    import matplotlib.pyplot as plt
    keys = [m for m in METRICS if m[0] in ("makespan", "tardiness", "imbalance", "wait_total")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (key, label, unit, lower) in zip(axes, keys):
        deltas, colors = [], []
        for cfg_x, cfg_y, _ in STEPS:
            deltas.append(stats.mean(paired(rows, cfg_x, cfg_y, key)))
            colors.append(COLORS[cfg_y])
        ax.bar(range(len(STEPS)), deltas, width=0.6, color=colors, edgecolor="white", linewidth=2)
        span = max(max(abs(d) for d in deltas), 1e-9)
        for xi, d in enumerate(deltas):
            dy = 0.03 * span
            if d < -1e-9:                                     # label under the bar
                ax.text(xi, d - dy, f"{d:+,.0f}", ha="center", va="top", fontsize=9, color=INK)
            elif d > 1e-9:
                ax.text(xi, d + dy, f"{d:+,.0f}", ha="center", va="bottom", fontsize=9, color=INK)
            else:                                            # a zero step: label under the zero line
                ax.text(xi, -dy, "+0  (plans identical)", ha="center", va="top", fontsize=8.5, color=INK2)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_ylim(min(min(deltas), 0) - 0.16 * span, 0.05 * span)
        ax.set_xticks(range(len(STEPS)))
        ax.set_xticklabels([f"{x}->{y}\n{s}" for x, y, s in STEPS], fontsize=8, color=INK2)
        ax.set_title(f"{label} ({unit})".replace(" ()", ""), fontsize=10, color=INK, loc="left")
        _style(ax)
    fig.suptitle(f"Mean paired change of one step (negative = better, {n_seeds} seeds)", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-30")
    ap.add_argument("--jobs", type=int, default=100)
    ap.add_argument("--agvs", type=int, default=20)
    ap.add_argument("--outdir", default="results_ablation")
    ap.add_argument("--skip-a2", action="store_true", help="skip the A2 == A demonstration")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(a.outdir, exist_ok=True)
    out = lambda f: os.path.join(a.outdir, f)
    prm = Params(eta=16.0, eta_p=16.0, delta=10.0, lam=1.0)
    seeds = parse_seeds(a.seeds)

    # ---- 1) run every configuration once per seed, verify every plan, check the two identities --
    plan_set = ("A", "A2", "B", "B2", "C") if not a.skip_a2 else ("A", "B", "B2", "C")
    rows, by = [], {}
    tic = time.perf_counter()
    for s in seeds:
        for c in plan_set:
            r, d = run_one(c, s, a.jobs, a.agvs, prm)
            assert r["ok"], f"verification failed: config {c}, seed {s}: {d['res']['violations'][:3]}"
            by[(c, s)] = d
            if c != "A2":                                     # A2 is a check, not a configuration
                rows.append(r)
        # A2 is not a separate configuration: under the paper's rule FCFS order gives A itself
        if not a.skip_a2:
            assert same_plans(by[("A", s)]["res"]["plans"], by[("A2", s)]["res"]["plans"]), \
                f"A2 (paper rule + FCFS order) differs from A (seed {s})"
        # the server split alone does not change the plans
        assert same_plans(by[("B2", s)]["res"]["plans"], by[("C", s)]["res"]["plans"]), \
            f"B2 != C on seed {s}: the server split changed the plans"
        print(f"  seed {s}: A {by[('A', s)]['m']['makespan_last_delivery']:8.0f}  "
              f"B {by[('B', s)]['m']['makespan_last_delivery']:8.0f}  "
              f"B2 {by[('B2', s)]['m']['makespan_last_delivery']:8.0f}  "
              f"C {by[('C', s)]['m']['makespan_last_delivery']:8.0f}   (makespan, s)")
    wall = time.perf_counter() - tic
    if not a.skip_a2:
        print(f"check: A2 (paper rule, FCFS order) == A on {len(seeds)} seeds  "
              f"-> FCFS order is vacuous under the paper's rule (see the module docstring)")
    print(f"check: B2 (one server, FCFS legs) == C (central + floor servers) on {len(seeds)} seeds  OK")
    details = {c: by[(c, seeds[0])] for c, _ in CONFIGS}
    with open(out("comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- 4) mean / std per configuration, paired difference per step -------------------------
    txt = [f"ABLATION: which part of the gain comes from what  -  {a.jobs} jobs, {a.agvs} AGVs, "
           f"seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} job sets)",
           "all plans verified conflict-free by agv_core.verify(); "
           "B2 (one server, FCFS legs) == C (central + floor servers) on every seed; "
           "A2 (paper rule, FCFS order) == A on every seed",
           "",
           "  " + "\n  ".join(f"{c:<3}{name}" for c, name in CONFIGS),
           "",
           "steps:   A->B  lobbies      (where an AGV may wait)",
           "         B->B2 FCFS order   (priority-ordered routes -> release-ordered legs)",
           "         B2->C server split (one global window store -> central + 3 floor stores)",
           "",
           f"{'metric (mean over seeds)':<42}" + "".join(f"{c:>12}" for c, _ in CONFIGS)
           + "".join(f"{x + '->' + y:>16}" for x, y, _ in STEPS)]
    for key, label, unit, lower in METRICS:
        cells = "".join(f"{fmt(stats.mean(r[key] for r in rows if r['config'] == c)):>12}"
                        for c, _ in CONFIGS)
        dcell = "".join(f"{stats.mean(paired(rows, x, y, key)):>+16,.1f}" for x, y, _ in STEPS)
        txt.append(f"{label + (f' ({unit})' if unit else ''):<42}" + cells + dcell)

    txt += ["", "paired difference per step: mean +- std over seeds, wins / ties / losses "
                "(win = the step lowered the metric)", "",
            f"{'metric':<34}" + "".join(f"{x + '->' + y + ' ' + s:>28}" for x, y, s in STEPS)]
    for key, label, unit, lower in METRICS:
        cells = ""
        for x, y, _ in STEPS:
            d = paired(rows, x, y, key)
            b, e, w = better_worse_equal(d, lower)
            cells += f"{stats.mean(d):+12,.1f} +- {stats.stdev(d) if len(d) > 1 else 0:8,.1f} {b:2d}/{e:2d}/{w:2d}"
        txt.append(f"{label + (f' ({unit})' if unit else ''):<34}" + cells)

    # ---- 5) the one-sentence verdict, computed from the numbers above ---------------------
    def step_share(cfg_x, cfg_y, key):
        total = stats.mean(paired(rows, CONFIGS[0][0], CONFIGS[-1][0], key))
        part = stats.mean(paired(rows, cfg_x, cfg_y, key))
        return 100.0 * part / total if abs(total) > 1e-9 else float("nan")

    ms_a, ms_b, ms_b2, ms_c = [stats.mean(r["makespan"] for r in rows if r["config"] == c)
                               for c, _ in CONFIGS]
    txt += ["", "VERDICT"]
    txt += [f"  Makespan {ms_a:,.0f} s (A) -> {ms_b:,.0f} s (B) -> {ms_b2:,.0f} s (B2) -> {ms_c:,.0f} s (C):"]
    for key in ("makespan", "tardiness", "imbalance"):
        shares = [step_share(x, y, key) for x, y, _ in STEPS]
        txt.append(f"    {key:<12} share of the A->C gain:  lobbies {shares[0]:5.1f}%   "
                   f"FCFS order {shares[1]:5.1f}%   server split {shares[2]:5.1f}%")
    sh = {key: [step_share(x, y, key) for x, y, _ in STEPS] for key in ("makespan", "tardiness", "imbalance")}
    same_sign = all(abs(v[2]) < 1e-9 for v in sh.values())
    txt += ["",
            "  ONE SENTENCE: "
            + f"the lobbies (A->B) account for {sh['makespan'][0]:.0f}% of the makespan gain, "
              f"{sh['tardiness'][0]:.0f}% of the tardiness gain and {sh['imbalance'][0]:.0f}% of the "
              f"imbalance gain; "
            + f"the FCFS leg order (B->B2) accounts for {sh['makespan'][1]:.0f}%, "
              f"{sh['tardiness'][1]:.0f}% and {sh['imbalance'][1]:.0f}%; "
            + ("the server split (B2->C) accounts for none of it - it changes the plans on no seed and "
               "leaves every schedule metric unchanged (0 s), paying only in central<->floor messages."
               if same_sign else
               "the server split (B2->C) is not schedule-neutral on these seeds - see the table above.")]
    txt += ["", "compute times are wall-clock measurements of one sequential process on one machine; they "
                "compare implementations (a leg-based planner pays for its orchestration), not "
                "asymptotic cost, and no parallelism or network latency is modelled",
            f"total run time {wall:.1f} s ({len(rows)} runs, sequential)"]
    open(out("summary.txt"), "w").write("\n".join(txt) + "\n")
    print("\n" + "\n".join(txt))

    fig_configs(rows, out("fig5_ablation_steps.png"), len(seeds))
    fig_steps(rows, out("fig6_effect_per_step.png"), len(seeds))
    print(f"\nfiles written to {os.path.abspath(a.outdir)}")


if __name__ == "__main__":
    main()
