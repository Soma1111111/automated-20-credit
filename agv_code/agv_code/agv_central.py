"""
agv_central.py - ONE centralized server with a choice of planning ORDER
=======================================================================
The paper's planner (agv_core.run_pipeline -> algorithm_2) plans whole routes one AGV after another,
always in priority order V_1, V_2, ...  The hybrid planner (agv_hybrid.run_hybrid) plans per-floor
LEGS in first-come-first-served order.  Config C therefore mixes two changes at once:
  (i)  the server split (one central + 3 floor servers),
  (ii) the planning order (FCFS legs instead of priority-ordered routes).

This module isolates (ii).  It is a plain centralized planner - one server, one global W / W' store,
no floor servers at all - that can plan in either order:

  order="priority" : all legs of V_1, then all legs of V_2, ...   (must reproduce config B exactly)
  order="fcfs"     : legs are planned in the order their AGV is released (earliest release first,
                     ties -> AGV priority).  This is the missing baseline B2.

The planner also owns the lifts (as run_hybrid's central server does): each lift trip is booked so
that two AGVs are never on one lift track within delta, and the AGV waits in the departure lobby.

split=False gives the paper's own model back: one leg per AGV (its whole route, pi -> pi), lift
conflicts handled by the edge windows W' of Algorithm 2 rather than by a booking table.  With
exempt = {pi} and order="fcfs" that is config A2 - see agv_ablation.py for why it collapses to A.

Requires the layout built with lobbies=True; with split=True the lobbies must also be in L.exempt.
"""
from __future__ import annotations

import heapq
import time
from collections import defaultdict

from agv_core import Log, Params, Plan, algorithm_1, f, freeze, plan_route, verify
from agv_hybrid import split_legs


class _OthersOnly:
    """W_n seen by one AGV: its own (earlier) legs are not obstacles for itself."""

    def __init__(self, W, agv_id):
        self.W, self.agv_id = W, agv_id

    def get(self, n, default=()):
        return [w for w in self.W.get(n, default) if w[2] != self.agv_id]


class CentralPlanner:
    """ONE server for the whole factory: W_n and W'_e of every node and track, plus the lift bookings."""

    def __init__(self, L, prm: Params, log: Log):
        self.L, self.prm, self.log = L, prm, log
        self.W, self.Wp = defaultdict(list), defaultdict(list)
        self.lift = defaultdict(list)                      # lift track -> [(depart, arrive, agv)]
        self.calls, self.cpu, self.legs = [], 0.0, 0
        self.t_free = 0.0                                  # end of the latest frozen window

    # -- one leg of one AGV, planned against the whole store --------------------------------
    def plan_leg(self, agv_id, R, dwell, t0):
        tic = time.perf_counter()
        assert R[0] in self.L.exempt, "a leg must start at pi or at a lobby (a place where AGVs may wait)"
        W = _OthersOnly(self.W, agv_id)
        Wtrack = lambda u, v: [w for w in self.Wp[(u, v)] + self.Wp[(v, u)] if w[2] != agv_id]
        P = plan_route(self.L, agv_id, R, dwell, t0, W, Wtrack, self.prm, self.log, self.calls, self.t_free)
        freeze(self.L, agv_id, R, P, self.W, self.Wp)
        self.t_free = max(self.t_free, P[-1][1])
        self.legs += 1
        self.cpu += time.perf_counter() - tic
        return P

    # -- the lifts ---------------------------------------------------------------------------
    def book_lift(self, agv_id, u, v, ready):
        """Earliest departure >= ready at which lift track u-v is free (padded by delta)."""
        dur = self.L.delta_edge(u, v)
        key = frozenset((u, v))
        others = [w for w in self.lift[key] if w[2] != agv_id]
        st, _ = f(ready, ready + dur, others, self.prm.delta)
        dep = ready + st
        self.lift[key].append((dep, dep + dur, agv_id))
        if st > 0:
            self.log.add("LIFT", f"{agv_id}: lift {u}->{v} busy, waits {st:.1f} s in lobby {u}",
                         depart=round(dep, 1))
        return dep

    @property
    def windows_held(self):
        return sum(map(len, self.W.values())) + sum(map(len, self.Wp.values())) + \
               sum(map(len, self.lift.values()))


def run_unified(L, jobs, agvs, prm: Params, log: Log, order="fcfs", split=True):
    """Same inputs / outputs as agv_core.run_pipeline and agv_hybrid.run_hybrid, one server.

    order="fcfs"     legs in (release, priority) order     -> config B2 (with lobbies) / A2 (without)
    order="priority" legs in (priority, index) order       -> must equal config B when split=True
    split=True       routes are cut into per-floor legs at the lift links and the lifts are booked
    split=False      one leg per AGV, lift conflicts handled by the edge windows (paper's model)
    """
    if split:
        assert L.lobbies and L.lobbies <= L.exempt, \
            "split=True needs lobbies in Layout.exempt (build with lobbies=True and make them exempt)"
    srv = CentralPlanner(L, prm, log)

    tic = time.perf_counter()
    estimate = algorithm_1(L, jobs, agvs, prm, log)              # Algorithm 1 - unchanged
    srv.cpu += time.perf_counter() - tic
    t_alg1 = srv.cpu

    tic = time.perf_counter()
    legs = {a.i: (split_legs(L, a.R) if split else [(0, len(a.R) - 1)]) for a in agvs}
    P_all = {a.i: [None] * len(a.R) for a in agvs}
    by_id = {a.i: a for a in agvs}
    heap = []

    def push(a, k, release):
        key = (release, a.priority, k) if order == "fcfs" else (a.priority, k, release)
        heapq.heappush(heap, (key, a.i, k, release))

    for a in agvs:                                               # every AGV leaves pi at time 0
        push(a, 0, 0.0)

    while heap:
        _, aid, k, release = heapq.heappop(heap)
        a, (s, e) = by_id[aid], legs[aid][k]
        R_leg, dwell = a.R[s:e + 1], a.dwell[s:e + 1]
        P_leg = srv.plan_leg(aid, R_leg, dwell, release)          # Algorithms 2 + 3 on the whole factory
        P_all[aid][s:e + 1] = P_leg
        if k + 1 < len(legs[aid]):                                # next leg starts after the lift trip
            u, v = a.R[e], a.R[e + 1]
            dep = srv.book_lift(aid, u, v, ready=P_leg[-1][1])
            P_all[aid][e][1] = dep                               # wait in the departure lobby
            push(a, k + 1, dep + L.delta_edge(u, v))
    srv.cpu += time.perf_counter() - tic

    plans = {a.i: Plan(a.i, list(a.R), P_all[a.i], list(a.dwell), list(a.events)) for a in agvs}
    for a in agvs:
        log.add("UNI", f"{a.i} (priority {a.priority}) planned", end=round(plans[a.i].end_time, 1),
                legs=len(legs[a.i]), jobs=len(a.jobs))
    violations = verify(L, plans, agvs, jobs, prm)
    servers = {"central": dict(cpu=srv.cpu, windows=srv.windows_held)}
    return dict(estimate=estimate, plans=plans, calls=srv.calls, violations=violations,
                t_alg1=t_alg1, t_alg2=srv.cpu - t_alg1,
                servers=servers, messages=0, payload=0,
                n_legs=sum(len(v) for v in legs.values()))


if __name__ == "__main__":
    import os
    import sys

    from agv_core import AGV
    from agv_hybrid import run_hybrid
    from agv_layouts import build_three_floors
    from agv_multifloor_demo import make_jobs

    seed, n_jobs, n_agvs = (int(sys.argv[1]) if len(sys.argv) > 1 else 1), 100, 20
    prm = Params(eta=16.0, eta_p=16.0, delta=10.0, lam=1.0)

    def prep(cfg):
        L = build_three_floors(lobbies=True)
        if cfg != "A":
            L.exempt = {L.pi} | L.lobbies
        jobs = make_jobs(L, prm, n_jobs, seed)
        return L, jobs, [AGV(f"AGV-{k:02d}", k) for k in range(1, n_agvs + 1)]

    same = lambda p, q, tol=1e-6: all(
        p[k].R == q[k].R and all(abs(x - y) <= tol for w1, w2 in zip(p[k].P, q[k].P) for x, y in zip(w1, w2))
        for k in p)

    L, jobs, agvs = prep("B")
    b2 = run_unified(L, jobs, agvs, prm, Log(), order="fcfs")
    c = run_hybrid(L, jobs, agvs, prm, Log(), order="fcfs")
    b = run_unified(L, jobs, agvs, prm, Log(), order="priority")
    print(f"seed {seed}: verify B2={b2['violations'] or 'ok'}  C={c['violations'] or 'ok'}")
    print(f"  B2 == C          : {same(b2['plans'], c['plans'])}")
    print(f"  unified(prio) == C? {same(b['plans'], c['plans'])}   (expected False: different order)")
    print(f"  B2 last delivery : {max(p.P[-1][1] for p in b2['plans'].values()):.1f} s"
          f"   C: {max(p.P[-1][1] for p in c['plans'].values()):.1f} s")
    print(f"  B2 windows held  : {b2['servers']['central']['windows']:,}   "
          f"C max per server: {max(s['windows'] for s in c['servers'].values()):,}")
    assert same(b2["plans"], c["plans"]), "B2 must equal C"
    assert not b2["violations"] and not c["violations"]
    print(os.path.basename(__file__), "OK")
