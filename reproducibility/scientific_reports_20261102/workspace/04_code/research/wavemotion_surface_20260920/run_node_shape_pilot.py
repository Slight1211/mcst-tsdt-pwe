"""Audited low-order epigraph-SLSQP pilot; NOT manuscript-quality N15 evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "msse_20260919/python"
sys.path[:0] = [str(ROOT / "src"), str(OLD)]
from inverse_circle_bending import prepare
from node_shape_bending import boundary, fourier, signed_area, convex_turns, AREA, clearance_design, clearance_coordinates
from node_shape_sensitivity import shape_point, sampled_gap


def ibz(grid):
    """Closed triangular Gamma-X-M region on a discrete grid."""
    return np.array([(np.pi*i/grid, np.pi*j/grid)
                     for i in range(grid+1) for j in range(i+1)])


class Evaluator:
    def __init__(self, N, wavevectors, ell_over_h, h_over_a=.08):
        self.N = N
        self.k = wavevectors
        self.ell_over_h = ell_over_h
        self.h_over_a = h_over_a
        self.last_y = None
        self.calls = 0
        self.seconds = 0.

    def evaluate(self, y):
        y = np.asarray(y, float)
        if self.last_y is not None and np.array_equal(self.last_y, y):
            return self.spectra, self.gradients
        tick = time.perf_counter()
        F, dF = fourier(y, self.N, derivative=True)
        cache = prepare(self.N, self.h_over_a, theory="TSDT", indicator_matrix=F)
        freq = []
        grads = []
        for k in self.k:
            f, g, _ = shape_point(y, self.N, k, h_over_a=self.h_over_a,
                                  ell_over_h=self.ell_over_h,
                                  geometry=(F, dF), prepared=cache)
            freq.append(f)
            grads.append(g)
        self.last_y = y.copy()
        self.spectra = np.array(freq)
        self.gradients = np.array(grads)
        self.calls += 1
        self.seconds += time.perf_counter() - tick
        return self.spectra, self.gradients


def optimize(evaluator, start, maxiter, uncapped_area=False, enforce_convexity=True):
    wall_start = time.perf_counter()
    nvar = len(start)
    f, _ = evaluator.evaluate(start)
    ref = sampled_gap(f)
    safe_scale = nvar == 7 and uncapped_area
    coordinates = np.asarray(start, float).copy()
    if safe_scale:
        coordinates = clearance_coordinates(start)
    x0 = np.r_[coordinates, ref["L"], ref["U"]]
    def decode(x):
        return (clearance_design(x[:nvar], True) if safe_scale else
                (np.asarray(x[:nvar]), np.eye(nvar)))
    def evaluate(x):
        y, jac = decode(x)
        freq, grad = evaluator.evaluate(y)
        return freq, grad@jac
    def objective(x):
        L, U = x[-2:]
        return -200*(U-L)/(U+L)
    def objective_jac(x):
        L, U = x[-2:]
        return np.r_[np.zeros(nvar), 400*U/(U+L)**2, -400*L/(U+L)**2]
    def c_lower(x):
        return x[-2] - evaluate(x)[0][:, 3]
    def c_upper(x):
        return evaluate(x)[0][:, 4] - x[-1]
    def j_lower(x):
        return np.column_stack((-evaluate(x)[1][:, 3, :],
                                np.ones(len(evaluator.k)), np.zeros(len(evaluator.k))))
    def j_upper(x):
        return np.column_stack((evaluate(x)[1][:, 4, :],
                                np.zeros(len(evaluator.k)), -np.ones(len(evaluator.k))))
    turn_scale = 2e-4
    def c_convex(x):
        return convex_turns(decode(x)[0])/turn_scale
    def j_convex(x):
        y, jac = decode(x)
        return np.column_stack((convex_turns(y,True)[1]@jac/turn_scale,
                                np.zeros(48), np.zeros(48)))
    def c_clearance(x):
        vertices = boundary(decode(x)[0])
        return np.r_[.49-vertices.ravel(), .49+vertices.ravel()]
    def j_clearance(x):
        y, transform = decode(x)
        _, jac = boundary(y, derivative=True)
        flat = jac.reshape(-1, nvar)@transform
        zero = np.zeros((flat.shape[0], 2))
        return np.vstack((np.column_stack((-flat, zero)),
                          np.column_stack((flat, zero))))
    history = []
    def record(x):
        y, _ = decode(x)
        actual = sampled_gap(evaluate(x)[0])
        history.append(dict(iteration=len(history)+1, J=actual["J"],
                            L=actual["L"], U=actual["U"],
                            epigraph_L=float(x[-2]), epigraph_U=float(x[-1]),
                            min_convex_turn=float(np.min(convex_turns(y))),
                            max_vertex=float(np.max(np.abs(boundary(y)))),
                            elapsed_seconds=time.perf_counter()-wall_start,
                            eigensolve_seconds=evaluator.seconds,
                            evaluations=evaluator.calls,
                            design=y.tolist()))
        print("ITER", len(history), "J", actual["J"], "evaluations", evaluator.calls,
              "elapsed_s", round(time.perf_counter()-wall_start, 2), flush=True)
    constraints = [dict(type="ineq", fun=c_lower, jac=j_lower),
                   dict(type="ineq", fun=c_upper, jac=j_upper)]
    if enforce_convexity:
        constraints.append(dict(type="ineq", fun=c_convex, jac=j_convex))
    # With safe_scale, every trial satisfies clearance by parameterization.
    sol = minimize(objective, x0, jac=objective_jac, method="SLSQP",
                   bounds=[(-.16, .16)]*6 +
                          ([(0., 1.)] if safe_scale else
                           [(float(.5*np.log(.15/AREA)),
                             float(.5*np.log(.40/AREA)))] if nvar == 7 else []) +
                          [(.001, 100.), (.001, 100.)],
                   constraints=constraints,
                   callback=record, options=dict(maxiter=maxiter, ftol=1e-8, disp=False))
    final_y, _ = decode(sol.x)
    spectra, _ = evaluator.evaluate(final_y)
    result = sampled_gap(spectra)
    result.update(design=final_y.tolist(), slsqp_success=bool(sol.success),
                  clearance_parameterization=safe_scale,
                  slsqp_message=str(sol.message), iterations=int(sol.nit),
                  evaluations=evaluator.calls, solve_seconds=evaluator.seconds,
                  wall_seconds=time.perf_counter()-wall_start,
                  epigraph_L=float(sol.x[-2]), epigraph_U=float(sol.x[-1]),
                  min_convex_turn=float(np.min(convex_turns(final_y))),
                  enforce_convexity=enforce_convexity,
                  max_vertex=float(np.max(np.abs(boundary(final_y)))),
                  trajectory=history,
                  max_constraint_violation=float(max(0, np.max(spectra[:, 3]-sol.x[-2]),
                                                    np.max(sol.x[-1]-spectra[:, 4]),
                                                    -np.min(c_clearance(sol.x))
                                                    if uncapped_area else 0.,
                                                    -np.min(convex_turns(final_y))
                                                    if enforce_convexity else 0.)))
    return result, spectra


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--grid", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=8)
    p.add_argument("--starts", type=int, default=1)
    p.add_argument("--seed-source", type=Path)
    p.add_argument("--a-um", type=float, default=75.)
    p.add_argument("--ell-epoxy-um", type=float, default=1.)
    p.add_argument("--h-over-a", type=float, default=.08)
    p.add_argument("--mcst-only", action="store_true")
    p.add_argument("--free-area", action="store_true")
    args = p.parse_args()
    if args.a_um <= 0 or args.h_over_a <= 0 or args.ell_epoxy_um < 0:
        raise ValueError("a and h/a must be positive; MCST length nonnegative")
    h_um = args.h_over_a * args.a_um
    ell_over_h = args.ell_epoxy_um / h_um
    if args.N > 15:
        raise ValueError("Optimization pilot is limited to N<=15; validate independently at higher order")
    seeds = None
    if args.seed_source is not None:
        parent = json.loads(args.seed_source.read_text(encoding="utf-8"))
        if parent["status"] != "PILOT_NOT_MANUSCRIPT_VALIDATED":
            raise ValueError("Seed source must be a pilot manifest")
        seeds = {}
        for name in (("mcst",) if args.mcst_only else ("classic", "mcst")):
            viable = [r for r in parent["records"] if r["model"] == name
                      and r["slsqp_success"] and r["max_constraint_violation"] < 1e-6]
            if not viable:
                raise ValueError(f"No viable {name} seed")
            seed = np.asarray(max(viable, key=lambda r:r["J"])["design"])
            seeds[name] = (np.r_[seed, 0.] if args.free_area and len(seed) == 6
                           else seed[:6] if not args.free_area and len(seed) == 7 else seed)
            if len(seeds[name]) != (7 if args.free_area else 6):
                raise ValueError("Seed dimension does not match area mode")
    prefix = "node_shape_free_area_" if args.free_area else (
        "node_shape_mcst_scale_" if args.mcst_only else "node_shape_pilot_")
    out = ROOT / "runs" / (prefix + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out.mkdir(parents=True)
    wavevectors = ibz(args.grid)
    np.savetxt(out/"ibz.csv", wavevectors, delimiter=",", header="kx_a,ky_a", comments="")
    records = []
    with threadpool_limits(limits=2):
        for name, ellh in (("mcst", ell_over_h),) if args.mcst_only else (("classic", 0.), ("mcst", ell_over_h)):
            for start_id in range(args.starts):
                rng = np.random.default_rng(20260927+start_id)
                nvar = 7 if args.free_area else 6
                start = ((np.zeros(nvar) if start_id == 0 else rng.uniform(-.003, .003, nvar))
                         if seeds is None else
                         (seeds[name].copy() if start_id == 0
                          else np.clip(seeds[name]+rng.uniform(-.001,.001,nvar),-.16,.16)))
                ev = Evaluator(args.N, wavevectors, ellh, args.h_over_a)
                result, spectra = optimize(ev, start, args.maxiter)
                result.update(model=name, start_id=start_id)
                result["area_fraction"] = float(signed_area(boundary(result["design"])))
                np.savetxt(out/f"{name}_{start_id}_spectra.csv", spectra, delimiter=",")
                np.savetxt(out/f"{name}_{start_id}_vertices.csv", boundary(result["design"]),
                           delimiter=",", header="x_over_a,y_over_a", comments="")
                records.append(result)
                print(name, start_id, result["J"], result["slsqp_message"], flush=True)
        # Compare the classical-optimal and MCST-optimal shapes under the
        # SAME MCST objective; no inference is made from different constitutive
        # models' individual objective values.
        best = {name: max((r for r in records if r["model"] == name
                            and r["slsqp_success"]
                            and r["max_constraint_violation"] < 1e-6
                            and r["min_convex_turn"] >= -1e-9),
                           key=lambda r:r["J"], default=None)
                for name in (("mcst",) if args.mcst_only else ("classic", "mcst"))}
        cross = {}
        for name, design in (("circle48", np.zeros(7 if args.free_area else 6)),
                             ("classic_design", None if args.mcst_only or best["classic"] is None else best["classic"]["design"]),
                             ("mcst_design", None if best["mcst"] is None else best["mcst"]["design"])):
            if design is not None:
                ev = Evaluator(args.N, wavevectors, ell_over_h, args.h_over_a)
                cross[name] = sampled_gap(ev.evaluate(np.asarray(design))[0])
    manifest = dict(status="PILOT_NOT_MANUSCRIPT_VALIDATED", N=args.N,
                    ibz_grid=args.grid, ibz_sample_points=len(wavevectors),
                    a_um=args.a_um, h_um=h_um, h_over_a=args.h_over_a,
                    ell_epoxy_um=args.ell_epoxy_um, ell_steel_um=0,
                    target_steel_area=AREA, area_check=signed_area(boundary(np.zeros(6))),
                    no_full_BZ_claim=True, no_N21_claim=True,
                    no_3d_FEM_claim=True, optimizer="SLSQP analytic matrix/eigen gradients",
                    convexity_constraint="all 48 consecutive edge cross products >= 0",
                    maxiter=args.maxiter, starts=args.starts, mcst_only=args.mcst_only,
                    free_area=args.free_area,
                    area_fraction_bounds=[.15,.40] if args.free_area else [AREA,AREA],
                    records=records,
                    seed_source=None if args.seed_source is None else str(args.seed_source.resolve()),
                    same_mcst_cross_evaluation=cross,
                    environment=dict(python=sys.version, platform=platform.platform()),
                    source_sha256={str(s): hashlib.sha256(s.read_bytes()).hexdigest()
                                   for s in (Path(__file__), ROOT/"src/node_shape_bending.py",
                                             ROOT/"src/node_shape_sensitivity.py",
                                             ROOT/"src/inverse_circle_bending.py",
                                             ROOT/"src/real_circle_bending.py")})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("OUTPUT", out, flush=True)

if __name__ == "__main__":
    main()
