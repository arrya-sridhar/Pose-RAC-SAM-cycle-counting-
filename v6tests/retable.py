"""Rebuild the per-cycle table of finished v6.1 runs from their saved traces (no GPU, seconds per video): rewrites
cycles.csv and the `cycles` / `timing` entries of result.json. The count is recomputed too and must not change.
    python v6tests/retable.py <runs folder> [--jobs jobs.json]   (--jobs only for runs made before gt_file was saved)
"""
import os, sys, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import run_v6 as V

ap = argparse.ArgumentParser(); ap.add_argument('runs'); ap.add_argument('--jobs', default=None)
a = ap.parse_args()
gts = {j['name']: j.get('gt') for j in json.load(open(a.jobs))} if a.jobs else {}
for f in sorted(glob.glob(os.path.join(a.runs, '**', 'result.json'), recursive=True)):
    od = os.path.dirname(f); r = json.load(open(f)); z = np.load(os.path.join(od, 'trace.npz'))
    cfg = dict(V.CFG, **{k: v for k, v in r.get('cfg', {}).items() if k in V.CFG})
    t, d, p, i0 = z['t'], z['d'], z['p'], int(z['i0'])
    dt = np.diff(t); pad = int(round(cfg['end_pad_s'] / max(dt[-1] if len(dt) else 0.5, 1e-6)))
    cyc, c = V.trigger_cycles(np.concatenate([p, np.repeat(p[-1:], pad)]))
    if c != r['count']:
        print(f"!! {r['name']}: count {c} from the trace vs {r['count']} in result.json, skipped"); continue
    gp = r.get('gt_file') or gts.get(r['name'])
    g = json.load(open(gp)) if gp else None
    rows, timing = V.cycle_table(V.cycle_rows(t, d, cyc, i0, cfg), r.get('start_s', 0) or 0, cfg['grid_fps'], g)
    r['cycles'] = rows
    if g is not None:
        r['timing'] = timing; r['gt_file'] = gp
    V.write_cycles_csv(os.path.join(od, 'cycles.csv'), r['name'], rows)
    json.dump(r, open(f, 'w'), indent=1, default=float)
    print(f"{r['name']}: {c} cycles" + (f"; timing {timing}" if g is not None else ''))
