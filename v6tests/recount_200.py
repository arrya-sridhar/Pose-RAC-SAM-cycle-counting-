"""Counting-stage check on all 200 dataset videos, from v5's stored outlines (no GPU, minutes).

Checks that v6's counter on evenly spaced frames reproduces v5's counts exactly, then measures the counting extras on
the 200 videos: the 120 s window cut-off, cycle end "at rest", and the warning flags (which videos get flagged, and
whether the 3 nachi_4 misses are among them).
    python v6tests/recount_200.py <v5 results folder> <dataset ground_truth folder> <out.json>
"""
import os, sys, json, glob
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
import run_v6 as V, run_real as RR

res_dir, gt_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for d in sorted(glob.glob(os.path.join(res_dir, 'nachi_*'))):
    n_ = os.path.basename(d)
    if not os.path.exists(os.path.join(d, 'outlines.npz')):
        continue
    r = json.load(open(os.path.join(d, 'results.json'))); g = json.load(open(os.path.join(gt_dir, n_ + '.json')))
    z = np.load(os.path.join(d, 'outlines.npz')); n = r['frames']
    M = np.unpackbits(z['ours__masks'], axis=1)[:, :V.GW * V.GH].reshape(n, V.GH, V.GW).astype(bool)
    t = np.arange(n) / 2.0; i0 = r['reference_frame']
    c, dd, p = V.count_samples(t, M, i0, V.CFG)
    ok = np.ones(n, bool); held = int(r.get('ours_held_frames', 0))
    fl, finfo = V.flags(t, M, dd, c['returns_s'], ok, c['count'], float(r['duration_s']))
    if held / n > 0.05:
        fl.append(f'{100 * held / n:.0f}% of frames held')
    ends = [cy['end_time'] for cy in g['cycles']][:-1]
    true = int(g['total_cycles'])
    rows.append(dict(video=n_, true=true, v5=int(r['methods']['ours']['counter']), v6_counter=c['count'],
                     window=c['count_window'], flags=fl, home_overlap=finfo['home_overlap_at_returns'],
                     autocorr=finfo['autocorr_count'],
                     ends_v5style=RR.boundary_scores(c['returns_s'], ends), ends_at_rest=RR.boundary_scores(c['rest_end_s'], ends)))
    print(n_, true, rows[-1]['v5'], c['count'], c['count_window'], len(fl), flush=True)

def tot(key, of='ends_v5style'):
    return sum(x[of]['found'] for x in rows), sum(x[of]['of'] for x in rows)
misses = [x['video'] for x in rows if x['v5'] != x['true']]
summ = dict(videos=len(rows),
            v6_counter_matches_v5=sum(x['v6_counter'] == x['v5'] for x in rows),
            exact_v5=sum(x['v5'] == x['true'] for x in rows), exact_window=sum(x['window'] == x['true'] for x in rows),
            window_changed=[(x['video'], x['true'], x['v5'], x['window']) for x in rows if x['window'] != x['v5']],
            flagged=[(x['video'], x['flags']) for x in rows if x['flags']],
            flagged_count=sum(bool(x['flags']) for x in rows), v5_misses=misses,
            misses_flagged=[m for m in misses if any(x['video'] == m and x['flags'] for x in rows)],
            lost_part_flag=[x['video'] for x in rows if any('part of the robot' in f for f in x['flags'])],
            ends_within_1s_v5style=tot('', 'ends_v5style'), ends_within_1s_at_rest=tot('', 'ends_at_rest'))
print(json.dumps({k: v for k, v in summ.items() if k != 'flagged'}, indent=1))
json.dump(dict(summary=summ, rows=rows), open(out, 'w'), indent=1, default=float)
