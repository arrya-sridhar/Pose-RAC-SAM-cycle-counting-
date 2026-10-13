"""Aggregate run_batch.py results into counting metrics.

Methods scored
  v5 (ours)            : v5 outlines + home/away counter
  Abhinav              : fixed-box outlines + area autocorrelation (count = round(duration / period))
  fixed box + counter  : Abhinav's outlines with our counter      (crossing, to separate outlines from counting)
  v5 + autocorrelation : our outlines with Abhinav's counting       (crossing)
Metrics per method: exact-count accuracy, OBO (|pred - true| <= 1), MAE (RepCount/PoseRAC style: mean |pred - true| / true),
mean absolute error in cycles, RMSE, bias (mean pred - true); overall, per robot program (job) and per cycle-count range.
"""
import os, sys, json, glob, csv
import numpy as np

METHODS = [('v5 (ours)', 'ours', 'counter'), ('Abhinav', 'abhinav', 'autocorr'),
           ('fixed box + our counter', 'abhinav', 'counter'), ('v5 outlines + autocorrelation', 'ours', 'autocorr')]
BINS = [(1, 5), (6, 10), (11, 20), (21, 30)]


def stats(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float); e = pred - true
    if len(true) == 0:
        return dict(n=0)
    return dict(n=int(len(true)), exact=round(float((e == 0).mean()), 4), obo=round(float((np.abs(e) <= 1).mean()), 4),
                mae=round(float((np.abs(e) / true).mean()), 4), mean_abs_err=round(float(np.abs(e).mean()), 3),
                rmse=round(float(np.sqrt((e ** 2).mean())), 3), bias=round(float(e.mean()), 3),
                max_abs_err=int(np.abs(e).max()))


def main(root, expected=None):
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, '*', 'results.json')))]
    out = dict(videos_scored=len(R), frames=int(sum(r['frames'] for r in R)), true_cycles=int(sum(r['true_cycles'] for r in R)))
    if expected:
        want = [l.strip() for l in open(expected) if l.strip()]
        out['videos_expected'] = len(want); out['missing'] = sorted(set(want) - {r['video'] for r in R})
    true = [r['true_cycles'] for r in R]
    out['overall'] = {name: stats([r['methods'][s][k] for r in R], true) for name, s, k in METHODS}
    jobs = sorted({r['job'] for r in R}, key=lambda j: int(j.split('_')[1]))
    out['per_job'] = {j: {name: stats([r['methods'][s][k] for r in R if r['job'] == j], [r['true_cycles'] for r in R if r['job'] == j])
                          for name, s, k in METHODS} for j in jobs}
    out['per_cycle_range'] = {f'{a}-{b}': {name: stats([r['methods'][s][k] for r in R if a <= r['true_cycles'] <= b],
                                                         [r['true_cycles'] for r in R if a <= r['true_cycles'] <= b])
                                            for name, s, k in METHODS} for a, b in BINS}
    for s, label in (('ours', 'v5 outlines'), ('abhinav', 'fixed-box outlines')):
        hits = sum(r['methods'][s]['boundary_hits'] for r in R); nb = sum(max(r['true_cycles'] - 1, 0) for r in R)
        offs = np.array([o for r in R for o in r['methods'][s]['home_entry_offsets_s']])
        out[f'boundaries ({label} + counter)'] = dict(
            true_boundaries=int(nb), inside_detected_home_period=int(hits), rate=round(hits / max(nb, 1), 4),
            extra_home_periods=int(sum(r['methods'][s]['boundary_extra'] for r in R)),
            median_home_entry_offset_s=round(float(np.median(offs)), 2) if len(offs) else None,
            within_1s=round(float((np.abs(offs) <= 1).mean()), 4) if len(offs) else None)
        out[f'outline quality ({label})'] = dict(
            mean_area_pct=round(float(np.mean([r['methods'][s]['area_pct_mean'] for r in R])), 2),
            videos_with_area_over_25pct=int(sum(r['methods'][s]['area_pct_mean'] > 25 for r in R)),
            sudden_jumps_total=int(sum(r['methods'][s]['sudden_jumps'] for r in R)),
            mean_consecutive_iou=round(float(np.mean([r['methods'][s]['mean_consecutive_iou'] for r in R])), 3))
    out['v5 held frames'] = dict(total=int(sum(r['ours_held_frames'] for r in R)),
                                 share=round(sum(r['ours_held_frames'] for r in R) / max(out['frames'], 1), 5))
    out['runtime'] = dict(mean_s_per_frame=round(float(np.mean([r['sam_s_per_frame'] for r in R])), 3),
                          gpu_hours=round(sum(r['sam_s_per_frame'] * r['frames'] for r in R) / 3600, 2))
    json.dump(out, open(os.path.join(root, 'metrics.json'), 'w'), indent=1)
    with open(os.path.join(root, 'per_video.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video', 'job', 'true_cycles', 'frames', 'v5_counter', 'abhinav_autocorr', 'fixedbox_counter', 'v5_autocorr',
                    'abhinav_period_s', 'v5_held_frames', 'v5_area_pct', 'fixedbox_area_pct', 'v5_jumps', 'fixedbox_jumps',
                    'v5_boundary_hits', 'boundaries', 's_per_frame'])
        for r in sorted(R, key=lambda r: (int(r['job'].split('_')[1]), r['video'])):
            o, a = r['methods']['ours'], r['methods']['abhinav']
            w.writerow([r['video'], r['job'], r['true_cycles'], r['frames'], o['counter'], a['autocorr'], a['counter'], o['autocorr'],
                        a['autocorr_period_s'], r['ours_held_frames'], o['area_pct_mean'], a['area_pct_mean'], o['sudden_jumps'],
                        a['sudden_jumps'], o['boundary_hits'], max(r['true_cycles'] - 1, 0), r['sam_s_per_frame']])
    L = [f"videos scored: {out['videos_scored']}" + (f" of {out.get('videos_expected')}" if expected else ''),
         f"frames: {out['frames']}, true cycles: {out['true_cycles']}", '',
         f"{'method':32s} {'exact':>7s} {'OBO':>7s} {'MAE':>7s} {'|err|':>7s} {'RMSE':>7s} {'bias':>7s}"]
    for name, _, _ in METHODS:
        s = out['overall'][name]
        if s.get('n'):
            L.append(f"{name:32s} {s['exact']:7.3f} {s['obo']:7.3f} {s['mae']:7.3f} {s['mean_abs_err']:7.2f} {s['rmse']:7.2f} {s['bias']:7.2f}")
    L += ['', 'per job (v5 exact / Abhinav exact):']
    for j, d in out['per_job'].items():
        L.append(f"  {j:9s} n={d['v5 (ours)']['n']:3d}  v5 {d['v5 (ours)']['exact']:.2f} (MAE {d['v5 (ours)']['mae']:.3f})   "
                 f"Abhinav {d['Abhinav']['exact']:.2f} (MAE {d['Abhinav']['mae']:.3f})")
    L += ['', 'per true-cycle range (v5 exact / Abhinav exact):']
    for b, d in out['per_cycle_range'].items():
        if d['v5 (ours)'].get('n'):
            L.append(f"  {b:6s} n={d['v5 (ours)']['n']:3d}  v5 {d['v5 (ours)']['exact']:.2f}   Abhinav {d['Abhinav']['exact']:.2f}")
    for k in [k for k in out if k.startswith('boundaries') or k.startswith('outline') or k in ('v5 held frames', 'runtime')]:
        L += ['', f'{k}: {json.dumps(out[k])}']
    if expected and out['missing']:
        L += ['', f"MISSING ({len(out['missing'])}): {' '.join(out['missing'])}"]
    open(os.path.join(root, 'metrics.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
