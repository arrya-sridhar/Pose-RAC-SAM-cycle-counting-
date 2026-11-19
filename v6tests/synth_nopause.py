"""Synthetic test for the dynamic frame rate, with exact outlines (no SAM): robots that never pause at home.

Videos are rendered at 30 fps with exact robot masks (src/synth). The same counter is then run on
  - fixed 2 fps (every 15th frame, = v5),
  - fixed 6 fps (= v5.1),
  - fixed 30 fps (every frame),
  - v6's dynamic rate (2 fps, switching to 6 fps or every frame when triggered),
so the only difference is which frames are looked at. Ground truth: 6 cycles per video.
    python v6tests/synth_nopause.py <out_dir>
"""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'src', 'synth'))
from make_video import make_video
import run_v6 as V

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'v6tests', 'synth_out')
CASES = [('nodwell', dict(program='inplane', home_dwell=0.0, home_dwell_jitter=0.0)),
         ('continuous', dict(program='continuous')),
         ('normal', dict(program='inplane'))]          # control: robot rests ~1.5 s at home (v5 already counts these)
SEEDS = [900, 901, 902, 903]
rows = []
for cname, kw in CASES:
    for s in SEEDS:
        cfg = dict(n_cycles=6, fps=30.0, appearance='real', seed=s, **kw)
        od = os.path.join(OUT, f'{cname}_s{s}')
        if not os.path.exists(os.path.join(od, 'gt.npz')):
            make_video(cfg, od)
        z = np.load(os.path.join(od, 'gt.npz')); M = z['masks'].astype(bool); t = z['t']; true = int(z['count']); n = len(M)
        res = {}
        for vname, ks in [('fixed_2fps', list(range(0, n, 15))), ('fixed_6fps', list(range(0, n, 5))), ('fixed_30fps', list(range(n)))]:
            c, _, _ = V.count_samples(t[ks], M[ks], 0, V.CFG); res[vname] = (c['count'], len(ks))
        out = V.adaptive_sample(n, t, 0, V.CFG, lambda k: dict(mask=M[k], ok=True), lambda: None, lambda s: None)
        ks = sorted(out); c, _, _ = V.count_samples(t[ks], M[ks], 0, V.CFG); res['dynamic'] = (c['count'], len(ks))
        row = dict(video=f'{cname}_s{s}', true=true, duration_s=round(float(t[-1]), 1),
                   **{f'{k}_count': v[0] for k, v in res.items()}, **{f'{k}_frames': v[1] for k, v in res.items()})
        rows.append(row); print(json.dumps(row), flush=True)
summ = {}
for cname, _ in CASES:
    r = [x for x in rows if x['video'].startswith(cname)]
    summ[cname] = {v: dict(exact=sum(x[f'{v}_count'] == x['true'] for x in r), of=len(r),
                           frames=sum(x[f'{v}_frames'] for x in r)) for v in ['fixed_2fps', 'fixed_6fps', 'fixed_30fps', 'dynamic']}
print(json.dumps(summ, indent=1))
json.dump(dict(rows=rows, summary=summ, cfg=V.CFG), open(os.path.join(OUT, 'synth_nopause_results.json'), 'w'), indent=1)
