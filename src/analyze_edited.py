"""Score the edited (non-identical cycles) test set.

For each video: true cycle count vs
  Abhinav's pipeline      = his fixed box masks + area autocorrelation (count = round(duration / period))
  ours                    = v5 following-box masks + home/away counter
  plus the two crossings (his masks + our counter, our masks + his counter) to separate masks from counting.
Timing: each true cycle end (robot arrives home) is 'found' if the counter registers a return home within
-1 s .. +2 s of it; 'extra' = registered returns home that match no true cycle end.
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import poserac as PR
from features import mask_iou_to
from analyze_real import unpack, refdist_prob

FPS = 2.0


def home_entries(p, hold=3):
    p = np.concatenate([p, np.repeat(p[-1:], hold)])
    c, ev, he = PR.count_official(p)
    return c, [i / FPS for i in he]


def timing(entries, ends, lo=-1.0, hi=2.0):
    used, found, errs = set(), 0, []
    for e in ends:
        cand = [(abs(x - e), j) for j, x in enumerate(entries) if j not in used and lo <= x - e <= hi]
        if cand:
            d, j = min(cand); used.add(j); found += 1; errs.append(entries[j] - e)
    return found, len(entries) - len(used), (round(float(np.median(errs)), 2) if errs else None)


def main(root):
    setups = json.load(open(os.path.join(root, 'setup.json')))
    rows = []
    for name in sorted(setups, key=lambda s: int(s.split('_')[1])):
        vd = os.path.join(root, name)
        if not os.path.exists(os.path.join(vd, 'results.json')):
            continue
        r = json.load(open(os.path.join(vd, 'results.json'))); gt = json.load(open(os.path.join(vd, 'gt.json')))
        z = np.load(os.path.join(vd, 'sam_vitb_masks.npz')); n = r['frames']
        ends = [c['end_time'] for c in gt['cycles']]
        row = dict(video=name, clip=gt['source_clip'], true=gt['total_cycles'], speeds=gt['plan']['speeds'],
                   waits=gt['plan']['wait_after_each_cycle_s'], mid_stop=gt['plan']['mid_cycle_stop'],
                   duration=gt['total_duration_sec'], held=r.get('ours_rejected_masks'))
        for s in ('abhinav', 'ours'):
            M = unpack(z[f'{s}__masks'], n); m = r['methods'][s]
            d = 1 - mask_iou_to(M, M[0]); c, he = home_entries(refdist_prob(d))
            f, extra, med = timing(he, ends)
            row[s] = dict(autocorr=m['autocorr']['count'], period=m['autocorr']['period_s'], counter=c,
                          ends_found=f, of=len(ends), extra=extra, median_offset_s=med, area=m['mask_area_pct']['mean'])
        rows.append(row)
    return rows


if __name__ == '__main__':
    root = sys.argv[1]; rows = main(root)
    json.dump(rows, open(os.path.join(root, 'analysis_edited.json'), 'w'), indent=1, default=float)
    print(f"{'video':8s} true | Abhinav (his masks+autocorr) | ours (our masks+counter) | his masks+counter | our masks+autocorr | ends found (ours) | held")
    for r in rows:
        a, o = r['abhinav'], r['ours']
        print(f"{r['video']:8s} {r['true']:4d} | {a['autocorr']:4d} (period {a['period']})"
              f" | {o['counter']:4d} | {a['counter']:4d} | {o['autocorr']:4d} (period {o['period']})"
              f" | {o['ends_found']}/{o['of']} +{o['extra']} (median {o['median_offset_s']} s) | {r['held']}")
    ex = lambda s, k: sum(r[s][k] == r['true'] for r in rows)
    print(f"exact: Abhinav pipeline {ex('abhinav','autocorr')}/{len(rows)} | ours {ex('ours','counter')}/{len(rows)} | "
          f"his masks+counter {ex('abhinav','counter')}/{len(rows)} | our masks+autocorr {ex('ours','autocorr')}/{len(rows)}")
    print('ours: cycle ends found', sum(r['ours']['ends_found'] for r in rows), '/', sum(r['ours']['of'] for r in rows),
          ' extra', sum(r['ours']['extra'] for r in rows))
