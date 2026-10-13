"""Summarise the real-video run: counts vs ground truth, whether each true cycle boundary falls inside a detected
'robot at home' period, and mask sanity.  Reads real/<video>/{results.json, sam_vitb_masks.npz, gt.json} + setup.json.

Ground-truth boundaries in this dataset are the points where one recording of the cycle ends and the next copy begins.
The robot is parked at home around that point, so the fair timing check is: does a detected home period contain it?
"""
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import poserac as PR
from features import mask_iou_to
from evaluate import otsu

FPS, GW, GH = 2.0, 640, 360


def unpack(x, n):
    return np.unpackbits(x, axis=1)[:, :GW * GH].reshape(n, GH, GW).astype(bool)


def refdist_prob(d):
    tau = otsu(d); lo, hi = tau / 2, (tau + d.max()) / 2
    return np.clip((hi - d) / max(hi - lo, 1e-6), 0, 1)


def home_periods(p, hold, enter=0.78, exit=0.4, momentum=0.4):
    """per frame: True while the official two-threshold trigger holds 'home' (smoothed score went above 0.78 and has
    not yet dropped below 0.4)"""
    p = np.concatenate([p, np.repeat(p[-1:], hold)])
    r1 = PR.ActionTrigger(enter, exit)
    prob, state = 0.5, []
    for o in p:
        prob = o * (1 - momentum) + momentum * prob
        r1(prob); state.append(r1.entered)
    state = np.array(state)
    iv, i = [], 0
    while i < len(state):
        if state[i]:
            j = i
            while j + 1 < len(state) and state[j + 1]: j += 1
            iv.append((i / FPS, j / FPS)); i = j + 1
        else:
            i += 1
    return iv


def boundary_check(iv, gt_internal, T, slack=1.0):
    caught = sum(any(s - slack <= b <= e + slack for s, e in iv) for b in gt_internal)
    inner = [(s, e) for s, e in iv if s > 0.5 and e < T - 0.5]          # home periods other than at the very start / end
    extra = sum(not any(s - slack <= b <= e + slack for b in gt_internal) for s, e in inner)
    return caught, extra


def main(root):
    setups = json.load(open(os.path.join(root, 'setup.json')))
    rows = []
    for name in sorted(setups, key=lambda s: int(s.split('_')[1])):
        vd = os.path.join(root, name)
        if not os.path.exists(os.path.join(vd, 'results.json')):
            continue
        r = json.load(open(os.path.join(vd, 'results.json'))); gt = json.load(open(os.path.join(vd, 'gt.json')))
        z = np.load(os.path.join(vd, 'sam_vitb_masks.npz')); n = r['frames']; f0 = setups[name].get('init_frame', 0)
        T = n / FPS; gt_internal = [c['end_time'] for c in gt['cycles']][:-1]
        row = dict(video=name, gt=r['true_cycles'], dur=round(r['duration_s'], 1), rejected=r.get('ours_rejected_masks'),
                   his_csv=r['methods'].get('abhinav_original_csv', {}).get('autocorr_count'))
        for s in ('abhinav', 'ours'):
            M = unpack(z[f'{s}__masks'], n); m = r['methods'][s]
            d = 1 - mask_iou_to(M, M[f0]); iv = home_periods(refdist_prob(d), int(round(1.5 * FPS)))
            caught, extra = boundary_check(iv, gt_internal, T)
            ci = np.array([(M[k] & M[k + 1]).sum() / max((M[k] | M[k + 1]).sum(), 1) for k in range(n - 1)])
            row[s + '_stability'] = dict(mean_consecutive_iou=round(float(ci.mean()), 3), jumps=int((ci < 0.6).sum()))
            row[s] = dict(area=m['mask_area_pct']['mean'], area_max=m['mask_area_pct']['max'], bad=m['implausible_mask_frames'],
                          autocorr=m['autocorr']['count'], lf=m['labelfree_counter']['count'], lfpr=m['labelfree_poserac']['count'],
                          caught=caught, of=len(gt_internal), extra=extra)
        rows.append(row)
    return rows


if __name__ == '__main__':
    rows = main(sys.argv[1])
    json.dump(rows, open(os.path.join(sys.argv[1], 'analysis.json'), 'w'), indent=1, default=float)
    hdr = f"{'video':18s} GT | his CSV | Abh setup: area bad AC LF PR bnd | ours: area max bad rej AC LF PR bnd"
    print(hdr)
    for r in rows:
        a, o = r['abhinav'], r['ours']
        print(f"{r['video']:18s} {r['gt']:2d} | {str(r['his_csv']):>7s} | {a['area']:5.1f} {a['bad']:3d} {a['autocorr']:2d} {a['lf']:2d} {a['lfpr']:2d} "
              f"{a['caught']}/{a['of']}+{a['extra']} | {o['area']:5.1f} {o['area_max']:5.1f} {o['bad']:3d} {str(r['rejected']):>3s} "
              f"{o['autocorr']:2d} {o['lf']:2d} {o['lfpr']:2d} {o['caught']}/{o['of']}+{o['extra']}")
    for s, k in [('abhinav', 'autocorr'), ('abhinav', 'lf'), ('abhinav', 'lfpr'), ('ours', 'autocorr'), ('ours', 'lf'), ('ours', 'lfpr')]:
        print(s, k, 'exact', sum(r[s][k] == r['gt'] for r in rows), '/', len(rows))
    print('his CSV autocorr exact', sum(r['his_csv'] == r['gt'] for r in rows), '/', len(rows))
