"""Counting methods on top of SAM (or oracle) robot masks, and their evaluation against ground truth.

Methods
  poserac_sup  : PoseRAC trained on pose-saliency annotations (home frame + farthest frame per cycle) from training videos
  poserac_rb   : label-free PoseRAC; pseudo-labels from distance to the frame-0 reference mask, trained on the test video itself
  refdist      : no learning; smoothed 'home probability' from mask IoU with the frame-0 mask + the same PoseRAC trigger
  autocorr     : Abhinav's approach; strongest autocorrelation period of the mask-area signal -> count = floor(T / period)
"""
import os, json, numpy as np
from features import unpack, compute, mask_iou_to, GW, GH
import poserac as PR

FEATS = {'grid': (99, 9, False), 'polar': (99, 9, False), 'geom': (7, 7, True)}   # dim, heads, z-score


# ---------------------------------------------------------------- data
def load_video(vd, source, pad_end=True):
    """source: 'gt' or '<model>:<strategy>' e.g. 'mobile:track_motion_pt'"""
    gt = np.load(os.path.join(vd, 'gt.npz'))
    if source == 'gt':
        M = unpack(gt['masks'])
        b = gt['boxes'].astype(float)
        wb = np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])
        w, h = wb[2] - wb[0], wb[3] - wb[1]
        wb = np.array([max(0, wb[0] - .03 * w - 8), max(0, wb[1] - .03 * h - 8), min(1279, wb[2] + .03 * w + 8), min(719, wb[3] + .03 * h + 8)])
        base = gt['base_px'] + 0.45 * (gt['shoulder_px'] - gt['base_px'])
    else:
        model, strat = source.split(':')
        sm = np.load(os.path.join(vd, f'sam_{model}.npz'))
        M = unpack(sm[f'{strat}__masks'])
        wb, base = sm['workspace_box'], sm['base_point']
    n = min(len(M), len(gt['t']))
    G = unpack(gt['masks'][:n])
    miou = ((M[:n] & G).sum((1, 2)) / np.maximum((M[:n] | G).sum((1, 2)), 1))
    v = dict(M=M[:n], G=G, t=gt['t'][:n], fps=float(gt['fps']), count=int(gt['count']), arrive=gt['arrive'],
             depart=gt['depart'], cycle=gt['cycle'][:n], wb=wb, base=base, miou=miou, padded=0)
    # evaluation protocol: if the robot is back home at the end of the clip, hold the last frame for 1.5 s so the
    # final cycle is not lost to the counter's smoothing lag.  Clips cut mid-cycle ('tim_partial') are not padded.
    if pad_end and gt['dist_home'][n - 1] < 1e-6:
        k = int(round(1.5 * v['fps']))
        for key in ('M', 'G', 'cycle', 'miou'):
            v[key] = np.concatenate([v[key], np.repeat(v[key][-1:], k, axis=0)])
        v['t'] = np.concatenate([v['t'], v['t'][-1] + np.arange(1, k + 1) / v['fps']])
        v['padded'] = k
    return v


def subsample(v, step):
    """keep every `step`-th frame (lower fps)"""
    out = dict(v)
    for k in ('M', 'G', 't', 'cycle', 'miou'):
        out[k] = v[k][::step]
    out['fps'] = v['fps'] / step
    return out


def saliency_annotation(v, n_cycles=None):
    """PoseRAC-style pose-saliency annotation from GT: per complete cycle, salient-1 = last home frame before
    departure, salient-2 = the frame visually farthest from home (max 1-IoU of GT masks)."""
    t, G = v['t'], v['G']
    s1, s2 = [], []
    ks = range(len(v['depart'])) if n_cycles is None else range(min(n_cycles, len(v['depart'])))
    for k in ks:
        d, a = v['depart'][k], v['arrive'][k]
        i1 = np.where(t <= d)[0]
        if len(i1) == 0:
            continue
        i1 = i1[-1]
        inside = np.where((t > d) & (t < a))[0]
        if len(inside) == 0:
            continue
        far = 1 - mask_iou_to(G[inside], G[i1])
        s1.append(i1); s2.append(inside[int(np.argmax(far))])
    return np.array(s1), np.array(s2)


# ---------------------------------------------------------------- metrics
def boundary_f1(pred_t, gt_t, tol):
    pred_t, gt_t = list(pred_t), list(gt_t)
    used, tp = set(), 0
    for g in gt_t:
        best = None
        for j, pt in enumerate(pred_t):
            if j in used or abs(pt - g) > tol:
                continue
            if best is None or abs(pt - g) < abs(pred_t[best] - g):
                best = j
        if best is not None:
            used.add(best); tp += 1
    P = tp / max(len(pred_t), 1); R = tp / max(len(gt_t), 1)
    return 2 * P * R / max(P + R, 1e-9)


def score(v, count, home_entries_idx=None, tol=1.5):
    gt = v['count']
    r = dict(pred=int(count), gt=gt, abs_err=abs(count - gt), mae=abs(count - gt) / gt, obo=int(abs(count - gt) <= 1),
             exact=int(count == gt))
    if home_entries_idx is not None:
        r['bf1'] = boundary_f1(v['t'][home_entries_idx] if len(home_entries_idx) else [], v['arrive'], tol)
    return r


# ---------------------------------------------------------------- methods
def otsu(x, bins=64):
    h, e = np.histogram(x, bins=bins); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))]


def ref_distance(v):
    return 1 - mask_iou_to(v['M'], v['M'][0])


def refdist_prob(d):
    tau = otsu(d)
    lo, hi = tau / 2, (tau + d.max()) / 2
    return np.clip((hi - d) / max(hi - lo, 1e-6), 0, 1), lo, hi


def m_refdist(v):
    d = ref_distance(v)
    p, _, _ = refdist_prob(d)
    c, ev, he = PR.count_official(p)
    return score(v, c, he), p


def m_autocorr(v, min_period=3.0):
    a = v['M'].mean((1, 2)).astype(float)
    a = a - a.mean()
    n = len(a); fps = v['fps']
    ac = np.correlate(a, a, 'full')[n - 1:] / (np.arange(n, 0, -1) * (a.var() + 1e-12))
    lo, hi = int(np.ceil(min_period * fps)), n // 2
    if hi <= lo + 1:
        return score(v, 0), None
    # strongest local peak
    seg = ac[lo:hi]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    k = (max(peaks, key=lambda i: seg[i]) if peaks else int(np.argmax(seg))) + lo
    period = k / fps
    return score(v, int(np.floor(v['t'][-1] / period))), dict(period=period, ac=float(ac[k]))


def features(v, kind):
    return compute(v['M'], kind, v['wb'], v['base'])


def m_poserac_sup(train_vs, test_v, kind='grid', mode='official', n_cycles=None, seed=42):
    dim, heads, z = FEATS[kind]
    X, Y = [], []
    for tv in train_vs:
        s1, s2 = saliency_annotation(tv, n_cycles)
        F = features(tv, kind)
        X += [F[s1], F[s2]]; Y += [np.ones((len(s1), 1)), np.zeros((len(s2), 1))]
    X = np.concatenate(X); Y = np.concatenate(Y)
    mu, sd = (X.mean(0), X.std(0) + 1e-6) if z else (0.0, 1.0)
    net = PR.train_poserac((X - mu) / sd, Y, dim, heads, mode=mode, seed=seed)
    p = PR.predict(net, (features(test_v, kind) - mu) / sd)
    c, ev, he = PR.count_official(p)
    return score(test_v, c, he), p


def m_poserac_rb(v, kind='grid', mode='official', seed=42):
    """label-free: pseudo salient-1 = frames close to the frame-0 reference, salient-2 = frames far from it"""
    dim, heads, z = FEATS[kind]
    d = ref_distance(v)
    _, lo, hi = refdist_prob(d)
    i1 = np.where(d <= lo)[0]; i2 = np.where(d >= hi)[0]
    rng = np.random.default_rng(seed)
    if len(i2) == 0 or len(i1) == 0:
        return score(v, 0, []), None
    if len(i1) > 2 * len(i2): i1 = rng.choice(i1, 2 * len(i2), replace=False)
    if len(i2) > 2 * len(i1): i2 = rng.choice(i2, 2 * len(i1), replace=False)
    F = features(v, kind)
    if z:
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    X = np.concatenate([F[i1], F[i2]]); Y = np.concatenate([np.ones((len(i1), 1)), np.zeros((len(i2), 1))])
    net = PR.train_poserac(X, Y, dim, heads, mode=mode, seed=seed)
    p = PR.predict(net, F)
    c, ev, he = PR.count_official(p)
    return score(v, c, he), p
