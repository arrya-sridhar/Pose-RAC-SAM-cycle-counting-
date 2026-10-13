"""(v3) End-to-end check of the stitched pipeline on REAL Nachi videos (frames already extracted at 2 fps, 1280x720).

For each video folder  real/<name>/{frames/*.jpg, gt.json, abhinav_sam_motion.csv}  and a setup entry
(init_box, base_point, init_frame) it:
  1. runs SAM ViT-B with three prompt setups on every frame (one image embedding per frame):
       abhinav      - Abhinav's fixed box [730,70,1450,1070]@1920x1080, first mask candidate (as in run_sam_10videos.py)
       abhinav_best - the same fixed box, highest-scoring mask
       ours         - box that follows the robot (previous mask + motion) + one click on the robot base
  2. counts cycles on each set of masks with
       - Abhinav's method: autocorrelation of mask area -> period -> count = round(duration / period)
       - label-free home/away counter (PoseRAC's two-threshold counter on overlap with the reference frame)
       - label-free PoseRAC (pseudo-labels from overlap with the reference frame, per-frame model, 11x9 grid features)
  3. scores against ground truth (cycle count; boundary times within +/-1 s) and writes overlays + an annotated video.
Supervised PoseRAC is not run: every other video of the same job is identical footage, so training on it would leak.
"""
import os, sys, json, time, argparse, glob, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, cv2, torch
import poserac as PR
from features import compute as feat_compute, mask_iou_to
from evaluate import otsu

FPS = 2.0
ABHINAV_BOX = np.array([730.0, 70.0, 1450.0, 1070.0]) * (1280 / 1920)   # his box, scaled to 1280x720
GW, GH = 640, 360


# ------------------------------------------------------------------ helpers
def expand(b, r, W=1280, H=720, pad=8):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([max(0, b[0] - r * w - pad), max(0, b[1] - r * h - pad), min(W - 1, b[2] + r * w + pad), min(H - 1, b[3] + r * h + pad)])

def clip_to(b, wb):
    return np.array([max(b[0], wb[0]), max(b[1], wb[1]), min(b[2], wb[2]), min(b[3], wb[3])])

def bbox(m):
    ys, xs = np.where(m)
    return None if len(xs) == 0 else np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)

def union(a, b):
    return np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])])

def motion_box(g0, g1, wb, thr=18, min_px=800):
    d = cv2.absdiff(cv2.GaussianBlur(g0, (0, 0), 2), cv2.GaussianBlur(g1, (0, 0), 2)) > thr
    roi = np.zeros_like(d); x0, y0, x1, y1 = wb.astype(int); roi[y0:y1, x0:x1] = True
    d = cv2.morphologyEx((d & roi).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]
    return bbox(np.isin(lab, keep)) if keep else None

def motion_envelope(grays, init_box, thr=30):
    """working area = everything that ever differs from the median frame (static camera) + the starting box"""
    small = np.array(grays).astype(np.int16)
    med = np.median(small, axis=0)
    moving = (np.abs(small - med) > thr).mean(0) > 0.02
    moving = cv2.morphologyEx(moving.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    b = bbox(moving)
    wb = init_box.copy() if b is None else union(init_box, b * 4)
    return expand(wb, 0.05)

K_NEAR = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))    # ~25 px at 1280x720
K_SPECK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

def clean_mask(m, min_px=1500):
    """drop specks and thin slivers (opening), then drop small leftover pieces"""
    o = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, K_SPECK)
    n, lab, st, _ = cv2.connectedComponentsWithStats(o)
    keep = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= min_px]
    return np.isin(lab, keep) if keep else m.astype(bool)

def load_predictor(device):
    from segment_anything import sam_model_registry, SamPredictor
    m = sam_model_registry['vit_b'](checkpoint=os.path.join(HERE, 'weights', 'sam_b.pt')).eval().to(device)
    return SamPredictor(m)


# ------------------------------------------------------------------ SAM
def run_sam(frames, setup, device, log):
    P = load_predictor(device)
    init_box = np.array(setup['init_box'], float); bp = np.array(setup['base_point'], float); f0 = setup.get('init_frame', 0)
    wb = motion_envelope([cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (320, 180), interpolation=cv2.INTER_AREA) for f in frames], init_box)
    prev_gray = None
    out = {s: dict(masks=[], score=[], box=[]) for s in ['abhinav', 'abhinav_best', 'ours']}
    prev_mask, prev_box, area_hist, n_rejected, area0, rejected_idx = None, init_box, [], 0, None, []
    t0 = time.time()
    for i, f in enumerate(frames):
        bgr = cv2.imread(f); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = lambda x: cv2.resize(x.astype(np.uint8), (GW, GH), interpolation=cv2.INTER_AREA) > 0
        with torch.inference_mode():
            P.set_image(rgb)
            m, sc, _ = P.predict(box=ABHINAV_BOX, multimask_output=True)
            for s, k in [('abhinav', 0), ('abhinav_best', int(np.argmax(sc)))]:
                out[s]['masks'].append(small(m[k])); out[s]['score'].append(float(sc[k])); out[s]['box'].append(ABHINAV_BOX)
            # ours (v3): box follows the robot (from init_frame on) + base click, with temporal consistency.
            # v1 unioned the previous box with ALL detected motion and adopted any mask, so the box could creep outward
            # until SAM picked the wall.  v2 added a mask-size check, but a few specks on a static wall panel still
            # stretched the box to the top of the image (nachi_5 on the M3 GPU).  v3: candidates are cleaned (specks and
            # slivers removed), and because the camera is fixed, any part of a candidate that was not robot in the previous
            # frame must lie where the image actually changed; a static wall panel fails that test.
            motion = None
            if prev_gray is not None:
                mv = cv2.absdiff(cv2.GaussianBlur(prev_gray, (0, 0), 2), cv2.GaussianBlur(gray, (0, 0), 2)) > 18
                motion = cv2.dilate(mv.astype(np.uint8), K_NEAR) > 0
            if i <= f0 or prev_mask is None:
                boxes = [init_box]
            else:
                pb = bbox(prev_mask); pb = prev_box if pb is None else pb
                near = expand(pb, 0.5)                                         # only motion close to the robot counts
                mb = motion_box(prev_gray, gray, clip_to(near, wb)) if prev_gray is not None else None
                boxes = [clip_to(expand(pb, 0.08), wb)]                        # tight box around the previous robot mask
                if mb is not None:
                    boxes.append(clip_to(expand(union(pb, mb), 0.08), wb))     # ... and one that also covers nearby motion
            cands = []
            for bx in boxes:
                m, sc, _ = P.predict(point_coords=bp[None], point_labels=np.array([1]), box=bx, multimask_output=True)
                for j in range(3):
                    cands.append((clean_mask(m[j]), float(sc[j]), bx))
            ok = True
            if prev_mask is None or i <= f0:
                mk, s_k, box = max(cands, key=lambda c: c[1])
            else:
                allowed = (cv2.dilate(prev_mask.astype(np.uint8), K_NEAR) > 0) | (motion if motion is not None else False)
                med_area = float(np.median(area_hist[-10:]))
                def judge(c):
                    a = c[0].sum(); u = (c[0] & ~allowed).sum() / max(a, 1)      # share of the mask that is new AND static
                    iou = (c[0] & prev_mask).sum() / max((c[0] | prev_mask).sum(), 1)
                    plaus = 0.6 * med_area <= a <= 1.6 * med_area and 0.4 * area0 <= a <= 2.0 * area0 and u <= 0.10
                    return iou + 0.1 * c[1] - 2.0 * u, plaus
                judged = [(judge(c), c) for c in cands]
                good = [x for x in judged if x[0][1]]
                if good:
                    mk, s_k, box = max(good, key=lambda x: x[0][0])[1]
                else:                                                          # nothing plausible: keep only the part of
                    mk, s_k, box = max(judged, key=lambda x: x[0][0])[1]       # the best candidate that was robot before
                    mk = clean_mask(mk & allowed); ok = False                  # or moved; don't let it steer the tracker
            if i >= f0 and ok:
                prev_mask, prev_box = mk, box; area_hist.append(mk.sum())
                area0 = mk.sum() if area0 is None else area0
            n_rejected += (i > f0 and not ok); rejected_idx += [i] if (i > f0 and not ok) else []
            out['ours']['masks'].append(small(mk)); out['ours']['score'].append(s_k); out['ours']['box'].append(box)
        prev_gray = gray
        if i % 20 == 0:
            log(f'    frame {i + 1}/{len(frames)}  {(time.time() - t0) / (i + 1):.2f} s/frame')
    for s in out:
        out[s]['masks'] = np.array(out[s]['masks'])
        out[s]['score'] = np.array(out[s]['score']); out[s]['box'] = np.array(out[s]['box'])
    out['ours']['rejected_frames'] = n_rejected; out['ours']['rejected_idx'] = rejected_idx
    return out, wb, (time.time() - t0) / len(frames)


# ------------------------------------------------------------------ counting
def autocorr_count(area, duration, min_period=3.0):
    """Abhinav's approach: strongest autocorrelation peak of the area signal -> period -> round(duration/period)"""
    a = np.asarray(area, float); a = a - a.mean(); n = len(a)
    if a.std() < 1e-9:
        return 0, None, 0.0
    ac = np.correlate(a, a, 'full')[n - 1:] / (n * a.var())        # biased (divide by N) normalisation
    lo, hi = int(np.ceil(min_period * FPS)), int(0.95 * n)
    peaks = [k for k in range(max(lo, 1), min(hi, n - 1)) if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
    if not peaks:
        return 0, None, 0.0
    k = max(peaks, key=lambda j: ac[j]); period = k / FPS
    return int(round(duration / period)), period, float(ac[k])

def counter(p, hold):
    """official PoseRAC counter; also returns departures from home and home re-entry frame indices"""
    p = np.concatenate([p, np.repeat(p[-1:], hold)])
    c, ev, he = PR.count_official(p)
    r1, prob, dep = PR.ActionTrigger(), 0.5, 0
    for o in p:
        prob = o * 0.6 + 0.4 * prob
        dep += r1(prob)
    return c, dep, he, p

def refdist_prob(d):
    tau = otsu(d); lo, hi = tau / 2, (tau + d.max()) / 2
    return np.clip((hi - d) / max(hi - lo, 1e-6), 0, 1), lo, hi

def labelfree_poserac(F, d, lo, hi, seed=42):
    i1 = np.where(d <= lo)[0]; i2 = np.where(d >= hi)[0]
    if len(i1) == 0 or len(i2) == 0:
        return np.full(len(d), 0.5)
    rng = np.random.default_rng(seed)
    if len(i1) > 2 * len(i2): i1 = rng.choice(i1, 2 * len(i2), replace=False)
    if len(i2) > 2 * len(i1): i2 = rng.choice(i2, 2 * len(i1), replace=False)
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-6)
    X = np.concatenate([Fz[i1], Fz[i2]]); Y = np.concatenate([np.ones((len(i1), 1)), np.zeros((len(i2), 1))])
    net = PR.train_poserac(X, Y, 99, 9, mode='perframe', seed=seed)
    return PR.predict(net, Fz)

def boundary_scores(pred_t, gt_t, tol=1.0):
    pred_t, used, tp, errs = list(pred_t), set(), 0, []
    for g in gt_t:
        cand = [(abs(pt - g), j) for j, pt in enumerate(pred_t) if j not in used and abs(pt - g) <= tol]
        if cand:
            e, j = min(cand); used.add(j); tp += 1; errs.append(e)
    P = tp / max(len(pred_t), 1); R = tp / max(len(gt_t), 1)
    return dict(found=tp, of=len(gt_t), extra=len(pred_t) - tp, f1=round(2 * P * R / max(P + R, 1e-9), 3),
                median_err_s=round(float(np.median(errs)), 2) if errs else None)


# ------------------------------------------------------------------ visuals
def overlay(img, mask, color, alpha=0.5):
    m = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    o = img.copy(); o[m] = ((1 - alpha) * o[m] + alpha * np.array(color)).astype(np.uint8); return o

def contact_sheet(frames, sam, path, n=8):
    idx = np.linspace(0, len(frames) - 1, n).astype(int); rows = []
    for i in idx:
        im = cv2.resize(cv2.imread(frames[i]), (640, 360))
        a = overlay(im, sam['abhinav']['masks'][i], (0, 0, 255)); b = overlay(im, sam['ours']['masks'][i], (0, 200, 0))
        cv2.putText(a, f't={i / FPS:.1f}s  Abhinav setup (red)', (10, 30), 0, 0.8, (255, 255, 255), 4); cv2.putText(a, f't={i / FPS:.1f}s  Abhinav setup (red)', (10, 30), 0, 0.8, (0, 0, 0), 2)
        cv2.putText(b, 'ours (green)', (10, 30), 0, 0.8, (255, 255, 255), 4); cv2.putText(b, 'ours (green)', (10, 30), 0, 0.8, (0, 0, 0), 2)
        rows.append(np.hstack([a, b]))
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])

def annotated_video(frames, sam, p_ours, gt_bounds, path):
    W, H = 1280, 720 + 150
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (W, H))
    r1, r2 = PR.ActionTrigger(), PR.ActionTrigger(); prob, count, curr, init = 0.5, 0, 'holder', 'pose_holder'
    for i, f in enumerate(frames):
        im = cv2.imread(f); o = overlay(im, sam['ours']['masks'][i], (60, 200, 60), 0.55)
        b = sam['ours']['box'][i].astype(int); cv2.rectangle(o, (b[0], b[1]), (b[2], b[3]), (0, 140, 255), 3)
        prob = p_ours[i] * 0.6 + 0.4 * prob; s1 = r1(prob); s2 = r2(1 - prob)
        if init == 'pose_holder': init = 'salient1' if s1 else ('salient2' if s2 else init)
        if init == 'salient1' and curr == 'salient1' and s2: count += 1
        if s1: curr = 'salient1'
        elif s2: curr = 'salient2'
        t = i / FPS; done = int(sum(1 for g in gt_bounds if g <= t + 1e-6))
        panel = np.full((150, W, 3), 255, np.uint8)
        cv2.putText(panel, f't = {t:5.1f} s   (2 frames per second, real Nachi footage)', (20, 40), 0, 1.0, (0, 0, 0), 2)
        cv2.putText(panel, f'home score: {prob:.2f}  -> {"HOME" if prob > 0.5 else "AWAY"}', (20, 85), 0, 1.0, (0, 0, 0), 2)
        cv2.putText(panel, f'cycles counted: {count}    true cycle boundaries passed: {done}', (20, 130), 0, 1.0, (0, 0, 0), 2)
        x0, x1 = 880, 1250; cv2.rectangle(panel, (x0, 62), (x1, 92), (0, 0, 0), 2); cv2.rectangle(panel, (x0, 62), (int(x0 + (x1 - x0) * prob), 92), (0, 0, 0), -1)
        fr = np.vstack([o, panel])
        for _ in range(5): vw.write(fr)
    vw.release()

def signal_plot(t, series, gt_bounds, path):
    W, H, pad = 1200, 170 * len(series), 60; img = np.full((H, W, 3), 255, np.uint8)
    for r, (name, y) in enumerate(series):
        top = r * 170 + 25; h = 120; y = np.asarray(y, float); yn = (y - y.min()) / (np.ptp(y) + 1e-9)
        pts = np.array([[pad + int((W - 2 * pad) * ti / max(t[-1], 1e-6)), top + h - int(h * v)] for ti, v in zip(t, yn)], np.int32)
        cv2.rectangle(img, (pad, top), (W - pad, top + h), (200, 200, 200), 1)
        for g in gt_bounds:
            x = pad + int((W - 2 * pad) * g / max(t[-1], 1e-6)); cv2.line(img, (x, top), (x, top + h), (0, 0, 0), 1)
        cv2.polylines(img, [pts], False, (0, 0, 0), 2)
        cv2.putText(img, name, (pad, top - 6), 0, 0.6, (0, 0, 0), 1)
    cv2.putText(img, 'vertical lines = true cycle boundaries', (pad, H - 8), 0, 0.5, (0, 0, 0), 1)
    cv2.imwrite(path, img)


# ------------------------------------------------------------------ main
def process(vdir, setup, device, log, max_frames=None):
    name = os.path.basename(vdir.rstrip('/'))
    frames = sorted(glob.glob(os.path.join(vdir, 'frames', '*.jpg')))[:max_frames]
    gt = json.load(open(os.path.join(vdir, 'gt.json')))
    n = len(frames); duration = float(gt['total_duration_sec']); f0 = setup.get('init_frame', 0)
    t = np.arange(n) / FPS
    gt_ends = [c['end_time'] for c in gt['cycles']]; gt_internal = gt_ends[:-1]
    log(f'== {name}: {n} frames, {gt["total_cycles"]} true cycles, {duration:.1f} s')
    sam, wb, spf = run_sam(frames, setup, device, log)
    np.savez_compressed(os.path.join(vdir, 'sam_vitb_masks.npz'), workspace_box=wb,
                        **{f'{s}__{k}': (np.packbits(v['masks'].reshape(n, -1), axis=1) if k == 'masks' else v[k]) for s, v in sam.items() for k in ('masks', 'score', 'box')})
    res = dict(video=name, true_cycles=gt['total_cycles'], duration_s=duration, frames=n, sam_s_per_frame=round(spf, 2),
               workspace_box=[round(float(x)) for x in wb], reference_frame=f0,
               ours_rejected_masks=int(sam['ours']['rejected_frames']), ours_rejected_idx=sam['ours']['rejected_idx'], methods={})
    # Abhinav's own CSV from his run (same videos) -> same autocorrelation
    csv = os.path.join(vdir, 'abhinav_sam_motion.csv')
    if os.path.exists(csv):
        import csv as _csv
        rows = list(_csv.DictReader(open(csv)))
        area = [float(r['area']) for r in rows]
        c, per, acv = autocorr_count(area, duration)
        res['methods']['abhinav_original_csv'] = dict(autocorr_count=c, period_s=per, autocorr=round(acv, 3))
    hold = int(round(1.5 * FPS))
    p_ours = None
    for s in ['abhinav', 'abhinav_best', 'ours']:
        M = sam[s]['masks']; area_frac = M.mean((1, 2))
        c_ac, per, acv = autocorr_count(M.sum((1, 2)), duration)
        d = 1 - mask_iou_to(M, M[f0]); p, lo, hi = refdist_prob(d)
        c_lf, dep_lf, he_lf, _ = counter(p, hold)
        F = feat_compute(M, 'grid', wb, np.array(setup['base_point']))
        prb = labelfree_poserac(F, d, lo, hi)
        c_pr, dep_pr, he_pr, _ = counter(prb, hold)
        he_t_lf = [i / FPS for i in he_lf if i < n]; he_t_pr = [i / FPS for i in he_pr if i < n]
        res['methods'][s] = dict(
            mask_area_pct=dict(mean=round(100 * area_frac.mean(), 2), min=round(100 * area_frac.min(), 2), max=round(100 * area_frac.max(), 2)),
            implausible_mask_frames=int(((area_frac < 0.01) | (area_frac > 0.25)).sum()),
            mean_sam_score=round(float(sam[s]['score'].mean()), 3),
            autocorr=dict(count=c_ac, period_s=per, value=round(acv, 3)),
            labelfree_counter=dict(count=c_lf, departures=dep_lf, boundaries=boundary_scores(he_t_lf, gt_internal)),
            labelfree_poserac=dict(count=c_pr, departures=dep_pr, boundaries=boundary_scores(he_t_pr, gt_internal)))
        if s == 'ours': p_ours = p
        log(f'   {s:12s} area {100*area_frac.mean():.1f}% | autocorr {c_ac} (period {per}) | label-free counter {c_lf} (departures {dep_lf}) | label-free PoseRAC {c_pr}')
    res['iou_ours_vs_abhinav_best_mean'] = round(float(np.mean([
        (a & b).sum() / max((a | b).sum(), 1) for a, b in zip(sam['ours']['masks'], sam['abhinav_best']['masks'])])), 3)
    contact_sheet(frames, sam, os.path.join(vdir, 'masks_abhinav_vs_ours.jpg'))
    annotated_video(frames, sam, p_ours, gt_ends, os.path.join(vdir, 'annotated_ours.mp4'))
    d_ours = 1 - mask_iou_to(sam['ours']['masks'], sam['ours']['masks'][f0])
    signal_plot(t, [('Abhinav setup: mask area', sam['abhinav']['masks'].sum((1, 2))),
                    ('ours: mask area', sam['ours']['masks'].sum((1, 2))),
                    ('ours: difference from reference frame (0 = same pose)', d_ours)], gt_ends, os.path.join(vdir, 'signals.png'))
    json.dump(res, open(os.path.join(vdir, 'results.json'), 'w'), indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=os.path.join(HERE, 'real'))
    ap.add_argument('--setup', default=os.path.join(HERE, 'real', 'setup.json')); ap.add_argument('--device', default='mps')
    ap.add_argument('--only', default=None); ap.add_argument('--max_frames', type=int, default=None)
    a = ap.parse_args()
    setups = json.load(open(a.setup)); logf = open(os.path.join(a.root, 'run_log.txt'), 'a')
    def log(msg):
        line = time.strftime('%H:%M:%S ') + msg; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    names = [a.only] if a.only else sorted(setups, key=lambda s: int(s.split('_')[1]))
    allres = []
    for nme in names:
        try:
            allres.append(process(os.path.join(a.root, nme), setups[nme], a.device, log, a.max_frames))
        except Exception as e:
            import traceback; log(f'!! {nme} failed: {e}\n{traceback.format_exc()}')
    json.dump(allres, open(os.path.join(a.root, 'summary.json'), 'w'), indent=1, default=float)
    log('ALL DONE')


if __name__ == '__main__':
    main()
