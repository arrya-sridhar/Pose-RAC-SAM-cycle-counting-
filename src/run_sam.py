"""Run SAM box prompting over a video's frames with several prompt strategies sharing one image embedding per frame.
Saves sam_<model>.npz in the video dir: per strategy packed 640x360 masks, chosen scores, boxes used."""
import sys, os, json, time, argparse, warnings, numpy as np, cv2, torch
WEIGHTS = os.environ.get('SAM_WEIGHTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights'))
warnings.filterwarnings('ignore')
GW, GH = 640, 360

def load_predictor(model):
    if model == 'vitb':
        from segment_anything import sam_model_registry, SamPredictor
        return SamPredictor(sam_model_registry['vit_b'](checkpoint=os.path.join(WEIGHTS, 'sam_b.pt')).eval())
    import mobile_sam
    return mobile_sam.SamPredictor(mobile_sam.sam_model_registry['vit_t'](checkpoint=os.path.join(WEIGHTS, 'mobile_sam.pt')).eval())

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
    if not keep:
        return None
    return bbox(np.isin(lab, keep))

STRATS = ['broad', 'broad_pt', 'broad_pt_small', 'oracle', 'track', 'track_pt', 'track_motion_pt']

def run(video_dir, model='mobile', strategies=STRATS, max_frames=None):
    gt = np.load(os.path.join(video_dir, 'gt.npz'))
    frames = sorted(os.listdir(os.path.join(video_dir, 'frames')))[:max_frames]
    N = len(frames)
    gtb = gt['boxes'][:N].astype(float)
    wb = expand(np.array([gtb[:, 0].min(), gtb[:, 1].min(), gtb[:, 2].max(), gtb[:, 3].max()]), 0.03)   # user-drawn workspace box
    b0 = expand(gtb[0], 0.03)                                                                             # user-drawn robot box on frame 0
    bp = gt['base_px'] + 0.45 * (gt['shoulder_px'] - gt['base_px'])                                        # user-clicked point on robot base
    P = load_predictor(model)
    out = {s: dict(masks=[], score=[], box=[], cand_scores=[]) for s in strategies}
    state = {s: dict(prev_mask=None, prev_box=b0.copy(), prev_area=None) for s in strategies}
    prev_gray = None
    t0 = time.time()
    for i, f in enumerate(frames):
        img = cv2.imread(os.path.join(video_dir, 'frames', f))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with torch.inference_mode():
            P.set_image(rgb)
            mb = motion_box(prev_gray, gray, wb) if prev_gray is not None else None
            for s in strategies:
                st = state[s]; pts = lbl = None
                if s.startswith('broad'):
                    box = wb
                elif s == 'oracle':
                    box = expand(gtb[i], 0.03)
                else:
                    if st['prev_mask'] is None:
                        box = b0
                    else:
                        pb = bbox(st['prev_mask']); pb = st['prev_box'] if pb is None else pb
                        if s.startswith('track_motion') and mb is not None:
                            pb = union(pb, mb)
                        box = clip_to(expand(pb, 0.08 if s.startswith('track_motion') else 0.12), wb)
                if '_pt' in s:
                    pts, lbl = bp[None, :], np.array([1])
                m, sc, _ = P.predict(point_coords=pts, point_labels=lbl, box=box, multimask_output=True)
                k = int(np.argmax(sc))
                if s.endswith('_small'):   # smallest candidate that contains the clicked base point
                    ar = m.mean((1, 2)); px, py = int(bp[0]), int(bp[1])
                    ok = [j for j in range(3) if m[j][py, px] and ar[j] > 0.005]
                    if ok: k = min(ok, key=lambda j: ar[j])
                mk = m[k]
                # sanity fallback for tracked strategies: implausible area jump -> retry with a wider box
                if s.startswith('track') and st['prev_area'] is not None:
                    a = mk.sum() / max(st['prev_area'], 1)
                    if a < 0.4 or a > 2.5:
                        box2 = clip_to(expand(box, 0.25), wb)
                        m2, sc2, _ = P.predict(point_coords=pts, point_labels=lbl, box=box2, multimask_output=True)
                        k2 = int(np.argmax(sc2)); a2 = m2[k2].sum() / max(st['prev_area'], 1)
                        if abs(np.log(max(a2, 1e-3))) < abs(np.log(max(a, 1e-3))):
                            m, sc, k, mk, box = m2, sc2, k2, m2[k2], box2
                st['prev_mask'] = mk; st['prev_box'] = box; st['prev_area'] = mk.sum()
                small = cv2.resize(mk.astype(np.uint8), (GW, GH), interpolation=cv2.INTER_AREA) > 0
                out[s]['masks'].append(np.packbits(small)); out[s]['score'].append(float(sc[k]))
                out[s]['box'].append(box.astype(np.float32)); out[s]['cand_scores'].append(sc.astype(np.float32))
        prev_gray = gray
        if i % 25 == 0:
            print(f'{video_dir} {i}/{N} {(time.time()-t0)/(i+1):.2f}s/frame', flush=True)
    save = {}
    for s in strategies:
        for k, v in out[s].items():
            save[f'{s}__{k}'] = np.array(v)
    save['workspace_box'] = wb; save['init_box'] = b0; save['base_point'] = bp
    np.savez_compressed(os.path.join(video_dir, f'sam_{model}.npz'), **save)
    return (time.time() - t0) / N

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('video_dir'); ap.add_argument('--model', default='mobile')
    ap.add_argument('--max_frames', type=int, default=None); a = ap.parse_args()
    print('done', run(a.video_dir, a.model, max_frames=a.max_frames))
