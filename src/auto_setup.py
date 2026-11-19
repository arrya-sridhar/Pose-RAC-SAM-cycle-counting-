"""(v6.1) Automatic setup: the video in, the tracker's setup out (box around the robot, a point on its base, the home
frame). Nothing else is needed: no box to draw, no click.

All steps look at the first `skim_s` seconds (default 120 s) of the video:
  1. skim    - the video at 6 fps, grey, 320x180 (plus 2 fps colour for pictures)
  2. region  - pixels that keep changing during the skim: the biggest such patch is where the robot works
  3. home    - the first frame (the convention every v6.1 result uses); if the robot is not visible there (a fade-in),
               the first still frame where it is
  4. sketch  - the home frame against the typical "robot away" picture, inside the region: a rough robot shape
  5. outline - SAM ViT-B prompted with the sketch three ways (box + points, points only, box only), 3 answers each;
               keep the answers that cover at least 80% of the sketch and take the one with the best SAM score minus
               twice its share of wall (pixels that look like the background)
  6. base    - the lowest part of the outline that never moves in the skim; box = the outline's bounding box

Tested on 2 NACHI videos only (nachi_1_seq_000: 9/9 cycles, nachi_1_nu05: 7/7; HISTORY.md, v6.1). Known weak spots:
the robot must be the biggest moving thing in the picture and at home (or still) early in the video; the outline can
take in a fixed object right next to the robot (on NACHI, the switch box on the wall). If the setup is wrong, give it
by hand (count_cycles.py --setup) and check setup.jpg.

This is the code that made the two demo videos (v61/explain/stage0/stage0_auto.py, 2 Nov 2026), unchanged in what it
computes; it is now a function, uses the repo's SAM weights and runs on CUDA, Apple Silicon or CPU.

    python src/auto_setup.py <video> <out_dir> [--skim 120] [--device auto|cuda|mps|cpu] [--debug]
writes <out_dir>/setup.json and setup.jpg (with --debug also stage0.npz: every intermediate step)
"""
import os, sys, json, subprocess, argparse, numpy as np, cv2, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_real as RR


def pick_device(device='auto'):
    if device and device != 'auto':
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _read(video, fps, w, h, pix, t, ss=0.0, n=None, ffmpeg='ffmpeg'):
    cmd = [ffmpeg, '-nostdin', '-loglevel', 'error'] + (['-ss', str(ss)] if ss else []) + ['-t', str(t), '-i', video,
           '-vf', f'fps={fps},scale={w}:{h}'] + (['-frames:v', str(n)] if n else []) + ['-pix_fmt', pix, '-f', 'rawvideo', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    c = 1 if pix == 'gray' else 3
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, c)[..., 0] if c == 1 else np.frombuffer(raw, np.uint8).reshape(-1, h, w, c)


def auto_setup(video, device='auto', skim_s=120.0, ffmpeg='ffmpeg', predictor=None):
    """Returns (setup, info). setup has init_box and base_point (1280x720 pixels), init_frame (at 2 fps, as the
    tracker expects) and how each was chosen. info holds the intermediate pictures and masks (for setup.jpg / debugging).
    predictor: an already loaded SamPredictor to reuse (count_cycles.py shares one); otherwise one is loaded."""
    SKIM = float(skim_s)
    read = lambda fps, w, h, pix, t=SKIM, ss=0.0, n=None: _read(video, fps, w, h, pix, t, ss, n, ffmpeg)
    # ---- 1. skim
    G = read(6, 320, 180, 'gray'); FPS = 6.0; n = len(G)
    if n < 3:
        raise RuntimeError(f'could not read enough frames from {video} ({n})')
    COL = read(2, 640, 360, 'rgb24')
    gb = np.array([cv2.GaussianBlur(f, (0, 0), 1.0) for f in G]).astype(np.int16)
    # ---- 2. what moves
    med = np.median(G, axis=0)
    frac = (np.abs(G.astype(np.int16) - med) > 30).mean(0)
    mov = cv2.morphologyEx((frac > 0.02).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    nl, lab, st, _ = cv2.connectedComponentsWithStats(mov.astype(np.uint8))
    if nl < 2:
        raise RuntimeError('nothing moves in the first part of the video: no robot to set up (give the setup by hand)')
    big = 1 + int(np.argmax(st[1:, 4])); R = lab == big
    reg = cv2.dilate(R.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    motion = np.array([0.0] + [float(np.mean(np.abs(gb[i] - gb[i - 1])[reg])) for i in range(1, n)])

    # ---- 3 + 4. home frame and the sketch of the robot there
    def sketch(h):
        dd = np.array([np.mean(np.abs(gb[h] - gb[i])[reg]) for i in range(n)])
        away = np.argsort(-dd)[:int(0.4 * n)]
        B = np.median(gb[away], axis=0)
        s = (np.abs(gb[h] - B) > 25) & reg
        s = cv2.morphologyEx(s.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        k, lb, ss, _ = cv2.connectedComponentsWithStats(s.astype(np.uint8))
        keep = [i for i in range(1, k) if ss[i, 4] >= 100]
        return np.isin(lb, keep), B, dd

    H = 0; S, B, dd = sketch(0); home_rule = 'first frame'
    if S.sum() < 0.15 * R.sum():                       # robot not visible on frame 0 (fade-in): first still frame with a sketch
        for h in [i for i in range(1, n - 1) if motion[i] < 0.5 and motion[i + 1] < 0.5]:
            S2, B2, dd2 = sketch(h)
            if S2.sum() >= 0.15 * R.sum():
                H, S, B, dd, home_rule = h, S2, B2, dd2, 'first still frame where the robot is visible'; break
    if not S.any():
        raise RuntimeError('no robot shape found at home (give the setup by hand)')
    t_home = H / FPS
    dt = cv2.distanceTransform(S.astype(np.uint8), cv2.DIST_L2, 3); cand = np.argwhere(dt >= 2.5)
    if len(cand) == 0:
        cand = np.argwhere(S)
    pts = [cand[np.argmax(dt[cand[:, 0], cand[:, 1]])]]
    for _ in range(4):
        pts.append(cand[np.argmax(np.min([np.hypot(*(cand - p).T) for p in pts], axis=0))])
    pts = np.array(pts)[:, ::-1] * 4.0
    ys, xs = np.where(S); seed_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()]) * 4.0
    # ---- 5. SAM outline
    rgb = read(30, 1280, 720, 'rgb24', t=1.0, ss=t_home, n=1)[0]
    P = predictor if predictor is not None else RR.load_predictor(pick_device(device))
    P.set_image(rgb)
    bx = RR.expand(seed_box, 0.03)
    up = lambda a, it=cv2.INTER_LINEAR: cv2.resize(a, (640, 360), interpolation=it)
    S6 = up(S.astype(np.uint8), cv2.INTER_NEAREST) > 0
    wall = (np.abs(up(gb[H].astype(np.float32)) - up(B.astype(np.float32))) < 12) & (up(frac.astype(np.float32)) > 0.02)
    names, masks, scores, cov, wl = [], [], [], [], []
    for name, kw in [('box + points', dict(box=bx, point_coords=pts, point_labels=np.ones(len(pts)))),
                     ('points only', dict(point_coords=pts, point_labels=np.ones(len(pts)))), ('box only', dict(box=bx))]:
        m, sc, _ = P.predict(multimask_output=True, **kw)
        for j in range(3):
            mm = up(RR.clean_mask(m[j]).astype(np.uint8), cv2.INTER_AREA) > 0
            names.append(f'{name} · answer {j + 1}'); masks.append(mm); scores.append(float(sc[j]))
            cov.append(float((mm & S6).sum() / max(S6.sum(), 1))); wl.append(float((mm & wall).sum() / max(mm.sum(), 1)))
    ok = [i for i in range(9) if cov[i] >= 0.8] or list(range(9))
    pick = max(ok, key=lambda i: scores[i] - 2 * wl[i]); M = masks[pick]
    if not M.any():
        raise RuntimeError('SAM gave an empty outline (give the setup by hand)')
    # ---- 6. base point and box
    static = M & (up(frac.astype(np.float32)) < 0.005)
    k, lb, ss, cen = cv2.connectedComponentsWithStats(static.astype(np.uint8))
    comps = [i for i in range(1, k) if ss[i, 4] >= 50]
    if comps:
        low = max(comps, key=lambda i: ss[i, 1] + ss[i, 3]); sy, sx = np.where(lb == low)
        sel = sy >= np.percentile(sy, 50); base = np.array([sx[sel].mean(), sy[sel].mean()]) * 2.0; base_rule = 'lowest never-moving part'
    else:                                              # nothing static: bottom-centre of the outline
        my, mx = np.where(M); y0 = np.percentile(my, 95); base = np.array([mx[my >= y0].mean(), my[my >= y0].mean()]) * 2.0
        low = 0; base_rule = 'bottom of the outline (no never-moving part found)'
    my, mx = np.where(M); box = np.array([mx.min(), my.min(), mx.max(), my.max()], float) * 2.0
    setup = dict(init_box=[round(float(v), 1) for v in box], base_point=[round(float(v), 1) for v in base],
                 init_frame=int(round(t_home * 2)), home_rule=home_rule, base_rule=base_rule, home_time_s=round(t_home, 2),
                 chosen_outline=names[pick], skim_s=SKIM)
    info = dict(gray=G, col=COL, frac=frac, region=R, motion=motion, B=B.astype(np.uint8), sketch=S, home=H, pts=pts,
                seed_box=seed_box, names=np.array(names), masks=np.array(masks), scores=np.array(scores), cov=np.array(cov),
                wall=np.array(wl), pick=pick, static=static, base_comp=(lb == low) if comps else static, base=base, box=box,
                home_rgb=cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA), home_rgb_full=rgb, mask=M)
    return setup, info


def draw_setup(setup, info, path):
    """setup.jpg: the home frame (1280x720) with the outline (green), the box (yellow) and the base point (red)"""
    img = cv2.cvtColor(np.ascontiguousarray(info['home_rgb_full']), cv2.COLOR_RGB2BGR)
    m = cv2.resize(info['mask'].astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, (60, 220, 60), 3)
    x0, y0, x1, y1 = [int(round(v)) for v in setup['init_box']]
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 220, 255), 2)
    bx, by = [int(round(v)) for v in setup['base_point']]
    cv2.circle(img, (bx, by), 9, (40, 40, 230), -1); cv2.circle(img, (bx, by), 9, (255, 255, 255), 2)
    put_lines(img, ['automatic setup', f"home: {setup['home_rule']} ({setup['home_time_s']} s)",
                    f"outline: {setup['chosen_outline']}", f"base point: {setup['base_rule']}",
                    'green = robot outline, yellow = box, red = base point'])
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def put_lines(img, lines, scale=0.7):
    """white text on a dark band in the top-left corner"""
    font, th, pad, gap = cv2.FONT_HERSHEY_SIMPLEX, 2, 10, 10
    sizes = [cv2.getTextSize(s, font, scale, th)[0] for s in lines]
    w = max(x for x, _ in sizes) + 2 * pad; lh = max(y for _, y in sizes) + gap; h = lh * len(lines) + pad
    band = img[:h, :w].astype(np.float32); img[:h, :w] = (band * 0.35).astype(np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (pad, pad + lh * i + sizes[i][1]), font, scale, (255, 255, 255), th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description='automatic setup for one video')
    ap.add_argument('video'); ap.add_argument('out')
    ap.add_argument('--skim', type=float, default=120.0); ap.add_argument('--device', default='auto')
    ap.add_argument('--debug', action='store_true', help='also save stage0.npz (every intermediate step)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ffmpeg = os.environ.get('FFMPEG', 'ffmpeg')
    setup, info = auto_setup(a.video, a.device, a.skim, ffmpeg)
    json.dump(setup, open(os.path.join(a.out, 'setup.json'), 'w'), indent=1)
    draw_setup(setup, info, os.path.join(a.out, 'setup.jpg'))
    if a.debug:
        np.savez_compressed(os.path.join(a.out, 'stage0.npz'), **{k: v for k, v in info.items() if k not in ('home_rgb_full', 'mask')})
    print(json.dumps(setup))


if __name__ == '__main__':
    main()
