"""Generate one synthetic robot-cycle video (frames at `fps`) with full ground truth.

cfg keys (all optional):
  program: inplane|outplane|pureyaw|wrist|viahome|pallet|continuous
  n_cycles, amp, vmax, speed_jitter, home_dwell, home_dwell_jitter, pause_prob, near_frac, period (continuous)
  appearance: real|lowcon|highcon|darkbg
  clutter: bool (static white fixtures inside the robot's workspace)
  occluder: int (number of person crossings in front of the robot)
  flicker: float (relative global-illumination amplitude)
  shake_px: float (camera shake std in px @1280x720)
  noise: float (gaussian sensor noise std), jpeg: int
  cam: None | dict(eye=..., target=..., fov=...)  (camera placement variation)
  fps, seed
"""
import sys, os, json, numpy as np, cv2, pybullet as p
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import Scene, load_plate, composite, APPEARANCE
from programs import build, Continuous, HOME

GT_W, GT_H = 640, 360


def person_mask(W, H, cx, height_frac=0.95, width_frac=0.11):
    m = np.zeros((H, W), np.uint8)
    h = int(H * height_frac); w = int(W * width_frac)
    top = H - h
    cv2.circle(m, (int(cx), top + int(0.09 * h)), int(0.07 * h), 1, -1)                         # head
    cv2.rectangle(m, (int(cx - w / 2), top + int(0.17 * h)), (int(cx + w / 2), H), 1, -1)     # torso+legs
    cv2.ellipse(m, (int(cx), top + int(0.22 * h)), (int(w * 0.62), int(0.07 * h)), 0, 180, 360, 1, -1)  # shoulders
    return m.astype(bool)


def make_video(cfg, out_dir):
    rng = np.random.default_rng(cfg.get('seed', 0))
    fps = cfg.get('fps', 2.0)
    W, H = 1280, 720
    prog = cfg.get('program', 'inplane')
    if prog == 'continuous':
        traj = Continuous(rng, n_cycles=cfg.get('n_cycles', 8), period=cfg.get('period', 10.0),
                          jitter=cfg.get('speed_jitter', 0.1), amp=cfg.get('amp', 1.0),
                          partial_end=cfg.get('partial_end', False))
        gt = traj.gt
    else:
        traj, gt = build(prog, rng, n_cycles=cfg.get('n_cycles', 8), amp=cfg.get('amp', 1.0), vmax=cfg.get('vmax', 0.55),
                         speed_jitter=cfg.get('speed_jitter', 0.1), home_dwell=cfg.get('home_dwell', 1.5),
                         home_dwell_jitter=cfg.get('home_dwell_jitter', 0.5), pause_prob=cfg.get('pause_prob', 0.0),
                         near_frac=cfg.get('near_frac', 0.12), partial_end=cfg.get('partial_end', False))
    T = gt['duration']
    cam = cfg.get('cam') or {}
    S = Scene(W, H, appearance=cfg.get('appearance', 'real'),
              **({} if not cam else dict(cam_eye=cam['eye'], cam_target=cam['target'], fov=cam.get('fov', 42))))
    clutter_ids = []
    if cfg.get('clutter', False):
        a = S.app['white']
        col = [a, a, a * 0.98, 1]
        for (he, pos) in [([0.07, 0.015, 0.11], [0.16, 0.30, 0.55]),        # white electrical panel behind robot
                          ([0.012, 0.012, 0.42], [-0.33, 0.22, 0.42]),      # white vertical pipe left of robot
                          ([0.09, 0.07, 0.05], [-0.42, 0.05, 0.05])]:       # white parts bin on the left (pick area)
            vs = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=col, physicsClientId=S.cid)
            clutter_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs, basePosition=pos, physicsClientId=S.cid))
    bg0 = load_plate(W, H, gain=S.app['plate_gain'])
    # occluder crossings: (t_start, duration, direction)
    occ = []
    for _ in range(int(cfg.get('occluder', 0))):
        occ.append((rng.uniform(0.1 * T, 0.85 * T), rng.uniform(3.0, 6.0), rng.choice([-1, 1])))
    ph0, ph1 = rng.uniform(0, 2 * np.pi, 2)
    flick = cfg.get('flicker', 0.0)
    shake = cfg.get('shake_px', 0.0)
    noise = cfg.get('noise', 1.5)
    jpeg = int(cfg.get('jpeg', 90))
    os.makedirs(os.path.join(out_dir, 'frames'), exist_ok=True)
    n = int(np.floor(T * fps)) + 1
    ts = np.arange(n) / fps
    Q, masks, boxes, occl_frac = [], [], [], []
    for i, t in enumerate(ts):
        q = traj(t)
        rgb, rm, pm = S.render(q)
        fg = rm | pm      # pedestal + static clutter are rendered but are NOT part of the GT robot mask
        frame = composite(bg0, rgb, fg)
        # illumination: slow drift + occasional steps
        if flick > 0:
            g = 1 + flick * (0.6 * np.sin(2 * np.pi * t / 23.0 + ph0) + 0.4 * np.sin(2 * np.pi * t / 7.3 + ph1))
            frame = frame * g
        vis = rm.copy()
        for (t0, dur, dr) in occ:
            if t0 <= t <= t0 + dur:
                u = (t - t0) / dur
                cx = (0.2 + 0.75 * u) * W if dr > 0 else (0.95 - 0.75 * u) * W
                pm_ = person_mask(W, H, cx)
                shade = np.array([58, 62, 70], np.float32) * (0.9 + 0.2 * rng.random())
                frame[pm_] = shade
                vis &= ~pm_
        if shake > 0:
            dx, dy = rng.normal(0, shake, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            vis = cv2.warpAffine(vis.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
        if noise > 0:
            frame = frame + rng.normal(0, noise, frame.shape)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, 'frames', f'{i:06d}.jpg'), frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        ys, xs = np.where(vis)
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()] if len(xs) else [0, 0, 0, 0])
        occl_frac.append(1 - vis.sum() / max(1, rm.sum()))
        masks.append(np.packbits(cv2.resize(vis.astype(np.uint8), (GT_W, GT_H), interpolation=cv2.INTER_AREA) > 0))
        Q.append(q)
    Q = np.array(Q)
    # per-frame ground truth
    dist_home = np.abs(Q - HOME).max(1)
    cyc = np.full(n, -1)
    for k, (d, a) in enumerate(zip(gt['depart'], gt['arrive'])):
        cyc[(ts >= d) & (ts < a)] = k
    np.savez_compressed(os.path.join(out_dir, 'gt.npz'), t=ts, q=Q, masks=np.array(masks), boxes=np.array(boxes),
                        occl=np.array(occl_frac), dist_home=dist_home, cycle=cyc, depart=gt['depart'],
                        arrive=gt['arrive'], count=gt['count'], duration=T, fps=fps,
                        base_px=S.project([0, 0, 0.0]), shoulder_px=S.project([0, 0, 0.27]))
    meta = dict(cfg=cfg, count=int(gt['count']), duration=float(T), n_frames=int(n),
                periods=np.diff(np.concatenate([[0], gt['arrive']])).round(2).tolist())
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'), indent=1, default=float)
    p.disconnect(S.cid)
    return meta


if __name__ == '__main__':
    cfg = json.loads(sys.argv[1]); out = sys.argv[2]
    print(json.dumps(make_video(cfg, out)))
