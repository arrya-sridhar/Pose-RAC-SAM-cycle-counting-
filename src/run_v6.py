"""(v6) SAM-Box PoseRAC with a dynamic frame rate and the set-up frame as a second memory.

What changes from v5 (src/run_real.py, tag v5), and why:

1. Dynamic frame rate (targets HISTORY 6.2 and the OP75 robot escaping the box).
   Frames are prepared at 30 per second. SAM normally runs on every 15th one (2 per second, the same frames v5 uses).
   Compared with the frame half a second earlier, it speeds up when:
     - the outline changed a lot (overlap below FAST_IOU): the robot is moving fast            -> 6 per second;
     - the outline moved towards or away from the set-up outline by more than HOME_CHANGE while near it (closer than
       HOME_NEAR): the robot is arriving at or leaving home                                    -> 6 per second;
     - no candidate outline was acceptable: the tracker is unsure                               -> 6 per second;
     - the robot is moving fast AND near home: a brief visit home can fall between frames even at 6 per second
       (the simulated palletizing videos: home for ~0.15 s per cycle)                          -> every frame.
   When it speeds up, it goes back and also processes the frames it skipped since the last one, so the start of the
   movement is not lost ("look-back"). The fast rate ends after CALM_S without a trigger, the every-frame rate as soon
   as its condition stops; it slows down only on a frame the slower rate also uses, so with no trigger v6 processes
   exactly v5's frames.
   Everything that v5 counted in frames is now in seconds: recovery after 1 s stuck, the size check's 5 s window,
   1.5 s end padding. The home score is smoothed per processed frame as in v5 (so it reacts faster when the rate is
   higher), and the cut-off is weighted by the time each frame stands for, so fast stretches don't outvote the rest.

2. The set-up frame as a second memory (targets HISTORY 6.1, nachi_4).
   v5 only remembers the last accepted outline, so a part that drops out (nachi_4's base) can never come back: it
   doesn't move, so re-adding it looks exactly like adding a wall. v6 also remembers the set-up frame, which the method
   already assumes shows the robot at home:
     - static-region check: a pixel that was inside the set-up outline and looks the same as in the set-up frame
       (same change threshold as the motion test, 18 grey levels after blurring) is allowed back;
     - the set-up box is always one of the candidate prompts (v5: only after 1 s stuck);
     - candidates are scored by the larger of their overlap with the last outline and with the set-up outline.
   Nothing new is assumed and no new tuned constant is added.

3. Counting extras, measured but not used for the count (Section 7 of HISTORY):
   - warning flags: outline at home much smaller than at the first return (part of the robot lost); many held
     frames. (Count vs duration / repeat period is reported but not flagged: too many false alarms on the 200.)
   - cycle end "at rest": the first frame after the return where the outline stops changing (overlap with the frame
     0.5 s earlier >= REST_IOU), reported next to the v5-style end time;
   - a cut-off from a 120 s window instead of the whole video (count reported as count_window).

Constants were set before any v6 run, from first principles (see comments), and are frozen for the test runs.

v6.1 (30 Oct 2026) changes nothing in how frames are chosen, tracked or counted:
   - videos are decoded while running (VideoSource) instead of first writing every 30 fps frame as a JPEG (the 200
     videos would be about 3 million images). Same ffmpeg settings, minus the JPEG compression; a quick first pass
     gives the motion envelope. "decode": "jpeg" in a job keeps the old way (for comparisons).
   - a per-cycle table (cycles.csv in each run folder, `cycles` in result.json): when the robot leaves home, when it
     is back, duration, time waiting at home, period, frame numbers, and the labelled cycle it matches.
     Departures come from the same trigger that counts (trigger_cycles reproduces poserac.count_official exactly).
   - --shared lets several cluster jobs work through one job list.

    python src/run_v6.py jobs.json --out <folder> --device cuda|mps [--only name ...] [--shared]
    python v6tests/collect_cycles.py <runs folder> --out <folder>      # all_cycles.csv, videos.csv, programs.csv
"""
import os, sys, json, time, glob, argparse, bisect, subprocess, traceback, tempfile, threading, queue, csv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, cv2, torch
import run_real as RR
import poserac as PR

GW, GH = 640, 360
CFG = dict(
    grid_fps=30.0,       # frames are prepared at this rate (the finest level uses all of them)
    rates=(2.0, 6.0, 30.0),   # normal rate (= v5), fast rate, every frame (only when moving fast near home)
    fast_iou=0.7,        # 0.5 s overlap below this -> fast mode. At 6 fps each step is then ~1/3 of the change,
                         # i.e. overlap ~0.9 per step, a change the tracker follows easily.
    home_change=0.1,     # distance to the set-up outline changed by more than this in 0.5 s ...
    home_near=0.5,       # ... while closer than half-way (distance = 1 - overlap with the set-up outline)
    calm_s=1.0,          # back to the normal rate after 1 s without a trigger
    calm_fine_s=0.0,     # every-frame mode only while 'fast near home' holds (it is the expensive one)
    stuck_s=1.0,         # recovery boxes after 1 s with no acceptable outline (v5: 2 frames at 2 fps)
    size_window_s=5.0,   # size check against the median of the last 5 s of accepted outlines (v5: 10 frames at 2 fps)
    end_pad_s=1.5,       # last score repeated for 1.5 s (v5: 3 frames at 2 fps)
    rest_iou=0.9,        # "at rest": overlap with the outline 0.5 s earlier at least this (cycle-end timing only)
    home_band=0.1,       # per-cycle table: home = within 10% of the way from the home level to the farthest point
    window_s=120.0,      # cut-off window for count_window (reported only)
    dynamic=True, memory=True)


# ------------------------------------------------------------------ small helpers
def iou(a, b):
    return float((a & b).sum()) / max(float((a | b).sum()), 1.0)


def small(m):
    return cv2.resize(m.astype(np.uint8), (GW, GH), interpolation=cv2.INTER_AREA) > 0


def wotsu(x, w, bins=64):
    """Otsu's cut-off with weights (identical to evaluate.otsu when all weights are equal)"""
    h, e = np.histogram(x, bins=bins, weights=w); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1e-12); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1e-12)
    return c[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))]


def prepare_frames(video, fdir, fps, start=0, dur=None, ffmpeg='ffmpeg'):
    """frames at a fixed rate, 1280x720, same ffmpeg settings as v5 (so every 3rd frame at 6 fps = v5's 2 fps frame).
    Only used when a job asks for it ("decode": "jpeg"); by default frames are decoded while running (VideoSource)."""
    if len(glob.glob(os.path.join(fdir, '*.jpg'))) == 0:
        os.makedirs(fdir, exist_ok=True)
        cmd = [ffmpeg, '-nostdin', '-loglevel', 'error'] + (['-ss', str(start)] if start else []) + \
              (['-t', str(dur)] if dur else []) + ['-i', video, '-vf', f'fps={fps:g},scale=1280:720', '-q:v', '3',
                                                   '-start_number', '1', os.path.join(fdir, '%06d.jpg')]
        subprocess.run(cmd, check=True)
    return sorted(glob.glob(os.path.join(fdir, '*.jpg')))


class Grid:
    """times of the frames on the fixed grid (k / fps); works for one index or a list of them"""
    def __init__(self, fps):
        self.fps = fps

    def __getitem__(self, k):
        return k / self.fps if np.isscalar(k) else np.asarray(k) / self.fps


class FrameList:
    """frames already on disk (the 2 fps development frames, the edited test videos, or "decode": "jpeg")"""
    def __init__(self, frames):
        self.frames, self.n = frames, len(frames)

    def has(self, k):
        return k < self.n

    def get(self, k):
        return cv2.imread(self.frames[k])

    def envelope(self, step):
        return [cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (320, 180), interpolation=cv2.INTER_AREA)
                for f in self.frames[::step]]

    def close(self):
        pass


def _ffmpeg_cmd(ffmpeg, video, fps, start, dur, pix_fmt, every=1):
    """raw frames on the grid; every=k keeps grid frames 0, k, 2k, ... (ffmpeg's own fps=fps/k would pick the middle
    frame of each step instead)"""
    sel = [f'select=not(mod(n\\,{every}))'] if every > 1 else []
    return [ffmpeg, '-nostdin', '-loglevel', 'error'] + (['-ss', str(start)] if start else []) + \
           (['-t', str(dur)] if dur else []) + ['-i', video, '-vf', ','.join([f'fps={fps:g}'] + sel + ['scale=1280:720'])] + \
           (['-vsync', 'passthrough'] if every > 1 else []) + ['-pix_fmt', pix_fmt, '-f', 'rawvideo', '-']


class VideoSource:
    """Frames of a video on the fixed grid, decoded while the tracker runs, so no frames are written to disk
    (a 30 fps grid for the 200 videos would be about 3 million images). Same ffmpeg settings as the JPEG frames
    (fps filter, scale to 1280x720) minus the JPEG compression. Only the last KEEP frames are held: the look-back never
    goes back more than one normal step (15 frames) and the tracker's caches hold 8. A reader thread keeps ffmpeg
    decoding ahead while SAM runs."""
    W, H, KEEP, AHEAD = 1280, 720, 48, 32

    def __init__(self, video, fps, start=0, dur=None, ffmpeg='ffmpeg'):
        self.args = (ffmpeg, video, fps, start, dur)
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(_ffmpeg_cmd(ffmpeg, video, fps, start, dur, 'bgr24'), stdout=subprocess.PIPE,
                                     stderr=self.err, bufsize=1 << 24)
        self.q = queue.Queue(maxsize=self.AHEAD)
        self.buf, self.next, self.n = {}, 0, None
        self.th = threading.Thread(target=self._read, daemon=True); self.th.start()

    def _read(self):
        size = self.W * self.H * 3
        try:
            while True:
                b = self.proc.stdout.read(size)
                if len(b) < size:
                    break
                self.q.put(np.frombuffer(b, np.uint8).reshape(self.H, self.W, 3))
        finally:
            self.q.put(None)

    def _fill(self, k):
        while self.n is None and self.next <= k:
            f = self.q.get()
            if f is None:
                self.n = self.next
                if self.proc.wait() != 0:
                    self.err.seek(0)
                    raise RuntimeError(f'ffmpeg failed: {self.err.read().decode(errors="replace")[-800:]}')
                break
            self.buf[self.next] = f; self.buf.pop(self.next - self.KEEP, None); self.next += 1

    def has(self, k):
        self._fill(k)
        return self.n is None or k < self.n

    def get(self, k):
        self._fill(k)
        if k not in self.buf:
            raise IndexError(f'frame {k} is no longer held (held {min(self.buf)}..{max(self.buf)})')
        return self.buf[k]

    def envelope(self, step):
        """v5's motion envelope frames (one per normal step, 320x180 grey), from a quick first pass over the video"""
        ffmpeg, video, fps, start, dur = self.args
        p = subprocess.Popen(_ffmpeg_cmd(ffmpeg, video, fps, start, dur, 'bgr24', every=step), stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, bufsize=1 << 24)
        size, out = self.W * self.H * 3, []
        while True:
            b = p.stdout.read(size)
            if len(b) < size:
                break
            g = cv2.cvtColor(np.frombuffer(b, np.uint8).reshape(self.H, self.W, 3), cv2.COLOR_BGR2GRAY)
            out.append(cv2.resize(g, (320, 180), interpolation=cv2.INTER_AREA))
        if p.wait() != 0:
            raise RuntimeError('ffmpeg failed in the first pass')
        return out

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        while self.n is None:                          # let the reader thread finish
            if self.q.get() is None:
                self.n = self.next
        self.th.join(timeout=10); self.proc.wait(); self.err.close(); self.buf = {}


# ------------------------------------------------------------------ the dynamic-rate sampler (works with any tracker)
def need_level(why):
    """0 = normal rate, 1 = fast rate, 2 = every frame. Every frame only when the robot moves fast near home: that is
    the one situation where a brief visit home can fall between two frames even at the fast rate."""
    return 2 if ('home' in why and 'fast' in why) else (1 if why else 0)


def adaptive_sample(n, times, k0, cfg, process, snapshot, restore):
    """Decide which frames to process. process(k) -> dict(mask=<640x360 bool>, ok=bool, ...) and updates the tracker;
    snapshot()/restore(s) save and restore the tracker so frames can be redone after a look-back.
    Frames k are on a grid at cfg['grid_fps']; cfg['rates'] lists the normal, fast and finest rates.
    n is the number of frames, or a function telling whether frame k exists (when decoding while running)."""
    exists = n if callable(n) else (lambda k: k < n)
    steps = [int(round(cfg['grid_fps'] / r)) for r in cfg['rates']]
    b = steps[0]
    out, done = {}, []
    level, t_req, k_prev, k, ref = 0, [-1e9] * len(steps), None, 0, None

    def back(k):                                        # the processed frame half a second (one normal step) earlier
        i = bisect.bisect_right(done, k - b) - 1
        return done[i] if i >= 0 and (k - b) - done[i] < steps[min(1, len(steps) - 1)] else None

    def reasons(k, r):
        kb = back(k)
        if kb is None or k <= k0:
            return []
        a, m, why = out[kb]['mask'], r['mask'], []
        if iou(m, a) < cfg['fast_iou']:
            why.append('fast')
        if ref is not None:
            d1, d0 = 1 - iou(m, ref), 1 - iou(a, ref)
            if abs(d1 - d0) > cfg['home_change'] and min(d1, d0) < cfg['home_near']:
                why.append('home')
        if not r['ok']:
            why.append('unsure')
        return why

    while exists(k):
        snap = snapshot() if cfg['dynamic'] else None
        r = process(k)
        if k == k0:
            ref = r['mask']
        why = reasons(k, r) if cfg['dynamic'] else []
        L = min(need_level(why), len(steps) - 1)
        for l in range(1, L + 1):
            t_req[l] = times[k]
        if L > level and k_prev is not None and k - k_prev > steps[L]:
            restore(snap)                               # look back: redo the skipped frames at the new rate
            for kk in range(k_prev + steps[L], k + 1, steps[L]):
                rr = process(kk); rr['level'] = L; rr['why'] = why if kk == k else []
                out[kk] = rr; done.append(kk)
            level, k_prev = L, k
            k += steps[level]
            continue
        level = max(level, L)
        r['level'] = level; r['why'] = why; out[k] = r; done.append(k)
        calm = [0.0, cfg['calm_s'], cfg['calm_fine_s']]  # the finest rate ends as soon as its condition stops
        new = max(l for l in range(len(steps)) if l == 0 or times[k] - t_req[l] < calm[l] - 1e-6)
        if new < level and k % steps[new] == 0:       # slow down only on a frame the slower rate also uses
            level = new
        k_prev = k
        k += steps[level]
    return out


# ------------------------------------------------------------------ v6 tracker (v5's rules + set-up memory)
class Tracker:
    def __init__(self, src, times, setup, cfg, device, k0, stress=None, log=print):
        self.src, self.times, self.cfg, self.k0, self.log = src, times, cfg, k0, log
        self.P = RR.load_predictor(device)
        self.init_box = np.array(setup['init_box'], float); self.bp = np.array(setup['base_point'], float)
        step = int(round(cfg['grid_fps'] / cfg['rates'][0]))
        self.wb = RR.motion_envelope(src.envelope(step), self.init_box)
        self.S = dict(prev_mask=None, prev_box=self.init_box, areas=[], area0=None, acc_gray=None, last_gray=None,
                      fail_since=None, setup_mask=None, setup_gray=None)
        self.emb, self.img = {}, {}
        self.n_embed, self.n_process, self.t_embed = 0, 0, 0.0
        self.stress, self.stress_done = stress, False

    def snapshot(self):
        s = dict(self.S); s['areas'] = list(self.S['areas']); return (s, self.stress_done)

    def restore(self, snap):
        self.S = dict(snap[0]); self.S['areas'] = list(snap[0]['areas']); self.stress_done = snap[1]

    def _image(self, k):
        if k not in self.img:
            bgr = self.src.get(k)
            self.img = {kk: v for kk, v in self.img.items() if kk > k - 8}
            self.img[k] = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        return self.img[k]

    def _embed(self, k, rgb):
        P = self.P
        if k in self.emb:                                   # reuse the image embedding (the expensive part)
            P.features, P.original_size, P.input_size = self.emb[k]; P.is_image_set = True
            return
        t0 = time.time()
        with torch.inference_mode():
            P.set_image(rgb)
        self.t_embed += time.time() - t0; self.n_embed += 1
        self.emb = {kk: v for kk, v in self.emb.items() if kk > k - 8}
        self.emb[k] = (P.features, P.original_size, P.input_size)

    def process(self, k):
        S, cfg, t = self.S, self.cfg, self.times[k]
        rgb, gray = self._image(k); self._embed(k, rgb); self.n_process += 1
        ref_gray = S['acc_gray'] if S['acc_gray'] is not None else S['last_gray']
        motion = None
        if ref_gray is not None:
            mv = cv2.absdiff(cv2.GaussianBlur(ref_gray, (0, 0), 2), cv2.GaussianBlur(gray, (0, 0), 2)) > 18
            motion = cv2.dilate(mv.astype(np.uint8), RR.K_NEAR) > 0
        first = k <= self.k0 or S['prev_mask'] is None
        stuck = S['fail_since'] is not None and t - S['fail_since'] >= cfg['stuck_s'] - 1e-6
        if first:
            boxes, kinds = [self.init_box], ['start']
        else:
            pb = RR.bbox(S['prev_mask']); pb = S['prev_box'] if pb is None else pb
            near = RR.expand(pb, 0.5)
            mb = RR.motion_box(ref_gray, gray, RR.clip_to(near, self.wb)) if ref_gray is not None else None
            boxes, kinds = [RR.clip_to(RR.expand(pb, 0.08), self.wb)], ['follow']
            if mb is not None:
                boxes.append(RR.clip_to(RR.expand(RR.union(pb, mb), 0.08), self.wb)); kinds.append('follow+motion')
            if stuck or cfg['memory']:                      # v5: set-up box only when stuck; v6: always
                boxes.append(self.init_box); kinds.append('start' if stuck else 'setup')
            if stuck and S['acc_gray'] is not None:
                mall = RR.motion_box(S['acc_gray'], gray, self.wb)
                if mall is not None:
                    boxes.append(RR.clip_to(RR.expand(mall, 0.08), self.wb)); kinds.append('motion')
        cands = []
        with torch.inference_mode():
            for bx, kd in zip(boxes, kinds):
                m, sc, _ = self.P.predict(point_coords=self.bp[None], point_labels=np.array([1]), box=bx, multimask_output=True)
                for j in range(3):
                    cands.append((RR.clean_mask(m[j]), float(sc[j]), bx, kd))
        ok = True
        if first:
            mk, s_k, box, kd = max(cands, key=lambda c: c[1])
        else:
            allowed = (cv2.dilate(S['prev_mask'].astype(np.uint8), RR.K_NEAR) > 0) | (motion if motion is not None else False)
            if cfg['memory'] and S['setup_mask'] is not None:   # robot pixels of the set-up frame that look the same now
                same = cv2.absdiff(cv2.GaussianBlur(S['setup_gray'], (0, 0), 2), cv2.GaussianBlur(gray, (0, 0), 2)) <= 18
                allowed = allowed | (S['setup_mask'] & same)
            recent = [a for (ta, a) in S['areas'] if ta >= t - cfg['size_window_s'] - 1e-6] or [a for (_, a) in S['areas'][-10:]]
            med = float(np.median(recent)); a0 = S['area0']; pm = S['prev_mask']; sm = S['setup_mask']

            def judge(c):
                a = c[0].sum(); u = (c[0] & ~allowed).sum() / max(a, 1)
                ov = (c[0] & pm).sum() / max((c[0] | pm).sum(), 1)
                if cfg['memory'] and sm is not None:
                    ov = max(ov, (c[0] & sm).sum() / max((c[0] | sm).sum(), 1))
                plaus = 0.5 * med <= a <= 2.0 * med and 0.4 * a0 <= a <= 2.0 * a0 and u <= 0.10
                return ov + 0.1 * c[1] - 2.0 * u, plaus
            judged = [(judge(c), c) for c in cands]
            good = [x for x in judged if x[0][1]]
            if good:
                mk, s_k, box, kd = max(good, key=lambda x: x[0][0])[1]
            else:
                _, s_k, box, kd = max(judged, key=lambda x: x[0][0])[1]
                mk, ok = S['prev_mask'], False                  # hold the last good outline
        if k >= self.k0 and ok:
            S['prev_mask'], S['prev_box'], S['acc_gray'] = mk, box, gray
            S['areas'] = (S['areas'] + [(t, mk.sum())])[-40:]      # 5 s at 6 fps is 30 entries
            if S['area0'] is None:
                S['area0'] = mk.sum()
            if S['setup_mask'] is None:
                S['setup_mask'], S['setup_gray'] = mk, gray
        S['fail_since'] = None if ok else (S['fail_since'] if S['fail_since'] is not None else t)
        S['last_gray'] = gray
        out_mask = small(mk)
        # stress test: remove the lower third of the outline once, as if the tracker had lost part of the robot
        if self.stress and not self.stress_done and t >= self.stress['t'] and k > self.k0 and S['prev_mask'] is not None:
            m = S['prev_mask'].copy(); ys = np.where(m.any(1))[0]
            cut = int(ys.max() - (ys.max() - ys.min()) / 3); m[cut:] = False
            S['prev_mask'] = m; S['areas'] = (S['areas'] + [(t, m.sum())])[-40:]; self.stress_done = True
            self.log(f'    stress: removed the lower third of the outline at {t:.1f} s')
        return dict(mask=out_mask, ok=ok, kind=kd, score=float(s_k), box=[float(x) for x in box])


# ------------------------------------------------------------------ counting on unevenly spaced frames
def count_samples(t, M, i0, cfg, gt_ends=None):
    n = len(t)
    d = 1 - np.array([iou(m, M[i0]) for m in M])
    dt = np.diff(t); dt = np.append(dt, dt[-1] if len(dt) else 0.5)
    tau = wotsu(d, dt); lo, hi = tau / 2, (tau + d.max()) / 2
    p = np.clip((hi - d) / max(hi - lo, 1e-6), 0, 1)
    pad = int(round(cfg['end_pad_s'] / max(dt[-1], 1e-6)))
    c, ev, he = PR.count_official(np.concatenate([p, np.repeat(p[-1:], pad)]))
    he = [i for i in he if i < n]
    # window cut-off (reported only)
    pw = np.empty(n)
    for i in range(n):
        sel = np.abs(t - t[i]) <= cfg['window_s'] / 2
        tw = wotsu(d[sel], dt[sel]); lw, hw = tw / 2, (tw + d[sel].max()) / 2
        pw[i] = np.clip((hw - d[i]) / max(hw - lw, 1e-6), 0, 1)
    cw, _, _ = PR.count_official(np.concatenate([pw, np.repeat(pw[-1:], pad)]))
    # cycle end at rest: first frame from the return on whose outline matches the one 0.5 s earlier
    rest = []
    for i in he:
        e = None
        for j in range(i, n):
            if t[j] - t[i] > 5.0:
                break
            jb = int(np.argmin(np.abs(t - (t[j] - 0.5))))
            if t[j] - t[jb] >= 0.4 and iou(M[j], M[jb]) >= cfg['rest_iou']:
                e = t[j]; break
        rest.append(round(float(e if e is not None else t[i]), 2))
    res = dict(count=int(c), count_window=int(cw), tau=round(float(tau), 4), returns_s=[round(float(t[i]), 2) for i in he],
               rest_end_s=rest)
    cyc, c2 = trigger_cycles(np.concatenate([p, np.repeat(p[-1:], pad)]))
    if c2 != c:                                          # cannot happen unless poserac.count_official changes
        raise RuntimeError(f'cycle table out of step with the count ({c2} vs {c})')
    res['cycles'] = cycle_rows(t, d, cyc, i0, cfg)
    return res, d, p


# ------------------------------------------------------------------ per-cycle times (v6.1)
def trigger_cycles(p, enter=0.78, exit=0.4, momentum=0.4):
    """poserac.count_official's loop, step for step, also recording when the robot leaves home. Returns one
    [leave, event, home] per counted cycle (indices into p; None if not seen) and the count:
      leave = first frame whose smoothed home score is below `enter` again before the departure trigger fired,
      event = the frame the cycle was counted on,
      home  = first frame above `enter` after the return (v5's return time, `returns_s`).
    A departure after the last counted cycle with no return is added as [leave, None, None] (not counted)."""
    r1, r2 = PR.ActionTrigger(enter, exit), PR.ActionTrigger(enter, exit)
    prob, count, curr, init = 0.5, 0, 'holder', 'pose_holder'
    cycles, was_in1, last_hi, leave, open_leave = [], False, None, None, None
    for i, o in enumerate(p):
        prob = o * (1. - momentum) + momentum * prob
        if prob > enter:
            last_hi = i
        s1 = r1(prob); s2 = r2(1 - prob)
        if s1:                                           # departure confirmed: it started when the score left `enter`
            leave = last_hi + 1 if last_hi is not None else i; open_leave = leave
        if init == 'pose_holder':
            if s1: init = 'salient1'
            elif s2: init = 'salient2'
        if init == 'salient1':
            if curr == 'salient1' and s2:
                count += 1; cycles.append([leave, i, None]); open_leave = None
        else:                                            # video starts away from home: counted when it leaves again
            if curr == 'salient2' and s1:
                count += 1; cycles.append([leave, i, None]); open_leave = None
        if s1: curr = 'salient1'
        elif s2: curr = 'salient2'
        if r1.entered and not was_in1 and curr == 'salient2' and cycles and cycles[-1][2] is None:
            cycles[-1][2] = i
        was_in1 = r1.entered
    if open_leave is not None:
        cycles.append([open_leave, None, None])
    return cycles, count


def cycle_rows(t, d, cyc, i0, cfg):
    """One row per cycle (times in seconds from the start of the run), from d = distance to the set-up outline.
    Between two cycles the robot is home: from the farthest point of one cycle to the departure of the next, the home
    level is the closest approach (lowest d), and the robot counts as home while d is within HOME_BAND (10%) of the way
    from that level to the farthest point. Then
      end   = first frame home after the cycle (it has arrived),
      start = last frame home before the cycle (it is about to leave);
      leave / return = when the smoothed home score crosses 0.78 (v5's times);
      duration = end - start (moving), wait = next start - end (at home), period = next start - start.
    The distance is used, not "has the outline stopped changing": a robot that moves slowly far from home (nachi_4)
    changes its outline as little as a robot at rest. A robot that never stops at home gets start = end = its closest
    approach."""
    n, band = len(t), cfg['home_band']
    far = []
    for lv, ev, hm in cyc:
        r = hm if hm is not None and hm < n else (ev if ev is not None and ev < n else None)
        a = min(lv, n - 1) if lv is not None else i0
        b = r if r is not None else n - 1
        far.append(a + int(np.argmax(d[a:b + 1])) if b >= a else a)

    def stay(a, b, ref):
        """first and last frame home in [a, b]; ref = the farthest distances on either side"""
        if b < a:
            return None, None
        m = float(d[a:b + 1].min()); lim = m + band * (max(ref) - m)
        idx = np.where(d[a:b + 1] <= lim + 1e-9)[0] + a
        return int(idx[0]), int(idx[-1])

    rows = []
    for c, (lv, ev, hm) in enumerate(cyc):
        r = hm if hm is not None and hm < n else (ev if ev is not None and ev < n else None)
        row = dict(cycle=c + 1, start_s=None, end_s=None, leave_s=None, return_s=None, complete=True, note='')
        if lv is not None:
            lv = min(lv, n - 1)
            a = far[c - 1] if c > 0 else i0
            _, dep = stay(a, lv, [d[far[c]]] + ([d[far[c - 1]]] if c > 0 else []))
            row.update(start_s=float(t[dep if dep is not None else lv]), leave_s=float(t[lv]))
        if r is not None:
            nxt = cyc[c + 1][0] if c + 1 < len(cyc) and cyc[c + 1][0] is not None else n - 1
            arr, _ = stay(far[c], min(nxt, n - 1), [d[far[c]]] + ([d[far[c + 1]]] if c + 1 < len(cyc) else []))
            row.update(end_s=float(t[arr if arr is not None else r]), return_s=float(t[r]))
        if ev is None:
            row.update(complete=False, note='left home, video ends before it is back (not counted)')
        elif r is None:
            row.update(end_s=float(t[-1]), complete=False, note='counted; back home only at the very end of the video')
        elif lv is None:
            row.update(complete=False, note='departure not seen')
        rows.append(row)
    for x, y in zip(rows, rows[1:]):
        if x['end_s'] is not None and y['start_s'] is not None:
            x['wait_s'] = y['start_s'] - x['end_s']
        if x['start_s'] is not None and y['start_s'] is not None:
            x['period_s'] = y['start_s'] - x['start_s']
    for x in rows:
        if x['start_s'] is not None and x['end_s'] is not None:
            x['duration_s'] = x['end_s'] - x['start_s']
    return rows


def gt_cycles(g):
    """labelled cycles as (start, end) in seconds. NACHI sets: start_time/end_time; palletizing: *_time_sec"""
    return [(float(c.get('start_time', c.get('start_time_sec'))), float(c.get('end_time', c.get('end_time_sec'))))
            for c in g['cycles']]


def match_cycles(rows, gts, tol=1.5):
    """Match our cycles one-to-one to the labelled ones by end time. A constant shift per video is removed for the
    matching only (the labels of some programs are offset: nachi_2 about +12 s, nachi_8 about -11 s); the errors
    written to the table are raw (ours - label). Returns {row index: label index} and the shift."""
    ours = [(i, r['end_s']) for i, r in enumerate(rows) if r['end_s'] is not None]     # every counted cycle
    if not ours or not gts:
        return {}, 0.0
    oe, ge = np.array([e for _, e in ours]), np.array([e for _, e in gts])
    L = float(np.median([e - s for s, e in gts]))
    best = (-1, 0.0)
    for s in sorted(np.arange(-L, L + 1e-9, 0.1), key=abs):         # most ends within tol; smallest shift wins ties
        k = int((np.abs(oe[:, None] - (ge[None, :] + s)).min(1) <= tol).sum())
        if k > best[0]:
            best = (k, float(s))
    s = best[1]
    near = [oe[a] - ge[b] for a, b in enumerate(np.abs(oe[:, None] - (ge[None, :] + s)).argmin(1))
            if abs(oe[a] - ge[b] - s) <= tol]
    s = float(np.median(near)) if near else s
    pairs = sorted((abs(oe[a] - ge[b] - s), a, b) for a in range(len(oe)) for b in range(len(ge)) if abs(oe[a] - ge[b] - s) <= L / 2)
    m, used = {}, set()
    for _, a, b in pairs:
        if ours[a][0] not in m and b not in used:
            m[ours[a][0]] = b; used.add(b)
    return m, s


CYCLE_COLS = ['video', 'cycle', 'start_s', 'end_s', 'duration_s', 'wait_s', 'period_s', 'start_frame', 'end_frame',
              'leave_s', 'return_s', 'complete', 'note', 'label_cycle', 'label_start_s', 'label_end_s', 'start_err_s',
              'end_err_s']


def cycle_table(rows, offset, fps, gt=None):
    """rows in video time, frame numbers, and the labelled cycle each one matches (if labels are given).
    Returns the table rows and a timing summary."""
    out = []
    for r in rows:
        o = dict(r)
        for k in ('start_s', 'end_s', 'leave_s', 'return_s'):
            if o.get(k) is not None:
                o[k] = o[k] + offset
        o['start_frame'] = int(round(o['start_s'] * fps)) if o.get('start_s') is not None else None
        o['end_frame'] = int(round(o['end_s'] * fps)) if o.get('end_s') is not None else None
        out.append(o)
    summ = {}
    if gt is not None:
        gts = gt_cycles(gt); m, shift = match_cycles(out, gts)
        for i, b in m.items():
            o = out[i]; o.update(label_cycle=b + 1, label_start_s=gts[b][0], label_end_s=gts[b][1],
                                 end_err_s=o['end_s'] - gts[b][1])
            if o.get('start_s') is not None:
                o['start_err_s'] = o['start_s'] - gts[b][0]
        es = [o['end_err_s'] for o in out if o.get('end_err_s') is not None]
        ss = [o['start_err_s'] for o in out if o.get('start_err_s') is not None]
        med = lambda x: round(float(np.median(x)), 2) if len(x) else None
        summ = dict(labelled=len(gts), matched=len(m), missed=len(gts) - len(m),
                    extra=sum(1 for o in out if o.get('end_s') is not None and o.get('label_cycle') is None), shift_s=round(shift, 2),
                    end_err_median_s=med(es), end_abs_err_median_s=med(np.abs(es)), end_within_1s=sum(abs(e) <= 1 for e in es),
                    start_err_median_s=med(ss), start_abs_err_median_s=med(np.abs(ss)), start_within_1s=sum(abs(e) <= 1 for e in ss))
    for o in out:
        for k, v in list(o.items()):
            if isinstance(v, float):
                o[k] = round(v, 2)
    return out, summ


def write_cycles_csv(path, name, rows, append=False):
    with open(path, 'a' if append else 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CYCLE_COLS, extrasaction='ignore')
        if not append or f.tell() == 0:
            w.writeheader()
        for r in rows:
            w.writerow(dict(r, video=name))


def flags(t, M, d, he_times, ok, count, duration):
    f = []
    # outline at home vs the first return: peak overlap with the set-up outline in the 3 s after each return
    peaks = [float(1 - d[(t >= x) & (t <= x + 3.0)].min()) for x in he_times if ((t >= x) & (t <= x + 3.0)).any()]
    if len(peaks) >= 2 and min(peaks[1:]) < 0.85 * peaks[0]:
        f.append(f'outline at home smaller than at the first return (overlap {min(peaks[1:]):.2f} vs {peaks[0]:.2f}): '
                 'part of the robot may have been lost')
    # count vs duration / repeat period (area resampled to 2 per second)
    tg = np.arange(0, t[-1] + 1e-9, 0.5); area = np.interp(tg, t, M.reshape(len(M), -1).mean(1))
    ac, per, _ = RR.autocorr_count(area, duration)
    # Reported, not flagged: on the 200 videos the repeat-period estimate disagreed on 32 videos whose count was right
    # (it fails on long and uneven videos), so as a flag it would mostly send people to check correct counts.
    disagree = bool(per is not None and abs(ac - count) > 1)
    held = 1 - float(np.mean(ok))
    if held > 0.05:
        f.append(f'{100 * held:.0f}% of frames held (no acceptable outline): tracking unreliable')
    return f, dict(home_overlap_at_returns=[round(x, 3) for x in peaks], autocorr_count=ac, count_vs_period_disagree=disagree,
                   held_share=round(held, 4))


def gt_ends_of(gt):
    return [c.get('end_time', c.get('end_time_sec')) for c in gt['cycles']]


# ------------------------------------------------------------------ sharing one job list between cluster jobs
def _alive(owner):
    """is the cluster job that claimed a video still running? (unknown -> assume it is)"""
    if not owner.isdigit():
        return True
    try:
        return bool(subprocess.run(['squeue', '-h', '-j', owner], capture_output=True, text=True, timeout=60).stdout.strip())
    except Exception:
        return True


def claim(od):
    """take a video for this cluster job: True if it is ours to run. A claim left by a job that is no longer running
    is taken over, so a stopped job can simply be resubmitted."""
    f, me = os.path.join(od, 'claim.txt'), os.environ.get('SLURM_JOB_ID', f'pid{os.getpid()}')
    for _ in range(2):
        try:
            fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, me.encode()); os.close(fd)
            return True
        except FileExistsError:
            owner = open(f).read().strip()
            if owner == me:
                return True
            if _alive(owner):
                return False
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
    return False


# ------------------------------------------------------------------ one job
def run_job(job, out_root, device, log, ffmpeg, shared=False):
    cfg = dict(CFG, **job.get('cfg', {}))
    od = os.path.join(out_root, job['name']); os.makedirs(od, exist_ok=True)
    if os.path.exists(os.path.join(od, 'result.json')):
        log(f"skip {job['name']} (done)"); return json.load(open(os.path.join(od, 'result.json')))
    if shared and not claim(od):
        log(f"skip {job['name']} (another job has it)"); return None
    t0 = time.time()
    step = int(round(cfg['grid_fps'] / cfg['rates'][0]))
    if job.get('frames_dir'):
        src, how = FrameList(sorted(glob.glob(os.path.join(job['frames_dir'], '*.jpg')))), 'frames on disk'
    elif job.get('decode') == 'jpeg':                    # the old way, all frames written as JPEGs first (A/B tests)
        src, how = FrameList(prepare_frames(job['video'], os.path.join(job.get('frames_root', od), 'frames'), cfg['grid_fps'],
                                            job.get('start', 0), job.get('dur'), ffmpeg)), 'JPEG frames'
    else:
        src, how = VideoSource(job['video'], cfg['grid_fps'], job.get('start', 0), job.get('dur'), ffmpeg), 'decoded while running'
    try:
        times = Grid(cfg['grid_fps'])
        setup = job['setup']
        k0 = int(round(setup.get('init_frame', 0) / 2.0 * cfg['grid_fps']))          # init_frame is given at 2 fps (v5)
        log(f"== {job['name']}: frames at {cfg['grid_fps']:g} fps, {how}; dynamic={cfg['dynamic']} memory={cfg['memory']} "
            f"stress={job.get('stress')}  (prep {time.time() - t0:.0f} s)")
        tr = Tracker(src, times, setup, cfg, device, k0, job.get('stress'), log)
        t1 = time.time()
        out = adaptive_sample(src.has, times, k0, cfg, tr.process, tr.snapshot, tr.restore)
        n = src.n
    finally:
        src.close()
    ks = sorted(out); t = times[ks]; M = np.array([out[k]['mask'] for k in ks]); ok = np.array([out[k]['ok'] for k in ks])
    level = np.array([out[k]['level'] for k in ks]); dense = level > 0; i0 = ks.index(k0) if k0 in ks else 0
    cres, d, p = count_samples(t, M, i0, cfg)
    duration = float(n / cfg['grid_fps'])
    fl, finfo = flags(t, M, d, cres['returns_s'], ok, cres['count'], duration)
    whys = {}
    for k in ks:
        for w in out[k]['why']:
            whys[w] = whys.get(w, 0) + 1
    # share of the video's time spent in fast mode
    dt = np.diff(np.append(t, t[-1] + 1 / cfg['rates'][0]))
    g = json.load(open(job['gt'])) if job.get('gt') else None
    rows, timing = cycle_table(cres.pop('cycles'), job.get('start', 0), cfg['grid_fps'], g)
    res = dict(name=job['name'], version='v6.1', program=job.get('program'), cfg={k: cfg[k] for k in CFG},
               video=os.path.basename(job.get('video', job.get('frames_dir', ''))), frames_from=how,
               start_s=job.get('start', 0), frames_available=n, frames_processed=len(ks), base_rate_frames=int(np.ceil(n / step)),
               sam_embeddings=tr.n_embed, sam_calls=tr.n_process, fast_time_share=round(float(dt[dense].sum() / dt.sum()), 3),
               every_frame_time_share=round(float(dt[level == 2].sum() / dt.sum()), 3),
               frames_by_level=[int((level == l).sum()) for l in range(len(cfg['rates']))],
               triggers=whys, held_frames=int((~ok[t >= t[i0]]).sum()), outline_area_pct_mean=round(100 * float(M.mean()), 2),
               s_per_embedding=round(tr.t_embed / max(tr.n_embed, 1), 3), runtime_s=round(time.time() - t0, 1),
               track_s=round(time.time() - t1, 1), flags=fl, flag_info=finfo, stress=job.get('stress'), **cres)
    if g is not None:
        true = int(g['total_cycles']); ends = gt_ends_of(g)
        res.update(true_cycles=true, exact=bool(cres['count'] == true), obo=bool(abs(cres['count'] - true) <= 1),
                   nmae=round(abs(cres['count'] - true) / true, 4), window_exact=bool(cres['count_window'] == true),
                   ends_v5style=RR.boundary_scores(cres['returns_s'], ends[:-1]),
                   ends_at_rest=RR.boundary_scores(cres['rest_end_s'], ends[:-1]), timing=timing,
                   gt_file=job['gt'], profile=g.get('profile'))
    res['cycles'] = rows
    write_cycles_csv(os.path.join(od, 'cycles.csv'), job['name'], rows)
    np.savez_compressed(os.path.join(od, 'trace.npz'), t=t, d=d, p=p, ok=ok, level=level, i0=i0,
                        masks=np.packbits(M.reshape(len(M), -1), axis=1))
    json.dump(res, open(os.path.join(od, 'result.json'), 'w'), indent=1, default=float)
    tm = (f" | ends within 1 s {timing['end_within_1s']}/{timing['matched']}" if timing else '')
    log(f"   done {job['name']}: count {res['count']}" + (f" / true {res.get('true_cycles')}" if g is not None else '') + tm +
        f" | processed {len(ks)} of {n} ({len(ks) / max(res['base_rate_frames'], 1):.2f}x v5) | fast {100 * res['fast_time_share']:.0f}% of time"
        f" | held {res['held_frames']} | {res['s_per_embedding']:.2f} s/embedding | {res['runtime_s']:.0f} s" + (f" | FLAGS: {len(fl)}" if fl else ''))
    del tr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('jobs'); ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--shared', action='store_true', help='several cluster jobs work through the same list')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); logf = open(os.path.join(a.out, 'run_log.txt'), 'a')
    def log(m):
        line = time.strftime('%H:%M:%S ') + m; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    ffmpeg = os.environ.get('FFMPEG', 'ffmpeg')
    jobs = json.load(open(a.jobs))
    if a.only:
        jobs = [j for j in jobs if j['name'] in a.only]
    log(f'v6: {len(jobs)} jobs on {a.device}')
    for j in jobs:
        try:
            run_job(j, a.out, a.device, log, ffmpeg, a.shared)
        except Exception as e:
            log(f"!! {j['name']} failed: {e}\n{traceback.format_exc()}")
    log('ALL DONE')


if __name__ == '__main__':
    main()
