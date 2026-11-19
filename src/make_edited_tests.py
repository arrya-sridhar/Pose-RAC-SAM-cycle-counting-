"""Build a fresh test set from the 10 real Nachi one-cycle clips: cycles that are NOT identical copies.

For each robot program (clip), the robot does 3-4 cycles back to back, and each cycle is changed the way real
production changes it:
  * speed: each cycle runs at 0.8x, 1.0x or 1.25x speed (random per cycle) -> different frames every cycle
  * waiting at home: after each cycle the robot waits at home 0-15 s (random), 1-3 s after the last one
  * a mid-cycle stop in about half the videos: the robot freezes somewhere in one cycle for 3-8 s
Frames are written at 2 per second, 1280x720, like the frames the pipeline already uses.  Ground truth (cycle count,
start/end times, pauses) is exact because we build the video.  The plan is drawn from a fixed random seed BEFORE any
method is run on it, and neither method is tuned on these videos afterwards.

nachi_9 is left out: its clip starts with a fade-in and ends mid-motion, so back-to-back copies would jump.
Pauses are frozen frames (no sensor noise), which is the one unrealistic part.
"""
import os, json, random, shutil, sys
import cv2

SRC = os.environ.get('NACHI_CLIPS', '/home/psaxena/Harshitha/nachi_robot_videos_trim')   # folder with nachi_1.mp4 ... nachi_10.mp4
OUT = sys.argv[1] if len(sys.argv) > 1 else '/scratch/psaxena/arrya_edited_test'
JOBS = [1, 2, 3, 4, 5, 6, 7, 8, 10]
SPEEDS = [0.8, 1.0, 1.25]
STEP = 1.0 / float(os.environ.get('EDIT_FPS', '2'))   # 2 frames per second (EDIT_FPS=30 for v6; same plan, same seed)
rng = random.Random(20260925)

os.makedirs(OUT, exist_ok=True)
plans = {}
for k in JOBS:
    n = rng.choice([3, 4])
    speeds = [rng.choice(SPEEDS) for _ in range(n)]
    waits = [round(rng.uniform(0, 15), 1) for _ in range(n - 1)] + [round(rng.uniform(1, 3), 1)]
    mid = None
    if rng.random() < 0.5:
        mid = dict(cycle=rng.randrange(n), at_fraction=round(rng.uniform(0.3, 0.7), 2), seconds=round(rng.uniform(3, 8), 1))
    plans[k] = dict(cycles=n, speeds=speeds, wait_after_each_cycle_s=waits, mid_cycle_stop=mid)

for k in JOBS:
    p = plans[k]
    cap = cv2.VideoCapture(os.path.join(SRC, f'nachi_{k}.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); D = (N - 1) / fps
    # piecewise timeline: list of (out_start, out_end, src_start, src_rate) ; src time = src_start + rate*(t - out_start)
    segs, t, cycles = [], 0.0, []
    for c in range(p['cycles']):
        v = p['speeds'][c]; start = t
        m = p['mid_cycle_stop']
        if m and m['cycle'] == c:
            s_stop = m['at_fraction'] * D
            segs.append((t, t + s_stop / v, 0.0, v)); t += s_stop / v
            segs.append((t, t + m['seconds'], s_stop, 0.0)); t += m['seconds']
            segs.append((t, t + (D - s_stop) / v, s_stop, v)); t += (D - s_stop) / v
        else:
            segs.append((t, t + D / v, 0.0, v)); t += D / v
        cycles.append(dict(cycle_index=c + 1, start_time=round(start, 3), end_time=round(t, 3), speed=v))
        w = p['wait_after_each_cycle_s'][c]
        segs.append((t, t + w, D, 0.0)); t += w
    total = t
    times = [j * STEP for j in range(int(total / STEP) + 1) if j * STEP <= total]
    def src_time(tt):
        for a, b, s0, r in segs:
            if a <= tt < b or (tt == b == total):
                return min(D, s0 + r * (tt - a))
        return D
    idx = [min(N - 1, int(round(src_time(tt) * fps))) for tt in times]
    need = sorted(set(idx))
    vdir = os.path.join(OUT, f'edit_{k}'); fdir = os.path.join(vdir, 'frames'); cache = os.path.join(vdir, '_src')
    os.makedirs(fdir, exist_ok=True); os.makedirs(cache, exist_ok=True)
    want, i = set(need), 0
    while True:
        ok, fr = cap.read()
        if not ok or i > need[-1]:
            break
        if i in want:
            cv2.imwrite(os.path.join(cache, f'{i:06d}.jpg'), cv2.resize(fr, (1280, 720), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        i += 1
    cap.release()
    for j, s in enumerate(idx):
        shutil.copyfile(os.path.join(cache, f'{s:06d}.jpg'), os.path.join(fdir, f'{j + 1:06d}.jpg'))
    shutil.rmtree(cache)
    gt = dict(video_id=f'edit_{k}', source_clip=f'nachi_{k}.mp4', built_from='real clip, edited (speed changes + waits)',
              fps=round(1.0 / STEP, 3), total_duration_sec=round(len(times) * STEP, 3), total_cycles=p['cycles'], cycles=cycles,
              plan=p, clip_duration_s=round(D, 3))
    json.dump(gt, open(os.path.join(vdir, 'gt.json'), 'w'), indent=1)
    print(f'edit_{k}: {p["cycles"]} cycles, speeds {p["speeds"]}, waits {p["wait_after_each_cycle_s"]}, '
          f'mid-stop {p["mid_cycle_stop"]}, {len(times)} frames ({len(need)} distinct)', flush=True)
json.dump(plans, open(os.path.join(OUT, 'plan.json'), 'w'), indent=1)
print('DONE')
