"""Run v5 (2 fps) and v5.1 (6 fps) on the new videos for Harshitha's slides.

v5.1 = v5 with frames sampled at 6 per second instead of 2. Every rule is unchanged and still counted in frames,
except the end padding, which stays 1.5 s (9 frames at 6 fps). Uses run_real.py (v5) from ../src (repo) or the parent folder (laptop).
The job lists hold the laptop paths used on 14 Oct 2026; edit them for your machine.
Resumable: finished jobs (result.json present) are skipped.

    caffeinate -i ../.venv/bin/python run_newtests.py jobs.json
"""
import os, sys, json, glob, time, subprocess, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(PKG, 'src'), PKG]          # repo layout (src/) or the laptop's flat repro_pkg layout
import numpy as np, cv2
import run_real as RR
from features import mask_iou_to

LOG = open(os.path.join(HERE, 'run_log.txt'), 'a')


def log(m):
    line = time.strftime('%H:%M:%S ') + m; print(line, flush=True); LOG.write(line + '\n'); LOG.flush()


def extract(video, fdir, fps, start, dur=None):
    if len(glob.glob(os.path.join(fdir, '*.jpg'))) == 0:
        os.makedirs(fdir, exist_ok=True)
        cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error'] + (['-ss', str(start)] if start else []) + \
              (['-t', str(dur)] if dur else []) + ['-i', video, '-vf', f'fps={fps},scale=1280:720', '-q:v', '3', '-start_number', '1', os.path.join(fdir, '%06d.jpg')]
        subprocess.run(cmd, check=True)
    return sorted(glob.glob(os.path.join(fdir, '*.jpg')))


def smoothed(p, hold):
    p = np.concatenate([p, np.repeat(p[-1:], hold)]); s, out = 0.5, []
    for o in p:
        s = 0.6 * o + 0.4 * s; out.append(s)
    return np.array(out)


def match_ends(returns, ends, lo=-1.0, hi=2.0):
    used, found, offs = set(), 0, []
    for e in ends:
        cand = [(abs(x - e), j) for j, x in enumerate(returns) if j not in used and lo <= x - e <= hi]
        if cand:
            _, j = min(cand); used.add(j); found += 1; offs.append(returns[j] - e)
    return found, len(returns) - len(used), (round(float(np.median(offs)), 2) if offs else None)


def run_job(job):
    od = os.path.join(HERE, 'out', job['name'])
    if os.path.exists(os.path.join(od, 'result.json')):
        log(f"skip {job['name']} (done)"); return
    fps = float(job['fps']); t0 = time.time()
    frames = extract(job['video'], os.path.join(od, 'frames'), fps, job.get('start', 0), job.get('dur'))
    n = len(frames); setup = job['setup']; f0 = setup.get('init_frame', 0)
    log(f"== {job['name']}: {job['version']} at {fps:g} fps, {n} frames ({n / fps:.1f} s), extract {time.time() - t0:.0f} s")
    sam, wb, spf = RR.run_sam(frames, setup, job.get('device', 'mps'), log)
    M = sam['ours']['masks']; hold = int(round(1.5 * fps))
    d = 1 - mask_iou_to(M, M[f0]); p, lo, hi = RR.refdist_prob(d)
    count, dep, he, _ = RR.counter(p, hold)
    returns = [round(i / fps + job.get('start', 0), 2) for i in he if i < n]
    area = M.mean((1, 2))
    ci = np.array([(M[k] & M[k + 1]).sum() / max((M[k] | M[k + 1]).sum(), 1) for k in range(n - 1)])
    res = dict(name=job['name'], version=job['version'], video=os.path.basename(job['video']), fps=fps,
               start_s=job.get('start', 0), frames=n, setup=setup, count=int(count), departures=int(dep),
               return_home_times_s=returns, held_frames=int(sam['ours']['rejected_frames']),
               outline_area_pct_mean=round(100 * float(area.mean()), 2), outline_area_pct_max=round(100 * float(area.max()), 2),
               sudden_jumps=int((ci < 0.6).sum()), s_per_frame=round(spf, 3), runtime_s=round(time.time() - t0, 1))
    if job.get('gt'):
        g = json.load(open(job['gt'])); true = int(g['total_cycles'])
        ends = [c['end_time_sec'] for c in g['cycles']]
        f, extra, med = match_ends(returns, ends)
        res.update(true_cycles=true, nmae=round(abs(count - true) / true, 4), obo=bool(abs(count - true) <= 1),
                   exact=bool(count == true), ends_found=f, ends_of=len(ends), ends_extra=extra, median_end_offset_s=med)
    np.savez_compressed(os.path.join(od, 'trace.npz'), d=d, p=p, s=smoothed(p, hold), lo=lo, hi=hi,
                        masks=np.packbits(M.reshape(n, -1), axis=1), boxes=np.array(sam['ours']['box'], np.float32))
    RR.contact_sheet(frames, sam, os.path.join(od, 'contact_sheet.jpg'))
    json.dump(res, open(os.path.join(od, 'result.json'), 'w'), indent=1)
    log(f"   done {job['name']}: count {count}" + (f" / true {res['true_cycles']} (NMAE {res['nmae']}, OBO {res['obo']})" if 'true_cycles' in res else '')
        + f" | {spf:.2f} s/frame | held {res['held_frames']} | area {res['outline_area_pct_mean']}% | {res['runtime_s']:.0f} s")


if __name__ == '__main__':
    jobs = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'jobs.json')))
    log(f'starting {len(jobs)} jobs')
    for j in jobs:
        try:
            run_job(j)
        except Exception as e:
            log(f"!! {j['name']} failed: {e}\n{traceback.format_exc()}")
    log('ALL DONE')
