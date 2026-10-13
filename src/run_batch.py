"""Run the v5 pipeline (run_real.py, unchanged tracker + counting) on many dataset videos.

For each video id:  extract frames at 2 fps / 1280x720 to node-local disk -> v5 SAM tracking (plus Abhinav's fixed box on the
same SAM image embedding) -> count cycles with
    ours    : v5 outlines + home/away counter (overlap with the reference frame, PoseRAC's two-threshold trigger)
    abhinav : fixed-box outlines + area autocorrelation, count = round(duration / period)
    (and the two crossings)  -> write results.json, a compact npz of both outline sets, a contact sheet; delete the frames.
Already-finished videos are skipped, so a job can simply be resubmitted.
"""
import os, sys, json, time, glob, shutil, subprocess, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import run_real as RR
from features import mask_iou_to

DS = os.environ.get('DS_ROOT', '/home/psaxena/Harshitha/dataset_out')
FPS = 2.0
try:                                     # one self-contained ffmpeg for every node (the cluster module is not usable everywhere)
    import imageio_ffmpeg; FFMPEG = os.environ.get('FFMPEG', imageio_ffmpeg.get_ffmpeg_exe())
except Exception:
    FFMPEG = os.environ.get('FFMPEG', 'ffmpeg')


def extract(mp4, fdir):
    os.makedirs(fdir, exist_ok=True)
    subprocess.run([FFMPEG, '-nostdin', '-loglevel', 'error', '-i', mp4, '-vf', 'fps=2,scale=1280:720', '-q:v', '3',
                    '-start_number', '1', os.path.join(fdir, '%06d.jpg')], check=True)
    return sorted(glob.glob(os.path.join(fdir, '*.jpg')))


def home_periods(p, hold=3, enter=0.78, exit_=0.4, momentum=0.4):
    """seconds intervals during which the official trigger holds 'home'"""
    p = np.concatenate([p, np.repeat(p[-1:], hold)]); r1 = RR.PR.ActionTrigger(enter, exit_); prob, st = 0.5, []
    for o in p:
        prob = o * (1 - momentum) + momentum * prob; r1(prob); st.append(r1.entered)
    iv, i = [], 0
    while i < len(st):
        if st[i]:
            j = i
            while j + 1 < len(st) and st[j + 1]: j += 1
            iv.append((i / FPS, j / FPS)); i = j + 1
        else:
            i += 1
    return iv


def boundary_hits(iv, bounds, T, slack=1.0):
    hit = sum(any(s - slack <= b <= e + slack for s, e in iv) for b in bounds)
    inner = [(s, e) for s, e in iv if s > 0.5 and e < T - 0.5]
    extra = sum(not any(s - slack <= b <= e + slack for b in bounds) for s, e in inner)
    return int(hit), int(extra)


def evaluate(M, f0, duration, gt_bounds, n):
    d = 1 - mask_iou_to(M, M[f0]); p, lo, hi = RR.refdist_prob(d)
    c_ctr, dep, he, _ = RR.counter(p, int(round(1.5 * FPS)))
    c_ac, per, acv = RR.autocorr_count(M.sum((1, 2)), duration)
    iv = home_periods(p); hit, extra = boundary_hits(iv, gt_bounds, n / FPS)
    he_t = [i / FPS for i in he if i < n]
    offs = [min(he_t, key=lambda x: abs(x - b)) - b for b in gt_bounds] if he_t and gt_bounds else []
    area = M.mean((1, 2))
    ci = np.array([(M[k] & M[k + 1]).sum() / max((M[k] | M[k + 1]).sum(), 1) for k in range(n - 1)]) if n > 1 else np.ones(1)
    return dict(counter=int(c_ctr), autocorr=int(c_ac), autocorr_period_s=per, boundary_hits=hit, boundary_extra=extra,
                home_entry_offsets_s=[round(float(o), 2) for o in offs],
                area_pct_mean=round(100 * float(area.mean()), 2), area_pct_max=round(100 * float(area.max()), 2),
                sudden_jumps=int((ci < 0.6).sum()), mean_consecutive_iou=round(float(ci.mean()), 3))


def process(vid, setups, out_root, tmp_root, device, log):
    od = os.path.join(out_root, vid)
    if os.path.exists(os.path.join(od, 'results.json')):
        return 'skip'
    os.makedirs(od, exist_ok=True)
    gt = json.load(open(os.path.join(DS, 'ground_truth', vid + '.json')))
    setup = setups[gt['job_id']]; f0 = setup.get('init_frame', 0)
    fdir = os.path.join(tmp_root, vid, 'frames'); t0 = time.time()
    frames = extract(os.path.join(DS, 'videos', vid + '.mp4'), fdir); n = len(frames); t_ext = time.time() - t0
    log(f'== {vid}: {n} frames, {gt["total_cycles"]} true cycles, {gt["total_duration_sec"]:.1f} s (extract {t_ext:.0f} s)')
    sam, wb, spf = RR.run_sam(frames, setup, device, log)
    duration = float(gt['total_duration_sec']); bounds = [c['end_time'] for c in gt['cycles']][:-1]
    res = dict(video=vid, job=gt['job_id'], true_cycles=int(gt['total_cycles']), duration_s=duration, frames=n,
               sam_s_per_frame=round(spf, 3), extract_s=round(t_ext, 1), reference_frame=f0,
               ours_held_frames=int(sam['ours']['rejected_frames']), methods={})
    for s in ('abhinav', 'ours'):
        res['methods'][s] = evaluate(sam[s]['masks'], f0, duration, bounds, n)
    np.savez_compressed(os.path.join(od, 'outlines.npz'), workspace_box=wb,
                        **{f'{s}__masks': np.packbits(sam[s]['masks'].reshape(n, -1), axis=1) for s in ('abhinav', 'ours')},
                        ours__box=sam['ours']['box'].astype(np.float32))
    RR.contact_sheet(frames, sam, os.path.join(od, 'outlines_abhinav_vs_ours.jpg'))
    json.dump(res, open(os.path.join(od, 'results.json'), 'w'), indent=1, default=float)
    shutil.rmtree(os.path.join(tmp_root, vid), ignore_errors=True)
    m = res['methods']
    log(f'   done {vid}: true {res["true_cycles"]} | ours {m["ours"]["counter"]} | abhinav {m["abhinav"]["autocorr"]} | '
        f'{spf:.2f} s/frame | held {res["ours_held_frames"]} | total {time.time() - t0:.0f} s')
    return 'done'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--setup', default=next((p for p in [os.path.join(HERE, 'job_setups.json'), os.path.join(HERE, '..', 'configs', 'job_setups.json')] if os.path.exists(p)), 'job_setups.json')); ap.add_argument('--device', default='cuda')
    ap.add_argument('--tmp', default=os.environ.get('TMPDIR', '/tmp'))
    a = ap.parse_args()
    setups = json.load(open(a.setup)); vids = [l.strip() for l in open(a.list) if l.strip()]
    os.makedirs(a.out, exist_ok=True)
    tmp_root = os.path.join(a.tmp, f'arrya_v5_{os.getpid()}'); os.makedirs(tmp_root, exist_ok=True)
    logf = open(os.path.join(a.out, f'log_{os.path.basename(a.list)}.txt'), 'a')
    def log(msg):
        line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    try:
        ver = subprocess.run([FFMPEG, '-version'], capture_output=True, text=True, check=True).stdout.split('\n')[0]
    except Exception as e:
        log(f'!! ffmpeg not usable ({FFMPEG}): {e} -- stopping before taking any video'); sys.exit(2)
    log(f'ffmpeg: {FFMPEG} ({ver})')
    import torch
    log(f'start: {len(vids)} videos, device {a.device}, torch {torch.__version__}, '
        f'gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}')
    claims = os.path.join(a.out, '.claims'); os.makedirs(claims, exist_ok=True); fails = 0
    for v in vids:
        if os.path.exists(os.path.join(a.out, v, 'results.json')):
            continue
        try:                                   # several jobs share one list: the first to create the claim folder takes the video
            os.mkdir(os.path.join(claims, v))
            open(os.path.join(claims, v, 'by'), 'w').write(f"{os.environ.get('SLURM_JOB_ID', '?')} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except FileExistsError:
            continue
        try:
            process(v, setups, a.out, tmp_root, a.device, log); fails = 0
        except Exception as e:
            log(f'!! {v} failed: {e}\n{traceback.format_exc()}')
            shutil.rmtree(os.path.join(tmp_root, v), ignore_errors=True)
            shutil.rmtree(os.path.join(claims, v), ignore_errors=True)      # give the video back to the pool
            fails += 1
            if fails >= 3:
                log('!! 3 failures in a row -- stopping this job'); break
    shutil.rmtree(tmp_root, ignore_errors=True)
    log('LIST DONE')


if __name__ == '__main__':
    main()
