"""Regression check: does a change keep (or improve) v5's results?

Run this before calling any change an improvement. It compares the current code against the stored v5 results on
the 10 development videos and the 9 edited videos (uneven cycles), and prints one row per video.

  recount   Seconds, no GPU. Re-runs only stage 2 (home score + counting: run_real.refdist_prob and
            run_real.counter) on the v5 outlines stored in results/. Use it for any change to stage 2.
            It cannot test a change to the tracker (stage 1), because the outlines are v5's.

  run       Whole pipeline (SAM tracking + counting) on the 19 videos, then compares. Use it for any change to
            stage 1. Needs the input frames (see 'prepare') and a GPU: about 4,100 frames, roughly 10-15 min on one
            cluster GPU plus start-up. Output goes to a new folder; nothing in results/ is touched.

  compare   Compare a finished 'run' folder with v5 (for example after a cluster job).

  prepare   Build the input frames once: the 10 development videos from the dataset (2 fps, 1280x720, same ffmpeg
            settings as the 200-video run) and the 9 edited videos (make_edited_tests.py, fixed seed 20260925).

    python src/regression_check.py recount
    python src/regression_check.py prepare --dataset /home/psaxena/Harshitha/dataset_out \\
                                           --clips /home/psaxena/Harshitha/nachi_robot_videos_trim --data $WORK/regression_data
    python src/regression_check.py run --data $WORK/regression_data --out $WORK/regression_runs/<name> --device cuda
    python src/regression_check.py compare $WORK/regression_runs/<name>

Pass rule (counts): no video ends up further from the true count than v5 was, and every video v5 counted exactly
stays exact. Cycle-end matches (within 1 s), outline size and held frames are reported for information; a drop in
matched cycle ends is flagged as a warning. Exit code 0 = pass, 1 = fail.
"""
import os, sys, json, glob, time, argparse, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
import numpy as np

FPS = 2.0
GW, GH = 640, 360          # stored outlines are 640x360, bit-packed
SETS = {                   # set name -> (stored v5 results, setup file)
    'dev': ('results/real_10_v5', 'configs/real_setup.json'),
    'edited': ('results/edited_9', 'configs/edited_setup.json'),
}


ONLY = None                # optional list of video names to restrict every step to (partial check)


def order(names):
    return sorted([n for n in names if ONLY is None or n in ONLY], key=lambda s: int(s.split('_')[1]))


def setups(set_name):
    return json.load(open(os.path.join(ROOT, SETS[set_name][1])))


def summarise(res):
    """the numbers we compare, from a run_real.py results.json"""
    o = res['methods']['ours']; lc = o['labelfree_counter']
    return dict(true=int(res['true_cycles']), count=int(lc['count']),
                ends_found=lc['boundaries']['found'], ends_of=lc['boundaries']['of'], ends_extra=lc['boundaries']['extra'],
                area=o['mask_area_pct']['mean'], held=int(res.get('ours_rejected_masks', 0)))


def baseline(set_name):
    d = os.path.join(ROOT, SETS[set_name][0])
    return {n: summarise(json.load(open(os.path.join(d, n, 'results.json')))) for n in order(setups(set_name))}


def git_rev():
    try:
        r = subprocess.run(['git', '-C', ROOT, 'describe', '--always', '--dirty', '--tags'], capture_output=True, text=True)
        return r.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


# ------------------------------------------------------------------ recount (stage 2 only, stored v5 outlines)
def recount(set_name):
    import run_real as RR
    from features import mask_iou_to
    d = os.path.join(ROOT, SETS[set_name][0]); su = setups(set_name); out = {}
    for n in order(su):
        res = json.load(open(os.path.join(d, n, 'results.json'))); gt = json.load(open(os.path.join(d, n, 'gt.json')))
        k = res['frames']; f0 = res.get('reference_frame', su[n].get('init_frame', 0))
        z = np.load(os.path.join(d, n, 'sam_vitb_masks.npz'))
        M = np.unpackbits(z['ours__masks'], axis=1)[:, :GW * GH].reshape(k, GH, GW).astype(bool)
        dist = 1 - mask_iou_to(M, M[f0]); p, lo, hi = RR.refdist_prob(dist)
        c, dep, he, _ = RR.counter(p, int(round(1.5 * FPS)))
        ends = [cy['end_time'] for cy in gt['cycles']][:-1]
        b = RR.boundary_scores([i / FPS for i in he if i < k], ends)
        out[n] = dict(true=int(gt['total_cycles']), count=int(c), ends_found=b['found'], ends_of=b['of'], ends_extra=b['extra'],
                      area=round(100 * float(M.mean()), 2), held=int(res.get('ours_rejected_masks', 0)))
    return out


# ------------------------------------------------------------------ prepare inputs
def prepare(a):
    os.makedirs(a.data, exist_ok=True)
    try:
        import imageio_ffmpeg; ff = os.environ.get('FFMPEG', imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        ff = os.environ.get('FFMPEG', 'ffmpeg')
    for n in order(setups('dev')):
        vd = os.path.join(a.data, 'dev', n); fd = os.path.join(vd, 'frames')
        if len(glob.glob(os.path.join(fd, '*.jpg'))) > 0:
            print('exists', n); continue
        os.makedirs(fd, exist_ok=True)
        subprocess.run([ff, '-nostdin', '-loglevel', 'error', '-i', os.path.join(a.dataset, 'videos', n + '.mp4'),
                        '-vf', 'fps=2,scale=1280:720', '-q:v', '3', '-start_number', '1', os.path.join(fd, '%06d.jpg')], check=True)
        shutil.copyfile(os.path.join(a.dataset, 'ground_truth', n + '.json'), os.path.join(vd, 'gt.json'))
        print('dev', n, len(glob.glob(os.path.join(fd, '*.jpg'))), 'frames', flush=True)
    ed = os.path.join(a.data, 'edited')
    if all(os.path.exists(os.path.join(ed, n, 'gt.json')) for n in setups('edited')):
        print('edited set exists')
    else:
        env = dict(os.environ, NACHI_CLIPS=a.clips)
        subprocess.run([sys.executable, os.path.join(HERE, 'make_edited_tests.py'), ed], check=True, env=env)
    # the edited set is rebuilt from a fixed seed; check it matches the stored ground truth
    for n in order(setups('edited')):
        new = json.load(open(os.path.join(ed, n, 'gt.json')))
        old = json.load(open(os.path.join(ROOT, SETS['edited'][0], n, 'gt.json')))
        if new['total_cycles'] != old['total_cycles'] or abs(new['total_duration_sec'] - old['total_duration_sec']) > 0.01:
            sys.exit(f'!! {n}: rebuilt edited video differs from the stored one -- do not compare against v5')
    print('inputs ready in', a.data)


# ------------------------------------------------------------------ full run
def run(a):
    import run_real as RR
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'run_log.txt'), 'a')
    def log(msg):
        line = time.strftime('%H:%M:%S ') + msg; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    log(f'code version: {git_rev()}  device: {a.device}')
    for s in a.sets:
        su = setups(s)
        for n in order(su):
            src = os.path.join(a.data, s, n); vd = os.path.join(a.out, s, n)
            if os.path.exists(os.path.join(vd, 'results.json')):
                log(f'skip {s}/{n} (done)'); continue
            if not os.path.isdir(os.path.join(src, 'frames')):
                sys.exit(f'missing frames: {src}/frames (run prepare first)')
            os.makedirs(vd, exist_ok=True)
            if not os.path.exists(os.path.join(vd, 'frames')):
                os.symlink(os.path.abspath(os.path.join(src, 'frames')), os.path.join(vd, 'frames'))
            shutil.copyfile(os.path.join(src, 'gt.json'), os.path.join(vd, 'gt.json'))
            RR.process(vd, su[n], a.device, log)
    return compare_folder(a.out, a.sets)


def compare_folder(folder, sets):
    rows = {}
    for s in sets:
        new = {}
        for n in order(setups(s)):
            f = os.path.join(folder, s, n, 'results.json')
            if os.path.exists(f):
                new[n] = summarise(json.load(open(f)))
        rows[s] = new
    return rows


# ------------------------------------------------------------------ report
def report(new_by_set, label, out_json=None):
    fail, warn, lines = False, [], []
    hdr = f"{'set':7s} {'video':17s} {'true':>4s} {'v5':>3s} {'new':>3s}  {'status':8s} {'ends±1s v5->new':>15s} {'area% v5->new':>14s} {'held v5->new':>12s}"
    lines += [f'Regression check ({label}), code version {git_rev()}',
              'ends±1s = labelled cycle ends (except the last) with a detected return home within 1 s; for information', hdr, '-' * len(hdr)]
    tot = dict(v5=0, new=0, n=0)
    for s, new in new_by_set.items():
        base = baseline(s)
        for n, b in base.items():
            if n not in new:
                lines.append(f'{s:7s} {n:17s} {b["true"]:4d} {b["count"]:3d}   -  MISSING'); fail = True; continue
            c = new[n]; e5, en = abs(b['count'] - b['true']), abs(c['count'] - c['true'])
            st = 'same' if c['count'] == b['count'] else ('better' if en < e5 else ('WORSE' if en > e5 else 'changed'))
            if st == 'WORSE' or (e5 == 0 and en > 0):
                fail = True
            if c['ends_found'] < b['ends_found'] or c['ends_extra'] > b['ends_extra']:
                warn.append(f'{s}/{n}: cycle-end matches {b["ends_found"]}/{b["ends_of"]} (+{b["ends_extra"]} extra) -> '
                            f'{c["ends_found"]}/{c["ends_of"]} (+{c["ends_extra"]} extra)')
            tot['v5'] += e5 == 0; tot['new'] += en == 0; tot['n'] += 1
            lines.append(f'{s:7s} {n:17s} {b["true"]:4d} {b["count"]:3d} {c["count"]:3d}  {st:8s} '
                         f'{b["ends_found"]:>7d}->{c["ends_found"]:<5d}   {b["area"]:6.2f}->{c["area"]:<6.2f} {b["held"]:>5d}->{c["held"]:<5d}')
    lines.append('-' * len(hdr))
    lines.append(f'exact counts: v5 {tot["v5"]}/{tot["n"]}, new {tot["new"]}/{tot["n"]}')
    for w in warn:
        lines.append('warning: ' + w)
    if ONLY is not None:
        lines.append(f'note: partial check on {tot["n"]} video(s) only (--only); run without --only before calling a change better')
    lines.append('RESULT: ' + ('FAIL (a count got worse or a video v5 got right is now wrong)' if fail else 'PASS'))
    text = '\n'.join(lines); print(text)
    if out_json:
        json.dump(dict(label=label, code_version=git_rev(), passed=not fail, warnings=warn, videos=new_by_set),
                  open(out_json, 'w'), indent=1)
        open(os.path.splitext(out_json)[0] + '.txt', 'w').write(text + '\n')
    return not fail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('recount'); r.add_argument('--sets', nargs='+', default=list(SETS)); r.add_argument('--json', default=None)
    p = sp.add_parser('prepare'); p.add_argument('--dataset', required=True); p.add_argument('--clips', required=True)
    p.add_argument('--data', required=True)
    u = sp.add_parser('run'); u.add_argument('--data', required=True); u.add_argument('--out', required=True)
    u.add_argument('--device', default='cuda'); u.add_argument('--sets', nargs='+', default=list(SETS))
    c = sp.add_parser('compare'); c.add_argument('folder'); c.add_argument('--sets', nargs='+', default=list(SETS))
    for q in (r, u, c):
        q.add_argument('--only', nargs='+', default=None, help='restrict to these video names (quick partial check)')
    a = ap.parse_args()
    global ONLY
    ONLY = getattr(a, 'only', None)
    if a.cmd == 'recount':
        ok = report({s: recount(s) for s in a.sets}, 'recount: current stage 2 on stored v5 outlines', a.json)
    elif a.cmd == 'prepare':
        prepare(a); return
    elif a.cmd == 'run':
        ok = report(run(a), f'full run in {a.out}', os.path.join(a.out, 'regression_report.json'))
    else:
        ok = report(compare_folder(a.folder, a.sets), f'compare {a.folder}', os.path.join(a.folder, 'regression_report.json'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
