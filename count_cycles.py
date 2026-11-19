#!/usr/bin/env python3
"""SAM-Box PoseRAC v6.1: count the work cycles of a robot arm in videos, fully automatically.

    python count_cycles.py my_video.mp4 --out results/              # one video
    python count_cycles.py a.mp4 b.mp4 c.mp4 --out results/         # several
    python count_cycles.py my_videos_folder/ --out results/         # every video in a folder

For each video, two steps:
  1. automatic setup (src/auto_setup.py): finds the robot, a point on its base and its home frame;
  2. the v6.1 counter (src/run_v6.py, unchanged): follows the robot's outline, counts each home -> away -> home trip
     and times every cycle.
Results go to <out>/<video name>/:
  setup.json   the setup used (box, base point, home frame, and how each was chosen)
  setup.jpg    the home frame with the outline, box and base point drawn on: check it at a glance
  result.json  the cycle count, warning flags, run details (and the score, if labels were given)
  cycles.csv   one row per cycle: start, end, duration, wait at home, period
  trace.npz    the robot's outline and the home score over time
With more than one video also <out>/summary/: videos.csv, all_cycles.csv, summary.txt.
A video that already has result.json is skipped, so a stopped run can simply be started again.

Options:
  --labels DIR     known cycles, DIR/<video name>.json (the NACHI label format: total_cycles and cycles with
                   start_time/end_time in seconds); the result is then scored
  --setup FILE|DIR give the setup by hand instead of the automatic one: one JSON file for every video (same camera) or
                   DIR/<video name>.json; keys init_box [x0, y0, x1, y1] and base_point [x, y] in 1280x720 pixels of the
                   home frame, init_frame (home frame index at 2 fps, default 0)
  --device         auto (default: cuda, else Apple Silicon mps, else cpu), cuda, mps or cpu
  --skim SECONDS   how much of the start of the video the automatic setup looks at (default 120)

Needs: Python packages from requirements.txt, and ffmpeg (a system ffmpeg, or the one that comes with imageio-ffmpeg).
The SAM ViT-B weights (375 MB) are downloaded to src/weights/ on the first run and checked.
"""
import os, sys, json, glob, time, csv, hashlib, shutil, argparse, urllib.request, traceback

REPO = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(REPO, 'src'); sys.path.insert(0, SRC)
WEIGHTS = os.path.join(SRC, 'weights', 'sam_b.pt')
WEIGHTS_URL = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_b.pt'
WEIGHTS_MD5 = '01ec64d29a2fca3f0661936605ae66f8'      # the same weights as Meta's sam_vit_b_01ec64.pth
VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm', '.mpg', '.mpeg', '.wmv'}


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def ensure_weights(log):
    if os.path.isfile(WEIGHTS):
        return
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    tmp = WEIGHTS + '.part'
    log(f'downloading the SAM ViT-B weights (375 MB, once) from {WEIGHTS_URL}')
    with urllib.request.urlopen(WEIGHTS_URL) as r, open(tmp, 'wb') as f:
        total, done, last = int(r.headers.get('Content-Length', 0)), 0, 0
        for b in iter(lambda: r.read(1 << 22), b''):
            f.write(b); done += len(b)
            if total and done * 10 // total > last:
                last = done * 10 // total; print(f'  {10 * last}%', flush=True)
    got = md5(tmp)
    if got != WEIGHTS_MD5:
        os.remove(tmp)
        sys.exit(f'the downloaded weights are not the expected file (md5 {got}, expected {WEIGHTS_MD5}); try again')
    os.replace(tmp, WEIGHTS); log('weights downloaded and checked')


def find_ffmpeg():
    f = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
    if f:
        return f
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit('ffmpeg not found: install it (e.g. brew install ffmpeg / apt install ffmpeg) or pip install imageio-ffmpeg')


def collect(inputs):
    vids = []
    for p in inputs:
        if os.path.isdir(p):
            vids += sorted(f for f in glob.glob(os.path.join(p, '*')) if os.path.splitext(f)[1].lower() in VIDEO_EXT)
        elif os.path.isfile(p):
            vids.append(p)
        else:
            sys.exit(f'not found: {p}')
    out, seen = [], {}
    for v in vids:
        n = os.path.splitext(os.path.basename(v))[0]
        seen[n] = seen.get(n, 0) + 1
        out.append((n if seen[n] == 1 else f'{n}_{seen[n]}', os.path.abspath(v)))
    return out


def given_setup(arg, name):
    """the setup given by hand for this video, or None"""
    if not arg:
        return None
    f = os.path.join(arg, name + '.json') if os.path.isdir(arg) else arg
    if not os.path.isfile(f):
        return None
    s = json.load(open(f))
    if len(s.get('init_box', [])) != 4 or len(s.get('base_point', [])) != 2:
        sys.exit(f'{f}: a setup needs init_box [x0, y0, x1, y1] and base_point [x, y] (1280x720 pixels)')
    return dict(init_box=[float(v) for v in s['init_box']], base_point=[float(v) for v in s['base_point']],
                init_frame=int(s.get('init_frame', 0)), source=f'given by hand: {os.path.abspath(f)}')


def draw_given(video, setup, path, ffmpeg):
    """setup.jpg for a setup given by hand: the home frame with the box (yellow) and base point (red)"""
    import cv2, auto_setup as AS
    rgb = AS._read(video, 30, 1280, 720, 'rgb24', 1.0, setup['init_frame'] / 2.0, 1, ffmpeg)[0]
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x0, y0, x1, y1 = [int(round(v)) for v in setup['init_box']]; bx, by = [int(round(v)) for v in setup['base_point']]
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 220, 255), 2)
    cv2.circle(img, (bx, by), 9, (40, 40, 230), -1); cv2.circle(img, (bx, by), 9, (255, 255, 255), 2)
    AS.put_lines(img, ['setup given by hand', 'yellow = box, red = base point'])
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def write_summary(out, rows, log):
    sd = os.path.join(out, 'summary'); os.makedirs(sd, exist_ok=True)
    cols = ['video', 'count', 'true_cycles', 'exact', 'flags', 'setup', 'runtime_s', 'error']
    with open(os.path.join(sd, 'videos.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    header, lines = None, []
    for r in rows:
        p = os.path.join(out, r['video'], 'cycles.csv')
        if os.path.isfile(p):
            ls = open(p).read().splitlines()
            if ls:
                header = header or ls[0]; lines += ls[1:]
    if header:
        open(os.path.join(sd, 'all_cycles.csv'), 'w').write('\n'.join([header] + lines) + '\n')
    done = [r for r in rows if not r['error']]; lab = [r for r in done if r['true_cycles'] != '']
    txt = [f'{len(rows)} videos: {len(done)} counted, {len(rows) - len(done)} failed; {sum(r["count"] for r in done)} cycles counted']
    if lab:
        txt.append(f'with labels: {len(lab)} videos, exact count on {sum(r["exact"] is True for r in lab)}/{len(lab)}')
    txt += [f'FAILED {r["video"]}: {r["error"]}' for r in rows if r['error']]
    txt += [f'FLAGS  {r["video"]}: {r["flags"]}' for r in done if r['flags']]
    open(os.path.join(sd, 'summary.txt'), 'w').write('\n'.join(txt) + '\n')
    for t in txt[:2]:
        log(t)


def main():
    ap = argparse.ArgumentParser(description='SAM-Box PoseRAC v6.1: count robot work cycles in videos, fully automatically',
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split('Options:')[0].split('\n\n', 1)[1])
    ap.add_argument('inputs', nargs='+', help='video files and/or folders of videos')
    ap.add_argument('--out', required=True, help='results folder')
    ap.add_argument('--labels', help='folder of label files <video name>.json (optional: scores the result)')
    ap.add_argument('--setup', help='setup given by hand: one JSON file for all videos, or a folder of <video name>.json')
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    ap.add_argument('--skim', type=float, default=120.0, help='seconds of video the automatic setup looks at (default 120)')
    a = ap.parse_args()

    videos = collect(a.inputs)
    if not videos:
        sys.exit('no videos found (extensions: ' + ' '.join(sorted(VIDEO_EXT)) + ')')
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'run_log.txt'), 'a')

    def log(m):
        line = time.strftime('%H:%M:%S ') + m; print(line, flush=True); logf.write(line + '\n'); logf.flush()

    ensure_weights(log)
    ffmpeg = find_ffmpeg()
    import torch, run_real as RR, run_v6 as V6, auto_setup as AS
    device = AS.pick_device(a.device)
    if device == 'cuda' and not torch.cuda.is_available():
        sys.exit('--device cuda: no CUDA GPU found')
    _load, _cache = RR.load_predictor, {}

    def shared_predictor(dev):                         # one SAM model for the setup and every video
        if dev not in _cache:
            _cache[dev] = _load(dev)
        return _cache[dev]
    RR.load_predictor = shared_predictor
    log(f'count_cycles v6.1: {len(videos)} video(s) on {device}; results in {os.path.abspath(a.out)}')

    rows = []
    for name, path in videos:
        od = os.path.join(a.out, name); os.makedirs(od, exist_ok=True)
        row = dict(video=name, count='', true_cycles='', exact='', flags='', setup='', runtime_s='', error='')
        t0 = time.time()
        try:
            sf = os.path.join(od, 'setup.json')
            setup = given_setup(a.setup, name)
            if setup is not None:
                json.dump(setup, open(sf, 'w'), indent=1); draw_given(path, setup, os.path.join(od, 'setup.jpg'), ffmpeg)
                log(f'{name}: setup given by hand')
            elif os.path.isfile(sf):
                setup = json.load(open(sf)); log(f'{name}: setup from an earlier run ({setup.get("source", "automatic")})')
            else:
                log(f'{name}: automatic setup ...')
                setup, info = AS.auto_setup(path, device, a.skim, ffmpeg, predictor=shared_predictor(device))
                setup['source'] = 'automatic'
                json.dump(setup, open(sf, 'w'), indent=1); AS.draw_setup(setup, info, os.path.join(od, 'setup.jpg'))
                log(f"{name}: setup done in {time.time() - t0:.0f} s: box {setup['init_box']}, base {setup['base_point']}, "
                    f"home {setup['home_rule']} ({setup['home_time_s']} s), outline {setup['chosen_outline']}")
            gt = os.path.join(a.labels, name + '.json') if a.labels else None
            if gt and not os.path.isfile(gt):
                log(f'{name}: no label file {gt}, counting without a score'); gt = None
            job = dict(name=name, video=path, gt=gt, setup={k: setup[k] for k in ('init_box', 'base_point', 'init_frame')})
            res = V6.run_job(job, a.out, device, log, ffmpeg)
            row.update(count=res['count'], true_cycles=res.get('true_cycles', ''), exact=res.get('exact', ''),
                       flags=' | '.join(res.get('flags', [])), setup=setup.get('source', 'automatic'),
                       runtime_s=round(time.time() - t0, 1))
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {e}'
            log(f'!! {name} failed: {e}\n{traceback.format_exc()}')
        rows.append(row)
    if len(videos) > 1:
        write_summary(a.out, rows, log)
    failed = [r['video'] for r in rows if r['error']]
    log('done' + (f'; FAILED: {", ".join(failed)}' if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
