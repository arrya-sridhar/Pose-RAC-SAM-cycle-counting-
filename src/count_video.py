"""Count robot cycles in one video with the v5 pipeline (no ground truth needed).

    python count_video.py --video clip.mp4 --box 600 156 944 719 --click 812 592
    python count_video.py --frames path/to/frames --box ... --click ... --device cuda

--box    x0 y0 x1 y1 : a box around the robot on the first frame, in 1280x720 pixel coordinates
--click  x y         : one point on the robot's base (1280x720 coordinates)
The robot should be at its home pose on the reference frame (--init_frame, default 0).
Prints the cycle count and the time of each return home; --out writes them to a JSON file.
"""
import os, sys, json, glob, time, argparse, tempfile, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import run_real as RR
from features import mask_iou_to

FPS = 2.0


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--video'); g.add_argument('--frames', help='folder of frames already at 2 fps, 1280x720')
    ap.add_argument('--box', type=float, nargs=4, required=True); ap.add_argument('--click', type=float, nargs=2, required=True)
    ap.add_argument('--init_frame', type=int, default=0)
    ap.add_argument('--device', default='cuda' if RR.torch.cuda.is_available() else ('mps' if RR.torch.backends.mps.is_available() else 'cpu'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    tmp = None
    if a.video:
        tmp = tempfile.mkdtemp(prefix='count_video_')
        subprocess.run([ffmpeg_exe(), '-nostdin', '-loglevel', 'error', '-i', a.video, '-vf', 'fps=2,scale=1280:720', '-q:v', '3',
                        '-start_number', '1', os.path.join(tmp, '%06d.jpg')], check=True)
        fdir = tmp
    else:
        fdir = a.frames
    frames = sorted(glob.glob(os.path.join(fdir, '*.jpg')) + glob.glob(os.path.join(fdir, '*.png')))
    if not frames:
        sys.exit(f'no frames found in {fdir}')
    setup = dict(init_box=a.box, base_point=a.click, init_frame=a.init_frame)
    log = lambda m: print(time.strftime('%H:%M:%S ') + m, flush=True)
    log(f'{len(frames)} frames ({len(frames) / FPS:.1f} s of video), device {a.device}')
    sam, wb, spf = RR.run_sam(frames, setup, a.device, log)
    M = sam['ours']['masks']; f0 = a.init_frame
    d = 1 - mask_iou_to(M, M[f0]); p, lo, hi = RR.refdist_prob(d)
    count, departures, home_entries, _ = RR.counter(p, int(round(1.5 * FPS)))
    returns = [round(i / FPS, 1) for i in home_entries if i < len(frames)]
    res = dict(cycles=int(count), return_home_times_s=returns, frames=len(frames), seconds_per_frame=round(spf, 3),
               held_frames=int(sam['ours']['rejected_frames']), outline_area_pct=round(100 * float(M.mean()), 2))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
