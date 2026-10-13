"""Reproduce the stitched-pipeline result on the synthetic 'normal' test video (nom_s2): SAM masks -> PoseRAC -> cycle count.

Steps (all deterministic):
  1. render synthetic videos nom_s0 (training) and nom_s2 (test) with ground truth
  2. download SAM weights (MobileSAM by default; --model vitb for SAM ViT-B) from the Ultralytics GitHub release
  3. run SAM with the 'following box + base click' prompt on every frame (2 fps)
  4. train PoseRAC on nom_s0 (2 labelled frames per cycle), count cycles on nom_s2
  5. print counts for PoseRAC, the label-free counter and Abhinav's area autocorrelation; write results.json

Usage:  python reproduce.py [--model mobile|vitb]
"""
import os, sys, json, time, argparse, urllib.request, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'synth'))
import numpy as np, torch
import experiments as E
from make_video import make_video
import run_sam, evaluate as EV

URLS = {'mobile': ('mobile_sam.pt', 'https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt'),
        'vitb': ('sam_b.pt', 'https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_b.pt')}

def download(url, dest):
    """download with certifi's certificates (python.org macOS installs ship without system certificates); fall back to curl"""
    import ssl, shutil, subprocess
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx) as r, open(dest + '.part', 'wb') as f:
            shutil.copyfileobj(r, f)
        os.replace(dest + '.part', dest)
    except Exception as e:
        print(f'    python download failed ({type(e).__name__}); trying curl ...', flush=True)
        subprocess.run(['curl', '-L', '--fail', '-o', dest, url], check=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='mobile', choices=['mobile', 'vitb'])
    ap.add_argument('--threads', type=int, default=2); a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    D = os.path.join(HERE, 'data', 'videos'); os.makedirs(D, exist_ok=True)
    for n in ['nom_s0', 'nom_s2']:
        if not os.path.exists(os.path.join(D, n, 'gt.npz')):
            m = make_video(E.REG[n], os.path.join(D, n)); print(f'[1] rendered {n}: {m["n_frames"]} frames, true cycles = {m["count"]}', flush=True)
    wdir = run_sam.WEIGHTS; os.makedirs(wdir, exist_ok=True)
    fn, url = URLS[a.model]
    if not os.path.exists(os.path.join(wdir, fn)):
        print(f'[2] downloading {fn} ...', flush=True); download(url, os.path.join(wdir, fn))
    strat = 'track_motion_pt'
    for n in ['nom_s0', 'nom_s2']:
        out = os.path.join(D, n, f'sam_{a.model}.npz')
        if not os.path.exists(out):
            spf = run_sam.run(os.path.join(D, n), a.model, [strat]); print(f'[3] SAM ({a.model}) on {n}: {spf:.2f} s/frame', flush=True)
    src = f'{a.model}:{strat}'
    train = [EV.load_video(os.path.join(D, 'nom_s0'), src)]
    test = EV.load_video(os.path.join(D, 'nom_s2'), src)
    r_pose, p = EV.m_poserac_sup(train, test, 'grid', 'perframe', seed=42)
    r_ref, _ = EV.m_refdist(test)
    r_ac, info = EV.m_autocorr(test)
    res = dict(model=a.model, prompt=strat, true_cycles=test['count'], mask_overlap_with_truth=round(float(test['miou'].mean()), 3),
               poserac_supervised=r_pose['pred'], label_free_overlap_counter=r_ref['pred'], abhinav_area_autocorrelation=r_ac['pred'],
               autocorrelation_period_s=info['period'] if info else None, minutes=round((time.time() - t0) / 60, 1))
    json.dump(res, open(os.path.join(HERE, f'results_{a.model}.json'), 'w'), indent=1)
    print('\n[5] RESULT'); [print(f'    {k}: {v}') for k, v in res.items()]

if __name__ == '__main__':
    main()
