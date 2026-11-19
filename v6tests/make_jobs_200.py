"""Job lists for v6.1 on the two 200-video sets (run on the cluster login node: it only lists files and reads labels).
  task1: the original 200 (dataset_out), per-cycle times for every cycle
  task2: the 200 non-uniform videos (dataset_nonuniform, made 25 Oct with gaps, speed changes and holds), counts
Each video uses its program's set-up (hpc/job_setups.json), as in v5's 200-video run. Longest videos first, so the
cluster jobs sharing a list (run_v6.py --shared) finish at about the same time.
    python v6tests/make_jobs_200.py --out $WORK/lists
"""
import os, sys, json, glob, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = dict(task1='/home/psaxena/Harshitha/dataset_out', task2='/home/psaxena/Harshitha/dataset_nonuniform')

ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
ap.add_argument('--only', nargs='*', default=list(SETS))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
progs = json.load(open(os.path.join(ROOT, 'hpc/job_setups.json')))
# nachi_9's set-up frame is at 7 s because the original recordings fade in from an empty cell. The non-uniform videos
# have no fade: the robot rests from frame 0 with its arm stretched left past the original box, starts moving at ~3 s,
# and some videos start with a ~3.5 s idle gap, so the 7 s frame can be mid-movement. Set up at frame 0 like every
# other program, box widened to the stretched arm (measured on nachi_9_nu01 frame 0: arm from x 652, top y 218).
OVERRIDE = {('task2', 'nachi_9'): dict(init_box=[630.0, 205.0, 990.0, 719.0], base_point=[885.0, 590.0], init_frame=0)}

allj = []
for task in a.only:
    root, jobs, hours = SETS[task], [], 0.0
    for v in sorted(glob.glob(os.path.join(root, 'videos', '*.mp4'))):
        name = os.path.splitext(os.path.basename(v))[0]
        gp = os.path.join(root, 'ground_truth', name + '.json')
        g = json.load(open(gp))
        prog = g.get('job_id') or name.split('_seq')[0].split('_nu')[0]
        su = OVERRIDE.get((task, prog)) or {k: v for k, v in progs[prog].items() if k in ('init_box', 'base_point', 'init_frame')}
        dur = float(g.get('total_duration_sec') or g['total_frames'] / g.get('fps', 30))
        jobs.append(dict(name=f'{task}/{name}', program=prog, video=v, gt=gp, setup=su, duration_s=round(dur, 1)))
        hours += dur / 3600
    jobs.sort(key=lambda j: -j['duration_s'])
    json.dump(jobs, open(os.path.join(a.out, f'jobs_{task}.json'), 'w'), indent=1)
    allj += jobs
    by = {}
    for j in jobs:
        by[j['program']] = by.get(j['program'], 0) + 1
    print(f'{task}: {len(jobs)} videos, {hours:.2f} h of video; per program {dict(sorted(by.items()))}')
json.dump(allj, open(os.path.join(a.out, 'jobs_both.json'), 'w'), indent=1)   # task 1 first, then task 2
print(f'jobs_both.json: {len(allj)} videos (task 1 first)')
