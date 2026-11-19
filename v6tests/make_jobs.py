"""Write the v6 test job lists (cluster paths). Only the tests each change needs:
  A: 10 development videos (regression), the 3 nachi_4 videos v5 got wrong (set-up memory), stress test
  B: 9 edited videos (regression), simulated palletizing videos (dynamic rate; only if --pallet), OP75 robot B 30 s
     (dynamic rate)
    python v6tests/make_jobs.py --work /scratch/psaxena/arrya_v6 [--pallet] [--extra name=json ...]
"""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = '/home/psaxena/Harshitha/dataset_out'

ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--pallet', action='store_true')
ap.add_argument('--pallet_cfg', default='{}', help='cfg overrides for the palletizing runs (JSON)')
a = ap.parse_args(); W = a.work

dev = json.load(open(os.path.join(ROOT, 'configs/real_setup.json')))
edited = json.load(open(os.path.join(ROOT, 'configs/edited_setup.json')))
progs = json.load(open(os.path.join(ROOT, 'hpc/job_setups.json')))

A, B = [], []
for n, su in sorted(dev.items(), key=lambda x: int(x[0].split('_')[1])):
    A.append(dict(name=f'dev/{n}', video=f'{DS}/videos/{n}.mp4', gt=f'{DS}/ground_truth/{n}.json', setup=su,
                  frames_root=f'{W}/frames/{n}'))
for n in ['nachi_4_seq_014', 'nachi_4_seq_016', 'nachi_4_seq_018']:
    su = {k: v for k, v in progs['nachi_4'].items() if k in ('init_box', 'base_point', 'init_frame')}
    A.append(dict(name=f'nachi4/{n}', video=f'{DS}/videos/{n}.mp4', gt=f'{DS}/ground_truth/{n}.json', setup=su,
                  frames_root=f'{W}/frames/{n}'))
# stress test: remove the lower third of the outline at 35 s (2nd cycle); with and without the set-up memory.
# Fixed rate (dynamic off) so only the memory differs.
for n in ['nachi_2_seq_003', 'nachi_4_seq_012']:
    for mem in [False, True]:
        A.append(dict(name=f'stress/{n}_memory_{"on" if mem else "off"}', video=f'{DS}/videos/{n}.mp4',
                      gt=f'{DS}/ground_truth/{n}.json', setup=dev[n], frames_root=f'{W}/frames/{n}',
                      stress=dict(t=35.0), cfg=dict(dynamic=False, memory=mem)))
for n, su in sorted(edited.items(), key=lambda x: int(x[0].split('_')[1])):
    B.append(dict(name=f'edited/{n}', frames_dir=f'{W}/edited30/{n}/frames', gt=f'{W}/edited30/{n}/gt.json', setup=su))
new = json.load(open(os.path.join(ROOT, 'newtests/jobs.json'))) + json.load(open(os.path.join(ROOT, 'newtests/jobs2.json')))
if a.pallet:
    pc = json.loads(a.pallet_cfg)
    for j in new:
        if j['name'].startswith('v51_') and ('nonuni' in j['name'] or 'uniform' in j['name']):
            v = os.path.basename(j['video']); g = os.path.basename(j['gt'])
            B.append(dict(name=f"sim/{j['name'][4:]}", video=f'{W}/newvideos/{v}', gt=f'{W}/newvideos/{g}', setup=j['setup'],
                          frames_root=f"{W}/frames/sim_{j['name'][4:]}", cfg=pc))
for j in new:
    if j['name'] == 'v51_op75_robotB_30s':
        B.append(dict(name='op75/robotB_30s', video=f"{W}/newvideos/{os.path.basename(j['video'])}", setup=j['setup'],
                      start=j['start'], dur=j['dur'], frames_root=f'{W}/frames/op75_robotB_30s'))
json.dump(A, open(os.path.join(ROOT, 'v6tests/jobs_A.json'), 'w'), indent=1)
json.dump(B, open(os.path.join(ROOT, 'v6tests/jobs_B.json'), 'w'), indent=1)
print('A:', len(A), 'jobs;', 'B:', len(B), 'jobs')
