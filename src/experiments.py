"""Registry of synthetic videos (name -> generator config) and the SAM processing order."""
NOM = dict(program='inplane', appearance='real', n_cycles=6, fps=2.0)
CAM_SHIFT = dict(eye=(-0.05, -1.45, 0.72), target=(-0.33, 0.0, 0.36), fov=40)

def V(**kw):
    c = dict(NOM); c.update(kw); return c

REG = {}
for s in range(5):  REG[f'nom_s{s}'] = V(seed=100 + s)
for app in ['lowcon', 'highcon', 'darkbg']:
    for s in range(2): REG[f'app_{app}_s{s}'] = V(appearance=app, seed=200 + s)
for k, kw in [('clutter', dict(clutter=True)), ('occ', dict(occluder=2)), ('flicker', dict(flicker=0.15)), ('shake', dict(shake_px=3.0))]:
    for s in range(2): REG[f'seg_{k}_s{s}'] = V(seed=300 + s, **kw)
for prog in ['outplane', 'pureyaw', 'wrist', 'viahome', 'pallet', 'continuous']:
    for s in range(3): REG[f'prog_{prog}_s{s}'] = V(program=prog, seed=400 + s)
for amp in [0.1, 0.25, 0.5]:
    for s in range(3): REG[f'amp_{amp}_s{s}'] = V(amp=amp, seed=500 + s)
for k, kw in [('jit30', dict(speed_jitter=0.3)), ('pause', dict(pause_prob=0.35)), ('nodwell', dict(home_dwell=0.0, home_dwell_jitter=0.0)),
              ('partial', dict(partial_end=True))]:
    for s in range(2): REG[f'tim_{k}_s{s}'] = V(seed=600 + s, **kw)
for s in range(2): REG[f'cam_shift_s{s}'] = V(cam=CAM_SHIFT, seed=700 + s)
for s in range(2): REG[f'via_near35_s{s}'] = V(program='viahome', near_frac=0.35, seed=800 + s)
for s in range(2): REG[f'fps6_nodwell_s{s}'] = V(fps=6.0, home_dwell=0.0, home_dwell_jitter=0.0, seed=900 + s)
for s in range(2): REG[f'fps6_cont_s{s}'] = V(fps=6.0, program='continuous', seed=950 + s)

MOBILE_ORDER = list(REG)   # all, in registry order (nominal first)
VITB_ORDER = ['nom_s0', 'nom_s2', 'nom_s3', 'app_lowcon_s1', 'seg_clutter_s1', 'seg_occ_s1', 'prog_pureyaw_s1', 'nom_s1']
