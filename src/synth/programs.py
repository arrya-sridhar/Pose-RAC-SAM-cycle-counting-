"""Robot cycle programs (joint-space waypoint sequences) and timeline builder with ground truth.

A cycle = leave HOME -> visit waypoints -> return to HOME (Abhinav's definition: return to reference configuration).
"""
import numpy as np

pi = np.pi
HOME = np.array([pi, -0.3, -0.9, 0.0, 1.2, 0.0])
POSE = {
    'PICK':      np.array([pi, 0.5, -1.3, 0.0, 0.9, 0.0]),     # reach down-left
    'PICK_UP':   np.array([pi, 0.2, -1.3, 0.0, 1.2, 0.0]),     # lift
    'PLACE_HI':  np.array([pi, -0.55, -1.35, 0.0, 1.3, 0.0]),  # arm raised, forearm up-left
    'Y_TOWARD':  np.array([pi + 1.1, 0.1, -0.8, 0.0, 1.3, 0.0]),  # base yawed toward camera (out of plane)
    'Y_AWAY':    np.array([pi - 0.9, 0.1, -0.8, 0.0, 1.3, 0.0]),  # base yawed away from camera
    'YAWP':      np.array([pi + 0.8, -0.3, -0.9, 0.0, 1.2, 0.0]),  # pure yaw, same arm shape
    'YAWM':      np.array([pi - 0.8, -0.3, -0.9, 0.0, 1.2, 0.0]),
    'WR1':       np.array([pi, -0.3, -0.9, 0.0, 0.6, 1.5]),     # wrist-only
    'WR2':       np.array([pi, -0.3, -0.9, 0.0, 1.8, -1.5]),
}

# program = list of (waypoint_name, dwell_after_seconds)
PROGRAMS = {
    'inplane':   [('PICK_UP', 0.2), ('PICK', 0.8), ('PICK_UP', 0.2), ('PLACE_HI', 0.8)],
    'outplane':  [('Y_TOWARD', 0.8), ('Y_AWAY', 0.8)],
    'pureyaw':   [('YAWP', 0.8), ('YAWM', 0.8)],
    'wrist':     [('WR1', 0.8), ('WR2', 0.8)],
    'viahome':   [('PICK', 0.8), ('HOME_NEAR', 0.5), ('PLACE_HI', 0.8)],
    'pallet':    [('PICK', 0.8), ('PLACE_K', 0.8)],
}


def min_jerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class Timeline:
    """piecewise trajectory: list of (t0, t1, q0, q1) segments (dwell = q0==q1)"""

    def __init__(self):
        self.segs = []
        self.t = 0.0
        self.q = HOME.copy()
        self.depart, self.arrive = [], []

    def dwell(self, d):
        if d > 0:
            self.segs.append((self.t, self.t + d, self.q.copy(), self.q.copy()))
            self.t += d

    def move(self, q1, vmax, speed):
        dq = np.abs(q1 - self.q).max()
        dur = max(0.5, dq / vmax) / speed
        self.segs.append((self.t, self.t + dur, self.q.copy(), np.array(q1, float).copy()))
        self.t += dur
        self.q = np.array(q1, float).copy()

    def __call__(self, t):
        for t0, t1, q0, q1 in self.segs:
            if t <= t1:
                s = min_jerk((t - t0) / max(t1 - t0, 1e-9))
                return q0 + (q1 - q0) * s
        return self.segs[-1][3].copy()


def build(program, rng, n_cycles=8, amp=1.0, vmax=0.55, speed_jitter=0.1, home_dwell=1.5, home_dwell_jitter=0.5,
          pause_prob=0.0, pause_range=(4, 10), near_frac=0.12, partial_end=False, pallet_targets=None):
    """returns (timeline, gt dict).  amp scales every waypoint's offset from HOME (1.0 = full motion)."""
    tl = Timeline()
    tl.dwell(home_dwell * (0.5 + rng.random()))                        # start at HOME (reference)
    steps = PROGRAMS[program]
    if pallet_targets is None:
        pallet_targets = [np.array([pi + dy, 0.5 - dr, -1.3 + 0.8 * dr, 0, 0.9, 0])
                          for dy in (-0.5, -0.25, 0.0, 0.25, 0.5) for dr in (0.0, 0.35)]

    def wp(name, k):
        if name == 'HOME_NEAR':
            q = HOME + near_frac * (POSE['PLACE_HI'] - HOME)
        elif name == 'PLACE_K':
            q = pallet_targets[k % len(pallet_targets)]
        else:
            q = POSE[name]
        return HOME + amp * (q - HOME)

    for k in range(n_cycles + (1 if partial_end else 0)):
        partial = partial_end and k == n_cycles
        sp = 1.0 + speed_jitter * (2 * rng.random() - 1)
        if k > 0:
            d = max(0.0, home_dwell + home_dwell_jitter * (2 * rng.random() - 1))
            if rng.random() < pause_prob:
                d += rng.uniform(*pause_range)
            tl.dwell(d)
        if not partial:
            tl.depart.append(tl.t)
        n_steps = len(steps) if not partial else max(1, int(rng.integers(1, len(steps))))
        for name, dw in steps[:n_steps]:
            tl.move(wp(name, k), vmax, sp)
            tl.dwell(dw * sp)
        if partial:
            break
        tl.move(HOME, vmax, sp)
        tl.arrive.append(tl.t)
    tl.dwell(0.5 + rng.random())
    gt = dict(count=n_cycles, depart=np.array(tl.depart), arrive=np.array(tl.arrive), duration=tl.t)
    return tl, gt


class Continuous:
    """no dwell anywhere: smooth closed loop through HOME, variable period per cycle"""

    def __init__(self, rng, n_cycles=8, period=10.0, jitter=0.1, amp=1.0, partial_end=False):
        Ts = period * (1 + jitter * (2 * rng.random(n_cycles + 1) - 1))
        self.bounds = np.concatenate([[0], np.cumsum(Ts)])
        self.amp = amp
        extra = (0.3 + 0.4 * rng.random()) * Ts[-1] if partial_end else 0.0
        self.T = self.bounds[n_cycles] + extra
        self.gt = dict(count=n_cycles, depart=self.bounds[:n_cycles].copy(), arrive=self.bounds[1:n_cycles + 1].copy(),
                       duration=self.T)
        self.d1 = POSE['PICK'] - HOME
        self.d2 = POSE['PLACE_HI'] - HOME

    def __call__(self, t):
        k = int(np.searchsorted(self.bounds, t, side='right') - 1)
        k = min(k, len(self.bounds) - 2)
        ph = 2 * pi * (t - self.bounds[k]) / (self.bounds[k + 1] - self.bounds[k])
        # closed loop (circle in joint space) through HOME at phase 0, passing HOME at full speed (no stop)
        return HOME + self.amp * (0.5 * (1 - np.cos(ph)) * self.d1 + 0.5 * np.sin(ph) * self.d2)
