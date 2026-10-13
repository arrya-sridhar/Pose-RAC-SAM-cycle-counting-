"""Mask -> fixed-length "pose" vectors (the robot analogue of PoseRAC's 33x3 BlazePose keypoints).

grid  : 11x9 occupancy grid of the mask inside the (fixed) workspace box            -> 99-d
polar : 33 angular sectors around the robot base point, (r_max, r_mean, mass) each    -> 99-d  (33 "keypoints" x 3)
geom  : Abhinav's mask geometry (area, centroid, bbox w/h, orientation as sin/cos 2θ)  -> 7-d
All masks are 640x360; boxes/points given in 1280x720 pixels.
"""
import numpy as np, cv2

GW, GH = 640, 360


def unpack(a):
    return np.unpackbits(a, axis=-1)[..., :GW * GH].reshape(*a.shape[:-1], GH, GW).astype(bool)


def feat_grid(M, wb, nx=11, ny=9):
    x0, y0, x1, y1 = (np.asarray(wb) / 2).astype(int)
    out = np.zeros((len(M), nx * ny), np.float32)
    for i, m in enumerate(M):
        crop = m[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
        out[i] = cv2.resize(crop, (nx, ny), interpolation=cv2.INTER_AREA).reshape(-1)
    return out


def feat_polar(M, base, scale, n=33):
    bx, by = np.asarray(base) / 2
    ys, xs = np.mgrid[0:GH, 0:GW]
    dx, dy = xs - bx, by - ys                          # y up
    ang = np.clip(np.arctan2(dy, dx), 0, np.pi)        # upper half plane (robot is above its base)
    sec = np.minimum((ang / np.pi * n).astype(int), n - 1)
    rad = np.sqrt(dx ** 2 + dy ** 2) / (scale / 2)
    out = np.zeros((len(M), n, 3), np.float32)
    for i, m in enumerate(M):
        s, r = sec[m], rad[m]
        tot = max(1, m.sum())
        cnt = np.bincount(s, minlength=n)
        rmax = np.zeros(n); np.maximum.at(rmax, s, r)
        rsum = np.bincount(s, weights=r, minlength=n)
        out[i, :, 0] = rmax
        out[i, :, 1] = rsum / np.maximum(cnt, 1)
        out[i, :, 2] = cnt / tot * 5.0
    return out.reshape(len(M), -1)


def feat_geom(M):
    out = np.zeros((len(M), 7), np.float32)
    for i, m in enumerate(M):
        ys, xs = np.where(m)
        if len(xs) < 5:
            continue
        cx, cy = xs.mean(), ys.mean()
        mu = cv2.moments(m.astype(np.uint8), binaryImage=True)
        th = 0.5 * np.arctan2(2 * mu['mu11'], mu['mu20'] - mu['mu02'] + 1e-9)
        out[i] = [m.mean() * 10, cx / GW, cy / GH, (xs.max() - xs.min()) / GW, (ys.max() - ys.min()) / GH,
                  np.sin(2 * th), np.cos(2 * th)]
    return out


def compute(M, kind, wb, base):
    if kind == 'grid':
        return feat_grid(M, wb)
    if kind == 'polar':
        return feat_polar(M, base, scale=(wb[3] - wb[1]))
    if kind == 'geom':
        return feat_geom(M)
    raise ValueError(kind)


def mask_iou_to(M, ref):
    inter = (M & ref).sum((1, 2)); uni = (M | ref).sum((1, 2))
    return inter / np.maximum(uni, 1)
