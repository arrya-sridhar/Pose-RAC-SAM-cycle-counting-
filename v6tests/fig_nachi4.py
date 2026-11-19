"""Figure: v5 vs v6 outline on nachi_4_seq_014 at the 6th return home (where v5 merged two cycles).
    python v6tests/fig_nachi4.py <v5 results dir> <v6 run dir> <frames dir (30 fps)> <out.jpg>
"""
import sys, json, numpy as np, cv2
v5d, v6d, fdir, out = sys.argv[1:5]
GW, GH = 640, 360
r6 = json.load(open(f'{v6d}/result.json')); z6 = np.load(f'{v6d}/trace.npz')
t6 = z6['t']; M6 = np.unpackbits(z6['masks'], axis=1)[:, :GW * GH].reshape(len(t6), GH, GW).astype(bool)
z5 = np.load(f'{v5d}/outlines.npz'); n5 = z5['ours__masks'].shape[0]
M5 = np.unpackbits(z5['ours__masks'], axis=1)[:, :GW * GH].reshape(n5, GH, GW).astype(bool)
ts = r6['returns_s'][5] + 2.0                         # robot settled at home after the 6th return
i5 = int(round(ts * 2)); i6 = int(np.argmin(np.abs(t6 - ts))); fr = int(round(t6[i6] * 30)) + 1
img = cv2.resize(cv2.imread(f'{fdir}/{fr:06d}.jpg'), (GW, GH))
def iou(a, b): return (a & b).sum() / max((a | b).sum(), 1)
panels = []
for name, m, ref in [('v5', M5[i5], M5[0]), ('v6', M6[i6], M6[int(z6['i0'])])]:
    o = img.copy(); o[m] = (0.45 * o[m] + 0.55 * np.array([60, 200, 60])).astype(np.uint8)
    cnt, _ = cv2.findContours(ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cnt, -1, (0, 140, 255), 2)
    ys, xs = np.where(ref | m); x0, x1 = max(xs.min() - 40, 0), min(xs.max() + 40, GW); y0, y1 = max(ys.min() - 30, 0), min(ys.max() + 30, GH)
    panels.append((name, o[y0:y1, x0:x1], round(float(iou(m, ref)), 2)))
h = max(p[1].shape[0] for p in panels)
row = [cv2.copyMakeBorder(cv2.resize(p[1], (int(p[1].shape[1] * h / p[1].shape[0]), h)), 0, 0, 0, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in panels]
big = cv2.resize(np.hstack(row), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(out, big, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(json.dumps(dict(time_s=round(float(ts), 2), v5_overlap=panels[0][2], v6_overlap=panels[1][2], frame=fr)))
