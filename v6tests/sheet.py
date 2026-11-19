"""Strip of frames with the v6 outline drawn in green, at given times.
    python v6tests/sheet.py <v6 run dir> <frames dir> <grid fps> <out.jpg> <t1> <t2> ...
"""
import sys, numpy as np, cv2
rd, fdir, gfps, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
ts = [float(x) for x in sys.argv[5:]]
GW, GH = 640, 360
z = np.load(f'{rd}/trace.npz'); t = z['t']; M = np.unpackbits(z['masks'], axis=1)[:, :GW * GH].reshape(len(t), GH, GW).astype(bool)
tiles = []
for x in ts:
    i = int(np.argmin(np.abs(t - x))); fr = int(round(t[i] * gfps)) + 1
    im = cv2.resize(cv2.imread(f'{fdir}/{fr:06d}.jpg'), (GW, GH)); m = M[i]
    im[m] = (0.4 * im[m] + 0.6 * np.array([60, 210, 60])).astype(np.uint8)
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, cnt, -1, (0, 255, 0), 2)
    cv2.putText(im, f'{t[i]:.0f} s', (12, 40), 0, 1.2, (255, 255, 255), 5); cv2.putText(im, f'{t[i]:.0f} s', (12, 40), 0, 1.2, (0, 0, 0), 2)
    tiles.append(cv2.copyMakeBorder(im, 0, 0, 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
cv2.imwrite(out, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('wrote', out)
