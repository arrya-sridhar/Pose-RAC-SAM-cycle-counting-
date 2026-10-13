#!/bin/bash
# One-time set-up on the lab cluster. Light work only, so it is fine on the login node: folders, SAM weights and three
# small pure-python packages installed into YOUR folder. No shared conda environment is modified.
#
#   cd <your clone of this repo>
#   export WORK=/scratch/psaxena/<your_name>_poserac      # your own folder; never point it at someone else's
#   bash hpc/setup_hpc.sh
#
# Note: /scratch is cleared after about 15 days. Re-run this script if $WORK has gone, and copy any results you want
# to keep into results/<version>/ in the repo.
set -e
: "${WORK:?set WORK to your own folder, e.g. export WORK=/scratch/psaxena/<your_name>_poserac}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/home/psaxena/.conda/envs/krishna/bin/python}        # shared env, used read-only (torch + CUDA, cv2, scipy)
DATASET=${DATASET:-/home/psaxena/Harshitha/dataset_out}       # videos/ and ground_truth/ of the 200-video dataset
mkdir -p "$WORK"/results "$WORK"/logs "$WORK"/lists "$WORK"/pylib "$REPO"/src/weights

if [ ! -f "$REPO"/src/weights/sam_b.pt ]; then
  curl -sL -o "$REPO"/src/weights/sam_b.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_b.pt
fi
echo "weights md5: $(md5sum "$REPO"/src/weights/sam_b.pt | cut -d' ' -f1)  (expected 01ec64d29a2fca3f0661936605ae66f8)"

"$PY" -m pip install --quiet --no-deps --target "$WORK"/pylib segment-anything pytorch-metric-learning imageio-ffmpeg 2>&1 | tail -2
ls "$WORK"/pylib

# video lists for the 200-video run: all.txt, queue.txt (longest first; jobs share it) and three balanced shards
WORK="$WORK" DATASET="$DATASET" python3 - <<'PY'
import json, glob, os
D, DS = os.environ['WORK'], os.environ['DATASET']
fs = sorted(glob.glob(f'{DS}/ground_truth/*.json'))
v = [(json.load(open(f))['total_duration_sec'] * 2, os.path.basename(f)[:-5]) for f in fs]
v = [x for x in v if os.path.exists(f'{DS}/videos/{x[1]}.mp4')]
open(f'{D}/lists/all.txt', 'w').write('\n'.join(n for _, n in sorted(v, key=lambda x: x[1])) + '\n')
open(f'{D}/lists/queue.txt', 'w').write('\n'.join(n for _, n in sorted(v, reverse=True)) + '\n')
K = 3; shards = [[] for _ in range(K)]; load = [0] * K
for fr, n in sorted(v, reverse=True):                       # longest first -> balanced shards
    i = load.index(min(load)); shards[i].append(n); load[i] += fr
for i, s in enumerate(shards):
    open(f'{D}/lists/shard{i}.txt', 'w').write('\n'.join(s) + '\n')
    print(f'shard{i}: {len(s)} videos, {int(load[i])} frames')
print('videos with mp4:', len(v))
PY
echo "set-up done: WORK=$WORK"
