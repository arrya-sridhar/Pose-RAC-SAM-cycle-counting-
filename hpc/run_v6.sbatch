#!/bin/bash
#SBATCH --job-name=arrya_v6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# v6 test runs: sbatch -p <one gpu partition> -o $WORK/logs/%x_%j.out --export=ALL,WORK=$WORK hpc/run_v6.sbatch <jobs.json> <out folder>
# Several cluster jobs can work through the same list: add SHARED=1 to --export (each video is claimed before it is
# run; a stopped job can simply be resubmitted). Videos are decoded while running (v6.1), nothing is written to disk
# but the results. 4 CPUs: the H200 partition allows at most 4 per GPU.
# Uses the v5 run's packages and SAM weights read-only (no shared conda env is modified). Set EDITED=1 to build the
# 30 fps edited videos first (fixed seed, same plan as v5's 2 fps ones).
: "${WORK:?set WORK}"
JOBS=${1:?jobs json}; OUT=${2:?output folder}
REPO=${REPO:-$WORK/code}
PY=${PY:-/home/psaxena/.conda/envs/krishna/bin/python}
export PYTHONPATH=$WORK/pylib
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
FB=$(ls $WORK/pylib/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-* | head -1)
LT=${TMPDIR:-/tmp}; cp $FB $LT/ffmpeg_$SLURM_JOB_ID && chmod +x $LT/ffmpeg_$SLURM_JOB_ID && export FFMPEG=$LT/ffmpeg_$SLURM_JOB_ID
echo "node $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  jobs $JOBS  start $(date)"
if [ -n "$EDITED" ] && [ ! -f $WORK/edited30/edit_10/gt.json ]; then
  EDIT_FPS=30 NACHI_CLIPS=/home/psaxena/Harshitha/nachi_robot_videos_trim $PY $REPO/src/make_edited_tests.py $WORK/edited30 || exit 1
fi
$PY -u $REPO/src/run_v6.py $JOBS --out $OUT --device cuda ${SHARED:+--shared}
STATUS=$?
rm -f $FFMPEG
echo "end $(date)  exit $STATUS"
exit $STATUS
