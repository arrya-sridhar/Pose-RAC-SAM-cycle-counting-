#!/bin/bash
#SBATCH --job-name=poserac_regress
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# Full regression check (tracking + counting) on the 10 development and 9 edited videos, compared with v5.
# Submit from the repo folder, after hpc/setup_hpc.sh. Give each run a name that says what changed:
#   sbatch -p <gpu partition> -o $WORK/logs/%x_%j.out --export=ALL,WORK=$WORK hpc/regression_check.sbatch v6_static_fix
# The first run also builds the input frames in $WORK/regression_data (a few minutes). The verdict is at the end of the
# log and in $WORK/regression_runs/<name>/regression_report.txt.
: "${WORK:?set WORK (pass --export=ALL,WORK=\$WORK)}"
NAME=${1:?give the run a name, e.g. v6_static_fix}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-/home/psaxena/.conda/envs/krishna/bin/python}
DATASET=${DATASET:-/home/psaxena/Harshitha/dataset_out}
CLIPS=${CLIPS:-/home/psaxena/Harshitha/nachi_robot_videos_trim}
export PYTHONPATH=$WORK/pylib
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
FB=$(ls $WORK/pylib/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-* | head -1)
LT=${TMPDIR:-/tmp}; cp $FB $LT/ffmpeg_$SLURM_JOB_ID && chmod +x $LT/ffmpeg_$SLURM_JOB_ID && export FFMPEG=$LT/ffmpeg_$SLURM_JOB_ID
echo "node $(hostname)  run $NAME  code $(git -C $REPO describe --always --dirty --tags 2>/dev/null)  start $(date)"
$PY $REPO/src/regression_check.py prepare --dataset $DATASET --clips $CLIPS --data $WORK/regression_data || exit 1
$PY -u $REPO/src/regression_check.py run --data $WORK/regression_data --out $WORK/regression_runs/$NAME --device cuda
STATUS=$?
rm -f $FFMPEG
echo "end $(date)  (exit $STATUS: 0 = pass, 1 = fail)"
exit $STATUS
