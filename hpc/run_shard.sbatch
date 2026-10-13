#!/bin/bash
#SBATCH --job-name=poserac_200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1-23:00:00
# Run the pipeline on the 200 dataset videos. Submit from the repo folder, after hpc/setup_hpc.sh.
# Start 1-3 of these at most; they share the work through lists/queue.txt and skip finished videos, so a stopped job can
# simply be resubmitted:
#   sbatch -p <gpu partition> -o $WORK/logs/%x_%j.out --export=ALL,WORK=$WORK hpc/run_shard.sbatch queue.txt
# The v5 run (2 GPUs: RTX PRO 6000 + H100) took 3 h 47 min; start-up alone is about 10 min (slow imports on the cluster).
: "${WORK:?set WORK (pass --export=ALL,WORK=\$WORK)}"
LIST=${1:-queue.txt}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-/home/psaxena/.conda/envs/krishna/bin/python}
export DS_ROOT=${DATASET:-/home/psaxena/Harshitha/dataset_out}
export PYTHONPATH=$WORK/pylib
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
# a private copy of the ffmpeg that ships with imageio-ffmpeg (the cluster's ffmpeg module did not work on every node)
FB=$(ls $WORK/pylib/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-* | head -1)
LT=${TMPDIR:-/tmp}; cp $FB $LT/ffmpeg_$SLURM_JOB_ID && chmod +x $LT/ffmpeg_$SLURM_JOB_ID && export FFMPEG=$LT/ffmpeg_$SLURM_JOB_ID
echo "node $(hostname)  list $LIST  code $(git -C $REPO describe --always --dirty --tags 2>/dev/null)  start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -u $REPO/src/run_batch.py --list $WORK/lists/$LIST --out $WORK/results --device cuda --tmp ${TMPDIR:-/tmp} \
    --setup $REPO/configs/job_setups.json
$PY $REPO/src/score_200.py $WORK/results $WORK/lists/all.txt > $WORK/results/metrics_after_${LIST%.txt}.txt 2>&1
rm -f $FFMPEG
echo "end $(date)"
