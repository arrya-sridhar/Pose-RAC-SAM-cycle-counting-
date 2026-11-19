# SAM-Box PoseRAC: counting robot work cycles from video

Counts the work cycles of an industrial robot arm (Nachi) filmed by a fixed camera, with no labelled training data.

Each frame, [SAM](https://github.com/facebookresearch/segment-anything) (ViT-Base) outlines the robot, prompted by a box that follows the robot plus one click on its base. Each outline is compared with the robot's outline on the first frame (its home pose). The two-threshold trigger from [PoseRAC](https://github.com/MiracleDance/PoseRAC) then counts each home → away → home trip as one cycle. The pipeline also reports the time of each return home.

The current version is **v6.1** (git tag `v6.1`). It is fully automatic: give it videos and it finds the robot by
itself, counts the cycles and times each one ([quick start](#quick-start-v61)). It is made of `count_cycles.py` (one
command for everything), `src/auto_setup.py` (finds the robot) and `src/run_v6.py` (tracks and counts: v6's method,
see [v6](#v6-dynamic-frame-rate-and-set-up-memory)). v5 (`src/run_real.py`, tag `v5`) is kept unchanged for comparison.

Every version has release notes and files on the [Releases page](../../releases); what changed in each version is in
[`CHANGELOG.md`](CHANGELOG.md), and how versions are numbered, kept apart and released is in [`RELEASING.md`](RELEASING.md).

**Before changing anything, read [`HISTORY.md`](HISTORY.md).** It lists every version, what each change fixed or broke,
what was tried and dropped, the known failures, and the fixes proposed but not yet tested. Before calling a change an
improvement, run the [regression check](#regression-check).

## Quick start (v6.1)

**Install once** (Python 3, tested with 3.14; and ffmpeg: `brew install ffmpeg`, `apt install ffmpeg`, or the copy
that comes with the `imageio-ffmpeg` package in requirements.txt):

```bash
git clone https://github.com/arrya-sridhar/Pose-RAC-SAM-cycle-counting-.git
cd Pose-RAC-SAM-cycle-counting-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The SAM ViT-B weights (375 MB) are downloaded to `src/weights/` on the first run and checked.

**Run it on your videos:**

```bash
python count_cycles.py my_video.mp4 --out results/              # one video
python count_cycles.py a.mp4 b.mp4 c.mp4 --out results/         # several
python count_cycles.py my_videos_folder/ --out results/         # every video in a folder
```

For each video it finds the robot (automatic setup), then follows its outline, counts every home → away → home trip
and times each cycle. Results, in `results/<video name>/`:

| File | What it holds |
| --- | --- |
| `setup.json` | where it found the robot: box, base point, home frame, and how each was chosen |
| `setup.jpg` | the home frame with the robot outline, box and base point drawn on: check it at a glance |
| `result.json` | the cycle count, warning flags and run details (and the score, if labels were given) |
| `cycles.csv` | one row per cycle: start, end, duration, wait at home, period |
| `trace.npz` | the robot's outline and the home score over time (for plots or later analysis) |

With several videos, `results/summary/` also gets `videos.csv` (one row per video), `all_cycles.csv` and `summary.txt`.
A video that already has a `result.json` is skipped, so a stopped run can simply be started again.

**What the videos need:** a fixed camera filming one robot, with the robot visible and at its home pose in the first
frame (after a fade-in, the first still frame where it is visible is used). Any common format, resolution and frame
rate (frames are read at 1280×720, 30 per second).

**Options:**
- `--labels <folder>`: known cycles, `<folder>/<video name>.json` (NACHI label format: `total_cycles` and `cycles`
  with `start_time`/`end_time` in seconds). The result is then scored.
- `--setup <file or folder>`: give the setup by hand instead (one JSON file for every video from the same camera, or
  `<folder>/<video name>.json`): `init_box` [x0, y0, x1, y1] and `base_point` [x, y] in 1280×720 pixels of the home
  frame, `init_frame` (the home frame's index at 2 per second, default 0).
- `--device auto|cuda|mps|cpu`: `auto` (default) picks a CUDA GPU, else Apple Silicon, else the CPU.
- `--skim <seconds>`: how much of the start of the video the automatic setup looks at (default 120).

**Check `setup.jpg` before trusting a count.** The automatic setup has been tested on 2 NACHI videos only, both
from program nachi_1 (9/9 and 7/7 cycles). The tracking and counting behind it were tested on 400 videos with the
setup given by hand (results below). If the outline in `setup.jpg` misses part of the robot or takes in something
else, give the setup by hand with `--setup`, delete that video's results folder and run it again. A warning in
`result.json` (`flags`) also means: look at that video before trusting its count.

**Speed:** about 0.3 s per second of video on a cluster GPU (H200; the 400-video run), and about 3 s per second of
video on an Apple M-series laptop GPU. CPU only works but is much slower. `count_cycles.py` has been run on a Mac
(Apple GPU); on the cluster, the 400-video run used `src/run_v6.py` with job lists (see
[v6](#v6-dynamic-frame-rate-and-set-up-memory)).

## Results

**v6.1 on all 400 NACHI videos** (lab cluster, 30 Oct 2026; setup given by hand per program, as in v5's run;
tables in [`results/v6.1_2026-10-30/`](results/v6.1_2026-10-30/)):

| Video set | Exact count | Within one | MAE | Cycle ends within 1 s of the label* |
| --- | --- | --- | --- | --- |
| Original 200 (3,141 cycles) | **196/200** (98.0%) | 199/200 | 0.0068 | 3,122 of 3,139 (99.5%) |
| 200 non-uniform (3,184 cycles; waits, holds, speed changes) | **187/200** (93.5%) | 195/200 | 0.0101 | 3,156 of 3,164 (99.7%) |

\* after removing one constant offset per video: the labels of some programs sit a fixed time away from the robot's
return home.

- Original 200: v5's three misses (nachi_4) are now exact. Four other videos are missed: nachi_10_seq_014 (5 counted,
  3 labelled), nachi_10_seq_015 (3 vs 2), nachi_9_seq_004 (21 vs 22) and nachi_9_seq_015 (6 vs 7). v5 got 197/200 on
  the same videos.
- Non-uniform 200: 11 of the 13 misses are nachi_4 (9/20 exact; each one to four cycles short). The others:
  nachi_10_nu08 (31 vs 30) and nachi_9_nu20 (2 vs 1).
- Per program, per video and per cycle: `programs.csv`, `videos.csv`, `all_cycles.csv` in each folder.

**v5 on the original 200 videos** (199,161 frames at 2 fps, 3,141 cycles, 1–30 cycles per video):

| Method | Exact count | OBO (within ±1) | MAE | Mean error (cycles) | RMSE |
| --- | --- | --- | --- | --- | --- |
| **v5 (this repo)** | **98.5%** (197/200) | **100%** | **0.002** | **0.01** | **0.12** |
| Fixed-box SAM + area autocorrelation (earlier pipeline) | 83.5% (167/200) | 83.5% | 0.221 | 2.27 | 5.88 |
| Fixed-box outlines + v5 counter | 70.5% | 81.5% | 0.158 | 2.19 | 5.58 |
| v5 outlines + autocorrelation | 83.5% | 84.0% | 0.143 | 2.12 | 5.86 |

MAE is the PoseRAC/RepCount definition, mean(|predicted − true| / true).

- v5's three misses are nachi_4_seq_014, 016 and 018. Each is one cycle short.
- On videos with 21–30 cycles, v5 is 100% exact and the earlier pipeline is 67%.
- v5 outlines cover 9.1% of the picture on average, against 15.3% for the fixed box. The fixed box picks up strips of wall.
- Sudden outline jumps: 715 for v5, 10,402 for the fixed box.
- The earlier pipeline's numbers come from our re-implementation of it.
- Full numbers: [`results/dataset_200/metrics.txt`](results/dataset_200/metrics.txt). Per video: [`per_video.csv`](results/dataset_200/per_video.csv).

**Edited test videos with uneven cycles** (9 videos, 33 cycles). Built from the real one-cycle recordings, each cycle runs at 0.8×, 1× or 1.25× speed, the robot waits 0–15 s at home between cycles, and 3 videos also stop mid-cycle:

| | Exact count |
| --- | --- |
| **v5** | **9/9** |
| Fixed-box SAM + autocorrelation | 6/9 |

All 33 cycle ends were detected exactly once, with a median of 0.8 s early.

**Development set** (the 10 real videos used while building v5): v5 gets 10/10, and the base stitched pipeline (v1) gets 5/10.

![Outlines on real frames: fixed box vs base stitched pipeline vs v5](docs/figures/fig1_outlines.png)

![edit_1: v5 detects every return home despite speed changes and waits](docs/figures/fig2_edit1_timeline.png)

**Important caveats**

- **The dataset repeats itself.** All 200 dataset videos are built from the same 10 one-cycle recordings, repeated back to back, so every cycle within a video is pixel-identical.
- **v5 is not independent of the dataset.** v5 was developed on 10 of those videos, so the 200-video result shows it holds up on longer videos but is not an independent test. The edited videos are the fairest test here.
- **New footage is still needed.** A new recording of the robot running several cycles in a row is the next real test.

## How v5 works

1. **Set up once per camera.** Draw a box around the robot on the first frame and click once on its base; the robot should be at home in that frame. These are stored in `configs/*.json`.
2. **Track the robot outline every frame (SAM ViT-B, 2 fps).**
   - **Candidates:** the box around the last good outline, and a second box that also covers movement near the robot. Each box gives 3 SAM outlines, so 6 candidates.
   - **Clean-up:** specks, thin slivers and small detached pieces are removed.
   - **Checks:** an outline is rejected if it adds a static region, meaning new area where the picture did not change since the last good outline (the camera is fixed, so a wall panel fails). It is also rejected if its size is outside 0.5–2× the recent size or 0.4–2× the first-frame size.
   - **Choice:** the candidate most like the last good outline is used.
   - **Getting unstuck:** if nothing passes for 1 s, it also tries the first-frame box and a box around everything that moved.
   - **Fallback:** if nothing passes at all, the last good outline is kept.
3. **Count cycles.** Measure the difference between each outline and the first-frame outline. A cut-off picked automatically from the data (Otsu's method) turns it into a home score, and PoseRAC's smoothed two-threshold trigger (on above 0.78, off below 0.4) counts each home → away → home trip.

Version history (v1 to v5, with results and side effects): [`HISTORY.md`](HISTORY.md). Earlier tracker versions are in
`history/`.

## v6: dynamic frame rate and set-up memory

`src/run_v6.py` (v5's `src/run_real.py` is unchanged). Full description, settings and test results: `HISTORY.md`, Section 10.

- **Dynamic frame rate.** SAM runs at 2 fps (v5's frames) and speeds up to 6 fps when the robot moves fast, arrives at or
  leaves home, or the tracker is unsure, and to every frame (30 fps) while the robot moves fast near home. It goes back
  over the frames it skipped, so the start of a movement is not missed.
- **Set-up memory.** Parts of the robot that were in the set-up outline and still look the same may come back into the
  outline (v5 could never re-add a still part it had lost: nachi_4's base).
- **Results** (`results/v6_2026-10-26/`): the 19 regression videos stay exact; the 3 nachi_4 videos v5 got wrong are
  now exact; the simulated palletizing videos go from 1/5, 1/6, 1/10, 4/12, 1/40 to 4/5, 5/6, 9/10, 11/12, 39/40 (the
  last cycle of each ends with the video, so its return cannot be seen); computing time 1.0–1.3× v5 on NACHI-style
  videos, ~4× on robots that move fast all the time.

**v6.1** (same tracker and counter as v6): videos are decoded while running (no frame extraction), each run writes a
per-cycle table (`cycles.csv`: start, end, duration, wait at home, period, frame numbers, matched labelled cycle), and
several cluster jobs can share one list (`--shared`, `SHARED=1`). It adds the automatic setup (`src/auto_setup.py`) and
one command for everything (`count_cycles.py`, see [quick start](#quick-start-v61)). Details: `HISTORY.md`, v6.1.
The cluster runs use job lists:

```
python src/run_v6.py jobs.json --out <folder> --device cuda      # jobs: v6tests/make_jobs.py, v6tests/make_jobs_200.py
sbatch -p <gpu partition> -c 4 -o $WORK/logs/%x_%j.out --export=ALL,WORK=$WORK hpc/run_v6.sbatch <jobs.json> <out>
python v6tests/collect_cycles.py <out> --out <tables folder>     # all_cycles.csv, videos.csv, programs.csv, summary.txt
python v6tests/retable.py <out>                                  # rebuild the tables from saved traces (no GPU)
```

## Repository layout

```
count_cycles.py         v6.1: one command, videos in, setup + counts + per-cycle times out
HISTORY.md              the truth sheet: versions, what was tried, known failures, proposed fixes, ground rules
CHANGELOG.md            one short entry per version: date, what changed, headline results
RELEASING.md            how versions are numbered, branched, tagged and released
src/
  run_v6.py             v6.1 tracker and counter (v6's method): dynamic frame rate, set-up memory, per-cycle table
  auto_setup.py         v6.1 automatic setup: finds the robot's box, base point and home frame in a video
  count_video.py        v5: count cycles in any video, box and click given by hand
  run_real.py           v5 pipeline + evaluation on folders of frames with ground truth
  regression_check.py   compare the current code with v5 on the 10 development + 9 edited videos
  run_batch.py          run v5 over many dataset videos (used for the 200-video run)
  score_200.py          metrics: exact, OBO, MAE, RMSE, bias, per program, per cycle range, boundaries
  analyze_real.py       scoring for the 10 real videos
  analyze_edited.py     scoring for the edited test videos
  make_edited_tests.py  builds the edited test videos from the one-cycle clips (fixed seed)
  poserac.py            PoseRAC model, loss and official counting trigger (ported)
  features.py           outline -> features (11x9 occupancy grid, geometry)
  evaluate.py           synthetic-benchmark evaluation; Otsu threshold used by the counter
  run_sam.py, reproduce.py, experiments.py, synth/, assets/   synthetic benchmark
configs/                box + base click per camera setup (real videos, dataset programs, edited set)
hpc/                    SLURM scripts for the lab cluster (setup, 200-video run, regression check)
history/                run_real_v1.py ... run_real_v4.py (v5 is src/run_real.py)
results/
  real_10_v5/           v5 on the 10 development videos: results, ground truth, outlines (used by the regression check)
  edited_9/             v5 on the 9 edited videos: results, ground truth, outlines (used by the regression check)
  dataset_200/          v5 on all 200 videos: metrics + per-video results
  new_videos_2026-10-14/  v5 and v5.1 on the 14 Oct videos (simulated palletizing, OP75): one result file per run
  v6_2026-10-26/        v6 tests: regression videos, nachi_4, palletizing, ablation, stress, synthetic
  v6.1_2026-10-30/      v6.1: streaming checks, and task1/ + task2/ = the 400-video run (tables per cycle, video, program)
newtests/               v5.1 experiment on the 14 Oct videos (6 fps): script and job lists
v6tests/                job lists and scripts for the v6 / v6.1 runs (make_jobs_200.py, collect_cycles.py, retable.py)
reports/                PDF reports (black on white)
docs/figures/           figures used above and in HISTORY.md
docs/releases/          release notes, one file per version (also the text of each GitHub release)
```

## Setup

`count_cycles.py` downloads the SAM weights by itself ([quick start](#quick-start-v61)). The other scripts expect them in
`src/weights/sam_b.pt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p src/weights
curl -L -o src/weights/sam_b.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_b.pt
# md5 01ec64d29a2fca3f0661936605ae66f8 (the same weights as Meta's sam_vit_b_01ec64.pth)
```

Runs on CUDA, Apple Silicon (`--device mps`) or CPU. Measured speeds:

| Device | Speed |
| --- | --- |
| RTX PRO 6000 | about 0.12 s per frame |
| Apple M3 | about 1.2 s per frame |
| CPU (2 cores) | 15–25 s per frame |

## Usage

**Count cycles in a new video with v5** (for v6.1, use `count_cycles.py`). The box and click are in 1280×720 pixel
coordinates on the reference frame:

```bash
python src/count_video.py --video clip.mp4 --box 430 180 950 719 --click 754 598 --out result.json
```

To get the box and click, save the first frame at 1280×720 and read the pixel positions off it in any image viewer
that shows coordinates:

```bash
ffmpeg -i clip.mp4 -vf scale=1280:720 -frames:v 1 first.png
```

- The **box** should hug the robot tightly. A loose box makes SAM pick the wall (see HISTORY.md, Section 2).
- The **click** goes on the robot's base.
- The robot must be **at its home pose and fully visible** in that frame. If the video starts with a fade or the robot
  is mid-move, pick a later frame and pass `--init_frame <frame index at 2 fps>`.
- The output gives the cycle count, the time of each return home, the outline size and the number of held frames.
  An outline far above ~10–15% of the frame, or many held frames, means the tracker has lost the robot.
- Known weak spot: a robot that never pauses at home, or very short cycles, at 2 fps (HISTORY.md, Section 6.2).

**Re-run the 10 real videos.** Each folder needs `frames/000001.jpg...` (extracted at 2 fps, 1280×720) and `gt.json`:

```bash
ffmpeg -i nachi_1_seq_007.mp4 -vf fps=2,scale=1280:720 -q:v 3 -start_number 1 data/real/nachi_1_seq_007/frames/%06d.jpg
python src/run_real.py --root data/real --setup configs/real_setup.json --device cuda
python src/analyze_real.py data/real        # needs data/real/setup.json (copy of configs/real_setup.json)
```

**Build and run the edited test set:**

```bash
NACHI_CLIPS=/path/to/nachi_robot_videos_trim python src/make_edited_tests.py data/edited
cp configs/edited_setup.json data/edited/setup.json
python src/run_real.py --root data/edited --setup data/edited/setup.json --device cuda
python src/analyze_edited.py data/edited
```

**On the lab cluster.** Clone the repo, choose your own working folder, and run the set-up once. Set-up is light
work, so it is fine on the login node. Everything heavy goes through SLURM.

```bash
export WORK=/scratch/psaxena/<your_name>_poserac      # your own folder; never someone else's
bash hpc/setup_hpc.sh                                 # SAM weights, 3 small packages into $WORK/pylib, video lists
```

It uses the shared `krishna` conda environment read-only. Never install into a shared environment. `/scratch` is
cleared after about 15 days: re-run the set-up if `$WORK` is gone, and copy results you want to keep into the repo.

**Run the full 200-video dataset on SLURM.** Run from the repo folder. Start 1–2 jobs at most (the account is shared with
other people); they share the work and skip finished videos:

```bash
sbatch -p <gpu partition> -o $WORK/logs/%x_%j.out --export=ALL,WORK=$WORK hpc/run_shard.sbatch queue.txt
python src/score_200.py $WORK/results $WORK/lists/all.txt     # also runs automatically at the end of each job
```

The v5 run took 3 h 47 min on two GPUs.

**Synthetic benchmark:**

```bash
pip install -r requirements-synthetic.txt
python src/reproduce.py --model vitb
```

It renders a Nachi-like arm into a photo of the real empty cell, with exact ground truth.

## Regression check

Every change must keep (or improve) v5's results on the 10 development and 9 edited videos. The check prints one row
per video (true count, v5, new, status) and ends with PASS or FAIL. It fails if any count gets further from the truth,
or if a video v5 got right is now wrong.

```bash
# changes to counting only (home score, thresholds, smoothing): seconds, no GPU, uses the stored v5 outlines
python src/regression_check.py recount

# changes to tracking: full pipeline on the cluster (the first run also builds the input frames)
sbatch -p <gpu partition> -o $WORK/logs/%x_%j.out --export=ALL,WORK=$WORK hpc/regression_check.sbatch <run_name>
cat $WORK/regression_runs/<run_name>/regression_report.txt

# a single video, for a quick look while developing (not a pass)
python src/regression_check.py recount --only nachi_7_seq_008
```

Passing the check is necessary, not sufficient. Also run the 200 videos for any change you want to keep, and record
the outcome in `HISTORY.md`.

## Working together

- **`main` always holds the latest released version.** Published tags (`v5`, `v6.1`) never move.
- **Work on your own branch** (e.g. `abhinav/static-fix`). One change per branch where possible.
- **Before merging:**
  - run the regression check;
  - add an entry to `HISTORY.md` (template at the end), including what didn't work;
  - give it a version number (`v6.2`, `v7`, ...; rules in [`RELEASING.md`](RELEASING.md)).

  Tag the merge with that number.
- **Fixes must be general.** No special cases for one robot, part, colour or video, and no settings tuned on the test
  videos (HISTORY.md, Sections 7–8).
- **Keep results you rely on in the repo** under `results/<version>/` (metrics and per-video results; not frames or
  videos).

## Data

The Nachi videos and ground truth are not included. They are the lab's dataset: 10 one-cycle recordings, 200 generated multi-cycle videos in `dataset_out/`, and 200 non-uniform videos made from the same recordings on 25 Oct 2026 (waits, holds and speed changes) in `dataset_nonuniform/`.

## Credits

- **SAM:** Kirillov et al., *Segment Anything*, ICCV 2023.
- **PoseRAC:** Yao, Cheng and Zou, *PoseRAC: Pose Saliency Transformer for Repetitive Action Counting*, arXiv:2303.08450, 2023. Its counting trigger and model are ported in `src/poserac.py`.
- **Fixed-box baseline:** the fixed-box SAM + area-autocorrelation pipeline is Abhinav's earlier work, re-implemented here for comparison.
