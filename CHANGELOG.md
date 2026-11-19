# Changelog

One entry per version, newest first: when, what changed, the headline results, and where the details are.
The full story of every change, including what was tried and dropped, is in [`HISTORY.md`](HISTORY.md).
How versions are numbered, tagged and released: [`RELEASING.md`](RELEASING.md).

Results use these test sets: *dev* = the 10 development videos, *edited* = the 9 edited videos with uneven cycles,
*200* = the full 200-video dataset, *non-uniform 200* = 200 videos made from the same recordings with waits, holds and
speed changes. *Exact* = count exactly right; *OBO* = within one cycle; *MAE* = mean(|predicted − true| / true).

## [v6.1](../../releases/tag/v6.1) – 20 Nov 2026 (includes v6, 26 Oct 2026, which was not released on its own)
- **Change (v6, tracker):** a dynamic frame rate: SAM looks at 2 frames a second, and at 6 a second or every frame
  (going back over the skipped ones) when the robot moves fast, arrives at or leaves home, or the tracker is unsure.
  The set-up frame becomes a second memory, so a still part that drops out of the outline (nachi_4's base) can come back.
- **Change (v6.1):** videos are decoded while running; a per-cycle time table (start, end, duration, wait at home,
  period); cluster jobs can share one list; an **automatic setup** (no box or click to give); and **one command**,
  `count_cycles.py`: videos or folders in, setup + count + cycle times out. The counting rule is unchanged since v1.
- **Results (setup typed in per program):** 200: 196/200 exact (v5: 197), 199/200 OBO, MAE 0.0068. v5's 3 nachi_4
  misses are fixed; 4 videos are newly missed (nachi_10_seq_014, nachi_10_seq_015, nachi_9_seq_004, nachi_9_seq_015).
  Non-uniform 200: 187/200 exact, 195/200 OBO, MAE 0.0101 (nachi_4 9/20). Cycle ends within 1 s of the label after one
  shift per video: 99.5% and 99.7%. v6 tests: dev 10/10, edited 9/9, simulated palletizing from 1/5–4/12 (v5) to
  4/5–11/12 and 39/40.
- **Automatic setup:** tested on 2 videos only, both from program nachi_1 (9/9 and 7/7 cycles); not yet run on the 400.
- **Code:** `count_cycles.py`, `src/auto_setup.py`, `src/run_v6.py` (v5's `src/run_real.py` unchanged).
  **Details:** [`HISTORY.md`](HISTORY.md) Section 10 (v6, v6.1); release notes `docs/releases/v6.1.md`;
  tables `results/v6.1_2026-10-30/`.

## v5.1 (experiment, not adopted; no tag, no release) – 14 Oct 2026
- **Change:** v5 with frames sampled at 6 per second instead of 2. Nothing else changed.
- **Result:** did not help. On the simulated palletizing videos (one uniform with 40 cycles, four non-uniform with
  5–12) it counted 1–2 cycles per video (v5: 1–4). The cause: those clips jump back to the start pose, so there is no
  visible return home to count.
- **Details:** [`HISTORY.md`](HISTORY.md) Section 10; script `newtests/run_newtests.py`; numbers in
  `results/new_videos_2026-10-14/`.

## [v5](../../releases/tag/v5) – results 25 Sep 2026, tagged 13 Oct 2026
- **Change (tracker):** after 1 s with no acceptable outline, also prompt with the first-frame box and a box around
  all motion (recovery); if nothing passes, keep the last good outline (hold).
- **Results:** dev 10/10; edited 9/9; 200 videos 197/200 exact (98.5%), 200/200 OBO, MAE 0.002.
  The 3 misses (nachi_4_seq_014, 016, 018) are each one cycle short.
- **Code:** `src/run_real.py` at tag `v5`. **Details:** [`HISTORY.md`](HISTORY.md) Section 3 (v5).
- **After tagging:** HISTORY.md was corrected on 13–14 Oct (wording and a fact-check); no code changed.

## v1–v4 (before git; not tagged) – 24–25 Sep 2026
The code of each is kept in `history/`. Details: [`HISTORY.md`](HISTORY.md) Section 3.
- **v4:** "changed" pixels measured against the last accepted frame; looser size check. nachi_7 still wrong.
  `history/run_real_v4.py`
- **v3:** clean-up of specks and slivers; static-region ("wall") check. Dev 9/10. `history/run_real_v3.py`
- **v2:** only motion near the robot can widen the box; 6 candidates; choice by continuity; size check.
  5/6 (run on nachi_1–6 only). `history/run_real_v2.py`
- **v1:** the base stitched pipeline: a box that follows the robot + the label-free home/away counter. Dev 5/10.
  `history/run_real_v1.py`

## Before v1: the team's earlier pipeline (Abhinav)
- One fixed broad box, SAM's first mask, autocorrelation of the mask area. 200 videos 167/200 exact (83.5%, our
  re-run); edited 6/9. Re-implemented as the `abhinav` set-up in `src/run_real.py`. Details: HISTORY.md Section 2.
