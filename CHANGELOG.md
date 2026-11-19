# Changelog

One entry per version, newest first: when, what changed, the headline results, and where the details are.
The full story of every change, including what was tried and dropped, is in [`HISTORY.md`](HISTORY.md).
How versions are numbered, tagged and released: [`RELEASING.md`](RELEASING.md).

Results use these test sets: *dev* = the 10 development videos, *edited* = the 9 edited videos with uneven cycles,
*200* = the full 200-video dataset. *Exact* = count exactly right; *OBO* = within one cycle; *MAE* = mean(|predicted − true| / true).

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
