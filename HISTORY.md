# HISTORY: what has been tried, what changed, and why

This is the truth sheet for SAM-Box PoseRAC. Read it before changing anything, so you don't redo something that has
already been tried. Add an entry for every change you make, including the ones that didn't work (template at the
end). The README says how to run the code; this file says why it looks the way it does.

Where the numbers come from:
- v5 results: the stored files in `results/`;
- v1–v4 development runs, the palletizing video and the per-return nachi_4 analysis: run outputs kept on Arrya's
  laptop (ask if you need them);
- the synthetic benchmark, DREAM, baselines and MobileSAM: the mid-semester report and the PDFs in `reports/`.

A few details come only from lab notes or code comments, and are marked "(notes)". Where an older note or report
disagrees with the raw results, trust the raw results (Section 9 lists the known cases).

Terms:
- *Dev* = the 10 development videos; *edited* = the 9 edited videos with uneven cycles; *200* = the full dataset.
- *Exact* = count exactly right; *OBO* = within one cycle.
- *MAE* = the RepCount/PoseRAC normalised error, mean(|predicted − true| / true).
- A *jump* = overlap (IoU) below 0.6 between consecutive outlines half a second apart.
- *Held frames* = frames where no candidate passed the checks and the last good outline was kept.

---

## 1. Summary

| Version | Main change | Dev (exact) | Edited (exact) | 200 videos (exact) | Code |
| --- | --- | --- | --- | --- | --- |
| Earlier pipeline (Abhinav) | One fixed broad box, SAM's first mask, area autocorrelation | 10/10 his own run; 9/10 our re-run | 6/9 (our re-run) | 83.5% (167; our re-run) | re-implemented as the `abhinav` set-up in `src/run_real.py` |
| v1 | Base stitched pipeline: following box + label-free home/away counter | 5/10 | – | – | `history/run_real_v1.py` |
| v2 | Only nearby motion widens the box; 6 candidates; continuity choice; size check | 5/6 (run on nachi_1–6 only) | – | – | `history/run_real_v2.py` |
| v3 | Clean-up; static-region ("wall") check | 9/10 | – | – | `history/run_real_v3.py` |
| v4 | "Changed" measured against the last accepted frame; looser size check | nachi_7 only: still wrong | – | – | `history/run_real_v4.py` |
| v5 | Recovery boxes after 1 s stuck; hold the last good outline | 10/10 | 9/9 | 98.5% (197), OBO 100%, MAE 0.002 | `src/run_real.py`, git tag `v5` |
| v6 | Dynamic frame rate (2 / 6 / 30 fps with look-back); set-up frame as a second memory | 10/10 | 9/9 | not re-run in full; the 3 v5 misses are now exact (Section 10) | `src/run_real.py` unchanged; v6 is `src/run_v6.py` (not released on its own; part of v6.1) |
| **v6.1** | v6's tracker and counter, unchanged; videos decoded while running; per-cycle times; automatic setup; one command | same tracker as v6 | same tracker as v6 | **98.0% (196)**, OBO 99.5%, MAE 0.0068 (setup typed in); 200 non-uniform videos 93.5% (187) (Section 10) | `count_cycles.py`, `src/auto_setup.py`, `src/run_v6.py`, git tag `v6.1` |

The counting stage (stage 2) is the same in v1–v5. Only the tracker (stage 1) changed between versions. v6 changes which
frames are looked at and adds the set-up memory to the tracker; the counting rule is unchanged (Section 10).

To see exactly what changed between two versions: `diff history/run_real_v3.py history/run_real_v4.py`
(v5 is `src/run_real.py` at tag `v5`).

---

## 2. Where it started

**The team's earlier pipeline (Abhinav).** SAM ViT-B with one fixed, broad box around the work area
(`[730, 70, 1450, 1070]` at 1920×1080), SAM's first mask, the mask area per frame, and autocorrelation of the area to
find the repeat period. Count = video length ÷ period. It was judged on how strong the periodicity looked (0.09–0.73
across 10 videos), which seemed unstable, so the line was set aside in favour of robot keypoints.

**What we found when we checked it against the true counts.** Turned into counts, its own run was right on all 10 of
its test videos (1–4 cycles each). The 10-video test was too small to show its weaknesses, which appear on the full
dataset and on uneven cycles:
- autocorrelation assumes every cycle has the same length and shape. On the edited videos, long waits made it count
  2 instead of 4 (edit_1, edit_5). On the 200 videos (our re-run), it is exact on only 67% of videos with 21–30
  cycles, and off by as much as 24;
- it also fails some very short videos. On 4 single-cycle videos it locks onto a short false period, e.g.
  nachi_7_seq_008 → 15 and nachi_3_seq_011 → 7. It is 89% exact on videos with 1–5 cycles;
- the fixed box picks up strips of wall. Its outlines average 15.3% of the frame (v5: 9.1%) and jump 10,402 times over
  the 200 videos (v5: 715). On edit_7, a flickering wall panel created a false 6 s period and a count of 32 instead
  of 3.

**Lesson:** judge a method on counts against ground truth, not on how strong a signal looks.

**Before v1 (24 Sep): testing the two halves separately.**
- *Prompt box, 9 real frames (SAM ViT-B; first frames taken from the figures in the team's report, 640×360 enlarged
  2×):*
  - a tight box gave the whole robot on 9/9 (SAM score 0.93–0.98, 8–14% of the frame);
  - a generic broad box (`[380, 0, 1120, 719]` at 1280×720) gave the wall, with a robot-shaped hole, on 9/9 (39–49% of
    the frame);
  - a box only 25% looser than the tight one gave 25–45% of the frame on 4/9.

  The robot is white and light grey against a light-grey wall, so with a loose box SAM picks the wall. Clicking on the
  base, or marking the wall as background, did **not** fix this for ViT-B.
- *Synthetic video with exact masks (40 frames):* a fixed broad box was badly wrong on 11/40 frames. A box that follows
  the robot plus a base click averaged IoU 0.96 with the true robot (worst frame 0.90).
- *Counting from perfect outlines (32 synthetic videos, 6 cycles each):*

  | Counter | Exact | Fails on |
  | --- | --- | --- |
  | Home score + PoseRAC trigger (used in v1–v5) | 25/32 | passing near home mid-cycle (2), wrist-only motion (2), no wait at home (2), pure base rotation (1) |
  | Supervised PoseRAC (2 labels per cycle), released code | 26/32 | base rotation, small motions |
  | Supervised PoseRAC, each frame classified on its own | 30/32 | base rotation |
  | Area autocorrelation | 22/32 | speed changes, partial last cycle, small motions, some normal videos |

  The supervised classifier counts best but needs labelled frames for every robot program, so the label-free home score
  was chosen.

---

## 3. Tracker versions in detail

All runs: SAM ViT-B, 2 fps, 1280×720, one box and one base click on the reference frame. The dev runs were on a laptop
GPU (Apple M3, MPS) on 24–25 Sep 2026.

### v1: base stitched pipeline
- **Change:** the first version to join SAM tracking to the counter. The box for the next frame goes around the last
  outline plus *any* movement in the work area, enlarged by 8%. It takes SAM's top-scoring mask. If the area changes
  outside 0.4–2.5× the previous outline, it retries with a 25% wider box and keeps the more plausible mask. There is no
  clean-up, wall check or fallback.
- **Result:** dev 5/10. On nachi_5, 7, 8, 9 and 10 the box kept growing until SAM outlined the wall, and from then on it
  followed the wall (outlines 42–82% of the frame instead of about 9–10%). Every one of those counted 0.

### v2: keep the box near the robot; choose by continuity
- **Change:**
  - only movement close to the robot can widen the box (within the previous box enlarged by 50%);
  - two boxes (tight around the previous outline; previous outline plus nearby motion) × three SAM masks give 6
    candidates (3 when no nearby motion is found);
  - choice = highest `IoU(candidate, last outline) + 0.1 × SAM score`;
  - size check: 0.5–2× the median of the last 10 accepted outlines. Only plausible outlines update the tracker.
- **Why:** the box creeping outward in v1.
- **Result:** run on nachi_1–6 only, 5/6. nachi_5 still slid onto the wall (mean outline 34% of the frame, count 0):
  specks on a wall panel stretched the box. This was seen on the laptop GPU.

### v3: clean-up and the static-region ("wall") check
- **Change:**
  - *Clean-up:* morphological opening (7×7 ellipse), then removal of pieces under 1,500 pixels. If nothing that big is
    left, the raw mask is kept.
  - *Static-region check:* `u` is the fraction of a candidate lying outside both a 51×51 dilation of the last accepted
    outline and the pixels that changed since the previous frame (also dilated 51×51; from v4 on, "changed" is
    measured against the last accepted frame). The candidate is rejected if `u > 0.10`, and `u` is
    also subtracted in the choice score (`− 2u`). The camera is fixed, so a wall panel that never changes cannot get in.
  - *Size check tightened:* 0.6–1.6× the recent median, and 0.4–2× the first-frame outline.
  - If no candidate passed, v3 output the best candidate trimmed to the allowed area (this was replaced by "hold" in v5).
- **Why:** specks and wall panels getting into the outline (nachi_5).
- **Result:** dev 9/10, nachi_5 fixed. **nachi_7 failed** (counted 0 instead of 1), in two stages
  (`docs/figures/nachi7_v3_failure_diagnosis.jpg`):
  1. at about 19 s the arm stretches diagonally across the wall, the box around it is mostly wall, and every SAM
     candidate is wall (a limit of SAM);
  2. the tracker never recovers: it keeps the oversized box from the last good outline. Prompting with the first-frame
     box at 37.5 s gets the robot back.
- **Side effect found later:** the static-region check cannot tell "re-adding a still part of the robot" from "adding a
  wall". This is half of the nachi_4 problem (Section 6.1).

### v4: measure "changed" against the last accepted frame; looser size check
- **Change:** pixels count as "changed" relative to the frame of the last *accepted* outline, not the previous frame.
  The recent-size check was loosened back to 0.5–2× (fast arm moves change the area quickly).
- **Why:** with the previous frame as reference, a robot that had moved and then stopped during a run of rejected frames
  looked static and was locked out (nachi_7).
- **Result:** tried on nachi_7 only. It still counted 0, so it **did not fix nachi_7**. Both changes were kept in v5.
- **Note:** the same "still looks static" effect is what later blocks the nachi_4 base from coming back (Section 6.1).
  v4 only moved the reference frame; it did not make still robot parts acceptable.

### v5: recovery and hold (current)
- **Change:**
  - *Recovery:* after 2 frames in a row (1 s) with no acceptable candidate, also prompt with the first-frame box and a
    box around all motion since the last accepted frame (up to 6 more candidates).
  - *Hold:* if nothing passes, keep the last good outline for that frame.
- **Why:** nachi_7, where the tracker was stuck with an oversized box.
- **Result:**
  - dev 10/10 (9 held frames of 1,508; 4 sudden outline jumps against 99 for the fixed box);
  - edited 9/9 (16 held of 2,583);
  - 200 videos 197/200 exact, 100% within one cycle, MAE 0.002 (1,811 held frames, 0.9%).
- **Full rule set:** README "How v5 works". The exact constants are in `src/run_real.py` (`run_sam`).

---

## 4. The counting stage (unchanged in v1–v5)

1. `d_t = 1 − IoU(outline_t, outline_ref)`, where the reference frame shows the robot at home: frame 0, except
   nachi_9, which uses frame 14 because of a 7 s fade-in. `init_frame` is set in `configs/`.
2. Otsu's method on all `d_t` of the video gives a cut-off `τ`. With `lo = τ/2` and `hi = (τ + max d)/2`, the home
   score is `p_t = clip((hi − d_t)/(hi − lo), 0, 1)`.
3. PoseRAC's released trigger, unchanged (`poserac.count_official`). The smoothed score is
   `s_t = 0.6 p_t + 0.4 s_{t−1}`, starting at 0.5. There are two triggers:
   - the *home* trigger is armed when `s > 0.78` and fires when `s` then drops below 0.4 (the robot has left home);
   - the *away* trigger is armed when `s < 0.22` and fires when `s` then rises above 0.6 (the robot has come back).

   A cycle is counted each time the away trigger fires after a home firing. So a return home counts as soon as the score
   climbs back above 0.6, but only if the robot was properly home (above 0.78) before it last left.
4. The reported cycle-end time is when `s` first goes above 0.78 after an away stretch.
5. The last score is repeated for 1.5 s (3 frames) so that a return home at the very end of a video still counts.

Consequences to keep in mind:
- a weak return (peak below 0.78) still counts its own cycle, but the trigger is not re-armed, so the **next** cycle is
  lost (two cycles merge). A weak *last* return costs nothing;
- the robot must look "home" for about 2 frames in a row (1 s at 2 fps) to reach 0.78;
- the cut-off uses the whole video, so the counter is not online, and the result for one cycle can depend on how long
  the video is (Section 6.1).

---

## 5. Tried and dropped

| Idea | Result | Status |
| --- | --- | --- |
| Fixed broad box (earlier pipeline) | Wall picked up; see Section 2 | Replaced by the following box |
| Base click or background click to rescue a broad box | Did not help ViT-B (does help MobileSAM) | Dropped |
| SAM point prompts only (team, before this work) | Masks covered 37–72% of the frame | Dropped |
| v1's tracker: box grows with any motion, SAM's top-scoring mask | Slid onto the wall on 5/10 | Replaced in v2 (box limited to nearby motion; 6 candidates + continuity) |
| MobileSAM instead of ViT-B | About 8× faster on CPU. On tight prompts it matches ViT-B with IoU 0.63–0.91 (usually ~0.8); on broad boxes it behaves differently (2–24% of the frame). One full synthetic video counted 6/6 (outline IoU 0.97). | Not used, so results stay comparable with the team's ViT-B work. Could be revisited for speed, but re-run the regression check. |
| Area + autocorrelation for counting | v5 outlines + autocorrelation: dev 8/10 (fails nachi_5, nachi_7), edited 7/9, 200: 83.5% | Replaced by home/away counting. Still useful as a cross-check, and it wins on a perfect loop with no pause (palletizing, 40/40). |
| Fixed-box outlines + v5 counter | Dev 7/10, edited 7/9, 200: 70.5% | Shows both halves of v5 are needed |
| Supervised PoseRAC on an 11×9 occupancy grid (99 numbers) | 26/32 synthetic; 30/32 with per-frame classification | Needs labels for every robot program |
| PoseRAC trained on pseudo-labels from the home score | Dev 5/10, worse than the home score alone | Dropped |
| Weaker smoothing on the palletizing video (diagnostic only) | Weight 0.2: 26/40; 0.1: 31/40; none: 33/40 | Not adopted. It shows frame-based smoothing is the problem (Section 6.2). |
| DREAM pretrained robot keypoints | 0 keypoints on 20 NACHI frames with both the Panda and Kuka models, 0 of 980 on 140 crops. The positive control on real Panda frames found 7/7. | Released models don't transfer to the CMZ05. HoRoPose, RoboPEPP and RoboPose need the same robot-specific training plus inputs we don't have. |
| Training our own keypoint detector | Scoped, not started: CAD → URDF → synthetic renders (6–10 weeks, CAD must be requested), or hand-label 300–800 frames and fine-tune ViTPose (3–4 weeks + ~2) | Only worth it if a later stage needs joint-level detail |
| Video-language models, RepNet, TransRAC (team runs) | VLMs give near-constant answers; TransRAC predicts 1 for every video; RepNet over-counts ~2.8× (untested guess: its 64-frame window is too short for 22–53 s cycles) | Not usable as counters here |

---

## 6. Known failures and open issues

### 6.1 nachi_4: a lost base plus a brief dip (the 3 misses in the 200)
- **Videos:** nachi_4_seq_014, 016 and 018 (9, 8 and 10 cycles). Each is one short, and always at the same return (the
  6th, about 147 s in).
- **What happens** (recomputed from the stored outlines):
  - the reference outline covers the whole robot (8.6% of the frame);
  - during the 2nd cycle the outline loses the black base and never gets it back. From then on, its overlap with the
    reference at each return home is about 0.76–0.77 instead of about 1 (outline about 6.6% of the frame at home);
  - at the 6th return there is a short dip, about 4 frames (2 s), where the upper arm also drops out (3.7% of the
    frame, overlap 0.41). The next frames are back to about 6.5%.

  With the base missing, every return home has little margin. The dip pushes the 6th return's smoothed peak to 0.77,
  just under 0.78, so the trigger is not re-armed and the 6th and 7th cycles merge (Section 4).
- **Why the safeguards miss it:**
  - the tracker's main memory is the last accepted outline (it also uses recent sizes and, when stuck, the first-frame
    box). The continuity choice keeps picking outlines without the base;
  - the static-region check blocks re-adding parts that are standing still. The base never moves, so adding it back
    looks exactly like adding a wall;
  - the size checks don't trigger: the base drops out over several frames, each step small enough to pass the
    recent-size check, and the dip's minimum is 0.42× the reference size, just above the 0.4× limit.
- **Why only these three:**
  - only nachi_4 loses part of the robot like this. Its average outline size is about 15–23% smaller on long videos than
    on short ones (more of the video comes after the loss); other programs stay within about ±1–2%;
  - in videos with exactly 6 cycles the weak 6th return is the last one, so nothing follows it to be lost.
    nachi_4_seq_004 peaks at 0.75 there and is still counted correctly;
  - with 14 or more cycles, the Otsu cut-off lands slightly differently and the same return just reaches 0.78.
- The earlier pipeline gets these three right, because it only needs the repeat period.

### 6.2 The robot doesn't pause at home, or cycles are fast
- **Synthetic no-wait videos:** 6/6 at 6 fps; at 3 fps 5 and 4; at 2 fps 1 and 2; at 1 fps and below, 1. The
  supervised classifier counts both correctly even at 2 fps. Its "home" covers a wider range of poses, so smoothing
  doesn't filter it out.
- **Palletizing video** (rendered scene from the Wipro PARI lab, different robot, 40 identical 5.97 s cycles): v5
  counts **1/40** at 2 fps.
  - Tracking was fine.
  - Each cycle has at most one frame that looks like home (a raw score of 1.0 in 21 of the 40 cycles). One frame later
    the raw score is down to about 0.46 or less, and the smoothed score never passes 0.64.
  - Autocorrelation on v5's outlines gets 40/40. So does the earlier pipeline, run with its NACHI box coordinates
    unchanged on this new scene.
- **Cause (corrected 14 Oct 2026):** the video loops a 5.98 s clip whose last frame is far from its first. The next
  frame jumps straight back: frame 178 is pixel-identical to frame 0, and the jump from frame 177 is about 56× a
  normal frame-to-frame change. So the robot never moves back to its start pose on screen: it reaches it only by the
  jump, and has moved away again within a fraction of a second (1–2 frames at 6 fps), at any frame rate. The earlier explanation (smoothing counted in frames) was wrong for this
  video. Sampling faster does not help (v5.1, Section 10). The same construction is used in all the palletizing test
  videos (`uniform_*`, `nonuniform_*`).
- **Repeat timing** (autocorrelation on v5's outlines) counts the uniform video correctly (40/40) because it is
  perfectly periodic. It fails on the non-uniform ones (2 for each of the 5-, 6-, 10- and 12-cycle videos), because the
  speed changes from cycle to cycle.
- The synthetic no-wait result above still stands: there the robot moves continuously through home, and 6 fps works.
- **What would count these videos:** a counter that does not rely on a visible return, e.g. detecting the loop jump
  itself. That is an artefact of how the videos were made, not robot behaviour, so it is not worth building into the
  method. Better: regenerate the test videos from a clip that starts and ends at the same pose.

### 6.3 Cycle-end timing is rougher than the counts
- **Edited set:**
  - all 33 ends matched exactly once in a −5 s to +2 s window;
  - median 0.8 s early (4.4 s early to 0.3 s late);
  - 21/33 within 1 s;
  - in a tighter −1 s to +2 s window: v5 21/33, fixed-box outlines with the same counter 7/33.
- **200 videos:**
  - for each label, the nearest detected return is a median 1.33 s early, and only 16.6% are within 1 s. (A separate
    measure, "label inside a detected home stretch, ±1 s", gives 2,650 of 2,941.);
  - 616 extra "home" stretches, all in nachi_8 and nachi_9;
  - offsets depend on the program (about +12 s for nachi_2 and −11 s for nachi_8), because a label marks where one copy
    of the recording joins the next.
- **nachi_9:** 0 of 288 boundaries fall inside a "home" stretch, because of the fade-in, although every count is correct.
- **Why early:** the counter calls "home" as soon as the outline is close to the reference, before the robot stops.

### 6.4 Other limits
- **The dataset loops 10 one-cycle recordings,** so cycles within a video are pixel-identical. The 200-video result
  tests consistency and long videos, not variation between cycles. v5 was developed on 10 of these videos.
- **The edited videos reuse the same recordings,** and their pauses are frozen frames with no camera noise.
- **Not yet run with the full pipeline:** the 14 synthetic disturbance videos (contrast, dark background, clutter,
  occluder, flicker, camera shake).
- **The counter double-counts** when the robot passes close to home mid-cycle. It can miss wrist-only motion or pure
  base rotation, where the outline barely changes.
- **No real multi-cycle footage** of the CMZ05 has been tested yet. This is the most important missing test.

---

## 7. Proposed fixes (all five tried in v6: 1, 3 adopted; 2, 4 measured, not adopted; 5 adopted in part. Section 10)

**Rule: fixes must work for any robot.** They must not single out a body part, a colour or this camera, and must not
be tuned to particular videos.

1. **Use the set-up frame as a second memory, not just the last frame** (targets 6.1). The method already assumes a
   fixed camera, a set-up frame at home, and returns home every cycle; nothing new is assumed.
   - *Static-region check:* still accept unchanging area if it was inside the set-up outline **and** looks the same as
     in the set-up frame. A pixel that was robot then and looks identical now is still robot. When the arm is away, those
     pixels show background, look different, and stay blocked. Reuse the existing change threshold; add no new tuned
     constant.
   - *Candidates:* always include the set-up box as a candidate (today only used when stuck). Score candidates by
     `max(IoU with last outline, IoU with set-up outline)`.
   - *Expected effect:* errors can still happen within a cycle but are corrected at each return home instead of piling
     up. If the lighting changes a lot during a long recording, the "looks the same" test stops matching and this just
     doesn't fire.
2. **Make the cut-off independent of video length** (targets 6.1, and needed for a live camera). Derive it from the
   set-up frame or a sliding window instead of Otsu on the whole video. On its own this does not fix the drift.
3. **Define smoothing, hold and recovery in seconds rather than frames, or sample faster** (targets the synthetic
   no-wait case in 6.2). 6 fps is about 3× the compute (about 20 GPU-hours for the 200 videos). Sampling at 6 fps was
   tried as v5.1 (Section 10) and does **not** help the palletizing videos, whose problem is the loop jump.
4. **Mark the cycle end when the robot comes to rest at home,** not when it first gets close (targets 6.3).
5. **Warning flags (they never change the count):**
   - the overlap with the set-up outline at each return home staying well below its value at the first return means
     the tracker has lost part of the robot (on nachi_4 it sits at about 0.77 from the 2nd return on);
   - the count disagreeing with duration ÷ autocorrelation period should be checked by a person.

**Don't:**
- special-case a body part or colour;
- nudge 0.78 or the 0.4× size limit just enough to catch the nachi_4 videos;
- choose any setting by looking at nachi_4 results.

**How to show a fix is general:**
- settings are frozen before the run;
- the regression check passes;
- nachi_4 is fixed with no new errors on the 200;
- the synthetic stress test passes: delete part of the outline mid-video and check it recovers by the next return
  home;
- it is then checked on the palletizing video and on new footage.

---

## 8. Ground rules for changes

1. **Run the regression check before calling anything better.** Use `python src/regression_check.py recount` for
   counting changes (seconds, no GPU). Use `hpc/regression_check.sbatch` for tracking changes. A change must not make
   any of the 19 videos worse.
2. **Keep changes general** (Section 7).
3. **Freeze settings before a test run.** If you change something after seeing the results on a set, that set becomes
   development data. Say so in the entry.
4. **One change at a time,** or also test the parts separately, so each effect can be attributed.
5. **Record negative results here too.** "Tried X, no effect" saves the next person a week.
6. **Never move a published tag** (`v5`, `v6.1`). New versions get new tags (`v6.2`, `v7`, ...; rules in
   `RELEASING.md`) and an entry below.
7. **Cluster:**
   - work only in your own `$WORK` folder;
   - no heavy work on the login node (use SLURM);
   - at most about 3 heavy GPU jobs at once (team guidance);
   - never modify the shared conda environments (install into `$WORK/pylib`);
   - `/scratch` is cleared after about 15 days, so copy results you want to keep into `results/<version>/` in the repo.

---

## 9. Gotchas

- **The counting code is duplicated.** If you change stage 2, change every copy, or better, make them all call
  `run_real`:
  - the pipeline's own path: `run_real.refdist_prob` and `run_real.counter`, used by `run_real.process`,
    `count_video.py`, `run_batch.evaluate` and `regression_check.py`;
  - the trigger itself (0.78 / 0.4 / momentum 0.4): `poserac.count_official` and `poserac.ActionTrigger`;
  - `run_real.counter` also has its own departure loop with 0.6/0.4 hard-coded, and `run_real.annotated_video` its own
    copy of the trigger (for drawing only);
  - `analyze_real.py`: its own `refdist_prob` and `home_periods` (dev boundary statistics). `analyze_edited.py` imports
    `analyze_real.refdist_prob` and has its own `home_entries`;
  - `run_batch.home_periods`: re-implements the trigger for the 200-video boundary statistics;
  - `evaluate.py`: another `refdist_prob` (synthetic benchmark). Its `otsu` is imported by `run_real` and
    `analyze_real`, so `evaluate.py` is a live part of v5.
- **`run_real.process` also runs the comparison methods:** the earlier pipeline's fixed box (`abhinav`, first mask),
  `abhinav_best` (top-scoring mask) and pseudo-label PoseRAC. In `results.json`, v5 is `methods.ours.labelfree_counter`.
  `run_batch.py` (200-video run) writes a different, smaller format: `methods.ours.counter`.
- **Coordinates:** box and click are in 1280×720 pixels; outlines are stored at 640×360, bit-packed.
- **The reference frame must show the robot at home and fully visible.** If a video starts with a fade, use
  `--init_frame` (nachi_9 uses 14).
- **Results can differ slightly between devices:**
  - the v2 nachi_5 failure was seen on the laptop GPU (MPS) (notes);
  - cluster and laptop gave the same count on nachi_1_seq_007, with outline IoU 0.986 (job 363873) (notes).

  Compare like with like.
- **Cluster:**
  - start-up takes about 10–11 minutes: about 7 min of imports from the network home, then about 4 min of CUDA
    initialisation (notes);
  - the cluster's ffmpeg module did not work on every node, so the scripts use the ffmpeg binary from `imageio-ffmpeg`,
    copied to local disk (code comment);
  - the first 200-video attempt (jobs 363890–363892) was stopped by a video-decoding problem on one node (the third job
    was cancelled while still queued) and resubmitted (notes). The final run, jobs 363905 and 363906, took 3 h 47 min:
    6.8 GPU-hours of processing (0.12 s per frame), about 7.6 GPU-hours allocated.
- **The edited set is rebuilt from a fixed seed (20260925)** by `make_edited_tests.py`. `regression_check.py prepare`
  checks that each rebuilt video has the stored cycle count and duration (not every cycle time). nachi_9 is excluded
  (fade-in at the start, ends mid-move).
- **Older sources that disagree with the raw data.** Trust the raw data:

  | Source | Says | Raw data says |
  | --- | --- | --- |
  | 24 Sep report | tight box SAM score 0.96–0.98; a box 25% looser gives the wall on 9/9 | 0.93–0.98; 25–45% of the frame on 4/9 |
  | lab notes | v5 dev held frames 10 | 9 |
  | mid-semester report | 1,502 dev frames | 1,508 |
  | mid-semester report | nachi_4 outline "slowly shrinks"; videos with ≤6 cycles "never reach the 6th return" | base lost once plus a brief dip; a weak last return costs nothing (Section 6.1) |
- **PoseRAC's released code has quirks** (only relevant if you use its learned model):
  - frames in a batch attend to each other, which blurs small motions; `poserac.py` adds a per-frame mode;
  - the loss combines triplet and BCE terms with a 0.01 weight and an unusual sign;
  - its evaluation picks the best class per video by lowest error (an oracle choice).

---

## 10. Log of changes after v5

Add new entries at the top of this section, using this template:

```
### v6 – <one line: what changed> (<name>, <date>)
- Change:
- Why (the failure that prompted it, with video names):
- General? (does it rely on anything specific to this robot, camera or video? if so, why is that acceptable?)
- Settings frozen before testing? (yes / no; if no, which sets became development data)
- Regression check: recount <PASS/FAIL, exact x/19>; full run <PASS/FAIL, exact x/19>, report: <path>
- 200 videos (if run): exact, OBO, MAE; which videos changed
- Other tests (palletizing, synthetic, new footage):
- Side effects / what it broke:
- Decision: kept (tag) / reverted
```

### v6.1 – the 400-video run, the automatic setup and one command (Arrya, 30 Oct – 20 Nov 2026)
Same tracker and counter as v6 and as the v6.1 entry below. This entry adds the full run and the two parts that make
v6.1 fully automatic. Released as **v6.1** (tag `v6.1`, 20 Nov 2026) together with the entry below.

- **400-video run** (lab cluster, 30 Oct; H200 GPUs, 2 jobs sharing one list; `v6tests/make_jobs_200.py`). The setup
  was typed in per program from `hpc/job_setups.json`, as in v5's run. One exception: on the non-uniform set nachi_9 is
  set up at frame 0 with a wider box, because those videos have no fade-in (reason in `make_jobs_200.py`). Tables:
  `results/v6.1_2026-10-30/task1/` (the original 200) and `task2/` (200 non-uniform videos made on 25 Oct from the same
  recordings: 4 profiles of 50, namely waits at home, waits + holds, waits + speed changes, and mixed).

  | Set | Exact | Within one | MAE | Cycles counted / labelled | Cycle ends within 1 s (after the per-video shift) |
  | --- | --- | --- | --- | --- | --- |
  | Original 200 | 196/200 (v5: 197) | 199/200 (v5: 200) | 0.0068 (v5: 0.002) | 3,142 / 3,141 | 3,122 of 3,139 (99.5%) |
  | Non-uniform 200 | 187/200 | 195/200 | 0.0101 | 3,166 / 3,184 | 3,156 of 3,164 (99.7%) |

  - Original 200: v5's three nachi_4 misses are now exact, as the v6 tests predicted. Four videos v5 got right are now
    missed: nachi_10_seq_014 (5 counted, 3 labelled), nachi_10_seq_015 (3 vs 2), nachi_9_seq_004 (21 vs 22),
    nachi_9_seq_015 (6 vs 7). Their causes have not been looked into yet.
  - Non-uniform 200: nachi_4 is 9/20 exact, every miss short by 1–4 cycles (the "nachi_4 counts wobble" warning in the
    entry below, made worse by holds and speed changes). The other misses: nachi_10_nu08 (31 vs 30; the lost-part flag
    fired) and nachi_9_nu20 (2 vs 1). Every other program is 19 or 20 out of 20.
  - Start times: on the original 200 the labelled start of a cycle is the end of the previous one (the clips loop back
    to back), so start errors include the robot's wait at home. Within 1 s after the shift: 2,307 of 3,139 starts
    (original 200) and 2,108 of 3,164 (non-uniform).
  - Compute: 8.0 and 8.1 GPU-hours for the two sets, about 0.29 s per second of video.
- **Automatic setup** (`src/auto_setup.py`; first written on 2 Nov as `stage0_auto.py` to make the two demo videos):
  - *Change:* the box around the robot, the click on its base and the home frame no longer have to be given. From the
    first 120 s of the video: (1) a quick skim at 6 fps; (2) the biggest patch of pixels that keep changing is where the
    robot works; (3) home = the first frame, or after a fade-in the first still frame where the robot is visible;
    (4) a rough robot shape = the home frame against the typical "robot away" picture, inside that patch; (5) SAM ViT-B
    prompted with that shape three ways (box + points, points only, box only), 3 answers each; of the answers that cover
    at least 80% of the shape, the one with the best SAM score minus twice its share of wall-like pixels; (6) base point
    = the lowest part of that outline that never moves during the skim; box = the outline's bounding box.
  - *Why:* so that a new video needs nothing but the video.
  - *General?* It assumes the robot is the biggest moving thing in view during the first 120 s and is at home (or
    still) early on. Both hold on the NACHI videos; neither is guaranteed elsewhere.
  - *Settings frozen before testing?* No separate test set: the constants were set while building it on nachi_1 (the
    two demo videos), so those two videos are development data for it.
  - *Tests:* 2 videos only, both from program nachi_1, with the full v6.1 counter: nachi_1_seq_000 9/9 and
    nachi_1_nu05 7/7; every cycle end within 1 s of its label (median 0.23 s early), starts a median 0.27 s late.
    Its box on nachi_1_seq_000 is [438, 188, 898, 638] against the typed-in [430, 180, 950, 719] (tighter; the
    typed-in one reaches the bottom of the picture).
  - *Not yet run* on the other 9 NACHI programs or on the 400 videos. Decided 20 Nov 2026 to release with this stated;
    the full automatic run can be done later.
  - *New simulated videos* (`gazebo_harshita`, one video of each kind, unchanged v6.1): right only on the plainest
    (5/5) and one short on a static CCTV view (3/4); on the other kinds the setup or the count failed. The reasons
    (several arms, people walking through, a moving camera, very short cycles and more) are the v6.2 work.
  - *Known weak spots:* on NACHI the outline can take in the switch box on the wall right next to the robot (seen while
    building it). A wrong setup shows in `setup.jpg`; the fix is to give the setup by hand (`count_cycles.py --setup`).
  - *Packaging (20 Nov):* the same computations as `stage0_auto.py`, now a function that uses the repo's SAM weights
    and picks CUDA, Apple Silicon or CPU, and stops with a clear message where the old script would crash (nothing
    moving, no robot shape, empty outline). Checked: identical setup on nachi_1_seq_000.
- **One command** (`count_cycles.py`, 20 Nov): takes videos and/or folders; for each video runs the automatic setup
  (or takes one given by hand), draws `setup.jpg`, then runs the v6.1 counter (`run_v6.run_job`, unchanged) and writes
  `setup.json`, `setup.jpg`, `result.json`, `cycles.csv`, `trace.npz`; with several videos also a summary. It
  downloads and checks the SAM weights on first use, picks the device and loads SAM once for all videos.
  *Check* (laptop, Apple GPU, 20 Nov): `python count_cycles.py nachi_1_seq_000.mp4 --labels ...` starting with no
  weights (downloaded and checked: md5 matches). Same setup as on 2 Nov, 9/9 cycles, the same 540 frames processed, and
  `cycles.csv` and every outline in `trace.npz` identical to the 2 Nov automatic run; 640 s for 202 s of video.
  (Only the text layout of `setup.jpg` was changed after this run, and checked on its own.)
- **Not changed:** `src/run_v6.py` (tracking, counting, timing) and `src/run_real.py` (v5).
- **Decision:** released as v6.1 (tag `v6.1`, 20 Nov 2026). v6 was not released on its own; its code is part of v6.1.

### v6.1 – videos decoded while running; a per-cycle time table (Arrya, 30 Oct 2026)
- **Change** (`src/run_v6.py`; nothing in how frames are chosen, tracked or counted changes):
  1. *Decoding while running.* v6 first wrote every 30 fps frame as a JPEG; for the 200 videos that is about 3 million
     images (~300 GB). Now ffmpeg streams the frames (same filter and scaling, no JPEG compression) and only the last 48
     are kept, more than the look-back ever needs (15). A first pass (every 15th frame) gives v5's motion envelope.
     `"decode": "jpeg"` in a job keeps the old way.
  2. *Per-cycle table* (`cycles.csv` per video; `v6tests/collect_cycles.py` combines them). For each cycle: start
     (robot leaves home), end (back home), duration, wait at home, period (start to next start), frame numbers, v5's
     threshold times, and the labelled cycle it matches. Departures come from the counter's own trigger
     (`trigger_cycles` reproduces `poserac.count_official` step for step: same count on 3,000 random score sequences).
     Start and end use the distance to the set-up outline: between the farthest point of one cycle and the departure of
     the next, the robot is "home" while within 10% of the way from its closest approach to the farthest point; end =
     first such frame, start = last.
  3. `--shared`: several cluster jobs work through one list (each video is claimed first; a stopped job can be resubmitted).
- **Why:** task 1 asks for start/end times of every cycle in the 200 videos; the frame pre-extraction made a 200-video
  v6 run impractical.
- **Tried and dropped for the table:** "at rest" = outline overlaps the one 0.5 s apart by ≥ 0.9 (v6's `rest_end_s`).
  On nachi_4 the robot moves so slowly far from home that consecutive outlines also overlap by ~0.9, so ends came out
  ~1.5 s early and starts landed at two different phases of an identical looped clip (spread 4.4 s). With the distance
  rule every cycle of nachi_4_seq_016 has the same phase (leaves 0.5–0.7 s after each labelled boundary, back 1.3–1.4 s
  before the next; period 24.5 s, clip 24.47 s). `rest_end_s` is still reported, unchanged.
- **General?** Yes: no new assumption beyond v5's (the set-up frame shows the robot at home); the 10% band is relative
  to each video's own distances.
- **Labels:** on the original 200 the labelled cycles are the looped clip's boundaries, back to back. For nachi_4 the
  boundary falls inside the robot's 2 s at home; for nachi_2 and nachi_8 it is ~12 s off. So label differences are
  reported raw and after removing one constant per video (`shift_s`).
- **Checks** (laptop, Apple GPU; `results/v6.1_2026-10-30/`):
  - streamed vs the cluster's v6 run (JPEG frames, CUDA): nachi_4_seq_016 8/8, the same 398 frames and the same return
    times; palletizing 12-cycle video 11/12 (as before; the last cycle ends with the video), identical return times,
    582 vs 586 frames, 78% of the time at the fast rates with no frame missing from the look-back.
  - the same video through the old JPEG path on the laptop gave **7/8**: during cycle 6 the outline kept only the base
    column while the arm was away (124–145 s) and picked up a box on the wall, so the distance to home never rose. Not
    caused by v6.1 (streaming on the same laptop gave 8/8), but it shows v6's nachi_4 fix is fragile: tiny pixel or
    GPU-arithmetic differences decide 7 or 8 on this video (v5 also got 7/8). Expect nachi_4 counts to wobble by one.
  - per-cycle times: exact on a synthetic robot (known times, 6 fps and 2 fps sampling, video ending mid-cycle);
    nachi_4_seq_016: every cycle at the same phase, after removing the per-video shift (−1.3 s) ends within 0.08 s
    median, 15/15 within 1 s (both runs); palletizing 12-cycle (never pauses): ends 0.21 s, starts 0.14 s from the
    labels (median), all 11 within 0.5 s; the labels drift by 0.4 s over the video because they run to 78.68 s while
    the file is 78.25 s long.
  - decoding cost (laptop, 195 s of 1080p): 51 CPU-s for the streamed frames, 13 CPU-s for the first pass; on the
    cluster's 4 CPUs this runs alongside SAM and stays below the GPU time.
- **Decision:** kept. Released as v6.1 (tag `v6.1`, 20 Nov 2026) together with the 400-video run, the automatic setup
  and the one-command entry point (entry above).

### v6 – dynamic frame rate and the set-up frame as a second memory (Arrya, 26 Oct 2026)
- **Change** (`src/run_v6.py`; v5's `src/run_real.py` is untouched):
  1. *Dynamic frame rate.* Frames are prepared at 30 fps; SAM runs on every 15th (2 fps, exactly v5's frames) and
     speeds up when, compared with the frame 0.5 s earlier, the outline changed a lot (overlap < 0.7), the robot is
     arriving at or leaving home (distance to the set-up outline changed by > 0.1 while < 0.5), or no outline was
     acceptable: then 6 fps. When the robot is moving fast *and* near home: every frame (30 fps), only while that
     holds. On speeding up it goes back and processes the frames it skipped since the last one (look-back), so the
     start of the movement is not missed. Back to 2 fps after 1 s without a trigger, on a frame v5 also uses.
     Recovery (1 s stuck), the size-check window (5 s) and end padding (1.5 s) are now in seconds. The home score is
     smoothed per processed frame as before; the Otsu cut-off is weighted by the time each frame stands for.
  2. *Set-up memory.* Static-region check: a pixel inside the set-up outline that looks the same as in the set-up
     frame (same 18-grey-level test as the motion check) is allowed back. The set-up box is always a candidate prompt.
     Candidates are scored by max(overlap with last outline, overlap with set-up outline).
  3. *Reported only (do not change the count):* warning flag "outline at home smaller than at the first return" and
     "many held frames"; cycle end "at rest"; a count with a 120 s window cut-off.
- **Why:** HISTORY 6.1 (nachi_4: base lost, never re-added) and 6.2 (robot that is home only briefly; OP75 robot
  outrunning the box). The 30 fps check on the uniform palletizing video (v5 rules on every frame, first 18 s) counted
  3/3: the start pose is visible for ~4–5 frames (0.15 s) per cycle at 30 fps but only 1 at 6 fps, hence the
  every-frame level.
- **General?** Yes. No body part, colour, camera or video is singled out. The set-up memory assumes only what v5
  already assumes (fixed camera, set-up frame at home). New constants (0.7, 0.1, 0.5, 1 s) were set from first
  principles before any v6 run (0.7: at 6 fps each step is then ~1/3 of the change, overlap ~0.9).
- **Settings frozen before testing?** Yes. One change after the first runs: the repeat-period disagreement was demoted
  from a flag to a reported number, because on the 200 it fired on 32 videos with correct counts (flags never change
  the count).
- **Tests run** (only what each change needs; cluster jobs 416296, 416297, 416311, 416299; H200 / RTX Pro 6000,
  0.04–0.05 s per SAM image embedding):

  | Test | v5 | v6 | v6 frames vs v5 |
  | --- | --- | --- | --- |
  | Regression: 10 development videos | 10/10 exact | 10/10 exact | 1.21× |
  | Regression: 9 edited videos (rebuilt at 30 fps, same seed) | 9/9 | 9/9 | 1.16× |
  | The 3 nachi_4 videos v5 got wrong (seq_014/016/018) | 8/9, 7/8, 9/10 | **9/9, 8/8, 10/10** | 1.05× |
  | Simulated palletizing, 4 uneven videos (5, 6, 10, 12 cycles) | 1, 1, 1, 4 (OBO 0%) | **4, 5, 9, 11** (OBO 100%) | 4.0× |
  | Simulated palletizing, uniform 40 cycles | 1 | **39** | 3.95× |
  | OP75 robot B, 30 s from 12 s (no ground truth) | tracked ~6 s | tracked to ~14 s, lost by 20 s (as fixed 6 fps) | 2.77× |

  - Simulated videos: every cycle whose return is visible is counted, and all 68 internal cycle ends are found
    within 1 s. The one missing cycle in each is the last: the video ends at the end of a cycle, before the clip
    jumps back to the start pose, so there is no return to see. NMAE (4 uneven): v5 0.800, v5.1 0.842, v6 0.138.
  - Fixed 6 fps (v5.1) costs 3× and still fails these; fixed 30 fps would cost 15×.
  - nachi_4: home overlap at every return stays 0.97–1.00 (v5: ~0.77 from the 2nd return; 0.44 during the 6th-return
    dip, `results/v6_2026-10-26/nachi4_return6.jpg`). The lost-part flag no longer fires on these videos.
  - *Ablation on the 3 nachi_4 videos:* set-up memory alone 9/9, 8/8, 10/10 (overlap 0.97–1.0). Dynamic rate alone
    9/9, 8/8, **11/10** (overlap stays ~0.96, and the faster reaction in fast mode double-counts once). The memory is
    the fix; the dynamic rate alone is not safe on a damaged outline.
  - *Stress test* (lower third of the outline removed at 35 s, fixed 2 fps): without memory the home overlap at later
    returns stays low on nachi_4_seq_012 (0.77, 0.75, 0.72) and on nachi_2_seq_003 recovers only at the last return
    (0.82, 0.81, 0.97); with memory it is back at the next return (0.97–1.0, same as unstressed). Counts 4/4 either
    way on these short videos.
  - *Synthetic videos with exact outlines* (`v6tests/synth_nopause.py`, rendered at 30 fps, 4 seeds × 6 cycles each;
    tests only which frames are looked at and the counting, not SAM). Exact videos and frames used:

    | Robot | fixed 2 fps (v5) | fixed 6 fps | every frame | v6 |
    | --- | --- | --- | --- | --- |
    | never pauses at home | 0/4 (308) | 4/4 (920) | 4/4 (4,590) | 4/4 (1,736) |
    | smooth loop through home | 4/4 (484) | 4/4 (1,450) | 4/4 (7,246) | 4/4 (1,024) |
    | rests ~1.5 s at home | 4/4 (376) | 4/4 (1,124) | 4/4 (5,616) | 4/4 (1,958) |

    v6 is right wherever any fixed rate is. Its cost depends on how fast the robot moves near home: the synthetic
    robot moves fast, so v6 uses 2.1–5.6× v5's frames there (NACHI: 1.0–1.3×).
- **Counting extras on all 200 videos** (from v5's stored outlines, `v6tests/recount_200.py`):
  - v6's counter on evenly spaced frames reproduces v5's count on 200/200 videos;
  - lost-part flag: fires on 19 of the 20 nachi_4 videos (all 3 misses) and on no other video;
  - 120 s window cut-off: fixes the 3 nachi_4 misses but over-counts 5 nachi_8 videos by one (195/200): **not adopted**;
  - cycle end "at rest": 1,025 of 2,941 labelled ends within 1 s, against 488 for v5's end time. Not adopted as the
    default: on robots that never rest (palletizing) it is worse (3/68 vs 68/68). Both are reported.
- **Regression check:** the 19 videos are all exact (same counts as v5). Cycle ends within 1 s: dev 3/15 (v5 2/15),
  edited 14/24 (v5 15/24).
- **200 videos:** not re-run with the tracker. A 30 fps frame grid for 27.7 h of video is ~3 M images; the full run
  needs on-the-fly decoding first. Expected: the 3 misses fixed (shown above), the rest unchanged (19/19 and the
  recount suggest so, but this is not a measured result).
- **Side effects:** compute 1.0–1.3× v5 on NACHI-style videos, ~4× on robots that move fast all the time. The
  every-frame level reacts within ~0.1 s, so a damaged outline can double count (ablation above).
- **Decision:** kept as v6. Not released or tagged on its own (decided 20 Nov 2026): its code is part of the v6.1
  release (tag `v6.1`), which also made the full 200-video run (v6.1 entries above).
- Results: `results/v6_2026-10-26/`; jobs: `v6tests/`; cluster script `hpc/run_v6.sbatch`.

### v5.1 – v5 with frames sampled at 6 per second instead of 2 (Arrya, 14 Oct 2026)
- **Change:** frame rate only (6 fps). Every rule is unchanged and still counted in frames, except the end padding,
  which stays 1.5 s (9 frames). Script: `newtests/run_newtests.py` on the laptop (Apple M3, about 1.2 s per frame).
- **Why:** the palletizing video failed at 2 fps (1/40), and the synthetic no-wait test had counted correctly at 6 fps.
- **General?** Yes (frame rate only), but 3× the compute.
- **Settings frozen before testing?** Yes. The uniform 40-cycle video motivated it; the 4 non-uniform videos were new.
- **Regression check:** not run. It needs 6 fps frames for the 19 videos; do it as a cluster job before using v5.1 on
  NACHI.
- **Results (count / true):**

  | Video | v5 (2 fps) | v5.1 (6 fps) |
  | --- | --- | --- |
  | uniform, 40 cycles | 1 / 40 | 1 / 40 |
  | non-uniform, 5 cycles | 1 / 5 | 1 / 5 |
  | non-uniform, 6 cycles | 1 / 6 | 1 / 6 |
  | non-uniform, 10 cycles | 1 / 10 | 1 / 10 |
  | non-uniform, 12 cycles | 4 / 12 | 2 / 12 |

  NMAE over the 4 non-uniform videos: v5 0.800, v5.1 0.842; OBO 0% for both.
- **Why it doesn't help:** see the corrected cause in 6.2. The clip jumps back to its start pose, so there is never a
  visible return home to count.
- **OP75 factory video** (real, phone camera, two robots riding an overhead rail, no ground truth), v5 at 2 fps from
  12 s:
  - robot A was tracked for about 5 s, then SAM merged it with the rail it rides on;
  - robot B was tracked for about 6 s, then it slid along the rail faster than the box followed, and the outline was
    left on the background;
  - the camera also moves (shifts of tens to hundreds of pixels), which breaks the fixed-camera assumption;
  - v5.1 (6 fps) on 30 s from 12 s: robot A tracked for about 7 s, then merged with the rail; robot B tracked for
    about 14 s, then lost. Faster sampling helps the tracker follow a robot that moves quickly, but the counts are not
    meaningful (tracking lost, moving camera, no ground truth);
  - per-run numbers: `results/new_videos_2026-10-14/`, script and job lists: `newtests/`.
- **Decision:** not kept as the default. Not tagged.

