"""Combine v6.1 runs into tables (no GPU, seconds):
  all_cycles.csv  every cycle of every video: start, end, duration, wait, period, frames, and the labelled cycle
  videos.csv      one row per video: count vs labelled, timing against the labels
  programs.csv    one row per program (and per profile for the non-uniform set)
  summary.txt     the totals
    python v6tests/collect_cycles.py <runs folder> [<runs folder> ...] --out <folder>
Timing errors are ours - label, in seconds. "after shift" removes one constant per video first: the labels of some
programs sit at a fixed offset from the robot's return home (nachi_2 about +12 s, nachi_8 about -11 s on the original
200), and the labelled start of a cycle on the original 200 is the previous cycle's end (the clips loop back to back),
so start errors there include the time the robot waits at home.
"""
import os, sys, csv, json, glob, argparse
import numpy as np

COLS = ['video', 'program', 'profile', 'cycle', 'start_s', 'end_s', 'duration_s', 'wait_s', 'period_s', 'start_frame',
        'end_frame', 'leave_s', 'return_s', 'complete', 'note', 'label_cycle', 'label_start_s', 'label_end_s',
        'start_err_s', 'end_err_s']


def med(x, f=np.median):
    x = [v for v in x if v is not None]
    return round(float(f(x)), 2) if x else ''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='+'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = []
    for r in a.runs:
        for f in sorted(glob.glob(os.path.join(r, '**', 'result.json'), recursive=True)):
            res.append(json.load(open(f)))
    res.sort(key=lambda r: r['name'])
    vids, cyc = [], []
    for r in res:
        prog = r.get('program') or os.path.basename(r['name']).split('_seq')[0].split('_nu')[0]
        tm, rows = r.get('timing') or {}, r.get('cycles', [])
        sh = tm.get('shift_s', 0.0) or 0.0
        for c in rows:
            cyc.append(dict(c, video=os.path.basename(r['name']), program=prog, profile=r.get('profile') or ''))
        ee = [c['end_err_s'] for c in rows if c.get('end_err_s') is not None]
        se = [c['start_err_s'] for c in rows if c.get('start_err_s') is not None]
        vids.append(dict(video=os.path.basename(r['name']), program=prog, profile=r.get('profile') or '',
                         count=r['count'], labelled=r.get('true_cycles', ''),
                         diff=(r['count'] - r['true_cycles']) if 'true_cycles' in r else '',
                         exact=r.get('exact', ''), matched=tm.get('matched', ''), missed=tm.get('missed', ''),
                         extra=tm.get('extra', ''), shift_s=tm.get('shift_s', ''),
                         end_abs_err_median_s=tm.get('end_abs_err_median_s', ''),
                         end_abs_err_after_shift_s=med([abs(e - sh) for e in ee]),
                         start_abs_err_median_s=tm.get('start_abs_err_median_s', ''),
                         start_abs_err_after_shift_s=med([abs(e - sh) for e in se]),
                         end_within_1s=tm.get('end_within_1s', ''), start_within_1s=tm.get('start_within_1s', ''),
                         incomplete=sum(1 for c in rows if not c.get('complete', True)),
                         frames_vs_v5=round(r['frames_processed'] / max(r['base_rate_frames'], 1), 2),
                         runtime_s=r.get('runtime_s'), flags=' | '.join(r.get('flags', []))))
    with open(os.path.join(a.out, 'all_cycles.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore'); w.writeheader(); w.writerows(cyc)
    with open(os.path.join(a.out, 'videos.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(vids[0])); w.writeheader(); w.writerows(vids)

    def group(key):
        out = {}
        for v in vids:
            out.setdefault(v[key], []).append(v)
        rows = []
        for k, vs in sorted(out.items()):
            names = {v['video'] for v in vs}
            cs = [c for c in cyc if c['video'] in names]
            lab = [v for v in vs if v['labelled'] != '']
            shift = {v['video']: (v['shift_s'] or 0.0) if v['shift_s'] != '' else 0.0 for v in vs}
            ee = [c['end_err_s'] for c in cs if c.get('end_err_s') is not None]
            ea = [abs(c['end_err_s'] - shift[c['video']]) for c in cs if c.get('end_err_s') is not None]
            sa = [abs(c['start_err_s'] - shift[c['video']]) for c in cs if c.get('start_err_s') is not None]
            rows.append({key: k, 'videos': len(vs), 'exact': sum(1 for v in lab if v['exact'] is True),
                         'within_1': sum(1 for v in lab if abs(v['diff']) <= 1),
                         'counted': sum(v['count'] for v in vs), 'labelled': sum(v['labelled'] for v in lab),
                         'nmae': round(float(np.mean([abs(v['diff']) / v['labelled'] for v in lab])), 4) if lab else '',
                         'cycles_matched': sum(v['matched'] or 0 for v in lab if v['matched'] != ''),
                         'median_shift_s': med([v['shift_s'] for v in lab if v['shift_s'] != '']),
                         'end_err_median_s': med(ee), 'end_abs_err_after_shift_median_s': med(ea),
                         'end_within_1s_after_shift': f"{sum(e <= 1 for e in ea)}/{len(ea)}" if ea else '',
                         'start_abs_err_after_shift_median_s': med(sa),
                         'start_within_1s_after_shift': f"{sum(e <= 1 for e in sa)}/{len(sa)}" if sa else '',
                         'duration_median_s': med([c.get('duration_s') for c in cs if c.get('complete', True)]),
                         'wait_median_s': med([c.get('wait_s') for c in cs if c.get('wait_s') not in (None, '')])})
        return rows
    progs = group('program')
    with open(os.path.join(a.out, 'programs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(progs[0])); w.writeheader(); w.writerows(progs)
    if any(v['profile'] for v in vids):
        prof = group('profile')
        with open(os.path.join(a.out, 'profiles.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(prof[0])); w.writeheader(); w.writerows(prof)
    lab = [v for v in vids if v['labelled'] != '']
    lines = [f"{len(vids)} videos, {len(cyc)} cycle rows ({sum(1 for c in cyc if not c.get('complete', True))} incomplete)"]
    if lab:
        lines.append(f"count exact {sum(v['exact'] is True for v in lab)}/{len(lab)}, within one {sum(abs(v['diff']) <= 1 for v in lab)}/{len(lab)}, "
                     f"counted {sum(v['count'] for v in lab)} of {sum(v['labelled'] for v in lab)} labelled cycles, "
                     f"NMAE {np.mean([abs(v['diff']) / v['labelled'] for v in lab]):.4f}")
    for p in progs:
        lines.append(f"  {p['program']}: exact {p['exact']}/{p['videos']}, {p['counted']}/{p['labelled']} cycles, shift {p['median_shift_s']} s, "
                     f"|end err| after shift {p['end_abs_err_after_shift_median_s']} s ({p['end_within_1s_after_shift']} within 1 s)")
    open(os.path.join(a.out, 'summary.txt'), 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
