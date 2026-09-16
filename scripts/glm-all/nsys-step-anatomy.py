#!/usr/bin/env python3
"""Per-step anatomy of a decode from an nsys recording (the .sqlite export).

Steps are found from a marker kernel that runs once per layer per forward pass (default:
gated_delta_net, the KDA/GDN kernel; pass --marker for another model). A cluster of marker
launches separated by more than --gap ms is one forward pass; clusters with fewer than
--min-markers launches (a draft-model pass, a partial) are ignored.

Reports, per step: wall time, busy time per device, time with no device running, the idle
gaps by (device before -> device after) so a handoff cost has a name, kernel time by class,
launches per step, and the CUDA API time on the host.

    nsys stats --force-export=true -q --report cuda_gpu_kern_sum rec.nsys-rep   # writes rec.sqlite
    nsys-step-anatomy.py rec.sqlite [--marker gated_delta_net] [--min-markers 29] [--gap 3] [--skip 10]
"""
import argparse
import bisect
import collections
import re
import sqlite3

CLASSES = collections.OrderedDict([
    ('dense mat-vec (quantized)',      r'^void mul_mat_vec_q<'),
    ('expert mat-vec (moe)',           r'^void mul_mat_vec_q_moe'),
    ('mat-mul (mmq / cublas)',         r'^void mul_mat_q<|nvjet|cublas|cutlass'),
    ('f32/f16 mat-vec',                r'^void mul_mat_vec_f|^void mul_mat_f'),
    ('quantize activations',           r'^quantize_'),
    ('attention',                      r'flash_attn|attn'),
    ('recurrent (gdn / ssm)',          r'gated_delta_net|ssm_'),
    ('topk / indexer',                 r'topk|DeviceTopK|indexer'),
    ('norms',                          r'norm'),
    ('elementwise glue',               r'k_bin_bcast|scale_f32|unary_op|cpy_|concat|get_rows|rope|silu|swiglu|sigmoid|softmax|clamp|argsort|sum_rows|fill|pad|set_rows|affine_sigmoid|dsv4_hc'),
])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sqlite')
    ap.add_argument('--marker', default='gated_delta_net')
    ap.add_argument('--min-markers', type=int, default=29)
    ap.add_argument('--gap', type=float, default=3.0, help='ms between marker launches that separates two passes')
    ap.add_argument('--skip', type=int, default=10, help='steps to drop at the start (warm-up) and end')
    a = ap.parse_args()

    db = sqlite3.connect(a.sqlite)
    c = db.cursor()

    kern = c.execute("SELECT k.start, k.end, k.deviceId, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                     "JOIN StringIds s ON k.demangledName = s.id ORDER BY k.start").fetchall()
    if not kern:
        raise SystemExit("no kernels in the recording")

    marks = [k[0] for k in kern if a.marker in k[3]]
    if not marks:
        raise SystemExit("marker kernel %r not found; pass --marker" % a.marker)

    clusters, cur = [], [marks[0]]
    for t, tp in zip(marks[1:], marks[:-1]):
        if t - tp > a.gap * 1e6:
            clusters.append(cur)
            cur = []
        cur.append(t)
    clusters.append(cur)
    steps = [cl[0] for cl in clusters if len(cl) >= a.min_markers]
    if len(steps) < a.skip * 2 + 2:
        raise SystemExit("only %d full passes found; lower --skip or --min-markers" % len(steps))
    steps = steps[a.skip:len(steps) - a.skip]
    n = len(steps) - 1
    starts = [k[0] for k in kern]

    wall = steps[-1] - steps[0]
    busy = collections.Counter()
    launches = collections.Counter()
    cls_time = collections.OrderedDict((k, 0) for k in CLASSES)
    cls_time['other'] = 0
    cls_n = collections.Counter()
    gap_by = collections.Counter()
    idle = 0
    tiny_n = tiny_t = 0

    i = bisect.bisect_left(starts, steps[0])
    j = bisect.bisect_left(starts, steps[-1])
    seg = kern[i:j]

    for s, e, d, name in seg:
        busy[d] += e - s
        launches[d] += 1
        if e - s < 5000:
            tiny_n += 1
            tiny_t += e - s
        for k, pat in CLASSES.items():
            if re.search(pat, name):
                cls_time[k] += e - s
                cls_n[k] += 1
                break
        else:
            cls_time['other'] += e - s
            cls_n['other'] += 1

    cs, ce, cd = seg[0][0], seg[0][1], seg[0][2]
    for s, e, d, _ in seg[1:]:
        if s <= ce:
            if e >= ce:
                ce, cd = e, d
        else:
            idle += s - ce
            gap_by[(cd, d)] += s - ce
            cs, ce, cd = s, e, d

    ms = lambda ns: ns / 1e6 / n

    print("%d steps analysed; mean step %.2f ms" % (n, wall / 1e6 / n))
    for d in sorted(busy):
        print("  dev%d busy %.2f ms/step, %.0f launches/step" % (d, ms(busy[d]), launches[d] / n))
    print("  no device running: %.2f ms/step (%.0f%%)" % (ms(idle), 100 * idle / wall))
    print("  idle by handoff (device before -> after), ms/step:")
    for (x, y), t in gap_by.most_common(8):
        print("    dev%d -> dev%d  %.2f" % (x, y, ms(t)))
    print("  kernel time by class, ms/step (launches/step):")
    for k, t in cls_time.items():
        if cls_n[k]:
            print("    %-30s %6.2f  (%5.0f)" % (k, ms(t), cls_n[k] / n))
    print("  kernels under 5 us: %.0f/step, %.2f ms/step" % (tiny_n / n, ms(tiny_t)))

    try:
        rows = c.execute("SELECT s.value, COUNT(*), SUM(r.end - r.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                         "JOIN StringIds s ON r.nameId = s.id WHERE r.start >= ? AND r.start < ? GROUP BY s.value "
                         "ORDER BY 3 DESC LIMIT 6", (steps[0], steps[-1])).fetchall()
        print("  host CUDA API, ms/step (calls/step):")
        for name, cnt, t in rows:
            print("    %-30s %6.2f  (%5.0f)" % (name[:30], ms(t), cnt / n))
    except sqlite3.OperationalError:
        pass


if __name__ == '__main__':
    main()
