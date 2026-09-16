#!/usr/bin/env python3
"""Paired per-chunk comparison of two llama-perplexity logs over the same text.

llama-perplexity prints the running estimate after every chunk ("[k]PPL,"). The running
estimate is exp(mean NLL over the first k chunks), so the k-th chunk's own mean NLL is
k*ln(P_k) - (k-1)*ln(P_{k-1}). Differencing those per chunk between two runs removes the
text's own variance, which dominates the +/- the tool reports.

    ppl-paired.py A.log B.log
"""
import math
import re
import sys


def series(path):
    return [float(x) for x in re.findall(r'\[\d+\]([0-9.]+),', open(path).read())]


def per_chunk(s):
    return [(k + 1) * math.log(s[k]) - k * math.log(s[k - 1]) if k else math.log(s[0]) for k in range(len(s))]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = series(sys.argv[1]), series(sys.argv[2])
    n = min(len(a), len(b))
    if n < 2:
        sys.exit("need at least two chunks in both logs")
    da, db = per_chunk(a[:n]), per_chunk(b[:n])
    d = [x - y for x, y in zip(da, db)]
    m = sum(d) / n
    sd = (sum((v - m) ** 2 for v in d) / (n - 1)) ** 0.5
    se = sd / n ** 0.5
    print("chunk    NLL(A)    NLL(B)   A-B")
    for k, (x, y) in enumerate(zip(da, db)):
        print("%5d  %8.4f  %8.4f  %+.4f" % (k + 1, x, y, x - y))
    print("final PPL: A %.4f  B %.4f" % (a[n - 1], b[n - 1]))
    print("paired mean dNLL(A-B) = %+.5f nats/token, sd %.4f, se %.5f, t = %.2f (n = %d); dPPL/PPL = %+.3f%%; A>B on %d chunks, B>A on %d"
          % (m, sd, se, m / se if se else float('nan'), n, 100 * (math.exp(m) - 1),
             sum(1 for v in d if v > 0), sum(1 for v in d if v < 0)))


if __name__ == '__main__':
    main()
