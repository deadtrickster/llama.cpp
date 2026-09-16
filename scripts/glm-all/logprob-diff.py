#!/usr/bin/env python3
"""Compare the top-k logprobs of the first generated token between two /completion responses.

Both responses must come from the same prompt with "n_probs": K (and ideally "temperature": 0,
"n_predict": 1). Prints, over the tokens present in both top-k lists, the max / mean / median
absolute logprob difference, whether the chosen token agrees, and the gap to the runner-up.
This is the token-level view of a numerical change; a distributional view is ppl-paired.py.

    logprob-diff.py A.json B.json [label]
"""
import json
import sys


def top(path):
    j = json.load(open(path))
    return j['completion_probabilities'][0]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, b = top(sys.argv[1]), top(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else "A vs B"
    ta = {t['id']: t['logprob'] for t in a['top_logprobs']}
    tb = {t['id']: t['logprob'] for t in b['top_logprobs']}
    shared = sorted(set(ta) & set(tb))
    d = sorted(abs(ta[k] - tb[k]) for k in shared)
    sb = sorted(tb.values(), reverse=True)
    gap = sb[0] - sb[1] if len(sb) > 1 else float('nan')
    print("%s: max %.3f  mean %.3f  median %.3f over %d shared of %d/%d; chosen %r / %r (%s); B's top1-top2 gap %.3f"
          % (label, d[-1], sum(d) / len(d), d[len(d) // 2], len(shared), len(ta), len(tb),
             a['token'], b['token'], "same" if a['id'] == b['id'] else "DIFFERENT", gap))


if __name__ == '__main__':
    main()
