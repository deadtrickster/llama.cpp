#!/usr/bin/env bash
# Temperature-0 completion against a running llama-server; sha of the text against a saved
# reference. Bit-exact changes reproduce the reference; changes that alter summation order
# (split-K, a different device for a layer) legitimately do not - judge those with
# ppl-paired.py instead.
#
#   greedy-compare.sh ref            # write the reference (ref.json)
#   greedy-compare.sh check [label]  # compare against ref.json
#
# PORT, PROMPT, N_PREDICT and REF are overridable in the environment.
set -euo pipefail

PORT="${PORT:-8080}"
PROMPT="${PROMPT:-Write a long, detailed essay about the history of the Roman aqueducts, their engineering, and their legacy.}"
N_PREDICT="${N_PREDICT:-300}"
REF="${REF:-ref.json}"

mode="${1:?ref|check}"
label="${2:-check}"

busy=$(curl -s "localhost:$PORT/metrics" | sed -n 's/^llamacpp:requests_processing //p')
if [ "${busy:-0}" != 0 ]; then
    echo "server is processing $busy request(s); a contended number is not a number" >&2
    exit 1
fi

body=$(printf '{"prompt":%s,"n_predict":%d,"temperature":0,"cache_prompt":false}' "$(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" "$N_PREDICT")
out=$(mktemp)
curl -s -m 600 "localhost:$PORT/completion" -H 'Content-Type: application/json' -d "$body" > "$out"

if [ "$mode" = ref ]; then
    mv "$out" "$REF"
    python3 -c "import json,hashlib; j=json.load(open('$REF')); t=j['timings']; print('ref: tg=%.1f t/s, %d tokens, sha=%s' % (t['predicted_per_second'], t['predicted_n'], hashlib.sha1(j['content'].encode()).hexdigest()[:12]))"
    exit 0
fi

python3 - "$out" "$REF" "$label" <<'PY'
import json, sys, hashlib
n = json.load(open(sys.argv[1])); r = json.load(open(sys.argv[2])); t = n['timings']
same = n['content'] == r['content']
i = next((k for k in range(min(len(r['content']), len(n['content']))) if r['content'][k] != n['content'][k]), None)
print("%s: tg=%.1f t/s, sha=%s, identical_to_ref=%s%s" % (sys.argv[3], t['predicted_per_second'],
      hashlib.sha1(n['content'].encode()).hexdigest()[:12], same, "" if same else ", first difference at char %s" % i))
PY
rm -f "$out"
