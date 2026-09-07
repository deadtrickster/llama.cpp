#!/usr/bin/env bash
# The memory fitter (common/fit.cpp) against two devices, without a GPU.
#
# Two ggml-rpc-server processes expose the CPU as two "GPU" devices; each reports all physical RAM as free, which
# is deterministic, so a per-device budget is expressed as a margin (--fit-target) of total minus budget. The fit
# never allocates (no_alloc), so a model far larger than the budget costs nothing but its file.
#
# What is asserted, each of which used to be a refusal or a wrong number:
#   1. -ngl 999 with --n-cpu-moe is fitted, not refused ("n_gpu_layers already set by user ... abort"), the user's
#      overrides stay in front of the fitted ones, and n_gpu_layers is at most what the user allowed
#   2. a user --tensor-split is kept ("tensor_split already set by user ... abort" used to be the answer) and the
#      context is sized for it
#   3. with everything pinned and nothing fitting, the fitter says so and fails - it does not report success
#   4. -ngl N is an upper bound
#   5. the pure-auto shape still fits
#
#   test-fit-rpc.sh <ggml-rpc-server> <llama-fit-params> <test-llama-archs>
set -euo pipefail

server=$1
fit=$2
gen=$3
test_dir=$(mktemp -d)

# ports below the ephemeral range (32768+ on Linux), so no outbound connection of another process can hold one;
# and verified unused right before use, because a bind with SO_REUSEADDR can succeed on a port that a live
# connection still owns and then never receive a client
free_port() {
    local port=$1
    while ss -tan 2>/dev/null | awk '{ print $4 }' | grep -q ":${port}$"; do
        port=$((port + 1))
    done
    echo "$port"
}
port_a=$(free_port $((20000 + $$ % 10000)))
port_b=$(free_port $((port_a + 1)))

cleanup() {
    kill "${pid_a:-}" "${pid_b:-}" 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

wait_for_port() {
    local port=$1
    for _ in {1..600}; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3>&-
            exec 3<&-
            return 0
        fi
        sleep 0.05
    done
    return 1
}

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

# a GLM-shaped dummy: 8 blocks, experts dominating the weights, a trained context with room to search
"$gen" -a glm5next -l 8 --n-ff 1024 --n-expert 4 --n-ctx 16384 -s 1 -o "$test_dir" > "$test_dir/gen.log" 2>&1
model="$test_dir/glm5next-moe.gguf"
[ -f "$model" ] || fail "dummy model was not generated: $(tail -3 "$test_dir/gen.log")"
model_mib=$(( $(stat -c %s "$model") / 1024 / 1024 ))

"$server" --device CPU --host 127.0.0.1 --port "$port_a" > "$test_dir/server-a.log" 2>&1 &
pid_a=$!
"$server" --device CPU --host 127.0.0.1 --port "$port_b" > "$test_dir/server-b.log" 2>&1 &
pid_b=$!
wait_for_port "$port_a" || { cat "$test_dir/server-a.log" >&2; fail "rpc server a did not listen on $port_a"; }
wait_for_port "$port_b" || { cat "$test_dir/server-b.log" >&2; fail "rpc server b did not listen on $port_b"; }

# the CPU device reports total physical memory as free; a budget is total minus margin
total_mib=$(awk -F'[(, ]+' '/MiB free/ { for (i = 1; i <= NF; i++) if ($i == "MiB") { print $(i-1); exit } }' "$test_dir/server-a.log")
for _ in {1..100}; do
    [ -n "$total_mib" ] && break
    sleep 0.1
    total_mib=$(awk -F'[(, ]+' '/MiB free/ { for (i = 1; i <= NF; i++) if ($i == "MiB") { print $(i-1); exit } }' "$test_dir/server-a.log")
done
[ -n "$total_mib" ] || fail "could not read the device memory from $(cat "$test_dir/server-a.log")"

margin_for() { echo $(( total_mib - $1 )); }

common=(-m "$model" --rpc "127.0.0.1:$port_a,127.0.0.1:$port_b" -b 256 -ub 256)

run_fit() {
    # run_fit <name> <expected exit> <args...>; stdout+stderr in $test_dir/<name>.log, the CLI line in $test_dir/<name>.out
    local name=$1 expect=$2
    shift 2
    set +e
    "$fit" "${common[@]}" "$@" > "$test_dir/$name.log" 2>&1
    local rc=$?
    set -e
    grep -E '^-c ' "$test_dir/$name.log" > "$test_dir/$name.out" || true
    if [ "$rc" -ne "$expect" ]; then
        cat "$test_dir/$name.log" >&2
        fail "$name: exit $rc, expected $expect"
    fi
    if grep -q "already set by user" "$test_dir/$name.log"; then
        fail "$name: the fitter refused a user setting instead of honouring it"
    fi
}

# budgets that hold about half the weights: the search has to leave experts (and maybe layers) on the host
budget_a=$(( model_mib * 40 / 100 ))
budget_b=$(( model_mib * 30 / 100 ))
targets_half="$(margin_for "$budget_a"),$(margin_for "$budget_b")"

# 1. the operator's shape: -ngl 999 and a --n-cpu-moe floor
run_fit ngl999_ncmoe 0 --fit-target "$targets_half" -ngl 999 --n-cpu-moe 2
out=$(cat "$test_dir/ngl999_ncmoe.out")
[[ "$out" == *"-ot \"blk\\.0\\.ffn_"* ]] || fail "user override for blk.0 is not first in: $out"
[[ "$out" == *"blk\\.0\\.ffn_"*"blk\\.1\\.ffn_"* ]] || fail "user overrides blk.0, blk.1 not kept in order in: $out"
ngl=$(sed -E 's/.*-ngl ([0-9]+).*/\1/' <<< "$out")
[ "$ngl" -le 9 ] || fail "n_gpu_layers $ngl exceeds the 8 + 1 layers of the model"
grep -q "fit result" "$test_dir/ngl999_ncmoe.log" || fail "no fit summary logged"

# 5. pure auto still works, and lands on the same placement without the user's floor
run_fit auto 0 --fit-target "$targets_half"
grep -q "0 user + " "$test_dir/auto.log" || fail "pure auto reported user overrides"

# 4. -ngl N is an upper bound
run_fit ngl_cap 0 --fit-target "$targets_half" -ngl 3
ngl=$(sed -E 's/.*-ngl ([0-9]+).*/\1/' "$test_dir/ngl_cap.out")
[ "$ngl" -le 3 ] || fail "n_gpu_layers $ngl exceeds the user's bound of 3"

# 2. a user split is kept and the context is sized for it: budgets that hold the weights, not the full context
budget_a=$(( model_mib * 75 / 100 ))
budget_b=$(( model_mib * 75 / 100 ))
targets_weights="$(margin_for "$budget_a"),$(margin_for "$budget_b")"
run_fit pinned_split 0 --fit-target "$targets_weights" -ngl 999 --n-cpu-moe 6 --tensor-split 1,1
grep -q "placement pinned by user" "$test_dir/pinned_split.log" || fail "pinned split was not reported as pinned"
[[ "$(cat "$test_dir/pinned_split.out")" == *"-ts 1,1"* ]] || fail "user split not kept: $(cat "$test_dir/pinned_split.out")"
ctx=$(sed -E 's/^-c ([0-9]+).*/\1/' "$test_dir/pinned_split.out")
[ "$ctx" -ge 4096 ] && [ "$ctx" -le 16384 ] || fail "context $ctx outside [4096, 16384]"
[ $(( ctx % 256 )) -eq 0 ] || fail "context $ctx not aligned"
# the derived context must project within the margin on every device
if ! grep -oE '[0-9]+ MiB left vs. margin of [0-9]+' "$test_dir/pinned_split.log" | awk '{ if ($1 + 0 < $7 + 0) bad = 1 } END { exit bad + 0 }'; then
    cat "$test_dir/pinned_split.log" >&2
    fail "derived context projects over the margin"
fi

# 3. everything pinned and nothing fits: a failure, said plainly
run_fit all_pinned 1 --fit-target "$targets_half" -ngl 999 --tensor-split 1,1 -c 16384
grep -q "nothing left to adjust" "$test_dir/all_pinned.log" || { tail -8 "$test_dir/all_pinned.log" >&2; fail "all-pinned case did not explain itself"; }

echo "test-fit-rpc: ok (model ${model_mib} MiB, budgets ${targets_half} / ${targets_weights} as margins of ${total_mib})"
