#!/usr/bin/env bash
# Profile ONE criterion bench in isolation: perf stat, perf record + flamegraph, heaptrack.
#
#   benches/profile.sh <bench-target> <filter> [seconds]
#   benches/profile.sh metadata 'stats/categorical/course' 5
#
# Outputs go to analysis/<date>/{perf,flamegraph,heaptrack}/<target>-<filter>.*
# Needs: perf, flamegraph (cargo install flamegraph), heaptrack, jq.
set -euo pipefail

target=$1 filter=$2 secs=${3:-5}
out="analysis/$(date +%F)"
name="$target-$(tr -d '^$' <<<"${filter//\//_}")"
mkdir -p "$out"/{perf,flamegraph,heaptrack}

bin=$(cargo bench --bench "$target" --no-run --message-format=json 2>/dev/null |
    jq -r 'select(.executable != null and .target.kind[0] == "bench") | .executable' | tail -1)
# --profile-time: run just this bench for N seconds, no statistics or reports.
run=("$bin" --bench --profile-time "$secs" "$filter")

echo "== perf stat"
perf stat -o "$out/perf/$name.stat.txt" \
    -e task-clock,context-switches,cpu-migrations,page-faults \
    -e cycles:u,instructions:u,branches:u,branch-misses:u,cache-references:u,cache-misses:u \
    -- "${run[@]}" >/dev/null
cat "$out/perf/$name.stat.txt"

echo "== perf record + flamegraph"
# dwarf unwinding: Rust std has no frame pointers. 8 KiB stacks at ~500 Hz keeps perf.data small.
perf record -q -F 499 --call-graph dwarf,8192 -o "$out/perf/$name.data" -- "${run[@]}" >/dev/null 2>&1
flamegraph --perfdata "$out/perf/$name.data" -o "$out/flamegraph/$name.svg" >/dev/null 2>&1
echo "$out/flamegraph/$name.svg"

echo "== heaptrack"
heaptrack --record-only -o "$out/heaptrack/$name" "${run[@]}" >/dev/null
heaptrack_print "$out/heaptrack/$name.zst" >"$out/heaptrack/$name.txt"
grep -E "^(total runtime|calls to allocation|temporary memory|peak heap|peak RSS|total memory leaked)" \
    "$out/heaptrack/$name.txt"
