#!/bin/bash
# Phase 3 pull (run on the local workstation): tar the small x0 artefacts on each host (cells/ with the frozen yaml bytes,
# results_trailing/, diagnostics/, runs/*/{identity,manifest}.json + train_log.jsonl + checkpoints/final.done -- never
# checkpoints, zarr or hdf5), pull the tarballs with tether, unpack them under <local>/<host>/ and merge the hosts into
# <local>/merged (exp.dp_nfe.analysis.merge_x0_hosts). Re-runnable: a previous pull of a host is replaced.
# usage: pull_x0_results.sh [<local dir>=exp/dp_nfe/data/x0_multimodal]
set -u -o pipefail
L=${1:-exp/dp_nfe/data/x0_multimodal}
say() { echo "== $(TZ=America/Chicago date +%H:%M:%S) PULL $*"; }
pull_host() {  # pull_host <name> <tether node> <X0_DATA on the host> <remote tmp dir> [<pre-command>]
  local name=$1 node=$2 remote=$3 rtmp=$4 pre=${5:-true} tgz
  say "$name: tar on $node"
  tether exec --timeout 120s "$node" -- bash -lc "$pre; cd $remote && tar czf $rtmp/x0_pull_$name.tgz cells results_trailing \$(ls -d diagnostics 2>/dev/null) \$(ls -d runs/*/identity.json runs/*/manifest.json runs/*/train_log.jsonl runs/*/checkpoints/final.done 2>/dev/null) && ls -la $rtmp/x0_pull_$name.tgz" 2>/dev/null | awk '!seen[$0]++' | tail -1
  tgz=$L/x0_pull_$name.tgz
  [ -f "$tgz" ] && rm "$tgz"
  tether pull "$node:$rtmp/x0_pull_$name.tgz" "$tgz" 2>&1 | tail -1 || { echo "PULL_FAIL $name"; return 1; }
  [ -d "$L/$name" ] && { say "$name: replacing $(find "$L/$name" -type f | wc -l) previously pulled files"; rm -r "$L/$name"; }
  mkdir -p "$L/$name" && tar xzf "$tgz" -C "$L/$name" && rm "$tgz"
  say "$name: $(find "$L/$name" -name summary.json | wc -l) summaries, $(ls "$L/$name/cells" | grep -c '\.yaml$') cell yamls"
}
mkdir -p "$L"
pull_host wls weilandserver /data/dp/x0_multimodal /tmp/x0 || exit 1
pull_host h100 h100 /data/dp_h100/x0_multimodal /data/dp_h100/tmp "export HOME=/home/exouser" || exit 1
[ -d "$L/merged" ] && { say "replacing previous merge ($(find "$L/merged" -type f | wc -l) files)"; rm -r "$L/merged"; }
uv run python -m exp.dp_nfe.analysis.merge_x0_hosts --host h100="$L/h100" --host wls="$L/wls" --out "$L/merged" 2>&1 | grep -v -i "warning" || { echo "PULL_FAIL merge"; exit 1; }
echo "PULL_DONE $(TZ=America/Chicago date +%H:%M:%S) $L/merged"
