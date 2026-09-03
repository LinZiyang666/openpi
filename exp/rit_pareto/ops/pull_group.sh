#!/bin/bash
# pull_group.sh <suite> <layer>  — timan107 raw -> local exp/rit_pareto/data/runs/<suite>_<layer>/ (sha-verified)
set -euo pipefail
SUITE=$1; LAYER=$2; G=${SUITE}_${LAYER}
D=/tmp/dsp_precheck/rit_pareto/$G
L=/home/weiland/projects/openpi/exp/rit_pareto/data/runs/$G
mkdir -p "$L"
tether exec --timeout 300s ${HOST:-timan108} -- bash -lc "export PATH=/usr/local/bin:/usr/bin:/bin; cd $D && tar -czf /tmp/dsp_precheck/rit_pareto/${G}.tgz journal.jsonl per_step.jsonl per_step.jsonl.launch.json apool_${SUITE}.local.yaml && sha256sum journal.jsonl per_step.jsonl per_step.jsonl.launch.json > /tmp/dsp_precheck/rit_pareto/${G}.sha && sha256sum /tmp/dsp_precheck/rit_pareto/${G}.tgz | cut -c1-64" | tail -1 > "$L/.tgz.sha"
tether pull ${HOST:-timan108}:/tmp/dsp_precheck/rit_pareto/${G}.tgz "$L/${G}.tgz" --force | tail -1
tether pull ${HOST:-timan108}:/tmp/dsp_precheck/rit_pareto/${G}.sha "$L/remote.sha" --force | tail -1
echo "$(cat "$L/.tgz.sha")  $L/${G}.tgz" | sha256sum -c -
tar -xzf "$L/${G}.tgz" -C "$L" && (cd "$L" && sha256sum -c remote.sha) && rm -f "$L/${G}.tgz"
wc -l "$L/journal.jsonl" "$L/per_step.jsonl"
