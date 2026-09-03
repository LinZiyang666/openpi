#!/bin/bash
# sync_k3_arms.sh <suite> — weilandserver -> local -> timan108 (same absolute paths), cross-verified
set -euo pipefail
SUITE=$1
T=$CLAUDE_JOB_DIR/tmp/${SUITE}_k3_arms.tgz
tether pull weilandserver:/tmp/dsp_shared/rit_pareto/${SUITE}_k3_arms.tgz "$T" --force | tail -1
H=$(sha256sum "$T" | cut -c1-64)
tether push --force "$T" timan108:/tmp/dsp_shared/rit_pareto/${SUITE}_k3_arms.tgz | tail -1
tether exec --timeout 120s timan108 -- bash -lc "export PATH=/usr/local/bin:/usr/bin:/bin; cd /tmp/dsp_shared/rit_pareto && echo '$H  ${SUITE}_k3_arms.tgz' | sha256sum -c - && rm -rf $SUITE/k3 && tar -xzf ${SUITE}_k3_arms.tgz && find $SUITE/k3 -type f | sort | xargs sha256sum > /tmp/dsp_precheck/${SUITE}_k3.t108.sha && wc -l < /tmp/dsp_precheck/${SUITE}_k3.t108.sha"
tether pull timan108:/tmp/dsp_precheck/${SUITE}_k3.t108.sha $CLAUDE_JOB_DIR/tmp/${SUITE}_k3.t108.sha --force | tail -1
tether push --force $CLAUDE_JOB_DIR/tmp/${SUITE}_k3.t108.sha weilandserver:/tmp/dsp_shared/verify/${SUITE}_k3.t108.sha | tail -1
tether exec --timeout 60s weilandserver -- bash -lc "export PATH=/usr/local/bin:/usr/bin:/bin; cd /tmp/dsp_shared/rit_pareto && echo CROSS_MISMATCH=\$(sha256sum -c /tmp/dsp_shared/verify/${SUITE}_k3.t108.sha 2>&1 | grep -vc ': OK$')"
