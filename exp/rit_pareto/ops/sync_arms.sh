#!/bin/bash
# sync_arms.sh <suite>  — weilandserver -> local -> timan107, same absolute paths, sha-verified
set -euo pipefail
SUITE=$1
T=$CLAUDE_JOB_DIR/tmp/${SUITE}_arms.tgz
tether pull weilandserver:/tmp/dsp_shared/rit_pareto/${SUITE}_arms.tgz "$T" --force | tail -1
H=$(sha256sum "$T" | cut -c1-64)
tether push --force "$T" timan107:/tmp/dsp_shared/rit_pareto/${SUITE}_arms.tgz | tail -1
tether exec --timeout 120s timan107 -- bash -lc "export PATH=/usr/local/bin:/usr/bin:/bin; cd /tmp/dsp_shared/rit_pareto && echo '$H  ${SUITE}_arms.tgz' | sha256sum -c - && tar -xzf ${SUITE}_arms.tgz && find $SUITE/export_tau1 $SUITE/arms -type f | sort | xargs sha256sum > /tmp/dsp_precheck/${SUITE}_arms.t107.sha && wc -l < /tmp/dsp_precheck/${SUITE}_arms.t107.sha"
tether pull timan107:/tmp/dsp_precheck/${SUITE}_arms.t107.sha $CLAUDE_JOB_DIR/tmp/${SUITE}_arms.t107.sha --force | tail -1
tether push --force $CLAUDE_JOB_DIR/tmp/${SUITE}_arms.t107.sha weilandserver:/tmp/dsp_shared/verify/${SUITE}_arms.t107.sha | tail -1
tether exec --timeout 60s weilandserver -- bash -lc "export PATH=/usr/local/bin:/usr/bin:/bin; cd /tmp/dsp_shared/rit_pareto && echo CROSS_MISMATCH=\$(sha256sum -c /tmp/dsp_shared/verify/${SUITE}_arms.t107.sha 2>&1 | grep -vc ': OK$')"
