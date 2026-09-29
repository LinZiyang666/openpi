"""Generate an unexecuted, stream-required calibration chain from pilot freeze."""
import difflib
from pathlib import Path
from .common import HERE,sha,write_json

def main():
    src=Path('/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/ops/chain_p3.stream.frozen.sh')
    original=src.read_text()
    target=original.replace('bash $ISL/run_arm_v2.sh','bash $ISL/run_arm_c_bval.sh')
    assert target!=original
    # Preserve all collection/verification/cleanup behavior. Require stream
    # before the chain creates state or could start any serving process.
    marker='RUN=${1:?run-root}; shift\n'
    guard='''# Q1 calibration: streaming is required; no whole-arm file-tar fallback.
if [ -f "$RUN/state/P3_STREAM_PORT" ]; then export P3_STREAM_PORT=$(cat "$RUN/state/P3_STREAM_PORT"); fi
: "${P3_STREAM_PORT:?calibration requires the run-specific P3 stream receiver}"
case "$P3_STREAM_PORT" in *[!0-9]*) echo 'invalid P3_STREAM_PORT' >&2;exit 2;; esac
'''
    assert target.count(marker)==1
    target=target.replace(marker,marker+guard)
    # Re-check after rereading the port file at each arm boundary, before plan.
    marker2='  MANIFEST=${OSCL_MANIFEST:-$(field_or "$arm" manifest \'\')}\n'
    assert target.count(marker2)==1
    target=target.replace(marker2,'  : "${P3_STREAM_PORT:?stream mode cannot be disabled during calibration}"\n'+marker2)
    target=target.replace('taskset -c 6-9,50-53 env','taskset -c 14-17,58-61 env')
    (HERE/'chain_calibration.sh').write_text(target)
    (HERE/'chain_calibration_v2.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),target.splitlines(True),
        fromfile='chain_p3.stream.frozen.sh',tofile='chain_calibration.sh')))
    write_json(HERE/'chain_calibration_source.json',dict(source=str(src),source_sha256=sha(src),
        current_p3_sha256=sha(HERE.parents[1]/'p3_profiling/chain_p3.sh'),output_sha256=sha(HERE/'chain_calibration.sh'),
        recommended_mode='stream required',executed=False,
        changes=['B-val launcher only for calibration arms','require receiver port before work and at each arm boundary',
                 'CPU helper affinity 14-17,58-61'],
        file_mode_limit='Original collect_client pulls a single uncompressed tar; pilot file collector splits compressed archives into 400000000-byte pieces.',
        spill_limit='Stream collector still pulls a single spill tar if transport failed; an oversize spill prevents DONE and needs coordinator chunked recovery. No telemetry is silently discarded.'))
    print('stream-required chain generated; not executed')

if __name__=='__main__':main()
