#!/bin/bash
# after_collect.sh MODEL SEED RUN: wait for "<RUN> ended rc=0" / "seed N ended rc=0" in chain.log, then finish the library on CPU
# and compute the seeds_1023 MT50 products for that seed (6 sizes in parallel, then manifest).
MODEL=$1; SEED=$2; RUN=$3
C=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/chain.log
R=/data/r17_library/seed$SEED/${MODEL}_mt50/campaigns/$RUN
log(){ echo "$(date +%H:%M:%S) [after-collect $MODEL s$SEED] $*" >> $C; }
until [ -f $R/RECEIPT.json ] && python3 -c "import json,sys;d=json.load(open('$R/RECEIPT.json'));sys.exit(0 if d.get('status')=='PASS' and d.get('all_owned_closed') else 1)" 2>/dev/null \
      && ! pgrep -f "[p]repared-run $R" >/dev/null && ! pgrep -f "[p]repared-run $(readlink -f $R)" >/dev/null; do sleep 60; done
cd /home/weiland/projects/openpi
P="nice -n 10 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python"
O=$CLAUDE_JOB_DIR/tmp/after_${MODEL}_$SEED; mkdir -p $O
log "finish $RUN"
$P -m exp.r17_integration.ops.library_campaign --model $MODEL --suite mt50 --library-root /data/r17_library/seed$SEED --finish $R > $O/finish.log 2>&1 || { log "finish FAILED"; exit 1; }
log "finish done; products"
B=${MODEL}
pids=(); for n in 250 500 1000 1500 2000 2500; do $P -m exp.offline_search.rounds.r17.seeds_1023.inputs cell --bundle $B --seed $SEED --suite mt50 --size $n > $O/cell_$n.log 2>&1 & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { log "products FAILED"; exit 1; }
$P -m exp.offline_search.rounds.r17.seeds_1023.inputs manifest --bundle $B --seed $SEED > $O/manifest.log 2>&1 || { log "manifest FAILED"; exit 1; }
log "products done"
