#!/usr/bin/env bash
# Idempotent even if tether exec h100 invokes the script twice. CPU only.
set -euo pipefail
export HOME=/home/exouser
DIR=/home/exouser/r08abl_clip_sc
mkdir -p "$DIR"
exec 9>"$DIR/proof.lock"
flock -x 9
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export R8_CLIP_WEIGHTS="$DIR/open_clip_model.safetensors"
cd "$DIR"
code_sha=$(sha256sum encoder.py | cut -d" " -f1)
for pair in "pi05:/home/exouser/openpi/.venv/bin/python" "groot:/home/exouser/gr00t_n15_venv/.venv/bin/python"; do
    tag=${pair%%:*}
    py=${pair#*:}
    if ! test -f "${tag}_encoder_proof.json" || ! test -f "${tag}_code.sha" || ! test "$(cat "${tag}_code.sha")" = "$code_sha"; then
        "$py" proof.py --inputs proof_inputs.npz --output "${tag}_encoder_proof.json" >"${tag}_encoder_proof.log" 2>&1
        echo "$code_sha" >"${tag}_code.sha"
    fi
    cat "${tag}_encoder_proof.json"
done
df -h /data /
