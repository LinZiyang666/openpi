#!/bin/bash
# weilandserver: fetch the official Diffusion Policy CNN checkpoints (best-epoch file of train_0) for the tasks we ladder.
# Files are ~4 GB each (with EMA + optimizer); stored under /scratch/zixuans8/dp/ckpts/<task>/. Idempotent.
set -u
export HOME=/home/weiland
D=/data/dp/ckpts; mkdir -p $D
for T in square_mh can_mh transport_mh tool_hang_ph pusht; do
  URL=https://diffusion-policy.cs.columbia.edu/data/experiments/image/$T/diffusion_policy_cnn/train_0/checkpoints/
  mkdir -p $D/$T
  if ls $D/$T/epoch=*.ckpt >/dev/null 2>&1; then echo "$T: present $(ls $D/$T/epoch=*.ckpt | head -1)"; continue; fi
  F=$(python3 - "$URL" <<'PY'
import re, sys, urllib.request, urllib.parse
html = urllib.request.urlopen(sys.argv[1], timeout=60).read().decode()
names = [urllib.parse.unquote(n) for n in re.findall(r'href="([^"]*ckpt)"', html) if 'epoch' in n]
print(names[-1] if names else "")
PY
)
  [ -z "$F" ] && { echo "$T: NO epoch ckpt in index"; continue; }
  F=$(python3 -c "import urllib.parse,sys; print(urllib.parse.unquote(sys.argv[1]))" "$F")
  echo "$T: downloading $F"
  wget -q -O "$D/$T/$F" "$URL$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$F")" && echo "$T: done $(du -h "$D/$T/$F" | cut -f1)" || echo "$T: DOWNLOAD FAIL"
done
echo "DP_CKPTS_DONE $(date +%H:%M:%S)"
