#!/bin/bash
# timan108: the env runners need the demo hdf5 only for env metadata + train-episode init states (we run n_train=0),
# so the 1 GB lowdim set replaces the 78 GB image set; PushT needs nothing but we fetch the 29 MB zarr for completeness.
set -u
export HOME=/home/zixuans8
D=/scratch/zixuans8/dp/data; mkdir -p $D; cd $D
[ -d robomimic ] || { wget -q -O robomimic_lowdim.zip https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip && unzip -q -o robomimic_lowdim.zip && rm -f robomimic_lowdim.zip; }
[ -d pusht ] || { wget -q -O pusht.zip https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip && unzip -q -o pusht.zip && rm -f pusht.zip; }
find $D -maxdepth 4 | head -40; du -sh $D
echo "DP_DATA_DONE $(date +%H:%M:%S)"
