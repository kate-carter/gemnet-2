#!/bin/bash
#SBATCH -o test.o%j
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH -c 3
#SBATCH --mem=32gb
echo "PFSDIR"="$PFSDIR"
cp -r /home/kxc750/gemnet-2/256_matrices/*.csv $PFSDIR
cp -r /home/kxc750/gemnet-2/gemnet-2-scripts/nondet_batch.py $PFSDIR
cd $PFSDIR
ls
module load Python/3.10.8-GCCcore-12.2.0
source /home/kxc750/bct_stats/bct_venv/bin/activate
which python
python -c "import scipy; print(scipy.__version__)"
python nondet_batch.py
cp -r $PFSDIR  /home/kxc750/gemnet-2/results
rm -rf "$PFSDIR"/*
