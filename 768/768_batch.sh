#!/bin/bash
#SBATCH -o test.o%j
#SBATCH --time=05:00:00
#SBATCH -N 1
#SBATCH -c 15
#SBATCH --mem=64gb
echo "PFSDIR"="$PFSDIR"
cp -r /home/kxc750/gemnet-2/768_matrices/*.csv $PFSDIR
cp -r /home/kxc750/gemnet-2/gemnet-2-scripts/nondet768.py $PFSDIR
cp -r /home/kxc750/gemnet-2/gemnet-2-scripts/det768.py $PFSDIR
cd $PFSDIR
module load Python/3.10.8-GCCcore-12.2.0
source /home/kxc750/bct_stats/bct_venv/bin/activate
which python
python -c "import scipy; print(scipy.__version__)"
python nondet768.py
python det768.py
cp -r $PFSDIR  /home/kxc750/gemnet-2/s_results
rm -rf "$PFSDIR"/*
 