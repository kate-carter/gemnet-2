#!/bin/bash
#SBATCH -o test.o%j
#SBATCH --time=00:20:00
#SBATCH -N 1
#SBATCH -c 15
#SBATCH --mem=64gb
echo "PFSDIR"="$PFSDIR"
cp -r /home/kxc750/gemnet-2/mg_corr/*.csv $PFSDIR
cp -r /home/kxc750/gemnet-2/gemnet-2-scripts/auto_threshold.py $PFSDIR
cd $PFSDIR
module load Python/3.10.8-GCCcore-12.2.0
source /home/kxc750/bct_stats/bct_venv/bin/activate
which python
python -c "import scipy; print(scipy.__version__)"
python auto_threshold.py
cp -r $PFSDIR  /home/kxc750/gemnet-2/mg_matrices
rm -rf "$PFSDIR"/*