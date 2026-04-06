#!/bin/bash
#SBATCH -o test.o%j
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH -c 3
#SBATCH --mem=32gb
echo "PFSDIR"="$PFSDIR"
cp -r /home/kxc750/3_corrvar/ma_matrices/*.csv $PFSDIR
cp -r /home/kxc750/3_corrvar/corrvar_scripts/ma_nondet.py $PFSDIR
cd $PFSDIR
ls
module load Python/3.10.8-GCCcore-12.2.0
source /home/kxc750/bct_stats/bct_venv/bin/activate
which python
python -c "import scipy; print(scipy.__version__)"
python nondet_batch.py
cp -r $PFSDIR  /home/kxc750/3_corrvar/batchtest
rm -rf "$PFSDIR"/*