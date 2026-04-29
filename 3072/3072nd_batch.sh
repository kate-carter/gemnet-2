#!/bin/bash
#SBATCH -o test.o%j
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH -c 30 
#SBATCH --mem=128gb 
echo "PFSDIR"="$PFSDIR"  
cp -r /home/kxc750/gemnet-2/3072_matrices/*.csv $PFSDIR 
cp -r /home/kxc750/gemnet-2/gemnet-2-scripts/nondet3072.py $PFSDIR
cd $PFSDIR
module load Python/3.10.8-GCCcore-12.2.0
source /home/kxc750/bct_stats/bct_venv/bin/activate
which python
python -c "import scipy; print(scipy.__version__)"
python nondet3072.py 
cp -r 3072_nondet_all_modularity1.csv /home/kxc750/gemnet-2/p_results
cp -r 3072_nondet_results.csv /home/kxc750/gemnet-2/p_results
cp -r 3072_nondet.log /home/kxc750/gemnet-2/p_results
cp -r 3072_nondet_first2iterations1.csv /home/kxc750/gemnet-2/p_results
cp -r 3072_nondet_results1.csv /home/kxc750/gemnet-2/p_results
rm -rf "$PFSDIR"/*
