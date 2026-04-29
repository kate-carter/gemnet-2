#!/bin/bash
#SBATCH -o test.o%j
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH -c 15
#SBATCH --mem=64gb
echo "PFSDIR"="$PFSDIR"
cp -r /home/kxc750/gemnet-2/512_matrices/*.csv $PFSDIR
cp -r /home/kxc750/gemnet-2/gemnet-2-scripts/nondet512.py $PFSDIR
cp -r /home/kxc750/gemnet-2/gemnet-2-scripts/det512.py $PFSDIR
cd $PFSDIR
module load Python/3.10.8-GCCcore-12.2.0
source /home/kxc750/bct_stats/bct_venv/bin/activate
which python
python -c "import scipy; print(scipy.__version__)"
python nondet512.py
python det512.py
cp -r 512_det_avgs.csv  /home/kxc750/gemnet-2/p_results
cp -r 512_det_individual_nodes.csv  /home/kxc750/gemnet-2/p_results
cp -r 512_det.log /home/kxc750/gemnet-2/p_results
cp -r 512_det_results.csv /home/kxc750/gemnet-2/p_results
cp -r 512_nondet_all_modularity1.csv /home/kxc750/gemnet-2/p_results
cp -r 512_nondet_results.csv /home/kxc750/gemnet-2/p_results
cp -r 512_nondet.log /home/kxc750/gemnet-2/p_results
cp -r 512_nondet_first2iterations1.csv /home/kxc750/gemnet-2/p_results
cp -r 512_nondet_results1.csv /home/kxc750/gemnet-2/p_results
rm -rf "$PFSDIR"/*
