# Parallel BCT Network Analysis
# set up to process 15 matrices at 18 different network thresholds
# outputs report individual network statistics as well as averaged
# statistics across concepts per network threshold
# bct network statistics (deterministic): core-fit, clustering coef (weighted unweighted),
# strength, and degree

#Set thread limits before any imports (Python 3.6 compatibility)
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


# Set numpy thread limits for HPC - MUST BE BEFORE IMPORTS
# Get total CPUs from SLURM
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 15))

# Number of concepts to process per threshold
num_concepts = 15  # Update this if your concept list changes

# Determine parallelization strategy
# Use one process per concept (up to available CPUs)
num_processes = min(total_cpus, num_concepts)
threads_per_process = 1  # Pure multiprocessing, no threading within processes

os.environ['OMP_NUM_THREADS'] = str(threads_per_process)
os.environ['MKL_NUM_THREADS'] = str(threads_per_process)

# NOW import numpy/scipy/bct AFTER setting environment variables
import numpy as np
import scipy
import bct
import pandas as pd
import logging
import time
from multiprocessing import Pool

#set up logging (for multiprocessing)
logging.basicConfig(
    filename='512_det.log',
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s'
)

#logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Configuration
threshold_values = np.arange(0.05, 0.95, 0.05)

concept_names = [
    'banana',
    'bottle',
    'car',
    'chocolate',
    'cookie',
    'grass',
    'key',
    'knife',
    'paper',
    'phone',
    'pickle',
    'pillow',
    'pumpkin',
    'table',
    'wood',
]

# Directories
output_dir = './'
input_dir = './'

#setting directories as enviornment variables
os.environ['OUTPUT_DIR'] = output_dir
os.environ['INPUT_DIR'] = input_dir

# Function to process a single network
def process_network(args):
    """Process one concept matrix at one threshold"""
    concept_name, threshold, input_dir = args
    
    try:
        # Construct filename
        filename = f'{concept_name}_512_{threshold:.2f}.csv'
        filepath = f'{input_dir}{filename}'
        
        # Load correlation matrix
        matrix = pd.read_csv(filepath, header=None).values
        
        # Check if matrix is too sparse (all or nearly all zeros)
        if np.count_nonzero(matrix) < (0.01 * matrix.size):
            logging.warning(f"Skipping {concept_name} at threshold {threshold:.2f}: matrix is more than 99% zeros")
            return None
        
        # Core-periphery fit
        _, corefit = bct.core_periphery_dir(matrix)
        
        # Clustering coefficient
        cc = bct.clustering_coef_wu(matrix)
        clustering_mean = float(np.mean(cc))
        
        # Strength
        s = bct.strengths_und(matrix)
        strength_mean = float(np.mean(s))
        
        # Degree
        d = bct.degrees_und(matrix)
        degree_mean = float(np.mean(d))
        
        return {
            'concept': str(concept_name),
            'threshold': float(threshold),
            'corefit': float(corefit),
            'clustering': float(clustering_mean),
            'strength': float(strength_mean),
            'degree': float(degree_mean),
            # Store arrays for individual node results
            'strength_array': np.asarray(s),
            'clustering_array': np.asarray(cc),
            'degree_array': np.asarray(d)
        }
        
    except Exception as e:
        logging.error(f"Error processing {concept_name} at threshold {threshold:.2f}: {e}")
        return None


if __name__ == "__main__":
    logging.info("="*60)
    logging.info("Starting BCT Deterministic Analysis with Thresholding (512 dimensions)")
    logging.info(f"Concepts: {len(concept_names)}")
    logging.info(f"Thresholds: {len(threshold_values)}")
    logging.info(f"Total combinations: {len(concept_names) * len(threshold_values)}")
    logging.info(f"Parallel processes: {num_processes}")
    logging.info("="*60)
    
    overall_start = time.time()
    all_results = []
    threshold_summaries = []
    
    # OUTER LOOP: Process each threshold value
    for threshold_idx, THRESHOLD in enumerate(threshold_values, 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"THRESHOLD {threshold_idx}/{len(threshold_values)}: {THRESHOLD:.2f}")
        logging.info(f"{'='*60}")
        
        start_time = time.time()
        
        # Create list of jobs for this threshold (all concepts)
        jobs = [(concept, THRESHOLD, input_dir) for concept in concept_names]
        
        # Process networks in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_network, jobs)
        
        # Filter out None results (failed networks)
        results = [r for r in results if r is not None]
        
        # Store all individual results
        all_results.extend(results)
        
        # Calculate averaged statistics across concepts for this threshold
        if results:
            avg_corefit = float(np.mean([r['corefit'] for r in results]))
            avg_clustering = float(np.mean([r['clustering'] for r in results]))
            avg_strength = float(np.mean([r['strength'] for r in results]))
            avg_degree = float(np.mean([r['degree'] for r in results]))
            
            threshold_summaries.append({
                'threshold': float(THRESHOLD),
                'avg_corefit': avg_corefit,
                'avg_clustering': avg_clustering,
                'avg_strength': avg_strength,
                'avg_degree': avg_degree,
                'n_networks': int(len(results))
            })
            
            logging.info(f"Threshold {THRESHOLD:.2f} Summary:")
            logging.info(f"  Avg Core-fit: {avg_corefit:.6f}")
            logging.info(f"  Avg Clustering: {avg_clustering:.6f}")
            logging.info(f"  Avg Strength: {avg_strength:.6f}")
            logging.info(f"  Avg Degree: {avg_degree:.6f}")
        
        threshold_time = time.time() - start_time
        logging.info(f"Threshold {THRESHOLD:.2f} completed in {threshold_time:.2f}s ({threshold_time/60:.2f} min) - {len(results)}/{len(concept_names)} networks")
    
    # Save all individual results
    all_results_df = pd.DataFrame(all_results)
    all_results_path = f'{output_dir}512_det_results.csv'
    all_results_df.to_csv(all_results_path, index=False)
    logging.info(f"\nSaved all individual results to: {all_results_path}")

    # Save individual per-node results (long format: one row per node)
    individual_node_results = []
    for r in all_results:
        concept = r['concept']
        threshold = r['threshold']
        strength_array = np.asarray(r['strength_array'])
        clustering_array = np.asarray(r['clustering_array'])
        degree_array = np.asarray(r['degree_array'])
        n_nodes = len(strength_array)
        
        for node_idx in range(n_nodes):
            individual_node_results.append({
                'concept': str(concept),
                'threshold': float(threshold),
                'node': int(node_idx),
                'strength': float(strength_array[node_idx]),
                'clustering': float(clustering_array[node_idx]),
                'degree': float(degree_array[node_idx])
            })
    
    individual_df = pd.DataFrame(individual_node_results)
    individual_path = f'{output_dir}512_det_individual_nodes.csv'
    individual_df.to_csv(individual_path, index=False)
    logging.info(f"Saved per-node individual results to: {individual_path}")
    
    # Save threshold-averaged results
    summary_df = pd.DataFrame(threshold_summaries)
    summary_path = f'{output_dir}512_det_avgs.csv'
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Saved threshold-averaged results to: {summary_path}")
    
    # Final summary
    total_time = time.time() - overall_start
    logging.info("\n" + "="*60)
    logging.info("All processing completed!")
    logging.info(f"Total time: {total_time/60:.2f} minutes")
    logging.info(f"Processed: {len(all_results)} total network-threshold combinations")
    logging.info(f"Output directory: {output_dir}")
    logging.info("="*60)