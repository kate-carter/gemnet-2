# Parallel BCT Network Analysis NONDETERMINISTIC
# set up to process x matrices at 18 different network thresholds
# outputs report individual network statistics as well as averaged
# statistics across concepts per network threshold
# bct network statistics (non-deterministic): modularity and diversity
# Each network is analyzed with 100 iterations, then averaged
# This specific version was modified on 3/23/27 to run 15 networks @512 dim

import os

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
import bct as bct
import pandas as pd
import logging
import time
from multiprocessing import Pool

# set up logging (for multiprocessing)
logging.basicConfig(
    filename='512_nondet.log',
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s'
)

# logging to console
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
input_dir = './'
output_dir = './'

# Number of iterations per network
nIt = 100


def run_single_iteration(thisnetwork):
    """
    Run a single iteration of modularity and diversity calculation.

    Parameters:
    -----------
    thisnetwork : np.ndarray
        The network adjacency matrix

    Returns:
    --------
    tuple : (Q, mean_Di) - modularity and mean diversity for this iteration
    """
    # Community detection (matching MATLAB line 21: [M,Q]=community_louvain)
    ci, Q = bct.modularity_louvain_und(thisnetwork, gamma=1)

    # Diversity coefficient (matching MATLAB line 22: Di= diversity_coef_sign)
    Di = bct.diversity_coef_sign(thisnetwork, ci)
    Di = np.asarray(Di)  # Convert to array if it's a tuple

    # diversity_coef_sign returns shape (2, n_nodes) for signed networks
    # Row 0: within-module positive, Row 1: within-module negative
    # Average across both to get total diversity per node
    if Di.ndim == 2:
        Di_averaged = np.mean(Di, axis=0)
    else:
        Di_averaged = Di

    # Return modularity, original Di array, and mean diversity
    return (float(Q), Di, float(np.mean(Di_averaged)))


# Function to process a single network
def process_network(args):
    """Process one concept matrix at one threshold with 100 iterations"""
    concept_name, threshold, input_dir = args

    try:
        # Construct filename
        filename = f'{concept_name}_512_{threshold:.2f}.csv'
        filepath = os.path.join(input_dir, filename)

        # Load correlation matrix
        matrix = pd.read_csv(filepath, header=None).values

        # Check if matrix is too sparse (all or nearly all zeros)
        if np.count_nonzero(matrix) < (0.01 * matrix.size):
            logging.warning(f"Skipping {concept_name} at threshold {threshold:.2f}: matrix is more than 99% zeros")
            return None

        # Run 100 iterations and collect results
        itMeasures_list = []
        first_two_iterations = []  # Store first 2 iterations for detailed output
        all_modularity_values = []  # Store ALL 100 modularity values

        for iteration_idx in range(nIt):
            Q, Di, mean_Di = run_single_iteration(matrix)
            itMeasures_list.append((Q, mean_Di))
            all_modularity_values.append(Q)  # Store this iteration's modularity

            # Save first 2 iterations with full Di array
            if iteration_idx < 2:
                first_two_iterations.append({
                    'iteration': int(iteration_idx + 1),
                    'modularity': float(Q),
                    'diversity_array': np.asarray(Di),  # Ensure array
                    'diversity_mean': float(mean_Di)
                })

        # Convert to array and calculate means
        itMeasures = np.array(itMeasures_list)
        mean_modularity = float(np.mean(itMeasures[:, 0]))  # Explicit float conversion
        mean_diversity = float(np.mean(itMeasures[:, 1]))  # Explicit float conversion

        return {
            'concept': concept_name,
            'threshold': float(threshold),  # Ensure float
            'modularity': mean_modularity,
            'diversity': mean_diversity,
            'first_two_iterations': first_two_iterations,
            'all_modularity_values': [float(q) for q in all_modularity_values]  # Ensure all floats
        }

    except Exception as e:
        logging.error(f"Error processing {concept_name} at threshold {threshold:.2f}: {e}")
        return None


if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Starting BCT Non-Deterministic Analysis with Thresholding")
    logging.info(f"Concepts: {len(concept_names)}")
    logging.info(f"Thresholds: {len(threshold_values)}")
    logging.info(f"Iterations per network: {nIt}")
    logging.info(f"Total combinations: {len(concept_names) * len(threshold_values)}")
    logging.info(f"Parallel processes: {num_processes}")
    logging.info(f"Threads per process: {threads_per_process}")
    logging.info("=" * 60)

    overall_start = time.time()
    all_results = []
    threshold_summaries = []

    # OUTER LOOP: Process each threshold value
    for threshold_idx, THRESHOLD in enumerate(threshold_values, 1):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"THRESHOLD {threshold_idx}/{len(threshold_values)}: {THRESHOLD:.2f}")
        logging.info(f"{'=' * 60}")

        start_time = time.time()

        # Create list of jobs for this threshold (all concepts)
        jobs = [(concept, THRESHOLD, input_dir) for concept in concept_names]

        # Process networks in parallel (each runs 100 iterations internally)
        logging.info(f"[{len(jobs)}/{len(jobs)}] matrices loaded, starting processing ({nIt} iterations each)...")
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_network, jobs)

        # Filter out None results (failed networks)
        results = [r for r in results if r is not None]
        logging.info(f"  ✓ Completed {len(results)}/{len(jobs)} networks")

        # Store all individual results
        all_results.extend(results)

        # Calculate averaged statistics across concepts for this threshold
        if results:
            avg_modularity = float(np.mean([r['modularity'] for r in results]))
            avg_diversity = float(np.mean([r['diversity'] for r in results]))

            threshold_summaries.append({
                'threshold': float(THRESHOLD),
                'avg_modularity': avg_modularity,
                'avg_diversity': avg_diversity,
                'n_networks': int(len(results))
            })

            logging.info(f"Threshold {THRESHOLD:.2f} Summary:")
            logging.info(f"  Avg Modularity: {avg_modularity:.6f}")
            logging.info(f"  Avg Diversity: {avg_diversity:.6f}")

        threshold_time = time.time() - start_time
        logging.info(
            f"Threshold {THRESHOLD:.2f} completed in {threshold_time:.2f}s ({threshold_time / 60:.2f} min) - {len(results)}/{len(concept_names)} networks")

    # Save summary results (means only, no iteration details)
    summary_results = []
    for r in all_results:
        summary_results.append({
            'concept': str(r['concept']),
            'threshold': float(r['threshold']),
            'modularity': float(r['modularity']),
            'diversity': float(r['diversity'])
        })

    all_results_df = pd.DataFrame(summary_results)
    all_results_path = f'{output_dir}512_nondet_results1.csv'
    all_results_df.to_csv(all_results_path, index=False)
    logging.info(f"\nSaved summary results to: {all_results_path}")

    # Save first 2 iterations per concept-threshold (per-node diversity values)
    first_iterations_data = []
    for r in all_results:
        concept = r['concept']
        threshold = r['threshold']

        # Debug: Check if first_two_iterations exists
        if 'first_two_iterations' not in r:
            logging.warning(f"No first_two_iterations for {concept} at {threshold}")
            continue

        for iter_data in r['first_two_iterations']:
            iteration = iter_data['iteration']
            modularity = iter_data['modularity']
            diversity_array = np.asarray(iter_data['diversity_array'])  # Ensure it's an array

            # diversity_coef_sign returns shape (2, n_nodes) for signed networks
            # Row 0: within-module positive, Row 1: within-module negative
            # We'll use the total diversity (mean of both)
            if diversity_array.ndim == 2:
                diversity_array = np.mean(diversity_array, axis=0)  # Average across positive/negative

            # One row per node
            n_nodes = len(diversity_array)

            for node_idx in range(n_nodes):
                first_iterations_data.append({
                    'concept': str(concept),
                    'threshold': float(threshold),
                    'iteration': int(iteration),
                    'modularity': float(modularity),
                    'node': int(node_idx),
                    'diversity': float(diversity_array[node_idx])
                })

    first_iterations_df = pd.DataFrame(first_iterations_data)
    first_iterations_path = f'{output_dir}512_nondet_first2iterations1.csv'
    first_iterations_df.to_csv(first_iterations_path, index=False)
    logging.info(f"Saved first 2 iterations (per-node) to: {first_iterations_path}")

    # Save all 100 modularity values per concept-threshold
    all_modularity_data = []
    for r in all_results:
        concept = r['concept']
        threshold = r['threshold']
        modularity_values = r['all_modularity_values']

        # One row per iteration
        for iteration_idx, Q in enumerate(modularity_values, start=1):
            all_modularity_data.append({
                'concept': str(concept),
                'threshold': float(threshold),
                'iteration': int(iteration_idx),
                'modularity': float(Q)
            })

    modularity_df = pd.DataFrame(all_modularity_data)
    modularity_path = f'{output_dir}512_nondet_all_modularity1.csv'
    modularity_df.to_csv(modularity_path, index=False)
    logging.info(f"Saved all 100 modularity values per concept-threshold to: {modularity_path}")

    # Save threshold-averaged results
    summary_df = pd.DataFrame(threshold_summaries)
    summary_path = f'{output_dir}512_nondet_results.csv'
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Saved threshold-averaged results to: {summary_path}")

    # Final summary
    total_time = time.time() - overall_start
    logging.info("\n" + "=" * 60)
    logging.info("All processing completed!")
    logging.info(f"Total time: {total_time / 60:.2f} minutes")
    logging.info(f"Processed: {len(all_results)} total network-threshold combinations")
    logging.info(f"Output directory: {output_dir}")
    logging.info("=" * 60)