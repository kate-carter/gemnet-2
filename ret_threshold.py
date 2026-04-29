# Parallel Network Thresholding for HPC with no negative values
# Applies multiple thresholds to multiple matrices in parallel
# disclosure: this script has been partially written by Kate Carter and partially using Claude 3.5 Sonnet

import os

# Set numpy thread limits for HPC - MUST BE BEFORE IMPORTS
num_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', 15))
threads_per_process = 1
os.environ['OMP_NUM_THREADS'] = str(threads_per_process)
os.environ['MKL_NUM_THREADS'] = str(threads_per_process)

# NOW import numpy/pandas AFTER setting environment variables
import numpy as np
import pandas as pd
import logging
import time
from multiprocessing import Pool

# Setup logging (multiprocessing-safe)
logging.basicConfig(
    filename='ret_threshold_spearman.log',
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s'
)
# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Input/output directories
input_dir = './'
output_dir = './'
stats_dir = './'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Configuration
concept_name = [
    "banana",
    "bottle",
    "car",
    "chocolate",
    "cookie",
    "grass",
    "key",
    "knife",
    "paper",
    "phone",
    "pickle",
    "pillow",
    "pumpkin",
    "table",
    "wood",
]

# Generate retention threshold values from 10% to 90% in steps of 10%
retention_values = np.arange(10, 90, 10)
retention_threshold = [f"{value:.2f}" for value in retention_values]

#define target dimensionalities
target_dimensionalities = (3072, 1536, 768, 512, 256)

def process_matrix_threshold(args):

    current_concept, dim, input_dir, output_dir, retention_threshold = args

    #construct filenames/list for logging
    input_filename = f"{current_concept}_{dim}_spearman.csv"
    output_filename = f"{current_concept}_{dim}_{retention_threshold}.csv"    
    
    try:
        # Load matrix
        # if there are 8 metadata lines before the header row:
        df = pd.read_csv(os.path.join(input_dir, input_filename), header=None)
        matrix = df.values
        print(f"  Matrix shape: {matrix.shape}")
        
        # Count values before thresholding
        total_values = matrix.size
        nonzero_before = np.count_nonzero(matrix)

        #remove negative values
        matrix = np.where(matrix <0, 0, matrix)

        # count values after removing negative values
        nonzero_after_neg = np.count_nonzero(matrix)
        values_neg = nonzero_before - nonzero_after_neg
        percent_neg = (values_neg / total_values) * 100

        #calculate retention threshold
        used_threshold = np.percentile(matrix, retention_threshold)
        
        # Apply threshold: set values where |value| < threshold to 0
        matrix_thresholded = np.where(matrix < used_threshold, 0, matrix)
        
        # Count values after thresholding
        nonzero_after = np.count_nonzero(matrix_thresholded)
        values_zeroed = nonzero_before - nonzero_after
        percent_zeroed = (values_zeroed / total_values) * 100
        
        # Save thresholded matrix
        output_path = os.path.join(output_dir, output_filename)
        pd.DataFrame(matrix_thresholded).to_csv(output_path, header=False, index=False)
        
        return {
            'concept': current_concept,
            'dim': dim,
            'percent_neg': percent_neg,
            'percent_zeroed': percent_zeroed,
            'target retention': retention_threshold,
            'retention': 100 - percent_zeroed,
            'used threshold': used_threshold
        }
        
    except Exception as e:
        logging.error(f"Error processing {input_filename} at threshold {retention_threshold}: {e}")
        return None

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Starting PARALLEL Network Thresholding")
    logging.info(f"Retention Thresholds: {len(retention_threshold)}")
    logging.info(f"Total combinations: {len(concept_name) * len(retention_threshold) * len(target_dimensionalities)}")
    logging.info(f"Parallel processes: {num_processes}")
    logging.info(f"Threads per process: {threads_per_process}")
    logging.info("=" * 60)
    
    overall_start = time.time()

    # OUTER LOOP: Process each target dimensionality
    for dim in target_dimensionalities:

        all_results = []
        threshold_summaries = []
        used_thresholds = []

        # INNER LOOP: Process each threshold value
        for retention_threshold_idx, retention_threshold in enumerate(retention_threshold, 1):
            logging.info(f"\n{'=' * 60}")
            logging.info(f"RETENTION THRESHOLD {retention_threshold}")
            logging.info(f"{'=' * 60}")
            
            start_time = time.time()
            
            # Create list of jobs for this threshold (all matrices)
            jobs = [(current_concept, dim, input_dir, output_dir, retention_threshold) for current_concept in concept_name]
            
            # Process all matrices for this threshold in parallel
            logging.info(f"Processing {len(jobs)} matrices in parallel...")
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_matrix_threshold, jobs)
            
            # Filter out None results (failed matrices)
            results = [r for r in results if r is not None]
            
            # Store all individual results
            all_results.extend(results)

            #store used threshold values
            used_threshold = np.mean([r['used threshold'] for r in results])

            # Calculate average statistics for this threshold
            if results:
                percent_neg = np.mean([r['percent_neg'] for r in results])
                avg_percent_zeroed = np.mean([r['percent_zeroed'] for r in results])
                avg_retention = 100 - avg_percent_zeroed
                
                threshold_summaries.append({
                    'retention_threshold': retention_threshold,
                    'avg_percent_neg': percent_neg,
                    'avg_percent_zeroed': avg_percent_zeroed,
                    'avg_retention': avg_retention,
                    'n_matrices': len(results),
                    'used threshold': used_threshold
                })
                
                logging.info(f"Threshold {retention_threshold} Summary:")
                logging.info(f"  Numerical Threshold: {used_threshold:.2f}")
                logging.info(f"  Avg Retention: {avg_retention:.2f}%")
            
            threshold_time = time.time() - start_time
            logging.info(f"Threshold {retention_threshold:.2f} completed in {threshold_time:.2f}s ({len(results)}/{len(concept_name)} matrices)")
        
        # Save individual dimension results
        all_results_df = pd.DataFrame(all_results)
        all_results_path = os.path.join(f'{stats_dir}{dim}_threshold_ind_spearman.csv')
        all_results_df.to_csv(all_results_path, index=False)
        logging.info(f"\nSaved all individual results to: {all_results_path}")
        
        # Save threshold-averaged results
        summary_df = pd.DataFrame(threshold_summaries)
        summary_path = os.path.join(f'{stats_dir}{dim}_threshold_summary_spearman.csv')
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Saved threshold summary to: {summary_path}")
    # Final summary
    total_time = time.time() - overall_start
    logging.info("\n" + "=" * 60)
    logging.info("All processing completed!")
    logging.info(f"Total time: {total_time / 60:.2f} minutes")
    logging.info(f"Processed: {len(all_results)} total matrix-threshold combinations")
    logging.info(f"Matrix output directory: {output_dir}")
    logging.info(f"Stats output directory: {stats_dir}")
    logging.info("=" * 60)