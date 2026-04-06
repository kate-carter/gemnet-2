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
    filename='net_threshold_3072.log',
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s'
)
# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Input/output directories
input_dir = '/home/kxc750/gemnet-2/3072_corr/'
output_dir = '/home/kxc750/gemnet-2/3072_matrices/'
stats_dir = '/home/kxc750/gemnet-2/results/'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Configuration
file_list = [
    "banana_3072_cosine.csv",
    "bottle_3072_cosine.csv",
    "car_3072_cosine.csv",
    "chocolate_3072_cosine.csv",
    "cookie_3072_cosine.csv",
    "grass_3072_cosine.csv",
    "key_3072_cosine.csv",
    "knife_3072_cosine.csv",
    "paper_3072_cosine.csv",
    "phone_3072_cosine.csv",
    "pickle_3072_cosine.csv",
    "pillow_3072_cosine.csv",
    "pumpkin_3072_cosine.csv",
    "table_3072_cosine.csv",
    "wood_3072_cosine.csv",
]

# Generate threshold values from 0.05 to 0.90 in steps of 0.05
threshold_values = np.arange(0.05, 0.95, 0.05)

def process_matrix_threshold(args):
    """
    Process one matrix at one threshold value.
    
    Parameters:
    -----------
    args : tuple
        (filename, threshold, input_dir, output_dir)
    
    Returns:
    --------
    dict : Results including concept name, threshold, and percent zeroed
    """
    filename, threshold, input_dir, output_dir = args
    
    try:
        concept_name = filename.replace('_3072_cosine.csv', '')
        
        # Load matrix
        # if there are 8 metadata lines before the header row:
        df = pd.read_csv(f'{input_dir}{filename}', header=None)
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
        
        # Apply threshold: set values where |value| < threshold to 0
        matrix_thresholded = np.where(matrix < threshold, 0, matrix)
        
        # Count values after thresholding
        nonzero_after = np.count_nonzero(matrix_thresholded)
        values_zeroed = nonzero_before - nonzero_after
        percent_zeroed = (values_zeroed / total_values) * 100
        
        # Save thresholded matrix
        output_filename = f'{concept_name}_3072_{threshold:.2f}.csv'
        output_path = f'{output_dir}{output_filename}'
        pd.DataFrame(matrix_thresholded).to_csv(output_path, header=False, index=False)
        
        return {
            'concept': concept_name,
            'threshold': threshold,
            'percent_neg': percent_neg,
            'percent_zeroed': percent_zeroed,
            'retention_percent': 100 - percent_zeroed
        }
        
    except Exception as e:
        logging.error(f"Error processing {filename} at threshold {threshold:.2f}: {e}")
        return None

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Starting PARALLEL Network Thresholding")
    logging.info(f"Matrices: {len(file_list)}")
    logging.info(f"Thresholds: {len(threshold_values)}")
    logging.info(f"Total combinations: {len(file_list) * len(threshold_values)}")
    logging.info(f"Parallel processes: {num_processes}")
    logging.info(f"Threads per process: {threads_per_process}")
    logging.info("=" * 60)
    
    overall_start = time.time()
    all_results = []
    threshold_summaries = []
    
    # OUTER LOOP: Process each threshold value
    for threshold_idx, THRESHOLD in enumerate(threshold_values, 1):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"THRESHOLD {threshold_idx}/{len(threshold_values)}: |value| < {THRESHOLD:.2f}")
        logging.info(f"{'=' * 60}")
        
        start_time = time.time()
        
        # Create list of jobs for this threshold (all matrices)
        jobs = [(filename, THRESHOLD, input_dir, output_dir) for filename in file_list]
        
        # Process all matrices for this threshold in parallel
        logging.info(f"Processing {len(jobs)} matrices in parallel...")
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_matrix_threshold, jobs)
        
        # Filter out None results (failed matrices)
        results = [r for r in results if r is not None]
        
        # Store all individual results
        all_results.extend(results)
        
        # Calculate average statistics for this threshold
        if results:
            percent_neg = np.mean([r['percent_neg'] for r in results])
            avg_percent_zeroed = np.mean([r['percent_zeroed'] for r in results])
            avg_retention = 100 - avg_percent_zeroed
            
            threshold_summaries.append({
                'threshold': THRESHOLD,
                'avg_percent_neg': percent_neg,
                'avg_percent_zeroed': avg_percent_zeroed,
                'avg_retention': avg_retention,
                'n_matrices': len(results)
            })
            
            logging.info(f"Threshold {THRESHOLD:.2f} Summary:")
            logging.info(f"  Avg Percent Zeroed: {avg_percent_zeroed:.2f}%")
            logging.info(f"  Avg Retention: {avg_retention:.2f}%")
        
        threshold_time = time.time() - start_time
        logging.info(f"Threshold {THRESHOLD:.2f} completed in {threshold_time:.2f}s ({len(results)}/{len(file_list)} matrices)")
    
    # Save all individual results
    all_results_df = pd.DataFrame(all_results)
    all_results_path = f'{stats_dir}3072_threshold_ind_results_noneg.csv'
    all_results_df.to_csv(all_results_path, index=False)
    logging.info(f"\nSaved all individual results to: {all_results_path}")
    
    # Save threshold-averaged results
    summary_df = pd.DataFrame(threshold_summaries)
    summary_path = f'{stats_dir}3072_threshold_summary.csv'
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