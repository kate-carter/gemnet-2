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
from collections import Counter

# Setup logging (multiprocessing-safe)
logging.basicConfig(
    filename='ret_threshold_mahalanobis.log',
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

# Generate retention targets in steps of 5%
retention_values = np.arange(95, 0, -5)
retention_targets = [value for value in retention_values]

#define target dimensionalities
target_dimensionalities = (3072, 1536, 768, 512, 256)

def process_matrix_threshold(args):

    current_concept, dim, input_dir, output_dir, retention_target = args

    #construct filenames/list for logging
    input_filename = f"{current_concept}_{dim}_mahalanobis.csv"
    output_filename = f"{current_concept}_{dim}_0.{retention_target}.csv"    
    
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
        noneg_matrix = np.where(matrix <0, 0, matrix)

        # count values after removing negative values
        nonzero_after_neg = np.count_nonzero(noneg_matrix)
        values_neg = nonzero_before - nonzero_after_neg
        percent_neg = (values_neg / total_values) * 100
        
        #calculate retention target and required kept count
        keep_fraction = retention_target / 100
        required_kept_count = int(np.ceil(keep_fraction * total_values))

        #calculate drop percentile
        drop_percentile_q = 100 - retention_target

        if nonzero_after_neg >= required_kept_count:
            #calculate retention threshold
            used_threshold = np.percentile(noneg_matrix.flatten(), drop_percentile_q)
            
            # Apply threshold: set values where |value| < threshold to 0
            matrix_thresholded = np.where(noneg_matrix < used_threshold, 0, noneg_matrix)
            
            # Count values after thresholding
            nonzero_after = np.count_nonzero(matrix_thresholded)
            values_zeroed = nonzero_before - nonzero_after
            percent_zeroed = (values_zeroed / total_values) * 100
            
            # Save thresholded matrix
            output_path = os.path.join(output_dir, output_filename)
            pd.DataFrame(matrix_thresholded).to_csv(output_path, header=False, index=False)

            return {
                'skipped': False,
                'skipped reason': None,
                'concept': current_concept,
                'dim': dim,
                'percent_neg': percent_neg,
                'percent_zeroed': percent_zeroed,
                'target retention': retention_target,
                'retention': 100 - percent_zeroed,
                'used threshold': used_threshold
            }
        else: 
            print(f"  Matrix {input_filename} has less than {retention_target} non-zero values after removing negative values")
            return {
                'skipped': True,
                'skipped reason': "insufficient values",
                'concept': current_concept,
                'dim': dim,
                'percent_neg': percent_neg,
                'target retention': retention_target,
                'used threshold': None,
            }
        
    except Exception as e:
        logging.error(f"Error processing {input_filename} at threshold {retention_target}: {e}")
        return None

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Starting PARALLEL Network Thresholding")
    logging.info(f"Retention Thresholds: {len(retention_targets)}")
    logging.info(f"Total combinations: {len(concept_name) * len(retention_targets) * len(target_dimensionalities)}")
    logging.info(f"Parallel processes: {num_processes}")
    logging.info(f"Threads per process: {threads_per_process}")
    logging.info("=" * 60)
    
    overall_start = time.time()
    total_dim_processed = []

    # OUTER LOOP: Process each target dimensionality
    for dim in target_dimensionalities:
        threshold_summaries = []

        # Store all individual results
        individual_results = []

        # INNER LOOP: Process each threshold value
        for retention_target_idx, retention_target in enumerate(retention_targets, 1):

            logging.info(f"\n{'=' * 60}")
            logging.info(f"RETENTION THRESHOLD {retention_target}")
            logging.info(f"{'=' * 60}")
            
            start_time = time.time()

            # Create list of jobs for this threshold (all matrices)
            jobs = [(current_concept, dim, input_dir, output_dir, retention_target) for current_concept in concept_name]
            
            # Process all matrices for this threshold in parallel
            logging.info(f"Processing {len(jobs)} matrices in parallel...")
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_matrix_threshold, jobs)
            
            # Filter out None results (failed matrices)
            results = [r for r in results if r is not None]
            individual_results.extend(results)

            # Calculate average statistics for this threshold
            if results and not all(r['skipped'] for r in results):
                processed_rows = [r for r in results if not r['skipped']]
                percent_neg = np.mean([r['percent_neg'] for r in processed_rows])
                avg_percent_zeroed = np.mean([r['percent_zeroed'] for r in processed_rows])
                avg_retention = 100 - avg_percent_zeroed
                
                #skipped rows
                skipped_rows = [r for r in results if r['skipped']]
                percent_skipped = (len(skipped_rows) / len(results)) * 100                
                used_threshold = float(np.mean([r['used threshold'] for r in processed_rows]))
            
                threshold_summaries.append({
                    'retention_threshold': retention_target,
                    'avg_percent_neg': percent_neg,
                    'avg_percent_zeroed': avg_percent_zeroed,
                    'avg_retention': avg_retention,
                    'n_processed': len(processed_rows),
                    'n_skipped': len(skipped_rows),
                    'percent_skipped': percent_skipped,
                    'used threshold': used_threshold
                })
                logging.info(f"Threshold {retention_target} Summary:")
                logging.info(f"Numerical Threshold: {used_threshold:.2f}")
                logging.info(f"Avg Retention: {avg_retention:.2f}%")
                if skipped_rows:
                    reason_counts = Counter(
                        r.get('skipped reason', 'unknown') for r in skipped_rows
                    )
                    logging.info(f"Skipped: {len(skipped_rows)} ({percent_skipped:.2f}%)")
                    logging.info(f"Skip reasons: {dict(reason_counts)}")
            else:
                skipped_rows = [r for r in results if r['skipped']]
                reason_counts = Counter(
                    r.get('skipped reason', 'unknown') for r in skipped_rows
                )
                logging.info(f"Threshold {retention_target} Summary:")
                logging.info(f"No matrices processed")
                logging.info(f"Skipped: {len(skipped_rows)}")
                logging.info(f"Skip reasons: {dict(reason_counts)}")


            threshold_time = time.time() - start_time
            logging.info(f"Threshold {retention_target} completed in {threshold_time:.2f}s ({len(results)}/{len(concept_name)} matrices)")
        
        # Save individual dimension results
        all_results_df = pd.DataFrame(individual_results)
        all_results_path = os.path.join(stats_dir, f"{dim}_threshold_ind_mahalanobis.csv")
        all_results_df.to_csv(all_results_path, index=False)
        logging.info(f"\nSaved all individual results to: {all_results_path}")
        
        # Save threshold-averaged results
        summary_df = pd.DataFrame(threshold_summaries)
        dim_total = len(individual_results) #total matrices
        dim_skipped = len([r for r in individual_results if r['skipped']]) #skipped matrices
        dim_processed = dim_total - dim_skipped
        summary_path = os.path.join(stats_dir, f"{dim}_threshold_summary_mahalanobis.csv")
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Saved threshold summary to: {summary_path}")
        logging.info(f"Total matrices: {dim_total}")
        logging.info(f"Skipped matrices: {dim_skipped}")
        logging.info(f"Processed matrices: {dim_processed}")
        total_dim_processed.append(dim_processed)
        
    # Final summary
    total_time = time.time() - overall_start
    logging.info("\n" + "=" * 60)
    logging.info("All processing completed!")
    logging.info(f"Total time: {total_time / 60:.2f} minutes")
    logging.info(f"Processed: {total_dim_processed} total matrix-threshold combinations")
    logging.info(f"Matrix output directory: {output_dir}")
    logging.info(f"Stats output directory: {stats_dir}")
    logging.info("=" * 60)