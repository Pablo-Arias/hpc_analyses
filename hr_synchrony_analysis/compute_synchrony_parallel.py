## sbatch --account project0028 compute_synchrony_parallel_job.sh

import os
# Force all background C/Rust engines to use EXACTLY 1 thread per Joblib worker
# NOTE: POLARS_MAX_THREADS is intentionally removed so the main process can use all SLURM cores!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Imports
import polars as pl
import polars.selectors as cs
pl.Config.set_fmt_str_lengths(100)
import numpy as np
import matplotlib.pylab as plt
import pickle
from scipy import signal
import random
from joblib import Parallel, delayed
import csv

# ---------------------------------------------------------
# WORKER FUNCTION (Pure Python & Heavy Math Only)
# ---------------------------------------------------------
def compute_synchrony_worker(task):
    target_folder = task["target_folder"]
    base_name = task["base_name"]
    lags_file = f"{target_folder}/{base_name}_df_lags.csv"
    avg_corr_file = f"{target_folder}/{base_name}_df_avg_corrs.csv"
    lock_file = f"{target_folder}/{base_name}.lock"
    
    # Atomic Lock Check
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return None # Another worker claimed it
        
    with open(lock_file, 'w') as f:
        f.write("computing")
    
    try:
        avg_corr_rows = []
        df_lags_rows = []
        
        for rep_data in task["repetitions_data"]:
            repetition_nb = rep_data["repetition_nb"]
            
            for feat_data in rep_data["features_data"]:
                source_feature = feat_data["source_feature"]
                target_feature = feat_data["target_feature"]
                X = feat_data["X"]
                y = feat_data["y"]
                
                start = 0
                window = 0
                while start + task["window_length"] < len(X):
                    sub_X  = X[start:start+task["window_length"]]
                    sub_Y =  y[start:start+task["window_length"]]

                    has_infs = np.isinf(sub_X).any() or np.isinf(sub_Y).any() 
                    has_nans = np.isnan(sub_X).any() or np.isnan(sub_Y).any()
                    if has_infs or has_nans:
                        start = start + task["overlap"]
                        window += 1
                        continue
    
                    if len(sub_X) < task["lag"]:
                        start = start + task["overlap"]
                        window += 1
                        continue
    
                    # Compute cross-correlation
                    std_X = np.nanstd(sub_X)
                    std_Y = np.nanstd(sub_Y)
                    if std_X == 0 or std_Y == 0:
                        start = start + task["overlap"]
                        window += 1
                        continue
                    
                    corr_X = (sub_X - np.nanmean(sub_X)) / np.nanstd(sub_X) 
                    corr_y = (sub_Y - np.nanmean(sub_Y)) / np.nanstd(sub_Y) 
                    corr = signal.correlate(corr_X, corr_y, mode='same') 
                    corr = corr / len(sub_X) 
    
                    # Select only cross correlation between +/- lag
                    selected_corr = corr[round(len(corr)/2 - task["lag"]): round(len(corr)/2 + task["lag"])]
    
                    has_infs = np.isinf(corr_X).any() or np.isinf(corr_y).any() 
                    has_nans = np.isnan(corr_X).any() or np.isnan(corr_y).any()
                    if has_infs or has_nans:
                        start = start + task["overlap"]
                        window += 1
                        continue 
    
                    # Compute pearson correlation and max cross correlation coeff
                    max_corr = np.arctanh(np.nanmax(selected_corr))
    
                    # Fisher Z Transformation
                    Z_scores = np.arctanh(np.clip(selected_corr, -0.9999, 0.9999))
                    mean_Z = np.nanmean(Z_scores) 
                    mean_corr = np.tanh(mean_Z) 
    
                    # Save results to lists as dictionaries (Zero DataFrame overhead)
                    avg_corr_rows.append({
                        "source_feature": source_feature,
                        "target_feature": target_feature,
                        "other_id": task["original_other_id"],
                        "user_id": task["user_id"],
                        "participant_manipulated": bool(task["participant_manipulated"]),
                        "other_manipulated": bool(task["other_manipulated"]),
                        "dyad": task["dyad"],
                        "sid": task["sid"],
                        "max_corr": float(max_corr),
                        "mean_corr": float(mean_corr),
                        "window": int(window),
                        "start": int(start),
                        "start_time": int(start),
                        "real_dyad": bool(task["real_dyad"]),
                        "repetition_nb": int(repetition_nb),
                        "analysis": task["analysis"]
                    })     
    
                    lag_range = range(-task["lag"], task["lag"])
                    for i, lag_val in enumerate(lag_range):
                        df_lags_rows.append({
                            "lag": int(lag_val),
                            "corr": float(selected_corr[i]),
                            "dyad": task["dyad"],
                            "source_feature": source_feature,
                            "target_feature": target_feature,
                            "participant_manipulated": bool(task["participant_manipulated"]),
                            "other_manipulated": bool(task["other_manipulated"]),
                            "other_id": task["original_other_id"],
                            "sid": task["sid"],
                            "user_id": task["user_id"],
                            "window": int(window),
                            "start": int(start),
                            "start_time": int(start),
                            "real_dyad": bool(task["real_dyad"]),
                            "repetition_nb": int(repetition_nb),
                            "analysis": task["analysis"]
                        })

                    if task["save_plot"]:                
                        plt.figure(figsize=(12,5))
                        plt.plot(sub_X)
                        plt.plot(sub_Y)
                        os.makedirs("plots/hr/", exist_ok=True)
                        plt.savefig(f"plots/hr/{task['sid']}{task['dyad']}{task['user_id']}{task['participant_manipulated']}.pdf")          
                        plt.close()
    
                    start = start + task["overlap"]
                    window += 1
                
        # Fast Pure-Python CSV writing
        if avg_corr_rows:
            with open(avg_corr_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=avg_corr_rows[0].keys())
                writer.writeheader()
                writer.writerows(avg_corr_rows)

        if df_lags_rows:
            with open(lags_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=df_lags_rows[0].keys())
                writer.writeheader()
                writer.writerows(df_lags_rows)
            
            print(f"[SUCCESS!] Saved CSVs: {task['sid']} {task['dyad']} Real: {task['real_dyad']}", flush=True)
        
    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)
    
    return None

# ---------------------------------------------------------
# DATA GENERATOR (Lazy Evaluation with Fast Dictionary Lookup)
# ---------------------------------------------------------
def generate_tasks(ts_df, ts_dict, target_folder):
    """Yields tasks one by one to prevent RAM ballooning."""
    for real_dyad in [True, False]:
        for other_manipulated in other_manipulations:
            for (sid, participant_manipulated, dyad, user_id, analysis), group_df in ts_df.group_by(["sid", "manipulated", "dyad", "user_id", "analysis"]):
                
                base_name = f"{sid}_{dyad}_{user_id}_{participant_manipulated}_{other_manipulated}_{analysis}_real_{real_dyad}"
                lags_file = f"{target_folder}/{base_name}_df_lags.csv"
                skip_file = f"{target_folder}/{base_name}.skipped"
                lock_file = f"{target_folder}/{base_name}.lock"
                
                # Fast exit: Skip if already computed, skipped, or locked
                if os.path.exists(lags_file) or os.path.exists(skip_file) or os.path.exists(lock_file):
                    continue

                # Check correct interactions
                found = any(sid == inter[0] and dyad == inter[1] for inter in correct_interactions)
                if not found:
                    continue

                # CRITICAL FIX: Sort chronologically before forward filling to avoid data corruption!
                participant_df = group_df.sort("time").with_columns(pl.all().forward_fill(limit=fill_limit)).with_columns(pl.all().backward_fill(limit=fill_limit))
                
                repetitions_data = []
                skip_task = False

                for repetition_nb in range(nb_of_surrogate_perms):
                    if repetition_nb >= 1 and real_dyad:
                        continue

                    other_id_series = participant_df.get_column("other_id")
                    if other_id_series.is_null().all():
                        with open(skip_file, 'w') as f: f.write("null_other_id")
                        skip_task = True
                        break

                    original_other_id = other_id_series.drop_nulls().unique()[0]

                    # Find Partner using the O(1) Dictionary Lookup
                    partner_df = pl.DataFrame()
                    
                    if real_dyad:
                        partner_other_id = original_other_id
                        lookup_key = (sid, other_manipulated, dyad, analysis)
                        
                        if lookup_key in ts_dict:
                            partner_df = ts_dict[lookup_key].filter(pl.col("user_id") == partner_other_id)
                    else:
                        filtered_list = [tup for tup in correct_interactions if tup[0] != sid]
                        attempts = 0
                        while len(partner_df) == 0 and attempts < max_attempts_for_surrogate:
                            partner_sid, partner_dyad = random.choice(filtered_list) 
                            lookup_key = (partner_sid, other_manipulated, partner_dyad, analysis)
                            
                            if lookup_key in ts_dict:
                                surrogate_session_df = ts_dict[lookup_key]
                                if len(surrogate_session_df) > 0:
                                    available_user_ids = surrogate_session_df.get_column("user_id").drop_nulls().unique().to_list()
                                    if len(available_user_ids) > 0:
                                        partner_other_id = random.choice(available_user_ids)
                                        partner_df = surrogate_session_df.filter(pl.col("user_id") == partner_other_id)
                            attempts += 1
                        
                        if len(partner_df) == 0:
                            print(f"[{sid}] Skipping repetition: No surrogate found after 15 attempts.", flush=True)
                            continue

                    if len(partner_df) == 0:
                        with open(skip_file, 'w') as f: f.write("missing_partner_data")     
                        skip_task = True
                        break

                    # Partner Data must also be sorted chronologically (Already sorted in dict, but safe to keep logic consistent)
                    partner_df = partner_df.sort("time").with_columns(pl.all().forward_fill(limit=fill_limit)).with_columns(pl.all().backward_fill(limit=fill_limit))

                    features_data = []
                    for source_feature in source_features:
                        for target_feature in target_features:
                            #With an inner join
                            merged_df = participant_df.join(partner_df, on="time", how="inner").sort("time")
                            if len(merged_df) == 0:
                                continue
                            X = merged_df.get_column(f"{source_feature}").to_numpy().copy()
                            y = merged_df.get_column(f"{target_feature}_right").to_numpy().copy()

                            # OLD SCRIPT LOGIC without inner join: Blind array extraction, no joining!
                            #X = participant_df.get_column(source_feature).to_numpy().copy()
                            #y = partner_df.get_column(target_feature).to_numpy().copy()

                            if len(y) == 0 or len(X) == 0 or len(y) < nb_min_samples or len(X) < nb_min_samples:
                                with open(skip_file, 'w') as f: f.write("data_too_short")   
                                skip_task = True
                                break            
                            
                            # Keep only the same number of samples for both (Row-index alignment)
                            max_len = np.min([len(X), len(y)])
                            X = X[0:max_len]
                            y = y[0:max_len]

                            features_data.append({
                                "source_feature": source_feature,
                                "target_feature": target_feature,
                                "X": X, 
                                "y": y
                            })
                            
                        if skip_task: break
                    if skip_task: break

                    if not skip_task and len(features_data) > 0:
                        repetitions_data.append({
                            "repetition_nb": repetition_nb,
                            "features_data": features_data
                        })

                if skip_task or len(repetitions_data) == 0:
                    continue

                yield {
                    "base_name": base_name,
                    "target_folder": target_folder,
                    "sid": sid,
                    "dyad": dyad,
                    "user_id": user_id,
                    "original_other_id": original_other_id,
                    "participant_manipulated": participant_manipulated,
                    "other_manipulated": other_manipulated,
                    "analysis": analysis,
                    "real_dyad": real_dyad,
                    "repetitions_data": repetitions_data,
                    "overlap": overlap,
                    "window_length": window_length,
                    "save_plot": save_plot,
                    "lag": lag
                }


# ---------------------------------------------------------
# Global Variables
# ---------------------------------------------------------
# Define global features
other_manipulations = [True, False]
source_features = ["bpmES"]
target_features = ["bpmES"]

## ---- Good parameters for MI estimation
lag = 20 # In seconds before and after
nb_min_samples = 130 # in seconds. 
window_length  = 45 # seconds
overlap = 1
save_plot = False
nb_of_surrogate_perms = 2 # nb of surrogate dyads to compute per dyad
max_attempts_for_surrogate = 15 # Nb of attempts to try to find a surrogate alternative
fill_limit = 30

# Load correct interactions globally
with open("data/correct_interactions.pickle", "rb") as handle:
    correct_interactions = pickle.load(handle)


# ---------------------------------------------------------
# MAIN LAUNCHER
# ---------------------------------------------------------
def main():
    print("Loading Data into Main Process...", flush=True)
    #all_data_df = pl.read_csv("data/hr_computed/hr.csv")
    #all_data_df = pl.read_csv("data/hr_computed/hr_preprocessed.csv")
    #all_data_df = pl.read_csv("data/hr_computed/hr_ws_6_with_filtering_June_2026_all_sessions_no_filtering_preprocessed.csv")
    #all_data_df = pl.read_csv("data/hr_computed/hr_ws_6_with_filtering_June_2026_preprocessed.csv")
    all_data_df = pl.read_csv("data/hr_computed/hr_ws_6_with_filtering_June_2026_clustering_preprocessed.csv")

    ts_df = all_data_df.group_by([
        "sid", "dyad", "user_id", "other_id", 
        "participant_condition", "other_condition", 
        "file_name", "manipulated", "analysis", "time"
    ]).agg(cs.numeric().mean()).sort(["sid", "dyad", "analysis", "user_id", "time"])
    
    # Pre-build the O(1) Lookup Dictionary for the generator
    print("Building fast lookup dictionary...", flush=True)
    ts_dict = {}
    for (sid, manipulated, dyad, analysis), group in ts_df.group_by(["sid", "manipulated", "dyad", "analysis"]):
        # Sort chunks immediately as they enter the dictionary
        ts_dict[(sid, manipulated, dyad, analysis)] = group.sort("time")

    target_folder = f"data/filtered_data_clustering_synchrony_new_parallel_PREPRO_repetition_{nb_of_surrogate_perms}_lag_{lag}_overlap_{overlap}_wl_{window_length}_fill_limit{fill_limit}"
    os.makedirs(target_folder, exist_ok=True)

    num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 32))
    print(f"Data Indexed. Streaming tasks lazily to {num_cores} cores...", flush=True)

    # Instantiate the generator
    task_generator = generate_tasks(ts_df, ts_dict, target_folder)

    # Launch Parallel
    Parallel(n_jobs=num_cores, backend="loky", pre_dispatch='2 * n_jobs', max_nbytes=None)(
        delayed(compute_synchrony_worker)(task) for task in task_generator
    )
    
    print("All parallel jobs completed successfully.", flush=True)

if __name__ == "__main__":
    main()