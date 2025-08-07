## sbatch --account project0028 all_synchrony_job.sh

# Imports
import polars as pl
pl.Config.set_fmt_str_lengths(100)
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pylab as plt
from conversions import get_file_without_path
import glob
import os
import pickle

from sklearn.feature_selection import mutual_info_regression
from scipy import signal
from scipy import stats
import random
from scipy.stats import spearmanr
import pingouin
import numpy as np
from scipy.special import erfcinv
from joblib import Parallel, delayed
import uuid

#For time series normalisation
def copnorm(x):
    """
    COPNORM Copula normalization
    cx = copnorm(x) returns standard normal samples with the same empirical
    CDF value as the input. Operates along the first axis.
    Equivalent to cx = norminv(ctransform(x))
    """
    # Rank the data
    ranks = np.argsort(np.argsort(x, axis=0), axis=0)
    
    # Normalize the ranks
    n = x.shape[0]
    x_normalized = (ranks + 1) / (n + 1)
    
    # Transform to standard normal
    x_standard_normal = -np.sqrt(2) * erfcinv(2 * x_normalized)
    
    return x_standard_normal

#Function to compute synchrony
import random; 
def compute_synchrony(sid, participant_manipulated, other_manipulated, dyad, user_id, dyad_df, ts_df, overlap, window_length, save_plot, nb_min_samples, lag, real_dyad, analysis):   
    #Check if interaction was recorded correctly (4 recordings in total).
    found=False
    for interaction in correct_interactions:
        if sid == interaction[0] and dyad == interaction[1]:
            found = True
            break
    if not found:
        #print("Skipping not correct recording - dyad : " + dyad + " sid : " + sid)
        return None, None

    #Prepare participants' time series
    participant_df = dyad_df.loc[dyad_df["user_id"] == user_id].copy()
    participant_df = participant_df.ffill(axis="rows")
    participant_df = participant_df.bfill(axis="rows")
    
    #Begin main loop
    for repetition_nb in range(nb_of_surrogate_perms):

        #If real dyad value has already been computed (repetition_nb=1) continue
        if repetition_nb>=1 and real_dyad:
            continue

        #Initialize DataFrames
        mi_df   = pl.DataFrame(schema=mi_df_schema)
        df_lags = pl.DataFrame(schema=df_lags_schema)
        
        

        #Get other id and check that other_id is not null, if it is, return.
        other_id = participant_df["other_id"]
        nulls = np.unique(other_id.isnull())
        if len(nulls)==1 and nulls[0]:
            return None, None

        #Select other_id
        other_id = np.unique(other_id)[0]  

        #If dyad is real, then select other id as the actual partner
        if real_dyad:
            partner_dyad = dyad
            partner_other_id= other_id

        #uf dyad is not real, find surrogate dyad
        else :
            #Choose a random participant
            filtered_list = [tup for tup in correct_interactions if tup[0] != user_id]
            filtered_list = [tup for tup in filtered_list if tup[0] != other_id]
            filtered_list = [tup for tup in filtered_list if tup[1] != user_id]
            filtered_list = [tup for tup in filtered_list if tup[1] != other_id]
            #Change other_id
            partner_sid, partner_dyad  = random.choice(filtered_list)
            partner_other_id  =  partner_sid + random.choice((partner_dyad[0:2], partner_dyad[2:4]))

        #Select time series of the partner
        partner_df = ts_df.loc[ts_df["user_id"] == partner_other_id].copy()
        partner_df = partner_df.loc[partner_df["dyad"] == partner_dyad]    
        partner_df = partner_df.loc[partner_df["manipulated"] == other_manipulated]        
        partner_df = partner_df.ffill(axis="rows")
        partner_df = partner_df.bfill(axis="rows")

        if not len(partner_df)>0:
            return None, None
            
        for source_feature in source_features:
            for target_feature in target_features:    
                #preapre participant time series
                X = participant_df[source_feature].values
    
                #preapre target time series
                y = partner_df[target_feature].values
    
                if len(y)==0 or len(X) ==0 or len(y)< nb_min_samples or len(X)< nb_min_samples:
                    continue            
                
                #Keep only the same number of samples for both
                max = np.min([len(X), len(y)])
                X = X[0:max]
                y = y[0:max]
    
                start = 0
                window = 0
                while start + window_length< len(X):
                    
                    sub_X  = X[start:start+window_length]
                    sub_Y =  y[start:start+window_length]
    
                    has_infs = np.isinf(sub_Y).any() or np.isinf(sub_Y).any() 
                    has_nans = np.isnan(sub_X).any() or np.isnan(sub_X).any()
                    if has_infs:
                        print("careful, Y or X has infs")
                    if has_nans:
                        print("careful, Y or X has nans")
    
                    if len(sub_X) < lag:
                        start = start+overlap
                        window += 1
                        continue
    
                    #cross-correlation
                    std_X = np.nanstd(sub_X)
                    std_Y = np.nanstd(sub_Y)
                    if std_X == 0 or std_Y == 0:
                        print(f"Zero standard deviation found")
                        #Update window information
                        start = start+overlap
                        window += 1
                        continue
                    
                    corr_X = [(val-np.nanmean(sub_X))/np.nanstd(sub_X) for val in sub_X]
                    corr_y = [(val-np.nanmean(sub_Y))/np.nanstd(sub_Y) for val in sub_Y]
                    corr = signal.correlate(corr_X, corr_y, mode='same') 
                    corr = corr  / len(sub_X)
    
                    #Select only cross correaltion between +/- lag
                    selected_corr = corr[round(len(corr)/2 - lag): round(len(corr)/2 + lag)]
    
                    #Check selected corrs:
                    has_infs = np.isinf(corr_X).any() or np.isinf(corr_y).any() 
                    has_nans = np.isnan(corr_X).any() or np.isnan(corr_y).any()
                    if has_infs:
                        print("careful, Corr has infs")
                    if has_nans:
                        print("careful, Corr has nans")
    
                    #Compute pearson correlation and max cross correlation coeff
                    pears_r, p = stats.pearsonr(corr_X, corr_y)
                    max_corr = np.arctanh(np.nanmax(selected_corr))
    
                    # Fisher Z Transformation
                    #Clamp values in array (don't allow 1 or -1 to be in the arrays)
                    Z_scores = np.arctanh(selected_corr)
                    has_infs = np.isinf(Z_scores).any() or np.isinf(Z_scores).any()
                    has_nans = np.isnan(corr_X).any() or np.isnan(corr_y).any()
                    if has_infs:
                        #print("careful, Z_scores has infs")
                        start = start+overlap
                        window += 1
                        continue
                    if has_nans:
                        #print("careful, Z_scores has nans")
                        start = start+overlap
                        window += 1
                        continue
                    
                    mean_Z = np.nanmean(Z_scores) # Compute the mean of Z scores
                    mean_corr = np.tanh(mean_Z) # Inverse Fisher Z Transformation to get the average correlation
    
                    #Compute MI
                    sub_X = sub_X.reshape(-1,1)      
                    mi = mutual_info_regression(X=copnorm(sub_X), y=copnorm(sub_Y)
                                    , discrete_features = 'auto'
                                    , n_neighbors = 15
                                    , copy = True
                                    , random_state=None
                                )
    
                    if save_plot:                
                        plt.figure(figsize=(12,5))
                        plt.plot(sub_X)
                        plt.plot(sub_Y)
                        plt.savefig("plots/hr/"+str(sid)+ str(dyad)+ str(user_id)+str(participant_manipulated)+".pdf")
    
    
                    #Save results to a DataFrame
                    aux_df = pl.DataFrame({
                        "source_feature": [source_feature],
                        "target_feature": [target_feature],
                        "other_id": [other_id],
                        "user_id": [user_id],
                        "participant_manipulated": [bool(participant_manipulated)],
                        "other_manipulated": [bool(other_manipulated)],
                        "dyad": [dyad],
                        "sid": [sid],
                        "mi": mi,
                        "max_corr": [max_corr],
                        "mean_corr": [mean_corr],
                        "window": [window],
                        "start": [start],
                        "start_time": [start],
                        "real_dyad": [bool(real_dyad)],
                        "repetition_nb": [repetition_nb],
                        "analysis": [analysis]
                    })
                    
    
                    aux_lags_df = pl.DataFrame({
                        "lag": pl.Series("lag", range(2 * lag)) - lag,
                        "corr": selected_corr,
                        "dyad": [dyad] * (2 * lag),
                        "source_feature": [source_feature] * (2 * lag),
                        "target_feature": [target_feature] * (2 * lag),
                        "participant_manipulated": [bool(participant_manipulated)] * (2 * lag),
                        "other_manipulated": [bool(other_manipulated)] * (2 * lag),
                        "other_id": [other_id] * (2 * lag),
                        "sid": [sid] * (2 * lag),
                        "user_id": [user_id] * (2 * lag),
                        "window": [window] * (2 * lag),
                        "start": [start] * (2 * lag),
                        "start_time": [start] * (2 * lag),
                        "real_dyad": [bool(real_dyad)] * (2 * lag),
                        "repetition_nb": [repetition_nb] * (2 * lag),
                        "analysis": [analysis] * (2 * lag)
                    })
                    aux_df = aux_df.with_columns([pl.col(column).cast(dtype) for column, dtype in mi_df_schema.items()])
                    aux_lags_df = aux_lags_df.with_columns([pl.col(column).cast(dtype) for column, dtype in df_lags_schema.items()])

                    df_lags = pl.concat([df_lags, aux_lags_df])
                    mi_df = pl.concat([mi_df, aux_df])

    
                    #Update window information
                    start = start+overlap
                    window += 1
    
    
        unique_id = str(uuid.uuid4())
        df_lags.write_csv("data/synchrony_for_each_ppg_method_repetition_2_lag_15_overlap_1_wl_30/" + unique_id +"_df_lags.csv")
        mi_df.write_csv("data/synchrony_for_each_ppg_method_repetition_2_lag_15_overlap_1_wl_30/"+ unique_id +"_mi_df.csv")


# Initialize the dataframes with the correct schema if they are not initialized
df_lags_schema = {
    "lag": pl.Int64,
    "corr": pl.Float64,
    "dyad": pl.Utf8,
    "source_feature": pl.Utf8,
    "target_feature": pl.Utf8,
    "participant_manipulated": pl.Boolean,
    "other_manipulated": pl.Boolean,
    "other_id": pl.Utf8,
    "sid": pl.Utf8,
    "user_id": pl.Utf8,
    "window": pl.Int64,
    "start": pl.Int64,
    "start_time": pl.Int64,
    "real_dyad": pl.Utf8,
    "repetition_nb": pl.Int64,
    "analysis": pl.Utf8
}

mi_df_schema = {
    "source_feature": pl.Utf8,
    "target_feature": pl.Utf8,
    "other_id": pl.Utf8,
    "user_id": pl.Utf8,
    "participant_manipulated": pl.Boolean,
    "other_manipulated": pl.Boolean,
    "dyad": pl.Utf8,
    "sid": pl.Utf8,
    "mi": pl.Float64,
    "max_corr": pl.Float64,
    "mean_corr": pl.Float64,
    "window": pl.Int64,
    "start": pl.Int64,
    "start_time": pl.Int64,
    "real_dyad": pl.Utf8,
    "repetition_nb": pl.Int64,
    "analysis": pl.Utf8
}

#define global features
other_manipulations = [True, False]
source_features = ["bpmES"]
target_features = ["bpmES"]

behavior_df = pl.read_csv("data/behavior/all_data_df.csv")
behavior_df = behavior_df.with_columns(
    pl.col("sid").str.replace_all(":", "").alias("sid"),
    pl.col("user_id").str.replace_all(":", "").alias("user_id"),
    pl.col("other_id").str.replace_all(":", "").alias("other_id")
)

all_data_df = pd.read_csv("data/hr_computed/hr.csv")

ts_df = all_data_df.groupby(["sid", "dyad", "user_id", "other_id", "participant_condition", "other_condition", "file_name", "manipulated", "analysis", "time"]).mean(numeric_only=True).reset_index()

#Check which interactions were recorded correctly and can be used for this analysis:
correct_interactions   = []
incorrect_interactions = []
for interaction in glob.glob("../preproc/prolific/*/trimed/*"):
    recordings = glob.glob(interaction + "/*.mp4")
    if len(recordings)==4:
        sid = interaction.split("/")[3]
        dyad = interaction.split("/")[5][2:]

        correct_interactions.append((sid, dyad))
    else:
        incorrect_interactions.append((sid, dyad))

print("Found "+ str(len(correct_interactions)) + " correct interactions")
print("Found "+ str(len(incorrect_interactions)) + " incorrect interactions")



## ---- Good parameters for MI estimation
#Both manipulated with the same value
lag = 15 # In seconds before and after
nb_min_samples = 150 #in seconds. 
window_length  = 30 #seconds
overlap = 1
save_plot = False
nb_of_surrogate_perms = 2 #nb of surrogate dyads to compute per dyad

# Create a process group function for parallelisation
def process_group(sid, participant_manipulated, other_manipulated, dyad, user_id, dyad_df, ts_df, overlap, window_length, save_plot, nb_min_samples, lag, real_dyad, analysis):
    return compute_synchrony(sid, participant_manipulated, other_manipulated, dyad, user_id, dyad_df, ts_df, overlap, window_length, save_plot, nb_min_samples, lag, real_dyad, analysis)


def main():
    tasks = []

    #Appending all the tasks
    for real_dyad in [True, False]:
        for other_manipulated in other_manipulations:
            for (sid, participant_manipulated, dyad, user_id, analysis), dyad_df in ts_df.groupby(["sid", "manipulated", "dyad", "user_id", "analysis"]):
                tasks.append((sid, participant_manipulated, other_manipulated, dyad, user_id, dyad_df, ts_df, overlap, window_length, save_plot, nb_min_samples, lag, real_dyad, analysis))

    # Using joblib.Parallel and delayed to handle multiple arguments
    print("Performing jobs in parallel - start")
    results = Parallel(n_jobs=-1)(delayed(process_group)(
        sid, participant_manipulated, other_manipulated, dyad, user_id, dyad_df, ts_df, overlap, window_length, save_plot, nb_min_samples, lag, real_dyad, analysis
    ) for sid, participant_manipulated, other_manipulated, dyad, user_id, dyad_df, ts_df, overlap, window_length, save_plot, nb_min_samples, lag, real_dyad, analysis in tasks)
    print("Performing jobs in parallel - stop")
    
    aux_lags_list = []
    aux_mi_df_list = []
    
    # Collecting results
    for aux_lags, aux_mi_df in results:
        if aux_lags is not None: 
            aux_lags_list.append(aux_lags)
        if aux_mi_df is not None:
            aux_mi_df_list.append(aux_mi_df)

    print("Concatenating results - start")
    # Concatenate all collected DataFrames at once
    df_lags = pl.concat( aux_lags_list, how="vertical")
    mi_df = pl.concat(aux_mi_df_list, how="vertical")
    print("Concatenating results - stop")
    
    return df_lags.to_pandas(), mi_df.to_pandas()

if __name__ == "__main__":
    df_lags, mi_df = main()
