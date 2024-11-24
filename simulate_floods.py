#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:22:46 2024

@author: tanvibansal
"""

import fake_data
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import ast
import math
from plotnine import *

def wide_to_long(df,cols_to_keep, cols_to_expand):
    df_exp = pd.DataFrame(df[cols_to_expand].to_list(),index = df.index)
    df_exp["t_pct"] = df_exp.apply(lambda x: x["time"]/x["time"].max(), axis=1)
    df_keep = df[cols_to_keep]
    df_out = df_keep.join(df_exp).explode(["time","depth","t_pct"])
    df_out = df_out.astype({"time":"int","depth":"float","t_pct":"float"})
    return df_out 
flood_df_long = wide_to_long(flood_df,["duration"],"signal")
# =============================================================================
# STEP 1: get raw signals for all the true floods
# =============================================================================
event_df = pd.read_pickle('/Users/tanvibansal/Documents/GitHub/ptsa_project/event_df_tidy')
flood_df = event_df.loc[event_df.label == "flood"]
suspect_floods = [4152648,2195521,8612524, 6101443, 7370250, 9364214, 4268458, 4298728, 3606063,
       3323611, 9829405, 3230015, 2860240, 5545019, 1929133, 1429885,
       8159541, 4487869, 1466876, 1343346, 9641026,6678695]

# =============================================================================
# STEP 2: get durations of all floods rounded up to the nearest minute
# =============================================================================
flood_df.loc[:,"duration"] = flood_df.apply(lambda x: math.ceil(x["signal"]["time"][-1]/60),axis=1)
#flood_df = flood_df.drop(suspect_floods)
flood_df = flood_df.loc[flood_df.duration > 4] #drop floods with duration <= 4 because these are all blips and boxes
#flood_df = flood_df.loc[flood_df.apply(lambda x: x["signal"]["depth"].max(),axis=1) > 30] #drop floods with max depth > 30 

# =============================================================================
# STEP 3: smooth floods and cut into rising/falling segments
# =============================================================================
x = flood_df.loc[4066145]
uuid = x.name
signal = pd.DataFrame(x.signal)
std = np.std((signal["depth"] - signal["depth"].mean())/signal["depth"].max())*signal["depth"].max()
smooth = signal.rolling(window=len(signal)//10, win_type="gaussian",center=True).mean(std=std)
ggplot(signal,aes(x="time",y="depth")) + geom_line(alpha=0.6) + geom_line(smooth,aes(x="time",y="depth"),color="skyblue",alpha=0.9)

# =============================================================================
# STEP 3: remove outliers
# =============================================================================
def remove_outliers(x,thresh):
    t = x.signal["time"]
    d = x.signal["depth"]
    z_scores = (d - d.mean())/d.std()
    mask = (np.abs(z_scores) > thresh)
    if mask.sum() > 0:
        t = t[~mask]
        d = d[~mask]
        x.signal={"time":t,"depth":d}
    return x.signal
flood_df["signal_smt"] = flood_df.apply(lambda x: remove_outliers(x,2),axis=1)

# =============================================================================
# STEP 3: generate simulated floods at all the durations and append onto dataframe
# =============================================================================
def generate_simulated_flood(x,signal_name):
    duration = x.duration
    peak = x[signal_name]["depth"].max()
    depth_sim = fake_data.flood(duration=duration, a=0.7, c=1.7, peak=peak, power=1., noise=0.)
    t_sim = np.arange(duration)*60
    return {"time":t_sim,"depth":depth_sim}

flood_df["signal_sim"] = flood_df.apply(lambda x: generate_simulated_flood(x,"signal_smt"),axis=1)
#flood_df["signal_sim"] = 
# =============================================================================
# STEP 4: get time step comparison of measured depth and simulated depth
# =============================================================================

flood_measure_df = pd.DataFrame(flood_df["signal_smt"].to_list(),index=flood_df.index)
flood_measure_df['time'] = flood_measure_df.time.apply(lambda x: np.round(x/60))
flood_measure_df = flood_measure_df.explode(["time","depth"]).astype({"time":"int","depth":"float"}).reset_index().set_index(["uuid","time"])

flood_simulate_df = pd.DataFrame(flood_df["signal_sim"].to_list(),index=flood_df.index)
flood_simulate_df['time'] = flood_simulate_df.time.apply(lambda x: np.round(x/60))
flood_simulate_df = flood_simulate_df.explode(["time","depth"]).astype({"time":"int","depth":"float"}).reset_index().set_index(["uuid","time"])

noise_est_df = flood_simulate_df.join(flood_measure_df,how="left",lsuffix=".sim",rsuffix=".meas")

# =============================================================================
# STEP 5: estimate noise between measured and simulated for each event
# =============================================================================
measurement_error = noise_est_df.dropna()
measurement_error.loc[:,"error"] = (measurement_error.loc[:,"depth.sim"] - measurement_error.loc[:,"depth.meas"]).rename("error")

event_error = measurement_error.reset_index().groupby("uuid").agg(rmse = ('error', lambda x: (x**2).mean())).sort_values(by="rmse",ascending=False)
event_error["pct_rank"] = event_error["rmse"].rank(pct=True)

# =============================================================================
# STEP 6: plot 
# =============================================================================
plt.hist(event_error["rmse"],bins=30)
plt.xlabel("Mean Event RMSE")
plt.title("Distribution of Flood Event RMSE between Simulated and Measured")

for i in range(len(event_error)):
    e = event_error.iloc[i]
    uuid = e.name
    pct = e.pct_rank 
    rmse = e.rmse
    
    p = ggplot(measurement_error.loc[uuid].reset_index(),aes(x="time",y="depth.meas",group="uuid")) + geom_line()  + \
        geom_line(aes(x = "time",y="depth.sim"),color="grey") + labs(title="Simulated vs Measured Flood Depth\nrmse = %s, pct_rank_rmse = %s"%(round(rmse,2), pct))
        
    ggsave(p,filename="/Users/tanvibansal/Documents/GitHub/ptsa_project/flood_simulation_tuning/%s.png"%(uuid),format="png")
x = flood_df.copy(deep=True).loc[5190612]
d = x["signal"]['depth']
d[np.abs((d - d.mean())/d.std()) > 2]
#explore taking out long runs of same vals from 

x = measurement_error.copy(deep=True).loc[4152648]
def find_runs(x):
    
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
flood_df.apply(lambda x: find_runs(x.signal["depth"]),axis=1)

run_values, run_starts, run_lengths = find_runs(x["depth.meas"])

run_lengths[run_lengths > 5]
run_starts[run_lengths > 5]

ggplot(x.reset_index(),aes(x="time",y="depth.meas")) + geom_line()
