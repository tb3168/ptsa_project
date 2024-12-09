#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:23:40 2024

@author: tanvibansal
"""
import pandas as pd
import numpy as np
import os 
os.chdir("/Users/tanvibansal/Documents/GitHub/ptsa_project/")
import fake_data
from scipy.optimize import fmin, minimize


# =============================================================================
# read files in 
# =============================================================================
event_df = pd.read_pickle("/Users/tanvibansal/Documents/GitHub/flood-filters/flood_filters/experimental_pipeline/phase 2/2.3/event_df_tidy")
flood_df = event_df.loc[event_df.label == "flood"]

# =============================================================================
# zero pad true  data before smoothing 
# =============================================================================
signal = flood_df.signal

def zero_pad_signal(signal,N):
    time = signal["time"]
    depth = signal["depth"]
    #create the timestamps for the front padding and back padding (not adjusting for negative times yet)
    t_pad_buffer = np.arange(1,(N+1))*60
    t_pad_before = np.array([time[0] - i for i in t_pad_buffer[::-1]])
    t_pad_after = np.array([time[-1] + i for i in t_pad_buffer])
    
    #concatenate the front padding timetamps, existing, and the back padding timestamps into an array. then shift them so that the first time stamp is 0
    t_pad = np.concatenate([t_pad_before, time, t_pad_after])
    t_pad_adj = time[0] - t_pad[0] 
    t_pad = t_pad + t_pad_adj
    
    #create the 0 padding before to concat onto the front and back of the depth array 
    d_pad_buffer = np.zeros(N)
    d_pad = np.concatenate([d_pad_buffer, depth, d_pad_buffer])
    
    return {"time":t_pad, "depth": d_pad}

signal_padded = signal.apply(lambda x: zero_pad_signal(x, 5))

# =============================================================================
# generate simulated data
# =============================================================================

# z-score remove outliers
def remove_outliers(x,thresh):
    t = x["time"]
    d = x["depth"]
    z_scores = (d - d.mean())/d.std()
    mask = (np.abs(z_scores) > thresh)
    if mask.sum() > 0:
        #t = t[~mask]
        d[mask] = np.mean(d)
        x={"time":t,"depth":d}
    return x
signal_outliers_rm = signal_padded.apply(lambda x: remove_outliers(x,2))

# smooth 
def smooth(signal):
    signal = pd.DataFrame(signal).set_index("time")
    std = np.std((signal["depth"] - signal["depth"].mean())/signal["depth"].max())*signal["depth"].max()
    smooth = signal.rolling(window=3, win_type="gaussian",center=True).mean(std=std).fillna(0).reset_index()
   
    return {"time":smooth.time.values, "depth": smooth.depth.values}
signal_smooth = signal_outliers_rm.apply(lambda x: smooth(x))

# generate drainage profile w/ params

def generate_flood_profile(signal, params):
    a = params[0]
    c = params[1]
    time = signal["time"]
    depth = signal["depth"]
    
    duration = round(time[-1]) + 1
    peak = depth.max()
    
    depth_sim = fake_data.flood(duration=duration, a=a, c=c, peak=peak, power=1., noise=0.)
    depth_sim = depth_sim[time.round().astype("int")]
    
    return {"time": time, "depth": depth_sim}

#quantify generated flood error 
def flood_profile_error(signal, params):
    depth = signal["depth"]
    peak = depth.max()
    
    signal_sim = generate_flood_profile(signal, params)
    depth_sim = signal_sim["depth"]
    
    #estimate rmse between fit profile and measured
    rmse = np.sqrt(np.sum((depth - depth_sim)**2)/(len(depth)*peak))

    return rmse

def optimize_flood_profile_error(params, signals):
    return signals.apply(lambda x: flood_profile_error(x, params)).mean() 

gamma_train = minimize(optimize_flood_profile_error, np.array([10,10]), args=(signal_smooth,),method = "Nelder-Mead",bounds=((0.1,None),(0.1, 1e10)))

#generate all the profiles with the optimized parameters 
signal_simulated = signal_smooth.apply(lambda x: generate_flood_profile(x,gamma_train.x))

# =============================================================================
# zero pad true and simulated data
# =============================================================================

signal_padded = signal_padded.apply(lambda x: zero_pad_signal(x, 5))
signal_sim_padded = signal_simulated.apply(lambda x: zero_pad_signal(x, 5))

# =============================================================================
# format df and write to disk 
# =============================================================================

flood_df_out = flood_df[["deployment_id","label","signal"]]
flood_df_out["signal_padded"] = signal_padded
flood_df_out["signal_sim"] = signal_simulated 
flood_df_out["signal_sim_padded"] = signal_sim_padded

flood_df_out.to_pickle("flood_df")
