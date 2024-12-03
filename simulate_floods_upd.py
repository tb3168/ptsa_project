#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:22:19 2024

@author: tanvibansal
"""

import os 
os.chdir("/Users/tanvibansal/Documents/GitHub/ptsa_project/")
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import fake_data
from scipy.optimize import fmin, minimize
event_df = pd.read_pickle('/Users/tanvibansal/Documents/GitHub/ptsa_project/event_df_tidy')
flood_df = event_df.loc[event_df.label == "flood"]


flood_simulations = pd.read_pickle('/Users/tanvibansal/Documents/GitHub/ptsa_project/flood_simulations')

plt.plot(x.signal["time"], x.signal["depth"], label="measured")
plt.plot(x.signal_sim["time"], x.signal_sim["depth"], label="simulated")

# =============================================================================
# STEP 1: generate simulated true state using a gamma distribution and optimizing the parameters to minimize error
# =============================================================================

def fit_flood_profile(ev, a, c):
    x = ev.copy(deep=True).dropna()
    duration = round(x.signal['time'][-1]) + 1
    peak = x.signal['depth'].max()
    
    depth_sim = fake_data.flood(duration=duration, a=a, c=c, peak=peak, power=1., noise=0.)
    depth_sim = depth_sim[x.signal['time'].round().astype("int")]
    
    depth_smooth = x.signal["depth"]
    
    #estimate rmse between fit profile and measured
    rmse = np.sqrt(np.sum((depth_smooth - depth_sim)**2)/(len(depth_smooth)*peak))

    return rmse

def apply_fit_flood_profile(params, flood_simulate_df):
    return flood_simulations.apply(lambda x: fit_flood_profile(x,params[0],params[1]),axis=1).sum()

gamma_train = minimize(apply_fit_flood_profile, np.array([10,10]), args=(flood_simulations,),method = "Nelder-Mead",bounds=((0.1,None),(0.1, 1e10)))
total_training_error = apply_fit_flood_profile(gamma_train.x, flood_simulations)

def generate_flood_profile(ev, a, c): 
    x = ev.copy(deep=True).dropna()
    duration = round(x.signal['time'][-1]) + 1
    peak = x.signal['depth'].max()
    
    depth_sim = fake_data.flood(duration=duration, a=a, c=c, peak=peak, power=1., noise=0.)
    depth_sim = depth_sim[x.signal['time'].round().astype("int")]
        
    return {"time": x.signal['time'].round().astype("int"), "depth": depth_sim}

flood_simulations["signal_sim"] = flood_simulations.apply(lambda x: generate_flood_profile(x, gamma_train.x[0], gamma_train.x[1]),axis=1)

#plot
for i in range(100):
    x = flood_simulations.iloc[i]
    plt.subplots()
    plt.plot(x["signal"]["time"],x["signal"]["depth"],label="measured")
    plt.plot(x["signal_sim"]["time"],x["signal_sim"]["depth"],label="simulated")
    
# =============================================================================
# STEP 2: zero pad the true signal 
# =============================================================================
x = flood_simulations.iloc[0]
time = x.signal["time"]
depth = x.signal["depth"] 

N = 5 #number of zero points to add onto each end

def zero_pad_signal(signal,N):
    time = signal["time"]
    depth = signal["depth"]
    #create the timestamps for the front padding and back padding (not adjusting for negative times yet)
    t_pad_buffer = np.arange(1,(N+1))*60
    t_pad_before = np.array([time[0] - i for i in t_pad_buffer[::-1]])
    t_pad_after = np.array([time[-1] + i for i in t_pad_buffer])
    
    #concatenate the front padding timetamps, existing, and the back padding timestamps into an array. then shift them so that the first time stamp is 0
    t_pad = np.concatenate([t_pad_before, time, t_pad_after])
    t_pad_adj = 0 - t_pad[0] 
    t_pad = t_pad + t_pad_adj
    
    #create the 0 padding before to concat onto the front and back of the depth array 
    d_pad_buffer = np.zeros(N)
    d_pad = np.concatenate([d_pad_buffer, depth, d_pad_buffer])
    
    return {"time":t_pad, "depth": d_pad}

flood_simulations["signal_padded"] = flood_simulations.apply(lambda x: zero_pad_signal(x.signal, 5),axis=1)
flood_simulations["signal_sim_padded"] = flood_simulations.apply(lambda x: zero_pad_signal(x.signal_sim, 5),axis=1)

flood_simulated_out = flood_simulations.copy(deep=True)[["deployment_id","label","inflection_t","signal","signal_padded","signal_sim","signal_sim_padded"]]
flood_simulated_out.to_pickle('/Users/tanvibansal/Documents/GitHub/ptsa_project/flood_df')

plt.plot(x["signal"]["time"],x["signal"]["depth"],label="measured")


plt.plot(x["signal_sim"]["time"],x["signal_sim"]["depth"],label="simulated")

# =============================================================================
# 
# =============================================================================
