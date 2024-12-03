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
from scipy.optimize import fmin 

def wide_to_long(df,cols_to_keep, cols_to_expand):
    df_exp = pd.DataFrame(df[cols_to_expand].to_list(),index = df.index)
    df_exp["t_pct"] = df_exp.apply(lambda x: x["time"]/x["time"].max(), axis=1)
    df_keep = df[cols_to_keep]
    df_out = df_keep.join(df_exp).explode(["time","depth","t_pct"])
    df_out = df_out.astype({"time":"int","depth":"float","t_pct":"float"})
    return df_out 
flood_df_long = wide_to_long(flood_df,["duration","inflection_t"],"signal")
# =============================================================================
# STEP 1: get raw signals for all the true floods
# =============================================================================
event_df = pd.read_pickle('/Users/tanvibansal/Documents/GitHub/ptsa_project/event_df_tidy')
flood_df = event_df.loc[event_df.label == "flood"]

# =============================================================================
# STEP 2: get durations of all floods rounded up to the nearest minute
# =============================================================================
flood_df.loc[:,"duration"] = flood_df.apply(lambda x: math.ceil(x["signal"]["time"][-1]/60),axis=1)
flood_df = flood_df.loc[flood_df.duration > 4] #drop floods with duration <= 4 because these are all blips and boxes
#flood_df = flood_df.loc[flood_df.apply(lambda x: x["signal"]["depth"].max(),axis=1) > 30] #drop floods with max depth > 30 

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

flood_df_mod = flood_df.copy(deep=True)
flood_df_mod["signal"] = flood_df_mod.apply(lambda x: remove_outliers(x,2),axis=1)

# =============================================================================
# STEP 4: locate peaky floods and drop 
# =============================================================================
flood_df_mod["mean_diff"] = flood_df_mod["signal"].apply(lambda x: np.abs(np.diff(x["depth"])).mean())
flood_df_mod = flood_df_mod.loc[flood_df_mod.mean_diff < 65]

flood_df_long = wide_to_long(flood_df_mod,["duration","mean_diff"],"signal")
ggplot(flood_df_long.reset_index(),aes(x="t_pct",y="depth",group="uuid",color="mean_diff")) + geom_line(alpha = 0.6) 

# =============================================================================
# STEP 5: drop the manually identified suspect floods 
# =============================================================================
ugly_flood = [8612524, 3230015]
suspect_floods = [2860240, 1929133]

flood_df_mod_filt = flood_df_mod.drop(suspect_floods + ugly_flood)
flood_df_long_filt = wide_to_long(flood_df_mod_filt,["duration","mean_diff"],"signal")

ggplot(flood_df_long_filt.reset_index(),aes(x="t_pct",y="depth",group="uuid")) + geom_line(alpha = 0.6) 

# =============================================================================
# STEP 5: smooth floods and cut into rising/falling segments
# =============================================================================
def smooth_and_cleave(ev):
    x = ev.copy(deep=True)
    signal = pd.DataFrame(x.signal)
    
    std = np.std((signal["depth"] - signal["depth"].mean())/signal["depth"].max())*signal["depth"].max()
    smooth = signal.rolling(window=len(signal)//5, win_type="gaussian",center=True).mean(std=std)
    cleave_ind = np.argwhere(smooth["depth"] == smooth["depth"].max()).flatten().min()
    
    rise = signal[:cleave_ind].to_dict(orient='list')
    fall = signal[cleave_ind:].to_dict(orient='list')
    
    smooth = smooth.to_dict(orient='list')
   
    
    return pd.Series({"smooth":smooth, "rise":rise,"fall":fall, "inflection_t":signal["time"][cleave_ind]})
flood_df_sim = flood_df_mod_filt.copy(deep=True)
flood_df_sim[["smooth","rise","fall","inflection_t"]] = flood_df_sim.apply(lambda x: smooth_and_cleave(x),axis=1)

flood_df_sim_long = wide_to_long(flood_df_sim,["duration","inflection_t"],"signal")
flood_df_sim_long.loc[(flood_df_sim_long.time > flood_df_sim_long.inflection_t),"acc"] = "fall"
flood_df_sim_long.loc[(flood_df_sim_long.time <= flood_df_sim_long.inflection_t),"acc"] = "rise"

# =============================================================================
# STEP 6: generate simulated floods for rising/falling action at all durations as an average of all flood samples
# =============================================================================
flood_simulate_df = pd.DataFrame(flood_df_sim["smooth"].to_list(),index = flood_df_sim.index).rename(columns={"time":"time.smooth","depth":"depth.smooth"}).join(flood_df_sim[["inflection_t","duration"]]).explode(["time.smooth","depth.smooth"]).astype({"time.smooth":"float","depth.smooth":"float"})
flood_simulate_df["t_pct"] = (flood_simulate_df["time.smooth"]/(flood_simulate_df["duration"]*60)).astype("float")
flood_simulate_df.loc[(flood_simulate_df["time.smooth"] >= flood_simulate_df.inflection_t),"acc"] = "fall"
flood_simulate_df.loc[(flood_simulate_df["time.smooth"] < flood_simulate_df.inflection_t),"acc"] = "rise"

flood_measure_df =pd.DataFrame(flood_df_sim["signal"].to_list(),index = flood_df_sim.index).join(flood_df_sim[["inflection_t","duration"]]).explode(["time","depth"]).astype({"time":"float","depth":"float"})
flood_measure_df["t_pct"] = (flood_measure_df["time"]/(flood_measure_df["duration"]*60)).astype("float")
flood_measure_df.loc[(flood_measure_df.time >= flood_measure_df.inflection_t),"acc"] = "fall"
flood_measure_df.loc[(flood_measure_df.time < flood_measure_df.inflection_t),"acc"] = "rise"

#plot drainage profiles and save them for later use 
flood_uuids = flood_df_sim.index.values
i = flood_uuids[30]
for i in flood_uuids:
    p = ggplot(flood_simulate_df.loc[i],aes(x = "t_pct", y = "depth.smooth",color="acc")) + geom_line(linetype="dashed") + \
        geom_line(flood_measure_df.loc[i],aes(x = "t_pct",y = "depth",color="acc"),linetype="solid") +\
        labs(x = "t_pct", y = "depth", title = "Depth Filtered and Smoothed UUID: %s"%(i))    
    ggsave(p,filename="/Users/tanvibansal/Documents/GitHub/ptsa_project/flood_drainage_tuning/%s.png"%(i),format="png")

#fit a gamma distribution to each flood, grab the drainage profiles, estimate error, and train params to minimize drainage errors
x = flood_simulate_df.loc[i]

def fit_drainage_profile(ev, a, c):
    x = ev.copy(deep=True).dropna()
    duration = round(x['time.smooth'].iloc[-1]) + 1
    peak = x['depth.smooth'].max()
    
    depth_sim = fake_data.flood(duration=duration, a=a, c=c, peak=peak, power=1., noise=0.)
    depth_sim = depth_sim[x['time.smooth'].round().values.astype("int")]
    
    #normalize smoothed and simulate depths and grab their drainage profiles
    if (x.acc == "fall").sum() > 0:
        drainage_start_ind = np.argwhere(x['time.smooth'] >= x.inflection_t).min()
        depth_smooth_drain = x["depth.smooth"].iloc[drainage_start_ind:].values
        depth_sim_drain = depth_sim[drainage_start_ind:]
        
        #estimate rmse between fit profile and measured
        rmse = np.sqrt(np.sum((depth_smooth_drain - depth_sim_drain)**2)/(len(depth_smooth_drain)*peak))
    
        return rmse

def optimize_drainage_profile(params, flood_simulate_df):
    a = params[0]
    c = params[1]
    x_rmse = []
    for i in flood_simulate_df.index.unique().values:
        ev = flood_simulate_df.loc[i].copy(deep=True)
        rmse = fit_drainage_profile(ev, a, c)
        if rmse:
            x_rmse.append(rmse)
    return np.sum(np.array(x_rmse))

gamma_train = fmin(optimize_drainage_profile, np.array([50,1]), args=(flood_simulate_df,))
optimize_drainage_profile(gamma_train, flood_simulate_df)


# =============================================================================
# STEP 7: using the optimized parameters simulate floods for each event, cleave into rising/falling, and output to csv
# =============================================================================
def simulate_floods(params, ev):
    a = params[0]
    c = params[1]
    
    x = ev.copy(deep=True).dropna()
    time = x['signal']['time']
    duration = round(time[-1]) + 1
    peak = x['signal']['depth'].max()
    inflection_t = x['inflection_t']
    inflection_t_ind = np.argwhere(time == inflection_t).flatten().item()
    
    depth_sim = fake_data.flood(duration=duration, a=a, c=c, peak=peak, power=1., noise=0.)
    depth_sim = depth_sim[time]
    
    signal_sim = {"time": time, "depth": depth_sim}
    signal_rise = {"time": time[0:inflection_t_ind], "depth": depth_sim[0:inflection_t_ind]}
    signal_fall = {"time": time[inflection_t_ind:], "depth": depth_sim[inflection_t_ind:]}
    
    return pd.Series({"signal_sim":signal_sim, "signal_rise":signal_rise, "signal_fall":signal_fall})
flood_df_out = flood_df_sim.copy(deep=True)[["deployment_id","label","signal","inflection_t"]]
flood_df_out[["signal_sim","signal_sim_rise","signal_sim_fall"]] = flood_df_out.apply(lambda ev: simulate_floods(gamma_train, ev),axis=1)
    
flood_df_long = flood_df_out.apply(lambda x: pd.Series({"time":x["signal"]['time'], "depth": x["signal"]["depth"], "depth_sim":x["signal_sim"]["depth"]}),axis=1).explode(["time","depth","depth_sim"]).astype({"time":"int","depth":"float","depth_sim":"float"})

i = flood_uuids[2]
plot_df = flood_df_long.loc[i]
ggplot(plot_df,aes(x="time",y="depth")) + geom_line(alpha=0.5) + geom_line(aes(x = "time",y="depth_sim"),color="blue",alpha=0.5,linetype="dashed")

# =============================================================================
# STEP 8: using the optimized parameters simulate floods for each event, cleave into rising/falling, and output to csv
# ============================================================================


flood_df_out.to_pickle("/Users/tanvibansal/Documents/GitHub/ptsa_project/flood_simulations")



