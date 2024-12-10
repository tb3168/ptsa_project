#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:08:38 2024

@author: tanvibansal
"""
import pandas as pd
import numpy as np 
import os 
os.chdir("/Users/tanvibansal/Documents/GitHub/ptsa_project/")
import glob
from run_kalman import apply_kalman_filter, apply_kalman_filter_3d
from plotnine import *
from pandarallel import pandarallel 
from particle_filter import run_particle_filter

root_fp = "/Users/tanvibansal/Documents/GitHub/flood-filters/flood_filters/"
data_fp = root_fp + "data-1015/"
key_fp = root_fp + "experimental_pipeline/phase 2/2.3/" + "sensor_events_1120.csv"

key = pd.read_csv(key_fp).dropna(subset=["label","deployment_id","start_time","end_time"])
sensors = key.deployment_id.unique()

s = sensors[1]
thresh = 0.33

def evaluate_filter(s, data_fp, thresh):
    def read_sensor_csvs(s, data_fp):
        #read in all the data from one sensor
        
        data_fps = glob.glob(data_fp + s + "/*")
        if len(data_fps) > 0:
            data = pd.concat([pd.read_csv(dfp) for dfp in data_fps])
            data = data.sort_values(by = "time").reset_index(drop = True)
            data = data.dropna(subset = ["time","depth_filt_mm"])
            data["time"] = pd.to_datetime(data["time"],format="ISO8601")#"%Y-%m-%d %H:%M:%S.%f+%z")
        
            return data 
    data = read_sensor_csvs(s, data_fp)
    
    if data is not None:
        #extract timestamps and measured data as vectors from the measured signal df
        time = data["time"]
        measured = data["depth_filt_mm"].values
        
        #compute or extract or read the filtered data as a vector. assumption is that the time stamps are identical to the measured
        def get_heuristic_filtered_signal(data): #adjust this function as needed for testing different filters
            filtered = data["depth_proc_mm"].values
            return filtered
        #filtered =  get_heuristic_filtered_signal(data)
#        filtered = apply_kalman_filter_3d(time.values, measured)
        filtered = run_particle_filter(time.values, measured, 2000)
        #find percent change between unfiltered and filtered depths for each point and mark as filtered if greater than thresh
        def get_filtered_indicator(measured, filtered, thresh):
            pct_change = np.empty((len(measured),),dtype="float")
            pct_change.fill(np.nan)
            
            #if unfiltered point is 0, percent change is 0 
            pct_change[measured == 0] = 0.0
            
            #if the unfiltered point is none, percent change is none. 
            
            #if unfiltered point is not none and filtered point is none, assume filtered point is 0 and replace the nan
            filtered[((~np.isnan(measured)) & (np.isnan(filtered)))] = 0 
            
            #for all the points where unfiltered is not zero or none, compute the percent change
            compute_mask = ((~np.isnan(measured)) & (measured != 0.0))
            pct_change[compute_mask] = (measured[compute_mask] - filtered[compute_mask])/measured[compute_mask]
            
            #convert the pct_change array to a binary filtered/unfiltered array 
            evl_mask = (~np.isnan(pct_change))
            #thresh = 0.33
            filtered_mask = (pct_change[evl_mask] > thresh).astype("int")
            return filtered_mask
        filtered_mask =  get_filtered_indicator(measured, filtered, thresh)
        
        #concat results to a dataframe that can be cross referenced with the key
        filt_eval_df = pd.DataFrame(data = {"time":time, "measured_depth": measured, "filtered_depth": filtered, "filtered_mask": filtered_mask})
        filt_eval_df["label"] = None
        filt_eval_df["event_id"] = None
        
        #cross reference all the events for the sensor of interest with the key and populate the event id/label column, then the noise column
        def populate_event_data(key, s, filt_eval_df):
            s_events = key.loc[key.deployment_id == s]
            filt_eval_df.set_index("time",inplace=True)
            for i in range(len(s_events)):
                s_ev = s_events.iloc[i]
                ev_mask = (filt_eval_df.index >= s_ev.start_time) & (filt_eval_df.index <= s_ev.end_time)
                filt_eval_df.loc[ev_mask,"label"] = s_ev.label
                filt_eval_df.loc[ev_mask,"event_id"] = s_ev.id
            filt_eval_df.reset_index(inplace = True)
            filt_eval_df["noise"] = filt_eval_df.label.apply(lambda x: 1 if x is not None and x != "flood" else 0)
            return filt_eval_df
        filt_eval_df = populate_event_data(key, s, filt_eval_df)
        
        #aggregate outputs: drop rows where depth = 0, then group by event id and count the number of points that are noise and number of points filtered
        agg_eval_df = filt_eval_df.loc[filt_eval_df.measured_depth != 0.0].groupby(["event_id","label"])[["filtered_mask","noise"]].sum()
        
        return agg_eval_df

#apply the function for each sensor and concatenate all the results into a df
eval_df = []
for s in sensors:
    eval_df.append(evaluate_filter(s, data_fp, thresh))
    print("sensor %s complete"%(s))
eval_df = pd.concat(eval_df)

# now we can extract our aggregate metrics of interest: sample and event level filtration scores
eval_df = eval_df.reset_index()

#sample level 
sample_eval = eval_df.loc[eval_df.label != "flood"].groupby("event_id")[["filtered_mask","noise"]].sum()
sample_eval["acc"] = sample_eval["filtered_mask"]/sample_eval["noise"]
sample_eval.acc.describe()

#event_level 
event_eval = sample_eval.copy(deep=True)
event_eval["filt"] = (event_eval["acc"] > 0.75).astype("int")
event_eval.filt.value_counts()
event_eval.filt.describe()

flood_df_proc = filt_eval_df.loc[filt_eval_df.label == "flood"]
for i in flood_df_proc.event_id.unique():
    x = flood_df_proc.loc[flood_df_proc.event_id == i] 
    plt.subplots()
    plt.plot(x.time, x.measured_depth, label = "measured")
    plt.plot(x.time, x.filtered_depth, label = "filtered_continuous")
    plt.plot(x.time, apply_kalman_filter_3d(x.time.values, x.measured_depth.values), label = "filtered_event")
    plt.legend()
    plt.title("%s"%(i))















