#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:18:59 2024

@author: tanvibansal
"""

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 
from plotnine import *
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

#get dataframe of floods
    
root_fp = "/Users/tanvibansal/Documents/GitHub/ptsa_project/"
event_df_tidy_flood = pd.read_pickle(root_fp + "flood_df")


# =============================================================================
# SMOOTH AND CLEAN
# =============================================================================
#function to remove outliers
def z_score_outlier_remove(signal):
    time = signal['time']
    depth = signal['depth']
    
    z_score = (depth - np.mean(depth))/np.std(depth)
    ind_to_keep = np.argwhere(z_score < 2).flatten()
    time_tidy = time[ind_to_keep]
    depth_tidy = depth[ind_to_keep]
    
    return {"time": time_tidy, "depth": depth_tidy}

#function to remove negative points
def drop_negative_depths(signal):
    time = signal["time"]
    depth = signal["depth"]
    
    ind_to_keep = np.argwhere(depth >= 0).flatten()
    
    time_tidy = time[ind_to_keep]
    depth_tidy = depth[ind_to_keep] 
    
    return {"time": time_tidy, "depth": depth_tidy}

#function to remove gradient violations
def gradient_filter(signal):
    time = signal["time"]
    depth = signal["depth"]
    
    grad = np.diff(depth,prepend=depth[0])
    ind_to_keep = np.argwhere(np.abs(grad) < 279.4).flatten()
    
    time_tidy = time[ind_to_keep]
    depth_tidy = depth[ind_to_keep] 
    
    return {"time": time_tidy, "depth": depth_tidy}

flood_outlier_rm = event_df_tidy_flood.copy(deep = True)
flood_outlier_rm["signal"] = flood_outlier_rm.signal.apply(lambda x: gradient_filter(drop_negative_depths(z_score_outlier_remove(x))))

# =============================================================================
# PREPROCESS
# =============================================================================
#function to normalize 
def normalize(signal):
    time = signal["time"]
    depth = signal["depth"]
    
    depth_max = depth.max()
    depth_norm = depth/depth_max 
    
    return({"time":time,"depth":depth_norm})
flood_pp = flood_outlier_rm.copy(deep=True)
flood_pp["signal"] = flood_outlier_rm.signal.apply(lambda x: normalize(x))

#transform timeseries signals from dictionary to matrix
T = flood_pp.apply(lambda x: len(x["signal"]["time"]), axis = 1).max()
N = len(flood_pp)
X = np.zeros((N, T, 1))

for i in range(N):
    x = flood_pp.iloc[i]
    signal = x["signal"]
    time = signal["time"]
    depth = signal["depth"]
    X[i, 0:len(depth)] = depth.reshape(-1,1)

#fit timeseries clustering
clusters = 9
samples = 500 
points = 180
model = TimeSeriesKMeans(n_clusters=clusters, metric="dtw", max_iter_barycenter=30, random_state=29)
y_pred = model.fit_predict(X[:samples,:points])

#plot clustered timseries
plt.figure(figsize=(8,8))
for yi in range(clusters):
    plt.subplot(3, 3, yi + 1)
    for xx in X[:samples][y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(model.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, points)
    #plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("DTW $k$-means")
    plt.tight_layout()
   
#plot full timeseries from clusters w/ suspect profiles
for xx in flood_pp.iloc[:samples][y_pred == 7]["signal"]:
    plt.plot(xx['depth'].ravel(),"k-",alpha=0.5)
    plt.title("Timeseries of Events in Cluster 7\ncount= %s"%((y_pred == 7).sum()))

for xx in flood_pp.iloc[:samples][y_pred == 6]["signal"]:
    plt.plot(xx['depth'].ravel(),"k-",alpha=0.5)
    plt.title("Timeseries of Events in Cluster 6\ncount= %s"%((y_pred == 6).sum()))

for xx in flood_pp.iloc[:samples][y_pred == 4]["signal"]:
    plt.plot(xx['depth'].ravel(),"k-",alpha=0.2)
    plt.title("Timeseries of Events in Cluster 4\ncount= %s"%((y_pred == 4).sum()))

print("count of events per cluster:\n",np.array(np.unique(y_pred,return_counts=True)).T)


##populate suspect list 
#suspect_flood_uuids = [9641026, 6155524, 3817509, 6455304, 1089798, 7370250, 8159541, 1343346, 5582583, 2914189, 
 #                      5951926, 7847079, 8010877, 1245241, 1069823, 2860240, 1381470, 8358636, 1608117, 2195521,
  #                     3323611, 3606063, 9829405, 2191264, 2679255, 6941412, 9717908, 9900112, 8114943, 7405880, 
   #                    9992012, 3959386]
#suspect_flood_event_df = event_df.loc[suspect_flood_uuids]
#suspect_flood_event_df.loc[suspect_flood_event_df.label == "suspect-flood"].to_csv("/Users/tanvibansal/Documents/GitHub/flood-filters/#flood_filters/experimental_pipeline/phase 2/2.2/suspect_floods.csv")
