#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:14:30 2024

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
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy import stats 

flood_df = pd.read_pickle("flood_df")

#TO DO LATER: generate the target signal with a gaussian smoother
#compute 1st and 2nd gradients of the true signal and the simulated signal
grad = flood_df.copy(deep = True)

grad[["d_time", "d_depth"]] = grad.signal_padded.apply(lambda x: pd.Series(data = {"d_time":np.diff(x["time"]), "d_depth":np.diff(x["depth"])}))
grad[["d2_time", "d2_depth"]] = grad.apply(lambda x: pd.Series(data = {"d2_time": x.d_time[1:], "d2_depth":np.diff(x.d_depth)}), axis=1)

grad[["d_sim_time", "d_sim_depth"]] = grad.signal_sim_padded.apply(lambda x: pd.Series(data = {"d_time":np.diff(x["time"]), "d_depth":np.diff(x["depth"])}))
grad[["d2_sim_time", "d2_sim_depth"]] = grad.apply(lambda x: pd.Series(data = {"d2_time": x.d_sim_time[1:], "d2_depth":np.diff(x.d_sim_depth)}), axis=1)

#position 
pos = grad.apply(lambda x: x.signal_sim_padded["depth"] - x.signal_padded["depth"], axis=1).rename("position").explode("position")

plt.hist(pos)

stats.describe(np.array(pos,dtype="float"))
pos_shape, pos_mu, pos_sigma = stats.genextreme.fit(np.array(pos, dtype = float))

#velocity 
vel = (grad["d_depth"]/grad["d_time"] - grad["d_sim_depth"]/grad["d_sim_time"]).rename("velocity").explode("velocity")

plt.hist(vel,bins = 30,density=True)
plt.xlim(-10,10)

vel_mu, vel_sigma = norm.fit(np.array(vel.values, dtype=float))

#acceleration 
acc = (grad["d2_depth"]/grad["d2_time"] - grad["d2_sim_depth"]/grad["d2_sim_time"]).rename("acceleration").explode("acceleration")

plt.hist(acc[(acc > -1) & (acc < 1)],bins=100,density=True)

acc_mu, acc_sigma = norm.fit(np.array(acc.values, dtype = float))


#repeat for noise
event_df = pd.read_pickle("/Users/tanvibansal/Documents/GitHub/flood-filters/flood_filters/experimental_pipeline/phase 2/2.3/event_df")
noise_df = event_df.loc[event_df.label != "flood"].dropna(subset = "signal")

grad_noise = noise_df.copy(deep = True)
grad_noise[["d_time", "d_depth"]] = grad_noise.signal.apply(lambda x: pd.Series(data = {"d_time":np.diff(x["time"]), "d_depth":np.diff(x["depth"])}))
grad_noise[["d2_time", "d2_depth"]] = grad_noise.apply(lambda x: pd.Series(data = {"d2_time": x.d_time[1:], "d2_depth":np.diff(x.d_depth)}), axis=1)

#position 
pos_noise = (grad_noise.signal.apply(lambda x: x["depth"])).rename("position").explode("position")
pos_noise = pos_noise.dropna()

plt.hist(pos_noise)

stats.describe(np.array(pos_noise,dtype=float))
pos_n_shape, pos_n_mu, pos_n_sigma = stats.gamma.fit(np.array(pos_noise,dtype=float))

#velocity 
vel_noise = (grad_noise["d_depth"]/grad_noise["d_time"]).rename("velocity").explode("velocity")
vel_noise = vel_noise.replace([-np.inf, np.inf],np.nan).dropna()
stats.describe(vel_noise)

plt.hist(vel_noise,bins = 30,density=True)

vel_n_shape, vel_n_mu, vel_n_sigma = stats.genextreme.fit(np.array(vel_noise, dtype = float))

#acceleration 
acc_noise = (grad_noise["d2_depth"]/grad_noise["d2_time"]).rename("acceleration").explode("acceleration")
acc_noise = acc_noise.replace([-np.inf, np.inf],np.nan).dropna()
stats.describe(acc_noise)

plt.hist(acc_noise[(acc_noise > -1) & (acc_noise < 1)],bins=100,density=True)

acc_n_shape, acc_n_mu, acc_n_sigma = stats.genextreme.fit(np.array(acc_noise, dtype = float))
