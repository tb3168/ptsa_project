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

# =============================================================================
# STEP 1: get raw signals for all the true floods
# =============================================================================
event_df = pd.read_pickle('/Users/tanvibansal/Documents/GitHub/ptsa_project/event_df_tidy')
flood_df = event_df.loc[event_df.label == "flood"]

# =============================================================================
# STEP 2: get durations of all floods rounded up to the nearest minute
# =============================================================================
flood_df.loc[:,"duration"] = flood_df.apply(lambda x: math.ceil(x["signal"]["time"][-1]/60),axis=1)

# =============================================================================
# STEP 3: generate simulated floods at all the durations and append onto dataframe
# =============================================================================
def generate_simulated_flood(x):
    duration = x.duration
    peak = x.signal["depth"].max()
    depth_sim = fake_data.flood(duration=duration, a=0.7, c=1.7, peak=peak, power=1., noise=0.)
    t_sim = np.arange(duration)*60
    return {"time":t_sim,"depth":depth_sim}

flood_df["signal_sim"] = flood_df.apply(lambda x: generate_simulated_flood(x),axis=1)

# =============================================================================
# STEP 4: get time step comparison of measured depth and simulated depth
# =============================================================================

flood_measure_df = pd.DataFrame(flood_df["signal"].to_list(),index=flood_df.index)
flood_measure_df['time'] = flood_measure_df.time.apply(lambda x: np.round(x/60))
flood_measure_df = flood_measure_df.explode(["time","depth"]).astype({"time":"int","depth":"float"}).reset_index().set_index(["uuid","time"])

flood_simulate_df = pd.DataFrame(flood_df["signal_sim"].to_list(),index=flood_df.index)
flood_simulate_df['time'] = flood_simulate_df.time.apply(lambda x: np.round(x/60))
flood_simulate_df = flood_simulate_df.explode(["time","depth"]).astype({"time":"int","depth":"float"}).reset_index().set_index(["uuid","time"])

noise_est_df = flood_simulate_df.join(flood_measure_df,how="left",lsuffix=".sim",rsuffix=".meas")

# =============================================================================
# STEP 5: estimate noise between measured and simulated for each event
# =============================================================================
event_error = noise_est_df.dropna()
event_error.loc[:,"error"] = (event_error.loc[:,"depth.sim"] - event_error.loc[:,"depth.meas"])

plt.hist(event_error.error,bins=30)
