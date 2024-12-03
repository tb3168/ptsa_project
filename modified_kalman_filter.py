#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:04:31 2024

@author: tanvibansal
"""

import pandas as pd 
import numpy as np 

# =============================================================================
# READ IN FLOOD EVENTS AND START DEVELOPMENT W/ ONE FLOOD
# =============================================================================
df_train = pd.read_pickle("flood_simulations")
x_obs = df_train.copy(deep=True).loc[1690089,"signal"]
inflection_t = df_train.copy(deep=True).loc[1690089,"inflection_t"]
inflection_t_ind = np.argwhere(x_obs["time"] == inflection_t).flatten().item()

x_obs = {"time": x_obs["time"][inflection_t_ind:], "depth": x_obs["depth"][inflection_t_ind:]}
x_true =  df_train.copy(deep=True).loc[1690089,"signal_sim_fall"]

# =============================================================================
# ASSIGN PRIORS
# =============================================================================
phi_k = -0.5 
Q = 0.025 

x_zero = x_true["depth"][0].round()
P_zero = 0.025 

T = len(x_obs["depth"])

# =============================================================================
# LATENT STATE TIME UPDATED PREDICTION (PRIOR)
# =============================================================================
x_priors = np.zeros(T+1)
P_priors = np.zeros(T+1) 

x_priors[0] = x_zero
P_priors[0] = P_zero

for t in range(1,T+1):
    x_priors[t] = phi_k*x_priors[t-1]
    P_priors[t] = phi_k*P_priors[t-1]*phi_k + Q 
    
# =============================================================================
# OBSERVATION STATE TIME UPDATE & SMOOTH (POSTERIOR)
# =============================================================================
f = 10 
theta_hat = 0.89 
z_i = [(theta_hat*x_priors[i] + np.random.poisson(lam=1.0, size=1)).item() for i in range(1,T+1)]
r_i = x_obs["depth"] - z_i






