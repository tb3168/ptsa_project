#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:10:32 2024

@author: tanvibansal
"""

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def observation_probability(latent: np.ndarray, observation: np.ndarray) -> float:

    # evaluate the likelihood of the observed position based on the latent position
    # 0.5 probability that the observation is noise, 0.5 probability that the observation is a flood
    p = np.random.rand()
    evaluation_mean = latent[0]
    if p < 0.7: 
        likelihood = stats.norm.pdf(observation, loc = 0, scale = 1)
    else:
        likelihood = stats.gamma.pdf(observation, 0.288, loc=evaluation_mean, scale=1)
        #likelihood = stats.norm.pdf(observation, loc = evaluation_mean, scale = 1)
    return likelihood

def latent_sample(delta_t, latent: np.ndarray,observation) -> np.ndarray:

    # Let's keep things somewhat 'simple' by making our distribution a sum of Gaussians
    transition_matrix = np.array([
        [1, delta_t, 0.5 * delta_t**2],
        [0, 1, delta_t] ,
        [0, 0, 1]
    ])
    
    #0.7 probability that the latent state is at rest, 0.3 probability that the latent state moves
    p = np.random.rand()
    if p < 0.7:
        sample=np.array([0,0,0])
    else:
        sample = transition_matrix @ (latent + np.random.laplace(0, .01))
    return sample
    
def compute_w(observation_t: np.ndarray, z_samples_t: np.ndarray) -> np.ndarray:
    
    likelihoods = np.apply_along_axis(lambda arr:observation_probability(arr,observation_t), axis=1, arr=z_samples_t)
    
    weights_t = likelihoods/np.sum(likelihoods)

    return weights_t

def run_particle_filter(t, observations,n_samples):
    #def particle_filter(measured:dict, n_samples: int, dim_z) -> Tuple[np.ndarray, np.ndarray]:
    #measured = flood_df.iloc[10]["signal_padded"]
    dim_z = 3 
    
    #t = measured["time"]
    #observations = measured["depth"]
    
    # Placeholder for all of our samples and weights.
    z_samples = np.zeros((len(observations) + 1, n_samples, dim_z))
    weights = np.zeros((len(observations) + 1, n_samples))
    
    # Draw initial samples and set initial weights.
    z_samples[0] = np.array([np.random.uniform(0,.01, size = n_samples), np.random.uniform(-.5,.5, size = n_samples),np.random.uniform(-.01,.01, size = n_samples) ]).T
    weights[0] = np.ones((n_samples))*(1/n_samples) 
    
    dts = np.diff(t,prepend=t[0])#(np.diff(t, prepend=t[0])/1000000000).astype("int")
    # Now let's start our particle filtering loop.
    for time in range(1,len(observations)+1):
            # Sample from the next latent state given the current latent state.
        dt = dts[time-1]
        #print(time)
        
        if np.isnan(sum(weights[time-1])):
            m = np.random.choice(n_samples,replace = True, size = n_samples)
            sample_choice = z_samples[time-1][m]
        else:
            m = np.random.choice(n_samples,p=weights[time-1],replace = True, size = n_samples)
            sample_choice = z_samples[time-1][m]
        z_samples[time] = np.apply_along_axis(lambda arr:latent_sample(dt,arr,observations[time-1]), axis=1, arr=sample_choice)
        weights[time] =  compute_w(observations[time-1],z_samples[time])
        
    
    #z_samples, weights = particle_filter(measured, 2000, 3)
    z_mean = np.nansum((z_samples * weights[:,:,np.newaxis]), axis=1)
    z_pos = z_mean[1:,0]
    z_pos = np.nan_to_num(z_pos,nan=0.0)
    
    return z_pos
    #plt.plot(measured['time'],measured['depth'],label="measured")
    #plt.plot(measured['time'],z_pos, label ="filtered")
    #plt.legend()
    
