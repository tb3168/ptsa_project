#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:10:32 2024

@author: tanvibansal
"""

from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters for our distribution p(observation|latent)
#GAUSS_WEIGHTS = [0.3, 0.3, 0.4]
#GAUSS_MEANS = [np.zeros(2), np.array([2.0,0.6]), np.array([-1.4,0.2])]
#GAUSS_COVS = [np.eye(2) * 3.0, np.eye(2)*2.0, np.eye(2) * 0.1]

def observation_probability(latent: np.ndarray, observation: np.ndarray) -> float:
    """Given an observation and corresponding latent state, evaluate the likelihood.

    Args:
        latent: Latent state at current time step.
        observation: Observation at current time step.

    Returns:
        Likelihood p(observation|latent).
    """
    # evaluate the likelihood of the observed position based on the latent position
    evaluation_mean = latent[0]
    if observation == 0: 
        likelihood = 1
    else:
    #likelihood = stats.gamma.pdf(observation, 0.288, loc=evaluation_mean, scale=6.01)
        likelihood = stats.norm.pdf(observation, loc = evaluation_mean, scale = 10)
    return likelihood

def observation_sample(latent: np.ndarray) -> np.ndarray:
    """Given the latent state, sample a observation.

    Args:
        latent: Latent state at current time step.

    Returns:
        Sampled observation.
    """
    # Let's keep things somewhat 'simple' by making our distribution a sum of Gaussians
    index = np.random.choice(np.arange(3), p=GAUSS_WEIGHTS)
    evaluation_mean = latent + GAUSS_MEANS[index]
    cov = GAUSS_COVS[index]
    return stats.multivariate_normal(mean=evaluation_mean, cov=cov).rvs()

def latent_sample(delta_t, latent: np.ndarray,observation) -> np.ndarray:
    """Given the latent state, sample the next latent state.

    Args:
        latent: Latent state at current time step.

    Returns:
        Sampled latent state.
    """
    # Let's keep things somewhat 'simple' by making our distribution a sum of Gaussians
    transition_matrix = np.array([
        [1, delta_t, 0.5 * delta_t**2],
        [0, 1 + np.random.rand(), delta_t],
        [0, 0, 1]
    ])
    
    if observation == 0:
        sample=latent
    else:
        sample = transition_matrix @ latent
    return sample
    

def particle_filter(measured:dict, n_samples: int, dim_z) -> Tuple[np.ndarray, np.ndarray]:
    
    t = measured["time"]
    observations = measured["depth"]

    # Placeholder for all of our samples and weights.
    z_samples = np.zeros((len(observations) + 1, n_samples, dim_z))
    weights = np.zeros((len(observations) + 1, n_samples))
    
    # Draw initial samples and set initial weights.
    z_samples[0] = np.zeros((n_samples,dim_z))#np.array([np.random.uniform(0,.01, size = n_samples), np.random.uniform(-.5,.5, size = n_samples),np.random.uniform(-.01,.01, size = n_samples) ]).T
    weights[0] = np.ones(n_samples) / n_samples
    
    # Now let's start our particle filtering loop.
    for time in range(1,len(observations)+1):
            # Sample from the next latent state given the current latent state.
        dt = t[time] - t[time-1]
        print(dt)
        for samp_i in range(n_samples):
            # Pick a sample with probability equal to its weight (resampling)
            #print(time, samp_i, weights[time-1])
            m = np.random.choice(n_samples, p=weights[time-1])
            sample_choice = z_samples[time-1][m]

            # Move the selected sample and save it
            
            z_samples[time, samp_i] = latent_sample(dt, sample_choice, observations[time-1]) 
            # Compute the weights for each of our new samples.
            #print(self.compute_w(observations[time-1], z_samples[time]))
            weights[time] =  compute_w(observations[time-1], z_samples[time])
        #print(np.sum((z_samples[time] * weights[time,:,np.newaxis]), axis=1)[0])
    return z_samples, weights

def compute_w(observation_t: np.ndarray, z_samples_t: np.ndarray) -> np.ndarray:

    # Placeholder for the weights.
    weights_t = np.zeros(len(z_samples_t))
   
    # Calculate each weight. Don't forget to normalize at the end!
    for i in range(len(weights_t)):
        z_t_i = z_samples_t[i]
       # print((z_t_i, observation_t))
        weights_t[i] =  observation_probability(z_t_i, observation_t) 
    weights_t /= np.sum(weights_t) 
        
    return weights_t
    
flood_df = pd.read_pickle('flood_df')
measured = flood_df.iloc[0]["signal_padded"]
measured_f = {"time":measured["time"]}
z_samples, weights = particle_filter(measured, 2000, 3)
z_mean = np.sum((z_samples * weights[:,:,np.newaxis]), axis=1)
print(z_mean)

plt.plot(measured['time'],measured['depth'],label="measured")
plt.plot(measured['time'],z_mean[1:,0], label ="filtered")
plt.legend()

# Get the expected value of z at each timestep 