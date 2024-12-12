#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:46:44 2024

@author: tanvibansal
"""
import pandas as pd 
import numpy as np 
from pykalman import KalmanFilter

def apply_kalman_filter(observations):
    global_params = pd.read_pickle("/Users/tanvibansal/Documents/GitHub/ptsa_project/global_params.pkl")
    
    transition_matrix = global_params["transition_matrix"]
    process_noise_cov = global_params["process_noise_base"] 
    observation_matrix = np.array([[1, 0, 0]])#global_params["observation_matrix"]
    observation_noise_cov = global_params["observation_covariance"]
    
    kf = KalmanFilter(
                transition_matrices=transition_matrix,
                observation_matrices=observation_matrix,  # Observation model
                transition_covariance=process_noise_cov,
                observation_covariance=observation_noise_cov,
                initial_state_mean=np.zeros(transition_matrix.shape[0]),
                initial_state_covariance=np.eye(transition_matrix.shape[0])
            )
    # Apply filtering and smoothing
    filtered_state_means, filtered_state_covariances = kf.filter(observations)
    smoothed_state_means, smoothed_state_covariances = kf.smooth(observations)
    
    return filtered_state_means[:,0]

def apply_kalman_filter_3d(times, observations):
    global_params = pd.read_pickle("/Users/tanvibansal/Documents/GitHub/ptsa_project/global_params.pkl")
     
    #time_diffs = np.diff(times)
    time_diffs = (np.diff(times, prepend=times[0])/1000000000).astype("int")  # Compute time gaps with 0 for the first element

    # Initialize state and covariance matrices
    n_observations = len(observations)
    state_dim = 3  # [depth, velocity, acceleration]
    filtered_means = np.zeros((n_observations, state_dim))
    smoothed_means = np.zeros((n_observations, state_dim))
    covariances = np.zeros((n_observations, state_dim, state_dim))

    # Initial state and covariance
    state_mean = np.array([0, 0, 0])  # Initial [depth, velocity, acceleration]
    state_covariance = np.eye(state_dim) * 1  # Initial uncertainty
    observation_matrix = np.array([[1, 0, 0]])  # Observation model
    observation_covariance = global_params["observation_covariance"]  # Observation noise
    process_noise_base = global_params["process_noise_base"] 

    epsilon = 1e-3  # Small value for regularization

    # Filtering
    for t in range(n_observations):
        if t > 0:
            delta_t = time_diffs[t - 1]
            transition_matrix = np.array([
                [1, delta_t, 0.5 * delta_t**2],
                [0, 1, delta_t],
                [0, 0, 1]
            ])
            process_noise = process_noise_base * delta_t

            # Predict step
            predicted_state_mean = np.dot(transition_matrix, state_mean)
            predicted_state_cov = (
                np.dot(transition_matrix, np.dot(state_covariance, transition_matrix.T)) + process_noise
            )

            # Update step
            innovation = observations[t] - np.dot(observation_matrix, predicted_state_mean)
            innovation_cov = (
                np.dot(observation_matrix, np.dot(predicted_state_cov, observation_matrix.T))
                + observation_covariance
            )
            kalman_gain = np.dot(
                predicted_state_cov,
                np.dot(observation_matrix.T, np.linalg.inv(innovation_cov + epsilon * np.eye(innovation_cov.shape[0])))
            )
            state_mean = predicted_state_mean + np.dot(kalman_gain, innovation)
            state_covariance = predicted_state_cov - np.dot(
                kalman_gain, np.dot(observation_matrix, predicted_state_cov)
            )

        # Store filtered results
        filtered_means[t] = state_mean
        covariances[t] = state_covariance

    # Smoothing
    smoothed_means[-1] = filtered_means[-1]
    smoothed_covariance = covariances[-1]
    for t in range(n_observations - 2, -1, -1):
        delta_t = time_diffs[t]
        transition_matrix = np.array([
            [1, delta_t, 0.5 * delta_t**2],
            [0, 1, delta_t],
            [0, 0, 1]
        ])

        # RTS smoother gain
        predicted_covariance = (
            np.dot(transition_matrix, np.dot(covariances[t], transition_matrix.T)) + process_noise_base * delta_t
        )
        predicted_covariance += epsilon * np.eye(predicted_covariance.shape[0])  # Regularization
        smoother_gain = np.dot(
            covariances[t],
            np.dot(transition_matrix.T, np.linalg.pinv(predicted_covariance))  # Use pseudo-inverse
        )

        # Update smoothed state
        smoothed_means[t] = (
            filtered_means[t]
            + np.dot(smoother_gain, (smoothed_means[t + 1] - np.dot(transition_matrix, filtered_means[t])))
        )
    return filtered_means[:,0]