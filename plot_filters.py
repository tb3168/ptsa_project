#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:11:27 2024

@author: tanvibansal
"""
from particle_filter import run_particle_filter
from run_kalman import apply_kalman_filter, apply_kalman_filter_3d
from matplotlib import pyplot as plt
import numpy as np 
def get_data():
    x = np.linspace(-5, 15, 150)
    # create flood event
    measurements = - (x**2 * 4 + 4*x - 2)  + np.random.normal(0, 2, len(x)) + 60
    measurements = np.concatenate([np.zeros(60), measurements, np.zeros(100)]) * 0.5
    measurements = np.clip(measurements, 0, None)
    measurements = measurements[:150]

    # box
    measurements[10:40] = 40
    measurements[23] = 64
    measurements[25] = 65
    measurements[30] = 60
    # blips
    measurements[50] = 60
    # measurements[51] = -20
    measurements[55] = 55
    measurements[56] = 65
    measurements[57] = 60
    # noise on top of flood
    measurements[90] = 75
    measurements[100:105] = 60
    return measurements

observations = get_data()
times = np.arange(len(observations))*60

#kalman_filtered = apply_kalman_filter_3d(times, observations)
particle_filtered = run_particle_filter(times, observations, 5000)

plt.plot(times,observations,label="measured")
#plt.plot(times,kalman_filtered,label="kalman")
plt.plot(times,particle_filtered,label="particle")
#plt.ylim(-10,70)
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("depth [mm]")
plt.title("Sample Depth vs. Time")