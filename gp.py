import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
from tslearn.clustering import TimeSeriesKMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

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

def normalize(signal):
    time = signal["time"]
    depth = signal["depth"]
    
    depth_max = depth.max()
    depth_norm = depth/depth_max 
    
    return({"time":time,"depth":depth_norm})


def clustering(flood_pp):
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

    print("count of events per cluster:\n",np.array(np.unique(y_pred,return_counts=True)).T)
    return y_pred


def preprocess_signal(signal):
    time = signal['time']
    depth = signal['depth']
    return pd.DataFrame({'time': time, 'depth': depth}).sort_values(by='time')


def main():
    flood = pd.read_pickle('flood_df')
    flood["signal"] = flood.signal.apply(lambda x: gradient_filter(drop_negative_depths(z_score_outlier_remove(x))))
    flood["signal"] = flood.signal.apply(lambda x: normalize(x))

    clusters = clustering(flood)
    
    flood['processed_signal'] = flood['signal'].apply(preprocess_signal)

    flood_type1 = flood.iloc[np.where(clusters==1)].copy()
    # flood_type2 = flood.iloc[np.where(np.isin(clusters, [2, 4, 5]))].copy()
    # flood_type3 = flood.iloc[np.where(np.isin(clusters,[3, 8]))].copy()

    print(f'number of flood events in flood type 1: {len(flood_type1)}')
    # print(f'number of flood events in flood type 2: {len(flood_type2)}')
    # print(f'number of flood events in flood type 3: {len(flood_type3)}')

    # Combine all flood events into a single dataframe
    flood1_events = flood_type1['processed_signal'].to_list()  # Replace with your actual dataframes
    for i, flood_df in enumerate(flood1_events):
        flood_df['event_id'] = i  # Add an identifier for each flood event

    flood_type1_combined = pd.concat(flood1_events, ignore_index=True)

    # Prepare features (X) and target (y)
    X = flood_type1_combined[['time', 'event_id']].values  # Use time and event_id as features
    y = flood_type1_combined['depth'].values

    # Define the kernel and GP regressor
    kernel = C(1.0, (1e-3, 1e3)) * RBF([10, 1], (1e-2, 1e2))  # Two-dimensional kernel
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # Fit the GP model
    gp.fit(X, y)
    
    # Generate predictions
    X_pred = []
    for event_id in range(len(flood1_events)):
        times = np.linspace(0, flood_type1_combined[flood_type1_combined['event_id'] == event_id]['time'].max(), 500)
        X_pred.append(np.column_stack((times, np.full_like(times, event_id))))
    X_pred = np.vstack(X_pred)

    y_pred, sigma = gp.predict(X_pred, return_std=True)

    # Visualization
    plt.figure(figsize=(14, 8))
    for event_id in range(len(flood1_events)):
        mask = flood_type1_combined['event_id'] == event_id
        plt.plot(flood_type1_combined.loc[mask, 'time'], flood_type1_combined.loc[mask, 'depth'], 'o', label=f'Event {event_id+1}')
        event_mask = X_pred[:, 1] == event_id
        plt.plot(X_pred[event_mask, 0], y_pred[event_mask], label=f'GP Prediction Event {event_id+1}')
        plt.fill_between(
            X_pred[event_mask, 0],
            y_pred[event_mask] - 1.96 * sigma[event_mask],
            y_pred[event_mask] + 1.96 * sigma[event_mask],
            alpha=0.2, label=f'95% CI Event {event_id+1}'
        )

    plt.xlabel('Time')
    plt.ylabel('Depth')
    plt.title('Combined GP Model for Multiple Flood Events')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()