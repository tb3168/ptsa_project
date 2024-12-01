import numpy as np


class KalmanFilter:
    """
    F: dynamics prediction matrix
        F @ [x1, v1, a1] -> [x2, v2, a2]
        3x3: 
            x = x + v*t + a*t^2 / 2
            v =     v +   a*t
            a =           a
    H: state estimation conversion matrix
        H @ [x, v, a] -> [x]

    predict(u | z, P; F, B, Q):
        z = Fz + Bu
        P = FPF^T + Q

    update(x | z, P; H, R):
        y = x - Hz
        S = HPH^T + R

        K = PH^TS^-1
        J = I - KH

        z = z + Ky
        P = JPJ^T + KRK^T

    """
    def __init__(self, x=0, B=None, H=None, Q=None, R=None, z=None, P=None, dt=None, max_age=None):
        self.dt = dt or 1  # default time step
        self.max_age = max_age  # maximum age of the filter between measurement and predict time in seconds
        self._update_F(self.dt)  # transition matrix

        # constants
        self.n = self.F.shape[1]  # state dimension
        self.I = np.eye(self.n)  # identity matrix

        # parameters
        self.H = np.eye(self.n)[:1] if H is None else H  # measurement function (state to measurement space)
        self.B = 0 if B is None else B  # control input
        self.Q = np.eye(self.n) if Q is None else Q  # process noise
        self.R = np.eye(self.n) if R is None else R  # measurement noise

        # state
        self.z = np.zeros((self.n, 1)) if z is None else z  # state estimate
        self.P = np.eye(self.n) if P is None else P  # state covariance
        self.tx = 0  # last measurement time
        self.tp = 0  # last prediction time

        # initialize filter position
        if x is not None:
            self.set_x(x)

    def _update_F(self, dt):
        '''Set state transition matrix using time delta'''
        # x = x + v*dt + a*dt^2 / 2
        # v =     v +   a*dt
        self.F = np.array([
            [1, dt, 0.5 * dt**2], 
            [0,  1,       dt], 
            [0,  0,        1]
        ])

    def _update_time(self, t):
        '''set prediction time and update transition matrix using time difference'''
        dt = t - self.tp if self.tp is not None else 1
        self.tp = t  # update prediction time
        self._update_F(dt)

    def predict(self, t, u=0):
        self._update_time(t)  # update timestamp and transition matrix

        self.z = self.F @ self.z + self.B * u  # state prediction  [n, 1]
        self.P = self.F @ self.P @ self.F.T + self.Q  # covariance prediction  [n, n]
        x_h = self.H @ self.z  # measurement prediction  [1]
        return x_h

    def estimate(self, x):
        y = x - self.H @ self.z  # measurement residual  [1]
        S = self.H @ self.P @ self.H.T + self.R  # residual covariance  [1, 1]
        return y, S

    def apply(self, y, S, x=None):
        self.tx = self.tp  # update measurement time to latest prediction time

        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain  [n, 1]
        J = self.I - K @ self.H  # [n, n]
        self.z = self.z + K @ y  # state update  [n, 1] + [n, 1] @ [1]
        self.P = J @ self.P @ J.T + K @ self.R @ K.T  # covariance update  [n, n] + [n, n]

        # set the filter position to the ground truth value to avoid drift
        if x is not None:
            self.set_x(x)
        return self.z, self.P

    def cost(self, y, S):
        '''Get how likely the measurement is given the state'''
        return (np.abs(y) / np.sqrt(np.diag(S)))
    
    def expired(self):
        '''Check if the filter hasn't been updated for a while'''
        return self.max_age and self.tx is not None and self.tp - self.tx > self.max_age

    def set_x(self, x):
        self.z[0] = x


class IMM:
    def __init__(self, filters, new_filter):
        self.filters = filters
        self.new_filter = new_filter

    @property
    def n(self):
        return len(self.filters)

    def predict(self, u=0):
        x_hs = np.zeros((self.n, 1, 1))
        for i in range(self.n):
            x_hs[i] = self.filters[i].predict(u)
        return x_hs

    def estimate(self, x):
        ys = np.zeros((self.n, 1, 1))
        Ss = np.zeros((self.n, 1, 1))
        for i in range(self.n):
            y, S = self.filters[i].estimate(x)
            ys[i], Ss[i] = y, S

        return ys, Ss

    def cost(self, ys, Ss):
        cost = np.zeros((self.n, 1, 3))
        for i in range(self.n):
            cost[i] = self.filters[i].cost(ys[i], Ss[i])
        return cost

    def apply(self, i, y, S, x):
        self.filters[i].apply(y, S, x)

        # drop stale filters
        self.filters = [f for f in self.filters if not f.expired()]

    def missed(self, x, t):
        # add new filter
        f = self.new_filter(x)
        self.filters.append(f)
        # warmup
        f.predict(t)
        y, S = f.estimate(x)
        f.apply(y, S)
        return y, S



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


# import ipdb
# @ipdb.iex
def example():

    # ---------------------------------------------------------------------------- #
    #                                 Create filter                                #
    # ---------------------------------------------------------------------------- #

    dt = 1.0/60
    Q = np.array([[6, 0.05, 0.0], [6, 0.1, 0.0], [0.0, 0.0, 0.0]])/10   # process noise
    R = np.array([[10.0]])                                              # measurement noise

    # create IMM filter with dynamic noise filters
    im = IMM([
        KalmanFilter(x=0, Q=Q, R=R, dt=dt), 
    ], (lambda x: KalmanFilter(x, Q=Q/50, R=R/50, dt=dt, max_age=3/60)))  # noise filters with different noise parameters

    # ---------------------------------------------------------------------------- #
    #                                  Create data                                 #
    # ---------------------------------------------------------------------------- #

    measurements = get_data()

    # ---------------------------------------------------------------------------- #
    #                                  Run filters                                 #
    # ---------------------------------------------------------------------------- #

    predictions = []
    uncertainties = []
    for i, x in enumerate(measurements):
        timestamp = dt * i
        # predict all filters into the future
        x_h = im.predict(timestamp)
        x_h = x_h[:, 0, 0]

        # estimate prediction error
        y, S = im.estimate(x)
        U = 3 * np.sqrt(S[:, 0,0])
        cost = im.cost(y, S)
        i = np.argmin(cost.mean((1,2)))  # get the best filter

        # check if the measurement is within the expected region
        if (cost[i] < 3).all() or (0 <= x < x_h[i]):
            im.apply(i, y[i], S[i], x)
            x_h[i] = x
        else:
            # if not, assume a novel noise source - create a new filter
            y, S = im.missed(x, dt*i)

            # append new filter prediction
            x_h = np.concatenate([x_h, [x]])
            U = np.concatenate([U, [3 * np.sqrt(S[0,0])]])

        predictions.append(x_h)
        uncertainties.append(U)

    # ---------------------------------------------------------------------------- #
    #                                 Plot results                                 #
    # ---------------------------------------------------------------------------- #

    # concatenate predictions and uncertainties
    max_n = max(len(p) for p in predictions)
    predictions = np.array([np.pad(p, (0, max_n - len(p)), constant_values=np.nan) for p in predictions]).T
    max_n = max(len(u) for u in uncertainties)
    uncertainties = np.array([np.pad(u, (0, max_n - len(u)), constant_values=np.nan) for u in uncertainties]).T

    # plot the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 6))
    for i in range(predictions.shape[0]):
        plt.fill_between(range(len(predictions[i])), predictions[i] - uncertainties[i], predictions[i] + uncertainties[i], alpha=0.5)
        plt.plot(predictions[i], label=f'filter {i}', marker='.')
    plt.plot(measurements, label='measured', color='mediumblue')
    plt.ylim(-30, measurements.max() + 10)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    import fire
    fire.Fire(example)
