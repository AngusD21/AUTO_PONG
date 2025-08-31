# kalman_filter.py  (extend your existing class)  :contentReference[oaicite:17]{index=17}
import numpy as np

class KalmanFilter3D: 
    def __init__(self, dt=1/90):

        self.dt = dt
        self.x = np.zeros((6,1))  # [x,y,z,vx,vy,vz]
        dt = self.dt
        self.F = np.array([[1,0,0,dt,0,0],
                           [0,1,0,0,dt,0],
                           [0,0,1,0,0,dt],
                           [0,0,0,1,0,0],
                           [0,0,0,0,1,0],
                           [0,0,0,0,0,1]])
        
        # control (constant acceleration)
        self.B = np.array([[0.5*dt*dt,0,0],
                           [0,0.5*dt*dt,0],
                           [0,0,0.5*dt*dt],
                           [dt,0,0],
                           [0,dt,0],
                           [0,0,dt]])
        
        self.H = np.array([[1,0,0,0,0,0],
                           [0,1,0,0,0,0],
                           [0,0,1,0,0,0]])
        
        self.P = np.eye(6)*10
        self.Q = np.diag([1e-4,1e-4,1e-4, 1e-2,1e-2,1e-2])
        self.R = np.diag([5e-4,5e-4, 5e-4])  # metres^2 (tune)

    def predict(self, acc=np.array([0,0,0], dtype=float)):
        a = acc.reshape(3,1)
        self.x = self.F @ self.x + self.B @ a
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]

    def update(self, z):
        z = np.reshape(z, (3,1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
