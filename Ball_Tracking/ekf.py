# ekf_drag.py
# Extended Kalman Filter (EKF) for 3D projectile with quadratic drag.
# State x = [px, py, pz, vx, vy, vz]^T  (metres, metres/second)
# Dynamics (continuous):
#   ṗ = v
#   v̇ = g - beta * ||v|| * v
# Discrete-time propagation uses RK4 for the mean, and a first-order EKF covariance update:
#   x_{k+1} ≈ x_k + dt * f(x_k)
#   F_k ≈ I + dt * ∂f/∂x |_{x̂_k}
#
# Measurement model: position only, z = H x + noise, with H = [I3  03x3]
#
# Usage in your loop:
#   from ekf_drag import EKF3DDrag
#   ekf = EKF3DDrag(dt=1/90, g=(0,9.81,0), beta_drag=0.02)
#   ekf.initialize(p_xyz=first_point, v_xyz=None)
#   ekf.predict()
#   ekf.update_xyz(measured_point, gate_alpha=0.99)   # optional gating

from __future__ import annotations
import numpy as np

_EPS = 1e-9

def _accel_with_drag(v: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
    """Compute acceleration g - beta*||v||*v"""
    speed = np.linalg.norm(v)
    if speed > _EPS:
        return g - beta * speed * v
    else:
        return g.copy()

def _jacobian_dvdt_dv(v: np.ndarray, beta: float) -> np.ndarray:
    """
    Jacobian ∂(v̇)/∂v for a = g - beta * ||v|| * v
      ∂a/∂v = -beta * ( ||v|| I + (v v^T) / ||v|| )
    Handle ||v|| ~ 0 with a small epsilon.
    """
    speed = np.linalg.norm(v)
    if speed < _EPS:
        # Near zero speed, approximate with -beta * ||v|| I ≈ 0
        return np.zeros((3, 3))
    I = np.eye(3)
    vvT = np.outer(v, v)
    return -beta * (speed * I + vvT / speed)

def _f(x: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
    """Continuous-time dynamics f(x) = [v; g - beta*||v||*v]"""
    v = x[3:]
    a = _accel_with_drag(v, g, beta)
    return np.hstack((v, a))

def _rk4_step(x: np.ndarray, dt: float, g: np.ndarray, beta: float) -> np.ndarray:
    """RK4 step for the nonlinear mean propagation."""
    k1 = _f(x, g, beta)
    k2 = _f(x + 0.5*dt*k1, g, beta)
    k3 = _f(x + 0.5*dt*k2, g, beta)
    k4 = _f(x + dt*k3, g, beta)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

class EKF3DDrag:
    """
    EKF for ballistic motion with quadratic drag.

    Parameters
    ----------
    dt : float
        Sampling period (s), e.g., 1/90 for D435 depth @ 90 FPS.
    g : array-like(3,)
        Gravity in the SAME frame as your measurements (e.g., RealSense camera frame).
        Example: (0, 9.81, 0) for x:right, y:down, z:forward.
    beta_drag : float
        Quadratic drag coefficient (1/m). Start ~0.01–0.05 and tune.
    Q_pos, Q_vel : float
        Process noise variances for position and velocity (diagonal Q).
    R_xyz : tuple of 3 floats
        Measurement noise variances (m^2) for x,y,z.
    init_pos_var, init_vel_var : float
        Initial covariance diagonals for pos/vel.

    Notes
    -----
    - Uses RK4 for the state mean to reduce discretization error.
    - Uses first-order linearization for covariance: F ≈ I + dt * df/dx.
    - Measurement model: z = H x + noise, H = [I3  03x3].
    - Optional Mahalanobis gating in update_xyz().
    """
    def __init__(
        self,
        dt: float = 1/90.0,
        g: np.ndarray | list = (0.0, 9.81, 0.0),
        beta_drag: float = 0.02,
        Q_pos: float = 1e-5,
        Q_vel: float = 1e-3,
        R_xyz: tuple[float, float, float] = (5e-4, 5e-4, 1e-3),
        init_pos_var: float = 1e-2,
        init_vel_var: float = 1e-1,
    ):
        self.n = 6
        self.dt = float(dt)
        self.g = np.asarray(g, dtype=float).reshape(3)
        self.beta_drag = float(beta_drag)

        # State and covariance
        self.x = np.zeros(self.n, dtype=float)
        self.P = np.diag([init_pos_var]*3 + [init_vel_var]*3).astype(float)

        # Process/measurement noise
        self.Q = np.diag([Q_pos]*3 + [Q_vel]*3).astype(float)
        self.R = np.diag(R_xyz).astype(float)

        # Measurement matrix H (position only)
        self.H = np.zeros((3, self.n), dtype=float)
        self.H[0,0] = self.H[1,1] = self.H[2,2] = 1.0

        # Identity
        self.I = np.eye(self.n, dtype=float)

    # -------- API --------

    def initialize(self, p_xyz: np.ndarray, v_xyz: np.ndarray | None = None):
        p = np.asarray(p_xyz, dtype=float).reshape(3)
        v = np.zeros(3, dtype=float) if v_xyz is None else np.asarray(v_xyz, dtype=float).reshape(3)
        self.x[:3] = p
        self.x[3:] = v

    def set_noise(self, Q_pos: float | None = None, Q_vel: float | None = None,
                  R_xyz: tuple[float, float, float] | None = None):
        if Q_pos is not None:
            for i in range(3): self.Q[i, i] = Q_pos
        if Q_vel is not None:
            for i in range(3,6): self.Q[i, i] = Q_vel
        if R_xyz is not None:
            self.R = np.diag(R_xyz).astype(float)

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """
        EKF time update.
        - Mean: RK4(x, dt) under drag dynamics.
        - Covariance: P <- F P F^T + Qd  (with F ≈ I + dt*df/dx)
          Here Qd is taken as continuous Q applied discretely (diagonal, tunable).
        """
        dt = self.dt
        # Mean propagation
        x_prev = self.x.copy()
        self.x = _rk4_step(self.x, dt, self.g, self.beta_drag)

        # Jacobian df/dx evaluated at x_prev (or midpoint; x_prev is standard)
        # f(x) = [ v ; g - beta*||v||*v ], df/dx = [[0 I],[0 J]]
        F = np.eye(self.n)
        # upper-right block: ∂ṗ/∂v = I
        F[0:3, 3:6] = dt * np.eye(3)
        # lower-right block: ∂v̇/∂v = J(v)
        v_prev = x_prev[3:]
        J = _jacobian_dvdt_dv(v_prev, self.beta_drag)      # 3x3
        F[3:6, 3:6] += dt * J

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update_xyz(self, z_xyz: np.ndarray, gate_alpha: float | None = None):
        """
        EKF measurement update with 3D position (metres). Optional Mahalanobis gating.

        Returns
        -------
        x, P, d2, thresh : ndarray, ndarray, float|None, float|None
            Filter state/cov after (possible) update, plus gating stats if gating applied.
        """
        z = np.asarray(z_xyz, dtype=float).reshape(3)
        H = self.H
        R = self.R

        # Innovation
        z_pred = H @ self.x
        y = z - z_pred

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Optional Mahalanobis gate
        d2 = thresh = None
        if gate_alpha is not None:
            try:
                d2 = float(y.reshape(1,3) @ np.linalg.solve(S, y.reshape(3,1)))
            except np.linalg.LinAlgError:
                d2 = float("inf")
            # Chi-square thresholds for df=3
            if gate_alpha >= 0.997:       thresh = 14.160  # ~99.7%
            elif gate_alpha >= 0.99:      thresh = 11.345  # ~99%
            else:                          thresh = 7.815   # ~95%
            if d2 > thresh:
                # Reject update; keep prediction
                return self.x.copy(), self.P.copy(), d2, thresh

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S + 1e-9*np.eye(3))

        # State & covariance update
        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P
        return self.x.copy(), self.P.copy(), d2, thresh

    # -------- Utilities --------

    def simulate_trajectory(self, T: float, step: float | None = None) -> np.ndarray:
        """Integrate mean dynamics forward for visualization/planning. Returns (N,3) positions."""
        if step is None:
            step = self.dt
        steps = max(1, int(np.ceil(T / step)))
        out = np.zeros((steps + 1, 3), dtype=float)
        xi = self.x.copy()
        out[0] = xi[:3]
        for k in range(1, steps + 1):
            xi = _rk4_step(xi, step, self.g, self.beta_drag)
            out[k] = xi[:3]
        return out

    def predict_intercept_with_plane(
        self, n: np.ndarray, d: float, t_max: float = 2.0, step: float | None = None
    ):
        """
        Find first intercept of mean trajectory with plane n·p + d = 0 within t_max.
        Returns (t_hit, p_hit) or (None, None).
        """
        if step is None:
            step = self.dt
        n = np.asarray(n, dtype=float).reshape(3)

        def phi(p):
            return float(n @ p + d)

        t = 0.0
        xi = self.x.copy()
        p_prev = xi[:3].copy()
        phi_prev = phi(p_prev)

        while t < t_max:
            x_next = _rk4_step(xi, step, self.g, self.beta_drag)
            t_next = t + step
            p_next = x_next[:3]
            phi_next = phi(p_next)

            if abs(phi_prev) < _EPS:
                return t, p_prev
            if phi_prev * phi_next < 0.0:
                # Bisection refine between t and t_next
                a_t, a_x = t, xi.copy()
                b_t, b_x = t_next, x_next.copy()
                for _ in range(24):
                    m_t = 0.5 * (a_t + b_t)
                    m_x = _rk4_step(a_x, (m_t - a_t), self.g, self.beta_drag)
                    m_p = m_x[:3]
                    if phi_prev * phi(m_p) <= 0.0:
                        b_t, b_x = m_t, m_x
                    else:
                        a_t, a_x = m_t, m_x
                t_hit = 0.5 * (a_t + b_t)
                p_hit = _rk4_step(self.x.copy(), t_hit, self.g, self.beta_drag)[:3]
                return t_hit, p_hit

            # advance
            t, xi = t_next, x_next
            p_prev, phi_prev = p_next, phi_next

        return None, None
    

# ======================= EKF (with drag) ======================
# _EPS = 1e-9

# def _accel_with_drag(v: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
# 	s = np.linalg.norm(v)
# 	return g - beta * s * v if s > _EPS else g.copy()

# def _jacobian_dvdt_dv(v: np.ndarray, beta: float) -> np.ndarray:
# 	s = np.linalg.norm(v)
# 	if s < _EPS:
# 		return np.zeros((3,3))
# 	I = np.eye(3)
# 	return -beta * (s * I + np.outer(v, v) / s)

# def _f(x: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
# 	v = x[3:]
# 	a = _accel_with_drag(v, g, beta)
# 	return np.hstack((v, a))

# def _rk4_step(x: np.ndarray, dt: float, g: np.ndarray, beta: float) -> np.ndarray:
# 	k1 = _f(x, g, beta)
# 	k2 = _f(x + 0.5*dt*k1, g, beta)
# 	k3 = _f(x + 0.5*dt*k2, g, beta)
# 	k4 = _f(x + dt*k3, g, beta)
# 	return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# class EKF3DDrag:
# 	"""x=[px,py,pz,vx,vy,vz]. Process: ṗ=v, v̇=g-β||v||v. Measurement: position (m)."""
# 	def __init__(self, dt=EKF_DT, g=G_VECTOR, beta_drag=BETA_DRAG, Q_pos=Q_POS, Q_vel=Q_VEL, R_xyz=R_XYZ):
# 		self.n = 6
# 		self.dt = float(dt)
# 		self.g = np.asarray(g, float).reshape(3)
# 		self.beta_drag = float(beta_drag)
# 		self.x = np.zeros(self.n, float)
# 		self.P = np.diag([1e-2]*3 + [1e-1]*3).astype(float)
# 		self.Q = np.diag([Q_pos]*3 + [Q_vel]*3).astype(float)
# 		self.R = np.diag(R_xyz).astype(float)
# 		self.H = np.zeros((3, self.n), float); self.H[0,0]=self.H[1,1]=self.H[2,2]=1.0
# 		self.I = np.eye(self.n).astype(float)

# 	def initialize(self, p_xyz, v_xyz=None):
# 		p = np.asarray(p_xyz, float).reshape(3)
# 		v = np.zeros(3, float) if v_xyz is None else np.asarray(v_xyz, float).reshape(3)
# 		self.x[:3], self.x[3:] = p, v

# 	def predict(self):
# 		dt = self.dt
# 		x_prev = self.x.copy()
# 		self.x = _rk4_step(self.x, dt, self.g, self.beta_drag)
# 		F = np.eye(self.n)
# 		F[0:3,3:6] = dt*np.eye(3)
# 		J = _jacobian_dvdt_dv(x_prev[3:], self.beta_drag)
# 		F[3:6,3:6] += dt*J
# 		self.P = F @ self.P @ F.T + self.Q
# 		return self.x.copy(), self.P.copy()

# 	def update_xyz(self, z_xyz, gate_alpha=GATE_ALPHA):
# 		z = np.asarray(z_xyz, float).reshape(3)
# 		H, R = self.H, self.R
# 		y = z - (H @ self.x)
# 		S = H @ self.P @ H.T + R
# 		if gate_alpha is not None:
# 			try:
# 				S_inv_y = np.linalg.solve(S, y.reshape(3,1))
# 				d2 = (y.reshape(1,3) @ S_inv_y).item()
# 			except np.linalg.LinAlgError:
# 				d2 = float("inf")
# 			# rough chisq thresholds for 3 dof
# 			thresh = 11.345 if gate_alpha >= 0.99 else 7.815
# 			if gate_alpha >= 0.997: thresh = 14.160
# 			if d2 > thresh:
# 				return self.x.copy(), self.P.copy(), d2, thresh
# 		try:
# 			K = self.P @ H.T @ np.linalg.inv(S)
# 		except np.linalg.LinAlgError:
# 			K = self.P @ H.T @ np.linalg.inv(S + 1e-9*np.eye(3))
# 		self.x = self.x + K @ y
# 		self.P = (self.I - K @ H) @ self.P
# 		return self.x.copy(), self.P.copy(), None, None

# 	def predict_intercept_with_plane(self, n, d, t_max=2.0, step=None):
# 		if step is None: step = self.dt
# 		n = np.asarray(n, float).reshape(3)
# 		def phi(p): return float(n @ p + d)
# 		t = 0.0; xi = self.x.copy()
# 		p_prev = xi[:3].copy(); phi_prev = phi(p_prev)
# 		while t < t_max:
# 			x_next = _rk4_step(xi, step, self.g, self.beta_drag)
# 			t_next = t + step
# 			p_next = x_next[:3]; phi_next = phi(p_next)
# 			if abs(phi_prev) < _EPS: return t, p_prev
# 			if phi_prev * phi_next < 0.0:
# 				a_t, a_x = t, xi.copy()
# 				b_t, b_x = t_next, x_next.copy()
# 				for _ in range(24):
# 					m_t = 0.5*(a_t+b_t)
# 					m_x = _rk4_step(a_x, (m_t-a_t), self.g, self.beta_drag)
# 					if phi_prev * phi(m_x[:3]) <= 0.0:
# 						b_t, b_x = m_t, m_x
# 					else:
# 						a_t, a_x = m_t, m_x
# 				t_hit = 0.5*(a_t+b_t)
# 				p_hit = _rk4_step(self.x.copy(), t_hit, self.g, self.beta_drag)[:3]
# 				return t_hit, p_hit
# 			t, xi = t_next, x_next
# 			p_prev, phi_prev = p_next, phi_next
# 		return None, None
# ===============================================================
