# smart_planner.py
from __future__ import annotations
import math, numpy as np
from dataclasses import dataclass

@dataclass
class Limits:
    vmax_axis: float = 300.0   # mm/s (per-axis normal cap)
    amax_fast: float = 4000.0  # mm/s^2 (fast accel for SMART)
    motor_vmax_mm_s: float = 600.0  # physical motor linear cap (belt/gear limit)

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

def choose_T(p0, v0, p1, v1, lim: Limits):
    dx = p1[0]-p0[0]; dy = p1[1]-p0[1]
    dist = math.hypot(dx, dy)
    v_allow = max(50.0, min(lim.vmax_axis, 0.9*lim.motor_vmax_mm_s))
    t_pos = dist / max(1.0, 0.6*v_allow)
    dvx = abs(v1[0]-v0[0]); dvy = abs(v1[1]-v0[1])
    t_vel = max(dvx, dvy) / max(1.0, 0.8*lim.amax_fast)
    return min(30.0, max(0.25, t_pos, t_vel))

def hermite_sample(p0, v0, p1, v1, T, N=320):
    def axis(p0, v0, p1, v1):
        def f(t):
            tau = max(0.0, min(1.0, t/T))
            h00 =  2*tau**3 - 3*tau**2 + 1
            h10 =      tau**3 - 2*tau**2 + tau
            h01 = -2*tau**3 + 3*tau**2
            h11 =      tau**3 -    tau**2
            return h00*p0 + h10*(T*v0) + h01*p1 + h11*(T*v1)
        return f
    fx = axis(p0[0], v0[0], p1[0], v1[0])
    fy = axis(p0[1], v0[1], p1[1], v1[1])
    ts = np.linspace(0.0, T, N+1)
    xs = np.array([fx(t) for t in ts]); ys = np.array([fy(t) for t in ts])
    return ts, xs, ys

def motor_sdot_limit(tx, ty, v_motor):
    # Mapping from path tangent to motor sum/diff limits like in the sim
    denom = np.maximum(np.maximum(np.abs(tx - ty), np.abs(tx + ty)), 1e-9)
    return v_motor / denom

def time_optimal(ts, xs, ys, v0_mag, v1_mag, lim: Limits):
    dx = np.diff(xs); dy = np.diff(ys)
    ds = np.hypot(dx, dy); ds[ds < 1e-9] = 1e-9
    tx = dx / ds; ty = dy / ds

    sdot_max_seg = motor_sdot_limit(tx, ty, lim.motor_vmax_mm_s)
    v_max_nodes = np.zeros(len(xs))
    v_max_nodes[0]  = sdot_max_seg[0]
    v_max_nodes[-1] = sdot_max_seg[-1]
    if len(xs) > 2:
        v_max_nodes[1:-1] = np.minimum(sdot_max_seg[:-1], sdot_max_seg[1:])

    v_nodes = np.zeros_like(v_max_nodes)
    v_nodes[0]  = min(v0_mag,  v_max_nodes[0])
    v_nodes[-1] = min(v1_mag,  v_max_nodes[-1])

    A = max(50.0, lim.amax_fast)

    # Forward pass
    for i in range(len(ds)):
        v_allow = math.sqrt(v_nodes[i]*v_nodes[i] + 2.0*A*ds[i])
        v_nodes[i+1] = min(v_allow, v_max_nodes[i+1])

    # Enforce end speed
    v_nodes[-1] = min(v_nodes[-1], v1_mag)

    # Backward pass
    for i in range(len(ds)-1, -1, -1):
        v_allow = math.sqrt(v_nodes[i+1]*v_nodes[i+1] + 2.0*A*ds[i])
        v_nodes[i] = min(v_nodes[i], v_allow, v_max_nodes[i])

    # Convert to time stamps
    dt_seg = 2.0 * ds / np.maximum(v_nodes[:-1] + v_nodes[1:], 1e-9)
    t_nodes = np.zeros_like(v_nodes); t_nodes[1:] = np.cumsum(dt_seg)
    return t_nodes, v_nodes

def plan_smart(p0, v0, p1, v1, lim: Limits, dt_stream: float = 0.02):
    """
    Returns a schedule of (t, x_mm, y_mm) waypoints for streaming every ~dt_stream seconds.
    """
    T = choose_T(p0, v0, p1, v1, lim)
    ts_h, xs, ys = hermite_sample(p0, v0, p1, v1, T)
    v0_mag = math.hypot(v0[0], v0[1]); v1_mag = math.hypot(v1[0], v1[1])
    t_nodes, v_nodes = time_optimal(ts_h, xs, ys, v0_mag, v1_mag, lim)

    # Resample to ~dt_stream
    T_total = float(t_nodes[-1])
    ts = np.arange(0.0, T_total + 1e-6, dt_stream)
    # linear interpolation on nodes
    xs_i = np.interp(ts, t_nodes, xs)
    ys_i = np.interp(ts, t_nodes, ys)

    schedule = [(float(t), float(x), float(y)) for t, x, y in zip(ts, xs_i, ys_i)]
    return schedule, T_total
