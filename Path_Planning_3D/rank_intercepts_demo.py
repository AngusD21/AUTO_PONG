#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank interception points along a 3D trajectory for a 6‑DOF arm (wx250s-ish).
- Uses simple DH model + numeric IK (Levenberg–Marquardt with finite‑difference Jacobian).
- Ranks each trajectory point by how much joint motion (Δθ from current/home) would be needed to reach it.
- Two orientation modes:
    1) "look_at": tool's +Z axis points opposite to the ball's velocity (i.e., facing the ball)
    2) "upright": tool's +Z axis points straight up (fixed pose)

This is a standalone demo: it generates synthetic trajectories and plots the
trajectory colored by the ranking score. Intended to slot into your pipeline later.
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional

# -------------------------
# Robot model (approximate)
# -------------------------

# DH row: (a, alpha_deg, d, theta_deg) with theta provided by joints
# These values were adapted from your notes; they are NOT guaranteed to be exact.
# Units: meters / radians internally (we convert degrees where relevant).
LINKS_MM = (110.25, 254.95, 171.55, 78.45, 65.0, 109.15)
LINKS_M  = tuple([x/1000.0 for x in LINKS_MM])
BETA_DEG = 11.3

# Joint limits (deg) — approximate, adjust to real robot as needed
JOINT_LIMITS_DEG = np.array([
    [-3.14159,  3.14159],   # j1
    [ -1.88496, 1.98968],   # j2
    [-2.14675,  1.60571],   # j3
    [-3.14159,  3.14159],   # j4
    [-1.74533,  2.14675],   # j5
    [-3.14159,  3.14159],   # j6
], dtype=float)

# Home (deg) — from your project, gripper facing down
HOME_DEG = np.array([0.0, 20.5, -18.45, 0.0, -95.0, 0.0], dtype=float)

# Step size for numeric Jacobian (radians)
FD_EPS = 1e-4

@dataclass
class IKConfig:
    max_iters: int = 200
    pos_w: float = 1.0     # weight for position error [m]
    ori_w: float = 0.3     # weight for orientation (tool z-axis) error [unitless]
    lm_damping: float = 1e-2
    step_clip: float = 0.2 # max |Δθ| per iter [rad]
    tol_pos: float = 1e-3  # [m]
    tol_ori: float = 3e-2  # ~1.7deg in z-axis difference-norm


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard DH transform (a, alpha, d, theta) all in radians/meters."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ ct, -st*ca,  st*sa, a*ct ],
        [ st,  ct*ca, -ct*sa, a*st ],
        [  0,     sa,     ca,    d  ],
        [  0,      0,      0,    1  ],
    ], dtype=float)


def fk_wx250s(j_deg: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Forward kinematics to tool tip.
    Returns (T_0_ee, frames), where 'frames' are the cumulative transforms of each joint (including base).
    Angles in DEGREES.
    """
    j = np.asarray(j_deg, dtype=float)
    th1, th2, th3, th4, th5, th6 = np.deg2rad(j)

    # DH table (a, alpha, d, theta) in radians/meters
    a1, a2, a3, a4, a5, a6 = LINKS_M[0], LINKS_M[1], LINKS_M[2], LINKS_M[3], LINKS_M[4], LINKS_M[5]
    beta = math.radians(BETA_DEG)

    # Following the table you shared (may not be exact to manufacturer)
    rows = [
        (0.0,              0.0,           a1,       th1),
        (0.0,       math.pi/2,            0.0, math.radians(90.0) + th2 - beta),
        (a2,               0.0,           0.0,       th3 + beta),
        (0.0,       math.pi/2,    (a3 + a4),       th4),
        (0.0,      -math.pi/2,          0.0,       th5),
        (0.0,       math.pi/2,           a5,       th6),
    ]

    T = np.eye(4)
    frames = [T.copy()]  # base
    for (a, alpha, d, th) in rows:
        T = T @ dh_transform(a, alpha, d, th)
        frames.append(T.copy())

    # Final tool offset along +Z of the last frame (a6)
    T = T @ dh_transform(0.0, 0.0, a6, 0.0)
    frames.append(T.copy())
    return T, frames


def tool_z_axis(T: np.ndarray) -> np.ndarray:
    """Return the tool's +Z axis (3,) from a 4x4 pose matrix."""
    return T[:3, 2] / (np.linalg.norm(T[:3, 2]) + 1e-12)


def numeric_jacobian(j_deg: np.ndarray, pos_w: float, ori_w: float, z_des: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finite-difference Jacobian for [ position; tool_z ] wrt joints.
    Returns (J 6x6, p 3, z 3) evaluated at j_deg.
    """
    T, _ = fk_wx250s(j_deg)
    p = T[:3, 3].copy()
    z = tool_z_axis(T)

    J = np.zeros((6, 6), dtype=float)
    base = j_deg.astype(float)
    for i in range(6):
        j2 = base.copy()
        j2[i] += np.rad2deg(FD_EPS)  # perturb in degrees for fk
        T2, _ = fk_wx250s(j2)
        p2 = T2[:3, 3]; z2 = tool_z_axis(T2)
        J[:3, i] = (p2 - p) / FD_EPS
        J[3:, i] = (z2 - z) / FD_EPS

    # Apply weights to rows to balance position/orientation
    W = np.diag([pos_w, pos_w, pos_w, ori_w, ori_w, ori_w])
    return W @ J, p, z


def ik_to_pose(
    p_des: np.ndarray,
    z_des: np.ndarray,
    j0_deg: np.ndarray,
    limits_deg: np.ndarray = JOINT_LIMITS_DEG,
    cfg: IKConfig = IKConfig(),
) -> Tuple[bool, np.ndarray, dict]:
    """
    Solve IK driving position to p_des and aligning tool +Z to z_des (no roll/yaw objective).
    Returns (success, j_deg, info).
    """
    j = j0_deg.astype(float).copy()
    # Clamp to limits initially
    j = np.minimum(np.maximum(j, limits_deg[:,0]), limits_deg[:,1])

    info = {"iters": 0, "err_pos": None, "err_ori": None}

    for it in range(cfg.max_iters):
        J, p, z = numeric_jacobian(j, cfg.pos_w, cfg.ori_w, z_des)
        e_pos = p - p_des
        e_ori = z - z_des
        r = np.concatenate([cfg.pos_w * e_pos, cfg.ori_w * e_ori])

        err_pos = float(np.linalg.norm(e_pos))
        err_ori = float(np.linalg.norm(e_ori))
        info["err_pos"], info["err_ori"] = err_pos, err_ori
        info["iters"] = it+1

        if (err_pos <= cfg.tol_pos) and (err_ori <= cfg.tol_ori):
            return True, j, info

        # LM step: (J^T J + λ I) Δθ = - J^T r
        JT = J.T
        H = JT @ J + (cfg.lm_damping**2) * np.eye(6)
        g = JT @ r
        try:
            dq = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dq = -np.linalg.pinv(H) @ g

        # Clip step & integrate (angles are in DEGREES in 'j', dq is in RADIANS because J is in m/rad)
        # Convert dq(rad) -> degrees:
        dq_deg = np.rad2deg(dq)
        # limit per-iter magnitude
        m = np.max(np.abs(dq_deg))
        if m > np.rad2deg(cfg.step_clip):
            dq_deg *= (np.rad2deg(cfg.step_clip) / (m + 1e-12))

        j += dq_deg
        # clamp to limits each iter
        j = np.minimum(np.maximum(j, limits_deg[:,0]), limits_deg[:,1])

    return False, j, info


# -------------------------
# Trajectory generation
# -------------------------

def ballistic_trajectory(p0: np.ndarray, v0: np.ndarray, g: np.ndarray, T: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple ballistic path: p(t) = p0 + v0 t + 0.5 g t^2. Returns (pts Nx3, vels Nx3)."""
    ts = np.arange(0.0, T + 1e-9, dt)
    pts = p0[None,:] + v0[None,:]*ts[:,None] + 0.5*g[None,:]*(ts[:,None]**2)
    vels = v0[None,:] + g[None,:]*ts[:,None]
    return pts, vels


def smooth_line(p_start: np.ndarray, p_end: np.ndarray, T: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Linear path with constant velocity (for testing)."""
    ts = np.arange(0.0, T + 1e-9, dt)
    v = (p_end - p_start) / max(1e-9, T)
    pts = p_start[None,:] + v[None,:]*ts[:,None]
    vels = np.tile(v, (len(ts), 1))
    return pts, vels


# -------------------------
# Ranking and visualization
# -------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def rank_along_trajectory(
    pts: np.ndarray,
    vels: np.ndarray,
    mode: str,
    j_home_deg: np.ndarray,
    limits_deg: np.ndarray = JOINT_LIMITS_DEG,
    cfg: IKConfig = IKConfig(),
) -> dict:
    """
    For each point, solve IK with orientation mode and compute cost = Σ |Δθ_i| from HOME (deg).
    Returns dict with fields:
        feasible_mask, costs, best_idx, J_solutions (Nx6), err_pos, err_ori
    """
    N = pts.shape[0]
    costs = np.full(N, np.inf, dtype=float)
    feasible = np.zeros(N, dtype=bool)
    err_pos = np.full(N, np.nan); err_ori = np.full(N, np.nan)
    J_solutions = np.full((N, 6), np.nan)

    # We'll warm‑start with home, then use previous solution
    j_guess = j_home_deg.copy()

    for i in range(N):
        p = pts[i]
        v = vels[i]

        if mode == "look_at":
            z_des = normalize(-v)   # point tool +Z toward the ball (opposite velocity)
            if np.linalg.norm(v) < 1e-6:
                z_des = np.array([0, 0, 1.0])
        elif mode == "upright":
            z_des = np.array([0, 0, 1.0])
        else:
            raise ValueError("mode must be 'look_at' or 'upright'")

        ok, j_sol, info = ik_to_pose(p_des=p, z_des=z_des, j0_deg=j_guess, limits_deg=limits_deg, cfg=cfg)
        if ok:
            feasible[i] = True
            J_solutions[i,:] = j_sol
            costs[i] = float(np.sum(np.abs(j_sol - j_home_deg)))  # L1 in joint space (deg)
            err_pos[i] = info["err_pos"]; err_ori[i] = info["err_ori"]
            j_guess = j_sol  # warm‑start next
        else:
            # leave as infeasible / inf cost; keep last good guess to help later segments
            err_pos[i] = info["err_pos"]; err_ori[i] = info["err_ori"]

    best_idx = int(np.nanargmin(costs)) if np.any(np.isfinite(costs)) else -1
    return dict(
        feasible_mask=feasible,
        costs=costs,
        best_idx=best_idx,
        J_solutions=J_solutions,
        err_pos=err_pos,
        err_ori=err_ori,
    )


def plot_results(ax3d, pts: np.ndarray, result: dict, title: str):
    """Color trajectory by cost, gray for infeasible."""
    feasible = result["feasible_mask"]
    costs = result["costs"]
    best = result["best_idx"]

    # Normalize colors over feasible points only
    finite_costs = costs[np.isfinite(costs)]
    if finite_costs.size > 0:
        cmin, cmax = np.percentile(finite_costs, 5), np.percentile(finite_costs, 95)
        cmin = min(cmin, np.min(finite_costs))
        cmax = max(cmax, np.max(finite_costs))
    else:
        cmin, cmax = 0.0, 1.0

    colors = np.zeros((len(pts), 4))
    for i in range(len(pts)):
        if feasible[i]:
            val = (costs[i] - cmin) / (cmax - cmin + 1e-12)
            val = float(np.clip(val, 0.0, 1.0))
            colors[i] = plt.cm.viridis(1.0 - val)  # low cost -> yellow/green
        else:
            colors[i] = (0.6, 0.6, 0.6, 0.5)       # infeasible -> gray

    ax3d.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors, s=18, depthshade=False)
    if best >= 0:
        ax3d.scatter([pts[best,0]],[pts[best,1]],[pts[best,2]], marker='*', s=160, c='red', label='best')
        ax3d.legend(loc='upper right')

    ax3d.set_title(title)
    ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
    ax3d.set_box_aspect((1,1,1))
    # Equal-ish axes
    mins = pts.min(axis=0); maxs = pts.max(axis=0)
    centers = 0.5*(mins+maxs); spans = maxs-mins; r = 0.5*float(np.max(spans)+1e-6)
    ax3d.set_xlim(centers[0]-r, centers[0]+r)
    ax3d.set_ylim(centers[1]-r, centers[1]+r)
    ax3d.set_zlim(max(0.0, centers[2]-r), centers[2]+r)


def draw_workspace_hint(ax3d, r_max: float):
    """Draw a translucent sphere as a crude reachable-radius hint."""
    u = np.linspace(0, 2*np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    xs = r_max*np.outer(np.cos(u), np.sin(v))
    ys = r_max*np.outer(np.sin(u), np.sin(v))
    zs = r_max*np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_wireframe(xs, ys, zs, rstride=3, cstride=3, linewidth=0.3, alpha=0.25)


def demo():
    np.set_printoptions(precision=3, suppress=True)

    # ---- Current pose: custom HOME you provided ----
    j_home = HOME_DEG.copy()

    # ---- Synthetic trajectories (pick one) ----
    # Base frame convention assumed: Z up. Tune start points to your setup.
    # 1) Ballistic arc across the front of the robot
    p0 = np.array([0.35, -0.10, 0.75])      # [m]
    v0 = np.array([-0.6,  0.20, 0.10])      # [m/s]
    g  = np.array([0.0, 0.0, -9.81]) * 0.0  # set to 0 for "slow" arc in the lab
    T_total = 0.8; dt = 0.02
    pts1, vels1 = ballistic_trajectory(p0, v0, g, T_total, dt)

    # 2) Straight line sweep (alternative)
    pA = np.array([0.50,  0.00, 0.15])
    pB = np.array([0.10, -0.20, 0.35])
    pts2, vels2 = smooth_line(pA, pB, T_total, dt)

    # Choose which to run
    # pts, vels = pts1, vels1
    pts, vels = pts2, vels2

    # ---- Solve & rank for both orientation modes ----
    cfg = IKConfig(max_iters=180, pos_w=1.0, ori_w=0.4, lm_damping=1e-2, step_clip=0.2, tol_pos=1e-3, tol_ori=2e-2)

    res_look  = rank_along_trajectory(pts, vels, mode='look_at', j_home_deg=j_home, cfg=cfg)
    res_up    = rank_along_trajectory(pts, vels, mode='upright',  j_home_deg=j_home, cfg=cfg)

    # ---- Plots ----
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')

    # Crude workspace hint (max reach ~ sum of main link lengths)
    r_max = LINKS_M[1] + LINKS_M[2] + LINKS_M[3] + LINKS_M[5]
    for ax in (ax1, ax2):
        draw_workspace_hint(ax, r_max=r_max)

    plot_results(ax1, pts, res_look, title="Ranking: look_at (tool +Z toward ball)")
    plot_results(ax2, pts, res_up,   title="Ranking: upright (tool +Z up)")
    plt.tight_layout()
    plt.show()

    # ---- Console summary ----
    for name, res in [("look_at", res_look), ("upright", res_up)]:
        feasible_count = int(np.sum(res["feasible_mask"]))
        best_idx = res["best_idx"]
        print(f"\nMode: {name}")
        print(f"  feasible points: {feasible_count}/{len(pts)}")
        if best_idx >= 0:
            print(f"  best idx: {best_idx}  position={pts[best_idx]}  cost(sum|Δθ| from HOME)={res['costs'][best_idx]:.1f} deg")
            print(f"  joints at best (deg): {np.round(res['J_solutions'][best_idx], 2)}")
        else:
            print("  no feasible IK solutions found.")

if __name__ == "__main__":
    demo()
