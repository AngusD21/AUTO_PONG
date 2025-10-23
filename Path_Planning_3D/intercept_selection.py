import numpy as np
from wx250s_kinematics import ik, fk   # uses your DH + IK routine  (HTM in mm, joints in deg)

def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps)

def make_pose_mm(p_m, v_mps, mode="face", up=np.array([0.0, 0.0, 1.0])):
    """Return 4x4 HTM (mm) with orientation per mode."""
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(p_m, float) * 1000.0  # m -> mm

    if mode == "up" or np.linalg.norm(v_mps) < 1e-6:
        R = np.eye(3)
    else:
        z = -_normalize(np.asarray(v_mps, float))           # tool-Z faces ball
        x = np.cross(up, z)
        if np.linalg.norm(x) < 1e-6:                        # fallback if colinear
            x = np.cross(np.array([0.0, 1.0, 0.0]), z)
        x = _normalize(x)
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])

    T[:3, :3] = R
    return T

def within_limits(q_deg, q_min=None, q_max=None):
    if q_min is None or q_max is None:
        return True
    q = np.asarray(q_deg, float)
    return np.all(q >= np.asarray(q_min)) and np.all(q <= np.asarray(q_max))

def rank_intercepts(traj_samples, q0_deg, mode="face",
                    q_min=None, q_max=None, qd_max=None):
    """
    traj_samples: list of (t, p_m, v_mps)
    q0_deg: 6-vector in degrees
    Returns: list of dicts per sample with fields:
      feasible, t, p, q_sol, score
    """
    results = []
    q_seed = np.array(q0_deg, float)

    for (t, p, v) in traj_samples:
        T_goal = make_pose_mm(p, v, mode=mode)
        q_sol = ik(q_seed.tolist(), T_goal)   # uses your solver (deg input/output)
        feasible = q_sol is not None

        score = np.inf
        if feasible:
            q_sol = np.asarray(q_sol, float)
            if not within_limits(q_sol, q_min, q_max):
                feasible = False
            else:
                dq = np.abs(q_sol - q_seed)
                if qd_max is not None:
                    score = np.max(dq / (np.asarray(qd_max, float) + 1e-9))
                else:
                    score = np.linalg.norm(dq)
                q_seed = q_sol  # warm-start next point for continuity

        results.append({
            "t": float(t),
            "p": np.asarray(p, float),
            "q_sol": q_sol if feasible else None,
            "feasible": feasible,
            "score": float(score)
        })

    return results

def score_to_rgb(score, s_good=0.0, s_bad=1.0):
    """Green (good) → Red (bad). score in [s_good, s_bad]."""
    if not np.isfinite(score):
        return (0.6, 0.6, 0.6)  # infeasible: gray
    a = (score - s_good) / max(1e-9, s_bad - s_good)
    a = float(np.clip(a, 0.0, 1.0))
    return (a, 1.0 - a, 0.0)
