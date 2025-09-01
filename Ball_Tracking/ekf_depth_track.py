#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depth-only ball tracker (Intel RealSense D435 @ 90 FPS) with:
- EKF + quadratic drag (gravity-aware),
- plane-intercept prediction,
- optional Arduino serial output,
- and REVIEW CAPTURE: ring-buffered frames + post-run overlay video and metadata.

Controls:
  f  -> toggle live UI (fast mode off/on)
  r  -> reset (drop lock & revert to global detect)
  q  -> quit

Tip: leave UI OFF for performance, rely on the review video afterwards.
"""

# =========================== CONFIG ===========================

SEND_TO_ARDUINO = False         # True: send to Arduino, False: print (x_mm,y_mm,t_ms) to terminal
SERIAL_PORT = "COM6"            # e.g., "/dev/tty.usbmodem1101" on macOS
SERIAL_BAUD = 115200

# Review capture (ring buffer)
ENABLE_REVIEW_CAPTURE = True
PREBUFFER_SEC = 1.0             # keep this many seconds BEFORE first detection
POSTBUFFER_SEC = 1.0            # save this many seconds AFTER tracking is lost
CAPTURE_KEEP_EVERY = 2          # save every Nth frame (2 halves memory/IO, 3 thirds it, etc.)
REVIEW_OUT_DIR = "captures"     # directory to save mp4 + metadata
REVIEW_PATH_HORIZON_S = 0.8     # draw predicted path this far into the future on each frame

# Catch/intercept plane (n·p + d = 0). For z = const plane:
CATCH_PLANE_Z_M = 1.20
PLANE_NORMAL = (0.0, 0.0, 1.0)
PLANE_D = -CATCH_PLANE_Z_M

# Gravity vector in the SAME frame as your depth points (D435 depth frame: x right, y down, z forward)
G_VECTOR = (0.0, 9.81, 0.0)

# Drag coefficient beta (1/m). Start ~0.01–0.05 and tune.
BETA_DRAG = 0.02

# Detection
DEPTH_TOLERANCE_MM = 15         # depth band for focused detection
SEARCH_RADIUS_PX = 200
INIT_MIN_RADIUS_PX = 5
INIT_SEARCH_CIRCLE_PX = 70

# EKF tuning
EKF_DT = 1/90.0                 # 90 FPS depth stream
Q_POS = 1e-5                    # process noise on pos (m^2)
Q_VEL = 1e-3                    # process noise on vel ((m/s)^2)
R_XYZ = (5e-4, 5e-4, 1.8e-3)    # meas noise (m^2). Z usually noisier.
GATE_ALPHA = 0.99               # Mahalanobis gate (None to disable)

# --- IPC publish rate limit (seconds) ---
IPC_MIN_PERIOD_S = 1.0 / 30.0

# =============================================================

import os, json, time, math
import numpy as np
import cv2
import pyrealsense2 as rs
from collections import deque
from datetime import datetime

import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# ---------- Optional serial ----------
try:
    import serial
except Exception:
    serial = None

try:
    from Communications.ipc import ControlSubscriber, InterceptPublisher
except Exception as e:
    print(f"[IPC] Warning: could not import Communications.ipc ({e}). Running without IPC.")
    ControlSubscriber = None
    InterceptPublisher = None


# ======================= EKF (with drag) =======================

_EPS = 1e-9

def _accel_with_drag(v: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
    s = np.linalg.norm(v)
    return g - beta * s * v if s > _EPS else g.copy()

def _jacobian_dvdt_dv(v: np.ndarray, beta: float) -> np.ndarray:
    s = np.linalg.norm(v)
    if s < _EPS:
        return np.zeros((3,3))
    I = np.eye(3)
    return -beta * (s * I + np.outer(v, v) / s)

def _f(x: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
    v = x[3:]
    a = _accel_with_drag(v, g, beta)
    return np.hstack((v, a))

def _rk4_step(x: np.ndarray, dt: float, g: np.ndarray, beta: float) -> np.ndarray:
    k1 = _f(x, g, beta)
    k2 = _f(x + 0.5*dt*k1, g, beta)
    k3 = _f(x + 0.5*dt*k2, g, beta)
    k4 = _f(x + dt*k3, g, beta)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

class EKF3DDrag:
    """State x=[px,py,pz,vx,vy,vz]. Process: ṗ=v, v̇=g-β||v||v. Measurement: position (m)."""
    def __init__(self, dt=EKF_DT, g=G_VECTOR, beta_drag=BETA_DRAG, Q_pos=Q_POS, Q_vel=Q_VEL, R_xyz=R_XYZ):
        self.n = 6
        self.dt = float(dt)
        self.g = np.asarray(g, float).reshape(3)
        self.beta_drag = float(beta_drag)
        self.x = np.zeros(self.n, float)
        self.P = np.diag([1e-2]*3 + [1e-1]*3).astype(float)
        self.Q = np.diag([Q_pos]*3 + [Q_vel]*3).astype(float)
        self.R = np.diag(R_xyz).astype(float)
        self.H = np.zeros((3, self.n), float); self.H[0,0]=self.H[1,1]=self.H[2,2]=1.0
        self.I = np.eye(self.n, float)

    def initialize(self, p_xyz, v_xyz=None):
        p = np.asarray(p_xyz, float).reshape(3)
        v = np.zeros(3, float) if v_xyz is None else np.asarray(v_xyz, float).reshape(3)
        self.x[:3], self.x[3:] = p, v

    def predict(self):
        dt = self.dt
        x_prev = self.x.copy()
        self.x = _rk4_step(self.x, dt, self.g, self.beta_drag)
        F = np.eye(self.n)
        F[0:3,3:6] = dt*np.eye(3)
        J = _jacobian_dvdt_dv(x_prev[3:], self.beta_drag)
        F[3:6,3:6] += dt*J
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update_xyz(self, z_xyz, gate_alpha=GATE_ALPHA):
        z = np.asarray(z_xyz, float).reshape(3)
        H, R = self.H, self.R
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        if gate_alpha is not None:
            try:
                d2 = float(y.reshape(1,3) @ np.linalg.solve(S, y.reshape(3,1)))
            except np.linalg.LinAlgError:
                d2 = float("inf")
            thresh = 11.345 if gate_alpha >= 0.99 else 7.815
            if gate_alpha >= 0.997: thresh = 14.160
            if d2 > thresh:
                return self.x.copy(), self.P.copy(), d2, thresh
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S + 1e-9*np.eye(3))
        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P
        return self.x.copy(), self.P.copy(), None, None

    def simulate_trajectory(self, T, step=None):
        if step is None: step = self.dt
        steps = max(1, int(math.ceil(T/step)))
        out = np.zeros((steps+1, 3), float)
        xi = self.x.copy(); out[0] = xi[:3]
        for k in range(1, steps+1):
            xi = _rk4_step(xi, step, self.g, self.beta_drag)
            out[k] = xi[:3]
        return out

    def predict_intercept_with_plane(self, n, d, t_max=2.0, step=None):
        if step is None: step = self.dt
        n = np.asarray(n, float).reshape(3)
        def phi(p): return float(n @ p + d)
        t = 0.0; xi = self.x.copy()
        p_prev = xi[:3].copy(); phi_prev = phi(p_prev)
        while t < t_max:
            x_next = _rk4_step(xi, step, self.g, self.beta_drag)
            t_next = t + step
            p_next = x_next[:3]; phi_next = phi(p_next)
            if abs(phi_prev) < _EPS: return t, p_prev
            if phi_prev * phi_next < 0.0:
                a_t, a_x = t, xi.copy()
                b_t, b_x = t_next, x_next.copy()
                for _ in range(24):
                    m_t = 0.5*(a_t+b_t)
                    m_x = _rk4_step(a_x, (m_t-a_t), self.g, self.beta_drag)
                    if phi_prev * phi(m_x[:3]) <= 0.0:
                        b_t, b_x = m_t, m_x
                    else:
                        a_t, a_x = m_t, m_x
                t_hit = 0.5*(a_t+b_t)
                p_hit = _rk4_step(self.x.copy(), t_hit, self.g, self.beta_drag)[:3]
                return t_hit, p_hit
            t, xi = t_next, x_next
            p_prev, phi_prev = p_next, phi_next
        return None, None

# ==================== Serial sender (optional) ====================

class InterceptSender:
    """Sends (x_mm, y_mm, t_ms) as ASCII: (TRACK,x_mm,y_mm,t_ms)\\n"""
    def __init__(self, enabled: bool, port: str, baud: int):
        self.enabled = bool(enabled) and (serial is not None)
        self.port = port; self.baud = baud; self.ser = None
        if self.enabled:
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
                time.sleep(0.1)
                print(f"[Serial] Opened {self.port} @ {self.baud}")
            except Exception as e:
                print(f"[Serial] ERROR opening {self.port}: {e}")
                self.enabled = False

    def send(self, x_m: float, y_m: float, t_s: float):
        x_mm = int(round(1000*x_m)); y_mm = int(round(1000*y_m))
        t_ms = max(0, int(round(1000*t_s)))
        msg = f"(TRACK,{x_mm},{y_mm},{t_ms})\n"
        if self.enabled and self.ser:
            try: self.ser.write(msg.encode("ascii"))
            except Exception as e: print(f"[Serial] write failed: {e} :: {msg.strip()}")
        else:
            print(f"[TRACK] x_mm={x_mm} y_mm={y_mm} t_ms={t_ms}")

    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception: pass

# ===================== Detection / rendering =====================

def display_depth_image(depth_image, depth_scale):
    depth_mm = depth_image * depth_scale * 1000.0
    norm = np.zeros_like(depth_mm, dtype=np.uint8)
    cv2.normalize(depth_mm, norm, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

def detect_circle_in_depth(depth_image, depth_scale):
    """Global detect: most circular blob in 0.5–3.0 m band. Returns ((cx,cy), r_px, depth_m) or None."""
    depth_mm = depth_image * depth_scale * 1000.0
    mask = ((depth_mm > 500) & (depth_mm < 3000)).astype(np.uint8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None; best_circ = 0.0; min_circ = 0.7
    for cnt in contours:
        if cv2.contourArea(cnt) < 50: continue
        (x,y), r = cv2.minEnclosingCircle(cnt)
        if r < INIT_MIN_RADIUS_PX: continue
        area = cv2.contourArea(cnt); per = cv2.arcLength(cnt, True)
        circ = 4*math.pi*area/(per*per) if per>0 else 0.0
        if circ < min_circ or circ <= best_circ: continue
        m = np.zeros_like(depth_image, np.uint8); cv2.drawContours(m,[cnt],0,255,-1)
        vals = depth_mm[m==255]; vals = vals[vals>0]
        if vals.size == 0: continue
        best = ((int(x),int(y)), int(r), float(np.median(vals))/1000.0)
        best_circ = circ
    return best

def depth_detection(depth_frame, depth_scale, ball_depth_mm, expected_radius_px, prediction_xy, fast_mode):
    """Focused detect around predicted pixel. Returns (center_xy, median_depth_mm) or (None, None)."""
    depth_img = np.asanyarray(depth_frame.get_data())
    depth_mm = depth_img * depth_scale * 1000.0

    band = (depth_mm > (ball_depth_mm-DEPTH_TOLERANCE_MM)) & (depth_mm < (ball_depth_mm+DEPTH_TOLERANCE_MM))
    depth_mask = (band.astype(np.uint8)*255)

    search_mask = np.zeros_like(depth_mask)
    px, py = int(np.clip(prediction_xy[0], 0, depth_mask.shape[1]-1)), int(np.clip(prediction_xy[1], 0, depth_mask.shape[0]-1))
    cv2.circle(search_mask, (px, py), SEARCH_RADIUS_PX, 255, -1)
    depth_mask = cv2.bitwise_and(depth_mask, search_mask)

    if not fast_mode:
        cv2.imshow("Depth Mask", depth_mask)
        cv2.imshow("Search Mask", search_mask)

    contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None; best_score = 1e9; found_center = None; found_depth_mm = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r <= 0: continue
        roundness = area / (math.pi*(r**2)) if r>0 else 0.0
        if roundness < 0.4: continue
        rdiff = abs(r - expected_radius_px)
        if rdiff < 10 and rdiff < best_score:
            best_score = rdiff; best_cnt = cnt; found_center = (int(x), int(y))

    if best_cnt is not None:
        m = np.zeros_like(depth_mask); cv2.drawContours(m,[best_cnt],-1,255,-1)
        vals = depth_mm[m==255]
        vals = vals[(vals>(ball_depth_mm-DEPTH_TOLERANCE_MM)) & (vals<(ball_depth_mm+DEPTH_TOLERANCE_MM))]
        if vals.size > 0:
            found_depth_mm = float(np.median(vals))
        return found_center, found_depth_mm

    return None, None

def predict_radius(calib_radius_px, calib_depth_mm, current_depth_mm):
    if current_depth_mm <= 0: return int(calib_radius_px)
    return int(calib_radius_px * (calib_depth_mm / current_depth_mm))

# ===================== Review capture logic =====================

class ReviewCapture:
    """
    Keeps a ring buffer of recent depth frames (+ per-frame metadata).
    On detection start: snapshot prebuffer.
    During tracking: record frames/metadata.
    After loss: record postbuffer, then render an MP4 with overlays + save meta.json.
    """
    def __init__(self, enabled: bool, fps: float, keep_every: int, depth_scale: float, intrinsics, out_dir: str):
        self.enabled = enabled
        self.keep_every = max(1, int(keep_every))
        self.depth_scale = float(depth_scale)
        self.intrinsics = intrinsics
        self.fps = float(fps) / self.keep_every
        self.pre_len = int(PREBUFFER_SEC * fps / self.keep_every)
        self.post_len = int(POSTBUFFER_SEC * fps / self.keep_every)
        self.prebuffer = deque(maxlen=self.pre_len)
        self.meta_pre = deque(maxlen=self.pre_len)
        self.active = False
        self.frames = []     # list of depth frames (uint16)
        self.meta = []       # list of dicts aligned with frames
        self.counter = 0     # downsampling counter
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _maybe_keep(self):
        self.counter += 1
        if self.counter >= self.keep_every:
            self.counter = 0
            return True
        return False

    def tick(self, depth_image_u16, meta_dict):
        """Always call each frame. Stores into prebuffer (downsampled)."""
        if not self.enabled:
            return
        if not self._maybe_keep():
            return
        self.prebuffer.append(depth_image_u16.copy())
        self.meta_pre.append(meta_dict.copy())

        if self.active:
            self.frames.append(depth_image_u16.copy())
            self.meta.append(meta_dict.copy())

    def start(self):
        if not self.enabled or self.active:
            return
        self.active = True
        # seed with prebuffer
        self.frames = list(self.prebuffer)
        self.meta = list(self.meta_pre)

    def stop_and_render(self):
        if not (self.enabled and self.active):
            self.active = False
            return
        # add postbuffer frames (already being collected via tick)
        # We simply wait until enough frames have passed externally, then call this.
        # Here we render immediately with what we have.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sess_dir = os.path.join(self.out_dir, f"session_{timestamp}")
        os.makedirs(sess_dir, exist_ok=True)
        video_path = os.path.join(sess_dir, "review.mp4")
        meta_path = os.path.join(sess_dir, "meta.json")

        # Save meta
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

        # Render MP4 with overlays
        if len(self.frames) > 0:
            h, w = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
            for depth_u16, m in zip(self.frames, self.meta):
                frame_bgr = display_depth_image(depth_u16, self.depth_scale)

                # Overlays: measured center, predicted ROI, predicted path
                if m.get("meas_px") is not None:
                    cx, cy = m["meas_px"]
                    cv2.circle(frame_bgr, (int(cx), int(cy)), 5, (0,255,0), 2)
                if m.get("pred_px") is not None:
                    cv2.circle(frame_bgr, (int(m["pred_px"][0]), int(m["pred_px"][1])),
                               SEARCH_RADIUS_PX, (255,255,255), 1, lineType=cv2.LINE_AA)

                # Predicted path from saved state (if provided)
                if m.get("state") is not None:
                    state = np.array(m["state"], float)
                    # Lightweight re-sim using same dynamics
                    # (we avoid creating a second EKF; just reuse functions)
                    x_tmp = state.copy()
                    pts_pix = []
                    step = m.get("dt", EKF_DT)
                    steps = max(1, int(math.ceil(REVIEW_PATH_HORIZON_S / step)))
                    for _ in range(steps):
                        # propagate
                        x_tmp = _rk4_step(x_tmp, step, np.array(G_VECTOR, float), BETA_DRAG)
                        # project to pixel
                        try:
                            uv = rs.rs2_project_point_to_pixel(self.intrinsics, x_tmp[:3].tolist())
                            u = int(np.clip(uv[0], 0, w-1)); v = int(np.clip(uv[1], 0, h-1))
                            pts_pix.append((u,v))
                        except Exception:
                            break
                    # draw path
                    for i in range(1, len(pts_pix)):
                        cv2.line(frame_bgr, pts_pix[i-1], pts_pix[i], (0,255,255), 1, lineType=cv2.LINE_AA)

                writer.write(frame_bgr)
            writer.release()

        print(f"[REVIEW] Saved {len(self.frames)} frames to {video_path}")
        print(f"[REVIEW] Metadata saved to {meta_path}")

        # reset
        self.active = False
        self.frames.clear(); self.meta.clear()

# ============================ Main ============================

def main():
    # IPC: subscribe to authority (SIM/LIVE) and publish intercepts for the sim
    ctrl_sub = ControlSubscriber() if ControlSubscriber else None
    itc_pub  = InterceptPublisher() if InterceptPublisher else None
    last_pub_ts = 0.0

    # Authority and Arduino-send flag (driven by sim). Start conservative.
    authority = "SIM"                    # SIM or LIVE (set by sim)
    send_arduino = False                 # sim checkbox; only matters in LIVE
    default_send_arduino = SEND_TO_ARDUINO  # fallback until sim tells us otherwise

    sender = InterceptSender(True, SERIAL_PORT, SERIAL_BAUD)

    # RealSense setup (unchanged)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_profile = profile.get_stream(rs.stream.depth)
    intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

    # Review (unchanged)
    review = ReviewCapture(
        enabled=ENABLE_REVIEW_CAPTURE,
        fps=1.0/EKF_DT,
        keep_every=CAPTURE_KEEP_EVERY,
        depth_scale=depth_scale,
        intrinsics=intrinsics,
        out_dir=REVIEW_OUT_DIR
    )

    # State (unchanged)
    fast_mode = True
    detected_ball = False
    undetected_count = 0
    detection_count = 0
    post_frames_remaining = 0

    ball_depth_mm = 0.0
    calib_radius_px = None
    calib_depth_mm = None

    ekf = None
    ball_centers_px = []

    def _fold_control(msg, mode_prev, send_prev):
        """Robustly extract mode + send_arduino from a variety of message shapes."""
        mode_out, send_out = mode_prev, send_prev
        try:
            if isinstance(msg, dict):
                mode_out = msg.get("mode", mode_prev)
                send_out = msg.get("send_arduino", msg.get("send_to_arduino", send_prev))
            else:
                # object with attributes
                mode_attr = getattr(msg, "mode", None) or getattr(msg, "text", None)
                if mode_attr: mode_out = mode_attr
                if hasattr(msg, "send_arduino"):
                    send_out = bool(getattr(msg, "send_arduino"))
                elif hasattr(msg, "send_to_arduino"):
                    send_out = bool(getattr(msg, "send_to_arduino"))
        except Exception:
            pass
        return mode_out, send_out

    try:
        while True:
            # Poll SIM/LIVE authority from sim (throttled by the publisher in the sim)
            if ctrl_sub:
                ctl = ctrl_sub.try_get()
                if ctl:
                    authority, send_arduino = _fold_control(ctl, authority, send_arduino)
            else:
                # No control channel: keep a safe fallback
                authority = "SIM"
                send_arduino = default_send_arduino

            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            h, w = depth_image.shape
            center_screen = (w//2, h//2)

            if not fast_mode:
                disp = display_depth_image(depth_image, depth_scale)

            meta = {
                "ts": time.time(),
                "pred_px": None,
                "meas_px": None,
                "state": ekf.x.tolist() if ekf is not None else None,
                "intercept_t": None,
                "intercept_xyz": None,
                "dt": EKF_DT
            }

            # ------------- detection / EKF -------------
            if not detected_ball:
                info = detect_circle_in_depth(depth_image, depth_scale)
                if info is not None:
                    (cx, cy), r_px, depth_m = info
                    ball_depth_mm = depth_m * 1000.0
                    calib_radius_px = r_px
                    calib_depth_mm = ball_depth_mm

                    p3d = rs.rs2_deproject_pixel_to_point(intrinsics, [float(cx), float(cy)], float(depth_m))
                    p3d = np.asarray(p3d, float)
                    ekf = EKF3DDrag(dt=EKF_DT, g=G_VECTOR, beta_drag=BETA_DRAG, Q_pos=Q_POS, Q_vel=Q_VEL, R_xyz=R_XYZ)
                    ekf.initialize(p3d, v_xyz=None)

                    ball_centers_px = [(cx, cy)]
                    detected_ball = True
                    detection_count = 1
                    undetected_count = 0
                    post_frames_remaining = 0

                    if ENABLE_REVIEW_CAPTURE:
                        review.start()

                    meta["meas_px"] = (cx, cy)
                    meta["state"] = ekf.x.tolist()

                    if not fast_mode:
                        cv2.circle(disp, (cx, cy), int(r_px), (0,255,0), 2)
                        cv2.putText(disp, "Initial Lock", (cx-30, cy-int(r_px)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                else:
                    if not fast_mode:
                        cv2.putText(disp, "No ball detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            else:
                ekf.predict()

                p_pred = ekf.x[:3].tolist()
                try:
                    pred_px = rs.rs2_project_point_to_pixel(intrinsics, p_pred)
                except Exception:
                    pred_px = ball_centers_px[-1] if ball_centers_px else center_screen
                pred_px = (int(np.clip(pred_px[0],0,w-1)), int(np.clip(pred_px[1],0,h-1)))
                meta["pred_px"] = pred_px
                meta["state"] = ekf.x.tolist()

                exp_r_px = calib_radius_px if calib_radius_px is not None else INIT_SEARCH_CIRCLE_PX
                exp_r_px = predict_radius(exp_r_px, calib_depth_mm or ball_depth_mm, ball_depth_mm or (calib_depth_mm or 1000.0))

                centre_info, depth_mm_found = depth_detection(
                    depth_frame, depth_scale, ball_depth_mm or calib_depth_mm or 1000.0,
                    exp_r_px, pred_px, fast_mode
                )

                if centre_info is not None and depth_mm_found is not None:
                    cx, cy = centre_info
                    depth_m = depth_mm_found / 1000.0
                    meas_p3d = rs.rs2_deproject_pixel_to_point(intrinsics, [float(cx), float(cy)], float(depth_m))
                    meas_p3d = np.asarray(meas_p3d, float)

                    ekf.update_xyz(meas_p3d, gate_alpha=GATE_ALPHA)

                    ball_depth_mm = depth_mm_found
                    ball_centers_px.append((cx, cy)); ball_centers_px = ball_centers_px[-2:]
                    detection_count += 1
                    undetected_count = 0
                    post_frames_remaining = int(POSTBUFFER_SEC * (1.0/EKF_DT) / CAPTURE_KEEP_EVERY)

                    # Compute intercept & communicate
                    t_hit, p_hit = ekf.predict_intercept_with_plane(PLANE_NORMAL, PLANE_D, t_max=2.0)
                    if t_hit is not None and p_hit is not None:
                        # Publish to sim when live, rate-limited
                        now = time.time()
                        if authority == "LIVE" and itc_pub and (now - last_pub_ts) >= IPC_MIN_PERIOD_S:
                            try:
                                itc_pub.publish(
                                    x_mm=float(p_hit[0]) * 1000.0,
                                    y_mm=float(p_hit[1]) * 1000.0,
                                    t_hit_s=float(t_hit),
                                    ttl=0.25
                                )
                                last_pub_ts = now
                            except Exception as e:
                                print(f"[IPC] Intercept publish failed: {e}")

                        # Send to Arduino only if authority == LIVE
                        if send_arduino and authority == "LIVE":
                            sender.send(p_hit[0], p_hit[1], t_hit)

                        meta["intercept_t"] = float(t_hit)
                        meta["intercept_xyz"] = [float(p_hit[0]), float(p_hit[1]), float(p_hit[2])]

                    meta["meas_px"] = (cx, cy)

                    if not fast_mode:
                        cv2.circle(disp, (cx, cy), int(exp_r_px), (0,255,0), 2)
                        cv2.circle(disp, pred_px, SEARCH_RADIUS_PX, (255,255,255), 1, lineType=cv2.LINE_AA)
                        cv2.putText(disp, "Tracking", (cx-20, cy-int(exp_r_px)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                else:
                    undetected_count += 1
                    if undetected_count > 10:
                        if ENABLE_REVIEW_CAPTURE and review.active:
                            review.stop_and_render()
                        detected_ball = False
                        ekf = None
                        ball_centers_px.clear()
                        detection_count = 0
                        undetected_count = 0
                        continue

            # Review tick (unchanged)
            if ENABLE_REVIEW_CAPTURE:
                review.tick(depth_image, meta)

            # UI & keys (unchanged)
            if not fast_mode:
                cv2.imshow("Depth Ball Tracker (EKF+Drag)", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if ENABLE_REVIEW_CAPTURE and review.active:
                    review.stop_and_render()
                detected_ball = False
                ekf = None
                ball_centers_px.clear()
                detection_count = 0
                undetected_count = 0
                print("[INFO] Reset tracking.")
            elif key == ord('f'):
                fast_mode = not fast_mode
                print("[INFO] Live UI:", "ON" if not fast_mode else "OFF")

    finally:
        try: pipeline.stop()
        except Exception: pass
        sender.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()