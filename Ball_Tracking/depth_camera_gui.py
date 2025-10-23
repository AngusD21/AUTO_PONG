# tracker_gui.py
# PyQt5 GUI that integrates RealSense + EKF(+drag) tracker
# IPC (ControlListener/InterceptPublisher) and Arduino serial
# ReviewCapture (pre/post ring buffer with mp4 + meta.json)

import sys, os, time, json, math, threading
from dataclasses import dataclass
from collections import deque
import numpy as np
import cv2
import serial

from PyQt5 import QtCore, QtGui, QtWidgets
import pyrealsense2 as rs

from Ball_Tracking.review_wizard import ReviewPage, VideoLabel, qimage_from_bgr
from Ball_Tracking.plane_wizard import PlaneSetupWizard, TablePlane

try:
	from Communications.ipc import ControlListener, InterceptPublisher
except Exception as e:
	print(f"[IPC] Warning: could not import Communications.ipc ({e}). Running without IPC.")
	ControlListener = None
	InterceptPublisher = None

B_LOGO_PATH  = "Assets/BULLSEYE.png"
W_LOGO_PATH  = "Assets/BULLSEYE_W.png"

# =========================== CONFIG ===========================
SEND_TO_ARDUINO_DEFAULT = False
SERIAL_PORT = "COM6"
SERIAL_BAUD = 115200

# Review capture
PREBUFFER_SEC = 1.0
POSTBUFFER_SEC = 1.0
CAPTURE_KEEP_EVERY = 2
REVIEW_OUT_DIR = "captures"
REVIEW_PATH_HORIZON_S = 0.8

# Catch plane (n·p + d = 0)
CATCH_PLANE_Z_M = 1.20
PLANE_NORMAL = (0.0, 0.0, 1.0)
PLANE_D = -CATCH_PLANE_Z_M

G_VECTOR = (0.0, 9.81, 0.0)
BETA_DRAG = 0.02

# Detection
DEPTH_TOLERANCE_MM = 15
SEARCH_RADIUS_PX = 200
INIT_MIN_RADIUS_PX = 5
INIT_SEARCH_CIRCLE_PX = 70

# EKF tuning
EKF_DT = 1/90.0
Q_POS = 1e-5
Q_VEL = 1e-3
R_XYZ = (5e-4, 5e-4, 1.8e-3)
GATE_ALPHA = 0.99

# IPC rate limit
IPC_MIN_PERIOD_S = 1.0 / 30.0
# ============================================================


# ======================= EKF (with drag) ======================
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
	"""x=[px,py,pz,vx,vy,vz]. Process: ṗ=v, v̇=g-β||v||v. Measurement: position (m)."""
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
		self.I = np.eye(self.n).astype(float)

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
			# rough chisq thresholds for 3 dof
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
# ===============================================================


# ==================== Utility / Display helpers ====================
def apply_dark_theme(app: QtWidgets.QApplication):
	app.setStyle("Fusion")
	p = QtGui.QPalette()
	# Window & panels
	p.setColor(QtGui.QPalette.Window,        QtGui.QColor("#111317"))
	p.setColor(QtGui.QPalette.Base,          QtGui.QColor("#0b0d11"))
	p.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#0f1217"))
	# Text
	p.setColor(QtGui.QPalette.WindowText,    QtGui.QColor("#e5e7eb"))
	p.setColor(QtGui.QPalette.Text,          QtGui.QColor("#e5e7eb"))
	p.setColor(QtGui.QPalette.ButtonText,    QtGui.QColor("#e5e7eb"))
	# Controls
	p.setColor(QtGui.QPalette.Button,        QtGui.QColor("#141821"))
	p.setColor(QtGui.QPalette.Highlight,     QtGui.QColor("#2563eb"))
	p.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
	app.setPalette(p)

	app.setStyleSheet("""
	QMainWindow { background: #111317; }
	QWidget#controlPanel { background: #0f1217; }
	QFrame.Card {
		background: #141821; border: 1px solid #1f2937; border-radius: 10px;
	}
	QPushButton {
		background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 6px 10px; color: #e5e7eb;
	}
	QPushButton:hover { background: #0f172a; }
	QPushButton:checked { background: #1d2a4d; }
	QCheckBox, QLabel { color: #e5e7eb; }
	QLineEdit, QComboBox, QPlainTextEdit {
		background: #0b0d11; border: 1px solid #1f2937; border-radius: 6px; color: #e5e7eb; padding: 4px 6px;
	}
	QSlider::groove:horizontal { height: 6px; background: #1f2937; border-radius: 3px; }
	QSlider::handle:horizontal { background: #e5e7eb; width: 14px; margin: -5px 0; border-radius: 7px; }
	QLabel#titleLabel { font-size: 22px; font-weight: 900; letter-spacing: 10px; color: #e5e7eb; }
	""")

class FaintLogo(QtWidgets.QLabel):
	def __init__(self, path, max_width=180, opacity=0.10, align=QtCore.Qt.AlignCenter, parent=None):
		super().__init__(parent)
		self.setAlignment(align)
		if os.path.exists(path):
			pm = QtGui.QPixmap(path)
			if not pm.isNull():
				pm = pm.scaledToWidth(max_width, QtCore.Qt.SmoothTransformation)
				self.setPixmap(pm)
				eff = QtWidgets.QGraphicsOpacityEffect(self)
				eff.setOpacity(opacity)
				self.setGraphicsEffect(eff)

def make_card(*widgets):
	card = QtWidgets.QFrame()
	card.setObjectName("Card")
	card.setProperty("class", "Card")
	layout = QtWidgets.QVBoxLayout(card)
	layout.setContentsMargins(12, 12, 12, 12)
	layout.setSpacing(8)
	for w in widgets:
		layout.addWidget(w)
	return card

def colourise_depth(depth_u16: np.ndarray,
                    depth_scale: float,
                    clip_mm=(400, 3000),  
                    use_auto_percentiles=False
                   ) -> np.ndarray:

    # Convert to millimetres (for D435, depth_scale~0.001 so this equals depth_u16)
    depth_mm = depth_u16.astype(np.float32) * (depth_scale * 1000.0)
    valid = depth_u16 > 0

    # Choose visualization range
    if use_auto_percentiles:
        if valid.any():
            lo = float(np.percentile(depth_mm[valid], 2))
            hi = float(np.percentile(depth_mm[valid], 98))
            if hi <= lo:   # fallback if scene is flat
                lo, hi = 400.0, 3000.0
        else:
            lo, hi = 400.0, 3000.0
    else:
        lo, hi = clip_mm

    # Clamp to [lo,hi] only on valid pixels
    vis = np.zeros_like(depth_mm, dtype=np.float32)
    if valid.any():
        vis[valid] = np.clip(depth_mm[valid], lo, hi)

    vis_u8 = np.zeros_like(depth_u16, dtype=np.uint8)
    rng = (hi - lo)
    if rng > 1e-6 and valid.any():
        vis_u8[valid] = np.round(255.0 * (vis[valid] - lo) / rng).astype(np.uint8)

    vis_bgr = cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)

    # Make invalid pixels black (optional but helpful)
    if valid.any():
        vis_bgr[~valid] = (0, 0, 0)

    return vis_bgr



def predict_radius(calib_r_px, calib_depth_mm, current_depth_mm):
	if current_depth_mm <= 0 or calib_depth_mm is None or calib_r_px is None:
		return int(calib_r_px or INIT_SEARCH_CIRCLE_PX)
	return int(calib_r_px * (calib_depth_mm / current_depth_mm))
# ==================================================================


# ===================== Review capture ==============
class ReviewCapture:
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
		self.frames = []
		self.meta = []
		self.counter = 0
		self.out_dir = out_dir
		os.makedirs(self.out_dir, exist_ok=True)

	def _maybe_keep(self):
		self.counter += 1
		if self.counter >= self.keep_every:
			self.counter = 0
			return True
		return False

	def tick(self, depth_image_u16, meta_dict):
		if not self.enabled: return
		if not self._maybe_keep(): return
		self.prebuffer.append(depth_image_u16.copy())
		self.meta_pre.append(dict(meta_dict))

		if self.active:
			self.frames.append(depth_image_u16.copy())
			self.meta.append(dict(meta_dict))

	def start(self):
		if not self.enabled or self.active: return
		self.active = True
		self.frames = list(self.prebuffer)
		self.meta = list(self.meta_pre)

	def stop_and_render(self):
		if not (self.enabled and self.active):
			self.active = False
			return
		timestamp = time.strftime("%Y%m%d_%H%M%S")
		sess_dir = os.path.join(self.out_dir, f"session_{timestamp}")
		os.makedirs(sess_dir, exist_ok=True)

		video_path = os.path.join(sess_dir, "review.mp4")
		meta_path = os.path.join(sess_dir, "meta.json")

		with open(meta_path, "w", encoding="utf-8") as f:
			json.dump(self.meta, f, indent=2)

		if len(self.frames) > 0:
			h, w = self.frames[0].shape
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
			for depth_u16, m in zip(self.frames, self.meta):
				frame_bgr = colourise_depth(depth_u16, self.depth_scale)
				# Overlay: measured center / search circle
				if m.get("meas_px") is not None:
					cx, cy = m["meas_px"]
					cv2.circle(frame_bgr, (int(cx), int(cy)), 5, (0,255,0), 2)
				if m.get("pred_px") is not None:
					cv2.circle(frame_bgr, (int(m["pred_px"][0]), int(m["pred_px"][1])),
							SEARCH_RADIUS_PX, (255,255,255), 1, lineType=cv2.LINE_AA)
				writer.write(frame_bgr)
			writer.release()

		print(f"[REVIEW] Saved {len(self.frames)} frames to {video_path}")
		print(f"[REVIEW] Metadata saved to {meta_path}")

		self.active = False
		self.frames.clear(); self.meta.clear()
# ==================================================================


# ===================== Serial sender (same format) =================
class InterceptSender:
	"""Sends ASCII: (TRACK,x_mm,y_mm,t_ms)\n"""
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
# ==================================================================


# =================== Core tracking worker (QThread) ===============
class CameraWorker(QtCore.QThread):
	frame = QtCore.pyqtSignal(QtGui.QImage)     # live frame for display
	status = QtCore.pyqtSignal(str)             # status line
	event = QtCore.pyqtSignal(str)              # log events
	camera_ok = QtCore.pyqtSignal(bool)         # camera connection

	def __init__(self, parent=None):
		super().__init__(parent)
		self._running = True

		# View mode
		self.req_depth = True
		self.req_rgb = False
		self.req_fast = False

		# Reconfig trigger
		self._reconfig_requested = False

		# FPS tracking
		self._fps = 0.0
		self._last_tick = None
		self._fps_ema = None
		self._last_ts = None

		self.view_mode = "Depth"
		self.fast_mode = False
		self.enable_record = False

		self.spatial = None
		self.temporal = None
		self.holes = None

		self.crop_top = 0.0
		self.crop_bottom = 0.0
		self.crop_left = 0.0
		self.crop_right = 0.0

		# Plane setup
		self.search_box = None
		self._last_cloud = None
		self._cloud_lock = threading.Lock()
		self._cloud_stride = 4
		self._cloud_every = 3
		self._cloud_counter = 0
		self.pc = None

		# IPC / authority
		self.ctrl = ControlListener() if ControlListener else None
		self.itc_pub = InterceptPublisher() if InterceptPublisher else None
		self.last_pub_ts = 0.0

		# Arduino sender (toggle from UI)
		self.send_to_arduino = SEND_TO_ARDUINO_DEFAULT
		self.serial_sender = InterceptSender(self.send_to_arduino, SERIAL_PORT, SERIAL_BAUD)

		# RealSense pipeline
		self.pipeline = None
		self.depth_scale = 0.001
		self.intrinsics = None
		self.has_rgb = False

		# EKF & detection state
		self.detected = False
		self.ekf = None
		self.ball_depth_mm = 0.0
		self.calib_radius_px = None
		self.calib_depth_mm = None
		self.ball_centers_px = []
		self.undetected_count = 0

		# Review
		self.review = None

	# ---------- Public setters from UI ----------
	def set_view_mode(self, mode: str):
		self.view_mode = mode
		self.fast_mode = (mode == "Fast")

	def set_send_arduino(self, enabled: bool):
		self.send_to_arduino = bool(enabled)
		# reopen serial if needed
		try:
			self.serial_sender.close()
		except Exception:
			pass
		self.serial_sender = InterceptSender(self.send_to_arduino, SERIAL_PORT, SERIAL_BAUD)

	def set_crop(self, top, bottom, left, right):
		self.crop_top, self.crop_bottom = float(top), float(bottom)
		self.crop_left, self.crop_right = float(left), float(right)

	# ---------- RealSense init ----------
	def _open_camera(self, want_depth: bool, want_rgb: bool):
		# Choose FPS combos RealSense
		# - depth only: 640x480@90
		# - depth+rgb: depth@30 + rgb@60
		# - rgb only:   rgb@60
		try:
			self.pipeline = rs.pipeline()
			cfg = rs.config()

			if want_depth and not want_rgb:
				cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
				self.has_rgb = False
			elif want_depth and want_rgb:
				cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
				cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
				self.has_rgb = True
			elif (not want_depth) and want_rgb:
				cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
				self.has_rgb = True
			else:
				# nothing requested → run in fast mode (no camera)
				return False

			profile = self.pipeline.start(cfg)

			# Depth metadata only if depth is enabled
			try:
				depth_sensor = profile.get_device().first_depth_sensor()
				self.depth_scale = depth_sensor.get_depth_scale()
			except Exception:
				self.depth_scale = 0.001

			try:
				if want_depth:
					depth_profile = profile.get_stream(rs.stream.depth)
					self.intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
				else:
					self.intrinsics = None
			except Exception:
				self.intrinsics = None

			self.pc = rs.pointcloud() if want_depth else None

			# (Re)build ReviewCapture with current scale/intrinsics
			self.review = ReviewCapture(
				enabled=self.enable_record,
				fps=1.0/EKF_DT,
				keep_every=CAPTURE_KEEP_EVERY,
				depth_scale=self.depth_scale,
				intrinsics=self.intrinsics,
				out_dir=REVIEW_OUT_DIR
			)

			# EXPLORE LATER
			# RealSense post-processing filters to reduce speckle & holes
			# try:
			# 	self.spatial = rs.spatial_filter()   # edge-preserving spatial
			# 	self.temporal = rs.temporal_filter() # temporal filter
			# 	self.holes = rs.hole_filling_filter()
			# except Exception:
			self.spatial = self.temporal = self.holes = None

			self.event.emit(f"Camera opened (depth={want_depth}, rgb={want_rgb}).")
			return True
		except Exception as e:
			self.event.emit(f"Camera open failed: {e}")
			try:
				if self.pipeline: self.pipeline.stop()
			except Exception:
				pass
			self.pipeline = None
			return False
	
	def _close_camera(self):
		"""Safely stop & tear down the RealSense pipeline."""
		try:
			if self.pipeline is not None:
				try:
					# Stop streaming (may raise if already stopped)
					self.pipeline.stop()
				except Exception:
					pass
		finally:
			self.pipeline = None
			# Reset per-stream flags so the next _open_camera() sets them afresh
			self.has_rgb = False
			self.intrinsics = None
			self.pc = None
			self._last_cloud = None
			# Let the UI know the camera is now “down”
			try:
				self.camera_ok.emit(False)
			except Exception:
				pass
	
	def get_point_cloud_snapshot(self):
		with self._cloud_lock:
			if self._last_cloud is None:
				return None
			return self._last_cloud.copy()
	
	# ---------- Request reconfiguration ----------
	@QtCore.pyqtSlot(bool, bool, bool)
	def request_reconfig(self, want_depth: bool, want_rgb: bool, fast_mode: bool):
		self.req_depth = bool(want_depth)
		self.req_rgb   = bool(want_rgb)
		self.req_fast  = bool(fast_mode)
		self._reconfig_requested = True

	# ---------- Detection helpers ----------
	def _global_detect(self, depth_u16):
		# circle-ish region between 0.5–3.0m
		depth_mm = depth_u16 * self.depth_scale * 1000.0
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
			m = np.zeros_like(depth_u16, np.uint8); cv2.drawContours(m,[cnt],0,255,-1)
			vals = depth_mm[m==255]; vals = vals[vals>0]
			if vals.size == 0: continue
			best = ((int(x),int(y)), int(r), float(np.median(vals))/1000.0)
			best_circ = circ
		return best

	def _focus_detect(self, depth_u16, ball_depth_mm, expected_radius_px, prediction_xy):
		depth_mm = depth_u16 * self.depth_scale * 1000.0
		band = (depth_mm > (ball_depth_mm-DEPTH_TOLERANCE_MM)) & (depth_mm < (ball_depth_mm+DEPTH_TOLERANCE_MM))
		depth_mask = (band.astype(np.uint8)*255)

		h, w = depth_mask.shape
		px = int(np.clip(prediction_xy[0], 0, w-1))
		py = int(np.clip(prediction_xy[1], 0, h-1))
		search_mask = np.zeros_like(depth_mask)
		cv2.circle(search_mask, (px, py), SEARCH_RADIUS_PX, 255, -1)
		depth_mask = cv2.bitwise_and(depth_mask, search_mask)

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

	# ---------- Main loop ----------
	def run(self):
		# initial open based on defaults
		ok = False
		if not self.req_fast:
			ok = self._open_camera(self.req_depth, self.req_rgb)
		self.camera_ok.emit(ok)

		while self._running:
			# Handle reconfig request
			if self._reconfig_requested:
				self._reconfig_requested = False

				# stop any current capture
				try:
					if self.enable_record and self.review and self.review.active:
						self.review.stop_and_render()
				except Exception:
					pass

				# close camera if open
				try:
					if self.pipeline: self._close_camera()
				except Exception:
					pass
				self.pipeline = None
				ok = False

				# Fast mode: keep camera closed
				if self.req_fast:
					self.view_mode = "Fast"
					self.fast_mode = True
					self.event.emit("Fast mode enabled (no camera).")
					self.camera_ok.emit(False)
				else:
					self.fast_mode = False
					# Reopen with requested streams
					ok = self._open_camera(self.req_depth, self.req_rgb)
					self.camera_ok.emit(ok)

			authority = "LIVE" if (self.ctrl and self.ctrl.live_enabled) else "SIM"

			# Fast mode
			if self.req_fast:
				self.status.emit("Camera: FAST mode (no live feed)")
				time.sleep(0.01)
				continue
			
			if not ok:
				# Show placeholder and retry occasionally
				frame = np.zeros((480, 640, 3), np.uint8)
				cv2.putText(frame, "No camera connection", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
				self.frame.emit(qimage_from_bgr(frame))
				self.status.emit("Camera: DISCONNECTED")
				time.sleep(0.2)
				ok = self._open_camera(self.req_depth, self.req_rgb)
				self.camera_ok.emit(ok)
				continue

			try:
				frames = self.pipeline.wait_for_frames(1000)
			except Exception:
				self.event.emit("Frame grab timeout.")
				ok = False
				self._close_camera()
				self.camera_ok.emit(False)
				continue
			
			if self.req_depth and self.req_rgb: 
				self.view_mode = "Both"
				depth_frame = frames.get_depth_frame()
				colour_frame = frames.get_color_frame()
			if self.req_depth and not self.req_rgb: 
				self.view_mode = "Depth"
				depth_frame = frames.get_depth_frame()
				colour_frame = None
			if not self.req_depth and self.req_rgb: 
				self.view_mode = "RGB"
				depth_frame = None
				colour_frame = frames.get_color_frame()

			# Apply RS filters if available
			# try:
			# 	if self.temporal: depth_frame = self.temporal.process(depth_frame)
			# 	if self.spatial:  depth_frame = self.spatial.process(depth_frame)
			# 	if self.holes:    depth_frame = self.holes.process(depth_frame)
			# except Exception:
			# 	pass

			# Assemble display frame depending on mode
			show_bgr = None
			if self.view_mode == "Depth":
				depth_u16 = np.asanyarray(depth_frame.get_data())
				h, w = depth_u16.shape
				show_bgr = colourise_depth(depth_u16, self.depth_scale)

			elif self.view_mode == "Both" and colour_frame is not None:
				depth_u16 = np.asanyarray(depth_frame.get_data())
				h, w = depth_u16.shape
				d = colourise_depth(depth_u16, self.depth_scale)
				c = np.asanyarray(colour_frame.get_data()).copy()
				show_bgr = np.hstack([d, c])

			elif self.view_mode == "RGB" and colour_frame is not None:
				colour_im = colour_frame.get_data()
				h, w = colour_im.shape[:2]
				show_bgr = np.asanyarray(colour_im).copy()


			# If RGB only, no tracking
			if self.view_mode != "RGB":
				# Meta per-frame (for review)
				meta = {
					"ts": time.time(),
					"pred_px": None,
					"meas_px": None,
					"state": self.ekf.x.tolist() if self.ekf is not None else None,
					"intercept_t": None,
					"intercept_xyz": None,
					"dt": EKF_DT
				}

				# Point cloud
				if self.pc is not None:
					self._cloud_counter = (self._cloud_counter + 1) % self._cloud_every
					if self._cloud_counter == 0:
						try:
							points = self.pc.calculate(depth_frame)
							v = np.asanyarray(points.get_vertices())
							xyz = np.asarray(v.tolist()).reshape(-1, 3)
							m = np.isfinite(xyz).all(axis=1) & (xyz[:,2] > 0)
							xyz = xyz[m][::self._cloud_stride]
							with self._cloud_lock:
								self._last_cloud = xyz
						except Exception:
							pass

				# ===== Tracking =====
				if not self.detected:
					info = self._global_detect(depth_u16)
					if info is not None:
						(cx, cy), r_px, depth_m = info
						self.ball_depth_mm = depth_m * 1000.0
						self.calib_radius_px = r_px
						self.calib_depth_mm = self.ball_depth_mm

						p3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(cx), float(cy)], float(depth_m))
						p3d = np.asarray(p3d, float)
						self.ekf = EKF3DDrag(dt=EKF_DT, g=G_VECTOR, beta_drag=BETA_DRAG, Q_pos=Q_POS, Q_vel=Q_VEL, R_xyz=R_XYZ)
						self.ekf.initialize(p3d, v_xyz=None)

						self.ball_centers_px = [(cx, cy)]
						self.detected = True
						self.undetected_count = 0

						if self.enable_record:
							self.review.start()

						meta["meas_px"] = (cx, cy)
						meta["state"] = self.ekf.x.tolist()
						self.event.emit(f"Ball detected at ({cx},{cy})")

						if show_bgr is not None:
							cv2.circle(show_bgr, (cx, cy), int(r_px), (0,255,0), 2)
					else:
						pass
				else:
					# Predict
					self.ekf.predict()
					p_pred = self.ekf.x[:3].tolist()
					try:
						pred_px = rs.rs2_project_point_to_pixel(self.intrinsics, p_pred)
					except Exception:
						pred_px = self.ball_centers_px[-1] if self.ball_centers_px else (w//2, h//2)
					pred_px = (int(np.clip(pred_px[0],0,w-1)), int(np.clip(pred_px[1],0,h-1)))
					meta["pred_px"] = pred_px
					meta["state"] = self.ekf.x.tolist()

					exp_r_px = predict_radius(self.calib_radius_px, self.calib_depth_mm, self.ball_depth_mm or (self.calib_depth_mm or 1000.0))
					centre_info, depth_mm_found = self._focus_detect(
						depth_u16, self.ball_depth_mm or self.calib_depth_mm or 1000.0,
						exp_r_px, pred_px
					)

					if centre_info is not None and depth_mm_found is not None:
						cx, cy = centre_info
						depth_m = depth_mm_found / 1000.0
						meas_p3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(cx), float(cy)], float(depth_m))
						meas_p3d = np.asarray(meas_p3d, float)
						self.ekf.update_xyz(meas_p3d, gate_alpha=GATE_ALPHA)

						self.ball_depth_mm = depth_mm_found
						self.ball_centers_px.append((cx, cy)); self.ball_centers_px = self.ball_centers_px[-2:]
						self.undetected_count = 0

						# Intercept + comms
						t_hit, p_hit = self.ekf.predict_intercept_with_plane(PLANE_NORMAL, PLANE_D, t_max=2.0)
						if t_hit is not None and p_hit is not None:
							now = time.time()
							# Publish to SIM (rate-limited)
							if (authority == "LIVE") and self.itc_pub and (now - self.last_pub_ts) >= IPC_MIN_PERIOD_S:
								try:
									self.itc_pub.publish(x_mm=float(p_hit[0])*1000.0,
														y_mm=float(p_hit[1])*1000.0,
														t_hit_s=float(t_hit),
														source="tracker")
									self.last_pub_ts = now
								except Exception as e:
									self.event.emit(f"[IPC] publish failed: {e}")

							# Arduino (only in LIVE and if enabled)
							if (authority == "LIVE") and self.send_to_arduino:
								self.serial_sender.send(p_hit[0], p_hit[1], t_hit)

							meta["intercept_t"] = float(t_hit)
							meta["intercept_xyz"] = [float(p_hit[0]), float(p_hit[1]), float(p_hit[2])]

						meta["meas_px"] = (cx, cy)

						if show_bgr is not None and self.view_mode != "Fast":
							cv2.circle(show_bgr, (cx, cy), int(exp_r_px), (0,255,0), 2)
							cv2.circle(show_bgr, pred_px, SEARCH_RADIUS_PX, (255,255,255), 1, lineType=cv2.LINE_AA)
					else:
						self.undetected_count += 1
						if self.undetected_count > 10:
							if self.enable_record and self.review.active:
								self.review.stop_and_render()
							self.detected = False
							self.ekf = None
							self.ball_centers_px.clear()
							self.undetected_count = 0
							self.event.emit("Ball lost.")

				# Review tick
				if self.enable_record:
					self.review.tick(depth_u16, meta)

			if (show_bgr is not None) and (self.view_mode != "Fast"):
				self.frame.emit(qimage_from_bgr(show_bgr))

			# FPS Calculation
			now = time.time()
			if self._last_ts is None:
				self._last_ts = now
			else:
				dt = now - self._last_ts
				self._last_ts = now
				if dt <= 0 or dt > 2.0:  # guard against bad timestamps and long stalls
					pass
				else:
					inst_fps = 1.0 / dt
					alpha = 0.15
					self._fps_ema = inst_fps if self._fps_ema is None else (1-alpha)*self._fps_ema + alpha*inst_fps

			fps_txt = f"{self._fps_ema:0.1f} FPS" if self._fps_ema else "— FPS"
			self.status.emit(f"Authority: {authority} | Arduino: {'ON' if self.send_to_arduino else 'OFF'} | {fps_txt}")

		# cleanup
		try:
			if self.enable_record and self.review and self.review.active:
				self.review.stop_and_render()
		except Exception:
			pass
		self._close_camera()
		self.serial_sender.close()

	def stop(self):
		self._running = False
# ==================================================================


# ========================== GUI main window =======================

class MainWindow(QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("DEADSHOT – Ball Tracker")
		if os.path.exists(B_LOGO_PATH):
			self.setWindowIcon(QtGui.QIcon(B_LOGO_PATH))

		app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
		apply_dark_theme(app)

		# Stacked central content: video label (main) vs plane setup
		self.stack = QtWidgets.QStackedWidget()
		self.setCentralWidget(self.stack)

		# Page 0: Live
		self.video = VideoLabel()
		live_page = QtWidgets.QWidget()
		live_layout = QtWidgets.QVBoxLayout(live_page)
		live_layout.setContentsMargins(0,0,0,0)
		live_layout.addWidget(self.video)
		self.stack.addWidget(live_page)   # index 0

		# Page 1: Review
		self.review_page = ReviewPage()
		self.stack.addWidget(self.review_page)  # index 1
		self.review_page.request_return.connect(self._return_to_live)

		# Page 2: Plane setup
		self.plane_setup = PlaneSetupWizard(worker=None, parent=self)
		self.stack.addWidget(self.plane_setup) # index 2

		# --- Right dock = controls ---
		dock = QtWidgets.QDockWidget("", self)
		dock.setObjectName("rightDock")
		dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
		dock.setTitleBarWidget(QtWidgets.QWidget(dock))
		self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

		panel = QtWidgets.QWidget()
		panel.setObjectName("controlPanel")
		root = QtWidgets.QVBoxLayout(panel)
		root.setContentsMargins(12, 12, 12, 12)
		root.setSpacing(12)

		# Header title
		header = QtWidgets.QHBoxLayout()
		title = QtWidgets.QLabel("D  E  A  D  S  H  T")   # logo where O would be
		title.setObjectName("titleLabel")
		header.addWidget(title, 1)

		# Insert the small logo instead of “O”
		if os.path.exists(W_LOGO_PATH):
			small_logo = QtWidgets.QLabel()
			pm = QtGui.QPixmap(W_LOGO_PATH).scaledToHeight(28, QtCore.Qt.SmoothTransformation)
			small_logo.setPixmap(pm)
			# Build spaced title with the logo as the “O”
			spaced = QtWidgets.QHBoxLayout()
			for ch in "DEADSH":
				lbl = QtWidgets.QLabel(ch)
				lbl.setObjectName("titleLabel")
				spaced.addWidget(lbl, 1, QtCore.Qt.AlignCenter)
			spaced.addWidget(small_logo, 1, QtCore.Qt.AlignCenter)   # <- this is the “O”
			lbl_t = QtWidgets.QLabel("T"); lbl_t.setObjectName("titleLabel")
			spaced.addWidget(lbl_t, 1, QtCore.Qt.AlignCenter)
			header = spaced

		root.addLayout(header)

		# --- Camera status + streams/fast card ---
		self.cam_status = QtWidgets.QLabel("Camera: …")

		streams_row = QtWidgets.QHBoxLayout()
		streams_row.addWidget(QtWidgets.QLabel("Streams"))
		self.chk_depth = QtWidgets.QCheckBox("Depth")
		self.chk_rgb   = QtWidgets.QCheckBox("RGB")
		self.chk_depth.setChecked(True)
		self.chk_rgb.setChecked(False)
		streams_row.addWidget(self.chk_depth)
		streams_row.addWidget(self.chk_rgb)
		streams_row.addStretch(1)

		self.chk_fast  = QtWidgets.QCheckBox("Fast mode (no camera)")
		self.chk_fast.setChecked(False)

		cam_box = QtWidgets.QWidget()
		cam_v = QtWidgets.QVBoxLayout(cam_box)
		cam_v.setContentsMargins(0,0,0,0)
		cam_v.setSpacing(6)
		cam_v.addWidget(self.cam_status)
		cam_v.addLayout(streams_row)
		cam_v.addWidget(self.chk_fast)

		card_cam = make_card(cam_box)
		root.addWidget(card_cam)

		# --- Send + recording card ---
		self.send_chk = QtWidgets.QCheckBox("Send to Arduino (LIVE only)")
		self.send_chk.setChecked(SEND_TO_ARDUINO_DEFAULT)

		self.rec_chk = QtWidgets.QCheckBox("Enable Recording")
		self.rec_chk.setChecked(False)

		self.btn_load_review = QtWidgets.QPushButton("Review Wizard")
		self.btn_plane = QtWidgets.QPushButton("Plane Wizard")

		card_send = make_card(self.send_chk, self.rec_chk, self.btn_load_review, self.btn_plane)
		root.addWidget(card_send)

		self.btn_plane.clicked.connect(self._enter_plane_setup)
		self.plane_setup.saved.connect(self._plane_saved)
		self.plane_setup.canceled.connect(self._plane_canceled)

		# --- ROI sliders card ---
		roi_box = QtWidgets.QWidget()
		roi_layout = QtWidgets.QVBoxLayout(roi_box)
		roi_layout.setContentsMargins(0,0,0,0)
		roi_layout.setSpacing(6)
		roi_layout.addWidget(QtWidgets.QLabel("Ignore region (crop)"))
		# build sliders
		def _mk_slider_row(text):
			roww = QtWidgets.QHBoxLayout()
			roww.addWidget(QtWidgets.QLabel(text))
			s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
			s.setRange(0, 50)  # 0..50%
			s.setSingleStep(1)
			roww.addWidget(s)
			rw = QtWidgets.QWidget()
			rw.setLayout(roww)
			return s, rw

		self.s_top,    row_top    = _mk_slider_row("Top")
		self.s_bottom, row_bottom = _mk_slider_row("Bottom")
		self.s_left,   row_left   = _mk_slider_row("Left")
		self.s_right,  row_right  = _mk_slider_row("Right")

		for rw in (row_top, row_bottom, row_left, row_right):
			roi_layout.addWidget(rw)

		card_roi = make_card(roi_box)
		root.addWidget(card_roi)

		# --- Events (log) card ---
		self.log = QtWidgets.QPlainTextEdit()
		self.log.setReadOnly(True)
		self.log.setMaximumBlockCount(500)
		self.log.setPlaceholderText("Runtime messages will appear here…")

		log_box = QtWidgets.QWidget()
		log_v = QtWidgets.QVBoxLayout(log_box)
		log_v.setContentsMargins(0,0,0,0)
		log_v.setSpacing(6)
		log_v.addWidget(QtWidgets.QLabel("Events"))
		log_v.addWidget(self.log)

		card_log = make_card(log_box)
		root.addWidget(card_log, 1)

		# Faint bottom logo
		root.addStretch(1)
		root.addWidget(FaintLogo(W_LOGO_PATH, max_width=160, opacity=0.08, align=QtCore.Qt.AlignCenter))

		dock.setWidget(panel)

		# Worker
		self.worker = CameraWorker()
		self.worker.frame.connect(self.on_frame)
		self.worker.status.connect(self.on_status)
		self.worker.event.connect(self.on_event)
		self.worker.camera_ok.connect(self.on_camera_ok)

		self.plane_setup._set_worker(self.worker)

		self.chk_depth.toggled.connect(self._apply_streams)
		self.chk_rgb.toggled.connect(self._apply_streams)
		self.chk_fast.toggled.connect(self._apply_streams)

		self.send_chk.toggled.connect(self.worker.set_send_arduino)
		for s in (self.s_top, self.s_bottom, self.s_left, self.s_right):
			s.valueChanged.connect(self._apply_crop)

		self.btn_load_review.clicked.connect(self._load_review)
		self.rec_chk.toggled.connect(self._on_toggle_record)

		# Start worker
		self.worker.start()

	def closeEvent(self, e):
		self.worker.stop()
		self.worker.wait(1500)
		# try:
		# 	if self.rev_cap:
		# 		self.rev_cap.release()
		# except Exception:
		# 	pass
		return super().closeEvent(e)
	
	def _enter_review(self, path: str):
		if self.review_page.load_video(path):
			self.worker.stop()        # stop live worker while reviewing
			self.worker.wait(1500)
			self.stack.setCurrentIndex(1)
			self.findChild(QtWidgets.QDockWidget, "rightDock").hide()

	def _return_to_live(self):
		# return to live
		self.stack.setCurrentIndex(0)
		self.findChild(QtWidgets.QDockWidget, "rightDock").show()
		# restart worker
		self.worker = CameraWorker()
		self._wire_worker_signals()  # helper to reconnect signals
		self.worker.start()

	def _enter_plane_setup(self):
		# give the wizard the current worker before showing it
		self.plane_setup._set_worker(self.worker)
		# hide the right dock while in the wizard
		self.findChild(QtWidgets.QDockWidget, "rightDock").hide()
		# show plane wizard page
		self.stack.setCurrentWidget(self.plane_setup)
		self.on_event("Plane setup: entering")

	def _plane_canceled(self):
		self.on_event("Plane setup: canceled")
		self.stack.setCurrentIndex(0)  # back to live
		self.findChild(QtWidgets.QDockWidget, "rightDock").show()

	def _plane_saved(self, box):
		self.on_event(f"Plane setup: saved box {box}")
		self.worker.search_box = box
		self.stack.setCurrentIndex(0)  # back to live
		self.findChild(QtWidgets.QDockWidget, "rightDock").show()

	def _wire_worker_signals(self):
		self.worker.frame.connect(self.on_frame)
		self.worker.status.connect(self.on_status)
		self.worker.event.connect(self.on_event)
		self.worker.camera_ok.connect(self.on_camera_ok)
		# Re-apply current UI settings to the worker
		self.worker.set_send_arduino(self.send_chk.isChecked())
		self._apply_crop()
		self._apply_streams()
		# Keep review capture toggle in sync if you use it
		self.worker.enable_record = self.rec_chk.isChecked()
		if getattr(self.worker, "review", None):
			self.worker.review.enabled = self.rec_chk.isChecked()

	def _mk_slider(self, label, parent_layout):
		row = QtWidgets.QHBoxLayout()
		row.addWidget(QtWidgets.QLabel(label))
		s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		s.setRange(0, 50)  # 0..50% of height/width
		s.setSingleStep(1)
		row.addWidget(s)
		parent_layout.addLayout(row)
		return s

	def _apply_crop(self):
		self.worker.set_crop(
			self.s_top.value()/100.0,
			self.s_bottom.value()/100.0,
			self.s_left.value()/100.0,
			self.s_right.value()/100.0,
		)
	
	def _apply_streams(self):
		want_fast  = self.chk_fast.isChecked()
		want_depth = self.chk_depth.isChecked()
		want_rgb   = self.chk_rgb.isChecked()

		# Disable depth/rgb checkboxes while in fast mode
		self.chk_depth.setEnabled(not want_fast)
		self.chk_rgb.setEnabled(not want_fast)

		# Request reconfigure in the worker thread
		self.worker.request_reconfig(want_depth, want_rgb, want_fast)

	# ---------- Live updates ----------
	@QtCore.pyqtSlot(QtGui.QImage)
	def on_frame(self, qimg):
		self.video.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
			self.video.width(), self.video.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
		))

	@QtCore.pyqtSlot(str)
	def on_status(self, s):
		self.cam_status.setText(s)

	@QtCore.pyqtSlot(bool)
	def on_camera_ok(self, ok):
		if not ok:
			# show black screen with text
			frame = np.zeros((480, 640, 3), np.uint8)
			cv2.putText(frame, "No camera connection", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
			self.on_frame(qimage_from_bgr(frame))

	@QtCore.pyqtSlot(str)
	def on_event(self, msg):
		ts = time.strftime("%H:%M:%S")
		self.log.appendPlainText(f"[{ts}] {msg}")

	# ---------- Review playback ----------
	def _load_review(self):
		path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open review.mp4", REVIEW_OUT_DIR, "MP4 files (*.mp4)")
		if not path:
			return
		self.on_event(f"Loaded review: {path}")
		self._enter_review(path)

		# if not path:
		# 	return
		# try:
		# 	if self.rev_cap:
		# 		self.rev_cap.release()
		# 	self.rev_cap = cv2.VideoCapture(path)
		# 	self.rev_total = int(self.rev_cap.get(cv2.CAP_PROP_FRAME_COUNT))
		# 	self.scrub.setRange(0, max(0, self.rev_total-1))
		# 	self.scrub.setEnabled(True)
		# 	self.btn_play.setEnabled(True)
		# 	self.rev_play = False
		# 	self._show_review_frame(0)
		# 	self.on_event(f"Loaded review: {path}")
		# except Exception as e:
		# 	self.on_event(f"Failed to load review: {e}")

	# def _show_review_frame(self, idx):
	# 	if not self.rev_cap: return
	# 	self.rev_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
	# 	ok, frame = self.rev_cap.read()
	# 	if not ok: return
	# 	self.on_frame(qimage_from_bgr(frame))

	# def _tick_review(self):
	# 	if not (self.rev_cap and self.rev_play): return
	# 	pos = int(self.rev_cap.get(cv2.CAP_PROP_POS_FRAMES))
	# 	ok, frame = self.rev_cap.read()
	# 	if not ok:
	# 		self.rev_play = False
	# 		self.rev_timer.stop()
	# 		return
	# 	self.on_frame(qimage_from_bgr(frame))
	# 	self.scrub.blockSignals(True)
	# 	self.scrub.setValue(min(self.rev_total-1, pos))
	# 	self.scrub.blockSignals(False)
	
	def _on_toggle_record(self, v):
		v = bool(v)
		self.worker.enable_record = v

		# If the ReviewCapture object exists, flip its internal flag too
		rc = getattr(self.worker, "review", None)
		if rc is not None:
			rc.enabled = v

		self.on_event(f"Review capture {'ENABLED' if v else 'DISABLED'}")
	


def main():
	app = QtWidgets.QApplication(sys.argv)
	win = MainWindow()
	win.resize(1200, 700)
	win.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()
