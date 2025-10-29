# tracker_gui.py
# PyQt5 GUI that integrates RealSense + EKF(+drag) tracker
# IPC (ControlListener/InterceptPublisher) and Arduino serial
# ReviewCapture (pre/post ring buffer with mp4 + meta.json)

import sys, os, time, json, math, threading
from dataclasses import dataclass
from collections import deque
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import pyrealsense2 as rs

from Ball_Tracking.ekf import EKF3DDrag
from Ball_Tracking.review_wizard import ReviewWizard, ReviewCapture
from Ball_Tracking.plane_wizard import PlaneSetupWizard, TablePlane
from Ball_Tracking.graphics_objects import AspectImageView, BackgroundLogo, CollapsibleCard, _apply_overlays, _apply_rotation, _as_np, _compute_cloud, _hstack_same_height, _map_px_for_display, _valid_frame, apply_dark_theme, make_card, bgr_np_to_qimage, _roi_colouring, qimage_to_bgr_np
from Ball_Tracking.plane_math import _render_interest_region_from_cloud, overlay_plane_and_roi_on_bgr_po, colourise_depth, plane_poly_px_from_overlay
from Communications.ipc import ControlListener, InterceptPublisher, InterceptSender

# =========================== CONFIG ===========================
SEND_TO_ARDUINO_DEFAULT = False
SERIAL_PORT = "COM6"
SERIAL_BAUD = 115200

# Catch plane (n·p + d = 0)
BETA_DRAG = 0.02

# Detection
DEPTH_TOLERANCE_MM = 15
INIT_MIN_RADIUS_PX = 5
INIT_SEARCH_CIRCLE_PX = 70
SEARCH_RADIUS_PX = 30

# EKF tuning
EKF_DT = 1/90.0
Q_POS = 1e-5
Q_VEL = 1e-3
R_XYZ = (5e-4, 5e-4, 1.8e-3)
GATE_ALPHA = 0.99

# IPC rate limit
IPC_MIN_PERIOD_S = 1.0 / 30.0

REVIEW_OUT_DIR = "captures"
B_LOGO_PATH  = "Assets/BULLSEYE.png"
W_LOGO_PATH  = "Assets/BULLSEYE_W.png"
FAST_IM = "Assets/FAST_MODE.png"
SEARCH_IM = "Assets/SEARCHING.png"
PLANE_W = "Assets/PLANE_W.png"
PLAY_W = "Assets/PLAY_W.png"
ARM_W = "Assets/ARM_W.png"
GANTRY_W = "Assets/GANTRY_W.png"
WIZARD_W = "Assets/WIZARD_W.png"

# ============================================================


# =================== Core tracking worker (QThread) ===============
class CameraWorker(QtCore.QThread):
	frame = QtCore.pyqtSignal(QtGui.QImage)     # live frame for display
	status = QtCore.pyqtSignal(str)             # status line
	event = QtCore.pyqtSignal(str)              # log events
	camera_ok = QtCore.pyqtSignal(bool)         # camera connection
	fast_ok = QtCore.pyqtSignal(bool)         	# fast connection
	rgb_for_wizard = QtCore.pyqtSignal(QtGui.QImage)

	def __init__(self, parent=None):
		super().__init__(parent)
		self._running = True

		# View mode
		self.req_depth = True
		self.req_rgb = False
		self.req_fast = False
		self.track = True

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
		self.rot_k = 1

		# Interest region controls
		self.roi_dict = {}

		# Plane setup
		self.search_box = None
		self._last_cloud = None
		self._cloud_lock = threading.Lock()
		self._cloud_stride = 4
		self._cloud_every = 3
		self._cloud_counter = 0
		self.pc = None
		self.enable_pointcloud = False
		self.display_hz = 25.0      # UI refresh cap (Hz)
		self._last_emit_ts = 0.0

		self.draw_roi = False
		self.draw_plane = True
		self.colour_roi = False

		self.plane_overlay = None
		self.plane_n = None
		self.plane_d = None

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
		self.color_intrinsics = None
		self.has_rgb = False

		# EKF & detection state
		self.detector_mode = "depth_roi"
		self.detected = False
		self.ekf = None
		self.ball_depth_mm = 0.0
		self.calib_radius_px = None
		self.calib_depth_mm = None
		self.ball_centers_px = []
		self.undetected_count = 0

		# Review
		self.review = None
		
		# --- Review pre/post buffers (±1s) ---
		self._prebuf = deque(maxlen=int(round(1.0 / EKF_DT))) 
		self._postroll_until = 0.0

		self.SEARCH_CARD = cv2.imread(SEARCH_IM)

	# ---------- Public setters from UI ----------

	def set_send_arduino(self, enabled: bool):
		self.send_to_arduino = bool(enabled)
		# reopen serial if needed
		try:
			self.serial_sender.close()
		except Exception:
			pass
		self.serial_sender = InterceptSender(self.send_to_arduino, SERIAL_PORT, SERIAL_BAUD)

	# ---------- Main loop ----------
	def run(self):
		# initial open based on defaults
		ok = False
		if not self.req_fast and self._device_present():
			ok = self._open_camera(self.req_depth, self.req_rgb)
		self.camera_ok.emit(ok)

		while self._running:
			# Handle reconfig request
			if self._reconfig_requested:
				self._reconfig_requested = False
				try:
					if self.enable_record and self.review and self.review.active:
						self.review.stop_and_render()
				except Exception:
					pass
				self._close_camera()
				ok = self._open_camera(self.req_depth, self.req_rgb)
				self.camera_ok.emit(ok)
				if self.req_fast:
					self.frame.emit(bgr_np_to_qimage(cv2.imread(FAST_IM)))

			authority = "LIVE" if (self.ctrl and self.ctrl.live_enabled) else "SIM"
			
			if not ok:
				now = time.time()
				# backoff: retry every 2s
				next_try = getattr(self, "_next_reopen_ts", 0.0)
				if now >= next_try:
					# self.frame.emit(qimage_from_bgr(self.SEARCH_CARD))
					self.status.emit("Camera: DISCONNECTED")
					# try reopen
					ok = self._open_camera(self.req_depth, self.req_rgb)
					self.camera_ok.emit(ok)
					# set next retry time
					self._next_reopen_ts = now + 5.0
				time.sleep(0.02)
				continue

			try:
				frames = self.pipeline.wait_for_frames(5000)
			except Exception:
				self.event.emit("Frame grab timeout.")
				ok = False
				self._close_camera()
				self.camera_ok.emit(False)
				continue

			depth_frame  = None
			colour_frame = None
			
			if self.req_fast:
				depth_frame = frames.get_depth_frame()
				if not _valid_frame(depth_frame):
					self.event.emit("Depth frame missing this cycle.")
					continue
				self.view_mode = "Fast"
			else:
				if self.req_depth:
					depth_frame = frames.get_depth_frame()
					if not _valid_frame(depth_frame):
						self.event.emit("Depth frame missing this cycle.")
						continue
				if self.req_rgb:
					colour_frame = frames.get_color_frame()
					if not _valid_frame(colour_frame):
						# colour often needs a cycle or two after restart
						self.event.emit("Color frame missing this cycle.")

				if self.req_depth and self.req_rgb: self.view_mode = "Both"
				if self.req_depth and not self.req_rgb: self.view_mode = "Depth"
				if not self.req_depth and self.req_rgb: self.view_mode = "RGB"

			# Apply RS filters if available
			# try:
			# 	if self.temporal: depth_frame = self.temporal.process(depth_frame)
			# 	if self.spatial:  depth_frame = self.spatial.process(depth_frame)
			# 	if self.holes:    depth_frame = self.holes.process(depth_frame)
			# except Exception:
			# 	pass

			# Compose Frames
			if 1:
				show_bgr = None
				c_bgr = None
				po = self.plane_overlay
				ro = self.roi_dict
				cloud = self._last_cloud

				if self.view_mode == "Fast":
					depth_u16 = np.asanyarray(depth_frame.get_data())

				elif self.view_mode == "Depth":
					depth_u16 = _as_np(depth_frame)
					if depth_u16 is None:
						continue
					cloud = _compute_cloud(depth_frame, cloud, self.colour_roi)
					d_img = _roi_colouring(depth_u16, self.intrinsics, cloud, po, self.colour_roi, ro, self.depth_scale)
					d_bgr = _apply_overlays(d_img, self.intrinsics, po, self.draw_plane, self.draw_roi, ro)
					d_bgr = _apply_rotation(self.rot_k, d_bgr)
					show_bgr = d_bgr

				elif self.view_mode == "Both":
					depth_u16 = _as_np(depth_frame)
					if depth_u16 is None:
						continue

					# Depth side
					cloud = _compute_cloud(depth_frame, cloud, self.colour_roi)
					d_img = _roi_colouring(depth_u16, self.intrinsics, cloud, po, self.colour_roi, ro, self.depth_scale)
					d_bgr = _apply_overlays(d_img, self.intrinsics, po, self.draw_plane, self.draw_roi, ro)
					d_bgr = _apply_rotation(self.rot_k, d_bgr)

					# RGB side
					if _valid_frame(colour_frame):
						c_raw = _as_np(colour_frame)
						if c_raw is not None:
							c_tmp = _apply_overlays(c_raw, self.color_intrinsics, po, self.draw_plane, self.draw_roi, ro) \
									if self.color_intrinsics is not None else c_raw
							c_bgr = _apply_rotation(self.rot_k, c_tmp)

					# Compose (depth | rgb) if both exist
					show_bgr = _hstack_same_height(d_bgr, c_bgr)

				elif self.view_mode == "RGB" and _valid_frame(colour_frame):
					c_raw = _as_np(colour_frame)
					if c_raw is None:
						continue
					c_tmp = _apply_overlays(c_raw, self.color_intrinsics, po, self.draw_plane, self.draw_roi, ro) \
							if self.color_intrinsics is not None else c_raw
					c_bgr = _apply_rotation(self.rot_k, c_tmp)
					show_bgr = c_bgr
				
				base_h, base_w = depth_u16.shape

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
                    "dt": EKF_DT,
                    "detector_mode": self.detector_mode,
                    "intrinsics": self._intrinsics_dict(),
                    "plane": (self.plane_overlay.copy() if self.plane_overlay else None)
                }

				# Point cloud
				if self.pc is not None and self.enable_pointcloud:
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
					
					if colour_frame is not None:
						try:
							self.rgb_for_wizard.emit(bgr_np_to_qimage(_as_np(colour_frame)))
						except Exception as e:
							print(e)
							pass

				if self.track:
					# ===== Tracking =====
					if not self.detected:
						# seed by chosen detector
						cxy = r_px = depth_m = p3d = None
						if self.detector_mode == "depth_roi":
							cxy, r_px, depth_m = self._detect_depth_roi_only(depth_u16)
							if cxy is not None:
								p3d = np.asarray(rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(cxy[0]), float(cxy[1])], float(depth_m)), float)
						elif self.detector_mode == "roi_3d_cluster":
							cxy, r_px, depth_mm_hint, p3d = self._detect_roi_3d_cluster(depth_u16)
							depth_m = (depth_mm_hint/1000.0) if depth_mm_hint else None
						else:  # "hybrid_2d3d"
							cxy, r_px, depth_m, p3d = self._detect_hybrid_2d3d(depth_u16)

						if cxy is not None and p3d is not None and depth_m is not None:
							cx, cy = cxy
							self.ball_depth_mm = depth_m * 1000.0
							self.calib_radius_px = r_px or self.calib_radius_px or 20
							self.calib_depth_mm = self.ball_depth_mm

							# EKF with correct gravity vector: g = -9.81 * n̂
							if self.plane_n is not None: n_up = np.asarray(self.plane_n, float).reshape(3)
							else: n_up = np.asarray([0,1,0], float).reshape(3)
							g_vec = -9.81 * n_up
							self.ekf = EKF3DDrag(dt=EKF_DT, g=g_vec, beta_drag=BETA_DRAG, Q_pos=Q_POS, Q_vel=Q_VEL, R_xyz=R_XYZ)
							self.ekf.initialize(p3d, v_xyz=None)

							self.ball_centers_px = [tuple(cxy)]
							self.detected = True
							self.undetected_count = 0

							if self.enable_record:
								self.review.start()
								# Pre-roll: dump last 1s from prebuffer
								try:
									for (db_u16, db_meta) in list(self._prebuf):
										self.review.tick(db_u16, db_meta)
								except Exception:
									pass

							meta["meas_px"] = tuple(cxy)
							meta["state"] = self.ekf.x.tolist()
							self.event.emit(f"Ball detected at {tuple(cxy)} via {self.detector_mode}")

							if show_bgr is not None and self.view_mode != "Fast":
								dx, dy = _map_px_for_display(self.rot_k, cx, cy, base_w, base_h)
								cv2.circle(show_bgr, (int(dx), int(dy)), int(r_px or 20), (0,255,0), 2)
						else:
							# not detected this frame; nothing else
							pass

					else:
						# Predict
						self.ekf.predict()
						p_pred = self.ekf.x[:3].tolist()
						try:
							pred_px = rs.rs2_project_point_to_pixel(self.intrinsics, p_pred)
						except Exception:
							pred_px = self.ball_centers_px[-1] if self.ball_centers_px else (base_w//2, base_h//2)
						pred_px = (int(np.clip(pred_px[0],0,base_w-1)), int(np.clip(pred_px[1],0,base_h-1)))
						meta["pred_px"] = pred_px
						meta["state"] = self.ekf.x.tolist()

						exp_r_px = int(self.calib_radius_px or 20)
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
							n_use = np.asarray(self.plane_n, float).reshape(3) if self.plane_n is not None else np.asarray((0,-1,0), float)
							d_use = float(self.plane_d) if self.plane_d is not None else float(1)
							t_hit, p_hit = self.ekf.predict_intercept_with_plane(n_use, d_use, t_max=2.0)

							if t_hit is not None and p_hit is not None:
								now = time.time()
								if (authority == "LIVE") and self.itc_pub and (now - self.last_pub_ts) >= IPC_MIN_PERIOD_S:
									try:
										self.itc_pub.publish(x_mm=float(p_hit[0])*1000.0,
															y_mm=float(p_hit[1])*1000.0,
															t_hit_s=float(t_hit),
															source="tracker")
										self.last_pub_ts = now
									except Exception as e:
										self.event.emit(f"[IPC] publish failed: {e}")

								if (authority == "LIVE") and self.send_to_arduino:
									self.serial_sender.send(p_hit[0], p_hit[1], t_hit)

								meta["intercept_t"] = float(t_hit)
								meta["intercept_xyz"] = [float(p_hit[0]), float(p_hit[1]), float(p_hit[2])]

							meta["meas_px"] = (cx, cy)

							if show_bgr is not None and self.view_mode != "Fast":
								mx, my = meta["meas_px"]
								dx, dy = _map_px_for_display(self.rot_k, mx, my, base_w, base_h)
								cv2.circle(show_bgr, (int(dx), int(dy)), int(exp_r_px), (0,255,0), 2)
								px, py = pred_px
								dx, dy = _map_px_for_display(self.rot_k, px, py, base_w, base_h)
								cv2.circle(show_bgr, (int(dx), int(dy)), SEARCH_RADIUS_PX, (255,255,255), 1, lineType=cv2.LINE_AA)

						else:
							self.undetected_count += 1
							if self.undetected_count > 10:
								# begin post-roll if we were recording
								if self.enable_record and self.review and self.review.active:
									self._postroll_until = time.time() + 1.0  # 1s tail
								self.detected = False
								self.ekf = None
								self.ball_centers_px.clear()
								self.undetected_count = 0
								self.event.emit("Ball lost.")

					self._prebuf.append((depth_u16.copy(), meta.copy()))

					# Review tick
					if self.enable_record:
						# If a post-roll is active, keep ticking until deadline
						if self._postroll_until > 0.0:
							if time.time() <= self._postroll_until:
								self.review.tick(depth_u16, meta)
							else:
								try:
									self.review.stop_and_render()
								except Exception:
									pass
								self._postroll_until = 0.0
						else:
							# normal ticking while recording (during pre/detect/track)
							self.review.tick(depth_u16, meta)
			
			if (show_bgr is not None) and (self.view_mode != "Fast"):
				now_emit = time.time()
				if (now_emit - self._last_emit_ts) >= (1.0 / self.display_hz):
					self._last_emit_ts = now_emit
					self.frame.emit(bgr_np_to_qimage(show_bgr))

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

	# ---------- RealSense init ----------
	def _open_camera(self, want_depth: bool, want_rgb: bool):
		if not self._device_present():
			self.event.emit("No RealSense device detected.")
			return False
		
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
				# nothing requested
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
			
			try:
				if want_rgb:
					color_profile = profile.get_stream(rs.stream.color)
					self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
				else:
					self.color_intrinsics = None
			except Exception:
				self.color_intrinsics = None

			self.pc = rs.pointcloud() if (want_depth and self.enable_pointcloud) else None

			# (Re)build ReviewCapture with current scale/intrinsics
			self.review = ReviewCapture(
				enabled=self.enable_record,
				fps=1.0/EKF_DT,
				keep_every=1,
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
	
	def stop(self):
		self._running = False
	
	def _device_present(self) -> bool:
		try:
			ctx = rs.context()
			return len(ctx.query_devices()) > 0
		except Exception:
			return False

	def set_enable_pointcloud(self, enable: bool):
		self.enable_pointcloud = bool(enable)
		# If depth stream is active and pipeline is up, create pc immediately
		if self.enable_pointcloud and self.pipeline is not None and self.pc is None and self.req_depth:
			try:
				self.pc = rs.pointcloud()
			except Exception:
				self.pc = None
		if not self.enable_pointcloud:
			self.pc = None
			# with self._cloud_lock:
			# 	self._last_cloud = None

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

	# def _plane_cam_vectors(self):
	# 	po = self.plane_overlay
	# 	if not po or not po.get("visible", True):
	# 		return None

	# 	p0 = np.asarray(po["p0"], float).reshape(3)
	# 	u  = np.asarray(po["u"],  float).reshape(3)
	# 	v  = np.asarray(po["v"],  float).reshape(3)
	# 	w  = float(po.get("width_m", 0.7))
	# 	l  = float(po.get("length_m", 0.7))

	# 	return p0, u, v, w, l
	
	# ----------- Detectors ---------------
	def _detect_hybrid_2d3d(self, depth_u16, pred_px=None, expected_r_px=20, depth_mm_hint=None):
		cxy, r_px, depth_m = self._detect_depth_roi_only(depth_u16, pred_px, expected_r_px, depth_mm_hint)
		if cxy is None: return None, None, None, None
		cx, cy = cxy

		# build a small disc and vector-deproject for robust 3D center
		disc = np.zeros_like(depth_u16, np.uint8)
		cv2.circle(disc, (cx, cy), int(max(4, r_px)), 255, -1)
		ys, xs = np.nonzero(disc)
		z_m = (depth_u16[ys, xs] * self.depth_scale).astype(np.float32)
		P = self._deproject_pixels(xs.astype(np.float32), ys.astype(np.float32), z_m)
		if P is not None and P.shape[0] >= 8:
			p3d = np.median(P, axis=0)
		else:
			p3d = np.asarray(rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(cx), float(cy)], float(depth_m)), float)
		return (cx, cy), r_px, depth_m, p3d
	
	def _detect_depth_roi_only(self, depth_u16, pred_px=None, expected_r_px=20, depth_mm_hint=None):
		h, w = depth_u16.shape
		roi = self._roi_polygon_mask((h, w))
		if roi is None:
			return None, None, None  # need plane/intrinsics

		depth_mm = depth_u16 * self.depth_scale * 1000.0
		mask = roi.copy()

		# If EKF predicted pixel available, crop to a disc
		if pred_px is not None:
			disc = np.zeros_like(mask)
			cv2.circle(disc, (int(pred_px[0]), int(pred_px[1])), SEARCH_RADIUS_PX, 255, -1)
			mask = cv2.bitwise_and(mask, disc)

		# If we have a depth hint, restrict a band; else use a broad sanity band
		if depth_mm_hint is not None:
			lo, hi = depth_mm_hint - DEPTH_TOLERANCE_MM, depth_mm_hint + DEPTH_TOLERANCE_MM
		else:
			lo, hi = 500.0, 3000.0
		band = ((depth_mm > lo) & (depth_mm < hi)).astype(np.uint8) * 255
		mask = cv2.bitwise_and(mask, band)

		cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		best, best_err = None, 1e9
		for c in cnts:
			area = cv2.contourArea(c)
			if area < 80: continue
			(x, y), r = cv2.minEnclosingCircle(c)
			if r <= 0: continue
			circ = area / (math.pi * r * r)
			if circ < 0.4: continue
			err = abs(r - expected_r_px)
			if err < best_err:
				best_err = err; best = (int(x), int(y), int(r))
		if best is None:
			return None, None, None

		x, y, r = best
		m = np.zeros_like(mask); cv2.circle(m, (x, y), r, 255, -1)
		vals = depth_mm[m == 255]
		vals = vals[(vals > lo) & (vals < hi)]
		if vals.size == 0:
			return None, None, None
		depth_m = float(np.median(vals)) / 1000.0
		return (x, y), r, depth_m
	
	def _clusters_voxel(self, pts, voxel=0.015, min_pts=30):
		if pts is None or pts.shape[0] == 0: return []
		q = np.floor(pts / float(voxel)).astype(np.int32)
		from collections import defaultdict, deque
		buckets = defaultdict(list)
		for i, key in enumerate(map(tuple, q)):
			buckets[key].append(i)
		keys = set(buckets.keys())
		vis = set(); out = []
		for k in keys:
			if k in vis: continue
			comp, dq = [], deque([k]); vis.add(k)
			while dq:
				kk = dq.popleft(); comp.append(kk); x,y,z = kk
				for nb in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
					if (nb in keys) and (nb not in vis): vis.add(nb); dq.append(nb)
			idx = [i for ck in comp for i in buckets[ck]]
			if len(idx) >= min_pts: out.append(np.asarray(idx, np.int32))
		return out

	def _detect_roi_3d_cluster(self, depth_u16):
		# sample pixels inside ROI to build a tiny point set (no rs.pointcloud)
		h, w = depth_u16.shape
		roi = self._roi_polygon_mask((h, w))
		if roi is None: return None, None, None, None

		# stride sample
		step = 2
		ys, xs = np.nonzero(roi[::step, ::step])
		ys = ys*step; xs = xs*step
		if ys.size < 20: return None, None, None, None

		z_m = (depth_u16[ys, xs] * self.depth_scale).astype(np.float32)
		valid = z_m > 0
		xs, ys, z_m = xs[valid].astype(np.float32), ys[valid].astype(np.float32), z_m[valid]
		P = self._deproject_pixels(xs, ys, z_m)
		if P is None or P.shape[0] < 50: return None, None, None, None

		clusters = self._clusters_voxel(P, voxel=0.012, min_pts=40)
		if not clusters: return None, None, None, None
		# pick cluster with best sphericity
		best, best_s = None, -1.0
		for idx in clusters:
			Q = P[idx]
			c = np.median(Q, axis=0)
			U, S, Vt = np.linalg.svd(Q - Q.mean(axis=0), full_matrices=False)
			sphericity = S[-1] / (S[0] + 1e-12)
			if sphericity > best_s:
				best_s = sphericity; best = (Q, c)
		Q, c3d = best
		# back-project c3d to a pixel for convenience
		try:
			px = rs.rs2_project_point_to_pixel(self.intrinsics, c3d.tolist())
			cx, cy = int(px[0]), int(px[1])
		except Exception:
			cx = cy = None
		# rough radius in pixels is optional; skip or estimate by median distance / fx
		return (cx, cy), None, float(np.median(z_m)*1000.0), c3d

	def _deproject_pixels(self, xs, ys, z_m):
		intr = self.intrinsics
		if intr is None or len(xs) == 0:
			return None
		fx, fy = float(intr.fx), float(intr.fy)
		cx = float(getattr(intr, "ppx", intr.ppx))
		cy = float(getattr(intr, "ppy", intr.ppy))
		X = (xs - cx) * z_m / fx
		Y = (ys - cy) * z_m / fy
		Z = z_m
		return np.column_stack([X, Y, Z])
	
	def _roi_polygon_mask(self, shape_hw):
		po = self.plane_overlay
		if not po or not self.intrinsics:
			return None
		p0 = po['p0']; u = po['u']; v = po['v']; w = po['width_m']; l = po['length_m']
		poly = plane_poly_px_from_overlay(p0, u, v, w, l, self.intrinsics).astype(np.int32)
		mask = np.zeros(shape_hw, np.uint8)
		cv2.fillPoly(mask, [poly], 255)
		return mask
	
	def _intrinsics_dict(self):
		intr = self.intrinsics
		if not intr:
			return None
		return {
			"fx": float(intr.fx), "fy": float(intr.fy),
			"ppx": float(getattr(intr, "ppx", intr.ppx)),
			"ppy": float(getattr(intr, "ppy", intr.ppy)),
			"width": int(intr.width), "height": int(intr.height)
		}

	def get_point_cloud_snapshot(self):
		with self._cloud_lock:
			if self._last_cloud is None:
				return None
			return self._last_cloud.copy()
	
	# ---------- Request reconfiguration ----------
	@QtCore.pyqtSlot(bool, bool, bool)
	def request_reconfig(self, want_depth: bool, want_rgb: bool, fast_mode: bool):
		if fast_mode:
			self.req_fast = True
			self.req_depth = True
			self.req_rgb = False
		else:
			self.req_fast = False
			self.req_depth = bool(want_depth)
			self.req_rgb   = bool(want_rgb)
		self._reconfig_requested = True


# ========================== GUI main window =======================
class MainWindow(QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("DEADSHOT - Ball Tracker")
		if os.path.exists(B_LOGO_PATH):
			self.setWindowIcon(QtGui.QIcon(B_LOGO_PATH))

		app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
		apply_dark_theme(app)

		# Stacked central content: video label (main) vs plane setup
		self.stack = QtWidgets.QStackedWidget()
		self.setCentralWidget(self.stack)

		# Worker
		self.worker = CameraWorker()
		self.worker.frame.connect(self.on_frame)
		self.worker.status.connect(self.on_status)
		self.worker.event.connect(self.on_event)
		self.worker.camera_ok.connect(self.on_camera_ok)

		# Page 0: Live
		self.video = AspectImageView(self)
		live_page = QtWidgets.QWidget()
		live_layout = QtWidgets.QVBoxLayout(live_page)
		live_layout.setContentsMargins(0,0,0,0)
		live_layout.addWidget(self.video)
		# BLACK background for the left content
		live_page.setStyleSheet("background:#000;")
		self.video.setStyleSheet("background:#000;")
		self.stack.addWidget(live_page)

		# Page 1: Review
		self.review_page = ReviewWizard()
		self.stack.addWidget(self.review_page)  # index 1
		self.review_page.request_return.connect(self._return_from_review)

		# Page 2: Plane setup
		self.plane_setup = PlaneSetupWizard(worker=self.worker, parent=self)
		self.stack.addWidget(self.plane_setup) # index 2
		self._plane_cfg_path = os.path.join(os.getcwd(), "Ball_Tracking/plane_wizard_state.json")
		self._plane_cfg_watcher = QtCore.QFileSystemWatcher(self)
		if os.path.isfile(self._plane_cfg_path):
			self._plane_cfg_watcher.addPath(self._plane_cfg_path)
		self._plane_cfg_watcher.fileChanged.connect(self._on_plane_cfg_changed)

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
		root.setSpacing(8)

		# Header title with logo in the word
		spaced = QtWidgets.QHBoxLayout()
		if os.path.exists(W_LOGO_PATH):
			pm = QtGui.QPixmap(W_LOGO_PATH).scaledToHeight(28, QtCore.Qt.SmoothTransformation)
		for ch in "DEADSH":
			lbl = QtWidgets.QLabel(ch); lbl.setObjectName("titleLabel"); spaced.addWidget(lbl, 1, QtCore.Qt.AlignCenter)
		if os.path.exists(W_LOGO_PATH):
			small_logo = QtWidgets.QLabel(); small_logo.setPixmap(pm); spaced.addWidget(small_logo, 1, QtCore.Qt.AlignCenter)
		lbl_t = QtWidgets.QLabel("T"); lbl_t.setObjectName("titleLabel"); spaced.addWidget(lbl_t, 1, QtCore.Qt.AlignCenter)
		root.addLayout(spaced)

		# --- Camera status + Streams + Detector (one compact card, no separate header) ---
		self.cam_status = QtWidgets.QLabel("Camera: …")
		root.addWidget(self.cam_status)

		row_streams = QtWidgets.QHBoxLayout()
		row_streams.setSpacing(6)
		row_streams.addWidget(QtWidgets.QLabel("Streams"))
		self.chk_depth = QtWidgets.QCheckBox("Depth"); self.chk_depth.setChecked(True)
		self.chk_rgb   = QtWidgets.QCheckBox("RGB");   self.chk_rgb.setChecked(False)
		self.chk_fast  = QtWidgets.QCheckBox("Fast");  self.chk_fast.setChecked(False)
		row_streams.addWidget(self.chk_depth); row_streams.addWidget(self.chk_rgb); row_streams.addWidget(self.chk_fast)

		row_detector = QtWidgets.QHBoxLayout()
		row_detector.setSpacing(6)
		row_detector.addWidget(QtWidgets.QLabel("Detector"))
		self.cb_detector = QtWidgets.QComboBox()
		self.cb_detector.addItems([
			"Hybrid 2D→3D",
			"Depth-only",
			"3D ROI Cluster"
		])
		row_detector.addWidget(self.cb_detector, 1)

		# --- Send + Recording (one row) ---
		row_io = QtWidgets.QHBoxLayout()
		row_io.setSpacing(6)
		self.send_chk = QtWidgets.QCheckBox("Send to Arduino"); self.send_chk.setChecked(SEND_TO_ARDUINO_DEFAULT)
		self.rec_chk  = QtWidgets.QCheckBox("Enable Recording"); self.rec_chk.setChecked(False)
		row_io.addWidget(self.send_chk); row_io.addWidget(self.rec_chk)

		cam_box = CollapsibleCard("CAMERA + DETECTOR")
		cam_box._button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
		cam_box._button.setStyleSheet(cam_box._button.styleSheet() + " text-align:center;")
		box_layout = cam_box.content_layout()
		box_layout.addLayout(row_streams); box_layout.addLayout(row_detector); box_layout.addLayout(row_io)
		cam_box._content.setVisible(False)
		root.addWidget(cam_box)

		# Detector mapping + apply
		_DET_MAP_UI2CODE = {
			"Hybrid 2D→3D": "hybrid_2d3d",
			"Depth-only":  "depth_roi",
			"3D ROI Cluster":  "roi_3d_cluster",
		}
		_DET_MAP_CODE2UI = {v:k for k,v in _DET_MAP_UI2CODE.items()}

		def _apply_detector_mode(_=None):
			mode = _DET_MAP_UI2CODE.get(self.cb_detector.currentText(), "hybrid_2d3d")
			if getattr(self, "worker", None):
				self.worker.detector_mode = mode
			self.on_event(f"Detector mode → {mode}")

		start_mode = getattr(self.worker, "detector_mode", "hybrid_2d3d")
		self.cb_detector.setCurrentText(_DET_MAP_CODE2UI.get(start_mode, "Hybrid 2D→3D (recommended)"))
		self.cb_detector.currentIndexChanged.connect(_apply_detector_mode)

		# --- VISUALS ---
		row_over = QtWidgets.QHBoxLayout()
		# row_over.setSpacing(6)
		self.chk_draw_plane = QtWidgets.QCheckBox("Draw plane"); self.chk_draw_plane.setChecked(True)
		self.chk_draw_roi   = QtWidgets.QCheckBox("Draw ROI");   self.chk_draw_roi.setChecked(False)
		row_over.addWidget(self.chk_draw_plane); row_over.addWidget(self.chk_draw_roi)

		row_colour = QtWidgets.QHBoxLayout()
		# row_colour.setSpacing(6)
		row_colour.addWidget(QtWidgets.QLabel("Colour by"))
		self.cb_colour_by = QtWidgets.QComboBox()
		self.cb_colour_by.addItems(["ROI", "Depth"])
		row_colour.addWidget(self.cb_colour_by, 1)

		self.btn_rotate90 = QtWidgets.QPushButton("Rotate 90°")
		self.btn_rotate90.setToolTip("Rotate output image 90° clockwise")

		over_card = CollapsibleCard("OVERLAYS", parent=panel, start_collapsed=True)
		over_card._button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
		over_card._button.setStyleSheet(over_card._button.styleSheet() + " text-align:center;")
		over_layout = over_card.content_layout()
		over_layout.addLayout(row_over)
		over_layout.addLayout(row_colour)
		over_layout.addWidget(self.btn_rotate90)
		root.addWidget(over_card)

		def _apply_overlay_toggles():
			self.worker.draw_plane = self.chk_draw_plane.isChecked()
			self.worker.draw_roi   = self.chk_draw_roi.isChecked()
			# Colour by dropdown: ROI vs Depth
			sel = self.cb_colour_by.currentText()
			self.worker.colour_roi = (sel == "ROI")
			if self.worker.colour_roi:
				self.worker.set_enable_pointcloud(True)

		self.chk_draw_plane.toggled.connect(_apply_overlay_toggles)
		self.chk_draw_roi.toggled.connect(_apply_overlay_toggles)
		self.cb_colour_by.currentIndexChanged.connect(lambda _: _apply_overlay_toggles())
		self.btn_rotate90.clicked.connect(self._on_rotate90_clicked)

		events_card = CollapsibleCard("EVENTS", parent=panel, start_collapsed=True)
		ev_layout = events_card.content_layout()
		events_card._button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
		events_card._button.setStyleSheet(events_card._button.styleSheet() + " text-align:center;")
		self.log = QtWidgets.QPlainTextEdit()
		self.log.setReadOnly(True)
		self.log.setMaximumBlockCount(500)
		self.log.setPlaceholderText("Runtime messages will appear here…")
		ev_layout.addWidget(self.log)
		root.addWidget(events_card)
		# root.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))
		root.addStretch(1)

		# Put a faint background logo behind everything; expanding events covers it
		if os.path.exists(W_LOGO_PATH):
			self._bg_logo = BackgroundLogo(W_LOGO_PATH, opacity=0.08, parent=panel)

			def _place_bg_logo_once():
				# scale to match the visible right-panel width (with a small margin)
				margin = 50
				target_w = max(80, panel.width() - 2*margin)  # avoid zero/silly values
				self._bg_logo.set_scaled_width(target_w)

				# bottom-center placement (fixed; no future resizing)
				pw, ph = panel.width(), panel.height()
				lw, lh = self._bg_logo.width(), self._bg_logo.height()
				x = (pw - lw) // 2
				y = ph//2 - lh + 140
				self._bg_logo.move(x, y)
				self._bg_logo.lower()
				self._bg_logo.show()

			QtCore.QTimer.singleShot(0, _place_bg_logo_once)

		# --- Wizards card (buttons) ---
		self.btn_load_review = self._mk_wizard_btn(PLAY_W,        "REVIEW",       self._enter_review)
		self.btn_plane       = self._mk_wizard_btn(PLANE_W,       "PLANE",        self._enter_plane_setup)
		self.btn_gantry      = self._mk_wizard_btn(GANTRY_W,      "GANTRY",       self._open_gantry_wizard)
		self.btn_manip       = self._mk_wizard_btn(ARM_W, "MANIPULATOR",  self._open_ARM_Wizard)

		root.addWidget(self.btn_load_review)
		root.addWidget(self.btn_plane)
		root.addWidget(self.btn_gantry)
		root.addWidget(self.btn_manip)

		self.btn_plane.clicked.connect(self._enter_plane_setup)
		self.plane_setup.saved.connect(self._plane_saved)
		self.plane_setup.canceled.connect(self._plane_canceled)
		self.btn_load_review.clicked.connect(self._enter_review)
		self.btn_gantry.clicked.connect(self._open_gantry_wizard)
		self.btn_manip.clicked.connect(self._open_ARM_Wizard)

		dock.setWidget(panel)

		# keep current behavior bindings
		self.plane_setup._set_worker(self.worker)
		self._load_plane_from_json()

		self.chk_depth.toggled.connect(self._apply_streams)
		self.chk_rgb.toggled.connect(self._apply_streams)
		self.chk_fast.toggled.connect(self._apply_streams)
		self.send_chk.toggled.connect(self.worker.set_send_arduino)
		self.rec_chk.toggled.connect(self._on_toggle_record)

		self.FAST_CARD = QtGui.QPixmap(FAST_IM)
		self.SEARCH_CARD = QtGui.QPixmap(SEARCH_IM)

		# Start worker
		self.worker.start()

	def closeEvent(self, e):
		self.worker.stop()
		self.worker.wait(1500)
		return super().closeEvent(e)
	
	def _load_plane_from_json(self, path=None):
		try_path = os.path.join(os.getcwd(), "Ball_Tracking/plane_wizard_state.json")

		state = None; chosen = None
		if try_path and os.path.isfile(try_path):
			try:
				with open(try_path, "r", encoding="utf-8") as f:
					state = json.load(f); chosen = try_path
			except Exception: state = None; chosen = None
	
		if not state:
			self.on_event("Plane config: no JSON found.")
			return

		self.on_event(f"Plane config: loaded {os.path.basename(chosen)}")

		# Push plane overlay
		flags = state.get("flags", {})
		flip_n = bool(flags.get("flip_n", False))
		visible = bool(flags.get("plane_visible", True))

		pl = state.get("plane")
		if pl:
			po = {
				"p0": pl.get("p0", [0,0,1]),
				"u":  pl.get("u",  [1,0,0]),
				"v":  pl.get("v",  [0,0,1]),
				"normal": pl.get("normal", [1,0,0]),
				"width_m":  float(pl.get("width_m", 0.7)),
				"length_m": float(pl.get("length_m", 0.7)),
				"visible":  bool(pl.get("visible", visible)),
				"flip_n": flip_n,
			}
			self.worker.plane_overlay = po

			n = np.asarray(pl.get("normal", [0,1,0]), float).reshape(3)
			p0 = np.asarray(pl.get("p0", [0,0,1]), float).reshape(3)

			n_norm = np.linalg.norm(n)
			if n_norm > 1e-9:
				n = n / n_norm

			d = -float(n @ p0)
			self.worker.plane_n = n
			self.worker.plane_d = d

			self.on_event(f"Plane set: n={n.round(3).tolist()}, d={d:.3f}, size=({po['width_m']:.3f},{po['length_m']:.3f})m, visible={po['visible']}")
		else:
			self.worker.plane_overlay = None
			self.worker.plane_n = None
			self.worker.plane_d = None

		roi = state.get("roi", {})
		if roi:
			self.worker.roi_dict = {
				"x_extend": float(roi.get("x_extend", 0.0)),
				"z_extend": float(roi.get("z_extend", 0.0)),
				"mirror_x": bool(roi.get("mirror_x", False)),
				"mirror_z": bool(roi.get("mirror_z", False)),
				"y_min": float(roi.get("y_min", 0.0)),
				"y_max": float(roi.get("y_max", 0.2))
			}

	def _on_plane_cfg_changed(self, path):
		# Re-arm watcher (Windows sometimes drops it after a change)
		try:
			if os.path.isfile(path) and (path not in self._plane_cfg_watcher.files()):
				self._plane_cfg_watcher.addPath(path)
		except Exception:
			pass
		self._load_plane_from_json()

	def _enter_review(self, path: str):
		# if self.review_page.load_video(path):
		# self.review_page.set_table_plane_from_overlay(self.plane_overlay.copy())
		# self.worker.stop()        # stop live worker while reviewing
		self.worker.track = False
		self.worker.wait(1500)
		self.stack.setCurrentIndex(1)
		self.findChild(QtWidgets.QDockWidget, "rightDock").hide()

	def _return_from_review(self):
		# return to live
		self.worker.track = True
		self.stack.setCurrentIndex(0)
		self.findChild(QtWidgets.QDockWidget, "rightDock").show()
		# restart worker
		# self.worker = CameraWorker()
		# self._wire_worker_signals()  # helper to reconnect signals
		# self.worker.start()

	def _enter_plane_setup(self):
		self.worker.track = False
		self.worker.set_enable_pointcloud(True)
		self._prev_cloud_stride = self.worker._cloud_stride
		self._prev_cloud_every  = self.worker._cloud_every
		self.worker._cloud_stride = 2
		self.worker._cloud_every  = 1

		self.worker.req_depth = True; self.worker.req_rgb = True; self.worker.req_fast = False
		self.chk_depth.setChecked(True); self.chk_rgb.setChecked(True); self.chk_fast.setChecked(False)
		self.worker.request_reconfig(True, True, False)  # depth, rgb, fast

		self.plane_setup._set_worker(self.worker)
		self.findChild(QtWidgets.QDockWidget, "rightDock").hide()
		self.stack.setCurrentWidget(self.plane_setup)
		self.on_event("Plane setup: entering")

	def _plane_canceled(self):
		self.worker.track = True
		self.worker.enable_pointcloud = False
		# restore cloud cadence
		self.worker._cloud_stride = getattr(self, "_prev_cloud_stride", 4)
		self.worker._cloud_every  = getattr(self, "_prev_cloud_every", 3)
		self.chk_draw_plane.setChecked(self.worker.draw_plane)
		self.chk_draw_roi.setChecked(self.worker.draw_roi)

		self.on_event("Plane setup: canceled")
		self.stack.setCurrentIndex(0)
		self.findChild(QtWidgets.QDockWidget, "rightDock").show()

	def _plane_saved(self, box):
		self.worker.track = True
		self.worker.enable_pointcloud = False
		# restore cloud cadence
		self.worker._cloud_stride = getattr(self, "_prev_cloud_stride", 4)
		self.worker._cloud_every  = getattr(self, "_prev_cloud_every", 3)
		self.chk_draw_plane.setChecked(self.worker.draw_plane)
		self.chk_draw_roi.setChecked(self.worker.draw_roi)

		self.on_event(f"Plane setup: saved box {box}")
		self.worker.search_box = box
		self.stack.setCurrentIndex(0)
		self.findChild(QtWidgets.QDockWidget, "rightDock").show()

	def _wire_worker_signals(self):
		self.worker.frame.connect(self.on_frame)
		self.worker.status.connect(self.on_status)
		self.worker.event.connect(self.on_event)
		self.worker.camera_ok.connect(self.on_camera_ok)
		# Re-apply current UI settings to the worker
		self.worker.set_send_arduino(self.send_chk.isChecked())
		# self._apply_crop()
		self._apply_streams()
		# Keep review capture toggle in sync if you use it
		self.worker.enable_record = self.rec_chk.isChecked()
		if getattr(self.worker, "review", None):
			self.worker.review.enabled = self.rec_chk.isChecked()
	
	def _apply_streams(self):
		want_fast  = self.chk_fast.isChecked()
		want_depth = self.chk_depth.isChecked()
		want_rgb   = self.chk_rgb.isChecked()

		# Disable depth/rgb checkboxes while in fast mode
		self.chk_depth.setEnabled(not want_fast)
		self.chk_rgb.setEnabled(not want_fast)

		# Request reconfigure in the worker thread
		self.worker.request_reconfig(want_depth, want_rgb, want_fast)
	
	def _on_rotate90_clicked(self):
		self.worker.rot_k = (self.worker.rot_k + 1) % 4
	
	def _on_toggle_record(self, v):
		v = bool(v)
		self.worker.enable_record = v

		# If the ReviewCapture object exists, flip its internal flag too
		rc = getattr(self.worker, "review", None)
		if rc is not None:
			rc.enabled = v

		self.on_event(f"Review capture {'ENABLED' if v else 'DISABLED'}")

	def _mk_wizard_btn(self, img_path: str, label: str, clicked_slot):
		text = f" {label} WIZARD"
		btn = QtWidgets.QPushButton(text)
		btn.setCursor(QtCore.Qt.PointingHandCursor)
		btn.setToolTip(text)

		# Left icon (relevant image)
		if img_path and os.path.exists(img_path):
			icon = QtGui.QIcon(img_path)
			btn.setIcon(icon)
			btn.setIconSize(QtCore.QSize(20, 20))

		# Optional trailing wizard glyph (right side background image)
		try:
			if os.path.exists(WIZARD_W):
				# space for right glyph; keep things neat
				btn.setStyleSheet(
					"QPushButton {"
					"  text-align: center;"
					"  padding: 6px 28px 6px 10px;"            
					f" background-image: url({WIZARD_W});"
					"  background-repeat: no-repeat;"
					"  background-position: right 6px center;"
					"}"
				)
			else:
				btn.setStyleSheet("QPushButton { text-align: center; padding: 6px 10px; }")
		except Exception:
			btn.setStyleSheet("QPushButton { text-align: center; padding: 6px 10px; }")

		if callable(clicked_slot):
			btn.clicked.connect(clicked_slot)
		return btn

	def _open_gantry_wizard(self):
		self.on_event("Launching Gantry Wizard…")
		# TODO: hook into your gantry simulation entrypoint here

	def _open_ARM_Wizard(self):
		self.on_event("Launching Manipulator Wizard…")
		# TODO: hook into your manipulator simulation entrypoint here

	@QtCore.pyqtSlot(QtGui.QImage)
	def on_frame(self, qimg: QtGui.QImage):
		# Just hand it to the view; it will repaint to current widget size
		self.video.setImage(qimg)

	@QtCore.pyqtSlot(str)
	def on_status(self, s):
		self.cam_status.setText(s)

	@QtCore.pyqtSlot(bool)
	def on_camera_ok(self, ok):
		if not ok:
			self.video.setPixmap(self.SEARCH_CARD.scaled(self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

	@QtCore.pyqtSlot(str)
	def on_event(self, msg):
		ts = time.strftime("%H:%M:%S")
		self.log.appendPlainText(f"[{ts}] {msg}")
	

# def predict_radius(calib_r_px, calib_depth_mm, current_depth_mm):
# 	if current_depth_mm <= 0 or calib_depth_mm is None or calib_r_px is None:
# 		return int(calib_r_px or INIT_SEARCH_CIRCLE_PX)
# 	return int(calib_r_px * (calib_depth_mm / current_depth_mm))

def main():
	app = QtWidgets.QApplication(sys.argv)
	win = MainWindow()
	win.resize(1200, 700)
	win.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()
