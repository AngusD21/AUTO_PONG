from collections import deque
import os, json, math, cv2, numpy as np
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from Ball_Tracking.graphics_objects import AspectImageView, AxisGlyph, CollapsibleCard, bgr_np_to_qimage
from Ball_Tracking.plane_math import (
	colourise_depth,
	project_points_px,
	project_points_px_masked,
	_project_uvn,
	TablePlane,
)

PLANE_W = "Assets/PLANE_W.png"
PLAY_W = "Assets/PLAY_W.png"
WIZARD_W = "Assets/WIZARD_W.png"

# Review capture
PREBUFFER_SEC = 1.0
POSTBUFFER_SEC = 1.0
REVIEW_PATH_HORIZON_S = 0.8
SEARCH_RADIUS_PX = 30

# ----------------------------- Helpers -----------------------------
@dataclass
class Intrinsics:
	fx: float
	fy: float
	cx: float
	cy: float

	@staticmethod
	def from_any(intr_obj) -> Optional["Intrinsics"]:
		if intr_obj is None:
			return None
		# RealSense intrinsics-like
		if hasattr(intr_obj, "fx") and hasattr(intr_obj, "fy"):
			cx = float(getattr(intr_obj, "ppx", getattr(intr_obj, "cx", 0.0)))
			cy = float(getattr(intr_obj, "ppy", getattr(intr_obj, "cy", 0.0)))
			return Intrinsics(float(intr_obj.fx), float(intr_obj.fy), cx, cy)
		# dict
		if isinstance(intr_obj, dict):
			return Intrinsics(
				float(intr_obj.get("fx", intr_obj.get("Fx", 0.0))),
				float(intr_obj.get("fy", intr_obj.get("Fy", 0.0))),
				float(intr_obj.get("ppx", intr_obj.get("cx", 0.0))),
				float(intr_obj.get("ppy", intr_obj.get("cy", 0.0))),
			)
		# 3x3 K
		K = np.asarray(intr_obj, float)
		if K.shape == (3, 3):
			return Intrinsics(float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]))
		return None

# Depth u16 (meters = depth_u16 * depth_scale)

def _depth_to_points(depth_u16: np.ndarray, intr: Intrinsics, depth_scale: float) -> np.ndarray:
	H, W = depth_u16.shape
	ys, xs = np.indices((H, W))
	Z = depth_u16.astype(np.float32) * float(depth_scale)
	valid = Z > 1e-6
	X = (xs - intr.cx) * Z / intr.fx
	Y = (ys - intr.cy) * Z / intr.fy
	P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
	return P[valid.reshape(-1)]

# ------------------------- Review data model ------------------------

class ReviewData:
	"""Container for one captured session."""
	def __init__(self):
		self.session_dir: Optional[str] = None
		self.video_path: Optional[str] = None
		self.meta_path: Optional[str] = None
		self.meta: List[Dict] = []
		self.depth_frames: Optional[np.ndarray] = None
		self.intrinsics: Optional[Intrinsics] = None
		self.depth_scale: Optional[float] = None
		self.table_plane: Optional[TablePlane] = None

	def count(self) -> int:
		if self.depth_frames is not None:
			return int(self.depth_frames.shape[0])
		if self.meta:
			return len(self.meta)
		return 0

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
		
		if len(self.meta) > 0:
			m0 = self.meta[0]
			# Intrinsics + scale
			m0.setdefault("depth_scale", float(self.depth_scale))
			if hasattr(self.intrinsics, "fx"):
				m0.setdefault("intrinsics", {
					"fx": float(self.intrinsics.fx),
					"fy": float(self.intrinsics.fy),
					"ppx": float(getattr(self.intrinsics, "ppx", getattr(self.intrinsics, "cx", 0.0))),
					"ppy": float(getattr(self.intrinsics, "ppy", getattr(self.intrinsics, "cy", 0.0))),
				})
			# Plane (if provided via setter below)
			if hasattr(self, "_plane_overlay") and self._plane_overlay:
				po = self._plane_overlay
				m0.setdefault("pl_p0", list(map(float, po["p0"])))
				m0.setdefault("pl_u",  list(map(float, po["u"])))
				m0.setdefault("pl_v",  list(map(float, po["v"])))
				m0.setdefault("pl_n",  list(map(float, po["normal"])))
		
		if len(self.frames) > 0:
			depth_npz = os.path.join(sess_dir, "depth_frames.npz")
			arr = np.stack(self.frames, axis=0).astype(np.uint16)
			np.savez_compressed(depth_npz, depth_u16=arr)

		if len(self.frames) > 0:
			h, w = self.frames[0].shape
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
			for depth_u16, m in zip(self.frames, self.meta):
				frame_bgr = colourise_depth(depth_u16, self.depth_scale)
				# Overlay: measured center / search circle
				if m.get("meas_px") is not None:
					cx, cy = m["meas_px"]
					self.meta["meas_px"] = (int(cx), int(cy))
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
	
	def set_table_plane(self, overlay: dict):
		# overlay with keys: 'p0','u','v','normal'
		self._plane_overlay = overlay.copy()

# ---------------------------- Main Widget ---------------------------

class ReviewWizard(QtWidgets.QWidget):
	request_return = QtCore.pyqtSignal()  # Return to main UI

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setObjectName("ReviewWizard")

		self.data = ReviewData()
		self.cap: Optional[cv2.VideoCapture] = None
		self.cur_idx: int = 0
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self._on_tick)

		# Accumulate state
		self.accumulate_enabled = False
		self.history_len = 200  # max frames to keep for onion-skin

		self._build_ui()
		self._wire_events()

	# ---------------- UI ----------------
	def _build_ui(self):
		root = QtWidgets.QHBoxLayout(self)
		root.setContentsMargins(8, 8, 8, 8)
		root.setSpacing(8)

		# Left: tabs + scrubber
		left = QtWidgets.QVBoxLayout(); left.setContentsMargins(0,0,0,0); left.setSpacing(6)
		root.addLayout(left, 1)

		self.tabs = QtWidgets.QTabWidget()
		self.tabs.setTabPosition(QtWidgets.QTabWidget.North)
		self.tabs.setDocumentMode(True)
		left.addWidget(self.tabs, 1)

		# --- Video tab ---
		tab_video = QtWidgets.QWidget()
		vlay = QtWidgets.QVBoxLayout(tab_video); vlay.setContentsMargins(0,0,0,0); vlay.setSpacing(6)
		self.lbl_video = QtWidgets.QLabel(); self.lbl_video.setAlignment(QtCore.Qt.AlignCenter)
		self.lbl_video.setStyleSheet("background:#000;")
		self.lbl_video.setMinimumSize(640, 480)
		vlay.addWidget(self.lbl_video, 1)
		self.tabs.addTab(tab_video, "Video")

		# --- 3D tab ---
		tab_3d = QtWidgets.QWidget()
		t3d = QtWidgets.QVBoxLayout(tab_3d); t3d.setContentsMargins(0,0,0,0); t3d.setSpacing(6)
		self.view3d = gl.GLViewWidget()
		self.view3d.setBackgroundColor((10, 10, 12))
		self.view3d.opts['distance'] = 2.0
		t3d.addWidget(self.view3d, 1)
		# cloud
		self.gl_cloud = gl.GLScatterPlotItem(size=1.5)
		self.gl_cloud.setGLOptions('translucent')
		self.view3d.addItem(self.gl_cloud)
		# path
		self.gl_path = gl.GLLinePlotItem(mode='line_strip', width=2.0)
		self.view3d.addItem(self.gl_path)
		self.tabs.addTab(tab_3d, "3D")

		# --- Projected tab (table UV) ---
		tab_proj = QtWidgets.QWidget()
		tproj = QtWidgets.QVBoxLayout(tab_proj); tproj.setContentsMargins(0,0,0,0); tproj.setSpacing(6)
		self.plot_uv = pg.PlotWidget()
		self.sp_uv = pg.ScatterPlotItem(size=3, pxMode=True)
		self.plot_uv.addItem(self.sp_uv)
		self.plot_uv.setLabel('bottom', 'v (m)'); self.plot_uv.setLabel('left', 'u (m)')
		tproj.addWidget(self.plot_uv, 1)
		# projected path
		self.curve_uv = self.plot_uv.plot([], [], pen=pg.mkPen(width=2))
		self.tabs.addTab(tab_proj, "Projected")

		# Scrubber + controls
		ctrl = QtWidgets.QHBoxLayout(); ctrl.setContentsMargins(0,0,0,0); ctrl.setSpacing(8)
		self.btn_play = QtWidgets.QPushButton("Play"); self.btn_play.setCheckable(True)
		self.lbl_time = QtWidgets.QLabel("00:00 / 00:00"); self.lbl_time.setMinimumWidth(120)
		ctrl.addWidget(self.btn_play)
		ctrl.addStretch(1)
		ctrl.addWidget(self.lbl_time)
		left.addLayout(ctrl)

		self.scrub = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.scrub.setRange(0, 0)
		self.scrub.setSingleStep(1)
		left.addWidget(self.scrub)

		# Right panel
		right_panel = QtWidgets.QFrame(); right_panel.setObjectName("ReviewRightPanel")
		right = QtWidgets.QVBoxLayout(right_panel)
		right.setContentsMargins(8, 8, 8, 8); right.setSpacing(6)
		root.addWidget(right_panel, 0)

		# Title header (match Plane Wizard look)
		spaced = QtWidgets.QHBoxLayout(); spaced.setContentsMargins(0,0,0,0); spaced.setSpacing(2)
		if os.path.exists(PLAY_W):
			small_logo = QtWidgets.QLabel()
			pm = QtGui.QPixmap(PLAY_W).scaledToHeight(28, QtCore.Qt.SmoothTransformation)
			small_logo.setPixmap(pm)
	
		for ch in " REVIEW WIZARD ":
			lbl = QtWidgets.QLabel(ch); lbl.setObjectName("titleLabel"); spaced.addWidget(lbl, 1, QtCore.Qt.AlignCenter)
		spaced.addWidget(small_logo, 1, QtCore.Qt.AlignCenter)
		spaced.addWidget(QtWidgets.QLabel(' '), 1, QtCore.Qt.AlignCenter)
		right.addLayout(spaced)

		# Load row
		self.btn_load = QtWidgets.QPushButton("Load…")
		right.addWidget(self.btn_load)

		# Options card
		opts = QtWidgets.QGroupBox("Display Options")
		f = QtWidgets.QFormLayout(opts); f.setLabelAlignment(QtCore.Qt.AlignLeft)

		# Colour by
		self.cmb_colour = QtWidgets.QComboBox()
		self.cmb_colour.addItems(["Depth", "Region", "Ball points"])  # applies where relevant
		f.addRow("Colour by", self.cmb_colour)

		# Toggles
		self.chk_plane   = QtWidgets.QCheckBox("Draw table plane")
		self.chk_path    = QtWidgets.QCheckBox("Draw predicted path")
		self.chk_ghost   = QtWidgets.QCheckBox("Draw ball ghost (pos+radius)")
		self.chk_accum   = QtWidgets.QCheckBox("Accumulate (onion-skin)")
		for c in (self.chk_plane, self.chk_path, self.chk_ghost, self.chk_accum):
			c.setChecked(True)
			f.addRow(c)
		right.addWidget(opts)

		right.addStretch(1)

		# Return button bottom
		self.btn_return = QtWidgets.QPushButton("← Return to Main UI")
		right.addWidget(self.btn_return)

		# initial state
		right_panel.setFixedWidth(360)

	def _wire_events(self):
		self.btn_return.clicked.connect(self.request_return.emit)
		self.btn_load.clicked.connect(self._on_load)

		self.btn_play.toggled.connect(self._on_play_toggled)
		self.scrub.valueChanged.connect(self._on_scrub)

		# repaint on toggles
		self.cmb_colour.currentIndexChanged.connect(lambda _: self._redraw_current())
		self.chk_plane.toggled.connect(lambda _: self._redraw_current())
		self.chk_path.toggled.connect(lambda _: self._redraw_current())
		self.chk_ghost.toggled.connect(lambda _: self._redraw_current())
		self.chk_accum.toggled.connect(self._on_accum_toggled)

	# ---------------- Load & session ----------------
	def _on_load(self):
		# Allow loading either a session folder or a meta.json file
		path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select review session folder")
		if not path:
			return
		sess_dir = path
		meta = os.path.join(sess_dir, "meta.json")
		video = os.path.join(sess_dir, "review.mp4")
		depth_npz = os.path.join(sess_dir, "depth_frames.npz")  # optional
		if not os.path.exists(meta):
			QtWidgets.QMessageBox.warning(self, "Missing file", "meta.json not found in the selected folder.")
			return
		if not os.path.exists(video):
			QtWidgets.QMessageBox.warning(self, "Missing file", "review.mp4 not found in the selected folder.")
			return

		# Close previous
		if self.cap is not None:
			self.cap.release(); self.cap = None

		self.data = ReviewData()
		self.data.session_dir = sess_dir
		self.data.meta_path = meta
		self.data.video_path = video
		try:
			with open(meta, "r", encoding="utf-8") as f:
				self.data.meta = json.load(f)
		except Exception as e:
			QtWidgets.QMessageBox.critical(self, "Invalid meta.json", str(e))
			return

		# Intrinsics / scale (try meta[0])
		if self.data.meta:
			m0 = self.data.meta[0]
			self.data.depth_scale = float(m0.get("depth_scale", m0.get("DepthScale", 0.001))) or 0.001
			self.data.intrinsics = Intrinsics.from_any(m0.get("intrinsics") or m0.get("Intrinsics") or m0.get("K"))

			# Optional table plane info in meta (p0,u,v,n)
			if all(k in m0 for k in ("pl_p0","pl_u","pl_v","pl_n")):
				p0 = np.array(m0["pl_p0"], float)
				u  = np.array(m0["pl_u"],  float)
				v  = np.array(m0["pl_v"],  float)
				n  = np.array(m0["pl_n"],  float)
				d  = -float(n @ p0)
				self.data.table_plane = TablePlane(n=n/np.linalg.norm(n), d=d, p0=p0, u=u/np.linalg.norm(u), v=v/np.linalg.norm(v))

		# Optional depth frames pack
		if os.path.exists(depth_npz):
			try:
				pack = np.load(depth_npz)
				self.data.depth_frames = pack.get("depth_u16")
			except Exception:
				self.data.depth_frames = None

		# Open video
		self.cap = cv2.VideoCapture(self.data.video_path)
		if not self.cap.isOpened():
			QtWidgets.QMessageBox.critical(self, "Open failed", f"Could not open video: {self.data.video_path}")
			return

		# Set ranges
		total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		self.scrub.blockSignals(True)
		self.scrub.setRange(0, max(0, total - 1))
		self.scrub.setValue(0)
		self.scrub.blockSignals(False)
		self._update_time_label(0, total)

		# Show first frame
		self.cur_idx = 0
		self._show_video_frame(0)
		self._redraw_current()

	# ---------------- Playback ----------------
	def _fps(self) -> float:
		if self.cap is None:
			return 30.0
		fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
		return 1.0 if fps <= 1e-6 else fps

	def _on_play_toggled(self, playing: bool):
		if playing:
			self.btn_play.setText("Pause")
			self.timer.start(int(1000.0 / max(1.0, self._fps())))
		else:
			self.btn_play.setText("Play")
			self.timer.stop()

	def _on_accum_toggled(self, checked: bool):
		self.accumulate_enabled = bool(checked)
		self._redraw_current()

	def _on_tick(self):
		if self.cap is None:
			return
		pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
		ok, frame = self.cap.read()
		if not ok:
			self.btn_play.setChecked(False)
			return
		self.cur_idx = min(pos, self.scrub.maximum())
		self._render_video_with_overlays(frame, self.cur_idx)
		self.scrub.blockSignals(True)
		self.scrub.setValue(self.cur_idx)
		self.scrub.blockSignals(False)
		total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		self._update_time_label(self.cur_idx, total)
		self._redraw_current(skip_video=True)

	def _on_scrub(self, idx: int):
		self.cur_idx = int(idx)
		if self.cap is not None:
			self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_idx)
			ok, frame = self.cap.read()
			if ok:
				self._render_video_with_overlays(frame, self.cur_idx)
		total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap is not None else (self.scrub.maximum()+1)
		self._update_time_label(self.cur_idx, total)
		self._redraw_current(skip_video=True)

	def _update_time_label(self, idx: int, total: int):
		fps = self._fps()
		cur_s = idx / max(1.0, fps)
		tot_s = total / max(1.0, fps)
		def mmss(t):
			return f"{int(t//60):02d}:{int(t%60):02d}"
		self.lbl_time.setText(f"{mmss(cur_s)} / {mmss(tot_s)}")

	# ---------------- Overlays (Video) ----------------
	def _render_video_with_overlays(self, frame_bgr: np.ndarray, idx: int):
		img = frame_bgr.copy()

		# Onion-skin accumulate (older = lower alpha)
		j0 = 0 if self.chk_accum.isChecked() else idx
		j1 = idx
		total = max(1, j1 - j0 + 1)

		for j, t in enumerate(range(j0, j1+1)):
			m = self._meta_at(t)
			if m is None:
				continue
			age = (j1 - t)
			a = 1.0 - (age / max(1.0, total))  # 1 .. small
			alpha = max(0.15, min(1.0, a))

			# Measured/ball points
			meas = m.get("meas_px") or m.get("ball_px")
			if meas is not None and self.cmb_colour.currentText() == "Ball points":
				cx, cy = int(meas[0]), int(meas[1])
				col = (0, int(255*alpha), 0)
				cv2.circle(img, (cx, cy), 4, col, 2, lineType=cv2.LINE_AA)

			# Predicted circle (ghost)
			if self.chk_ghost.isChecked():
				pred_px = m.get("pred_px")
				pred_r  = m.get("pred_r_px") or m.get("pred_radius_px")
				if pred_px is not None and pred_r is not None:
					g = int(220*alpha)
					cv2.circle(img, (int(pred_px[0]), int(pred_px[1])), int(pred_r), (g,g,g), 1, lineType=cv2.LINE_AA)

		qimg = bgr_np_to_qimage(img)
		self.lbl_video.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
			self.lbl_video.width(), self.lbl_video.height(),
			QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

	# ---------------- 3D & Projected ----------------
	def _redraw_current(self, skip_video: bool=False):
		idx = self.cur_idx
		m = self._meta_at(idx)

		# 3D cloud
		pts = None
		if self.data.depth_frames is not None and self.data.intrinsics is not None and self.data.depth_scale is not None:
			depth = self.data.depth_frames[min(idx, self.data.depth_frames.shape[0]-1)]
			pts = _depth_to_points(depth, self.data.intrinsics, self.data.depth_scale)
			if pts is not None and pts.size:
				# Colouring mode
				if self.cmb_colour.currentText() == "Depth":
					z = pts[:,2]
					z = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z) + 1e-6)
					cols = np.c_[z, 1.0-z, 0.4*np.ones_like(z), 0.9*np.ones_like(z)]
				else:
					cols = np.tile(np.array([[0.7,0.8,1.0,0.9]]), (pts.shape[0],1))
				self.gl_cloud.setData(pos=pts, color=cols)
			else:
				self.gl_cloud.setData(pos=np.zeros((0,3)))
		else:
			# No depth available for session
			self.gl_cloud.setData(pos=np.zeros((0,3)))

		# 3D path / ghost
		if self.chk_path.isChecked():
			path_pts = self._get_path_segment_up_to(idx)
			if path_pts is not None and len(path_pts):
				P = np.array(path_pts, float)
				self.gl_path.setData(pos=P, color=(0.9,0.9,0.2,0.95))
			else:
				self.gl_path.setData(pos=np.zeros((0,3)))
		else:
			self.gl_path.setData(pos=np.zeros((0,3)))

		# Projected (onto table u,v)
		if self.data.table_plane is not None:
			# project either full cloud or just path
			uvs = []
			if pts is not None and pts.size:
				uvn = _project_uvn(pts, self._plane_R(), 0,0,0, False, np.zeros(3), self.data.table_plane.p0)
				uvs = uvn[:,[0,2]]  # u,v
				self.sp_uv.setData(uvs[:,1] if uvs.__len__() else [], uvs[:,0] if uvs.__len__() else [])
			else:
				self.sp_uv.setData([], [])

			if self.chk_path.isChecked():
				path_pts = self._get_path_segment_up_to(idx)
				if path_pts is not None and len(path_pts):
					uvn_p = _project_uvn(np.asarray(path_pts, float), self._plane_R(), 0,0,0, False, np.zeros(3), self.data.table_plane.p0)
					uu, vv = uvn_p[:,0], uvn_p[:,2]
					self.curve_uv.setData(vv, uu)
				else:
					self.curve_uv.setData([], [])
		else:
			self.sp_uv.setData([], [])
			self.curve_uv.setData([], [])

	def _plane_R(self) -> np.ndarray:
		# Build [u n v] from plane
		pl = self.data.table_plane
		if pl is None:
			return np.eye(3)
		return np.column_stack([pl.u, pl.n, pl.v])

	# ---------------- Utilities ----------------
	def _meta_at(self, idx: int) -> Optional[Dict]:
		if not self.data.meta:
			return None
		i = max(0, min(idx, len(self.data.meta)-1))
		return self.data.meta[i]

	def _get_path_segment_up_to(self, idx: int) -> Optional[List[List[float]]]:
		"""Return list of 3D points (world) representing predicted/measured path up to idx.
		Supports two meta layouts:
		1) Per-frame key 'pred_world' = [x,y,z] (single point per frame)
		2) Global key in meta[0]: 'predicted_path' = Nx3 list
		"""
		if not self.data.meta:
			return None
		# Global path
		g = self.data.meta[0].get("predicted_path")
		if isinstance(g, list) and len(g) and isinstance(g[0], (list, tuple)):
			N = min(idx+1, len(g)) if self.chk_accum.isChecked() else 1
			return g[:N]
		# Accumulate per-frame
		pts = []
		N = idx+1 if self.chk_accum.isChecked() else 1
		j0 = 0 if self.chk_accum.isChecked() else idx
		for j in range(j0, min(len(self.data.meta), idx+1)):
			m = self.data.meta[j]
			p = m.get("pred_world") or m.get("ball_world")
			if p is not None:
				pts.append([float(p[0]), float(p[1]), float(p[2])])
		return pts

	# def _set_plane_from_overlay(self, overlay: dict):
	# 	"""overlay: {'p0','u','v','normal'} in world coords."""
	# 	self.plane_overlay = overlay.copy()
	# 	n = np.asarray(overlay['normal'], float).reshape(3)
	# 	n /= (np.linalg.norm(n) + 1e-12)
	# 	p0 = np.asarray(overlay['p0'], float).reshape(3)
	# 	self.plane_n = n
	# 	self.plane_d = -float(n @ p0)
# ---------------- Convenience launcher ----------------
if __name__ == "__main__":
	import sys
	app = QtWidgets.QApplication(sys.argv)
	w = ReviewWizard()
	w.resize(1280, 720)
	w.show()
	sys.exit(app.exec_())
