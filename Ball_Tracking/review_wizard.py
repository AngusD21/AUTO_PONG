import sys, os, time, json, math, threading
from dataclasses import dataclass
from collections import deque
import numpy as np
import cv2
import serial

from PyQt5 import QtCore, QtGui, QtWidgets
import pyrealsense2 as rs

from Ball_Tracking.graphics_objects import bgr_np_to_qimage
from Ball_Tracking.plane_math import colourise_depth

PLANE_W = "Assets/PLANE_W.png"
PLAY_W = "Assets/PLAY_W.png"
WIZARD_W = "Assets/WIZARD_W.png"

PREBUFFER_SEC = 1.0
POSTBUFFER_SEC = 1.0
CAPTURE_KEEP_EVERY = 2
REVIEW_OUT_DIR = "captures"
REVIEW_PATH_HORIZON_S = 0.8
SEARCH_RADIUS_PX = 200

# ===================== Review Capture Object =====================
class ReviewCapture:
	def __init__(self, enabled, fps, depth_scale, intrinsics, out_dir = REVIEW_OUT_DIR, keep_every = CAPTURE_KEEP_EVERY):
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

# =========================== HELPERS ===========================
class VideoLabel(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: black;")
# ===============================================================


# =========================== REVIEW UI =========================
class ReviewPage(QtWidgets.QWidget):
    """Standalone review player: big video, long slider, play/pause, arrow-key frame step."""
    request_return = QtCore.pyqtSignal()  # emitted when user clicks Return

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        # Video area
        self.video = VideoLabel() 
        v.addWidget(self.video, 1)

        # Long scrub + controls
        ctrl = QtWidgets.QHBoxLayout()
        self.btn_return = QtWidgets.QPushButton("← Return to Live")
        self.btn_play   = QtWidgets.QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.lbl_time   = QtWidgets.QLabel("00:00 / 00:00")
        self.lbl_time.setMinimumWidth(120)
        ctrl.addWidget(self.btn_return)
        ctrl.addStretch(1)
        ctrl.addWidget(self.btn_play)
        ctrl.addSpacing(8)
        ctrl.addWidget(self.lbl_time)
        v.addLayout(ctrl)

        self.scrub = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrub.setRange(0, 0)
        self.scrub.setSingleStep(1)
        v.addWidget(self.scrub)

        # Player state
        self.cap = None
        self.total = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)

        # Wire
        self.btn_return.clicked.connect(self.request_return.emit)
        self.btn_play.toggled.connect(self._toggle_play)
        self.scrub.valueChanged.connect(self._scrub_to)

        # Keyboard focus for arrow keys
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def load_video(self, path: str):
        # close previous
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(path)
        if not self.cap or not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Load failed", f"Could not open:\n{path}")
            return False
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.scrub.setRange(0, max(0, self.total - 1))
        self._show_frame(0)
        self.btn_play.setChecked(False)  # paused by default
        self._update_time_label(0)
        return True

    def _update_time_label(self, idx):
        if not self.cap or self.total <= 0:
            self.lbl_time.setText("00:00 / 00:00")
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        cur_s = idx / fps
        tot_s = self.total / fps
        def mmss(t): return f"{int(t//60):02d}:{int(t%60):02d}"
        self.lbl_time.setText(f"{mmss(cur_s)} / {mmss(tot_s)}")

    def _show_frame(self, idx):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = self.cap.read()
        if not ok: return
        qimg = bgr_np_to_qimage(frame)
        self.video.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video.width(), self.video.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))

    def _tick(self):
        if not self.cap: return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = self.cap.read()
        if not ok:
            self.btn_play.setChecked(False)  # stop
            return
        qimg = bgr_np_to_qimage(frame)
        self.video.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video.width(), self.video.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))
        self.scrub.blockSignals(True)
        self.scrub.setValue(min(self.total-1, pos))
        self.scrub.blockSignals(False)
        self._update_time_label(pos)

    def _toggle_play(self, playing: bool):
        if playing:
            self.btn_play.setText("Pause")
            self.timer.start(33)  # ~30fps
        else:
            self.btn_play.setText("Play")
            self.timer.stop()

    def _scrub_to(self, idx: int):
        self._show_frame(int(idx))
        self._update_time_label(int(idx))

    # Arrow keys: when paused, step ±1 frame
    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if not self.cap:
            return super().keyPressEvent(e)
        if e.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            if self.timer.isActive():
                return  # ignore while playing
            cur = self.scrub.value()
            delta = -1 if e.key() == QtCore.Qt.Key_Left else 1
            nxt = int(np.clip(cur + delta, 0, max(0, self.total-1)))
            self.scrub.setValue(nxt)
            self._show_frame(nxt)
            self._update_time_label(nxt)
        else:
            super().keyPressEvent(e)

    def _load_review(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open review.mp4", REVIEW_OUT_DIR, "MP4 files (*.mp4)")
        if not path:
            return
        self.on_event(f"Loaded review: {path}")
        self._enter_review(path)