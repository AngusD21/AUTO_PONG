import sys, os, time, json, math, threading
from dataclasses import dataclass
from collections import deque
import numpy as np
import cv2
import serial

from PyQt5 import QtCore, QtGui, QtWidgets
import pyrealsense2 as rs


# =========================== HELPERS ===========================
def qimage_from_bgr(img_bgr):
	h, w = img_bgr.shape[:2]
	rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
	return qimg.copy()

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
        qimg = qimage_from_bgr(frame)
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
        qimg = qimage_from_bgr(frame)
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