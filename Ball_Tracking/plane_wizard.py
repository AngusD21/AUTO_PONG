import sys, os, time, json, math, threading
from dataclasses import dataclass
from collections import deque
import numpy as np
import cv2
import serial

from PyQt5 import QtCore, QtGui, QtWidgets
import pyrealsense2 as rs

import pyqtgraph as pg
import pyqtgraph.opengl as gl


def _points_in_box(xyz, center, half_extents, ypr_deg):
	cx, cy, cz = center
	ex, ey, ez = half_extents
	yaw, pitch, roll = np.deg2rad(ypr_deg)

	cy_, sy = np.cos(yaw),   np.sin(yaw)
	cp, sp = np.cos(pitch), np.sin(pitch)
	cr, sr = np.cos(roll),  np.sin(roll)
	Rz = np.array([[ cy_,-sy, 0],[ sy, cy_, 0],[0,0,1]])
	Ry = np.array([[ cp, 0, sp],[ 0, 1, 0],[-sp, 0, cp]])
	Rx = np.array([[ 1, 0, 0],[ 0, cr,-sr],[0, sr, cr]])
	R = Rz @ Ry @ Rx

	# transform points into box local frame
	X = xyz - np.array([cx,cy,cz])
	Xloc = X @ R
	m = (np.abs(Xloc[:,0]) <= ex) & (np.abs(Xloc[:,1]) <= ey) & (np.abs(Xloc[:,2]) <= ez)
	return xyz[m]


@dataclass
class TablePlane:
	n: np.ndarray   # (3,) unit normal pointing “up” from the table
	d: float        # plane offset
	p0: np.ndarray  # a point on the plane
	u: np.ndarray   # unit axis 1 on plane
	v: np.ndarray   # unit axis 2 on plane

	# Signed height above plane (meters)
	def height(self, p: np.ndarray) -> float:
		return float(self.n @ p + self.d)

	# Local plane coordinates (meters) of p projected to plane
	def uv(self, p: np.ndarray) -> tuple[float,float]:
		# project p onto plane, then dot with u,v from p0
		h = self.height(p)
		p_proj = p - h * self.n
		rel = p_proj - self.p0
		return float(self.u @ rel), float(self.v @ rel)
	

class PlaneSetupWizard(QtWidgets.QWidget):
	saved = QtCore.pyqtSignal(dict)   # emits box dict on Save
	canceled = QtCore.pyqtSignal()    # emits on Cancel

	def __init__(self, worker=None, parent=None):
		super().__init__(parent)
		self.worker = worker

		# Layout: GL on the left, controls on the right
		root = QtWidgets.QHBoxLayout(self)
		root.setContentsMargins(8,8,8,8)
		root.setSpacing(8)

		# GL view
		self.gl = gl.GLViewWidget()
		self.gl.opts['distance'] = 2.0
		self.gl.setBackgroundColor((10,10,12))
		root.addWidget(self.gl, 1)

		# Cloud item
		self.cloud_item = gl.GLScatterPlotItem(size=1.5)  # point size in px
		self.cloud_item.setGLOptions('translucent')
		self.gl.addItem(self.cloud_item)

		# Search box wireframe (12 edges)
		self.box_lines = gl.GLLinePlotItem(width=2.0, mode='lines')
		self.gl.addItem(self.box_lines)

		# Right panel
		panel = QtWidgets.QFrame()
		panel.setObjectName("PlaneRightPanel")
		panel_layout = QtWidgets.QVBoxLayout(panel)
		panel_layout.setContentsMargins(8,8,8,8)
		panel_layout.setSpacing(6)
		root.addWidget(panel, 0)

		title = QtWidgets.QLabel("Plane Setup")
		title.setStyleSheet("font-weight:700; font-size:16px;")
		panel_layout.addWidget(title)

		# Center (m)
		self.cx = self._dspin(-1.0, 1.0, 0.001, 0.0, "Center X (m)")
		self.cy = self._dspin(-1.0, 1.0, 0.001, 0.0, "Center Y (m)")
		self.cz = self._dspin( 0.0, 3.0, 0.001, 1.0, "Center Z (m)")  # z forward

		# Half-extents (m)
		self.ex = self._dspin(0.05, 2.0, 0.001, 0.8, "Half-extent X (m)")
		self.ey = self._dspin(0.01, 1.0, 0.001, 0.2, "Half-extent Y (m)")
		self.ez = self._dspin(0.05, 2.0, 0.001, 0.8, "Half-extent Z (m)")

		# Rotation (deg)
		self.yaw   = self._dspin(-180, 180, 0.5, 0.0, "Yaw (deg)")
		self.pitch = self._dspin(-180, 180, 0.5, 0.0, "Pitch (deg)")
		self.roll  = self._dspin(-180, 180, 0.5, 0.0, "Roll (deg)")

		# Buttons
		btn_row = QtWidgets.QHBoxLayout()
		self.btn_suggest = QtWidgets.QPushButton("Suggest orientation")
		self.btn_save = QtWidgets.QPushButton("Save")
		self.btn_cancel = QtWidgets.QPushButton("Cancel")
		btn_row.addWidget(self.btn_suggest)
		btn_row.addStretch(1)
		btn_row.addWidget(self.btn_cancel)
		btn_row.addWidget(self.btn_save)
		panel_layout.addLayout(btn_row)

		panel_layout.addStretch(1)

		# Timer to refresh cloud/box at ~15 Hz
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self._tick)
		self.timer.start(66)

		# Wire up
		for w in (self.cx,self.cy,self.cz,self.ex,self.ey,self.ez,self.yaw,self.pitch,self.roll):
			w.valueChanged.connect(self._update_box)

		self.btn_suggest.clicked.connect(self._suggest_orientation)
		self.btn_save.clicked.connect(self._save)
		self.btn_cancel.clicked.connect(self.canceled.emit)

		# Initial draw
		self._update_box()
		
	def _set_worker(self, worker):
		self.worker = worker

	def _dspin(self, lo, hi, step, val, label):
		box = QtWidgets.QGroupBox(label)
		lay = QtWidgets.QHBoxLayout(box)
		spin = QtWidgets.QDoubleSpinBox()
		spin.setRange(lo, hi)
		spin.setSingleStep(step)
		spin.setDecimals(4)
		spin.setValue(val)
		spin.setAlignment(QtCore.Qt.AlignRight)
		lay.addWidget(spin)
		self.layout().itemAt(1).widget().layout().insertWidget(self.layout().itemAt(1).widget().layout().count()-1, box)
		return spin

	def _tick(self):
		# Update cloud
		xyz = self.worker.get_point_cloud_snapshot()
		if xyz is not None and xyz.shape[0] > 0:
			self.cloud_item.setData(pos=xyz)
		# box redraw
		self._update_box()

	def _update_box(self):
		# Build wireframe box from center/half-extents/rotation
		c = np.array([self.cx.value(), self.cy.value(), self.cz.value()], float)
		e = np.array([self.ex.value(), self.ey.value(), self.ez.value()], float)

		# Rotation R = Rz(yaw)*Ry(pitch)*Rx(roll) in camera frame
		yaw, pitch, roll = np.deg2rad([self.yaw.value(), self.pitch.value(), self.roll.value()])
		cy, sy = np.cos(yaw),   np.sin(yaw)
		cp, sp = np.cos(pitch), np.sin(pitch)
		cr, sr = np.cos(roll),  np.sin(roll)
		Rz = np.array([[ cy,-sy, 0],[ sy, cy, 0],[0,0,1]])
		Ry = np.array([[ cp, 0, sp],[ 0, 1, 0],[-sp, 0, cp]])
		Rx = np.array([[ 1, 0, 0],[ 0, cr,-sr],[0, sr, cr]])
		R = Rz @ Ry @ Rx

		# 8 corners in local box space (+/- ex,ey,ez)
		corners_local = np.array([[ sx, sy, sz] for sx in (-e[0],e[0]) for sy in (-e[1],e[1]) for sz in (-e[2],e[2])])
		corners = (corners_local @ R.T) + c

		# 12 edges index pairs
		idx = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
		lines = []
		for a,b in idx:
			lines.append(corners[a])
			lines.append(corners[b])
		lines = np.asarray(lines, float)

		self.box_lines.setData(pos=lines, color=(1,1,1,0.9))

	def _suggest_orientation(self):
		"""
		Quick PCA suggestion:
		- Take the current cloud, do PCA.
		- Align yaw to the first principal axis projection on XZ.
		- Set pitch roughly to make the second axis more horizontal.
		It's a heuristic, just to get close.
		"""
		xyz = self.worker.get_point_cloud_snapshot()
		if xyz is None or xyz.shape[0] < 200:
			QtWidgets.QMessageBox.information(self, "Suggest", "Not enough points for suggestion.")
			return
		P = xyz - xyz.mean(axis=0, keepdims=True)
		try:
			U, S, Vt = np.linalg.svd(P, full_matrices=False)
			axes = Vt  # rows: principal directions
			# Use first axis as "long" direction; set yaw from its XZ projection
			a1 = axes[0]
			yaw = math.degrees(math.atan2(a1[0], a1[2]))  # rotate around Y so Z aligns with a1's forward
			# Keep pitch/roll 0 for now (simple)
			self.yaw.setValue(yaw)
			# Position the box center at median Z of cloud
			med = np.median(xyz, axis=0)
			self.cx.setValue(float(med[0]))
			self.cy.setValue(float(med[1]))
			self.cz.setValue(float(med[2]))
		except Exception:
			pass

	def _save(self):
		out = dict(
			center=[self.cx.value(), self.cy.value(), self.cz.value()],
			half_extents=[self.ex.value(), self.ey.value(), self.ez.value()],
			ypr_deg=[self.yaw.value(), self.pitch.value(), self.roll.value()]
		)
		self.saved.emit(out)
