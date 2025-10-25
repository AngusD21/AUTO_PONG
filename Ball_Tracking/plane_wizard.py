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

from Ball_Tracking.graphics_objects import  CollapsibleCard, RangeSlider, AxisGlyph, AXIS_COLORS
from Ball_Tracking.graphics_objects import  _backing_spin, _colored_span, _link_half_to_span, _link_span_to_half,make_full_slider_row, _span_box

#======================= PLANE OBJECT =========================
@dataclass
class TablePlane:
	n: np.ndarray   # unit normal pointing up from the table
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
#===========================================================	

class PlaneSetupWizard(QtWidgets.QWidget):
	saved = QtCore.pyqtSignal(dict)   # emits box dict on Save
	canceled = QtCore.pyqtSignal()    # emits on Cancel

	def __init__(self, worker=None, parent=None, config_path=None):
		super().__init__(parent)
		self.worker = worker
		self.config_path = (config_path or os.path.join(os.path.dirname(__file__), "plane_wizard_state.json"))

		self.cam_axes = []
		self.box_axes = []
		self.box_visible = True
		self.pts_visible = True
		self.roi_visible = True
		self._visual_flip_y = True

		# Table plane state
		self.plane = None     
		self.pl_visible = True
		self.pl_size_x = 0.70 
		self.pl_size_z = 0.70 

		self._pl_R_base = np.eye(3)     # columns [u, n, v]
		self._pl_p0_base = np.zeros(3)
		self._pl_off_local = np.zeros(3)  # (dx, dy, dz) in local (u, n, v)
		self._pl_yaw = 0.0
		self._pl_pitch = 0.0
		self._pl_roll = 0.0

		# Layout: GL on the left, controls on the right
		root = QtWidgets.QHBoxLayout(self)
		root.setContentsMargins(8,8,8,8)
		root.setSpacing(8)

		# ---------- Left panel (GL) ----------
		if 1: 
			self.gl = gl.GLViewWidget()
			self.gl.opts['distance'] = 2.0
			self.gl.setBackgroundColor((10,10,12))
			self.gl.installEventFilter(self)
			# root.addWidget(self.gl, 1)

			# left column with GL on top and axis HUD at bottom
			left_col = QtWidgets.QWidget()
			left_v = QtWidgets.QVBoxLayout(left_col)
			left_v.setContentsMargins(0,0,0,0); left_v.setSpacing(6)
			left_v.addWidget(self.gl, 1)

			# axis HUD row
			hud = QtWidgets.QWidget()
			hud_h = QtWidgets.QHBoxLayout(hud); hud_h.setContentsMargins(0,0,0,0); hud_h.setSpacing(8)
			self.glyph_cam  = AxisGlyph("Camera")
			self.glyph_box  = AxisGlyph("Box")
			self.glyph_plane= AxisGlyph("Plane")
			hud_h.addWidget(self.glyph_cam); hud_h.addWidget(self.glyph_box); hud_h.addWidget(self.glyph_plane)
			left_v.addWidget(hud, 0)

			root.addWidget(left_col, 1)

			# Start in a pose that’s behind/above the camera looking down +Z
			self.gl.opts['fov'] = 45
			self._set_default_view()

			# Cloud item
			self.cloud_item = gl.GLScatterPlotItem(size=1.5)  # point size in px
			self.cloud_item.setGLOptions('translucent')
			self.gl.addItem(self.cloud_item)

			# Search box wireframe (12 edges)
			self.box_lines = gl.GLLinePlotItem(width=2.0, mode='lines')
			self.gl.addItem(self.box_lines)

			# Camera body (0.05 x 0.05 x 0.20 m)
			self.cam_mesh = self._make_box_mesh(0.15, 0.05, 0.05, color=(0.85,0.85,0.95,0.9))
			self.gl.addItem(self.cam_mesh)

			# Camera forward normal (along +Z)
			self.cam_forward = gl.GLLinePlotItem(mode='lines', width=2.0)
			self.gl.addItem(self.cam_forward)
			# Set it once; it's fixed in world-coords
			cam_forward_pts = np.array([[0,0,0], [0,0,0.35]], dtype=float)  # 35 cm ray
			self.cam_forward.setData(pos=cam_forward_pts, color=(0.3,0.7,1.0,0.95))

			# Camera line
			L = 0.25
			self.cam_axes.append(self._make_axis(np.zeros(3), np.array([0,0,L]), (0,0,1,0.9)))

			self.global_axis_x = gl.GLLinePlotItem(mode='line_strip', width=1.2)
			self.global_axis_y = gl.GLLinePlotItem(mode='line_strip', width=1.2)
			self.global_axis_z = gl.GLLinePlotItem(mode='line_strip', width=1.2)
			self.gl.addItem(self.global_axis_x)
			self.gl.addItem(self.global_axis_y)
			self.gl.addItem(self.global_axis_z)

			# GL items for the plane
			self.plane_mesh = gl.GLMeshItem(smooth=False, drawEdges=False)
			self.plane_mesh.setGLOptions('translucent')
			self.plane_edges = gl.GLLinePlotItem(mode='line_strip', width=3.0)
			self.gl.addItem(self.plane_mesh)
			self.gl.addItem(self.plane_edges)
			self._update_plane_draw()  # empty at start

			if self._visual_flip_y:
				for it in (self.cloud_item, self.box_lines, self.plane_mesh, self.plane_edges,
								self.global_axis_x, self.global_axis_y, self.global_axis_z,
								self.cam_mesh, self.cam_forward):
					it.scale(1.0, -1.0, 1.0)

		# ---------- Right panel ----------
		if 1:
			panel = QtWidgets.QFrame()
			panel.setObjectName("PlaneRightPanel")
			panel_layout = QtWidgets.QVBoxLayout(panel)
			panel_layout.setContentsMargins(8, 8, 8, 8)
			panel_layout.setSpacing(6)
			root.addWidget(panel, 0)

			# Title
			title = QtWidgets.QLabel("PLANE WIZARD")
			title.setStyleSheet("font-weight:700; font-size:16px;")
			title.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
			title.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
			panel_layout.addWidget(title)

			# Center (m)
			self.cx = _backing_spin(-10.0, 10.0, 0.001, 0.0)
			self.cy = _backing_spin(-10.0, 10.0, 0.001, 0.0)
			self.cz = _backing_spin(-10.0, 10.0, 0.001, 1.0)
			# Half-extents (m)
			self.ex = _backing_spin(0.05, 5.0, 0.001, 0.40)
			self.ey = _backing_spin(0.01, 1.0, 0.001, 0.05)
			self.ez = _backing_spin(0.05, 5.0, 0.001, 0.40)
			# Rotation (deg)
			self.yaw   = _backing_spin(-180.0, 180.0, 0.5, 0.0)
			self.pitch = _backing_spin(-180.0, 180.0, 0.5, 0.0)
			self.roll  = _backing_spin(-180.0, 180.0, 0.5, 0.0)

			# --- Right sidebar (cards) ---
			scroll = QtWidgets.QScrollArea()
			scroll.setWidgetResizable(True)
			panel_layout.addWidget(scroll, 1)

			right_host = QtWidgets.QWidget()
			right = QtWidgets.QVBoxLayout(right_host)
			right.setContentsMargins(0,0,0,0)
			right.setSpacing(12)
			scroll.setWidget(right_host)

		# ---------- PLANE ROI CARD ----------
		if 1:
			card_box = CollapsibleCard("SELECT PLANE POINTS")
			# header full width + centered text
			card_box._button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
			card_box._button.setStyleSheet(card_box._button.styleSheet() + " text-align:center;")
			box_layout = card_box.content_layout()

			b_vis_row = QtWidgets.QHBoxLayout()
			self.chk_box_visible = QtWidgets.QCheckBox("Visible")
			self.chk_box_visible.setChecked(True)
			self.chk_pts_visible = QtWidgets.QCheckBox("Colour Pts")
			self.chk_pts_visible.setChecked(True)
			self.btn_suggest = QtWidgets.QPushButton("Suggest orientation")
			
			b_vis_row.addWidget(self.chk_box_visible)
			b_vis_row.addWidget(self.chk_pts_visible)
			b_vis_row.addStretch(1)
			b_vis_row.addWidget(self.btn_suggest)
			box_layout.addLayout(b_vis_row)

			self.chk_pts_visible.toggled.connect(lambda v: (setattr(self, "pts_visible", bool(v)), self._update_plane_draw()))
			self.chk_box_visible.toggled.connect(lambda v: (setattr(self, "box_visible", bool(v)), self._update_plane_draw()))
			self.btn_suggest.clicked.connect(self._suggest_orientation)

			# Spans: input boxes in a single row (no sliders)
			row_spans = QtWidgets.QWidget()
			row_lay = QtWidgets.QHBoxLayout(row_spans); row_lay.setContentsMargins(0,0,0,0); row_lay.setSpacing(8)

			row_x, span_x = _colored_span("X span", "x", (0.10, 10.0), self.ex.value())
			row_y, span_y = _colored_span("Y span", "y", (0.02,  2.0), self.ey.value())
			row_z, span_z = _colored_span("Z span", "z", (0.10, 10.0), self.ez.value())
			row_lay.addWidget(row_x); row_lay.addWidget(row_y); row_lay.addWidget(row_z)
			box_layout.addWidget(row_spans)

			_link_span_to_half(span_x, self.ex); _link_half_to_span(self.ex, span_x)
			_link_span_to_half(span_y, self.ey); _link_half_to_span(self.ey, span_y)
			_link_span_to_half(span_z, self.ez); _link_half_to_span(self.ez, span_z)

			# Sliders: Center (m)
			self.row_cx, self.s_cx, self.sp_cx, i2v_cx, v2i_cx, _ = make_full_slider_row("Center X", -10.0, 10.0, 0.005, self.cx.value(), "x", " m")
			self.row_cy, self.s_cy, self.sp_cy, i2v_cy, v2i_cy, _ = make_full_slider_row("Center Y", -10.0, 10.0, 0.005, self.cy.value(), "y", " m")
			self.row_cz, self.s_cz, self.sp_cz, i2v_cz, v2i_cz, _ = make_full_slider_row("Center Z", -10.0, 10.0, 0.005, self.cz.value(), "z", " m")

			self.s_cx.valueChanged.connect(lambda i: (self.cx.setValue(i2v_cx(i)), self._update_box()))
			self.s_cy.valueChanged.connect(lambda i: (self.cy.setValue(i2v_cy(i)), self._update_box()))
			self.s_cz.valueChanged.connect(lambda i: (self.cz.setValue(i2v_cz(i)), self._update_box()))
			self.sp_cx.valueChanged.connect(lambda v: (self.cx.setValue(v), self._update_box()))
			self.sp_cy.valueChanged.connect(lambda v: (self.cy.setValue(v), self._update_box()))
			self.sp_cz.valueChanged.connect(lambda v: (self.cz.setValue(v), self._update_box()))
			self.cx.valueChanged.connect(lambda v: (self.s_cx.blockSignals(True), self.s_cx.setValue(v2i_cx(v)), self.s_cx.blockSignals(False), self.sp_cx.blockSignals(True), self.sp_cx.setValue(v), self.sp_cx.blockSignals(False)))
			self.cy.valueChanged.connect(lambda v: (self.s_cy.blockSignals(True), self.s_cy.setValue(v2i_cy(v)), self.s_cy.blockSignals(False), self.sp_cy.blockSignals(True), self.sp_cy.setValue(v), self.sp_cy.blockSignals(False)))
			self.cz.valueChanged.connect(lambda v: (self.s_cz.blockSignals(True), self.s_cz.setValue(v2i_cz(v)), self.s_cz.blockSignals(False), self.sp_cz.blockSignals(True), self.sp_cz.setValue(v), self.sp_cz.blockSignals(False)))

			box_layout.addWidget(self.row_cx); box_layout.addWidget(self.row_cy); box_layout.addWidget(self.row_cz)

			# Sliders: Rotation (deg)
			self.row_yw, self.s_yw, self.sp_yw, i2v_yw, v2i_yw, _ = make_full_slider_row("Yaw",   -180.0, 180.0, 0.5, self.yaw.value(),   "y", "°")
			self.row_pt, self.s_pt, self.sp_pt, i2v_pt, v2i_pt, _ = make_full_slider_row("Pitch", -180.0, 180.0, 0.5, self.pitch.value(), "x", "°")
			self.row_rl, self.s_rl, self.sp_rl, i2v_rl, v2i_rl, _ = make_full_slider_row("Roll",  -180.0, 180.0, 0.5, self.roll.value(),  "z", "°")

			self.s_yw.valueChanged.connect(lambda i: (self.yaw.setValue(i2v_yw(i)), self._update_box()))
			self.s_pt.valueChanged.connect(lambda i: (self.pitch.setValue(i2v_pt(i)), self._update_box()))
			self.s_rl.valueChanged.connect(lambda i: (self.roll.setValue(i2v_rl(i)), self._update_box()))
			self.sp_yw.valueChanged.connect(lambda v: (self.yaw.setValue(v), self._update_box()))
			self.sp_pt.valueChanged.connect(lambda v: (self.pitch.setValue(v), self._update_box()))
			self.sp_rl.valueChanged.connect(lambda v: (self.roll.setValue(v), self._update_box()))
			self.yaw.valueChanged.connect(  lambda v: (self.s_yw.blockSignals(True), self.s_yw.setValue(v2i_yw(v)), self.s_yw.blockSignals(False), self.sp_yw.blockSignals(True), self.sp_yw.setValue(v), self.sp_yw.blockSignals(False)))
			self.pitch.valueChanged.connect(lambda v: (self.s_pt.blockSignals(True), self.s_pt.setValue(v2i_pt(v)), self.s_pt.blockSignals(False), self.sp_pt.blockSignals(True), self.sp_pt.setValue(v), self.sp_pt.blockSignals(False)))
			self.roll.valueChanged.connect( lambda v: (self.s_rl.blockSignals(True), self.s_rl.setValue(v2i_rl(v)), self.s_rl.blockSignals(False), self.sp_rl.blockSignals(True), self.sp_rl.setValue(v), self.sp_rl.blockSignals(False)))

			box_layout.addWidget(self.row_yw); box_layout.addWidget(self.row_pt); box_layout.addWidget(self.row_rl)


			right.addWidget(card_box)
		
		# ---------- PLANE CARD ----------
		if 1:
			card_pln = CollapsibleCard("GENERATE PLANE")
			card_pln._button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
			card_pln._button.setStyleSheet(card_pln._button.styleSheet() + " text-align:center;")
			pln_layout = card_pln.content_layout()

			# Row: Visible + Fit
			vis_row = QtWidgets.QHBoxLayout()
			self.chk_plane_visible = QtWidgets.QCheckBox("Visible")
			self.chk_plane_visible.setChecked(True)
			self.btn_fit_plane = QtWidgets.QPushButton("Fit Plane (from ROI)")
			vis_row.addWidget(self.chk_plane_visible)
			vis_row.addStretch(1)
			vis_row.addWidget(self.btn_fit_plane)
			pln_layout.addLayout(vis_row)

			# Row: plane spans (boxes only)
			row_psp = QtWidgets.QWidget()
			psp = QtWidgets.QHBoxLayout(row_psp); psp.setContentsMargins(0,0,0,0); psp.setSpacing(8)

			row_w, self.sp_plane_w = _span_box("Width (X)", self.pl_size_x)
			row_l, self.sp_plane_l = _span_box("Length (Z)", self.pl_size_z)
			psp.addWidget(row_w); psp.addWidget(row_l)
			pln_layout.addWidget(row_psp)

			# Position (local coordinates, along plane axes)
			pln_layout.addWidget(QtWidgets.QLabel("Plane Center (local)"))
			row_lx, s_lx, self.sp_lx, i2v_lx, v2i_lx, _ = make_full_slider_row("Local X", -2.0, 2.0, 0.002, 0.0, "x", " m")
			row_ly, s_ly, self.sp_ly, i2v_ly, v2i_ly, _ = make_full_slider_row("Local Y", -2.0, 2.0, 0.002, 0.0, "y", " m")
			row_lz, s_lz, self.sp_lz, i2v_lz, v2i_lz, _ = make_full_slider_row("Local Z", -2.0, 2.0, 0.002, 0.0, "z", " m")
			pln_layout.addWidget(row_lx); pln_layout.addWidget(row_ly); pln_layout.addWidget(row_lz)

			# Rotation (about plane's local axes)
			pln_layout.addWidget(QtWidgets.QLabel("Plane Rotation (local)"))
			row_yw2, s_yw2,self.sp_yw2, i2v_yw2, v2i_yw2, _ = make_full_slider_row("Yaw (about n)",   -180.0, 180.0, 0.5, 0.0, "y", "°")
			row_pt2, s_pt2,self.sp_pt2, i2v_pt2, v2i_pt2, _ = make_full_slider_row("Pitch (about u)", -89.0,   89.0, 0.5, 0.0, "x", "°")
			row_rl2, s_rl2,self.sp_rl2, i2v_rl2, v2i_rl2, _ = make_full_slider_row("Roll (about v)",  -89.0,   89.0, 0.5, 0.0, "z", "°")
			pln_layout.addWidget(row_yw2); pln_layout.addWidget(row_pt2); pln_layout.addWidget(row_rl2)

			right.addWidget(card_pln)

			# --- Plane UI wiring ---
			self.sp_plane_w.valueChanged.connect(self._plane_apply_size)
			self.sp_plane_l.valueChanged.connect(self._plane_apply_size)

			self.chk_plane_visible.toggled.connect(lambda v: (setattr(self, "pl_visible", bool(v)), self._update_plane_draw()))

			# Local position offsets
			s_lx.valueChanged.connect(lambda i: (
				self.sp_lx.blockSignals(True), self.sp_lx.setValue(i2v_lx(i)), self.sp_lx.blockSignals(False),
				setattr(self, "_pl_off_local", np.array([i2v_lx(i), self._pl_off_local[1], self._pl_off_local[2]])),
				self._update_plane_from_ui()
			))
			s_ly.valueChanged.connect(lambda i: (
				self.sp_ly.blockSignals(True), self.sp_ly.setValue(i2v_ly(i)), self.sp_ly.blockSignals(False),
				setattr(self, "_pl_off_local", np.array([self._pl_off_local[0], i2v_ly(i), self._pl_off_local[2]])),
				self._update_plane_from_ui()
			))
			s_lz.valueChanged.connect(lambda i: (
				self.sp_lz.blockSignals(True), self.sp_lz.setValue(i2v_lz(i)), self.sp_lz.blockSignals(False),
				setattr(self, "_pl_off_local", np.array([self._pl_off_local[0], self._pl_off_local[1], i2v_lz(i)])),
				self._update_plane_from_ui()
			))

			# Local rotations
			s_yw2.valueChanged.connect(lambda i: (
				self.sp_yw2.blockSignals(True), self.sp_yw2.setValue(i2v_yw2(i)), self.sp_yw2.blockSignals(False),
				setattr(self, "_pl_yaw", i2v_yw2(i)), self._update_plane_from_ui()
			))
			s_pt2.valueChanged.connect(lambda i: (
				self.sp_pt2.blockSignals(True), self.sp_pt2.setValue(i2v_pt2(i)), self.sp_pt2.blockSignals(False),
				setattr(self, "_pl_pitch", i2v_pt2(i)), self._update_plane_from_ui()
			))
			s_rl2.valueChanged.connect(lambda i: (
				self.sp_rl2.blockSignals(True), self.sp_rl2.setValue(i2v_rl2(i)), self.sp_rl2.blockSignals(False),
				setattr(self, "_pl_roll", i2v_rl2(i)), self._update_plane_from_ui()
			))

			self.btn_fit_plane.clicked.connect(self._fit_table_plane)

		# ---------- SEARCH AREA CARD --------
		if 1:
			card_roi = CollapsibleCard("SEARCH BOUNDS (above plane)")
			card_roi._button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
			card_roi._button.setStyleSheet(card_roi._button.styleSheet() + " text-align:center;")
			roi_layout = card_roi.content_layout()

			# Row: Visible
			vis_row = QtWidgets.QHBoxLayout()
			self.chk_roi_visible = QtWidgets.QCheckBox("Visible")
			self.chk_roi_visible.setChecked(True)
			vis_row.addWidget(self.chk_roi_visible)
			roi_layout.addLayout(vis_row)

			self.chk_roi_visible.toggled.connect(lambda v: (setattr(self, "roi_visible", bool(v)), self._update_plane_draw()))

			# X extend with Mirror toggle
			self.row_x, self.sld_x, self.spn_x, i2v_x, v2i_x, _ = make_full_slider_row("Extend X", -10.0, 10.0, 0.05, 
																			  float(getattr(self, "roi_x_extend", 0.0)),  "x", "°")
			self.chk_roi_mx = QtWidgets.QCheckBox("Mirror")
			xwrap = QtWidgets.QWidget(); xh = QtWidgets.QHBoxLayout(xwrap); xh.setContentsMargins(0,0,0,0); xh.setSpacing(8)
			xh.addWidget(self.row_x, 1); xh.addWidget(self.chk_roi_mx, 0)
			roi_layout.addWidget(xwrap)

			# Z extend with Mirror toggle
			self.row_z, self.sld_z, self.spn_z, i2v_z, v2i_z, _ = make_full_slider_row("Extend Z", -10.0, 10.0, 0.05, 
																			  float(getattr(self, "roi_z_extend", 0.0)), "z", "°")
			self.chk_roi_mz = QtWidgets.QCheckBox("Mirror")
			zwrap = QtWidgets.QWidget(); zh = QtWidgets.QHBoxLayout(zwrap); zh.setContentsMargins(0,0,0,0); zh.setSpacing(8)
			zh.addWidget(self.row_z, 1); zh.addWidget(self.chk_roi_mz, 0)
			roi_layout.addWidget(zwrap)

			# Height band (min/max) above plane, meters
			row_h = QtWidgets.QWidget(); hh = QtWidgets.QHBoxLayout(row_h); hh.setContentsMargins(0,0,0,0); hh.setSpacing(8)
			lab_h = QtWidgets.QLabel("Height above plane")
			self.sp_roi_ymin = QtWidgets.QDoubleSpinBox()
			self.sp_roi_ymin.setRange(0.0, 3.0); self.sp_roi_ymin.setDecimals(3); self.sp_roi_ymin.setSingleStep(0.01); self.sp_roi_ymin.setSuffix(" m")
			self.sp_roi_ymin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
			self.sp_roi_ymax = QtWidgets.QDoubleSpinBox()
			self.sp_roi_ymax.setRange(0.0, 3.0); self.sp_roi_ymax.setDecimals(3); self.sp_roi_ymax.setSingleStep(0.01); self.sp_roi_ymax.setSuffix(" m")
			self.sp_roi_ymax.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)

			self.rs_y = RangeSlider(QtCore.Qt.Horizontal)
			self.rs_y.setMinimum(0); self.rs_y.setMaximum(3000)  # map mm 0..5000
			# init from current values (meters -> mm)
			ymin0 = int(round(1000.0 * float(getattr(self, "roi_y_min", 0.0))))
			ymax0 = int(round(1000.0 * float(getattr(self, "roi_y_max", 0.2))))
			self.rs_y.setLowerValue(ymin0)
			self.rs_y.setUpperValue(ymax0)
			self.sp_roi_ymin.setValue(ymin0 / 1000.0)
			self.sp_roi_ymax.setValue(ymax0 / 1000.0)

			hh.addWidget(lab_h)
			hh.addWidget(self.sp_roi_ymin)
			hh.addWidget(self.rs_y, 1)
			hh.addWidget(self.sp_roi_ymax)
			roi_layout.addWidget(row_h)

			def _sync_y_from_slider():
				self.sp_roi_ymin.blockSignals(True); self.sp_roi_ymin.setValue(self.rs_y.lowerValue()/1000.0); self.sp_roi_ymin.blockSignals(False)
				self.sp_roi_ymax.blockSignals(True); self.sp_roi_ymax.setValue(self.rs_y.upperValue()/1000.0); self.sp_roi_ymax.blockSignals(False)
				self.roi_y_min = self.sp_roi_ymin.value()
				self.roi_y_max = self.sp_roi_ymax.value()
				self._update_roi_preview()

			def _sync_slider_from_spin():
				lo = int(round(self.sp_roi_ymin.value()*1000.0))
				hi = int(round(self.sp_roi_ymax.value()*1000.0))
				self.rs_y.blockSignals(True); self.rs_y.setLowerValue(min(lo, hi)); self.rs_y.setUpperValue(max(lo, hi)); self.rs_y.blockSignals(False)
				self.roi_y_min = float(min(self.sp_roi_ymin.value(), self.sp_roi_ymax.value()))
				self.roi_y_max = float(max(self.sp_roi_ymin.value(), self.sp_roi_ymax.value()))
				self._update_roi_preview()

			self.rs_y.lowerValueChanged.connect(lambda _: _sync_y_from_slider())
			self.rs_y.upperValueChanged.connect(lambda _: _sync_y_from_slider())
			self.sp_roi_ymin.valueChanged.connect(lambda _: _sync_slider_from_spin())
			self.sp_roi_ymax.valueChanged.connect(lambda _: _sync_slider_from_spin())

			right.addWidget(card_roi)
			right.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

			self.roi_edges = gl.GLLinePlotItem(mode='lines', width=2.0)
			self.gl.addItem(self.roi_edges)
			if self._visual_flip_y:
				self.roi_edges.scale(1.0, -1.0, 1.0)

			def _apply_roi_from_ui():
				self.roi_x_extend = float(self.spn_x.value())
				self.roi_z_extend = float(self.spn_z.value())
				self.roi_mirror_x = bool(self.chk_roi_mx.isChecked())
				self.roi_mirror_z = bool(self.chk_roi_mz.isChecked())
				self.roi_y_min = float(min(self.sp_roi_ymin.value(), self.sp_roi_ymax.value()))
				self.roi_y_max = float(max(self.sp_roi_ymin.value(), self.sp_roi_ymax.value()))
				self._update_roi_preview()

			# initial populate (if loaded earlier)
			self.chk_roi_mx.setChecked(getattr(self, "roi_mirror_x", False))
			self.chk_roi_mz.setChecked(getattr(self, "roi_mirror_z", False))
			self.sp_roi_ymin.setValue(getattr(self, "roi_y_min", 0.0))
			self.sp_roi_ymax.setValue(getattr(self, "roi_y_max", 0.2))

			# wire events
			self.sld_x.valueChanged.connect(lambda i: (
				self.spn_x.blockSignals(True), self.spn_x.setValue(i2v_x(i)), self.spn_x.blockSignals(False),
				_apply_roi_from_ui()))
			self.spn_x.valueChanged.connect(lambda v: (
				self.sld_x.blockSignals(True), self.sld_x.setValue(v2i_x(v)), self.sld_x.blockSignals(False),
				_apply_roi_from_ui()))

			self.sld_z.valueChanged.connect(lambda i: (
				self.spn_z.blockSignals(True), self.spn_z.setValue(i2v_z(i)), self.spn_z.blockSignals(False),
				_apply_roi_from_ui()))
			self.spn_z.valueChanged.connect(lambda v: (
				self.sld_z.blockSignals(True), self.sld_z.setValue(v2i_z(v)), self.sld_z.blockSignals(False),
				_apply_roi_from_ui()))

			self.sp_roi_ymin.valueChanged.connect(lambda _: _apply_roi_from_ui())
			self.sp_roi_ymax.valueChanged.connect(lambda _: _apply_roi_from_ui())

		# ---------- BOTTOM BUTTONS ----------
		if 1:
			btn_row = QtWidgets.QHBoxLayout()
			self.btn_reset   = QtWidgets.QPushButton("Reset View")
			self.btn_view_u = QtWidgets.QPushButton("<-")
			self.btn_view_v = QtWidgets.QPushButton("->")
			self.btn_save    = QtWidgets.QPushButton("Save")
			self.btn_cancel  = QtWidgets.QPushButton("Cancel")

			btn_row.addWidget(self.btn_reset) 
			btn_row.insertWidget(1, self.btn_view_u)
			btn_row.insertWidget(2, QtWidgets.QLabel("Plane View"))
			btn_row.insertWidget(3, self.btn_view_v)
			btn_row.addStretch(1)
			btn_row.addWidget(self.btn_cancel)
			btn_row.addWidget(self.btn_save)
			panel_layout.addLayout(btn_row)

			self.current_view = 0
			# Wire updates into draw path
			for w in (self.cx,self.cy,self.cz,self.ex,self.ey,self.ez,self.yaw,self.pitch,self.roll):
				w.valueChanged.connect(self._update_box)
			self.btn_reset.clicked.connect(self._set_default_view) 
			self.btn_view_u.clicked.connect(lambda: self._view_from_axis(1))
			self.btn_view_v.clicked.connect(lambda: self._view_from_axis(-1))
			self.btn_save.clicked.connect(self._save)
			self.btn_cancel.clicked.connect(self.canceled.emit)

		panel.setFixedWidth(420) 
		scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		# right_host.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

		# Load previous session defaults if present
		self._load_defaults()
		self._update_box()
		self._update_plane_draw()

		# Timer (~15 Hz)
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self._tick)
		self.timer.start(66)

	def _tick(self):
		# Update cloud
		xyz = self.worker.get_point_cloud_snapshot()
		if xyz is not None and xyz.shape[0] > 0:

			# current box params
			c = np.array([self.cx.value(), self.cy.value(), self.cz.value()], float)
			e = np.array([self.ex.value(), self.ey.value(), self.ez.value()], float)
			R = self._R_yxz(self.yaw.value(), self.pitch.value(), self.roll.value())

			Xloc = (xyz - c) @ R
			inside = (np.abs(Xloc[:,0]) <= e[0]) & (np.abs(Xloc[:,1]) <= e[1]) & (np.abs(Xloc[:,2]) <= e[2])

			colors = np.zeros((xyz.shape[0], 4), float)
			colors[:] = (1,1,1, 0.60)
			if self.pts_visible:
				colors[inside] = (0.10, 1.00, 0.30, 0.95)

			self.cloud_item.setData(pos=xyz, color=colors)

		self._update_box()
		if self.plane is not None:
			self._update_plane_draw()
			self._update_global_axes()
		
		# --- update axis HUDs ---
		try:
			VM = np.array(self.gl.viewMatrix().data()).reshape(4,4).T
			Rw = VM[:3,:3]  # world -> camera
		except Exception:
			Rw = np.eye(3)

		# Camera glyph: identity in camera space
		R_cam_world = Rw.T
		self.glyph_cam.setRotationCam(R_cam_world)

		# Box glyph: local axes in world -> camera
		R_box_world = self._R_yxz(self.yaw.value(), self.pitch.value(), self.roll.value())
		self.glyph_box.setRotationCam(Rw @ R_box_world)

		# Plane glyph: [u n v] in world -> camera
		if self.plane is not None:
			R_plane_world = np.column_stack([self.plane.u, self.plane.n, self.plane.v])
			self.glyph_plane.setRotationCam(Rw @ R_plane_world)

	def _load_defaults(self):
		#Read JSON state (if exists) and apply as current defaults
		if not os.path.isfile(self.config_path):
			return
		try:
			with open(self.config_path, "r", encoding="utf-8") as f:
				state = json.load(f)
			self._apply_state(state)
			# After applying, make sure the visuals update
			self._update_box()
			self._update_plane_draw()
		except Exception as e:
			QtWidgets.QMessageBox.warning(self, "Load error", f"Could not load config:\n{e}")

	def _apply_state(self, state: dict):
		# ---- Box ----
		box = state.get("box", {})
		c = box.get("center")
		e = box.get("extents")
		ypr = box.get("ypr_deg")
		if c:   self.cx.setValue(float(c[0])); self.cy.setValue(float(c[1])); self.cz.setValue(float(c[2]))
		if e:   self.ex.setValue(float(e[0])); self.ey.setValue(float(e[1])); self.ez.setValue(float(e[2]))
		if ypr: self.yaw.setValue(float(ypr[0])); self.pitch.setValue(float(ypr[1])); self.roll.setValue(float(ypr[2]))

		roi = state.get("roi", {})
		self.roi_x_extend = float(roi.get("x_extend", 0.0))
		self.roi_z_extend = float(roi.get("z_extend", 0.0))
		self.roi_mirror_x = bool(roi.get("mirror_x", False))
		self.roi_mirror_z = bool(roi.get("mirror_z", False))
		self.roi_y_min    = float(roi.get("y_min", 0.0))
		self.roi_y_max    = float(roi.get("y_max", 0.2))
		
		self.spn_x.blockSignals(True); self.spn_x.setValue(self.roi_x_extend); self.spn_x.blockSignals(False)
		self.spn_z.blockSignals(True); self.spn_z.setValue(self.roi_z_extend); self.spn_z.blockSignals(False)
		self.chk_roi_mx.setChecked(self.roi_mirror_x)
		self.chk_roi_mz.setChecked(self.roi_mirror_z)
		self.sp_roi_ymin.setValue(self.roi_y_min)
		self.sp_roi_ymax.setValue(self.roi_y_max)

		cp = box.get("color_points")
		if cp is not None:
			self.pts_visible = bool(cp)
			self.chk_pts_visible.blockSignals(True)
			self.chk_pts_visible.setChecked(self.pts_visible)
			self.chk_pts_visible.blockSignals(False)

		# ---- Plane ----
		pl = state.get("plane")
		if pl:
			p0 = np.array(pl.get("p0", [0,0,0]), float)
			n  = np.array(pl.get("normal", [0,1,0]), float); n = n/np.linalg.norm(n) if np.linalg.norm(n)>0 else n
			u  = np.array(pl.get("u", [1,0,0]), float)
			v  = np.array(pl.get("v", [0,0,1]), float)
			# Rebuild base pose
			self._pl_R_base = np.column_stack([u, n, v])
			self._pl_p0_base = p0

			# Local edits
			loff = pl.get("local_offsets", [0,0,0])
			self._pl_off_local = np.array(loff, float)
			lypr = pl.get("local_yaw_pitch_roll_deg", [0,0,0])
			self._pl_yaw, self._pl_pitch, self._pl_roll = [float(a) for a in lypr]

			# Size & visibility
			self.pl_size_x = float(pl.get("width_m", 0.70))
			self.pl_size_z = float(pl.get("length_m", 0.70))
			
			self.sp_lx.setValue(self._pl_off_local[0])
			self.sp_ly.setValue(self._pl_off_local[1])
			self.sp_lz.setValue(self._pl_off_local[2])
			self.sp_yw2.setValue(self._pl_yaw)
			self.sp_pt2.setValue(self._pl_pitch)
			self.sp_rl2.setValue(self._pl_roll)

			# Populate UI controls to reflect size/visibility/local edits
			self.sp_plane_w.blockSignals(True); self.sp_plane_w.setValue(self.pl_size_x); self.sp_plane_w.blockSignals(False)
			self.sp_plane_l.blockSignals(True); self.sp_plane_l.setValue(self.pl_size_z); self.sp_plane_l.blockSignals(False)
			self.chk_plane_visible.blockSignals(True); self.chk_plane_visible.setChecked(self.pl_visible); self.chk_plane_visible.blockSignals(False)
			self.sp_cx.blockSignals(True); self.sp_cx.setValue(self.cx.value()); self.sp_cx.blockSignals(False)
			self.sp_cy.blockSignals(True); self.sp_cy.setValue(self.cy.value()); self.sp_cy.blockSignals(False)
			self.sp_cz.blockSignals(True); self.sp_cz.setValue(self.cz.value()); self.sp_cz.blockSignals(False)
			self.sp_yw.blockSignals(True); self.sp_yw.setValue(self.yaw.value()); self.sp_yw.blockSignals(False)
			self.sp_pt.blockSignals(True); self.sp_pt.setValue(self.pitch.value()); self.sp_pt.blockSignals(False)
			self.sp_rl.blockSignals(True); self.sp_rl.setValue(self.roll.value()); self.sp_rl.blockSignals(False)

			# Create/refresh plane object and draw
			R = self._plane_R_from_base_and_loc()
			p0c = self._plane_p0_from_base_and_loc()
			u_cur, n_cur, v_cur = R[:,0], R[:,1], R[:,2]
			self.plane = TablePlane(n=n_cur/np.linalg.norm(n_cur), d=-(n_cur @ p0c), p0=p0c, u=u_cur/np.linalg.norm(u_cur), v=v_cur/np.linalg.norm(v_cur))
			self._update_plane_draw()
			self._update_global_axes()

		# ---- Camera ----
		self._set_default_view()

		# ---- Flags ----
		fl = state.get("flags", {})
		if "plane_visible" in fl:
			self.pl_visible = bool(fl["plane_visible"])
			self.chk_plane_visible.blockSignals(True)
			self.chk_plane_visible.setChecked(self.pl_visible)
			self.chk_plane_visible.blockSignals(False)
		
		if "box_visible" in fl:
			self.box_visible = bool(fl["box_visible"])
			self.chk_box_visible.blockSignals(True)
			self.chk_box_visible.setChecked(self.box_visible)
			self.chk_box_visible.blockSignals(False)

		if "roi_visible" in fl:
			self.roi_visible = bool(fl["roi_visible"])
			self.chk_roi_visible.blockSignals(True)
			self.chk_roi_visible.setChecked(self.roi_visible)
			self.chk_roi_visible.blockSignals(False)

	def _state_to_dict(self):
		"""Collect full UI state (box, plane, camera, flags) -> dict."""
		state = {
			"box": {
				"center": [self.cx.value(), self.cy.value(), self.cz.value()],
				"extents": [self.ex.value(), self.ey.value(), self.ez.value()],
				"ypr_deg": [self.yaw.value(), self.pitch.value(), self.roll.value()],
				"color_points": bool(getattr(self, "pts_visible", True)),
			},
			"plane": None,
			"flags": {
				"plane_visible": bool(getattr(self, "pl_visible", True)),
				"box_visible": bool(getattr(self, "box_visible", True)),
				"roi_visible": bool(getattr(self, "roi_visible", True)),
			},
			"roi": {
				"x_extend": float(getattr(self, "roi_x_extend", 0.0)),
				"z_extend": float(getattr(self, "roi_z_extend", 0.0)),
				"mirror_x": bool(getattr(self, "roi_mirror_x", False)),
				"mirror_z": bool(getattr(self, "roi_mirror_z", False)),
				"y_min":    float(getattr(self, "roi_y_min", 0.0)),
				"y_max":    float(getattr(self, "roi_y_max", 0.2)),
			}
		}

		if self.plane is not None:
			state["plane"] = {
				"p0": self.plane.p0.tolist(),
				"normal": self.plane.n.tolist(),
				"u": self.plane.u.tolist(),
				"v": self.plane.v.tolist(),
				"width_m": float(self.pl_size_x),
				"length_m": float(self.pl_size_z),
				"local_offsets": self._pl_off_local.tolist(),
				"local_yaw_pitch_roll_deg": [float(self._pl_yaw),
											float(self._pl_pitch),
											float(self._pl_roll)],
				"visible": bool(self.pl_visible),
			}

		return state

	def _save(self):
		
		self.saved.emit(None)

		# Write the full state to JSON
		state = self._state_to_dict()
		try:
			with open(self.config_path, "w", encoding="utf-8") as f:
				json.dump(state, f, indent=2)
			QtWidgets.QToolTip.showText(self.mapToGlobal(QtCore.QPoint(0,0)),
										f"Saved config → {os.path.basename(self.config_path)}")
		except Exception as e:
			QtWidgets.QMessageBox.warning(self, "Save error", f"Could not write config:\n{e}")

	def eventFilter(self, obj, ev):
		if obj is self.gl and ev.type() == QtCore.QEvent.MouseButtonDblClick:
			pos = ev.pos()
			self._pick_and_center(pos.x(), pos.y())
			return True
		return super().eventFilter(obj, ev)

	def _set_worker(self, worker):
		self.worker = worker

#============== Plane Functions =========================	
	def _plane_R_from_base_and_loc(self):
		"""Compute [u n v] from base pose + local yaw/pitch/roll, no self.plane dependency."""
		y, p, r = np.deg2rad([self._pl_yaw, self._pl_pitch, self._pl_roll])
		Rloc = _Ry(y) @ _Rx(p) @ _Rz(r)
		return self._pl_R_base @ Rloc

	def _plane_p0_from_base_and_loc(self):
		"""Compute current plane center from base + local offsets along (u,n,v), no self.plane dependency."""
		R = self._plane_R_from_base_and_loc()
		u, n, v = R[:,0], R[:,1], R[:,2]
		off = self._pl_off_local
		return self._pl_p0_base + off[0]*u + off[1]*n + off[2]*v

	def _plane_current_R(self):
		"""R columns are [u, n, v] after applying local rotations (Y,X,Z in that local order)."""
		if self.plane is None:
			return None
		y, p, r = np.deg2rad([self._pl_yaw, self._pl_pitch, self._pl_roll])
		Rloc = _Ry(y) @ _Rx(p) @ _Rz(r)        # about [Y=normal, X=u, Z=v] in local
		return self._pl_R_base @ Rloc

	def _plane_current_p0(self):
		"""Current plane center from base + local offsets along [u, n, v]."""
		if self.plane is None:
			return None
		R = self._plane_current_R()
		u, n, v = R[:,0], R[:,1], R[:,2]
		off = self._pl_off_local
		return self._pl_p0_base + off[0]*u + off[1]*n + off[2]*v

	def _update_plane_from_ui(self):
		"""Refresh self.plane (geometry) from UI offsets/rotations/size and redraw."""
		if self.plane is None:
			self._update_plane_draw()
			return
		R = self._plane_current_R()
		p0 = self._plane_current_p0()
		u, n, v = R[:,0], R[:,1], R[:,2]
		self.plane = TablePlane(n=n, d=-(n @ p0), p0=p0, u=u, v=v)
		self._update_plane_draw()

	def _update_plane_draw(self):
		"""Draw low-alpha plane surface and higher-alpha edges."""
		if (self.plane is None) or (not self.pl_visible):
			# Hide
			self.plane_mesh.setMeshData(meshdata=None)
			self.plane_edges.setData(pos=np.empty((0,3)))
			return
		w = float(self.pl_size_x)
		l = float(self.pl_size_z)
		u, n, v = self.plane.u, self.plane.n, self.plane.v
		p0 = self.plane.p0

		# 4 corners in world
		hx, hz = 0.5*w, 0.5*l
		corners = [
			p0 + (-hx)*u + (-hz)*v,
			p0 + ( hx)*u + (-hz)*v,
			p0 + ( hx)*u + ( hz)*v,
			p0 + (-hx)*u + ( hz)*v,
		]
		verts = np.vstack(corners).astype(float)
		faces = np.array([[0,1,2], [0,2,3]], dtype=int)

		md = gl.MeshData(vertexes=verts, faces=faces)
		self.plane_mesh.setMeshData(meshdata=md)
		self.plane_mesh.setColor((0.2, 0.8, 1.0, 0.15))    # translucent surface

		edge_loop = np.vstack([verts, verts[0:1]])
		self.plane_edges.setData(pos=edge_loop, color=(0.2, 0.8, 1.0, 0.7))

		self._update_global_axes()
		self._update_roi_preview()

	def _plane_apply_size(self):
		self.pl_size_x = float(self.sp_plane_w.value())
		self.pl_size_z = float(self.sp_plane_l.value())
		self._update_plane_from_ui()

	def _fit_table_plane(self):
		"""Fit a plane from points inside the ROI box; seed base pose & zero local adjustments."""
		xyz = self.worker.get_point_cloud_snapshot()
		if xyz is None or xyz.shape[0] < 200:
			QtWidgets.QMessageBox.information(self, "Plane Fit", "Not enough points.")
			return

		# ROI mask using the current box
		c = np.array([self.cx.value(), self.cy.value(), self.cz.value()], float)
		e = np.array([self.ex.value(), self.ey.value(), self.ez.value()], float)
		Rb = self._R_yxz(self.yaw.value(), self.pitch.value(), self.roll.value())
		Xloc = (xyz - c) @ Rb
		inside = (np.abs(Xloc[:,0]) <= e[0]) & (np.abs(Xloc[:,1]) <= e[1]) & (np.abs(Xloc[:,2]) <= e[2])
		pts = xyz[inside]
		if pts.shape[0] < 200:
			QtWidgets.QMessageBox.information(self, "Plane Fit", "Not enough points inside ROI.")
			return

		# Fit plane (normal up-ish)
		n, d, c_patch, rms = _fit_plane_svd(pts)

		# Pick an in-plane axis u by PCA (stable heading), get v = u × n
		h = (pts - c_patch) @ n
		proj = pts - np.outer(h, n)
		U, S, Vt = np.linalg.svd(proj - proj.mean(axis=0, keepdims=True), full_matrices=False)
		u = Vt[0]; u = u - (u @ n)*n; u = u / np.linalg.norm(u)
		v = np.cross(u, n); v = v/np.linalg.norm(v)
		R = np.column_stack([u, n, v])

		# Seed base pose and zero local offsets/rotations
		self._pl_R_base = R
		self._pl_p0_base = c_patch
		self._pl_off_local[:] = 0.0
		self._pl_yaw = 0.0; self._pl_pitch = 0.0; self._pl_roll = 0.0

		# Create/refresh the TablePlane and draw
		self.plane = TablePlane(n=n, d=-(n @ c_patch), p0=c_patch, u=u, v=v)
		self._update_plane_from_ui()

		QtWidgets.QToolTip.showText(self.mapToGlobal(QtCore.QPoint(0,0)),
									f"Plane RMS: {rms*1000:.1f} mm on {pts.shape[0]} pts")

	def _global_axis_origin(self):
		"""Axis anchored at the table's physical center: base p0 + (dx*u + dz*v), ignoring local Y offset."""
		if self.plane is None:
			return np.zeros(3), np.eye(3)
		# current [u n v] and local offsets
		R = self._plane_R_from_base_and_loc()
		u, n, v = R[:,0], R[:,1], R[:,2]
		off = self._pl_off_local
		# ignore local Y offset (normal)
		origin = self._pl_p0_base + off[0]*u + off[2]*v
		return origin, R

	def _update_global_axes(self):
		if (self.plane is None) or (not self.pl_visible):
			for it in (self.global_axis_x, self.global_axis_y, self.global_axis_z):
				it.setData(pos=np.empty((0,3)))
			return
		O, R = self._global_axis_origin()
		u, n, v = R[:,0]/(np.linalg.norm(R[:,0])+1e-12), R[:,1]/(np.linalg.norm(R[:,1])+1e-12), R[:,2]/(np.linalg.norm(R[:,2])+1e-12)
		# very long axes (meters)
		L = 2.5
		px, cx = self._fade_line(O, u, L, (1.0, 0.15, 0.15))
		py, cy = self._fade_line(O, n, L, (0.2, 0.95, 0.2))
		pz, cz = self._fade_line(O, v, L, (0.2, 0.5, 1.0))
		self.global_axis_x.setData(pos=px, color=cx)
		self.global_axis_y.setData(pos=py, color=cy)
		self.global_axis_z.setData(pos=pz, color=cz)

	def _project_points(self, pts):
		try:
			PM = np.array(self.gl.projectionMatrix().data()).reshape(4,4).T
			VM = np.array(self.gl.viewMatrix().data()).reshape(4,4).T
		except Exception:
			return None

		pts_vis = np.asarray(pts, dtype=float)
		if getattr(self, "_visual_flip_y", False):
			pts_vis = pts_vis.copy()
			pts_vis[:, 1] *= -1.0

		P = PM @ VM
		N = pts_vis.shape[0]
		hom = np.c_[pts_vis, np.ones((N,1))]
		clip = hom @ P.T
		w = clip[:, 3:4]
		ok = np.abs(w[:, 0]) > 1e-9
		clip = clip[ok]; w = w[ok]
		ndc = clip[:, :3] / w  # [-1,1]

		W = float(self.gl.width())
		H = float(self.gl.height())
		x_px = (ndc[:, 0] * 0.5 + 0.5) * W
		y_px = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H  # Qt y down

		out = np.empty((N, 2), dtype=float)
		out[:] = np.nan
		out[ok, 0] = x_px
		out[ok, 1] = y_px
		return out
	
	def _R_yxz(self, yaw_deg, pitch_deg, roll_deg):
		y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
		return _Ry(y) @ _Rx(p) @ _Rz(r)
	
	def _euler_from_R_yxz(self, R):
		pitch = -np.degrees(np.arcsin(R[1,2]))
		yaw   =  np.degrees(np.arctan2(R[0,2], R[2,2]))
		roll  =  np.degrees(np.arctan2(R[1,0], R[1,1]))
		return _wrap_deg(yaw), _wrap_deg(pitch), _wrap_deg(roll)
#=========================================================		
	
#================= Box Functions =========================
	def _update_box(self):
		if not self.box_visible:
			# Hide
			self.box_lines.setData(pos=np.empty((0,3)))
			for axis in self.box_axes:
				axis.setData(pos=np.empty((0,3)))
			return
		
		# Build wireframe box from center/half-extents/rotation
		c = np.array([self.cx.value(), self.cy.value(), self.cz.value()], float)
		e = np.array([self.ex.value(), self.ey.value(), self.ez.value()], float)
		R = self._R_yxz(self.yaw.value(), self.pitch.value(), self.roll.value())

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
	
	def _make_box_mesh(self, sx, sy, sz, color=(0.85,0.85,0.90,0.9)):
		# axis-aligned cuboid centered at origin
		x,y,z = sx*0.5, sy*0.5, sz*0.5
		verts = np.array([
			[-x,-y,-z], [ x,-y,-z], [ x, y,-z], [-x, y,-z],  # back
			[-x,-y, z], [ x,-y, z], [ x, y, z], [-x, y, z],  # front
		], dtype=float)
		faces = np.array([
			[0,1,2],[0,2,3],  # back
			[4,5,6],[4,6,7],  # front
			[0,1,5],[0,5,4],  # bottom
			[2,3,7],[2,7,6],  # top
			[1,2,6],[1,6,5],  # right
			[0,3,7],[0,7,4],  # left
		], dtype=int)
		md = gl.MeshData(vertexes=verts, faces=faces)
		m  = gl.GLMeshItem(meshdata=md, smooth=False, drawEdges=False)
		m.setGLOptions('translucent')
		m.setColor(color)
		return m

	def _pick_and_center(self, px, py):
		xyz = self.worker.get_point_cloud_snapshot()
		if xyz is None or xyz.shape[0] == 0:
			return

		scr = self._project_points(xyz)
		if scr is None:
			return
		j = int(np.nanargmin((scr[:,0]-px)**2 + (scr[:,1]-py)**2))
		p = xyz[j]  # already in scene coords
		self.cx.setValue(float(p[0]))
		self.cy.setValue(float(p[1]))
		self.cz.setValue(float(p[2]))

	def _suggest_orientation(self):

		xyz = self.worker.get_point_cloud_snapshot()
		if xyz is None or xyz.shape[0] < 100:
			QtWidgets.QMessageBox.information(self, "Suggest", "Not enough points.")
			return

		# Current box center and half-extents
		cx, cy, cz = float(self.cx.value()), float(self.cy.value()), float(self.cz.value())
		ex, ey, ez = float(self.ex.value()), float(self.ey.value()), float(self.ez.value())

		# Select a neighborhood: rectangle in XZ around the center (slightly larger than box)
		tol = 1.15  # 15% margin
		m = (np.abs(xyz[:,0] - cx) <= ex * tol) & (np.abs(xyz[:,2] - cz) <= ez * tol)
		pts = xyz[m]
		if pts.shape[0] < 200:
			# If too sparse, relax once
			tol = 1.5
			m = (np.abs(xyz[:,0] - cx) <= ex * tol) & (np.abs(xyz[:,2] - cz) <= ez * tol)
			pts = xyz[m]
		if pts.shape[0] < 100:
			QtWidgets.QMessageBox.information(self, "Suggest", "Not enough local points near the box.")
			return

		# Downsample if huge (speed)
		if pts.shape[0] > 80000:
			step = int(np.ceil(pts.shape[0] / 80000.0))
			pts = pts[::step]

		# Fit plane
		n, d, c_local, rms = _fit_plane_svd(pts)

		Rprev = _RzRyRx(self.yaw.value(), self.pitch.value(), self.roll.value())
		xprev = Rprev[:, 0]
		R = _R_from_n_and_u(n, xprev)   # columns are world directions of local x,y,z
		# Sanity: R[:,1] (the 'y' column) is the plane normal
		# assert np.allclose(R[:,1] / np.linalg.norm(R[:,1]), n, atol=1e-6)

		yaw, pitch, roll = self._euler_from_R_yxz(R)

		self.yaw.setValue(float(yaw))
		self.pitch.setValue(float(pitch))
		self.roll.setValue(float(roll))

		# Optional: micro feedback
		QtWidgets.QToolTip.showText(self.mapToGlobal(QtCore.QPoint(0,0)),
									f"Plane fit RMS: {rms*1000:.1f} mm on {pts.shape[0]} pts")
#=========================================================

#================= ROI Functions =========================
	def _roi_corners_ext(self, p0, u, v, n, w, l, y0, y1):
		# same logic as main UI: mirror vs single-side, using self.roi_* values
		hx, hz = 0.5*w, 0.5*l
		ex, ez = float(getattr(self, "roi_x_extend", 0.0)), float(getattr(self, "roi_z_extend", 0.0))

		# along u
		if getattr(self, "roi_mirror_x", False):
			hx_neg = hx + abs(ex); hx_pos = hx + abs(ex)
		else:
			hx_neg = hx + (abs(ex) if ex < 0 else 0.0)
			hx_pos = hx + (abs(ex) if ex > 0 else 0.0)

		# along v
		if getattr(self, "roi_mirror_z", False):
			hz_neg = hz + abs(ez); hz_pos = hz + abs(ez)
		else:
			hz_neg = hz + (abs(ez) if ez < 0 else 0.0)
			hz_pos = hz + (abs(ez) if ez > 0 else 0.0)

		# base 4
		c00 = p0 + (-hx_neg)*u + (-hz_neg)*v
		c10 = p0 + ( +hx_pos)*u + (-hz_neg)*v
		c11 = p0 + ( +hx_pos)*u + ( +hz_pos)*v
		c01 = p0 + (-hx_neg)*u + ( +hz_pos)*v
		base = np.stack([c00,c10,c11,c01], axis=0)

		lo = base + n[None,:]*y0
		hi = base + n[None,:]*y1
		return np.vstack([lo, hi])  # 8x3

	def _update_roi_preview(self):
		if self.plane is None or not self.roi_visible:
			self.roi_edges.setData(pos=np.empty((0,3))); return
		# plane basis/size
		p0 = np.asarray(self.plane.p0, float)
		u  = np.asarray(self.plane.u,  float)
		v  = np.asarray(self.plane.v,  float)
		n  = np.asarray(self.plane.n,  float)
		w  = float(self.pl_size_x)
		l  = float(self.pl_size_z)
		y0 = float(getattr(self, "roi_y_min", 0.0))
		y1 = float(getattr(self, "roi_y_max", 0.2))

		V = self._roi_corners_ext(p0, u, v, n, w, l, y0, y1)  # 8x3
		# 12 edges
		E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
		lines = []
		for i,j in E:
			lines.append(V[i]); lines.append(V[j])
		self.roi_edges.setData(pos=np.asarray(lines, float), color=(0.85,0.85,0.85,0.9))
#=========================================================

#================ Graphics ===============================
	def _view_from_axis(self, dir):
	
		if self.plane is None:
			return
		
		self.current_view += dir
		if self.current_view < 0: self.current_view = 3
		elif self.current_view > 3: self.current_view = 0
		
		if self.current_view % 2: which = 'v'
		if not self.current_view % 2: which = 'u'
		if self.current_view < 2: sign = 1
		if self.current_view >= 2: sign = -1

		# table centre, ignoring local Y offset
		O, Rw = self._global_axis_origin()
		u, n, v = Rw[:,0], Rw[:,1], Rw[:,2]

		# choose direction
		dir_vec = (u if which == 'u' else v) * float(sign)
		dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)

		# camera placement: back away along dir, lift slightly along n
		back = 2.0 
		lift = 0.35
		center = np.asarray(O, dtype=float)
		eye = center - dir_vec * back + n * lift

		# center
		self.gl.opts['center'] = pg.Vector(float(center[0]), float(center[1]), float(center[2]))

		# azimuth from ±U/±V projection on XZ
		az = math.degrees(math.atan2(dir_vec[0], dir_vec[2]))
		elev = 15.0  # modest tilt

		dist = float(np.linalg.norm(eye - center))
		self.gl.setCameraPosition(distance=dist, elevation=float(elev), azimuth=float(az))

	def _set_default_view(self):
		center_z = 1.0
		center = pg.Vector(0.0, 0.0, center_z)

		self.gl.opts['center'] = center

		# Distance from the center, elevation (deg above XZ), azimuth (deg around Y)
		self.gl.setCameraPosition(distance=2, elevation=-80, azimuth=90)
		self.gl.orbit(0, 0)

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
	
	def _fade_line(self, origin, direction, half_len, color_rgb, steps=21):
		"""Returns (pos, rgba) for a long line with end-fade."""
		t = np.linspace(-half_len, half_len, steps)
		pos = origin[None,:] + t[:,None]*direction[None,:]
		# alpha tapered toward ends; keep thin lines darker outwards
		a = 0.90 * (1.0 - (np.abs(t)/half_len)*0.75)
		r,g,b = color_rgb
		rgba = np.column_stack([np.full_like(a, r), np.full_like(a, g), np.full_like(a, b), a])
		return pos.astype(np.float32), rgba.astype(np.float32)

	def _make_axis(self, origin, direction, color, width=2.0):
		p = np.array([origin, origin + direction], dtype=float)
		item = gl.GLLinePlotItem(pos=p, color=color, width=width, mode='lines')
		self.gl.addItem(item)
		return item
#=================================================================


def _fit_plane_svd(pts: np.ndarray):
	c = pts.mean(axis=0)
	P = pts - c
	# smallest singular vector = normal
	_, _, Vt = np.linalg.svd(P, full_matrices=False)
	n = Vt[-1]
	n = n / np.linalg.norm(n)
	# Make normal point "up-ish" for consistency
	if n[1] < 0:
		n = -n
	d = -float(n @ c)
	h = P @ n
	rms = float(np.sqrt(np.mean(h*h)))
	return n, d, c, rms


def _Ry(a):
	c, s = np.cos(a), np.sin(a)
	return np.array([[ c,0, s],
					[ 0,1, 0],
					[-s,0, c]], float)

def _Rx(a):
	c, s = np.cos(a), np.sin(a)
	return np.array([[1, 0, 0],
					[0, c,-s],
					[0, s, c]], float)

def _Rz(a):
	c, s = np.cos(a), np.sin(a)
	return np.array([[ c,-s, 0],
					[ s, c, 0],
					[ 0, 0, 1]], float)

def _wrap_deg(x):
	return ((x + 180.0) % 360.0) - 180.0


def _R_from_n_and_u(n: np.ndarray, u_hint: np.ndarray = None):
	y = n / np.linalg.norm(n)
	if u_hint is None or np.linalg.norm(u_hint) < 1e-8:
		u_hint = np.array([1.0, 0.0, 0.0]) if abs(y[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
	# project hint into plane
	x = u_hint - (u_hint @ y) * y
	nrm = np.linalg.norm(x)
	if nrm < 1e-9:
		x = np.array([1.0, 0.0, 0.0]) - y[0]*y
		x /= np.linalg.norm(x)
	else:
		x /= nrm
	z = np.cross(x, y)
	z /= np.linalg.norm(z)
	return np.column_stack([x, y, z])

def _RzRyRx(yaw_deg: float, pitch_deg: float, roll_deg: float):
	y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
	cy, sy = np.cos(y), np.sin(y)
	cp, sp = np.cos(p), np.sin(p)
	cr, sr = np.cos(r), np.sin(r)
	Rz = np.array([[ cy,-sy, 0],[ sy, cy, 0],[0,0,1]])
	Ry = np.array([[ cp, 0, sp],[ 0, 1, 0],[-sp, 0, cp]])
	Rx = np.array([[ 1, 0, 0],[ 0, cr,-sr],[0, sr, cr]])
	return Rz @ Ry @ Rx
	
