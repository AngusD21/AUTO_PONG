import sys, os, time, json, math, threading
from dataclasses import dataclass
from collections import deque
import numpy as np
import cv2
import serial

from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph.exporters import ImageExporter
import pyrealsense2 as rs

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from Ball_Tracking.graphics_objects import  AspectImageView, CollapsibleCard, RangeSlider, AxisGlyph, AXIS_COLORS
from Ball_Tracking.graphics_objects import  _backing_spin, _colored_span, _link_half_to_span, _link_span_to_half,make_full_slider_row, _span_box

PLANE_W = "Assets/PLANE_W.png"
PLAY_W = "Assets/PLAY_W.png"
WIZARD_W = "Assets/WIZARD_W.png"

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
		
		self.flip_u = False
		self.flip_v = False
		self.flip_n = False

		# Layout: GL on the left, controls on the right
		root = QtWidgets.QHBoxLayout(self)
		root.setContentsMargins(8,8,8,8)
		root.setSpacing(8)

		# ---------- 3D Tab ----------
		if 1:
			self.leftTabs = QtWidgets.QTabWidget()
			self.leftTabs.setTabPosition(QtWidgets.QTabWidget.North)
			self.leftTabs.setDocumentMode(True)
			root.addWidget(self.leftTabs, 1)

			# --- Tab 1: 3D (existing GL + HUD) ---
			tab3d = QtWidgets.QWidget()
			t3d_v = QtWidgets.QVBoxLayout(tab3d); t3d_v.setContentsMargins(0,0,0,0); t3d_v.setSpacing(6)

			self.gl = gl.GLViewWidget()
			self.gl.opts['distance'] = 2.0
			self.gl.setBackgroundColor((10,10,12))
			self.gl.installEventFilter(self)
			t3d_v.addWidget(self.gl, 1)

			hud = QtWidgets.QWidget()
			hud_h = QtWidgets.QHBoxLayout(hud); hud_h.setContentsMargins(0,0,0,0); hud_h.setSpacing(8)
			self.glyph_cam  = AxisGlyph("Camera")
			self.glyph_box  = AxisGlyph("Box")
			self.glyph_plane= AxisGlyph("Plane")
			hud_h.addWidget(self.glyph_cam); hud_h.addWidget(self.glyph_box); hud_h.addWidget(self.glyph_plane)
			t3d_v.addWidget(hud, 0)
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

			self.leftTabs.addTab(tab3d, "3D")

		# Camera Tab
		if 1:
			tabCam = QtWidgets.QWidget()
			tcam_v = QtWidgets.QVBoxLayout(tabCam); tcam_v.setContentsMargins(0,0,0,0)
			self.camLabel = AspectImageView(self)
			# self.camLabel.setAlignment(QtCore.Qt.AlignCenter)
			# self.camLabel.setMinimumHeight(200)
			# self.camLabel.setStyleSheet("background:#111; color:#888;")
			tcam_v.addWidget(self.camLabel, 1)
			self.leftTabs.addTab(tabCam, "Camera")

			self.cam_running = True
			self.cam_show_plane = False
			self.cam_show_roi   = False
			self._last_cam_ts   = 0.0
			self._cam_interval  = 1.0 / 30.0

		# Planar Tab
		if 1:
			tab2d = QtWidgets.QWidget()
			t2d_v = QtWidgets.QVBoxLayout(tab2d); t2d_v.setContentsMargins(0,0,0,0); t2d_v.setSpacing(6)

			# small tab bar for uv / u-n / v-n
			self.plane2dTabs = QtWidgets.QTabWidget()
			t2d_v.addWidget(self.plane2dTabs, 1)

			# U-V
			w_uv = QtWidgets.QWidget(); l_uv = QtWidgets.QVBoxLayout(w_uv); l_uv.setContentsMargins(4,4,4,4)
			self.plot_uv = pg.PlotWidget()
			self.sp_uv = pg.ScatterPlotItem(size=3, pxMode=True)  # 3 px points
			self.plot_uv.addItem(self.sp_uv)
			self.plot_uv.setLabel('bottom', 'z (m)'); self.plot_uv.setLabel('left', 'x (m)')
			self.plane2dTabs.addTab(w_uv, "Top Down")
			l_uv.addWidget(self.plot_uv)

			# U-N
			w_un = QtWidgets.QWidget(); l_un = QtWidgets.QVBoxLayout(w_un); l_un.setContentsMargins(4,4,4,4)
			self.plot_un = pg.PlotWidget()
			self.sp_un = pg.ScatterPlotItem(size=3, pxMode=True)
			self.plot_un.addItem(self.sp_un)
			self.plot_un.setLabel('bottom', 'z (m)'); self.plot_un.setLabel('left', 'y (m)')
			self.plane2dTabs.addTab(w_un, "Side X")
			l_un.addWidget(self.plot_un)

			# V-N
			w_vn = QtWidgets.QWidget(); l_vn = QtWidgets.QVBoxLayout(w_vn); l_vn.setContentsMargins(4,4,4,4)
			self.plot_vn = pg.PlotWidget()
			self.sp_vn = pg.ScatterPlotItem(size=3, pxMode=True)
			self.plot_vn.addItem(self.sp_vn)
			self.plot_vn.setLabel('bottom', 'x (m)'); self.plot_vn.setLabel('left', 'y (m)')
			self.plane2dTabs.addTab(w_vn, "Side Z")
			l_vn.addWidget(self.plot_vn)

			self.leftTabs.addTab(tab2d, "Plane 2D")

			# Overlays
			self.uv_box = pg.PlotDataItem(pen=pg.mkPen(200, 200, 200, 160), connect='finite')
			self.uv_roi = pg.PlotDataItem(pen=pg.mkPen(200, 200, 50, 255), connect='finite')
			self.plot_uv.addItem(self.uv_box); self.plot_uv.addItem(self.uv_roi)

			self.un_box = pg.PlotDataItem(pen=pg.mkPen(200, 200, 200, 160), connect='finite')
			self.un_roi = pg.PlotDataItem(pen=pg.mkPen(200, 200, 50, 255), connect='finite')
			self.plot_un.addItem(self.un_box); self.plot_un.addItem(self.un_roi)

			self.vn_box = pg.PlotDataItem(pen=pg.mkPen(200, 200, 200, 160), connect='finite')
			self.vn_roi = pg.PlotDataItem(pen=pg.mkPen(200, 200, 50, 255), connect='finite')
			self.plot_vn.addItem(self.vn_box); self.plot_vn.addItem(self.vn_roi)

			# Planes
			self.uv_plane_poly = pg.PlotDataItem(pen=pg.mkPen(colour=(255, 180, 185, 220), width=3), connect='all')
			self.un_plane_poly = pg.PlotDataItem(pen=pg.mkPen(colour=(255, 180, 185, 220), width=3), connect='all')
			self.vn_plane_poly = pg.PlotDataItem(pen=pg.mkPen(colour=(255, 180, 185, 220), width=3), connect='all')
			self.plot_uv.addItem(self.uv_plane_poly)
			self.plot_un.addItem(self.un_plane_poly)
			self.plot_vn.addItem(self.vn_plane_poly)


		# ---------- Right panel ----------
		if 1:
			panel = QtWidgets.QFrame()
			panel.setObjectName("PlaneRightPanel")
			panel_layout = QtWidgets.QVBoxLayout(panel)
			panel_layout.setContentsMargins(8, 8, 8, 8)
			panel_layout.setSpacing(6)
			root.addWidget(panel, 0)

			# Title
			if os.path.exists(PLANE_W):
				small_logo = QtWidgets.QLabel()
				pm = QtGui.QPixmap(PLANE_W).scaledToHeight(28, QtCore.Qt.SmoothTransformation)
				small_logo.setPixmap(pm)
				spaced = QtWidgets.QHBoxLayout()
				for ch in " PLANE WIZARD ":
					lbl = QtWidgets.QLabel(ch)
					lbl.setObjectName("titleLabel")
					spaced.addWidget(lbl, 1, QtCore.Qt.AlignCenter)
				spaced.addWidget(small_logo, 1, QtCore.Qt.AlignCenter) 
				spaced.addWidget(QtWidgets.QLabel(' '), 1, QtCore.Qt.AlignCenter) 
			panel_layout.addLayout(spaced)

			# Center (m)
			self.cx = _backing_spin(-10.0, 10.0, 0.001, 0.0)
			self.cy = _backing_spin(-10.0, 10.0, 0.001, 0.0)
			self.cz = _backing_spin(-10.0, 10.0, 0.001, 1.0)
			# Half-extents (m)
			self.ex = _backing_spin(0.05, 5.0, 0.001, 0.40)
			self.ey = _backing_spin(0.01, 1.0, 0.001, 0.05)
			self.ez = _backing_spin(0.05, 5.0, 0.001, 0.40)
			# Rotation (deg)
			self.yaw   = _backing_spin(-180.0, 180.0, 0.1, 0.0)
			self.pitch = _backing_spin(-180.0, 180.0, 0.1, 0.0)
			self.roll  = _backing_spin(-180.0, 180.0, 0.1, 0.0)
		
		# Context Row
		if 1:
			# Changes per left tab
			self.contextStack = QtWidgets.QStackedWidget()
			panel_layout.addWidget(self.contextStack)

			# 3D tab
			row3d_top = QtWidgets.QWidget()
			r3_top = QtWidgets.QHBoxLayout(row3d_top); r3_top.setContentsMargins(0,0,0,0); r3_top.setSpacing(6)
			btnResetView = QtWidgets.QPushButton("Reset View")
			self.btn_view_u = QtWidgets.QPushButton("<")
			self.btn_view_v = QtWidgets.QPushButton(">")
			self.planeVLabel = QtWidgets.QLabel("Plane View"); self.planeVLabel.setAlignment(QtCore.Qt.AlignCenter)
			r3_top.addWidget(btnResetView); r3_top.addSpacing(20)
			r3_top.addWidget(self.btn_view_u); r3_top.addWidget(self.planeVLabel); r3_top.addWidget(self.btn_view_v)

			# build 2 identical bottom rows
			self.row3d_bot, self.chkDepthColour3D, self.chkPlaneFade3D = self._build_bot_controls_row()
			self.row2d_bot, self.chkDepthColour2D, self.chkPlaneFade2D = self._build_bot_controls_row()
			self.chkDepthColour3D.setChecked(True)
			self.chkPlaneFade3D.setChecked(True)
			self.chkDepthColour2D.setChecked(True)
			self.chkPlaneFade2D.setChecked(True)

			def _sync(a, b):
				a.blockSignals(True); b.blockSignals(True)
				b.setChecked(a.isChecked())
				a.blockSignals(False); b.blockSignals(False)

			self.chkDepthColour3D.toggled.connect(lambda _: _sync(self.chkDepthColour3D, self.chkDepthColour2D))
			self.chkDepthColour2D.toggled.connect(lambda _: _sync(self.chkDepthColour2D, self.chkDepthColour3D))
			self.chkPlaneFade3D.toggled.connect(lambda _: _sync(self.chkPlaneFade3D, self.chkPlaneFade2D))
			self.chkPlaneFade2D.toggled.connect(lambda _: _sync(self.chkPlaneFade2D, self.chkPlaneFade3D))

			self.chkDepthColour = self.chkDepthColour3D
			self.chkPlaneFade   = self.chkPlaneFade3D

			page3d = QtWidgets.QWidget()
			page3d_v = QtWidgets.QVBoxLayout(page3d); page3d_v.setContentsMargins(0,0,0,0); page3d_v.setSpacing(4)
			page3d_v.addWidget(row3d_top)
			page3d_v.addWidget(self.row3d_bot)
			self.contextStack.addWidget(page3d)

			# wire the top-row buttons
			btnResetView.clicked.connect(self._set_default_view)
			self.btn_view_u.clicked.connect(lambda: self._view_from_axis(1))
			self.btn_view_v.clicked.connect(lambda: self._view_from_axis(-1))

			# Row for Camera tab
			rowCam = QtWidgets.QWidget(); rc = QtWidgets.QHBoxLayout(rowCam); rc.setContentsMargins(0,0,0,0); rc.setSpacing(6)
			self.chkCamPlane = QtWidgets.QCheckBox("Display Plane")
			self.chkCamROI   = QtWidgets.QCheckBox("Display ROI")
			self.btnCamStart = QtWidgets.QPushButton("Start")
			self.btnCamStop  = QtWidgets.QPushButton("Stop")
			self.btnCamSnap  = QtWidgets.QPushButton("Snapshot")
			rc.addWidget(self.chkCamPlane); rc.addWidget(self.chkCamROI); rc.addStretch(1)
			rc.addWidget(self.btnCamStart); rc.addWidget(self.btnCamStop); rc.addWidget(self.btnCamSnap)
			self.contextStack.addWidget(rowCam)

			self.chkCamPlane.toggled.connect(lambda v: setattr(self, "cam_show_plane", bool(v)))
			self.chkCamROI.toggled.connect(lambda v: setattr(self, "cam_show_roi",   bool(v)))

			self.btnCamStart.clicked.connect(lambda: setattr(self, "cam_running", True))
			self.btnCamStop.clicked.connect(lambda: setattr(self, "cam_running", False))
			self.btnCamSnap.clicked.connect(self._snapshot_camera_frame)

			# Plane 2D tab
			self.contextStack.addWidget(self.row2d_bot)

			# Switch the context row when the left tabs change
			self.leftTabs.currentChanged.connect(self._on_left_tab_changed)

		# --- Right sidebar (cards) ---
		scroll = QtWidgets.QScrollArea()
		scroll.setWidgetResizable(True)
		panel_layout.addWidget(scroll, 1)

		right_host = QtWidgets.QWidget()
		right = QtWidgets.QVBoxLayout(right_host)
		right.setContentsMargins(0,0,0,0)
		right.setSpacing(12)
		scroll.setWidget(right_host)

		# ---------- ROI CARD ----------
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
			self.row_yw, self.s_yw, self.sp_yw, i2v_yw, v2i_yw, _ = make_full_slider_row("Yaw",   -180.0, 180.0, 0.1, self.yaw.value(),   "y", "°")
			self.row_pt, self.s_pt, self.sp_pt, i2v_pt, v2i_pt, _ = make_full_slider_row("Pitch", -90.0, 90.0, 0.1, self.pitch.value(), "x", "°")
			self.row_rl, self.s_rl, self.sp_rl, i2v_rl, v2i_rl, _ = make_full_slider_row("Roll",  -90.0, 90.0, 0.1, self.roll.value(),  "z", "°")

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

			card_box._content.setVisible(False)
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

			# Local position (store sliders & mappers)
			row_lx, self.s_lx, self.sp_lx, self.i2v_lx, self.v2i_lx, _ = make_full_slider_row("Local X", -2.0, 2.0, 0.002, 0.0, "x", " m")
			row_ly, self.s_ly, self.sp_ly, self.i2v_ly, self.v2i_ly, _ = make_full_slider_row("Local Y", -2.0, 2.0, 0.002, 0.0, "y", " m")
			row_lz, self.s_lz, self.sp_lz, self.i2v_lz, self.v2i_lz, _ = make_full_slider_row("Local Z", -2.0, 2.0, 0.002, 0.0, "z", " m")
			pln_layout.addWidget(row_lx); pln_layout.addWidget(row_ly); pln_layout.addWidget(row_lz)

			# Local rotations (store sliders & mappers)
			row_yw2, self.s_yw2, self.sp_yw2, self.i2v_yw2, self.v2i_yw2, _ = make_full_slider_row("Yaw (about n)",   -180.0, 180.0, 0.1, 0.0, "y", "°")
			row_pt2, self.s_pt2, self.sp_pt2, self.i2v_pt2, self.v2i_pt2, _ = make_full_slider_row("Pitch (about u)",  -89.0,   89.0, 0.1, 0.0, "x", "°")
			row_rl2, self.s_rl2, self.sp_rl2, self.i2v_rl2, self.v2i_rl2, _ = make_full_slider_row("Roll (about v)",   -89.0,   89.0, 0.1, 0.0, "z", "°")
			pln_layout.addWidget(row_yw2); pln_layout.addWidget(row_pt2); pln_layout.addWidget(row_rl2)

			card_pln._content.setVisible(False)
			right.addWidget(card_pln)

			# --- Plane UI wiring ---
			self.sp_plane_w.valueChanged.connect(self._plane_apply_size)
			self.sp_plane_l.valueChanged.connect(self._plane_apply_size)

			self.chk_plane_visible.toggled.connect(lambda v: (setattr(self, "pl_visible", bool(v)), self._update_plane_draw()))

			# Local position offsets
			self.s_lx.valueChanged.connect(lambda i: (
				self.sp_lx.blockSignals(True), self.sp_lx.setValue(self.i2v_lx(i)), self.sp_lx.blockSignals(False),
				setattr(self, "_pl_off_local", np.array([self.i2v_lx(i), self._pl_off_local[1], self._pl_off_local[2]])),
				self._update_plane_from_ui()
			))
			self.s_ly.valueChanged.connect(lambda i: (
				self.sp_ly.blockSignals(True), self.sp_ly.setValue(self.i2v_ly(i)), self.sp_ly.blockSignals(False),
				setattr(self, "_pl_off_local", np.array([self._pl_off_local[0], self.i2v_ly(i), self._pl_off_local[2]])),
				self._update_plane_from_ui()
			))
			self.s_lz.valueChanged.connect(lambda i: (
				self.sp_lz.blockSignals(True), self.sp_lz.setValue(self.i2v_lz(i)), self.sp_lz.blockSignals(False),
				setattr(self, "_pl_off_local", np.array([self._pl_off_local[0], self._pl_off_local[1], self.i2v_lz(i)])),
				self._update_plane_from_ui()
			))

			# Local rotations (sliders drive state)
			self.s_yw2.valueChanged.connect(lambda i: (
				self.sp_yw2.blockSignals(True), self.sp_yw2.setValue(self.i2v_yw2(i)), self.sp_yw2.blockSignals(False),
				setattr(self, "_pl_yaw", self.i2v_yw2(i)), self._update_plane_from_ui()
			))
			self.s_pt2.valueChanged.connect(lambda i: (
				self.sp_pt2.blockSignals(True), self.sp_pt2.setValue(self.i2v_pt2(i)), self.sp_pt2.blockSignals(False),
				setattr(self, "_pl_pitch", self.i2v_pt2(i)), self._update_plane_from_ui()
			))
			self.s_rl2.valueChanged.connect(lambda i: (
				self.sp_rl2.blockSignals(True), self.sp_rl2.setValue(self.i2v_rl2(i)), self.sp_rl2.blockSignals(False),
				setattr(self, "_pl_roll", self.i2v_rl2(i)), self._update_plane_from_ui()
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
			card_roi._content.setVisible(False)
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
			self.btn_save    = QtWidgets.QPushButton("Save")
			self.btn_cancel  = QtWidgets.QPushButton("Cancel")
			btn_row.addWidget(self.btn_cancel)
			btn_row.addWidget(self.btn_save)
			panel_layout.addLayout(btn_row)

			self.current_view = 0
			# Wire updates into draw path
			for w in (self.cx,self.cy,self.cz,self.ex,self.ey,self.ez,self.yaw,self.pitch,self.roll):
				w.valueChanged.connect(self._update_box)
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
		idx = self.leftTabs.currentIndex()

		# Tab 0: 3D
		if idx == 0:
			xyz = self.worker.get_point_cloud_snapshot()
			if xyz is not None and xyz.shape[0] > 0 and self.plane is not None:
				m8 = self._mask_radius_from_plane_center(xyz, 5)
				xyz_vis = xyz[m8]
			else:
				m8 = None
				xyz_vis = None

			if xyz_vis is not None and xyz_vis.shape[0] > 0:
				c = np.array([self.cx.value(), self.cy.value(), self.cz.value()], float)
				e = np.array([self.ex.value(), self.ey.value(), self.ez.value()], float)
				R = self._R_yxz(self.yaw.value(), self.pitch.value(), self.roll.value())

				Xloc = (xyz_vis - c) @ R
				inside = (np.abs(Xloc[:,0]) <= e[0]) & (np.abs(Xloc[:,1]) <= e[1]) & (np.abs(Xloc[:,2]) <= e[2])

				# ROI (on full cloud so alpha can lift ROI even when far)
				roi_full = self._roi_mask(xyz) if getattr(self, "roi_visible", True) else np.zeros(xyz.shape[0], bool)
				roi_vis  = roi_full[m8]

				# Base color/alpha
				colors = np.zeros((xyz_vis.shape[0], 4), float)
				colors[:] = (1, 1, 1, 0.60)

				# Depth-to-camera colouring
				if self.chkDepthColour.isChecked():
					rgb = self._apply_jet_colour_by_cam_dist(xyz_vis)
					colors[:, :3] = rgb

				# Plane-center fade for alpha?
				if self.chkPlaneFade.isChecked() and self.plane is not None:
					a = self._alpha_by_plane_center(xyz_vis, roi_vis)
					colors[:, 3] = a

				# ROI highlight (ensure ROI points very visible)
				colors[roi_vis, :3] = np.minimum(colors[roi_vis, :3] + 0.25, 1.0)
				colors[roi_vis, 3]  = 1.0

				if self.pts_visible:
					colors[inside] = (1.0, 1.0, 1.0, 0.8)
				self.cloud_item.setData(pos=xyz_vis, color=colors)

		# Tab 1: Camera
		elif idx == 1:
			frm = getattr(self.worker, "get_rgb_frame", lambda: None)()
			if frm is not None:
				# assume BGR uint8
				if len(frm.shape) == 3 and frm.shape[2] == 3:
					rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
					h, w, _ = rgb.shape
					qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
					self.camLabel.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
						self.camLabel.width(), self.camLabel.height(),
						QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
		
		# Tab 2: Plane 2D
		else:
			xyz = self.worker.get_point_cloud_snapshot()
			if xyz is None or xyz.shape[0] == 0 or self.plane is None:
				# clear current view quickly
				idx2 = self.plane2dTabs.currentIndex()
				if idx2 == 0: self.sp_uv.setData([], [])
				elif idx2 == 1: self.sp_un.setData([], [])
				else: self.sp_vn.setData([], [])
				return

			m8 = self._mask_radius_from_plane_center(xyz, 2)
			if not m8.any():
				idx2 = self.plane2dTabs.currentIndex()
				if idx2 == 0: self.sp_uv.setData([], [])
				elif idx2 == 1: self.sp_un.setData([], [])
				else: self.sp_vn.setData([], [])
				return

			down_sample = 4
			xyz8 = xyz[m8][::down_sample]
			roi8 = self._roi_mask(xyz)[m8][::down_sample]

			# --- Project once ---
			R = self._plane_R_from_base_and_loc()  # [u n v]
			u, n, v = R[:,0], R[:,1], R[:,2]
			p0 = self._plane_p0_from_base_and_loc()
			d = xyz8 - p0[None, :]
			u_coord = d @ u
			n_coord = d @ n
			v_coord = d @ v

			# ---- Colour/alpha only if needed ----
			want_depth = self.chkDepthColour.isChecked()
			want_fade  = self.chkPlaneFade.isChecked()

			if not want_depth and not want_fade:
				# FAST PATH: one brush for all points, no per-spot objects
				brush = pg.mkBrush(255, 255, 255, 230)
				idx2 = self.plane2dTabs.currentIndex()
				if idx2 == 0:
					self.sp_uv.setData(u_coord, v_coord, brush=brush, pen=None, size=3)
				elif idx2 == 1:
					self.sp_un.setData(u_coord, n_coord, brush=brush, pen=None, size=3)
				else:
					self.sp_vn.setData(v_coord, n_coord, brush=brush, pen=None, size=3)
				return

			# SLOW PATH (still optimized): compute RGBA once, quantize, reuse brushes
			# Base color
			if want_depth:
				rgb01 = self._apply_jet_colour_by_cam_dist(xyz8)
			else:
				rgb01 = np.ones((xyz8.shape[0], 3), dtype=float)

			# Alpha
			if want_fade:
				a01 = self._alpha_by_plane_center(xyz8, roi8)
			else:
				a01 = np.ones(xyz8.shape[0], dtype=float)

			# ROI lift (brighten & full alpha)
			if roi8.any():
				rgb01[roi8] = np.minimum(rgb01[roi8] + 0.25, 1.0)
				a01[roi8]   = 1.0

			# pack to uint8 once
			rgba = np.empty((xyz8.shape[0], 4), dtype=np.uint8)
			rgba[:,0:3] = np.clip((rgb01 * 255.0), 0, 255).astype(np.uint8)
			rgba[:,3]   = np.clip((a01   * 255.0), 0, 255).astype(np.uint8)

			uniq_rgba, inv = self._quantize_rgba(rgba, q=32)
			pool = self._brush_pool_from_rgba(uniq_rgba)
			brushes = [pool[i] for i in inv]

			# Update ONLY the selected sub-tab
			idx2 = self.plane2dTabs.currentIndex()
			if idx2 == 0:
				self.sp_uv.setData(u_coord, v_coord, brush=brushes, pen=None, size=3)
			elif idx2 == 1:
				self.sp_un.setData(u_coord, n_coord, brush=brushes, pen=None, size=3)
			else:
				self.sp_vn.setData(v_coord, n_coord, brush=brushes, pen=None, size=3)
			
			self._lock_viewbox(idx2)

			# Build polylines for each projection (separate segments with NaN)
			def segs(xs, ys):
				X, Y = [], []
				for a,b in edges:
					X += [xs[a], xs[b], np.nan]
					Y += [ys[a], ys[b], np.nan]
				return np.array(X), np.array(Y)

			edges = [(0,1),(2,3),(4,5),(6,7),
					(0,2),(1,3),(4,6),(5,7), 
					(0,4),(1,5),(2,6),(3,7)] 

			# Overlays
			# Box
			if self.box_visible:
				boxW = self._box_corners_world()
				uvn  = self._project_uvn(boxW)  
				U, N, V = uvn[:,0], uvn[:,1], uvn[:,2]
				if idx2 == 0: x_uv, y_uv = segs(U, V); self.uv_box.setData(x_uv, y_uv)
				if idx2 == 1: x_un, y_un = segs(U, N); self.un_box.setData(x_un, y_un)
				if idx2 == 2: x_vn, y_vn = segs(V, N); self.vn_box.setData(x_vn, y_vn)
			else:
				self.uv_box.setData([],[]); self.un_box.setData([],[]); self.vn_box.setData([],[])

			# ROI
			if self.roi_visible:
				p0 = np.asarray(self.plane.p0, float)
				u  = np.asarray(self.plane.u,  float)
				v  = np.asarray(self.plane.v,  float)
				n  = np.asarray(self.plane.n,  float)
				w  = float(self.pl_size_x)
				l  = float(self.pl_size_z)
				y0 = float(getattr(self, "roi_y_min", 0.0))
				y1 = float(getattr(self, "roi_y_max", 0.2))

				roiW = self._roi_corners_ext(p0, u, v, n, w, l, y0, y1)   # (8,3) world
				uvn  = self._project_uvn(roiW)                            # (8,3) projected (with flips)

				# --- UV uses the rectangle polygon derived from min/max ---
				if idx2 == 0:
					UVx, UVy = self._roi_square_uv_poly(uvn)
					self.uv_roi.setData(UVx, UVy)   # closed loop

				# --- UN / VN keep the 12-edge wireframe (your existing code) ---
				else:
					U, N, V = uvn[:,0], uvn[:,1], uvn[:,2]
					if idx2 == 1:
						x_un, y_un = segs(U, N); self.un_roi.setData(x_un, y_un)
					else:
						x_vn, y_vn = segs(V, N); self.vn_roi.setData(x_vn, y_vn)
			else:
				self.uv_roi.setData([], []); self.un_roi.setData([], []); self.vn_roi.setData([], [])

			if self.pl_visible:
				pc = self._plane_corners_world()
				if pc is not None:
					uvn_plane = self._project_uvn(pc)  
					Uc, Nc, Vc = uvn_plane[:,0], uvn_plane[:,1], uvn_plane[:,2]

					UVx = np.r_[Uc, Uc[:1]]; UVy = np.r_[Vc, Vc[:1]]
					UNx = np.r_[Uc, Uc[:1]]; UNy = np.r_[Nc, Nc[:1]]
					VNx = np.r_[Vc, Vc[:1]]; VNy = np.r_[Nc, Nc[:1]]

					if idx2 == 0: self.uv_plane_poly.setData(UVx, UVy)
					if idx2 == 1: self.un_plane_poly.setData(UNx, UNy)
					if idx2 == 2: self.vn_plane_poly.setData(VNx, VNy)
			else:
				self.uv_plane_poly.setData([], []); self.un_plane_poly.setData([], []); self.vn_plane_poly.setData([], [])

		self._update_box()
		if self.plane is not None:
			self._update_plane_draw()
			self._update_global_axes()
		self._update_axis_huds()

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
			
			# ----- reflect local offsets into spin + slider -----
			self.sp_lx.blockSignals(True); self.sp_lx.setValue(self._pl_off_local[0]); self.sp_lx.blockSignals(False)
			self.sp_ly.blockSignals(True); self.sp_ly.setValue(self._pl_off_local[1]); self.sp_ly.blockSignals(False)
			self.sp_lz.blockSignals(True); self.sp_lz.setValue(self._pl_off_local[2]); self.sp_lz.blockSignals(False)
			self.s_lx.blockSignals(True); self.s_lx.setValue(self.v2i_lx(self._pl_off_local[0])); self.s_lx.blockSignals(False)
			self.s_ly.blockSignals(True); self.s_ly.setValue(self.v2i_ly(self._pl_off_local[1])); self.s_ly.blockSignals(False)
			self.s_lz.blockSignals(True); self.s_lz.setValue(self.v2i_lz(self._pl_off_local[2])); self.s_lz.blockSignals(False)

			# ----- reflect local rotations into spin + slider -----
			self.sp_yw2.blockSignals(True); self.sp_yw2.setValue(self._pl_yaw);   self.sp_yw2.blockSignals(False)
			self.sp_pt2.blockSignals(True); self.sp_pt2.setValue(self._pl_pitch); self.sp_pt2.blockSignals(False)
			self.sp_rl2.blockSignals(True); self.sp_rl2.setValue(self._pl_roll);  self.sp_rl2.blockSignals(False)
			self.s_yw2.blockSignals(True); self.s_yw2.setValue(self.v2i_yw2(self._pl_yaw));   self.s_yw2.blockSignals(False)
			self.s_pt2.blockSignals(True); self.s_pt2.setValue(self.v2i_pt2(self._pl_pitch)); self.s_pt2.blockSignals(False)
			self.s_rl2.blockSignals(True); self.s_rl2.setValue(self.v2i_rl2(self._pl_roll));  self.s_rl2.blockSignals(False)

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
		
		self.saved.emit({})

		# Write the full state to JSON
		self._canonicalize_base_from_current()
		state = self._state_to_dict()
		try:
			with open(self.config_path, "w", encoding="utf-8") as f:
				json.dump(state, f, indent=2)
			QtWidgets.QToolTip.showText(self.mapToGlobal(QtCore.QPoint(0,0)),
										f"Saved config → {os.path.basename(self.config_path)}")
		except Exception as e:
			QtWidgets.QMessageBox.warning(self, "Save error", f"Could not write config:\n{e}")

	def _canonicalize_base_from_current(self):
		if self.plane is None:
			return

		# Current displayed plane pose (already includes flip_n & locals)
		R_cur = np.column_stack([np.asarray(self.plane.u), 
								np.asarray(self.plane.n),
								np.asarray(self.plane.v)])
		# R_cur = self._orthonormalize(R_cur)
		p0_cur = np.asarray(self.plane.p0, float)

		# Local transforms currently in the UI
		R_loc = self._R_yxz(self._pl_yaw, self._pl_pitch, self._pl_roll)  
		off_loc = np.asarray(self._pl_off_local, float)
		R_base = R_cur @ R_loc.T

		# R_base = self._orthonormalize(R_base)
		p0_base = p0_cur - R_base @ off_loc

		self._pl_R_base = R_base
		self._pl_p0_base = p0_base

	def eventFilter(self, obj, ev):
		if obj is self.gl and ev.type() == QtCore.QEvent.MouseButtonDblClick:
			pos = ev.pos()
			self._pick_and_center(pos.x(), pos.y())
			return True
		return super().eventFilter(obj, ev)

	def _set_worker(self, worker):
		self.worker = worker
		if hasattr(worker, "rgb_for_wizard"):
			worker.rgb_for_wizard.connect(self._on_rgb_for_wizard)

	def _on_left_tab_changed(self, idx:int):
		# 0: 3D, 1: Camera, 2: Plane 2D
		self.contextStack.setCurrentIndex(idx)

	def _snapshot_camera_frame(self):
		frm = getattr(self.worker, "get_rgb_frame", lambda: None)()
		if frm is None:
			return
		ts = time.strftime("%Y%m%d_%H%M%S")
		path = os.path.join(os.path.expanduser("~"), f"camera_{ts}.png")
		try:
			cv2.imwrite(path, frm)
			QtWidgets.QToolTip.showText(self.mapToGlobal(QtCore.QPoint(0,0)), f"Saved {os.path.basename(path)}")
		except Exception as e:
			QtWidgets.QMessageBox.warning(self, "Snapshot", f"Could not save:\n{e}")

#============== Projection Funcs ========================
	def _plane_corners_world(self):
		"""Return (4,3) world coords of plane rectangle corners using current size/pose."""
		if self.plane is None:
			return None
		w = float(self.pl_size_x); l = float(self.pl_size_z)
		hx, hz = 0.5*w, 0.5*l
		u, n, v = self.plane.u, self.plane.n, self.plane.v
		p0 = self.plane.p0
		corners = np.vstack([
			p0 + (-hx)*u + (-hz)*v,
			p0 + ( +hx)*u + (-hz)*v,
			p0 + ( +hx)*u + ( +hz)*v,
			p0 + (-hx)*u + ( +hz)*v,
		])
		return corners

	def _box_corners_world(self):
		# box center/extends/rotation from your existing controls
		c = np.array([self.cx.value(), self.cy.value(), self.cz.value()], float)
		e = np.array([self.ex.value(), self.ey.value(), self.ez.value()], float)
		Rloc = self._R_yxz(self.yaw.value(), self.pitch.value(), self.roll.value())
		# 8 corners in local coords
		s = np.array([[-1,-1,-1],[+1,-1,-1],[-1,+1,-1],[+1,+1,-1],
					[-1,-1,+1],[+1,-1,+1],[-1,+1,+1],[+1,+1,+1]], float) * e
		return c[None,:] + s @ Rloc.T  # (8,3) world
	
	def _roi_corners_world(self):
		# ROI center/extends/rotation from your existing controls
		c = np.array([self.cx.value(), self.cy.value(), self.cz.value()], float)
		e = np.array([self.ex.value(), self.ey.value(), self.ez.value()], float)
		Rloc = self._R_yxz(self.yaw.value(), self.pitch.value(), self.roll.value())
		# 8 corners in local coords
		s = np.array([[-1,-1,-1],[+1,-1,-1],[-1,+1,-1],[+1,+1,-1],
					[-1,-1,+1],[+1,-1,+1],[-1,+1,+1],[+1,+1,+1]], float) * e
		return c[None,:] + s @ Rloc.T  # (8,3) world

	def _project_uvn(self, P):
		"""Project Nx3 world points into (u,n,v)."""
		R = self._plane_R_from_base_and_loc()  # [u n v]
		p0 = self._plane_p0_from_base_and_loc()
		D = P - p0[None,:]
		uvn = D @ R  # columns are [u, n, v]
		return uvn  # (N,3)

	def _quantize_rgba(self, rgba_u8: np.ndarray, q: int = 16):
		if rgba_u8.dtype != np.uint8:
			rgba_u8 = rgba_u8.astype(np.uint8, copy=False)
		rgba_u8 = np.ascontiguousarray(rgba_u8)

		# snap channels to {0, step, 2*step, ..., 255}
		qstep = max(1, 255 // q)
		q_rgba = (rgba_u8 // qstep) * qstep  # uint8

		# pack to 32-bit for fast unique
		packed = (q_rgba[:, 0].astype(np.uint32) << 24) | \
				(q_rgba[:, 1].astype(np.uint32) << 16) | \
				(q_rgba[:, 2].astype(np.uint32) << 8)  | \
				(q_rgba[:, 3].astype(np.uint32))

		uniq, inv = np.unique(packed, return_inverse=True)  # <-- fixed name

		uniq_rgba = np.column_stack([
			(uniq >> 24) & 0xFF,
			(uniq >> 16) & 0xFF,
			(uniq >> 8)  & 0xFF,
			(uniq)       & 0xFF
		]).astype(np.uint8)

		return uniq_rgba, inv

	def _brush_pool_from_rgba(self, uniq_rgba_u8: np.ndarray):
		if not hasattr(self, "_brush_cache"):
			self._brush_cache = {}  # key: (r,g,b,a) -> QBrush
		pool = []
		for r, g, b, a in uniq_rgba_u8:
			key = (int(r), int(g), int(b), int(a))
			br = self._brush_cache.get(key)
			if br is None:
				br = pg.mkBrush(key[0], key[1], key[2], key[3])
				self._brush_cache[key] = br
			pool.append(br)
		return pool
#========================================================

#============== Plane Functions =========================	
	def _plane_R_from_base_and_loc(self):
		# if self.plane is None:
		# 	return None
		y, p, r = np.deg2rad([self._pl_yaw, self._pl_pitch, self._pl_roll])
		Rloc = _Ry(y) @ _Rx(p) @ _Rz(r)
		R0 = self._pl_R_base @ Rloc         
		u = R0[:, 0]
		n = R0[:, 1]
		v = R0[:, 2]
		if getattr(self, "flip_n", False):
			n = -n
		return np.column_stack([u, n, v])

	def _plane_p0_from_base_and_loc(self):
		# if self.plane is None:
		# 	return None		
		R = self._plane_R_from_base_and_loc()
		u, n, v = R[:,0], R[:,1], R[:,2]
		off = self._pl_off_local
		return self._pl_p0_base + off[0]*u + off[1]*n + off[2]*v

	def _update_plane_from_ui(self):
		"""Refresh self.plane (geometry) from UI offsets/rotations/size and redraw."""
		if self.plane is None:
			self._update_plane_draw()
			return
		R = self._plane_R_from_base_and_loc()
		p0 = self._plane_p0_from_base_and_loc()
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

	def _order_roi8_uvn(self, uvn):
		assert uvn.shape == (8, 3)
		U, N, V = uvn[:, 0], uvn[:, 1], uvn[:, 2]

		# Split into two faces by N (plane-normal axis in projection)
		# Use a numeric split to avoid sign/center assumptions.
		orderN = np.argsort(N)
		lower_idxs = orderN[:4]   # 4 smallest N
		upper_idxs = orderN[4:]   # 4 largest  N

		def _ccw_on_face(idxs):
			pts = uvn[idxs][:, [0, 2]]          # (U, V) plane
			c = pts.mean(axis=0)
			ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
			return idxs[np.argsort(ang)]        # CCW

		lower_ccw = _ccw_on_face(lower_idxs)
		upper_ccw = _ccw_on_face(upper_idxs)

		return np.concatenate([lower_ccw, upper_ccw])

	def _roi_square_uv_poly(self, uvn: np.ndarray):

		assert uvn.shape == (8, 3)
		U = uvn[:, 0]
		V = uvn[:, 2]

		# The ROI in u–v is axis-aligned ⇒ corners are at (min/max U) × (min/max V)
		umin, umax = float(U.min()), float(U.max())
		vmin, vmax = float(V.min()), float(V.max())

		# Order consistently (CCW): (umin,vmin) → (umax,vmin) → (umax,vmax) → (umin,vmax) → back to start
		UVx = np.array([umin, umax, umax, umin, umin], dtype=float)
		UVy = np.array([vmin, vmin, vmax, vmax, vmin], dtype=float)
		return UVx, UVy
#=========================================================

#================ Graphics ===============================
	def _plane_center_world(self):
		# plane origin at base + local offset
		return self._plane_p0_from_base_and_loc() if self.plane is not None else np.zeros(3)

	def _camera_origin_world(self):
		# Try worker first, else fall back to (0,0,0)
		p = getattr(self.worker, "get_camera_origin_world", lambda: None)()
		return p if p is not None else np.zeros(3)

	def _mask_radius_from_plane_center(self, xyz, rad):
		p0 = self._plane_center_world()
		return np.linalg.norm(xyz - p0[None, :], axis=1) <= rad

	def _roi_mask(self, xyz):
		# Same ROI logic you already use in 3D (box in local coords)
		c = np.array([self.cx.value(), self.cy.value(), self.cz.value()], float)
		e = np.array([self.ex.value(), self.ey.value(), self.ez.value()], float)
		R = self._R_yxz(self.yaw.value(), self.pitch.value(), self.roll.value())
		Xloc = (xyz - c) @ R
		return (np.abs(Xloc[:,0]) <= e[0]) & (np.abs(Xloc[:,1]) <= e[1]) & (np.abs(Xloc[:,2]) <= e[2])

	def _apply_jet_colour_by_cam_dist(self, xyz):
		"""Return RGB in [0,1] using JET mapping of distance to camera (auto range)."""
		d = np.linalg.norm(xyz, axis=1)
		if d.size == 0:
			return np.zeros((0,3))
		# Robust min/max to avoid flicker; tweak as needed
		d0, d1 = np.percentile(d, 1), np.percentile(d, 95)
		if d1 <= d0: d1 = d0 + 1e-6
		t = np.clip((d - d0) / (d1 - d0), 0.0, 1.0)
		return self._jet_rgb01(t)
	
	def _lock_viewbox(self, idx2):
		vb = (self.plot_uv if idx2==0 else self.plot_un if idx2==1 else self.plot_vn).getViewBox()
		vb.disableAutoRange(pg.ViewBox.XYAxes)

	def _alpha_by_plane_center(self, xyz, roi_mask):
		"""Alpha fades with radius from plane center; ROI points stay opaque."""
		p0 = self._plane_center_world()
		r = np.linalg.norm(xyz - p0[None, :], axis=1)
		# 0m→1.0, 8m→~0.1 (tweak floor if you want)
		a = 1.0 - (r / 3.0)
		a = np.clip(a, 0.1, 0.6)
		a[roi_mask] = 1.0
		return a

	def _jet_rgb01(self, t):
		"""t in [0,1] → approximate JET RGB in [0,1]."""
		# piecewise JET (blue→cyan→yellow→red)
		r = np.clip(1.5 - np.abs(4.0*(t - 0.75)), 0, 1)
		g = np.clip(1.5 - np.abs(4.0*(t - 0.50)), 0, 1)
		b = np.clip(1.5 - np.abs(4.0*(t - 0.25)), 0, 1)
		return np.stack([r, g, b], axis=1)

	def _update_axis_huds(self):
		try:
			VM = np.array(self.gl.viewMatrix().data()).reshape(4,4).T
			Rw = VM[:3,:3]
		except Exception:
			Rw = np.eye(3)
		R_cam_world = Rw.T
		self.glyph_cam.setRotationCam(R_cam_world)
		R_box_world = self._R_yxz(self.yaw.value(), self.pitch.value(), self.roll.value())
		self.glyph_box.setRotationCam(Rw @ R_box_world)
		if self.plane is not None:
			R_plane_world = np.column_stack([self.plane.u, self.plane.n, self.plane.v])
			self.glyph_plane.setRotationCam(Rw @ R_plane_world)

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
	
	def _build_bot_controls_row(self):
		row = QtWidgets.QWidget()
		h = QtWidgets.QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.setSpacing(6)

		btnToggleBox   = QtWidgets.QPushButton("Box")
		btnTogglePlane = QtWidgets.QPushButton("Plane")
		btnToggleRoi   = QtWidgets.QPushButton("ROI")
		chkDepth       = QtWidgets.QCheckBox("Depth")
		chkFade        = QtWidgets.QCheckBox("Fade")

		h.addWidget(btnToggleBox)
		h.addWidget(btnTogglePlane)
		h.addWidget(btnToggleRoi)
		h.addStretch(1)
		h.addWidget(chkDepth)
		h.addWidget(chkFade)

		# wire buttons to the same state you already use
		btnToggleBox.clicked.connect(lambda: (setattr(self, "box_visible", not getattr(self, "box_visible", True)),
											self.chk_box_visible.setChecked(self.box_visible),
											self._update_box()))
		btnTogglePlane.clicked.connect(lambda: (setattr(self, "pl_visible", not getattr(self, "pl_visible", True)),
												self.chk_plane_visible.setChecked(self.pl_visible),
												self._update_plane_draw()))
		btnToggleRoi.clicked.connect(lambda: (setattr(self, "roi_visible", not getattr(self, "roi_visible", True)),
											self.chk_roi_visible.setChecked(self.roi_visible),
											self._update_roi_preview()))
		return row, chkDepth, chkFade

	def resizeEvent(self, ev: QtGui.QResizeEvent):
		super().resizeEvent(ev)
		# Recompute once per resize
		if hasattr(self, "camLabel"):
			# Cache the *content* size available to the label (layout margins already handled by layouts)
			self._cam_target_size = self.camLabel.size()

	@QtCore.pyqtSlot(QtGui.QImage)
	def _on_rgb_for_wizard(self, qimg: QtGui.QImage):
		self.camLabel.setImage(qimg)

#=================================================================

def _plane_R_from_base_and_loc(_pl_R_base, _pl_yaw, _pl_pitch, _pl_roll, flip_n=False):
	y, p, r = np.deg2rad([_pl_yaw, _pl_pitch, _pl_roll])
	Rloc = _Ry(y) @ _Rx(p) @ _Rz(r)
	R0 = _pl_R_base @ Rloc         
	u = R0[:, 0]
	n = R0[:, 1]
	v = R0[:, 2]
	if flip_n:
		n = -n
	return np.column_stack([u, n, v])

def _plane_corners_world(plane, pl_size_x, pl_size_z):
	"""Return (4,3) world coords of plane rectangle corners using current size/pose."""
	if plane is None:
		return None
	w = float(pl_size_x); l = float(pl_size_z)
	hx, hz = 0.5*w, 0.5*l
	u, n, v = plane.u, plane.n, plane.v
	p0 = plane.p0
	corners = np.vstack([
		p0 + (-hx)*u + (-hz)*v,
		p0 + ( +hx)*u + (-hz)*v,
		p0 + ( +hx)*u + ( +hz)*v,
		p0 + (-hx)*u + ( +hz)*v,
	])
	return corners

def _box_mask(xyz, cx, cy, cz, ex, ey, ez, yaw, pitch, roll):
	c = np.array([cx, cy, cz], float)
	e = np.array([ex, ey, ez], float)
	R = _R_yxz(yaw, pitch, roll)
	Xloc = (xyz - c) @ R
	return (np.abs(Xloc[:,0]) <= e[0]) & (np.abs(Xloc[:,1]) <= e[1]) & (np.abs(Xloc[:,2]) <= e[2])

def corners_world(cx, cy, cz, ex, ey, ez, yaw, pitch, roll):
	# ROI center/extends/rotation from your existing controls
	c = np.array([cx, cy, cz], float)
	e = np.array([ex, ey, ez], float)
	Rloc = _R_yxz(yaw, pitch, roll)
	# 8 corners in local coords
	s = np.array([[-1,-1,-1],[+1,-1,-1],[-1,+1,-1],[+1,+1,-1],
				[-1,-1,+1],[+1,-1,+1],[-1,+1,+1],[+1,+1,+1]], float) * e
	return c[None,:] + s @ Rloc.T  # (8,3) world

def _project_uvn(P):
	"""Project Nx3 world points into (u,n,v)."""
	R = _plane_R_from_base_and_loc()  # [u n v]
	p0 = _plane_p0_from_base_and_loc()
	D = P - p0[None,:]
	uvn = D @ R  # columns are [u, n, v]
	return uvn  # (N,3)

def _plane_p0_from_base_and_loc(_pl_off_local, _pl_p0_base):
	R = _plane_R_from_base_and_loc()
	u, n, v = R[:,0], R[:,1], R[:,2]
	off = _pl_off_local
	return _pl_p0_base + off[0]*u + off[1]*n + off[2]*v

def _R_yxz(yaw_deg, pitch_deg, roll_deg):
	y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
	return _Ry(y) @ _Rx(p) @ _Rz(r)

def _euler_from_R_yxz(R):
	pitch = -np.degrees(np.arcsin(R[1,2]))
	yaw   =  np.degrees(np.arctan2(R[0,2], R[2,2]))
	roll  =  np.degrees(np.arctan2(R[1,0], R[1,1]))
	return _wrap_deg(yaw), _wrap_deg(pitch), _wrap_deg(roll)


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
	
