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

from Ball_Tracking.plane_math import _render_interest_region_from_cloud, colourise_depth, overlay_plane_and_roi_on_bgr_po

AXIS_COLORS = {
	"x": QtGui.QColor(220,  0,  0),
	"y": QtGui.QColor( 0, 200,  0),
	"z": QtGui.QColor( 0, 60, 220),
}

class CollapsibleCard(QtWidgets.QWidget):
	def __init__(self, title: str, parent=None, start_collapsed=True):
		super().__init__(parent)

		# Header button
		self._button = QtWidgets.QToolButton(self)
		self._button.setText(title)
		self._button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
		self._button.setCheckable(True)
		self._button.setChecked(not start_collapsed)
		self._button.setArrowType(QtCore.Qt.DownArrow if not start_collapsed else QtCore.Qt.RightArrow)

		# Outer frame that draws the border
		self._frame = QtWidgets.QFrame(self)
		self._frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self._frame.setObjectName("CardFrame")

		# Content widget inside the frame
		self._content = QtWidgets.QWidget(self._frame)
		self._content.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
		self._content_layout = QtWidgets.QVBoxLayout(self._content)
		self._content_layout.setContentsMargins(12, 8, 12, 12)
		self._content_layout.setSpacing(8)

		fv = QtWidgets.QVBoxLayout(self._frame)
		fv.setContentsMargins(0, 0, 0, 0)
		fv.setSpacing(0)
		fv.addWidget(self._content)

		lay = QtWidgets.QVBoxLayout(self)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.setSpacing(4)
		lay.addWidget(self._button)
		lay.addWidget(self._frame)

		# style
		self.setStyleSheet("""
			QToolButton { font-weight:600; padding:6px 10px; }
			QFrame#CardFrame { border:1px solid #404040; border-radius:8px; }
		""")

		# initial state
		self._apply_state(expanded=not start_collapsed)

		# connect
		self._button.toggled.connect(self._on_toggled)

	def _on_toggled(self, checked: bool):
		self._button.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
		self._apply_state(expanded=checked)
		self.updateGeometry()

	def _apply_state(self, *, expanded: bool):
		if expanded:
			self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
			self._content.setVisible(True)
			self._frame.setVisible(True)
			self._frame.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)
		else:
			self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
			self._frame.setVisible(False)
			self._content.setVisible(False)
			self._frame.setMaximumHeight(0)
			# (optional) limit overall height to the button’s height
			# h = self._button.sizeHint().height() + 4
			# self.setMaximumHeight(h)
		self.adjustSize()

	def content_layout(self) -> QtWidgets.QVBoxLayout:
		return self._content_layout
	
class RangeSlider(QtWidgets.QSlider):
	lowerValueChanged = QtCore.pyqtSignal(int)
	upperValueChanged = QtCore.pyqtSignal(int)

	def __init__(self, orient=QtCore.Qt.Horizontal, parent=None):
		super().__init__(orient, parent)
		self._lower = self.minimum()
		self._upper = self.maximum()
		self._handle_w = 12
		self.setMouseTracking(True)
		self.setTickPosition(QtWidgets.QSlider.TicksBelow)

	def lowerValue(self): return self._lower
	def upperValue(self): return self._upper
	def setLowerValue(self, v):
		v = max(self.minimum(), min(v, self._upper))
		if v != self._lower:
			self._lower = v
			self.lowerValueChanged.emit(v)
			self.update()

	def setUpperValue(self, v):
		v = min(self.maximum(), max(v, self._lower))
		if v != self._upper:
			self._upper = v
			self.upperValueChanged.emit(v)
			self.update()

	def pixelPosToRangeValue(self, x):
		return QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), x, self.width())

	def mousePressEvent(self, e):
		v = self.pixelPosToRangeValue(e.x())
		if abs(v - self._lower) < abs(v - self._upper):
			self._drag = 'lower'
		else:
			self._drag = 'upper'
		super().mousePressEvent(e)

	def mouseMoveEvent(self, e):
		v = self.pixelPosToRangeValue(e.x())
		if getattr(self, '_drag', None) == 'lower':
			self.setLowerValue(v)
		elif getattr(self, '_drag', None) == 'upper':
			self.setUpperValue(v)
		super().mouseMoveEvent(e)

	def mouseReleaseEvent(self, e):
		self._drag = None
		super().mouseReleaseEvent(e)

	def paintEvent(self, ev):
		# simple painter: groove + two handles + filled span
		p = QtGui.QPainter(self)
		rect = self.rect().adjusted(6, self.height()//2 - 3, -6, -(self.height()//2 - 3))
		groove = QtCore.QRect(rect)
		p.fillRect(groove, QtGui.QColor("#2d2d2f"))

		def x_for(v):
			t = (v - self.minimum()) / max(1, (self.maximum() - self.minimum()))
			return int(rect.left() + t * rect.width())

		x1 = x_for(self._lower); x2 = x_for(self._upper)
		p.fillRect(QtCore.QRect(min(x1,x2), rect.top(), abs(x2-x1), rect.height()), QtGui.QColor(80,160,255,80))

		for x in (x1, x2):
			hrect = QtCore.QRect(x-6, rect.top()-6, 12, rect.height()+12)
			p.setBrush(QtGui.QBrush(QtGui.QColor(AXIS_COLORS['y'])))
			p.setPen(QtGui.QColor(255,255,255,40))
			p.drawRoundedRect(hrect, 3, 3)

		p.end()

class AxisGlyph(QtWidgets.QFrame):
	def __init__(self, title:str, parent=None):
		super().__init__(parent)
		self.setFixedHeight(64)
		self.setMinimumWidth(120)
		self._R_cam = np.eye(3)  # axes expressed in *camera* space
		self._title = title
		self.setStyleSheet("QFrame { background: #151518; border: 1px solid #303036; border-radius: 6px; }")

	def setRotationCam(self, R_cam: np.ndarray):
		if R_cam is None: return
		self._R_cam = np.array(R_cam, float)
		self.update()

	def paintEvent(self, ev):
		p = QtGui.QPainter(self)
		p.setRenderHint(QtGui.QPainter.Antialiasing, True)
		w, h = self.width(), self.height()
		cx, cy = w*0.5, h*0.62
		scale = min(w, h) * 0.36  # slightly longer

		p.setPen(QtGui.QColor("#bbbbbb"))
		p.drawText(QtCore.QRectF(0, 2, w, 18), QtCore.Qt.AlignHCenter, self._title)

		# Project using camera X (→) and Z (↑) so Z is clearly visible
		axes = [("X", AXIS_COLORS["x"], self._R_cam[:,0]),
				("Y", AXIS_COLORS["y"], self._R_cam[:,1]),
				("Z", AXIS_COLORS["z"], self._R_cam[:,2])]
		for _, color, vec in axes:
			vx = float(vec[0])         # camera X
			vz = float(vec[2])         # camera Z
			sx = cx + scale * vx
			sy = cy - scale * vz       # up is -Z on screen

			pen = QtGui.QPen(color); pen.setWidth(2)   # thinner
			p.setPen(pen)
			p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(sx, sy))
			# arrow head
			dx, dy = sx-cx, sy-cy
			L = math.hypot(dx, dy) + 1e-9
			ux, uy = dx/L, dy/L
			px, py = -uy, ux
			head_len = 12.0; head_w = 5.0
			tip = QtCore.QPointF(sx, sy)
			left = QtCore.QPointF(sx - ux*head_len + px*head_w, sy - uy*head_len + py*head_w)
			right= QtCore.QPointF(sx - ux*head_len - px*head_w, sy - uy*head_len - py*head_w)
			p.setBrush(color)
			p.drawPolygon(QtGui.QPolygonF([tip, left, right]))
		p.end()

class AspectImageView(QtWidgets.QWidget):
	"""
	A view that draws a QImage/QPixmap scaled to the widget's rect
	with KeepAspectRatio. It does NOT let the image dictate window size.
	"""
	def __init__(self, parent=None):
		super().__init__(parent)
		self._pm = None
		# Important: don't let the pixmap's natural size affect the layout
		self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		self.setMinimumSize(1, 1)
		self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, False)  # keep background via style
		self.setAutoFillBackground(True)

	def sizeHint(self):
		# Modest default hint; won't bloat layout
		return QtCore.QSize(320, 240)

	def minimumSizeHint(self):
		return QtCore.QSize(1, 1)

	@QtCore.pyqtSlot(QtGui.QImage)
	def setImage(self, img: QtGui.QImage):
		# Store as QPixmap once; paintEvent handles scaling
		if img is None or img.isNull():
			self._pm = None
		else:
			self._pm = QtGui.QPixmap.fromImage(img)
		self.update()

	@QtCore.pyqtSlot(QtGui.QPixmap)
	def setPixmap(self, pm: QtGui.QPixmap):
		self._pm = pm
		self.update()

	def clear(self):
		self._pm = None
		self.update()

	def paintEvent(self, ev: QtGui.QPaintEvent):
		p = QtGui.QPainter(self)
		try:
			# Fill background using style (so you don't see “creep” artifacts)
			opt = QtWidgets.QStyleOption()
			opt.initFrom(self)
			self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, p, self)

			if not self._pm or self._pm.isNull():
				return

			# Compute target rect with aspect ratio preserved
			wnd = self.rect()
			pm_size = self._pm.size()
			target = pm_size.scaled(wnd.size(), QtCore.Qt.KeepAspectRatio)
			x = wnd.x() + (wnd.width()  - target.width())  // 2
			y = wnd.y() + (wnd.height() - target.height()) // 2
			p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
			p.drawPixmap(QtCore.QRect(x, y, target.width(), target.height()), self._pm)
		finally:
			p.end()

class BackgroundLogo(QtWidgets.QWidget):
	def __init__(self, path, opacity=0.08, parent=None):
		super().__init__(parent)
		self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
		self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
		self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
		self.src = QtGui.QPixmap(path) if path and os.path.exists(path) else None
		self.pix = None
		self.opacity = float(opacity)

	def set_scaled_width(self, target_w: int):
		"""Scale once to target width; keeps aspect ratio and fixes widget size."""
		if not self.src or target_w <= 0:
			return
		# preserve aspect ratio
		self.pix = self.src.scaledToWidth(int(target_w), QtCore.Qt.SmoothTransformation)
		self.setFixedSize(self.pix.size())

	def paintEvent(self, ev):
		if not self.pix:
			return
		p = QtGui.QPainter(self)
		p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
		p.setOpacity(self.opacity)
		p.drawPixmap(0, 0, self.pix)

def slider_style_for(color: QtGui.QColor) -> str:
	c = color.name()
	# neutral groove, colored square handle
	return f"""
	QSlider::groove:horizontal {{
		height: 6px; background: #2d2d2f; border-radius: 3px;
	}}
	QSlider::sub-page:horizontal {{ background: transparent; }}
	QSlider::add-page:horizontal {{ background: transparent; }}
	QSlider::handle:horizontal {{
		background: {c};
		width: 14px; height: 14px; margin: -5px 0; border-radius: 3px;
		border: 1px solid rgba(255,255,255,0.15);
	}}
	QSlider::tick-mark:horizontal {{
		background: #6a6a6a;
	}}
	"""

def make_full_slider_row(label_text: str, minv: float, maxv: float, step: float,
						init: float, axis_key: str, suffix: str):
	container = QtWidgets.QWidget()
	v = QtWidgets.QVBoxLayout(container); v.setContentsMargins(0,0,0,0); v.setSpacing(2)

	row = QtWidgets.QWidget()
	h = QtWidgets.QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.setSpacing(8)

	lbl = QtWidgets.QLabel(label_text); lbl.setMinimumWidth(60)

	sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
	sld.setStyleSheet(slider_style_for(AXIS_COLORS[axis_key]))
	sld.setCursor(QtCore.Qt.PointingHandCursor)
	sld.setTickPosition(QtWidgets.QSlider.TicksBelow)
	# tick interval ~25 steps
	total_steps = max(1, int(round((maxv - minv) / step)))
	sld.setMinimum(0); sld.setMaximum(total_steps)
	sld.setSingleStep(1); sld.setPageStep(max(2, total_steps // 20))
	sld.setTickInterval(max(1, total_steps // 12))

	spin = QtWidgets.QDoubleSpinBox()
	spin.setRange(minv, maxv); spin.setSingleStep(step); spin.setDecimals(3); spin.setSuffix(suffix)
	spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons); spin.setFixedWidth(60)

	def idx_to_value(i: int) -> float:
		return float(minv + i * step)
	def value_to_idx(v: float) -> int:
		return int(round((v - minv) / step))
	def set_spin_display(v: float):
		spin.setValue(v)

	# init
	sld.blockSignals(True); sld.setValue(value_to_idx(init)); sld.blockSignals(False)
	spin.blockSignals(True); spin.setValue(init); spin.blockSignals(False)

	h.addWidget(lbl)
	h.addWidget(sld, 1)
	h.addWidget(spin)

	# ticks labels row under the slider
	ticks = QtWidgets.QWidget()
	th = QtWidgets.QHBoxLayout(ticks); th.setContentsMargins(96, 0, 90, 0); th.setSpacing(0)  # align under slider (accounts for label+spin widths)
	lb_min = QtWidgets.QLabel(f"{minv:.3f}{suffix}")
	lb_mid = QtWidgets.QLabel(f"{(0.5*(minv+maxv)):.3f}{suffix}")
	lb_max = QtWidgets.QLabel(f"{maxv:.3f}{suffix}")
	lb_min.setStyleSheet("color:#aaaaaa;"); lb_mid.setStyleSheet("color:#aaaaaa;"); lb_max.setStyleSheet("color:#aaaaaa;")
	th.addWidget(lb_min, 0, QtCore.Qt.AlignLeft)
	th.addWidget(lb_mid, 1, QtCore.Qt.AlignHCenter)
	th.addWidget(lb_max, 0, QtCore.Qt.AlignRight)

	v.addWidget(row)
	v.addWidget(ticks)

	return container, sld, spin, idx_to_value, value_to_idx, set_spin_display

def _backing_spin(minv, maxv, step, init):
	b = QtWidgets.QDoubleSpinBox()
	b.setRange(minv, maxv); b.setSingleStep(step); b.setDecimals(6); b.setValue(init)
	b.hide()
	return b

def _link_span_to_half(span_box, half_spin):
	def f(v):
		half_spin.blockSignals(True); half_spin.setValue(v*0.5); half_spin.blockSignals(False)
	span_box.valueChanged.connect(f)

def _link_half_to_span(half_spin, span_box):
	def f(v):
		span_box.blockSignals(True); span_box.setValue(v); span_box.blockSignals(False)
	half_spin.valueChanged.connect(f)

def _span_box(label, init):
	wrap = QtWidgets.QWidget(); hl = QtWidgets.QHBoxLayout(wrap); hl.setContentsMargins(0,0,0,0); hl.setSpacing(6)
	lab = QtWidgets.QLabel(label)
	spn = QtWidgets.QDoubleSpinBox(); spn.setRange(0.05, 10.0); spn.setDecimals(3); spn.setSuffix(" m")
	spn.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons); spn.setFixedWidth(60)
	spn.setValue(init)
	hl.addWidget(lab); hl.addWidget(spn)
	return wrap, spn

def _colored_span(label, color_key, rng, init):
	wrap = QtWidgets.QWidget()
	hl = QtWidgets.QHBoxLayout(wrap); hl.setContentsMargins(0,0,0,0); hl.setSpacing(6)
	lab = QtWidgets.QLabel(label); lab.setStyleSheet(f"color:{AXIS_COLORS[color_key].name()}")
	spn = QtWidgets.QDoubleSpinBox(); spn.setRange(*rng); spn.setDecimals(3); spn.setValue(init); spn.setSuffix(" m")
	spn.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons); spn.setFixedWidth(60)
	hl.addWidget(lab); hl.addWidget(spn)
	return wrap, spn

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

def make_card(*items):
	card = QtWidgets.QFrame()
	card.setObjectName("Card")
	card.setProperty("class", "Card")

	layout = QtWidgets.QVBoxLayout(card)
	layout.setContentsMargins(12, 12, 12, 12)
	layout.setSpacing(8)

	for it in items:
		if it is None:
			continue
		if isinstance(it, QtWidgets.QLayout):
			layout.addLayout(it)              # accept layouts directly
		elif isinstance(it, QtWidgets.QWidget):
			layout.addWidget(it)
		else:
			# If someone passes a plain string or unknown object, wrap in a QLabel
			layout.addWidget(QtWidgets.QLabel(str(it)))

	return card
	
def qimage_to_bgr_np(qimg: QtGui.QImage) -> np.ndarray:
	fmt = qimg.format()
	qimg = qimg.convertToFormat(QtGui.QImage.Format_RGBA8888)  # unify
	w, h = qimg.width(), qimg.height()
	ptr = qimg.bits()
	ptr.setsize(qimg.byteCount())
	arr = np.frombuffer(ptr, np.uint8).reshape(h, w, 4)  # RGBA
	bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
	return bgr

def bgr_np_to_qimage(img) -> QtGui.QImage:
	if img is None:
		return QtGui.QImage()

	if img.dtype != np.uint8:
		img = np.clip(img, 0, 255).astype(np.uint8)
	img = np.ascontiguousarray(img)

	if img.ndim == 2:
		h, w = img.shape
		bytes_per_line = img.strides[0]
		# QImage expects Grayscale8 for 1-channel
		qimg = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
		return qimg.copy()

	if img.ndim == 3:
		h, w, c = img.shape
		if c == 3:
			# BGR -> RGB
			rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			rgb = np.ascontiguousarray(rgb)
			bytes_per_line = rgb.strides[0]  # use actual stride, not 3*w
			qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
			return qimg.copy()
		elif c == 4:
			# BGRA -> RGBA
			rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
			rgba = np.ascontiguousarray(rgba)
			bytes_per_line = rgba.strides[0]
			qimg = QtGui.QImage(rgba.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)
			return qimg.copy()

	bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	bgr = np.ascontiguousarray(bgr)
	h, w = bgr.shape[:2]
	return QtGui.QImage(bgr.data, w, h, bgr.strides[0], QtGui.QImage.Format_RGB888).copy()

def _apply_rotation(k, img):
	if img is None: 
		return None
	if   k % 4 == 0: out = img
	elif k % 4 == 1: out = np.rot90(img, 3)   # 90° CW
	elif k % 4 == 2: out = np.rot90(img, 2)
	else:            out = np.rot90(img, 1)   # 270° CW
	return np.ascontiguousarray(out)          # OpenCV-safe

def _ensure_bgr_drawable(img):
	if img is None: return None
	if img.ndim == 2:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	if img.dtype != np.uint8:
		img = np.clip(img, 0, 255).astype(np.uint8)
	return np.ascontiguousarray(img)

def _hstack_same_height(left, right):
	"""Return left|right with matching heights (resize right to left’s height)."""
	if left is None:  return right
	if right is None: return left
	left  = _ensure_bgr_drawable(left)
	right = _ensure_bgr_drawable(right)
	hL, wL = left.shape[:2]
	hR, wR = right.shape[:2]
	if hL != hR:
		scale = float(hL) / float(hR)
		right = cv2.resize(right, (int(round(wR*scale)), hL), interpolation=cv2.INTER_LINEAR)
	return np.hstack([left, right])

def _apply_overlays(img_bgr, intrinsics, po, draw_plane, draw_roi, roi_dict):
	if img_bgr is None: 
		return None
	if po and (draw_plane or draw_roi) and (intrinsics is not None):
		return overlay_plane_and_roi_on_bgr_po(
			img_bgr, intrinsics, po, roi_dict, draw_plane, draw_roi
		)
	return img_bgr

def _compute_cloud(depth_frame, cloud, need_roi):
	if cloud is not None or not need_roi:
		return cloud
	try:
		pc_local = rs.pointcloud()
		points = pc_local.calculate(depth_frame)
		v = np.asanyarray(points.get_vertices())
		cloud = np.asarray(v.tolist()).reshape(-1, 3)
		m = np.isfinite(cloud).all(axis=1) & (cloud[:, 2] > 0)
		return cloud[m]
	except Exception:
		return None

def _roi_colouring(depth_u16, intrinsics, cloud, po, need_roi, roi_dict, scale):
	im = None
	if depth_u16 is not None:
		if need_roi: im = _render_interest_region_from_cloud(depth_u16, intrinsics, cloud, po, scale, roi_dict)
		else: im = colourise_depth(depth_u16, scale)
	return im

def _map_px_for_display(k, x, y, w, h):
	if k == 0:
		return x, y
	elif k == 1:  # 90° CW
		return (h - 1 - y, x)
	elif k == 2:  # 180°
		return (w - 1 - x, h - 1 - y)
	else:        # k == 3, 270° CW
		return (y, w - 1 - x)
	
def _valid_frame(f):
	return (f is not None) and getattr(f, "is_frame", lambda: False)()

def _as_np(f):
	return np.asanyarray(f.get_data()) if _valid_frame(f) else None
