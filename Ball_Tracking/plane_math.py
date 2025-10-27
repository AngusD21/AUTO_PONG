import numpy as np
from dataclasses import dataclass
import cv2

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

def _corners_world(cx, cy, cz, ex, ey, ez, yaw, pitch, roll):
	# ROI center/extends/rotation from your existing controls
	c = np.array([cx, cy, cz], float)
	e = np.array([ex, ey, ez], float)
	Rloc = _R_yxz(yaw, pitch, roll)
	# 8 corners in local coords
	s = np.array([[-1,-1,-1],[+1,-1,-1],[-1,+1,-1],[+1,+1,-1],
				[-1,-1,+1],[+1,-1,+1],[-1,+1,+1],[+1,+1,+1]], float) * e
	return c[None,:] + s @ Rloc.T  # (8,3) world

def _project_uvn(P, _pl_R_base, _pl_yaw, _pl_pitch, _pl_roll, flip_n, _pl_off_local, _pl_p0_base):
	"""Project Nx3 world points into (u,n,v)."""
	R = _plane_R_from_base_and_loc(_pl_R_base, _pl_yaw, _pl_pitch, _pl_roll, flip_n)  # [u n v]
	p0 = _plane_p0_from_base_and_loc(R, _pl_off_local, _pl_p0_base)
	D = P - p0[None,:]
	uvn = D @ R  # columns are [u, n, v]
	return uvn  # (N,3)

def _plane_p0_from_base_and_loc(R, _pl_off_local, _pl_p0_base):
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

def _roi_corners_world(p0, u, v, n, w, l, y0, y1, mirror_x, mirror_z, ex, ez):

	hx, hz = 0.5*w, 0.5*l

	# along u
	if mirror_x:
		hx_neg = hx + abs(ex); hx_pos = hx + abs(ex)
	else:
		hx_neg = hx + (abs(ex) if ex < 0 else 0.0)
		hx_pos = hx + (abs(ex) if ex > 0 else 0.0)

	# along v
	if mirror_z:
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
	

def project_points_px(pts3d: np.ndarray, intr) -> np.ndarray:

	P = np.asarray(pts3d, float).reshape(-1, 3)

	# --- Parse intrinsics robustly ---
	fx = fy = cx = cy = None
	# RealSense intrinsics object?
	if hasattr(intr, "fx") and hasattr(intr, "fy"):
		fx = float(intr.fx); fy = float(intr.fy)
		# RealSense uses ppx/ppy (principal point)
		cx = float(getattr(intr, "ppx", getattr(intr, "cx", 0.0)))
		cy = float(getattr(intr, "ppy", getattr(intr, "cy", 0.0)))
	# dict-like?
	elif isinstance(intr, dict):
		fx = float(intr.get("fx", intr.get("Fx", 0.0)))
		fy = float(intr.get("fy", intr.get("Fy", 0.0)))
		cx = float(intr.get("ppx", intr.get("cx", 0.0)))
		cy = float(intr.get("ppy", intr.get("cy", 0.0)))
	# 3x3 K matrix?
	else:
		K = np.asarray(intr, float)
		if K.shape == (3, 3):
			fx = float(K[0, 0]); fy = float(K[1, 1])
			cx = float(K[0, 2]); cy = float(K[1, 2])
		else:
			raise TypeError("Unsupported intrinsics type for project_points_px")

	# --- Project ---
	X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
	eps = 1e-9
	invZ = 1.0 / np.where(np.abs(Z) < eps, np.sign(Z) * eps, Z)
	x = X * fx * invZ + cx
	y = Y * fy * invZ + cy

	return np.column_stack([np.rint(x).astype(np.int32),
							np.rint(y).astype(np.int32)])



def project_points_px_masked(pts3d: np.ndarray, intr, z_eps: float = 1e-6):
	"""
	Project Nx3 -> (px Nx2 int32, valid Nx bool).
	Marks points with Z<=z_eps invalid so callers can drop edges that cross behind camera.
	Supports RealSense intrinsics obj, dict, or 3x3 K.
	"""
	P = np.asarray(pts3d, float).reshape(-1, 3)
	# parse intrinsics (same logic you already have)
	fx = fy = cx = cy = None
	if hasattr(intr, "fx"):
		fx, fy = float(intr.fx), float(intr.fy)
		cx = float(getattr(intr, "ppx", getattr(intr, "cx", 0.0)))
		cy = float(getattr(intr, "ppy", getattr(intr, "cy", 0.0)))
	elif isinstance(intr, dict):
		fx, fy = float(intr.get("fx", 0.0)), float(intr.get("fy", 0.0))
		cx = float(intr.get("ppx", intr.get("cx", 0.0)))
		cy = float(intr.get("ppy", intr.get("cy", 0.0)))
	else:
		K = np.asarray(intr, float)
		fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])

	X, Y, Z = P[:,0], P[:,1], P[:,2]
	valid = Z > z_eps
	# avoid dividing invalid rows
	x = np.empty_like(X); y = np.empty_like(Y)
	x[:] = np.nan; y[:] = np.nan
	x[valid] = X[valid] * fx / Z[valid] + cx
	y[valid] = Y[valid] * fy / Z[valid] + cy
	px = np.column_stack([np.rint(x), np.rint(y)]).astype(np.int32)
	return px, valid


def plane_corners_world_from_overlay(p0, u, v, width_m, length_m) -> np.ndarray:
	"""Return 4x3 plane corners (c00,c10,c11,c01)."""
	p0 = np.asarray(p0, float); u = np.asarray(u, float); v = np.asarray(v, float)
	hx, hz = 0.5*float(width_m), 0.5*float(length_m)
	return np.vstack([
		p0 + (-hx)*u + (-hz)*v,  # c00
		p0 + ( +hx)*u + (-hz)*v, # c10
		p0 + ( +hx)*u + ( +hz)*v,# c11
		p0 + (-hx)*u + ( +hz)*v, # c01
	])


def roi_corners_world_from_overlay(
		p0, u, v, n,
		width_m, length_m,
		y_min, y_max,
		mirror_x: bool, mirror_z: bool,
		x_extend: float, z_extend: float
	) -> np.ndarray:
	"""
	Build the extended rectangle on the plane and extrude along n to [y_min, y_max].
	Returns (8,3) bottom(4) then top(4).
	"""
	p0 = np.asarray(p0, float); u = np.asarray(u, float); v = np.asarray(v, float); n = np.asarray(n, float)
	u = u / (np.linalg.norm(u)+1e-12)
	v = v / (np.linalg.norm(v)+1e-12)

	hx, hz = 0.5*float(width_m), 0.5*float(length_m)
	ex, ez = float(x_extend), float(z_extend)

	if mirror_x:
		hx_min = hx + abs(ex); hx_max = hx + abs(ex)
		neg_push_x = pos_push_x = 0.0
	else:
		hx_min = hx; hx_max = hx
		neg_push_x = abs(ex) if ex < 0 else 0.0
		pos_push_x = abs(ex) if ex > 0 else 0.0

	if mirror_z:
		hz_min = hz + abs(ez); hz_max = hz + abs(ez)
		neg_push_z = pos_push_z = 0.0
	else:
		hz_min = hz; hz_max = hz
		neg_push_z = abs(ez) if ez < 0 else 0.0
		pos_push_z = abs(ez) if ez > 0 else 0.0

	c00 = p0 + (-(hx_min + neg_push_x))*u + (-(hz_min + neg_push_z))*v
	c10 = p0 + ( +(hx_max + pos_push_x))*u + (-(hz_min + neg_push_z))*v
	c11 = p0 + ( +(hx_max + pos_push_x))*u + ( +(hz_max + pos_push_z))*v
	c01 = p0 + (-(hx_min + neg_push_x))*u + ( +(hz_max + pos_push_z))*v
	base = np.stack([c00, c10, c11, c01], axis=0)

	lo = base + n[None, :]*float(y_min)
	hi = base + n[None, :]*float(y_max)
	return np.vstack([lo, hi])

def plane_poly_px_from_overlay(plane_overlay, intr) -> np.ndarray:
	"""
	plane_overlay keys: p0,u,v,width_m,length_m
	returns (4,2) int32 polygon in pixel coords (c00..c01)
	"""
	p0 = plane_overlay["p0"]; u = plane_overlay["u"]; v = plane_overlay["v"]
	w  = plane_overlay.get("width_m", 0.7)
	l  = plane_overlay.get("length_m", 0.7)
	corners = plane_corners_world_from_overlay(p0, u, v, w, l)  # (4,3)
	return project_points_px(corners, intr)

def roi_box_edges_px_from_overlay(*, p0, u, v, n, width_m, length_m, intr, y_min, y_max, x_extend, z_extend, mirror_x, mirror_z):

	roi8 = roi_corners_world_from_overlay(
		p0, u, v, n, width_m, length_m,
		y_min, y_max, mirror_x, mirror_z, x_extend, z_extend
	)
	px = project_points_px(roi8, intr)  # (8,2)
	edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
	return px, edges

def roi_mask_points_world(pts, p0, u, v, n, width_m, length_m, y_min, y_max, x_extend, z_extend, mirror_x, mirror_z):
	P = np.asarray(pts, float)
	p0 = np.asarray(p0, float); u = np.asarray(u, float); v = np.asarray(v, float); n = np.asarray(n, float)
	# plane frame
	R = np.column_stack([
		u/ (np.linalg.norm(u)+1e-12),
		n/ (np.linalg.norm(n)+1e-12),
		v/ (np.linalg.norm(v)+1e-12)
	])
	D = P - p0[None,:]
	uvn = D @ R  # columns: [U, N, V]
	U, N, V = uvn[:,0], uvn[:,1], uvn[:,2]

	hx, hz = 0.5*float(width_m), 0.5*float(length_m)
	ex, ez = float(x_extend), float(z_extend)

	if mirror_x:
		u_min, u_max = -hx - abs(ex), +hx + abs(ex)
	else:
		u_min, u_max = -hx + min(ex, 0.0), +hx + max(ex, 0.0)

	if mirror_z:
		v_min, v_max = -hz - abs(ez), +hz + abs(ez)
	else:
		v_min, v_max = -hz + min(ez, 0.0), +hz + max(ez, 0.0)

	y0 = min(float(y_min), float(y_max))
	y1 = max(float(y_min), float(y_max))

	inside = (U >= u_min) & (U <= u_max) & (V >= v_min) & (V <= v_max) & (N >= y0) & (N <= y1)
	return inside

def plane_poly_px_from_overlay(*, p0, u, v, width_m, length_m, intr):
	corners = plane_corners_world_from_overlay(p0, u, v, width_m, length_m)
	return project_points_px(corners, intr)

_INSIDE, _LEFT, _RIGHT, _BOTTOM, _TOP = 0, 1, 2, 4, 8
def _outcode(x, y, w, h):
	code = _INSIDE
	if x < 0:      code |= _LEFT
	elif x >= w:   code |= _RIGHT
	if y < 0:      code |= _TOP
	elif y >= h:   code |= _BOTTOM
	return code

def clip_segment_to_image(p0, p1, w, h):
	"""Clip line segment p0->p1 to [0..w-1]x[0..h-1]. Returns (q0,q1) or None."""
	x0, y0 = float(p0[0]), float(p0[1])
	x1, y1 = float(p1[0]), float(p1[1])
	c0, c1 = _outcode(x0,y0,w,h), _outcode(x1,y1,w,h)
	while True:
		if not (c0 | c1):
			return (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1)))
		if c0 & c1:
			return None
		c_out = c0 or c1
		if c_out & _TOP:
			x = x0 + (x1-x0) * (0 - y0) / (y1 - y0 + 1e-12); y = 0
		elif c_out & _BOTTOM:
			y = h-1; x = x0 + (x1-x0) * (y - y0) / (y1 - y0 + 1e-12)
		elif c_out & _RIGHT:
			x = w-1; y = y0 + (y1-y0) * (x - x0) / (x1 - x0 + 1e-12)
		else: # _LEFT
			x = 0;   y = y0 + (y1-y0) * (x - x0) / (x1 - x0 + 1e-12)
		if c_out == c0:
			x0, y0 = x, y; c0 = _outcode(x0,y0,w,h)
		else:
			x1, y1 = x, y; c1 = _outcode(x1,y1,w,h)

def clip_poly_points_to_image(poly_xy, w, h):
	"""Clamp points for fills (safe for cv2.fillPoly)."""
	p = np.asarray(poly_xy, np.int32).copy()
	p[:,0] = np.clip(p[:,0], 0, w-1)
	p[:,1] = np.clip(p[:,1], 0, h-1)
	return p

# ================= ROI/Plane projected helpers =================
def roi_footprint_poly_px_from_overlay(
	*, p0, u, v, n, width_m, length_m,
	x_extend, z_extend, mirror_x, mirror_z,
	intr):
	"""
	Project the ROI's base rectangle (with mirror/extends) to pixels as a 4x2 int32 polygon.
	"""
	# Build extended base on the plane (bottom ring of roi_corners_world_from_overlay)
	base8 = roi_corners_world_from_overlay(
		p0, u, v, n, width_m, length_m,
		y_min=0.0, y_max=0.0,  # base
		mirror_x=mirror_x, mirror_z=mirror_z,
		x_extend=x_extend, z_extend=z_extend
	)
	base4 = base8[:4]  # bottom loop
	poly = project_points_px(base4, intr)
	return poly.astype(np.int32)


def roi_box_edges_px_from_overlay_clipped(
	*, p0, u, v, n, width_m, length_m, intr,
	y_min, y_max, x_extend, z_extend, mirror_x, mirror_z,
	image_shape
):
	# build 8 corners (bottom[0..3], top[4..7]) using your existing function
	roi8 = roi_corners_world_from_overlay(
		p0, u, v, n, width_m, length_m,
		y_min, y_max, mirror_x, mirror_z, x_extend, z_extend
	)
	px8, valid = project_points_px_masked(roi8, intr)  # << use masked projector
	h, w = image_shape[:2]
	edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
	segs = []
	for i, j in edges:
		if not (valid[i] and valid[j]):
			continue  # drop segments with an endpoint behind the camera
		seg = clip_segment_to_image(px8[i], px8[j], w, h)
		if seg is not None:
			segs.append(seg)
	return segs


def overlay_plane_and_roi_on_bgr_po(bgr, intr, po, roi_dict, show_plane, show_roi):
	
	y_min    = roi_dict["roi_y_min"]
	y_max    = roi_dict["roi_y_max"]
	x_extend = roi_dict["roi_x_extend"]
	z_extend = roi_dict["roi_z_extend"]
	mirror_x = roi_dict["roi_mirror_x"]
	mirror_z = roi_dict["roi_mirror_z"]
	
	return overlay_plane_and_roi_on_bgr(bgr, intr, po["p0"], po["u"], po["v"], po["normal"], po["width_m"], po["length_m"], y_min, y_max, 
								x_extend, z_extend, mirror_x, mirror_z, 
								show_plane, show_roi)
	

def overlay_plane_and_roi_on_bgr(bgr, intr, p0, u, v, n, width_m, length_m, y_min, y_max, 
								x_extend, z_extend, mirror_x, mirror_z, 
								show_plane, show_roi, plane_fill_alpha = 0.25, 
								roi_color = (255, 255, 255), plane_edge_color = (255, 255, 255), edge_thickness = 2):
	out = bgr.copy()
	h, w = out.shape[:2]
	
	if show_plane:
		# 4-plane corners in pixels (clip for fill)
		plane4 = plane_corners_world_from_overlay(p0, u, v, width_m, length_m)
		poly_px = project_points_px(plane4, intr).astype(np.int32)
		poly_px = clip_poly_points_to_image(poly_px, w, h)

		# fill (alpha) + outline
		overlay = out.copy()
		cv2.fillPoly(overlay, [poly_px], color=(plane_edge_color[2], plane_edge_color[1], plane_edge_color[0]))
		cv2.addWeighted(overlay, plane_fill_alpha, out, 1.0 - plane_fill_alpha, 0, out)
		cv2.polylines(out, [poly_px], isClosed=True,
		              color=(plane_edge_color[2], plane_edge_color[1], plane_edge_color[0]),
		              thickness=edge_thickness, lineType=cv2.LINE_AA)

	if show_roi:
		segs = roi_box_edges_px_from_overlay_clipped(
			p0=p0, u=u, v=v, n=n,
			width_m=width_m, length_m=length_m,
			intr=intr,
			y_min=y_min, y_max=y_max,
			x_extend=x_extend, z_extend=z_extend,
			mirror_x=mirror_x, mirror_z=mirror_z,
			image_shape=out.shape
		)
		bgr_color = (roi_color[2], roi_color[1], roi_color[0])

		for seg in segs:
			if len(seg) == 2 and hasattr(seg[0], "__len__"):
				(x0, y0), (x1, y1) = seg
			else:
				x0, y0, x1, y1 = seg
			cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)),
					bgr_color, edge_thickness, cv2.LINE_AA)
	return out

def _render_interest_region_from_cloud(depth_u16, intr, xyz_full, po, depth_scale, roi_dict):

	# Basic guards
	if depth_u16 is None or intr is None or xyz_full is None or xyz_full.size == 0:
		return colourise_depth(depth_u16, depth_scale)

	if roi_dict:
		roi_y_min    = roi_dict["roi_y_min"]
		roi_y_max    = roi_dict["roi_y_max"]
		roi_x_extend = roi_dict["roi_x_extend"]
		roi_z_extend = roi_dict["roi_z_extend"]
		roi_mirror_x = roi_dict["roi_mirror_x"]
		roi_mirror_z = roi_dict["roi_mirror_z"]
	else:
		print("No ROI")
		return colourise_depth(depth_u16, depth_scale)
	
	# Background image
	# fast greyscale from depth (mm), then to BGR
	depth_mm = depth_u16.astype(np.float32) * (depth_scale * 1000.0)
	valid = depth_u16 > 0
	if valid.any():
		lo = float(np.percentile(depth_mm[valid], 2))
		hi = float(np.percentile(depth_mm[valid], 98))
		if hi <= lo:
			lo, hi = 400.0, 3000.0
	else:
		lo, hi = 400.0, 3000.0
	norm = np.zeros_like(depth_mm, np.float32)
	if valid.any():
		norm[valid] = np.clip((depth_mm[valid] - lo) / (hi - lo + 1e-6), 0, 1)
	grey_u8 = (norm * 255.0).astype(np.uint8)
	out = cv2.cvtColor(grey_u8, cv2.COLOR_GRAY2BGR)
	out[~valid] = (0, 0, 0)

	h, w = out.shape[:2]

	# Plane/ROI params
	if not po:
		return out
	p0 = np.asarray(po["p0"], float)
	u  = np.asarray(po["u"],  float)
	v  = np.asarray(po["v"],  float)
	n  = np.asarray(po["normal"], float)

	width_m  = float(po.get("width_m",  0.7))
	length_m = float(po.get("length_m", 0.7))

	# --- World-space ROI mask over full cloud ---
	inside_full = roi_mask_points_world(
		xyz_full, p0, u, v, n,
		width_m, length_m,
		roi_y_min, roi_y_max,
		roi_x_extend, roi_z_extend,
		roi_mirror_x, roi_mirror_z
	)
	if not inside_full.any():
		return out

	pts_in = xyz_full[inside_full]

	# --- Project those ROI points to depth pixels (skip behind camera) ---
	px, valid = project_points_px_masked(pts_in, intr)
	px = px[valid]
	if px.size == 0:
		return out

	# Build/clean a pixel mask
	xs = np.clip(px[:, 0], 0, w - 1)
	ys = np.clip(px[:, 1], 0, h - 1)
	mask = np.zeros((h, w), np.uint8)
	mask[ys, xs] = 255
	k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	mask = cv2.dilate(mask, k, iterations=1)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

	# --- Green overlay (alpha blend) ---
	overlay = out.copy()
	overlay[mask > 0] = (0, 255, 0)
	out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0.0)

	return out


def colourise_depth(depth_u16, depth_scale, clip_mm=(400, 3000), use_auto_percentiles=False):

	# Convert to millimetres
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