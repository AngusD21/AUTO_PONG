import threading
import time
import socket
import math
import queue
import sys
from dataclasses import dataclass
import os

# ---- Force Qt5Agg early (your env has PyQt5) ----
os.environ["MPLBACKEND"] = "Qt5Agg"
import matplotlib
matplotlib.use("Qt5Agg")
print("Matplotlib backend ->", matplotlib.get_backend())

# Matplotlib / Qt glue
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Qt
from PyQt5 import QtWidgets, QtCore

# ===============================
# Configuration
# ===============================

HOST = "127.0.0.1"
PORT = 8765

DT = 1.0 / 200.0
STATUS_HZ = 1.0

# Workspace (mm)
X_MAX = 800.0
Y_MAX = 800.0

# Visual geometry (mm)
RAIL_W = 20.0
CROSSBAR_H = 16.0
CARR_W = 60.0
CARR_H = 40.0
FRAME_MARGIN = 120.0
GEAR_RADIUS_MM = 22.0
IDLER_RADIUS_MM = 10.0

MAX_V_NORMAL = 300.0
MAX_A_NORMAL = 1500.0
MAX_A_FAST   = 3000.0
DECEL_A_SLOW = 250.0

IDLE_TIMEOUT_S = 1.0

CIRCLE_R = min(X_MAX, Y_MAX) * 0.3
CIRCLE_W = 2.0 * math.pi / 10.0  # one loop / 10s

DASH_SCALE = 0.15
DASH_STYLE = (0, (6, 6))

SNAP_POS = 0.05
SNAP_VEL = 0.20

MOTOR_RPM_MAX = 600.0

# ===============================
# Utility
# ===============================

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def close_enough(a, b, eps=1e-3):
    return abs(a - b) <= eps

def _saturate_bound(p, v, lo, hi):
    if p < lo:  return lo, 0.0 if v < 0 else v
    if p > hi:  return hi, 0.0 if v > 0 else v
    return p, v

def motor_vmax_mm_s():
    return (MOTOR_RPM_MAX * 2.0 * math.pi / 60.0) * GEAR_RADIUS_MM

# ===============================
# Modes
# ===============================

MODE_IDLE    = "I"
MODE_TARGET  = "T"
MODE_SMART   = "U"
MODE_CIRCLE  = "C"
MODE_SQUARE  = "S"
MODE_MOUSE   = "M"
MODE_STOP    = "X"
MODE_REHOME  = "R"

# ===============================
# Simulator Core
# ===============================

@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    mode: str = MODE_IDLE
    t0: float = time.time()
    thetaA: float = 0.0
    thetaB: float = 0.0

class GantrySim:
    def __init__(self):
        self.state = State(x=X_MAX/2.0, y=Y_MAX/2.0, mode=MODE_IDLE)
        self.target = [self.state.x, self.state.y]
        self.lock = threading.Lock()
        self.cmd_q = queue.Queue()
        self.last_cmd_time = time.time()
        self.last_status_time = 0.0
        self.circle_center = (X_MAX/2.0, Y_MAX/2.0)
        self.circle_phase = 0.0
        self.running = True

        self.square_w = 200.0
        self.square_h = 200.0
        self.square_i = 0
        self.square_corners = []

        self.mouse_inside = False
        self.mouse_xy = (self.state.x, self.state.y)

        self.smart_active = False
        self.smart_t0 = 0.0
        self.smart_T  = 0.0
        self.smart_p0 = (self.state.x, self.state.y)
        self.smart_v0 = (0.0, 0.0)
        self.smart_p1 = (self.state.x, self.state.y)
        self.smart_v1 = (0.0, 0.0)
        self.smart_path = ([], [])

        self.smart_T_tail = 0.0
        self.smart_T_total = 0.0
        self.smart_decel_path = ([], [])  # (xs_tail, ys_tail)
        self.smart_speed = 0.0            # |v1| at intercept
        self.smart_u = (0.0, 0.0)         # unit direction of v1

        self.belt_ofs = {
            'top_l':0.0, 'top_r':0.0, 'bot':0.0,
            'left_outer':0.0, 'right_outer':0.0,
            'l_vert_top':0.0, 'l_vert_bot':0.0,
            'r_vert_top':0.0, 'r_vert_bot':0.0,
        }

    # ---------------------------------
    # Command Handling
    # ---------------------------------

    def handle_command(self, raw_line, sock=None):
        line = raw_line.strip()
        if not line:
            return
        if line.startswith(">"):
            line = line[1:]
        parts = [p.strip() for p in line.split(",")]
        cmd = parts[0].upper() if parts else ""
        now = time.time()

        reply = "ACK"

        try:
            if cmd in (MODE_IDLE, MODE_TARGET, MODE_SMART, MODE_CIRCLE, MODE_SQUARE, MODE_MOUSE):
                if cmd == MODE_SMART:
                    x = float(parts[1]); y = float(parts[2])
                    vx = float(parts[3]); vy = float(parts[4])
                    self.plan_smart(x, y, vx, vy)
                else:
                    with self.lock:
                        if cmd == MODE_IDLE:
                            self.state.mode = MODE_IDLE
                        elif cmd == MODE_TARGET:
                            x = float(parts[1]); y = float(parts[2])
                            self.target[0] = clamp(x, 0.0, X_MAX)
                            self.target[1] = clamp(y, 0.0, Y_MAX)
                            self.state.mode = MODE_TARGET
                        elif cmd == MODE_CIRCLE:
                            x0 = float(parts[1]); y0 = float(parts[2])
                            self.circle_center = (clamp(x0, 0.0, X_MAX), clamp(y0, 0.0, Y_MAX))
                            self.state.mode = MODE_CIRCLE
                        elif cmd == MODE_SQUARE:
                            w = float(parts[1]); h = float(parts[2])
                            self.square_w = max(1.0, min(w, X_MAX))
                            self.square_h = max(1.0, min(h, Y_MAX))
                            cx, cy = X_MAX/2.0, Y_MAX/2.0
                            w2, h2 = self.square_w/2.0, self.square_h/2.0
                            self.square_corners = [
                                (cx - w2, cy + h2),
                                (cx + w2, cy + h2),
                                (cx + w2, cy - h2),
                                (cx - w2, cy - h2),
                            ]
                            self.square_i = 0
                            self.target[:] = self.square_corners[self.square_i]
                            self.state.mode = MODE_SQUARE
                        elif cmd == MODE_MOUSE:
                            self.state.mode = MODE_MOUSE
                self.last_cmd_time = now

            elif cmd == MODE_STOP or cmd == "X":
                with self.lock:
                    self.state.mode = MODE_STOP
                    self.target[0] = self.state.x
                    self.target[1] = self.state.y
                    self.state.vx = 0.0
                    self.state.vy = 0.0
                self.last_cmd_time = now

            elif cmd == MODE_REHOME:
                with self.lock:
                    self.state.mode = MODE_REHOME
                self.last_cmd_time = now

            elif cmd == "P":
                key = parts[1].upper()
                val = float(parts[2])
                global MAX_V_NORMAL, MAX_A_NORMAL, MAX_V_FAST, MAX_A_FAST, GEAR_RADIUS_MM, MOTOR_RPM_MAX
                if key == "MAX_V": MAX_V_NORMAL = val
                elif key == "MAX_A": MAX_A_NORMAL = val
                elif key == "MAX_V_FAST": MAX_V_FAST = val
                elif key == "MAX_A_FAST": MAX_A_FAST = val
                elif key == "GEAR_R": GEAR_RADIUS_MM = val
                elif key == "MOTOR_RPM": MOTOR_RPM_MAX = val
                else:
                    reply = f"ERR,unknown_param,{key}"
            elif cmd == "Q":
                pass
            else:
                reply = "ERR,unknown_cmd"
        except Exception as e:
            reply = f"ERR,parse,{e}"

        if sock:
            try:
                if cmd == "Q":
                    s = self.build_status()
                    sock.sendall((s + "\n").encode("utf-8"))
                else:
                    sock.sendall((reply + "\n").encode("utf-8"))
            except Exception:
                pass

    def build_status(self):
        with self.lock:
            return f"POS,{self.state.x:.3f},{self.state.y:.3f},{self.state.mode}"

    # ---------------------------------
    # Physics / Motion
    # ---------------------------------

    def _hermite_coeffs(self, p0, v0, p1, v1, T):
        T = max(T, 1e-6)
        def axis_coeff(p0, v0, p1, v1):
            def f(t):
                tau = clamp(t / T, 0.0, 1.0)
                h00 =  2*tau**3 - 3*tau**2 + 1
                h10 =      tau**3 - 2*tau**2 + tau
                h01 = -2*tau**3 + 3*tau**2
                h11 =      tau**3 -    tau**2
                return h00*p0 + h10*(T*v0) + h01*p1 + h11*(T*v1)
            return f
        fx = axis_coeff(p0[0], v0[0], p1[0], v1[0])
        fy = axis_coeff(p0[1], v0[1], p1[1], v1[1])
        return lambda t: (fx(t), fy(t))

    def _choose_T(self, p0, v0, p1, v1):
        dx = p1[0]-p0[0]; dy = p1[1]-p0[1]
        dist = math.hypot(dx, dy)
        vmax_axis = MAX_V_NORMAL
        amax_axis = MAX_A_NORMAL
        v_motor   = motor_vmax_mm_s()
        v_allow   = max(50.0, min(vmax_axis, 0.9*v_motor))
        t_pos     = dist / max(1.0, 0.6*v_allow)
        dvx = abs(v1[0]-v0[0]); dvy = abs(v1[1]-v0[1])
        t_vel = max(dvx, dvy) / max(1.0, 0.8*amax_axis)
        T = max(0.25, t_pos, t_vel)
        return min(T, 30.0)

    def plan_smart(self, x, y, vx, vy):
        with self.lock:
            p0 = (self.state.x, self.state.y)
            v0 = (self.state.vx, self.state.vy)
        p1 = (clamp(x, 0.0, X_MAX), clamp(y, 0.0, Y_MAX))
        v1 = (vx, vy)

        T  = self._choose_T(p0, v0, p1, v1)
        f  = self._hermite_coeffs(p0, v0, p1, v1, T)

        # --- SAMPLE THE HERMITE PATH (this was missing) ---
        N = 160
        xs, ys = [], []
        for i in range(N + 1):
            t = T * (i / N)
            px, py = f(t)
            xs.append(clamp(px, 0.0, X_MAX))
            ys.append(clamp(py, 0.0, Y_MAX))

        # --- Decel tail after intercept (pink) ---
        speed = math.hypot(v1[0], v1[1])
        if speed > 1e-6:
            ux, uy = v1[0] / speed, v1[1] / speed
            T_tail = speed / DECEL_A_SLOW
            Nt = 120
            xs_tail, ys_tail = [], []
            for i in range(Nt + 1):
                t = T_tail * (i / Nt)
                s = speed * t - 0.5 * DECEL_A_SLOW * t * t
                xs_tail.append(clamp(p1[0] + ux * s, 0.0, X_MAX))
                ys_tail.append(clamp(p1[1] + uy * s, 0.0, Y_MAX))
        else:
            T_tail = 0.0
            xs_tail, ys_tail = [], []
            ux, uy = 0.0, 0.0

        with self.lock:
            self.smart_active = True
            self.smart_t0 = time.time()
            self.smart_T  = T
            self.smart_p0, self.smart_v0 = p0, v0
            self.smart_p1, self.smart_v1 = p1, v1
            self.smart_path = (xs, ys)    

            self.smart_T_tail  = T_tail
            self.smart_T_total = T + T_tail
            self.smart_decel_path = (xs_tail, ys_tail)
            self.smart_speed = speed
            self.smart_u = (ux, uy)

            self.state.mode = MODE_SMART

    def _approach_1d(self, pos, vel, target, vmax, amax, dt):
        dx = target - pos
        stop_dist = (vel*vel) / (2.0*amax) if amax > 0 else 0.0

        if abs(dx) <= 1e-6 and abs(vel) < 1e-3:
            return target, 0.0

        if abs(dx) <= stop_dist:
            desired_acc = -math.copysign(amax, vel) if abs(vel) > 1e-6 else 0.0
        else:
            desired_acc = amax * math.copysign(1.0, dx)

        v_next = max(min(vel + desired_acc*dt, vmax), -vmax)
        pos_next = pos + v_next * dt

        crossed = (pos - target) * (pos_next - target) <= 0.0
        if crossed or (abs(pos_next - target) <= SNAP_POS and abs(v_next) <= SNAP_VEL):
            return target, 0.0

        return pos_next, v_next

    def predict_to_target(self, max_secs=15.0, dt_step=DT*2.0):
        with self.lock:
            mode = self.state.mode
            x  = float(self.state.x);  y  = float(self.state.y)
            vx = float(self.state.vx); vy = float(self.state.vy)
            tx = float(self.target[0]); ty = float(self.target[1])

        axis_v_cap = motor_vmax_mm_s()
        vmax = min(MAX_V_NORMAL, axis_v_cap)
        amax = MAX_A_FAST if mode == MODE_SMART else MAX_A_NORMAL

        if mode in (MODE_CIRCLE, MODE_SQUARE, MODE_REHOME, MODE_STOP, MODE_IDLE):
            return [], [], 0.0

        xs = [x]; ys = [y]; t = 0.0
        while t < max_secs:
            _, vx_star = self._approach_1d(x, vx, tx, vmax, amax, dt_step)
            _, vy_star = self._approach_1d(y, vy, ty, vmax, amax, dt_step)

            v_lin_max = motor_vmax_mm_s()
            if v_lin_max > 1e-6:
                vA_star =  vx_star - vy_star
                vB_star = -vy_star - vx_star
                denom = max(abs(vA_star), abs(vB_star), 1e-9)
                s = min(1.0, v_lin_max / denom)
            else:
                s = 0.0

            vx_new = vx_star * s
            vy_new = vy_star * s

            xn = x + vx_new * dt_step
            yn = y + vy_new * dt_step

            def _snap_axis(p_old, p_new, v_new, targ):
                crossed = (p_old - targ) * (p_new - targ) <= 0.0
                near    = (abs(p_new - targ) <= SNAP_POS) and (abs(v_new) <= SNAP_VEL)
                if crossed or near: return targ, 0.0
                return p_new, v_new

            x, vx = _snap_axis(x, xn, vx_new, tx)
            y, vy = _snap_axis(y, yn, vy_new, ty)

            xs.append(x); ys.append(y)
            t += dt_step

            done_x = abs(x - tx) <= SNAP_POS and abs(vx) <= SNAP_VEL
            done_y = abs(y - ty) <= SNAP_POS and abs(vy) <= SNAP_VEL
            if done_x and done_y:
                break

        return xs, ys, t

    def update(self, dt):
        with self.lock:
            st = self.state

            if st.mode == MODE_SMART and self.smart_active:
                t = time.time() - self.smart_t0
                T = self.smart_T
                if t <= T:
                    # Hermite segment
                    f = self._hermite_coeffs(self.smart_p0, self.smart_v0, self.smart_p1, self.smart_v1, T)
                    rx, ry = f(max(0.0, min(t, T)))
                    self.target[0] = clamp(rx, 0.0, X_MAX)
                    self.target[1] = clamp(ry, 0.0, Y_MAX)
                else:
                    # Tail decel segment
                    tau = t - T
                    if tau <= self.smart_T_tail and self.smart_speed > 1e-6:
                        ux, uy = self.smart_u
                        s = self.smart_speed*tau - 0.5*DECEL_A_SLOW*tau*tau
                        rx = self.smart_p1[0] + ux*s
                        ry = self.smart_p1[1] + uy*s
                        self.target[0] = clamp(rx, 0.0, X_MAX)
                        self.target[1] = clamp(ry, 0.0, Y_MAX)
                    else:
                        # Tail done -> hand off to tracker at final stop point
                        if self.smart_speed > 1e-6:
                            d_tail = (self.smart_speed*self.smart_speed)/(2.0*DECEL_A_SLOW)
                            ux, uy = self.smart_u
                            stop_x = clamp(self.smart_p1[0] + ux*d_tail, 0.0, X_MAX)
                            stop_y = clamp(self.smart_p1[1] + uy*d_tail, 0.0, Y_MAX)
                        else:
                            stop_x, stop_y = self.smart_p1
                        self.smart_active = False
                        st.mode = MODE_TARGET
                        self.target[0], self.target[1] = stop_x, stop_y

            if st.mode == MODE_IDLE:
                if (time.time() - self.last_cmd_time) > IDLE_TIMEOUT_S:
                    self.target[0] = X_MAX/2.0
                    self.target[1] = Y_MAX/2.0

            if st.mode == MODE_CIRCLE:
                cx, cy = self.circle_center
                self.circle_phase += CIRCLE_W * dt
                tx = cx + CIRCLE_R * math.cos(self.circle_phase)
                ty = cy + CIRCLE_R * math.sin(self.circle_phase)
                self.target[0] = clamp(tx, 0.0, X_MAX)
                self.target[1] = clamp(ty, 0.0, Y_MAX)

            if st.mode == MODE_SQUARE and self.square_corners:
                tx, ty = self.target
                if abs(st.x - tx) <= SNAP_POS and abs(st.y - ty) <= SNAP_POS and abs(st.vx) <= SNAP_VEL and abs(st.vy) <= SNAP_VEL:
                    self.square_i = (self.square_i + 1) % 4
                    nx, ny = self.square_corners[self.square_i]
                    self.target[0] = nx
                    self.target[1] = ny

            if st.mode == MODE_MOUSE:
                if self.mouse_inside:
                    mx, my = self.mouse_xy
                    self.target[0] = clamp(mx, 0.0, X_MAX)
                    self.target[1] = clamp(my, 0.0, Y_MAX)

            if st.mode == MODE_REHOME:
                self.target[0] = 0.0
                self.target[1] = 0.0
                if close_enough(st.x, 0.0, 0.01) and close_enough(st.y, 0.0, 0.01) and abs(st.vx) < 0.5 and abs(st.vy) < 0.5:
                    st.mode = MODE_IDLE
                    self.last_cmd_time = time.time()

            if st.mode == MODE_STOP:
                self.target[0] = st.x
                self.target[1] = st.y

            tx, ty = self.target

            axis_v_cap = motor_vmax_mm_s()
            vmax = min(MAX_V_NORMAL, axis_v_cap)
            amax = MAX_A_FAST if st.mode == MODE_SMART else MAX_A_NORMAL

            x0, y0 = st.x, st.y
            _, vx_star = self._approach_1d(st.x, st.vx, tx, vmax, amax, DT)
            _, vy_star = self._approach_1d(st.y, st.vy, ty, vmax, amax, DT)

            v_lin_max = motor_vmax_mm_s()
            if v_lin_max > 1e-6:
                vA_star =  vx_star - vy_star
                vB_star = -vy_star - vx_star
                denom = max(abs(vA_star), abs(vB_star), 1e-9)
                s = min(1.0, v_lin_max / denom)
            else:
                s = 0.0

            vx_new = vx_star * s
            vy_new = vy_star * s

            def _snap_axis(p_old, p_new, v_new, targ):
                crossed = (p_old - targ) * (p_new - targ) <= 0.0
                near    = (abs(p_new - targ) <= SNAP_POS) and (abs(v_new) <= SNAP_VEL)
                if crossed or near:
                    return targ, 0.0
                return p_new, v_new

            x1 = x0 + vx_new * DT
            y1 = y0 + vy_new * DT
            st.x, st.vx = _snap_axis(x0, x1, vx_new, tx)
            st.y, st.vy = _snap_axis(y0, y1, vy_new, ty)

            st.x, st.vx = _saturate_bound(st.x, st.vx, 0.0, X_MAX)
            st.y, st.vy = _saturate_bound(st.y, st.vy, 0.0, Y_MAX)

            settled_x = abs(self.target[0] - st.x) <= SNAP_POS and abs(st.vx) <= SNAP_VEL
            settled_y = abs(self.target[1] - st.y) <= SNAP_POS and abs(st.vy) <= SNAP_VEL
            vx_vis = 0.0 if settled_x else st.vx
            vy_vis = 0.0 if settled_y else st.vy

            v_top_l   = -vx_vis
            v_top_r   =  vx_vis
            v_l_t_inner = vx_vis - vy_vis
            v_l_b_inner = vx_vis - vy_vis
            v_r_t_inner = -vx_vis - vy_vis
            v_r_b_inner = -vx_vis - vy_vis
            v_l_outer  = -(vx_vis - vy_vis)
            v_r_outer  = -(-vx_vis - vy_vis)
            v_bot      = vx_vis

            k = DASH_SCALE
            for key, speed in [
                ('top_l', v_top_l), ('top_r', v_top_r), ('bot', v_bot),
                ('left_outer', v_l_outer), ('l_vert_top', v_l_t_inner), ('l_vert_bot', v_l_b_inner),
                ('right_outer', v_r_outer), ('r_vert_top', v_r_t_inner), ('r_vert_bot', v_r_b_inner),
            ]:
                self.belt_ofs[key] = (self.belt_ofs[key] + speed * DT * k)

            vA = vx_vis - vy_vis
            vB = -vy_vis - vx_vis

            if GEAR_RADIUS_MM > 1e-6:
                wA = vA / GEAR_RADIUS_MM
                wB = vB / GEAR_RADIUS_MM
            else:
                wA = wB = 0.0

            st.thetaA = (st.thetaA + wA * DT) % (2.0 * math.pi)
            st.thetaB = (st.thetaB + wB * DT) % (2.0 * math.pi)

        now = time.time()
        if now - self.last_status_time >= 1.0/STATUS_HZ:
            self.last_status_time = now
            print(self.build_status(), flush=True)

    # ---------------------------------
    # Networking
    # ---------------------------------

    def server_thread(self):
        while self.running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind((HOST, PORT))
                    s.listen(1)
                    print(f"Server listening on {HOST}:{PORT}")
                    s.settimeout(1.0)
                    while self.running:
                        try:
                            conn, addr = s.accept()
                        except socket.timeout:
                            continue
                        with conn:
                            conn.settimeout(0.5)
                            print(f"Client connected from {addr}")
                            try:
                                conn.sendall(b"ACK,hello\n")
                            except Exception:
                                pass
                            buf = b""
                            while self.running:
                                try:
                                    data = conn.recv(1024)
                                    if not data:
                                        break
                                    buf += data
                                    while b"\n" in buf:
                                        line, buf = buf.split(b"\n", 1)
                                        self.handle_command(line.decode("utf-8"), sock=conn)
                                except socket.timeout:
                                    continue
                                except ConnectionResetError:
                                    break
                                except Exception:
                                    break
                            print("Client disconnected")
            except Exception as e:
                print(f"Server error: {e}")
                time.sleep(1.0)

    def sim_loop_thread(self):
        prev = time.time()
        acc = 0.0
        while self.running:
            try:
                while True:
                    cmd = self.cmd_q.get_nowait()
                    self.handle_command(cmd)
            except queue.Empty:
                pass

            now = time.time()
            acc += now - prev
            prev = now
            while acc >= DT:
                self.update(DT)
                acc -= DT
            time.sleep(0.001)

# ===============================
# Embeddable Sim Canvas (Qt)
# ===============================

def build_sim_canvas(sim: GantrySim, get_smart_vx, get_smart_vy):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    fig = Figure(figsize=(10, 7))
    fig.patch.set_facecolor("#efe8d5")     # beige panel vibe
    canvas = FigureCanvas(fig)

    ax = fig.add_subplot(1,1,1)
    ax.set_title("Simulator", fontsize=12, fontweight="bold")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(-FRAME_MARGIN, X_MAX + FRAME_MARGIN)
    ax.set_ylim(-FRAME_MARGIN, Y_MAX + FRAME_MARGIN)
    ax.set_navigate(False)

    _pred_cache = {"tx": None, "ty": None, "ts": 0.0, "xs": [], "ys": [], "eta": "—"}

    ws_rect = Rectangle((0, 0), X_MAX, Y_MAX, facecolor=(0.92, 0.92, 0.92), edgecolor='none', zorder=1)
    ax.add_patch(ws_rect)

    rail_left  = Rectangle((-RAIL_W-CARR_W/2, -CARR_H/2), RAIL_W, Y_MAX+CARR_H, facecolor=(0.3, 0.3, 0.3), edgecolor='none', zorder=2)
    rail_right = Rectangle((X_MAX+CARR_W/2,  -CARR_H/2),  RAIL_W, Y_MAX+CARR_H, facecolor=(0.3, 0.3, 0.3), edgecolor='none', zorder=2)
    ax.add_patch(rail_left); ax.add_patch(rail_right)

    crossbar = Rectangle((-CARR_W/2, 0), X_MAX+CARR_W, CROSSBAR_H, facecolor=(0.3, 0.3, 0.3), edgecolor='none', zorder=3)
    ax.add_patch(crossbar)

    carriage = Rectangle((0, 0), CARR_W, CARR_H, facecolor=(0.2, 0.2, 0.2), edgecolor='white', linewidth=1.0, zorder=4)
    ax.add_patch(carriage)

    (center_dot,) = ax.plot([], [], marker='o', markersize=4, color='white', zorder=5)
    (target_dot,) = ax.plot([], [], marker='x', markersize=6, color='red', zorder=5)

    def motor_centers():
        return (
            (-RAIL_W/2.0 - CARR_W/2.0,          -CARR_H/2.0 - GEAR_RADIUS_MM*1.1),
            (X_MAX + RAIL_W/2.0 + CARR_W/2.0,   -CARR_H/2.0 - GEAR_RADIUS_MM*1.1)
        )
    motorA = Circle(motor_centers()[0], radius=GEAR_RADIUS_MM, facecolor='black', edgecolor='black', zorder=6)
    motorB = Circle(motor_centers()[1], radius=GEAR_RADIUS_MM, facecolor='black', edgecolor='black', zorder=6)
    ax.add_patch(motorA); ax.add_patch(motorB)
    motorA_dot, = ax.plot([], [], marker='o', markersize=4, color='white', zorder=7)
    motorB_dot, = ax.plot([], [], marker='o', markersize=4, color='white', zorder=7)

    # Idlers pinned to the ends/top
    idler_end_L = Circle((0, 0), radius=IDLER_RADIUS_MM*0.8, facecolor='aqua', edgecolor='none', zorder=6)
    idler_end_R = Circle((0, 0), radius=IDLER_RADIUS_MM*0.8, facecolor='aqua', edgecolor='none', zorder=6)
    ax.add_patch(idler_end_L); ax.add_patch(idler_end_R)

    idler_top_L = Circle((0, 0), radius=IDLER_RADIUS_MM*0.8, facecolor='aqua', edgecolor='none', zorder=6)
    idler_top_R = Circle((0, 0), radius=IDLER_RADIUS_MM*0.8, facecolor='aqua', edgecolor='none', zorder=6)
    idler_bot_L = Circle((0, 0), radius=IDLER_RADIUS_MM*0.8, facecolor='aqua', edgecolor='none', zorder=6)
    idler_bot_R = Circle((0, 0), radius=IDLER_RADIUS_MM*0.8, facecolor='aqua', edgecolor='none', zorder=6)
    for c in (idler_top_L, idler_top_R, idler_bot_L, idler_bot_R):
        ax.add_patch(c)

    attach_L = Circle((0.0, 0.0), radius=IDLER_RADIUS_MM*0.8, facecolor='black', edgecolor='none', zorder=6)
    attach_R = Circle((0.0, 0.0), radius=IDLER_RADIUS_MM*0.8, facecolor='black', edgecolor='none', zorder=6)
    ax.add_patch(attach_L); ax.add_patch(attach_R)

    belt_left  = Line2D([], [], linewidth=2.0, linestyle=DASH_STYLE)
    belt_right = Line2D([], [], linewidth=2.0, linestyle=DASH_STYLE)
    ax.add_line(belt_left); ax.add_line(belt_right)

    belt_l_vert_top = Line2D([], [], linewidth=2.0, linestyle=DASH_STYLE)
    belt_l_vert_bot = Line2D([], [], linewidth=2.0, linestyle=DASH_STYLE)
    belt_r_vert_top = Line2D([], [], linewidth=2.0, linestyle=DASH_STYLE)
    belt_r_vert_bot = Line2D([], [], linewidth=2.0, linestyle=DASH_STYLE)
    ax.add_line(belt_l_vert_top); ax.add_line(belt_l_vert_bot)
    ax.add_line(belt_r_vert_top); ax.add_line(belt_r_vert_bot)

    belt_top_l = Line2D([], [], linewidth=2.0, linestyle=DASH_STYLE)
    belt_top_r = Line2D([], [], linewidth=2.0, linestyle=DASH_STYLE)
    ax.add_line(belt_top_l); ax.add_line(belt_top_r)

    belt_bot = Line2D([], [], linewidth=2.0, linestyle=DASH_STYLE)
    ax.add_line(belt_bot)

    path_line, = ax.plot([], [], '-', color='red', linewidth=2.0, alpha=0.6, zorder=8)
    smart_path_line, = ax.plot([], [], '-', color='red', linewidth=2.2, alpha=0.6, zorder=8)

    # Deceleration tail (pink)
    smart_decel_line, = ax.plot([], [], '-', linewidth=2.2, alpha=0.6, color='#ff69b4', zorder=8)
    (intercept_dot,) = ax.plot([], [], marker='x', markersize=6, color='green', zorder=7)

    hud = ax.text(0.5, 0.94, "", transform=ax.transAxes, va='bottom', ha='center', fontsize=10, fontweight='bold', color='black')
    spd = ax.text(0.5, 0.02, "", transform=ax.transAxes, va='bottom', ha='center', fontsize=10, fontweight='bold', color='black')

    # --- recompute anything that depends on X_MAX, Y_MAX, GEAR_RADIUS_MM ---
    def refresh_dims():
        # axes & workspace
        ax.set_xlim(-FRAME_MARGIN, X_MAX + FRAME_MARGIN)
        ax.set_ylim(-FRAME_MARGIN, Y_MAX + FRAME_MARGIN)
        ws_rect.set_width(X_MAX); ws_rect.set_height(Y_MAX)

        # rails & crossbar
        rail_right.set_x(X_MAX + CARR_W/2.0)
        rail_right.set_height(Y_MAX + CARR_H)
        rail_left.set_height(Y_MAX + CARR_H)
        crossbar.set_width(X_MAX + CARR_W)

        # motor centers & radius
        a_c, b_c = motor_centers()
        motorA.center = a_c; motorB.center = b_c
        motorA.set_radius(GEAR_RADIUS_MM); motorB.set_radius(GEAR_RADIUS_MM)

        # idler ends track the top outer points
        x_left  = -CARR_W/2.0
        x_right = X_MAX + CARR_W/2.0
        idler_end_L.center = (x_left  - RAIL_W/2.0, Y_MAX + CARR_H/2.0 + IDLER_RADIUS_MM)
        idler_end_R.center = (x_right + RAIL_W/2.0, Y_MAX + CARR_H/2.0 + IDLER_RADIUS_MM)

    # call once
    refresh_dims()
    last_dims = {"x": X_MAX, "y": Y_MAX, "r": GEAR_RADIUS_MM}

    # --- geometry that tracks the carriage position (and dims each frame) ---
    def update_geometry(x, y):
        # derive current left/right from (possibly updated) X_MAX:
        x_left  = -CARR_W/2.0
        x_right = X_MAX + CARR_W/2.0

        crossbar.set_xy((-CARR_W/2, y - CROSSBAR_H/2.0))
        carriage.set_xy((x - CARR_W/2.0, y - CARR_H/2.0))

        idler_top_L.center = (x_left,  y + CARR_H/2.0)
        idler_top_R.center = (x_right, y + CARR_H/2.0)
        idler_bot_L.center = (x_left,  y - CARR_H)
        idler_bot_R.center = (x_right, y - CARR_H)

        belt_l_vert_top.set_data([x_left, x_left], [y + CARR_H/2.0, Y_MAX+CARR_H/2.0+IDLER_RADIUS_MM])
        belt_l_vert_bot.set_data([x_left+GEAR_RADIUS_MM-RAIL_W/2.0, x_left], [-CARR_H/2.0-GEAR_RADIUS_MM, y - CARR_H])
        belt_r_vert_top.set_data([x_right, x_right], [y + CARR_H/2.0, Y_MAX+CARR_H/2.0+IDLER_RADIUS_MM])
        belt_r_vert_bot.set_data([x_right-GEAR_RADIUS_MM+RAIL_W/2.0, x_right], [-CARR_H/2.0-GEAR_RADIUS_MM, y - CARR_H])

        belt_left.set_data([x_left-GEAR_RADIUS_MM-RAIL_W/2.0, x_left-RAIL_W],
                           [-CARR_H/2.0-GEAR_RADIUS_MM, Y_MAX+CARR_H/2.0+IDLER_RADIUS_MM])
        belt_right.set_data([x_right+GEAR_RADIUS_MM+RAIL_W/2.0, x_right+RAIL_W],
                            [-CARR_H/2.0-GEAR_RADIUS_MM, Y_MAX+CARR_H/2.0+IDLER_RADIUS_MM])

        belt_top_l.set_data([x_left, x-CARR_W/2.0], [y + CARR_H/2.0, y + CARR_H/2.0])
        belt_top_r.set_data([x_right, x+CARR_W/2.0], [y + CARR_H/2.0, y + CARR_H/2.0])
        belt_bot.set_data([x_left, x_right], [y - CARR_H, y - CARR_H])

        attach_L.center = (x - CARR_W/2.0, y + CARR_H/2.0)
        attach_R.center = (x + CARR_W/2.0, y + CARR_H/2.0)

    def set_dash_offset(line: Line2D, offset_pts: float):
        line.set_linestyle((offset_pts, DASH_STYLE[1]))

    # Mouse -> sim commands
    def on_motion(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            with sim.lock: sim.mouse_inside = False
            return
        x, y = float(event.xdata), float(event.ydata)
        inside = (0.0 <= x <= X_MAX) and (0.0 <= y <= Y_MAX)
        with sim.lock:
            sim.mouse_inside = inside
            if inside: sim.mouse_xy = (x, y)

    def on_leave(_event):
        with sim.lock: sim.mouse_inside = False

    def on_click(event):
        if event.inaxes is not ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if not (0.0 <= x <= X_MAX and 0.0 <= y <= Y_MAX):
            return

        mode = sim.state.mode
        if mode == MODE_SMART:
            vx = float(get_smart_vx()); vy = float(get_smart_vy())
            sim.cmd_q.put(f"{MODE_SMART},{x:.3f},{y:.3f},{vx:.3f},{vy:.3f}")
        elif mode == MODE_TARGET:
            sim.cmd_q.put(f"{MODE_TARGET},{x:.3f},{y:.3f}")

    # Connect events on the embedded canvas
    canvas.mpl_connect('motion_notify_event', on_motion)
    canvas.mpl_connect('figure_leave_event', on_leave)
    canvas.mpl_connect('button_press_event', on_click)

    def update_plot(_frame):
        # If dimensions changed via control panel, refresh static geometry once
        if (last_dims["x"] != X_MAX) or (last_dims["y"] != Y_MAX) or (last_dims["r"] != GEAR_RADIUS_MM):
            refresh_dims()
            last_dims["x"], last_dims["y"], last_dims["r"] = X_MAX, Y_MAX, GEAR_RADIUS_MM

        with sim.lock:
            x, y   = sim.state.x, sim.state.y
            tx, ty = sim.target
            mode   = sim.state.mode
            thA, thB = sim.state.thetaA, sim.state.thetaB
            vx_now, vy_now = sim.state.vx, sim.state.vy

        update_geometry(x, y)

        center_dot.set_data([x], [y])
        target_dot.set_data([tx], [ty])

        # Use live patch centers so updates to centers are reflected
        r_indicator = GEAR_RADIUS_MM * 0.65
        axA = motorA.center[0] + r_indicator * math.cos(-thA)
        ayA = motorA.center[1] + r_indicator * math.sin(-thA)
        axB = motorB.center[0] + r_indicator * math.cos( thB)
        ayB = motorB.center[1] + r_indicator * math.sin( thB)
        motorA_dot.set_data([axA], [ayA])
        motorB_dot.set_data([axB], [ayB])

        set_dash_offset(belt_top_l,      sim.belt_ofs['top_l'])
        set_dash_offset(belt_top_r,      sim.belt_ofs['top_r'])
        set_dash_offset(belt_bot,        sim.belt_ofs['bot'])
        set_dash_offset(belt_left,       sim.belt_ofs['left_outer'])
        set_dash_offset(belt_right,      sim.belt_ofs['right_outer'])
        set_dash_offset(belt_l_vert_top, sim.belt_ofs['l_vert_top'])
        set_dash_offset(belt_l_vert_bot, sim.belt_ofs['l_vert_bot'])
        set_dash_offset(belt_r_vert_top, sim.belt_ofs['r_vert_top'])
        set_dash_offset(belt_r_vert_bot, sim.belt_ofs['r_vert_bot'])

        settled = (abs(tx - x) <= SNAP_POS and abs(ty - y) <= SNAP_POS
                   and abs(vx_now) <= SNAP_VEL and abs(vy_now) <= SNAP_VEL)

        eta_txt = "—"
        show_preview = (not settled) and (mode not in (MODE_CIRCLE, MODE_SQUARE, MODE_REHOME, MODE_STOP, MODE_IDLE, MODE_SMART))
        if show_preview:
            t_now = time.time()
            xs, ys, eta = sim.predict_to_target(max_secs=20.0, dt_step=DT*2.0)
            _pred_cache.update({
                "tx": tx, "ty": ty, "ts": t_now,
                "xs": xs or [], "ys": ys or [],
                "eta": f"{eta:.2f}s" if xs and ys else "—"
            })

        if show_preview:
            path_line.set_data(_pred_cache["xs"], _pred_cache["ys"])
            eta_txt = _pred_cache["eta"]
        else:
            path_line.set_data([], [])
            _pred_cache["tx"] = _pred_cache["ty"] = None

        # with sim.lock:
        #     smart_active = sim.smart_active
        #     if smart_active:
        #         xs_s, ys_s = sim.smart_path
        #         smart_t = time.time() - sim.smart_t0
        #         eta_txt = f"{max(sim.smart_T - smart_t, 0.0):.2f}s"
        #     else:
        #         xs_s, ys_s = ([], [])
        # smart_path_line.set_data(xs_s, ys_s)

        with sim.lock:
            smart_active = sim.smart_active
            p1 = sim.smart_p1
            p0 = sim.smart_p0
            xs_s, ys_s = sim.smart_path if smart_active else ([], [])
            xs_tail, ys_tail = sim.smart_decel_path if smart_active else ([], [])

        smart_path_line.set_data(xs_s, ys_s)
        smart_decel_line.set_data(xs_tail, ys_tail)
        # intercept_dot.set_data([p0],[p1])

        spd.set_text(f"ETA: {eta_txt}")
        hud.set_text(f"Mode: {mode}  |  Target: ({tx:.1f}, {ty:.1f})  |  Pos: ({x:.1f}, {y:.1f})")
        return ()

    ani = FuncAnimation(fig, update_plot, interval=1000/60.0)
    fig._ani = ani
    canvas._ani = ani
    return fig, canvas

# ===============================
# Control Panel (Qt, side-by-side fields + old colors)
# ===============================

def run_control_panel(sim: GantrySim):
    # Shared values for sim click handler
    smart_vx_val = {"v": 0.0}
    smart_vy_val = {"v": 200.0}
    tgt_x_val    = {"v": X_MAX/2.0}
    tgt_y_val    = {"v": Y_MAX/2.0}
    cx_val       = {"v": X_MAX/2.0}
    cy_val       = {"v": Y_MAX/2.0}
    sqw_val      = {"v": min(300.0, X_MAX)}
    sqh_val      = {"v": min(300.0, Y_MAX)}

    w = QtWidgets.QWidget()
    w.setObjectName("controlPanel")
    w.setStyleSheet("""
        QWidget#controlPanel { background-color: #f6f0df; }
        QGroupBox { font-weight: bold; border: 1px solid #d9d0b8; margin-top: 6px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
        QLabel { font-weight: normal; }
    """)
    root = QtWidgets.QVBoxLayout(w)

    def mk_spin(val, step, minv, maxv, cb, decimals=3, width=90):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minv, maxv); spin.setSingleStep(step); spin.setDecimals(decimals)
        spin.setValue(val["v"])
        spin.setMaximumWidth(width)
        def on_change(v):
            val["v"] = float(v); cb(float(v))
        spin.valueChanged.connect(on_change)
        return spin

    def pair_row(label_left, spin_left, label_right, spin_right):
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel(label_left))
        row.addWidget(spin_left)
        row.addSpacing(10)
        row.addWidget(QtWidgets.QLabel(label_right))
        row.addWidget(spin_right)
        row.addStretch(1)
        return row

    # --- Geometry / Params ---
    gb_geo = QtWidgets.QGroupBox("Geometry & Params")
    geo = QtWidgets.QVBoxLayout(gb_geo)

    def on_xmax(v):
        global X_MAX
        X_MAX = max(50.0, v)
    def on_ymax(v):
        global Y_MAX
        Y_MAX = max(50.0, v)
    def on_gearr(v):
        global GEAR_RADIUS_MM
        GEAR_RADIUS_MM = max(1.0, v)
    def on_motor_rpm(v):
        global MOTOR_RPM_MAX
        MOTOR_RPM_MAX = max(1.0, v)

    spin_xmax = mk_spin({"v": X_MAX}, 10.0, 50.0, 5000.0, on_xmax, decimals=1)
    spin_ymax = mk_spin({"v": Y_MAX}, 10.0, 50.0, 5000.0, on_ymax, decimals=1)
    geo.addLayout(pair_row("Workspace X", spin_xmax, "Workspace Y", spin_ymax))

    spin_gearr  = mk_spin({"v": GEAR_RADIUS_MM}, 0.5, 1.0, 200.0, on_gearr, decimals=2)
    spin_rpm    = mk_spin({"v": MOTOR_RPM_MAX}, 10.0, 1.0, 50000.0, on_motor_rpm, decimals=0)
    geo.addLayout(pair_row("Gear radius", spin_gearr, "Motor RPM", spin_rpm))

    root.addWidget(gb_geo)

    # --- Target & Smart ---
    gb_target = QtWidgets.QGroupBox("Target & Smart")
    lay_t = QtWidgets.QVBoxLayout(gb_target)

    spin_tx = mk_spin(tgt_x_val, 5.0, 0.0, 10000.0, lambda _: None, decimals=1)
    spin_ty = mk_spin(tgt_y_val, 5.0, 0.0, 10000.0, lambda _: None, decimals=1)
    lay_t.addLayout(pair_row("Target X", spin_tx, "Target Y", spin_ty))

    spin_svx = mk_spin(smart_vx_val, 10.0, -5000.0, 5000.0, lambda _: None, decimals=1)
    spin_svy = mk_spin(smart_vy_val, 10.0, -5000.0, 5000.0, lambda _: None, decimals=1)
    lay_t.addLayout(pair_row("Smart vx", spin_svx, "Smart vy", spin_svy))

    root.addWidget(gb_target)

    # --- Circle & Square ---
    gb_shapes = QtWidgets.QGroupBox("Circle & Square")
    lay_s = QtWidgets.QVBoxLayout(gb_shapes)

    spin_cx = mk_spin(cx_val, 5.0, 0.0, 10000.0, lambda _: None, decimals=1)
    spin_cy = mk_spin(cy_val, 5.0, 0.0, 10000.0, lambda _: None, decimals=1)
    lay_s.addLayout(pair_row("Circle X", spin_cx, "Circle Y", spin_cy))

    spin_sw = mk_spin(sqw_val, 5.0, 1.0, 10000.0, lambda _: None, decimals=1)
    spin_sh = mk_spin(sqh_val, 5.0, 1.0, 10000.0, lambda _: None, decimals=1)
    lay_s.addLayout(pair_row("Square W", spin_sw, "Square H", spin_sh))

    root.addWidget(gb_shapes)

    # --- Buttons ---
    gb_btns = QtWidgets.QGroupBox("Commands")
    grid = QtWidgets.QGridLayout(gb_btns)

    def add_btn(r, c, text, f):
        b = QtWidgets.QPushButton(text); b.clicked.connect(f); grid.addWidget(b, r, c)

    add_btn(0, 0, "Idle (I)",   lambda: sim.handle_command(MODE_IDLE))
    add_btn(0, 1, "Stop (X)",   lambda: sim.handle_command(MODE_STOP))
    add_btn(1, 0, "Rehome (R)", lambda: sim.handle_command(MODE_REHOME))
    add_btn(1, 1, "Mouse (M)",  lambda: sim.handle_command(MODE_MOUSE))

    def do_target():
        x = tgt_x_val["v"]; y = tgt_y_val["v"]
        sim.handle_command(f"{MODE_TARGET},{x},{y}")
    def do_smart():
        x = tgt_x_val["v"]; y = tgt_y_val["v"]
        vx = smart_vx_val["v"]; vy = smart_vy_val["v"]
        sim.handle_command(f"{MODE_SMART},{x},{y},{vx},{vy}")
    def do_circle():
        sim.handle_command(f"{MODE_CIRCLE},{cx_val['v']},{cy_val['v']}")
    def do_square():
        sim.handle_command(f"{MODE_SQUARE},{sqw_val['v']},{sqh_val['v']}")

    add_btn(2, 0, "Target (T)", do_target)
    add_btn(2, 1, "Smart (U)",  do_smart)
    add_btn(3, 0, "Circle (C)", do_circle)
    add_btn(3, 1, "Square (S)", do_square)

    root.addWidget(gb_btns)
    root.addStretch(1)

    # Return widget + getters used by sim click handler
    return (w, lambda: smart_vx_val["v"], lambda: smart_vy_val["v"])

# ===============================
# Main Window (dockable panel)
# ===============================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sim: GantrySim):
        super().__init__()
        self.setWindowTitle("Thesis Digital Twin")

        # Build control panel first to pass getters into the sim canvas
        panel_widget, get_svx, get_svy = run_control_panel(sim)
        fig, canvas = build_sim_canvas(sim, get_svx, get_svy)

        # Central: sim canvas
        self.setCentralWidget(canvas)

        # Dock: control panel
        dock = QtWidgets.QDockWidget("Control Panel", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea |
                             QtCore.Qt.TopDockWidgetArea  | QtCore.Qt.BottomDockWidgetArea)
        dock.setWidget(panel_widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        # Keep refs alive
        self._keep = (fig, canvas, dock, panel_widget)

# ===============================
# Main
# ===============================

def main():
    sim = GantrySim()
    th_server = threading.Thread(target=sim.server_thread, daemon=True); th_server.start()
    th_sim    = threading.Thread(target=sim.sim_loop_thread, daemon=True); th_sim.start()

    try:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        win = MainWindow(sim)
        win.resize(1400, 860)
        win.show()
        app.exec_()
    finally:
        sim.running = False
        th_server.join(timeout=1.0); th_sim.join(timeout=1.0)

if __name__ == "__main__":
    main()
