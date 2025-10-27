# Communications/ipc.py
from __future__ import annotations
import json, socket, threading, time, queue
from dataclasses import dataclass
import serial

CTRL_PORT = 48080        # sim -> tracker (mode control)
INTERCEPT_PORT = 48081   # tracker -> sim (intercept events)
BIND_ADDR = "127.0.0.1"  # change to LAN IP if processes on different machines

def _mk_recv_sock(port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((BIND_ADDR, port))
    s.setblocking(False)
    return s

def _mk_send_sock() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setblocking(False)
    return s

def _send_json(sock: socket.socket, addr: tuple[str,int], obj: dict):
    data = (json.dumps(obj) + "\n").encode("utf-8")
    try: sock.sendto(data, addr)
    except Exception: pass

def _try_recv_json(sock: socket.socket):
    try:
        data, _ = sock.recvfrom(65536)
        line = data.decode("utf-8", errors="ignore").strip()
        return json.loads(line)
    except BlockingIOError:
        return None
    except Exception:
        return None

# -------- Control (sim -> tracker) --------

class ControlPublisher:
    """Sim publishes current mode periodically: {'type':'control','mode':'SIM'|'LIVE','ttl':seconds}"""
    def __init__(self, port: int = CTRL_PORT):
        self.sock = _mk_send_sock()
        self.addr = (BIND_ADDR, port)

    def publish(self, mode: str, ttl: float = 2.0):
        _send_json(self.sock, self.addr, {"type":"control", "mode":mode, "ttl":float(ttl), "ts":time.time()})

class ControlListener:
    """Tracker listens for mode. live_enabled true while last message is fresh."""
    def __init__(self, port: int = CTRL_PORT):
        self.sock = _mk_recv_sock(port)
        self.live_enabled = False
        self._expiry = 0.0
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        while True:
            msg = _try_recv_json(self.sock)
            now = time.time()
            if msg and msg.get("type") == "control":
                mode = str(msg.get("mode","SIM")).upper()
                ttl = float(msg.get("ttl", 2.0))
                self._expiry = now + max(0.5, ttl)
                self.live_enabled = (mode == "LIVE")
            if now > self._expiry:
                self.live_enabled = False
            time.sleep(0.01)

# -------- Intercepts (tracker -> sim) --------

@dataclass
class Intercept:
    ts: float
    x_mm: float
    y_mm: float
    t_hit_s: float
    source: str = "tracker"

class InterceptPublisher:
    def __init__(self, port: int = INTERCEPT_PORT):
        self.sock = _mk_send_sock()
        self.addr = (BIND_ADDR, port)

    def publish(self, x_mm: float, y_mm: float, t_hit_s: float, source: str = "tracker"):
        obj = {"type":"intercept","ts":time.time(),"x_mm":float(x_mm),"y_mm":float(y_mm),
               "t_hit_s":float(t_hit_s),"source":source}
        _send_json(self.sock, self.addr, obj)

# ===================== Serial sender (same format) =================
class InterceptSender:
	"""Sends ASCII: (TRACK,x_mm,y_mm,t_ms)\n"""
	def __init__(self, enabled: bool, port: str, baud: int):
		self.enabled = bool(enabled) and (serial is not None)
		self.port = port; self.baud = baud; self.ser = None
		if self.enabled:
			try:
				self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
				time.sleep(0.1)
				print(f"[Serial] Opened {self.port} @ {self.baud}")
			except Exception as e:
				print(f"[Serial] ERROR opening {self.port}: {e}")
				self.enabled = False

	def send(self, x_m: float, y_m: float, t_s: float):
		x_mm = int(round(1000*x_m)); y_mm = int(round(1000*y_m))
		t_ms = max(0, int(round(1000*t_s)))
		msg = f"(TRACK,{x_mm},{y_mm},{t_ms})\n"
		if self.enabled and self.ser:
			try: self.ser.write(msg.encode("ascii"))
			except Exception as e: print(f"[Serial] write failed: {e} :: {msg.strip()}")
		else:
			print(f"[TRACK] x_mm={x_mm} y_mm={y_mm} t_ms={t_ms}")

	def close(self):
		try:
			if self.ser and self.ser.is_open:
				self.ser.close()
		except Exception: pass
# ==================================================================

class InterceptSubscriber:
    """Non-blocking queue of Intercept messages."""
    def __init__(self, port: int = INTERCEPT_PORT):
        self.sock = _mk_recv_sock(port)
        self.q = queue.Queue(maxsize=256)
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        while True:
            msg = _try_recv_json(self.sock)
            if msg and msg.get("type") == "intercept":
                try:
                    itc = Intercept(ts=float(msg["ts"]), x_mm=float(msg["x_mm"]),
                                    y_mm=float(msg["y_mm"]), t_hit_s=float(msg["t_hit_s"]),
                                    source=str(msg.get("source","tracker")))
                    self.q.put_nowait(itc)
                except Exception:
                    pass
            time.sleep(0.005)

    def try_get(self) -> Intercept | None:
        try: return self.q.get_nowait()
        except queue.Empty: return None
