# gantry_comm.py
import time
import serial

class Gantry:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.05):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(0.1)

    def close(self):
        try: self.ser.close()
        except Exception: pass

    def _send(self, line: str):
        if not line.endswith("\n"): line += "\n"
        self.ser.write(line.encode("ascii"))

    def idle(self):          self._send("I")
    def stop(self):          self._send("X")
    def rehome(self):        self._send("R")
    def set_param(self, k, v): self._send(f"P,{k},{v}")

    def target(self, x_mm: float, y_mm: float):
        self._send(f"T,{x_mm:.3f},{y_mm:.3f}")

    def smart(self, x_mm: float, y_mm: float, vx: float = 0.0, vy: float = 0.0):
        # Planning stays on Python; Arduino just uses fast accel caps.
        self._send(f"U,{x_mm:.3f},{y_mm:.3f},{vx:.3f},{vy:.3f}")

    def query(self) -> tuple[float, float, str] | None:
        self._send("Q")
        try:
            line = self.ser.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("POS,"):
                _, sx, sy, smode = line.split(",", 3)
                return float(sx), float(sy), smode
        except Exception:
            pass
        return None

    # Stream a time-stamped waypoint schedule: list of (t_monotonic, x_mm, y_mm)
    def play_schedule(self, schedule, lead_time_s: float = 0.0):
        t0 = time.monotonic() + max(0.0, lead_time_s)
        for (t_when, x, y) in schedule:
            # Sleep until it's time to send this waypoint
            while time.monotonic() < t0 + t_when:
                time.sleep(0.0005)
            self.target(x, y)
