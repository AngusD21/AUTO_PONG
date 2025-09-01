# gantry_controller.py
import time, threading
from typing import Optional, Tuple
from Path_Planning_2D.smart_path_planner import Limits, plan_smart
from Communications.gantry_comm import Gantry

class GantryController:
    def __init__(self, port: str, baud: int = 115200):
        self.g = Gantry(port, baud)
        self.lock = threading.Lock()
        self.plan_thread: Optional[threading.Thread] = None
        self.abort_flag = False
        self.latest_status = (0.0, 0.0, "I")

    def close(self):
        self.g.close()

    def read_status_once(self) -> Tuple[float, float, str]:
        st = self.g.query()
        if st: self.latest_status = st
        return self.latest_status

    def start_intercept(self, x_mm: float, y_mm: float, t_hit_s: float, v_end: Tuple[float,float]=(0.0,0.0)):
        """
        Plan a SMART path to (x_mm, y_mm) that ends with velocity v_end (mm/s).
        Streams ~50 Hz waypoint targets to Arduino (which runs trapezoids per-axis).
        """
        with self.lock:
            self.abort_flag = True
            if self.plan_thread and self.plan_thread.is_alive():
                self.plan_thread.join()
            self.abort_flag = False

        # Snapshot current pos/vel (we only have pos; assume ~0 start vel)
        px, py, _ = self.read_status_once()
        p0 = (px, py); v0 = (0.0, 0.0)     # if you estimate gantry velocity, put it here
        p1 = (x_mm, y_mm); v1 = v_end

        # Build schedule (identical to simulator logic)
        limits = Limits()
        schedule, T_total = plan_smart(p0, v0, p1, v1, limits, dt_stream=0.02)

        # Optionally, “lead” by t_hit_s if you want to sync to arrival time
        lead = max(0.0, t_hit_s - T_total)

        def run():
            self.g.smart(x_mm, y_mm, v_end[0], v_end[1])  # sets fast accel caps
            self.g.play_schedule(schedule, lead_time_s=lead)

        self.plan_thread = threading.Thread(target=run, daemon=True)
        self.plan_thread.start()
