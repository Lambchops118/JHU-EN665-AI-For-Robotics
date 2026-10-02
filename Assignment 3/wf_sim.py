#!/usr/bin/env python3
"""
AI for Robotics - Module 03 Programming Assignment
Wall-Following Robot: Feedback Control, Timing, and Safe Autonomy

A minimal 2-D differential-drive (unicycle) simulator. The robot drives forward
at a constant speed and must hold a target distance d* from a wall (the x-axis,
y = 0) using a NOISY distance sensor. You will implement and study feedback
controllers and the effects of sampling rate, latency, and jitter, then add an
open-world safety override.

Geometry / sign convention (matches Module 03 slide 7):
    measured distance d(t) ~ y (+ noise)
    error  e = d* - d
        e > 0  -> too FAR from wall
        e < 0  -> too CLOSE to wall
    steering law omega = controller(e): with the unicycle below,
    omega = Kp * e steers the robot back toward d*.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless save-to-file; comment out to view live
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque


# ----------------------------------------------------------------------------
# Simulation configuration
# ----------------------------------------------------------------------------
@dataclass
class SimConfig:
    d_star: float = 1.0           # desired distance to wall (m)
    v: float = 0.5                # constant forward speed (m/s)
    y0: float = 2.0               # initial distance from wall (m)
    theta0: float = 0.0           # initial heading (rad, +x)
    T: float = 40.0               # total simulated time (s)
    dt_nominal: float = 0.05      # nominal loop period (s)  -> 20 Hz
    deadline: float = 0.05        # control deadline (s); a cycle slower than this "misses"
    sensor_noise_std: float = 0.02
    latency_steps: int = 0        # controller acts on a measurement delayed by N steps
    jitter_std: float = 0.0       # std of loop-period jitter (s)
    setpoint_step_time: float = None   # if set, d* changes at this time...
    setpoint_step_value: float = None  # ...to this value
    obstacle_x: float = None      # if set, a blockage crosses the path at this x (m)
    safety_override_enabled: bool = True  # set False only to demonstrate failure
    seed: int = 0

    OMEGA_MAX: float = 1.5        # bounded action space (rad/s) -- slide 51
    SAFE_DIST: float = 0.6        # front clearance that triggers the override (m)


# ----------------------------------------------------------------------------
# Controllers
# ----------------------------------------------------------------------------
class PController:
    """Proportional steering law (GIVEN - use it in Part 1)."""
    def __init__(self, Kp):
        self.Kp = Kp

    def reset(self):
        pass

    def __call__(self, e, dt):
        return self.Kp * e


class PIDController:
    """
    PID steering law.

    >>> PART 2 TODO <<<
    The P term is given. Implement the I and D terms following slide 9:
        u = Kp*e + Ki * (integral of e dt) + Kd * (de/dt)
    Handle the first call (no previous error yet) gracefully.
    """
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_e = None

    def __call__(self, e, dt):
        p = self.Kp * e

        # Accumulate the discrete-time integral.  The clamp prevents a long
        # saturated turn from storing an unbounded correction (windup).
        if dt > 0:
            self._integral = float(np.clip(self._integral + e * dt, -2.0, 2.0))
        i = self.Ki * self._integral

        # The first sample has no earlier error; use a zero derivative then.
        derivative = 0.0
        if self._prev_e is not None and dt > 0:
            derivative = (e - self._prev_e) / dt
        d = self.Kd * derivative

        self._prev_e = e   # store for the next derivative estimate
        return p + i + d


# ----------------------------------------------------------------------------
# World helpers
# ----------------------------------------------------------------------------
def front_obstacle_distance(x, cfg):
    """Distance straight ahead (+x) to a blockage centered at cfg.obstacle_x.
    Returns np.inf if there is no obstacle or it is already behind the robot."""
    if cfg.obstacle_x is None:
        return np.inf
    gap = cfg.obstacle_x - x
    return gap if gap >= 0 else np.inf


# ----------------------------------------------------------------------------
# Core sense-decide-act loop  (slide 5)
# ----------------------------------------------------------------------------
def run_sim(cfg, controller):
    rng = np.random.default_rng(cfg.seed)
    x, y, theta = 0.0, cfg.y0, cfg.theta0
    d_star = cfg.d_star
    controller.reset()
    meas_buffer = deque()

    keys = ["t", "x", "y", "theta", "d_star", "d_meas", "e",
            "omega", "v", "dt", "deadline_miss", "overridden"]
    log = {k: [] for k in keys}

    t = 0.0
    while t < cfg.T:
        # --- loop timing: jitter (slide 14) ---
        dt = cfg.dt_nominal
        if cfg.jitter_std > 0:
            dt = max(0.001, cfg.dt_nominal + rng.normal(0.0, cfg.jitter_std))
        deadline_miss = dt > cfg.deadline + 1e-12

        # --- setpoint step (transient demo) ---
        if cfg.setpoint_step_time is not None and t >= cfg.setpoint_step_time:
            d_star = cfg.setpoint_step_value

        # --- SENSE: noisy distance signal (slide 32) ---
        d_meas = y + rng.normal(0.0, cfg.sensor_noise_std)

        # --- latency: act on a delayed measurement (slide 13) ---
        meas_buffer.append(d_meas)
        if len(meas_buffer) > cfg.latency_steps:
            d_used = meas_buffer.popleft()
        else:
            d_used = meas_buffer[0]

        # --- error + DECIDE (slide 6) ---
        e = d_star - d_used
        omega = controller(e, dt)
        v = cfg.v
        overridden = False

        # --- reactive open-world safety override --------------------------
        # The wall-follower assumes a clear path -- a closed-world assumption.
        # Use front_obstacle_distance(x, cfg) to check the path ahead. If a
        # blockage is closer than cfg.SAFE_DIST, this controller must NOT keep
        # driving forward. Intervene with a bounded, safe fallback (decide what
        # to do with v and omega so the robot stays safe and gets clear of the
        # wall), and set `overridden = True` when you intervene.
        # ------------------------------------------------------------------
        if cfg.safety_override_enabled and front_obstacle_distance(x, cfg) < cfg.SAFE_DIST:
            v = 0.0
            omega = cfg.OMEGA_MAX
            overridden = True

        # --- bounded action space (slide 51) ---
        omega = float(np.clip(omega, -cfg.OMEGA_MAX, cfg.OMEGA_MAX))

        # --- ACT: integrate unicycle kinematics ---
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        for k, val in zip(keys, [t, x, y, theta, d_star, d_meas, e,
                                 omega, v, dt, deadline_miss, overridden]):
            log[k].append(val)
        t += dt

    for k in log:
        log[k] = np.array(log[k])
    return log


# ----------------------------------------------------------------------------
# Performance metric  (a small "utility"/cost, slides 37-38)
# ----------------------------------------------------------------------------
def metrics(log, settle_after=10.0):
    t = log["t"]
    true_err = log["d_star"] - log["y"]          # tracking error vs setpoint
    mask = t >= settle_after                      # ignore the startup transient
    rms = float(np.sqrt(np.mean(true_err[mask] ** 2))) if mask.any() else float("nan")
    overshoot = float(np.max(np.abs(true_err)))   # worst excursion (incl. transient)
    effort = float(np.sum(np.abs(log["omega"]) * log["dt"]))  # control effort
    miss_pct = 100.0 * float(np.mean(log["deadline_miss"]))
    return dict(rms_error=rms, max_overshoot=overshoot,
                control_effort=effort, deadline_miss_pct=miss_pct)


def autotune(cfg, budget):
    """Search a bounded PID gain grid with at most budget evaluations."""
    if budget <= 0:
        raise ValueError("budget must be a positive number of candidates")

    kp_values = [0.05, 0.10, 0.20, 0.40, 0.80]
    ki_values = [0.0, 0.001, 0.005, 0.010]
    kd_values = [0.0, 0.10, 0.20, 0.30, 0.40]

    best_gains = None
    best_metrics = None
    best_cost = np.inf
    evaluated = 0

    for Kp in kp_values:
        for Ki in ki_values:
            for Kd in kd_values:
                if evaluated >= budget:
                    return best_gains, best_metrics, best_cost, evaluated

                log = run_sim(cfg, PIDController(Kp, Ki, Kd))
                candidate_metrics = metrics(log)
                cost = (candidate_metrics["rms_error"]
                        + 0.01 * candidate_metrics["control_effort"])
                evaluated += 1

                if cost < best_cost:
                    best_gains = (Kp, Ki, Kd)
                    best_metrics = candidate_metrics
                    best_cost = cost

    return best_gains, best_metrics, best_cost, evaluated


def print_metrics(name, log):
    m = metrics(log)
    print(f"{name:<28}  RMS={m['rms_error']:.3f} m   "
          f"overshoot={m['max_overshoot']:.3f} m   "
          f"effort={m['control_effort']:.2f}   "
          f"deadline_miss={m['deadline_miss_pct']:.1f}%")


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_runs(runs, title, fname):
    """runs: list of (label, log). Top = trajectory, bottom = error vs time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    for label, log in runs:
        ax1.plot(log["x"], log["y"], label=label, lw=1.6)
        ax2.plot(log["t"], log["d_star"] - log["y"], label=label, lw=1.4)
    # wall and setpoint reference
    d_star0 = runs[0][1]["d_star"][0]
    ax1.axhline(0.0, color="k", lw=3, label="wall")
    ax1.axhline(d_star0, color="gray", ls="--", lw=1, label="desired d*")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y = distance to wall (m)")
    ax1.set_title(title); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.axhline(0.0, color="gray", ls="--", lw=1)
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("tracking error d* - y (m)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  saved {fname}")


# ----------------------------------------------------------------------------
# Demo: runs out of the box with the provided P controller.
# Replace / extend in the assignment parts.
# ----------------------------------------------------------------------------
def main():
    print("== Part 1 demo: proportional control, sweeping Kp ==")
    runs = []
    for Kp in [0.4, 1.5, 4.0]:
        cfg = SimConfig()
        log = run_sim(cfg, PController(Kp))
        runs.append((f"P, Kp={Kp}", log))
        print_metrics(f"P  Kp={Kp}", log)
    plot_runs(runs, "Proportional control: effect of gain", "part1_pcontrol.png")

    print("\n== PID scaffold (Part 2 - currently behaves like P until you fill it in) ==")
    cfg = SimConfig(setpoint_step_time=20.0, setpoint_step_value=1.5)
    log = run_sim(cfg, PIDController(Kp=1.5, Ki=0.3, Kd=0.8))
    print_metrics("PID (your gains)", log)
    plot_runs([("PID", log)], "PID with a setpoint step at t=20s", "part2_pid.png")


if __name__ == "__main__":
    main()
