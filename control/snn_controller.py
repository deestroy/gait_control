import numpy as np
from snn.brian2_cpg_drive import CPGDrive

class SNNController:
    """
    Produces a smooth StepVelocity command from Brian2 spike activity.

    Key features:
    - Uses per-tick spike count (not cumulative)
    - Low-pass filters velocity
    - Slew-rate limits changes per tick
    - Conservative gain/clamps for stable gait
    """
    def __init__(self, dt_s: float = 0.01):
        self.dt_s = float(dt_s)
        self.cpg = CPGDrive(dt_s=self.dt_s)

        # Baseline walking speed
        self.base_vel = 1.2

        # Start conservative; tune later
        self.gain = 2e-4            # was 0.002 (too large)
        self.min_vel = 0.9
        self.max_vel = 1.5

        # Smoothing / stability
        self.beta = 0.05            # EMA low-pass factor (0.02–0.1)
        self.max_dv = 0.02          # max velocity change per control tick

        # Internal state
        self.vel_filt = self.base_vel

        # If CPGDrive returns cumulative spikes, we convert to per-tick
        self.prev_total_spikes = 0

    def compute_step_velocity(self, state):
        # Get spikes from CPGDrive for this tick
        spikes_val = self.cpg.step()

        # Convert to "spikes this tick" if spikes_val looks cumulative
        # Heuristic: if it never goes down and keeps growing large, treat as cumulative.
        # We'll still compute a delta safely.
        total = int(spikes_val)
        spikes_tick = total - self.prev_total_spikes
        if spikes_tick < 0:
            # If the monitor resets internally, just treat spikes_val as per-tick
            spikes_tick = total
        self.prev_total_spikes = total

        # Raw velocity from spikes
        vel_raw = self.base_vel + self.gain * spikes_tick
        vel_raw = float(np.clip(vel_raw, self.min_vel, self.max_vel))

        # Low-pass filter
        vel_lp = (1.0 - self.beta) * self.vel_filt + self.beta * vel_raw

        # Slew-rate limit (prevents bounce)
        dv = float(np.clip(vel_lp - self.vel_filt, -self.max_dv, self.max_dv))
        self.vel_filt += dv

        return self.vel_filt, spikes_tick