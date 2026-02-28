from brian2 import *

class CPGDrive:
    """
    Minimal Brian2 oscillator producing a scalar drive signal.
    Returns spikes PER CONTROL TICK (delta), not cumulative.
    """
    def __init__(self, dt_s=0.01):
        self.dt_s = float(dt_s)
        defaultclock.dt = self.dt_s * second

        eqs = """
        dv/dt = (I - v)/tau : 1
        I : 1
        tau : second
        """

        self.g = NeuronGroup(20, eqs, threshold="v>1", reset="v=0", method="euler")
        self.g.v = 0
        self.g.I = '1.02 + 0.06*rand()'
        self.g.tau = 0.05 * second
        self.g.v = 'rand()'

        self.sm = SpikeMonitor(self.g)
        self.net = Network(self.g, self.sm)

        # Track cumulative count to compute per-tick delta
        self.prev_total_spikes = 0

    def step(self) -> int:
        # Run exactly one control tick
        self.net.run(defaultclock.dt)

        total = int(self.sm.num_spikes)                 # cumulative
        spikes_tick = total - self.prev_total_spikes    # per-tick delta
        self.prev_total_spikes = total

        # Safety: if Brian2 resets internally for any reason
        if spikes_tick < 0:
            spikes_tick = total

        return spikes_tick