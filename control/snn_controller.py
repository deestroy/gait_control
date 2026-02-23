import numpy as np
from snn.brian2_cpg import QuadrupedCPG

class SNNController:
    def __init__(self):
        self.cpg = QuadrupedCPG(dt=0.01)

    def compute_gait_params(self):

        spike_counts = self.cpg.step()

        # Convert spike rate to velocity modulation
        velocity = 1.0 + 0.2 * np.mean(spike_counts)

        StepLength = 0.08
        LateralFraction = 0.0
        YawRate = 0.0
        StepVelocity = velocity
        ClearanceHeight = 0.06
        PenetrationDepth = 0.01

        return StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth
