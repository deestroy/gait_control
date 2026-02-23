import os
import sys
import numpy as np
import copy

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMODULE_ROOT = os.path.join(REPO_ROOT, "external", "spot_mini_mini")
sys.path.insert(0, SUBMODULE_ROOT)

from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.OpenLoopSM.SpotOL import BezierStepper
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.util.gui import GUI


N_STAND = 300
N_RAMP  = 800

# -----------------------------
# SNN hook (currently no-op)
# -----------------------------
def snn_override_gait_params(
    state,
    StepLength, LateralFraction, YawRate, StepVelocity,
    ClearanceHeight, PenetrationDepth, SwingPeriod
):
    """
    Later: take IMU + contacts from state and output modulated gait params from Brian2.
    For now: return inputs unchanged (baseline).
    """
    return StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth, SwingPeriod


def main():
    env = spotBezierEnv(
        render=True,
        on_rack=False,
        height_field=False,
        draw_foot_path=False,
        env_randomizer=None
    )

    state = env.reset()
    g_u_i = GUI(env.spot.quadruped)

    spot = SpotModel()
    T_bf0 = spot.WorldToFoot
    T_bf = copy.deepcopy(T_bf0)

    bzg = BezierGait(dt=env._time_step)
    bz_step = BezierStepper(dt=env._time_step, mode=1)

    # Placeholder action; env.step() uses internal self.ja (set by pass_joint_angles)
    try:
        action = env.action_space.sample()
    except Exception:
        action = None

    contacts = state[-4:]

    # -------- Baseline gait params (SNN will modulate later) --------
    StepLength = 0.08
    LateralFraction = 0.0
    YawRate = 0.0
    StepVelocity = 1.2
    ClearanceHeight = 0.06
    PenetrationDepth = 0.01
    SwingPeriod = 0.25  # seconds

    # If BezierGait supports Tswing
    if hasattr(bzg, "Tswing"):
        bzg.Tswing = SwingPeriod

    # Optional yaw stabilization
    AUTO_YAW = False
    P_yaw = 5.0

    max_timesteps = 20000
    for t in range(max_timesteps):

        bz_step.ramp_up()
        pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth = bz_step.StateMachine()
        # pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth, SwingPeriod = g_u_i.UserInput()

        # Force baseline walking targets (we'll later replace these with SNN output)
        StepLength_target = 0.08
        StepVelocity_target = 1.2
        ClearanceHeight_target = 0.06
        PenetrationDepth_target = 0.01

        StepLength = StepLength_target
        StepVelocity = StepVelocity_target
        ClearanceHeight = ClearanceHeight_target
        PenetrationDepth = PenetrationDepth_target

        bzg.Tswing = SwingPeriod
        # ---- Your SNN hook (currently no-op) ----
        # StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth, SwingPeriod = \
        #     snn_override_gait_params(
        #         state,
        #         StepLength, LateralFraction, YawRate, StepVelocity,
        #         ClearanceHeight, PenetrationDepth, SwingPeriod
        #     )

        # if t < N_STAND:
        #     StepLength = 0.0
        #     StepVelocity = 0.0
        # else:
        #     alpha = min(1.0, (t - N_STAND) / float(N_RAMP))
        #     StepLength *= alpha
        #     StepVelocity *= alpha


        if hasattr(bzg, "Tswing"):
            bzg.Tswing = SwingPeriod

        if AUTO_YAW:
            yaw = env.return_yaw()
            YawRate += -yaw * P_yaw

        contacts = state[-4:]

        # Generate desired foot trajectories
        T_bf = bzg.GenerateTrajectory(
            StepLength, LateralFraction, YawRate,
            StepVelocity, T_bf0, T_bf,
            ClearanceHeight, PenetrationDepth,
            contacts
        )

        # IK (from SpotModel)
        joint_angles = spot.IK(orn, pos, T_bf)

        # CRITICAL: set env.ja so env.step() won't crash
        env.pass_joint_angles(joint_angles.reshape(-1))

        # Step physics
        state, reward, done, info = env.step(action)
        if t % 50 == 0:
            print(f"t={t} roll={state[0]:+.3f} pitch={state[1]:+.3f} StepVel={StepVelocity:.2f} StepLen={StepLength:.2f}")

        if done:
            state = env.reset()

    env.close()


if __name__ == "__main__":
    main()
