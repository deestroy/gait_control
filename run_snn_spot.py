import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMODULE_ROOT = os.path.join(REPO_ROOT, "external", "spot_mini_mini")
sys.path.insert(0, SUBMODULE_ROOT)

from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.Kinematics.SpotKinematics import SpotModel


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

    # Placeholder action; env.step() uses internal self.ja (set by pass_joint_angles)
    try:
        action = env.action_space.sample()
    except Exception:
        action = None

    # Gait + kinematics
    bzg = BezierGait()
    kin = SpotModel()

    # Trajectory buffers (identity transforms initial)
    T_bf0 = {
        "FL": np.eye(4),
        "FR": np.eye(4),
        "BL": np.eye(4),
        "BR": np.eye(4),
    }
    T_bf = dict(T_bf0)

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

        # ---- Your SNN hook (currently no-op) ----
        StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth, SwingPeriod = \
            snn_override_gait_params(
                state,
                StepLength, LateralFraction, YawRate, StepVelocity,
                ClearanceHeight, PenetrationDepth, SwingPeriod
            )

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

        # Get robot base pose from env (convert to mutable arrays so IK can modify them)
        pos = np.array(env.spot.GetBasePosition(), dtype=float)
        orn = np.array(env.spot.GetBaseOrientation(), dtype=float)

        # IK (from SpotKinematics)
        joint_angles = kin.IK(orn, pos, T_bf)

        # CRITICAL: set env.ja so env.step() won't crash
        env.pass_joint_angles(joint_angles.reshape(-1))

        # Step physics
        state, reward, done, info = env.step(action)

        if done:
            state = env.reset()

    env.close()


if __name__ == "__main__":
    main()
