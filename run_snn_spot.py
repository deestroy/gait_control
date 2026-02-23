# """Run an SNN-driven Spot gait demo.

# This script boots the spot bezier environment, steps a Bezier gait
# generator and uses `SpotModel.IK` to compute joint angles.
# """

# import os
# import sys
# import copy
# import numpy as np
# import matplotlib.pyplot as plt

# REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# SUBMODULE_ROOT = os.path.join(REPO_ROOT, "external", "spot_mini_mini")
# sys.path.insert(0, SUBMODULE_ROOT)

# from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
# from spotmicro.GaitGenerator.Bezier import BezierGait
# from spotmicro.OpenLoopSM.SpotOL import BezierStepper
# from spotmicro.Kinematics.SpotKinematics import SpotModel
# from control.snn_controller import SNNController


# # -----------------------------
# # SNN hook (currently no-op)
# # -----------------------------
# def snn_override_gait_params(
#     state,
#     StepLength, LateralFraction, YawRate, StepVelocity,
#     ClearanceHeight, PenetrationDepth, SwingPeriod
# ):
#     """
#     Later: take IMU + contacts from state and output modulated gait params from Brian2.
#     For now: return inputs unchanged (baseline).
#     """
#     return StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth, SwingPeriod


# def main():
#     env = spotBezierEnv(
#         render=True,
#         on_rack=False,
#         height_field=False,
#         draw_foot_path=False,
#         env_randomizer=None
#     )

#     controller = SNNController(dt_s=float(env._time_step))

#     state = env.reset()

#     spot = SpotModel()
#     T_bf0 = spot.WorldToFoot
#     T_bf = copy.deepcopy(T_bf0)

#     bzg = BezierGait(dt=env._time_step)
#     bz_step = BezierStepper(dt=env._time_step, mode=1)

#     # Placeholder action; env.step() uses internal self.ja (set by pass_joint_angles)
#     try:
#         action = env.action_space.sample()
#     except Exception:
#         action = None

#     contacts = state[-4:]

#     # -------- Baseline gait params (SNN will modulate later) --------
#     StepLength = 0.08
#     LateralFraction = 0.0
#     YawRate = 0.0
#     StepVelocity = 1.2
#     ClearanceHeight = 0.06
#     PenetrationDepth = 0.01
#     SwingPeriod = 0.25  # seconds

#     # If BezierGait supports Tswing
#     if hasattr(bzg, "Tswing"):
#         bzg.Tswing = SwingPeriod

#     # Optional yaw stabilization
#     AUTO_YAW = False
#     P_yaw = 5.0

#     max_timesteps = 20000

#     # Logging buffers
#     time_log = []
#     step_velocity_log = []
#     spike_log = []

#     try:
#         for t in range(max_timesteps):

#             bz_step.ramp_up()
#             pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth = bz_step.StateMachine()

#             # Ensure position/orientation are mutable numeric arrays; some
#             # downstream code does in-place modification of these values.
#             pos = np.array(pos, dtype=float)
#             orn = np.array(orn, dtype=float)

#             # Force baseline walking targets (we'll later replace these with SNN output)
#             StepLength_target = 0.08
#             StepVelocity_target = 1.2
#             ClearanceHeight_target = 0.06
#             PenetrationDepth_target = 0.01

#             StepLength = StepLength_target
#             # StepVelocity = StepVelocity_target
#             StepVelocity, spikes = controller.compute_step_velocity(state)
#             ClearanceHeight = ClearanceHeight_target
#             PenetrationDepth = PenetrationDepth_target

#             # Log data
#             time_log.append(t)
#             step_velocity_log.append(StepVelocity)
#             spike_log.append(spikes)

#             bzg.Tswing = SwingPeriod

#             # Optional SNN hook (commented out for now)
#             # StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth, SwingPeriod = \
#             #     snn_override_gait_params(
#             #         state,
#             #         StepLength, LateralFraction, YawRate, StepVelocity,
#             #         ClearanceHeight, PenetrationDepth, SwingPeriod
#             #     )

#             if hasattr(bzg, "Tswing"):
#                 bzg.Tswing = SwingPeriod

#             if AUTO_YAW:
#                 yaw = env.return_yaw()
#                 YawRate += -yaw * P_yaw

#             contacts = state[-4:]

#             # Generate desired foot trajectories
#             T_bf = bzg.GenerateTrajectory(
#                 StepLength, LateralFraction, YawRate,
#                 StepVelocity, T_bf0, T_bf,
#                 ClearanceHeight, PenetrationDepth,
#                 contacts
#             )

#             # IK (from SpotModel)
#             joint_angles = spot.IK(orn, pos, T_bf)

#             # CRITICAL: set env.ja so env.step() won't crash
#             env.pass_joint_angles(joint_angles.reshape(-1))

#             # Step physics
#             state, reward, done, info = env.step(action)
#             if t % 50 == 0:
#                 print(f"t={t} roll={state[0]:+.3f} pitch={state[1]:+.3f} StepVel={StepVelocity:.2f} StepLen={StepLength:.2f}")

#             if done:
#                 state = env.reset()
#     finally:
#         env.close()
        
#         np.savez(
#             "results_step_velocity_spikes.npz",
#             time=np.array(time_log),
#             step_velocity=np.array(step_velocity_log),
#             spikes=np.array(spike_log),
#         )
#         print("Saved results_step_velocity_spikes.npz")




# if __name__ == "__main__":
#     main()
