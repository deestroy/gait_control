#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMODULE_ROOT = os.path.join(REPO_ROOT, "external", "spot_mini_mini")
sys.path.insert(0, SUBMODULE_ROOT)

from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.util.gui import GUI
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.Kinematics.LieAlgebra import RPY
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.spot_env_randomizer import SpotEnvRandomizer
from control.snn_controller import SNNController

# TESTING
from spotmicro.OpenLoopSM.SpotOL import BezierStepper

import time
import os

import argparse

# ARGUMENTS
descr = "Spot Mini Mini Environment Tester (No Joystick)."
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("-hf",
                    "--HeightField",
                    help="Use HeightField",
                    action='store_true')
parser.add_argument("-r",
                    "--DebugRack",
                    help="Put Spot on an Elevated Rack",
                    action='store_true')
parser.add_argument("-p",
                    "--DebugPath",
                    help="Draw Spot's Foot Path",
                    action='store_true')
parser.add_argument("-ay",
                    "--AutoYaw",
                    help="Automatically Adjust Spot's Yaw",
                    action='store_true')
parser.add_argument("-ar",
                    "--AutoReset",
                    help="Automatically Reset Environment When Spot Falls",
                    action='store_true')
parser.add_argument("-dr",
                    "--DontRandomize",
                    help="Do NOT Randomize State and Environment.",
                    action='store_true')
ARGS = parser.parse_args()


def main():
    """ The main() function. """

    print("STARTING SPOT TEST ENV")
    seed = 0
    max_timesteps = 4e6

    # Find abs path to this file
    my_path = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(my_path, "../results")
    models_path = os.path.join(my_path, "../models")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if ARGS.DebugRack:
        on_rack = True
    else:
        on_rack = False

    if ARGS.DebugPath:
        draw_foot_path = True
    else:
        draw_foot_path = False

    if ARGS.HeightField:
        height_field = True
    else:
        height_field = False

    if ARGS.DontRandomize:
        env_randomizer = None
    else:
        env_randomizer = SpotEnvRandomizer()

    env = spotBezierEnv(render=True,
                        on_rack=on_rack,
                        height_field=height_field,
                        draw_foot_path=draw_foot_path,
                        env_randomizer=env_randomizer)

    # Set seeds
    env.seed(seed)
    np.random.seed(seed)

    controller = SNNController(dt_s=float(env._time_step))

    state_dim = env.observation_space.shape[0]
    print("STATE DIM: {}".format(state_dim))
    action_dim = env.action_space.shape[0]
    print("ACTION DIM: {}".format(action_dim))
    max_action = float(env.action_space.high[0])

    state = env.reset()

    g_u_i = GUI(env.spot.quadruped)

    spot = SpotModel()
    T_bf0 = spot.WorldToFoot
    T_bf = copy.deepcopy(T_bf0)

    bzg = BezierGait(dt=env._time_step)

    bz_step = BezierStepper(dt=env._time_step, mode=0)

    action = env.action_space.sample()

    FL_phases = []
    FR_phases = []
    BL_phases = []
    BR_phases = []

    FL_Elbow = []

    yaw = 0.0

    print("STARTED SPOT TEST ENV")
    t = 0
    time_log = []
    step_velocity_sm_log = []
    step_velocity_final_log = []
    spike_log = []
    dv_log = []
    try:
        while t < (int(max_timesteps)):

            bz_step.ramp_up()

            pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth = bz_step.StateMachine(
            )

            # --- SNN modulation (small residual on top of StateMachine velocity) ---
            StepVelocity_snn, spikes = controller.compute_step_velocity(state)
            dv = StepVelocity_snn - controller.base_vel

            # Convert absolute SNN velocity to a delta around base_vel
            StepVelocity_sm = StepVelocity  # from StateMachine

            k = 0.10  # residual strength (0.05–0.20 typical)
            StepVelocity = float(np.clip(StepVelocity_sm + k * dv, 0.9, 1.5))
                        
            bz_step.StepVelocity = StepVelocity

            time_log.append(t)
            step_velocity_sm_log.append(StepVelocity_sm)
            step_velocity_final_log.append(StepVelocity)
            spike_log.append(spikes)
            dv_log.append(dv)

            # # Apply residual + clamp (tight bounds to keep it smooth)
            # StepVelocity = float(np.clip(StepVelocity + dv, 0.8, 1.6))
            # Update Swing Period
            bzg.Tswing = 0.25

            yaw = env.return_yaw()

            P_yaw = 5.0

            if ARGS.AutoYaw:
                YawRate += -yaw * P_yaw

            # print("YAW RATE: {}".format(YawRate))

            # TEMP
            bz_step.StepLength = StepLength
            bz_step.LateralFraction = LateralFraction
            bz_step.YawRate = YawRate

            contacts = state[-4:]

            FL_phases.append(env.spot.LegPhases[0])
            FR_phases.append(env.spot.LegPhases[1])
            BL_phases.append(env.spot.LegPhases[2])
            BR_phases.append(env.spot.LegPhases[3])

            # Get Desired Foot Poses
            T_bf = bzg.GenerateTrajectory(StepLength, LateralFraction, YawRate,
                                        StepVelocity, T_bf0, T_bf,
                                        ClearanceHeight, PenetrationDepth,
                                        contacts)
            joint_angles = spot.IK(orn, pos, T_bf)

            FL_Elbow.append(np.degrees(joint_angles[0][-1]))

            # for i, (key, Tbf_in) in enumerate(T_bf.items()):
            #     print("{}: \t Angle: {}".format(key, np.degrees(joint_angles[i])))
            # print("-------------------------")

            env.pass_joint_angles(joint_angles.reshape(-1))
            # Get External Observations
            env.spot.GetExternalObservations(bzg, bz_step)
            # Step
            state, reward, done, _ = env.step(action)
            # print("IMU Roll: {}".format(state[0]))
            # print("IMU Pitch: {}".format(state[1]))
            # print("IMU GX: {}".format(state[2]))
            # print("IMU GY: {}".format(state[3]))
            # print("IMU GZ: {}".format(state[4]))
            # print("IMU AX: {}".format(state[5]))
            # print("IMU AY: {}".format(state[6]))
            # print("IMU AZ: {}".format(state[7]))
            # print("-------------------------")
            if done:
                print("DONE")
                if ARGS.AutoReset:
                    env.reset()
                    # plt.plot()
                    # # plt.plot(FL_phases, label="FL")
                    # # plt.plot(FR_phases, label="FR")
                    # # plt.plot(BL_phases, label="BL")
                    # # plt.plot(BR_phases, label="BR")
                    # plt.plot(FL_Elbow, label="FL ELbow (Deg)")
                    # plt.xlabel("dt")
                    # plt.ylabel("value")
                    # plt.title("Leg Phases")
                    # plt.legend()
                    # plt.show()

            # time.sleep(1.0)

            t += 1
        print(joint_angles)
    finally:
        np.savez(
            "results_sm_plus_snn.npz",
            time=np.array(time_log),
            step_velocity_sm=np.array(step_velocity_sm_log),
            step_velocity=np.array(step_velocity_final_log),
            spikes=np.array(spike_log),
            dv=np.array(dv_log)
        )
        env.close()


if __name__ == '__main__':
    main()
