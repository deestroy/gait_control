# File Structure
```quad_snn_cpg/
  sim/
    pybullet_env.py          # loads URDF, steps physics, exposes sensors + applies torques
    spotmicro_assets/        # URDF + meshes
  snn/
    brian2_cpg.py            # CPG core network definition (E/I per leg, coupling)
    brian2_feedback.py       # sensory spike encoding + connections to CPG
    brian2_stdp.py           # STDP rule + training protocol toggles
    io_encoding.py           # sensor->spike encoders, spike->torque decoders
  control/
    controller.py            # orchestrates: get sensors -> encode -> run SNN -> decode -> torques
    baselines.py             # PID / classical CPG baseline for comparisons
  experiments/
    exp1_gait.py             # stable trot
    exp2_push_recovery.py    # perturbation tests
    exp3_efficiency.py       # compute cost/spike sparsity
  results/
    figures/
    logs/
  paper/
    ieee_template/
```
