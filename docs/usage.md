# Running DC and PMSM experiments

Install the project from the source checkout with `python -m pip install -e .`. Run commands from the workspace containing your `RL_models/` and `outputs/`, or set `RLLIB_CONTROL_WORKSPACE` to that directory. `python -m rllib_control` is equivalent to the `rllib-control` entry point.

## Classical baselines

```sh
rllib-control results --motor dc --controller pi --show
rllib-control results --motor pmsm --controller foc --show
```

The baseline gains and environment settings are packaged in `src/rllib_control/data/`. DC PI was selected for a specified nominal reversal overshoot target; PMSM PI/FOC was tuned on validation seeds 501/502. Neither command trains or needs a model checkpoint. Omit `--show` for headless use.

## Fresh PPO and PPO + ICM

The commands below are **smoke runs**, not research-grade models. Use a new run name and a planned interaction budget for actual experiments. Both members of each pair use the same external reward, physical constraints, reference generation, and training budget.

```sh
rllib-control train --motor dc --controller ppo --run dc_smoke_ppo --iterations 1 --num-env-runners 0 --train-batch-size 512 --minibatch-size 128 --num-epochs 1
rllib-control train --motor dc --controller icm --run dc_smoke_icm --iterations 1 --num-env-runners 0 --train-batch-size 512 --minibatch-size 128 --num-epochs 1
rllib-control train --motor pmsm --controller ppo --run pmsm_smoke_ppo --iterations 1 --num-env-runners 0 --train-batch-size 512 --minibatch-size 128 --num-epochs 1 --stability-profile pmsm-v1 --tracking-absolute-weight 20 --current-rate-weight 0.002
rllib-control train --motor pmsm --controller icm --run pmsm_smoke_icm --iterations 1 --num-env-runners 0 --train-batch-size 512 --minibatch-size 128 --num-epochs 1 --stability-profile pmsm-v1 --tracking-absolute-weight 20 --current-rate-weight 0.002
```

Check settings without training by appending `--dry-run`. Each run writes to `RL_models/<motor>/<controller>/<run>/`. The default evaluation profile is `mixed`; training still uses its configured random reference generator.

```sh
# One learned model
rllib-control results --motor dc --controller ppo --ppo-run dc_smoke_ppo
rllib-control results --motor pmsm --controller icm --icm-run pmsm_smoke_icm

# Compare paired checkpoints with the corresponding frozen baseline
rllib-control results --motor dc --ppo-run dc_smoke_ppo --icm-run dc_smoke_icm
rllib-control results --motor pmsm --ppo-run pmsm_smoke_ppo --icm-run pmsm_smoke_icm
```

If you change the environment, freeze a matching baseline with `rllib-control tune-baseline` and pass its JSON via `--baseline-config`. Comparisons reject inconsistent checkpoint environments instead of silently substituting a default.

## Existing thesis runs

The preset DC runs are `tdk_v1_main_ppo_seed123` and `tdk_v1_main_icm_reduced_seed123` (1000 iterations); the PMSM presets are `tdk_confirm700_v1_ppo_seed456` and `tdk_confirm700_v1_icm_fixed_seed456` (700 iterations). These checkpoints are not distributed with the source. If they are already in your local workspace, run:

```sh
rllib-control results --motor dc --show
rllib-control results --motor pmsm --show
```

Each learned run needs its `manifest.json` and the final `iteration_XXXX/` directory. Keep the original folder and manifest together. Only load checkpoints you trust: serialized models are executable input, not ordinary data files.

Use `--profile steps|random|sine|mixed` (one value), `--seed`, and `--scenario` for controlled evaluation. Available disturbances include load steps, parameter shifts, resistance jumps, and sensor noise. A nominal mixed demo is not a robustness study. A single-seed comparison does not establish general ICM benefit.

## Research and figures

```sh
python -m rllib_control.research --help
python -m rllib_control.validate --help
python -m rllib_control.tdk_extension --help
rllib-control thesis-figures --help
```

The thesis exporter takes a saved batch with `evidence/<motor>/<case>` via `--input`. It does not train, simulate, or smooth the data. Historical local protocols are source-fingerprinted; do not resume them under changed code. Create a new protocol for new measurements. Keep validation and held-out test seeds separate, and report incomplete and failed cases.
