# RLlib Drive Control

Python simulations for comparing classical PI/FOC, PPO, and PPO with intrinsic curiosity (ICM) on brushed DC and permanent-magnet synchronous motor (PMSM) speed tracking.

The learned controller produces a continuous current reference. A shared inner PI/FOC loop handles the voltage command and physical limits. The classical comparator replaces the **outer** learned policy with speed PI. This is a simulation research project, not a hardware controller or a formal stability guarantee.

## Prior work

[DC_PMSM_simulation](https://github.com/Laxi03/DC_PMSM_simulation) contains the previous semester’s TD3 research. This repository is a separate study focused on PPO and the effect of adding ICM.

## Quick start

Use Python 3.12. From the repository directory on macOS or Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .

# Run the classical baselines; no checkpoints or private data required.
rllib-control results --motor dc --controller pi
rllib-control results --motor pmsm --controller foc
```

On Linux, to use CPU-only PyTorch, install it in the environment before installing this project:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

Both demos use the **mixed** reference by default and save a report, tracking plots, a current histogram, and raw signals under `outputs/final_results/`. Add `--show` to open figures locally. Reward and t-SNE plots are optional; t-SNE requires `python -m pip install '.[tsne]'`.

## Train and compare

See the [usage guide](docs/usage.md) for separate PPO/ICM training, learned-policy evaluation, classical baselines, disturbances, and the existing thesis checkpoints. Models and generated outputs are intentionally not in Git. Train fresh models or provide trusted checkpoints explicitly; the PI/FOC demos work without them.

```sh
rllib-control --help
rllib-control results --help
```

## Repository layout

```text
src/rllib_control/  CLI, environments, control, evaluation and research protocols
src/sim_*           motor simulators; historical import names retained for checkpoints
tests/              self-contained unit and baseline smoke tests
docs/               usage, reproducibility and GitHub publishing
scripts/            allowlisted public source export
.github/workflows/  wheel build and test workflow
```

Run from your experiment directory or set `RLLIB_CONTROL_WORKSPACE` to an explicit path. Checkpoints and outputs live in that workspace, never inside the installed Python package.

## Development and publication

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

See [contributing](CONTRIBUTING.md) and the [GitHub publishing checklist](docs/publishing.md). Cite this software using [CITATION.cff](CITATION.cff). Source code and repository documentation are released under the [MIT License](LICENSE).
