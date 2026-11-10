from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pmsm_curiosity_mpl_cache")
)

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from gymnasium import spaces
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import DEFAULT_MODULE_ID, Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.learners.classes.intrinsic_curiosity_learners import (
    ICM_MODULE_ID,
    PPOTorchLearnerWithCuriosity,
)
from ray.rllib.examples.rl_modules.classes.intrinsic_curiosity_model_rlm import (
    IntrinsicCuriosityModel,
)
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN
from ray.tune.registry import register_env

plt.style.use("seaborn-v0_8-whitegrid")

ENV_NAME = "pmsm_speed_curiosity_v0"
from rllib_control.workspace import workspace_root

SCRIPT_DIR = workspace_root()
MODEL_ROOT = SCRIPT_DIR / "RL_models" / "pmsm_experiments"
TRAIN_ROOT = SCRIPT_DIR / "outputs" / "pmsm_training"
EVAL_ROOT = SCRIPT_DIR / "outputs" / "pmsm_evaluations"
COMPARE_ROOT = SCRIPT_DIR / "outputs" / "pmsm_comparisons"


@dataclass
class PMSMParams:
    R_pp: float = 10.8
    Ld_pp: float = 1.61e-3
    Lq_pp: float = 1.61e-3
    kT: float = 22.4e-3
    J: float = 2.0e-5
    B: float = 5.5e-6
    pole_pairs: int = 4
    Vdc: float = 24.0

    def phase_values(self) -> tuple[float, float, float, float]:
        resistance = self.R_pp / 2.0
        ld = self.Ld_pp / 2.0
        lq = self.Lq_pp / 2.0
        flux = self.kT / (1.5 * self.pole_pairs)
        return resistance, ld, lq, flux


@dataclass
class PMSMState:
    id: float = 0.0
    iq: float = 0.0
    omega: float = 0.0
    theta_e: float = 0.0


class PMSMSpeedReference:
    SUPPORTED = ("random", "steps", "sine", "mixed", "random_full")

    def __init__(
        self,
        profile: str,
        duration: float,
        rng: np.random.Generator,
        amplitude_scale: float = 1.0,
    ):
        if profile not in self.SUPPORTED:
            raise ValueError(f"Unsupported reference profile: {profile}")
        if not 0.0 < amplitude_scale <= 1.0:
            raise ValueError("amplitude_scale must be in (0, 1].")
        self.profile = profile
        self.duration = float(duration)
        self.rng = rng
        self.scale = float(amplitude_scale)
        self.times = np.array([0.0, self.duration])
        self.values = np.zeros(2)
        if profile == "random":
            n = int(rng.integers(6, 11))
            durations = rng.uniform(0.4, 1.1, n)
            interior = np.cumsum(durations)
            self.times = np.concatenate(
                ([0.0], interior[interior < self.duration], [self.duration])
            )
            self.values = rng.uniform(-400.0, 400.0, len(self.times))
            for index in range(1, len(self.values)):
                self.values[index] = (
                    0.6 * self.values[index - 1] + 0.4 * self.values[index]
                )
        elif profile == "steps":
            self.times = np.linspace(0.0, self.duration, 9)
            self.values = np.array(
                [0.0, 200.0, -200.0, 350.0, 100.0, -350.0, 300.0, 0.0, 0.0]
            )
        elif profile == 'random_full':
            n=int(rng.integers(7,12))
            interior=np.cumsum(rng.uniform(.7,1.3,n))
            self.times=np.concatenate(([0.],interior[interior<self.duration],[self.duration]))
            self.values=rng.uniform(-400.,400.,len(self.times))
            self.values[0]=0.
            self.values[-1]=self.values[-2]
            # Independent, unsmoothed targets: large reversals are possible.
            # No test-profile levels or held-out seeds are encoded here.

    def value(self, time_s: float) -> float:
        time_s = float(min(max(time_s, 0.0), self.duration))
        if self.profile == "sine":
            raw = 350.0 * math.sin(2.0 * math.pi * time_s / self.duration)
        elif self.profile == "mixed":
            phase = time_s / max(self.duration, 1e-12)
            if phase < 0.25:
                raw = 300.0 if phase < 0.125 else -300.0
            elif phase < 0.5:
                raw = 300.0 - 600.0 * ((phase - 0.25) / 0.25)
            elif phase < 0.75:
                raw = 350.0 * math.sin(2.0 * math.pi * (phase - 0.5) / 0.25)
            else:
                raw = 200.0 if phase < 0.875 else -200.0
        else:
            index = int(np.searchsorted(self.times, time_s, side="right") - 1)
            raw = float(self.values[min(max(index, 0), len(self.values) - 1)])
        return float(self.scale * raw)


class PMSMSpeedCuriosityEnv(gym.Env[np.ndarray, int]):
    """PMSM speed control with a discrete 7x7 dq-voltage vector map.

    Observation: omega, id, iq, vd, vq, reference, error, integrated error,
    and load torque. All values are normalized.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any] | None = None):
        super().__init__()
        cfg = dict(env_config or {})
        self.params = PMSMParams()
        self.dt = float(cfg.get("dt", 5e-5))
        self.frame_skip = int(cfg.get("frame_skip", 20))
        self.episode_length = float(cfg.get("episode_length", 6.0))
        self.max_steps = max(
            1, int(math.ceil(self.episode_length / (self.dt * self.frame_skip)))
        )
        self.profile_name = str(cfg.get("profile", "random"))
        self.reference_scale = float(cfg.get("reference_scale", 1.0))
        self.training_mode = bool(cfg.get("training_mode", True))
        self.domain_randomization = bool(cfg.get("domain_randomization", True))
        self.paired_domain_rng = bool(cfg.get("paired_domain_rng", False))
        self.parameter_jump = bool(cfg.get("parameter_jump", True))
        self.base_seed = int(cfg.get("seed", 0))
        self.speed_limit = float(cfg.get("speed_limit", 600.0))
        self.current_limit = float(cfg.get("current_limit", 3.0))
        self.num_axis_levels = int(cfg.get("num_axis_levels", 7))
        if self.num_axis_levels < 3 or self.num_axis_levels % 2 == 0:
            raise ValueError("num_axis_levels must be an odd integer >= 3.")

        self.Vmax_nominal = self.params.Vdc / math.sqrt(3.0)
        normalized_levels = np.linspace(-1.0, 1.0, self.num_axis_levels)
        self.action_vectors = np.asarray(
            [(vd, vq) for vd in normalized_levels for vq in normalized_levels],
            dtype=np.float64,
        )
        magnitudes = np.linalg.norm(self.action_vectors, axis=1)
        outside = magnitudes > 1.0
        self.action_vectors[outside] /= magnitudes[outside, None]
        self.action_space = spaces.Discrete(len(self.action_vectors))
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(9,), dtype=np.float32
        )

        self.rng = np.random.default_rng(self.base_seed)
        self.state = PMSMState()
        self.reference = PMSMSpeedReference(
            self.profile_name, self.episode_length, self.rng, self.reference_scale
        )
        self.R_nom, self.Ld_nom, self.Lq_nom, self.flux_nom = (
            self.params.phase_values()
        )
        self.R = self.R_nom
        self.Ld = self.Ld_nom
        self.Lq = self.Lq_nom
        self.flux = self.flux_nom
        self.J = self.params.J
        self.load_torque = 0.0
        self.vd = 0.0
        self.vq = 0.0
        self.prev_vd = 0.0
        self.prev_vq = 0.0
        self.error_integral = 0.0
        self.time = 0.0
        self.step_count = 0
        self.last_reference = 0.0
        self.jump_time: float | None = None
        self.jump_factor = 1.0
        self.jump_applied = False
        self.safety_violation = False

    @property
    def control_period(self) -> float:
        return self.dt * self.frame_skip

    def _configure_episode(self) -> None:
        self.R = self.R_nom
        self.Ld = self.Ld_nom
        self.Lq = self.Lq_nom
        self.flux = self.flux_nom
        self.J = self.params.J
        self.load_torque = 0.006
        if self.training_mode and (self.domain_randomization or self.paired_domain_rng):
            scales=[float(self.rng.uniform(0.85, 1.15)),float(self.rng.uniform(0.85, 1.20)),
                    float(self.rng.uniform(0.25, 1.5))]
            if self.domain_randomization:
                self.R *= scales[0]
                self.J *= scales[1]
                self.load_torque *= scales[2]
        self.jump_time = None
        self.jump_factor = 1.0
        self.jump_applied = False
        if self.training_mode and self.parameter_jump and self.rng.random() < 0.5:
            self.jump_time = float(
                self.rng.uniform(0.2 * self.episode_length, 0.8 * self.episode_length)
            )
            self.jump_factor = float(self.rng.uniform(0.75, 1.35))

    def _derivatives(
        self, state: PMSMState, vd: float, vq: float
    ) -> PMSMState:
        electrical_speed = self.params.pole_pairs * state.omega
        did = (
            vd - self.R * state.id + electrical_speed * self.Lq * state.iq
        ) / self.Ld
        diq = (
            vq
            - self.R * state.iq
            - electrical_speed * (self.Ld * state.id + self.flux)
        ) / self.Lq
        torque = 1.5 * self.params.pole_pairs * (
            self.flux * state.iq
            + (self.Ld - self.Lq) * state.id * state.iq
        )
        domega = (
            torque - self.load_torque - self.params.B * state.omega
        ) / self.J
        return PMSMState(did, diq, domega, electrical_speed)

    def _rk4(self, state: PMSMState, vd: float, vq: float) -> PMSMState:
        def add(x: PMSMState, k: PMSMState, scale: float) -> PMSMState:
            return PMSMState(
                x.id + scale * k.id,
                x.iq + scale * k.iq,
                x.omega + scale * k.omega,
                x.theta_e + scale * k.theta_e,
            )

        k1 = self._derivatives(state, vd, vq)
        k2 = self._derivatives(add(state, k1, self.dt / 2.0), vd, vq)
        k3 = self._derivatives(add(state, k2, self.dt / 2.0), vd, vq)
        k4 = self._derivatives(add(state, k3, self.dt), vd, vq)
        return PMSMState(
            state.id + self.dt * (k1.id + 2 * k2.id + 2 * k3.id + k4.id) / 6,
            state.iq + self.dt * (k1.iq + 2 * k2.iq + 2 * k3.iq + k4.iq) / 6,
            state.omega
            + self.dt * (k1.omega + 2 * k2.omega + 2 * k3.omega + k4.omega) / 6,
            state.theta_e
            + self.dt
            * (k1.theta_e + 2 * k2.theta_e + 2 * k3.theta_e + k4.theta_e)
            / 6,
        )

    def _observation(self) -> np.ndarray:
        error = (self.last_reference - self.state.omega) / self.speed_limit
        return np.asarray(
            [
                np.clip(self.state.omega / self.speed_limit, -1, 1),
                np.clip(self.state.id / self.current_limit, -1, 1),
                np.clip(self.state.iq / self.current_limit, -1, 1),
                np.clip(self.vd / self.Vmax_nominal, -1, 1),
                np.clip(self.vq / self.Vmax_nominal, -1, 1),
                np.clip(self.last_reference / self.speed_limit, -1, 1),
                np.clip(error, -1, 1),
                np.clip(self.error_integral, -1, 1),
                np.clip(self.load_torque / 0.012, -1, 1),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._configure_episode()
        self.reference = PMSMSpeedReference(
            self.profile_name, self.episode_length, self.rng, self.reference_scale
        )
        self.state = PMSMState()
        self.vd = self.vq = self.prev_vd = self.prev_vq = 0.0
        self.error_integral = 0.0
        self.time = 0.0
        self.step_count = 0
        self.last_reference = self.reference.value(0.0)
        self.safety_violation = False
        return self._observation(), self._info(0.0)

    def step(self, action: int):
        action_index = int(np.asarray(action).item())
        if not self.action_space.contains(action_index):
            raise ValueError(f"Invalid action: {action_index}")
        vd_target, vq_target = (
            self.action_vectors[action_index] * self.Vmax_nominal
        )
        return self.step_voltage(np.array([vd_target, vq_target]))

    def step_voltage(self, voltage):
        """Integrate a physical dq command projected onto the inverter disk."""
        voltage = np.asarray(voltage, dtype=float)
        voltage = voltage * min(1.0, self.Vmax_nominal / max(np.linalg.norm(voltage), 1e-12))
        vd_target, vq_target = voltage
        reward_sum = 0.0
        for _ in range(self.frame_skip):
            if hasattr(self, "before_substep_callback"):
                self.before_substep_callback()
            if (
                not self.jump_applied
                and self.jump_time is not None
                and self.time >= self.jump_time
            ):
                self.R *= self.jump_factor
                self.jump_applied = True
            if hasattr(self, "current_control_callback"):
                vd_target, vq_target = self.current_control_callback()
            alpha = min(max(self.dt / 5e-4, 0.0), 1.0)
            self.vd += alpha * (vd_target - self.vd)
            self.vq += alpha * (vq_target - self.vq)
            self.state = self._rk4(self.state, self.vd, self.vq)
            self.time += self.dt
            self.last_reference = self.reference.value(self.time)
            if hasattr(self, "after_substep_callback"):
                self.after_substep_callback()
            error = (self.last_reference - self.state.omega) / self.speed_limit
            self.error_integral = float(
                min(max(self.error_integral + error * self.dt, -0.5), 0.5)
            )
            current_norm_sq = (
                self.state.id**2 + self.state.iq**2
            ) / self.current_limit**2
            voltage_norm_sq = (self.vd**2 + self.vq**2) / self.Vmax_nominal**2
            delta_voltage_sq = (
                (self.vd - self.prev_vd) ** 2 + (self.vq - self.prev_vq) ** 2
            ) / self.Vmax_nominal**2
            raw_reward = (
                -20.0 * error**2
                - 0.015 * current_norm_sq
                - 0.01 * voltage_norm_sq
                - getattr(self, "substep_smoothness_weight", 0.01) * delta_voltage_sq
                - 0.1 * self.error_integral**2
                + (0.1 if abs(error) < 0.01 else 0.0)
            )
            reward_sum += (self.reward_callback() if hasattr(self, "reward_callback")
                           else float(min(max(raw_reward, -1.0), 0.5)))
            self.prev_vd, self.prev_vq = self.vd, self.vq
            if (
                not np.isfinite(self.state.omega)
                or not np.isfinite(self.state.id)
                or not np.isfinite(self.state.iq)
                or abs(self.state.omega) > 1.25 * self.speed_limit
                or math.hypot(self.state.id, self.state.iq) > 1.5 * self.current_limit
            ):
                self.safety_violation = True
                break
        self.step_count += 1
        terminated = self.safety_violation
        truncated = self.step_count >= self.max_steps or self.time >= self.episode_length
        reward = -5.0 if terminated else reward_sum / self.frame_skip
        return self._observation(), reward, terminated, truncated, self._info(reward)

    def _info(self, reward: float) -> dict[str, Any]:
        return {
            "time": self.time,
            "omega": self.state.omega,
            "id": self.state.id,
            "iq": self.state.iq,
            "current_magnitude": math.hypot(self.state.id, self.state.iq),
            "vd": self.vd,
            "vq": self.vq,
            "voltage_magnitude": math.hypot(self.vd, self.vq),
            "reference": self.last_reference,
            "error": self.last_reference - self.state.omega,
            "reward": reward,
            "safety_violation": self.safety_violation,
        }


class PMSMPhysicalStateICM(IntrinsicCuriosityModel):
    """ICM sees physical state/action history but not ref/error/integral."""

    def _forward_train(self, batch: dict[str, torch.Tensor], **kwargs: Any):
        masked = dict(batch)
        observations = batch[Columns.OBS].clone()
        next_observations = batch[Columns.NEXT_OBS].clone()
        observations[..., 5:8] = 0.0
        next_observations[..., 5:8] = 0.0
        masked[Columns.OBS] = observations
        masked[Columns.NEXT_OBS] = next_observations
        return super()._forward_train(masked, **kwargs)


def register_pmsm_environment() -> None:
    register_env(ENV_NAME, lambda config: PMSMSpeedCuriosityEnv(dict(config)))


def make_config(args: argparse.Namespace) -> PPOConfig:
    env_config = {
        "seed": args.seed,
        "episode_length": args.episode_length,
        "profile": "random",
        "reference_scale": args.train_reference_scale,
        "training_mode": True,
        "domain_randomization": not args.no_domain_randomization,
        "parameter_jump": not args.no_parameter_jump,
        "num_axis_levels": args.num_axis_levels,
    }
    return (
        PPOConfig()
        .environment(ENV_NAME, env_config=env_config)
        .framework("torch")
        .debugging(seed=args.seed, log_level="ERROR", log_sys_usage=False)
        .env_runners(num_env_runners=0, batch_mode="truncate_episodes")
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,
            learner_class=PPOTorchLearnerWithCuriosity,
            learner_config_dict={
                "intrinsic_reward_coeff": args.intrinsic_reward_coeff,
                "forward_loss_weight": args.forward_loss_weight,
            },
        )
        .training(
            gamma=0.99,
            lambda_=0.95,
            lr=3e-4,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.005,
            grad_clip=10.0,
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    DEFAULT_MODULE_ID: RLModuleSpec(
                        model_config={
                            "fcnet_hiddens": list(args.policy_hidden_sizes),
                            "fcnet_activation": "tanh",
                            "vf_share_layers": False,
                        }
                    ),
                    ICM_MODULE_ID: RLModuleSpec(
                        module_class=PMSMPhysicalStateICM,
                        learner_only=True,
                        model_config={
                            "feature_dim": 16,
                            "feature_net_hiddens": (32, 16),
                            "feature_net_activation": "relu",
                            "inverse_net_hiddens": (32, 16),
                            "inverse_net_activation": "relu",
                            "forward_net_hiddens": (32, 16),
                            "forward_net_activation": "relu",
                        },
                    ),
                }
            ),
            algorithm_config_overrides_per_module={
                ICM_MODULE_ID: AlgorithmConfig.overrides(lr=1e-3)
            },
        )
    )


def _find_numeric(data: Any, key: str) -> float | None:
    if isinstance(data, dict):
        if key in data and isinstance(data[key], (int, float, np.number)):
            return float(data[key])
        for value in data.values():
            found = _find_numeric(value, key)
            if found is not None:
                return found
    return None


def _episode_return(result: dict[str, Any]) -> float:
    env_results = result.get(ENV_RUNNER_RESULTS, {})
    value = env_results.get(EPISODE_RETURN_MEAN)
    return float("nan") if value is None else float(value)


def _intrinsic_coeff(algorithm: Algorithm) -> float:
    config = getattr(algorithm, "config", None)
    learner = getattr(config, "learner_config_dict", {}) or {}
    return float(learner.get("intrinsic_reward_coeff", 0.0))


def _policy_action(module: torch.nn.Module, observation: np.ndarray) -> int:
    device = next(module.parameters()).device
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)[None]
    with torch.no_grad():
        output = module.forward_inference({Columns.OBS: tensor})
    return int(torch.argmax(output[Columns.ACTION_DIST_INPUTS], dim=-1).item())


def save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    if not history:
        return
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    iteration = [row["iteration"] for row in history]
    axes[0].plot(iteration, [row["return"] for row in history])
    axes[0].set_ylabel("Extrinsic return")
    axes[1].plot(iteration, [row["intrinsic"] for row in history])
    axes[1].set_ylabel("Mean intrinsic reward")
    axes[1].set_xlabel("Training iteration")
    fig.suptitle("PMSM PPO training")
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=180)
    plt.close(fig)


def train(args: argparse.Namespace) -> Path:
    register_pmsm_environment()
    run_name = args.run_name or (
        f"pmsm_{'curiosity' if args.intrinsic_reward_coeff > 0 else 'ppo'}_"
        f"{args.iterations}_seed{args.seed}"
    )
    model_root = (
        Path(args.model_dir).resolve() if args.model_dir else MODEL_ROOT / run_name
    )
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else TRAIN_ROOT / run_name
    )
    if (model_root.exists() and any(model_root.iterdir())) or (
        output_dir / "training_config.json"
    ).exists():
        raise FileExistsError(f"Run already exists: {run_name}")
    model_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "motor": asdict(PMSMParams()),
                "run_name": run_name,
                "iterations": args.iterations,
                "seed": args.seed,
                "policy_hidden_sizes": args.policy_hidden_sizes,
                "train_reference_scale": args.train_reference_scale,
                "evaluation_reference_scale": args.evaluation_reference_scale,
                "intrinsic_reward_coeff": args.intrinsic_reward_coeff,
                "forward_loss_weight": args.forward_loss_weight,
                "num_axis_levels": args.num_axis_levels,
                "action_count": args.num_axis_levels**2,
            },
            stream,
            indent=2,
        )
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        logging_level="ERROR",
        num_cpus=1,
    )
    algorithm = make_config(args).build_algo()
    history: list[dict[str, Any]] = []
    latest = model_root
    try:
        for iteration in range(1, args.iterations + 1):
            result = algorithm.train()
            intrinsic = _find_numeric(result, "mean_intrinsic_rewards")
            history.append(
                {
                    "iteration": iteration,
                    "return": _episode_return(result),
                    "intrinsic": 0.0 if intrinsic is None else intrinsic,
                    "steps": int(result.get("num_env_steps_sampled_lifetime", 0)),
                }
            )
            print(
                f"iteration={iteration:04d} return={history[-1]['return']: .5f} "
                f"intrinsic={history[-1]['intrinsic']:.6f}"
            )
            if iteration % args.checkpoint_frequency == 0 or iteration == args.iterations:
                latest = model_root / f"iteration_{iteration:04d}"
                algorithm.save_to_path(latest)
                save_history(history, output_dir)
    finally:
        algorithm.stop()
        ray.shutdown()
    print(f"Saved PMSM checkpoint: {latest}")
    return latest


def run_episode(
    module: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, float | bool]]:
    env = PMSMSpeedCuriosityEnv(
        {
            "seed": args.seed,
            "episode_length": args.episode_length,
            "profile": args.profile,
            "reference_scale": args.evaluation_reference_scale,
            "training_mode": False,
            "domain_randomization": False,
            "parameter_jump": False,
            "num_axis_levels": args.num_axis_levels,
        }
    )
    observation, _ = env.reset(seed=args.seed)
    logs: dict[str, list[float]] = {
        key: []
        for key in (
            "time_s",
            "reference_rad_s",
            "omega_rad_s",
            "id_A",
            "iq_A",
            "current_magnitude_A",
            "vd_V",
            "vq_V",
            "voltage_magnitude_V",
            "reward",
        )
    }
    done = False
    while not done:
        action = _policy_action(module, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        logs["time_s"].append(info["time"])
        logs["reference_rad_s"].append(info["reference"])
        logs["omega_rad_s"].append(info["omega"])
        logs["id_A"].append(info["id"])
        logs["iq_A"].append(info["iq"])
        logs["current_magnitude_A"].append(info["current_magnitude"])
        logs["vd_V"].append(info["vd"])
        logs["vq_V"].append(info["vq"])
        logs["voltage_magnitude_V"].append(info["voltage_magnitude"])
        logs["reward"].append(reward)
        done = terminated or truncated
    arrays = {key: np.asarray(value) for key, value in logs.items()}
    error = arrays["reference_rad_s"] - arrays["omega_rad_s"]
    dt = env.control_period
    metrics: dict[str, float | bool] = {
        "episode_return_extrinsic": float(np.sum(arrays["reward"])),
        "MAE_speed_rad_s": float(np.mean(np.abs(error))),
        "RMSE_speed_rad_s": float(np.sqrt(np.mean(error**2))),
        "max_abs_speed_error_rad_s": float(np.max(np.abs(error))),
        "IAE_speed": float(np.sum(np.abs(error)) * dt),
        "ISE_speed": float(np.sum(error**2) * dt),
        "mean_current_magnitude_A": float(np.mean(arrays["current_magnitude_A"])),
        "max_current_magnitude_A": float(np.max(arrays["current_magnitude_A"])),
        "current_I2t_A2s": float(np.sum(arrays["current_magnitude_A"] ** 2) * dt),
        "RMS_voltage_magnitude_V": float(
            np.sqrt(np.mean(arrays["voltage_magnitude_V"] ** 2))
        ),
        "safety_violation": bool(env.safety_violation),
    }
    env.close()
    return arrays, metrics


def current_histogram(axis, series: dict[str, np.ndarray], limit: float) -> None:
    values = {label: 100 * np.abs(data) / limit for label, data in series.items()}
    upper = max(110.0, 10.0 * math.ceil(max(map(np.max, values.values())) / 10.0))
    bins = np.linspace(0, upper, 29)
    for label, data in values.items():
        axis.hist(
            data,
            bins=bins,
            weights=np.full(len(data), 100 / len(data)),
            histtype="stepfilled",
            alpha=0.24,
            linewidth=1.6,
            label=label,
        )
    axis.axvline(100, color="tab:red", linestyle="--", label=f"scale ({limit:g} A)")
    axis.set_xlabel("dq current magnitude utilization [%]")
    axis.set_ylabel("Share of samples [%]")
    axis.set_title("PMSM current-use histogram")
    axis.legend()


def save_evaluation(
    output_dir: Path,
    controller_data: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]],
    show: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {label: metrics for label, (_, metrics) in controller_data.items()},
            stream,
            indent=2,
        )
    metric_names = list(next(iter(controller_data.values()))[1])
    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", *controller_data])
        for metric in metric_names:
            writer.writerow(
                [metric, *[data[1][metric] for data in controller_data.values()]]
            )
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    first_logs = next(iter(controller_data.values()))[0]
    axes[0].plot(
        first_logs["time_s"], first_logs["reference_rad_s"], "k--", label="reference"
    )
    for label, (logs, _) in controller_data.items():
        time_s = logs["time_s"]
        axes[0].plot(time_s, logs["omega_rad_s"], label=label)
        axes[1].plot(time_s, logs["reference_rad_s"] - logs["omega_rad_s"], label=label)
        axes[2].plot(time_s, logs["current_magnitude_A"], label=label)
        axes[3].plot(time_s, logs["voltage_magnitude_V"], label=label)
    axes[0].set_ylabel("Speed [rad/s]")
    axes[0].set_title("PMSM PPO full-range evaluation")
    axes[0].legend(ncol=3)
    axes[1].set_ylabel("Error [rad/s]")
    axes[2].set_ylabel("|i_dq| [A]")
    axes[3].set_ylabel("|v_dq| [V]")
    axes[3].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "tracking.png", dpi=180)

    fig2, axis = plt.subplots(figsize=(9, 5))
    current_histogram(
        axis,
        {label: logs["current_magnitude_A"] for label, (logs, _) in controller_data.items()},
        3.0,
    )
    fig2.tight_layout()
    fig2.savefig(output_dir / "current_histogram.png", dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    plt.close(fig2)


def evaluate(args: argparse.Namespace) -> None:
    register_pmsm_environment()
    checkpoints = (
        {"PPO": Path(args.ppo_checkpoint), "PPO + curiosity": Path(args.curiosity_checkpoint)}
        if args.compare
        else {"evaluated policy": Path(args.checkpoint)}
    )
    for checkpoint in checkpoints.values():
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint.resolve()}")
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        logging_level="ERROR",
        num_cpus=1,
    )
    controller_data = {}
    try:
        for label, checkpoint in checkpoints.items():
            algorithm = Algorithm.from_checkpoint(checkpoint.resolve())
            try:
                module = algorithm.get_module(DEFAULT_MODULE_ID)
                controller_data[label] = run_episode(module, args)
                print(f"\n{label} (intrinsic coefficient={_intrinsic_coeff(algorithm):g})")
                for metric, value in controller_data[label][1].items():
                    print(f"{metric:34s}: {value}")
            finally:
                algorithm.stop()
    finally:
        ray.shutdown()
    name = (
        f"ppo_vs_curiosity_{args.profile}_seed{args.seed}"
        if args.compare
        else f"{Path(args.checkpoint).parent.name}_{args.profile}_seed{args.seed}"
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        (COMPARE_ROOT if args.compare else EVAL_ROOT) / name
    )
    save_evaluation(output_dir, controller_data, args.show)
    print(f"Saved PMSM evaluation: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PMSM PPO and PPO+ICM speed control")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-ppo", action="store_true")
    mode.add_argument("--train-curiosity", action="store_true")
    mode.add_argument("--eval", action="store_true")
    mode.add_argument("--compare", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--episode-length", type=float, default=6.0)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--checkpoint-frequency", type=int, default=10)
    parser.add_argument("--policy-hidden-sizes", nargs="+", type=int, default=[16, 8])
    parser.add_argument("--train-reference-scale", type=float, default=1.0)
    parser.add_argument("--evaluation-reference-scale", type=float, default=1.0)
    parser.add_argument("--num-axis-levels", type=int, default=7)
    parser.add_argument("--intrinsic-reward-coeff", type=float, default=None)
    parser.add_argument("--forward-loss-weight", type=float, default=None)
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--no-parameter-jump", action="store_true")
    parser.add_argument("--profile", choices=PMSMSpeedReference.SUPPORTED, default="mixed")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--ppo-checkpoint", default="")
    parser.add_argument("--curiosity-checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0 < args.train_reference_scale <= 1:
        parser.error("--train-reference-scale must be in (0, 1].")
    if not 0 < args.evaluation_reference_scale <= 1:
        parser.error("--evaluation-reference-scale must be in (0, 1].")
    if args.train_batch_size < args.minibatch_size:
        parser.error("--train-batch-size must be >= --minibatch-size.")
    if args.num_axis_levels < 3 or args.num_axis_levels % 2 == 0:
        parser.error("--num-axis-levels must be an odd integer >= 3.")
    if args.train_ppo:
        args.intrinsic_reward_coeff = 0.0
        args.forward_loss_weight = 0.0
        train(args)
    elif args.train_curiosity:
        args.intrinsic_reward_coeff = (
            0.02 if args.intrinsic_reward_coeff is None else args.intrinsic_reward_coeff
        )
        args.forward_loss_weight = (
            0.2 if args.forward_loss_weight is None else args.forward_loss_weight
        )
        train(args)
    elif args.eval:
        if not args.checkpoint:
            parser.error("--eval requires --checkpoint.")
        evaluate(args)
    else:
        if not args.ppo_checkpoint or not args.curiosity_checkpoint:
            parser.error("--compare requires both PMSM checkpoint paths.")
        evaluate(args)


if __name__ == "__main__":
    main()
