from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dc_curiosity_mpl_cache")
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


plt.style.use("seaborn-v0_8-darkgrid")

ENV_NAME = "dc_motor_curiosity_v0"
from rllib_control.workspace import workspace_root

SCRIPT_DIR = workspace_root()
DEFAULT_MODEL_EXPERIMENT_ROOT = SCRIPT_DIR / "RL_models" / "experiments"
DEFAULT_TRAINING_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "training"
DEFAULT_EVALUATION_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "evaluations"
DEFAULT_COMPARISON_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "comparisons"
OBSERVATION_LABELS = (
    "speed",
    "current",
    "applied voltage",
    "reference",
    "tracking error",
    "integrated error",
)


class _ExpectedRayMessageFilter(logging.Filter):
    """Hide one harmless RLlib notice while preserving other warnings/errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "You are running PPO on the new API stack" not in record.getMessage()


logging.getLogger("ray.rllib.algorithms.algorithm_config").addFilter(
    _ExpectedRayMessageFilter()
)
warnings.filterwarnings(
    "ignore",
    message=r".*Overriding environment rllib-single-agent-env-v0.*",
    category=UserWarning,
    module=r"gymnasium\.envs\.registration",
)


# -----------------------------------------------------------------------------
# Motor model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AMax32Params:
    """Nominal Maxon A-max 32 parameters used by the existing project."""

    R: float = 3.99
    L: float = 0.556e-3
    kT: float = 35.2e-3
    Ke: float = 35.2e-3
    J: float = 45.3e-7
    B: float = 1.0e-6
    T_coulomb: float = 0.001506
    n0_rpm: float = 6460.0


@dataclass
class MotorState:
    current: float = 0.0
    omega: float = 0.0
    theta: float = 0.0


class AMax32Motor:
    """Continuous-time brushed DC motor model."""

    def __init__(self, params: AMax32Params | None = None):
        self.nominal = params or AMax32Params()
        self.R = self.nominal.R
        self.L = self.nominal.L
        self.kT = self.nominal.kT
        self.Ke = self.nominal.Ke
        self.J = self.nominal.J
        self.B = self.nominal.B
        self.T_coulomb = self.nominal.T_coulomb

    def set_relative_parameters(
        self,
        *,
        resistance: float = 1.0,
        inductance: float = 1.0,
        inertia: float = 1.0,
        torque_constant: float = 1.0,
    ) -> None:
        self.R = self.nominal.R * resistance
        self.L = self.nominal.L * inductance
        self.J = self.nominal.J * inertia
        self.kT = self.nominal.kT * torque_constant
        self.Ke = self.nominal.Ke * torque_constant

    def derivatives(
        self,
        state: MotorState,
        voltage: float,
        load_torque: float,
    ) -> MotorState:
        current_dot = (
            voltage - self.R * state.current - self.Ke * state.omega
        ) / self.L

        if abs(state.omega) <= 1e-6:
            direction = 0.0
        else:
            direction = 1.0 if state.omega > 0.0 else -1.0

        friction = self.B * state.omega + self.T_coulomb * direction
        electromagnetic_torque = self.kT * state.current
        omega_dot = (
            electromagnetic_torque - load_torque - friction
        ) / self.J

        return MotorState(
            current=current_dot,
            omega=omega_dot,
            theta=state.omega,
        )

    def rk4_step(
        self,
        state: MotorState,
        voltage: float,
        load_torque: float,
        dt: float,
    ) -> MotorState:
        def add(x: MotorState, k: MotorState, scale: float) -> MotorState:
            return MotorState(
                current=x.current + scale * k.current,
                omega=x.omega + scale * k.omega,
                theta=x.theta + scale * k.theta,
            )

        k1 = self.derivatives(state, voltage, load_torque)
        k2 = self.derivatives(add(state, k1, 0.5 * dt), voltage, load_torque)
        k3 = self.derivatives(add(state, k2, 0.5 * dt), voltage, load_torque)
        k4 = self.derivatives(add(state, k3, dt), voltage, load_torque)

        return MotorState(
            current=state.current
            + (dt / 6.0)
            * (k1.current + 2.0 * k2.current + 2.0 * k3.current + k4.current),
            omega=state.omega
            + (dt / 6.0)
            * (k1.omega + 2.0 * k2.omega + 2.0 * k3.omega + k4.omega),
            theta=state.theta
            + (dt / 6.0)
            * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta),
        )


# -----------------------------------------------------------------------------
# Reference profiles
# -----------------------------------------------------------------------------


class SpeedReference:
    """Speed-reference generator used by both training and evaluation."""

    SUPPORTED = ("random", "steps", "sine", "mixed")

    def __init__(
        self,
        profile: str,
        duration: float,
        rng: np.random.Generator,
        amplitude_scale: float = 1.0,
    ):
        if profile not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported profile {profile!r}; choose one of {self.SUPPORTED}."
            )
        self.profile = profile
        self.duration = float(duration)
        self.rng = rng
        self.amplitude_scale = float(amplitude_scale)
        if not 0.0 < self.amplitude_scale <= 1.0:
            raise ValueError("Reference amplitude_scale must be in (0, 1].")
        self.times = np.array([0.0, self.duration], dtype=np.float64)
        self.values = np.zeros(2, dtype=np.float64)
        self._build()

    def _build(self) -> None:
        if self.profile == "random":
            # Same construction as sim_DC_RL_Speed.py, but with an explicit RNG
            # so seeded evaluations are exactly reproducible.
            n_segments = int(self.rng.integers(6, 12))
            durations = self.rng.uniform(0.4, 1.2, n_segments)
            interior = np.cumsum(durations)
            interior = interior[interior < self.duration]
            self.times = np.concatenate(([0.0], interior, [self.duration]))
            self.values = self.rng.uniform(-600.0, 600.0, len(self.times))
            for index in range(1, len(self.values)):
                self.values[index] = (
                    0.6 * self.values[index - 1] + 0.4 * self.values[index]
                )

        elif self.profile == "steps":
            fractions = np.array(
                [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
            )
            self.times = fractions * self.duration
            self.values = np.array(
                [0.0, 250.0, -250.0, 400.0, 100.0, -400.0, 300.0, 0.0, 0.0]
            )

        elif self.profile == "sine":
            self.times = np.array([0.0, self.duration], dtype=np.float64)
            self.values = np.zeros(2, dtype=np.float64)

        elif self.profile == "mixed":
            # A deterministic combination inspired by the notebook's square,
            # ramp, and sinusoidal reference examples.
            self.times = np.linspace(0.0, self.duration, 9)
            self.values = np.array(
                [0.0, 300.0, 300.0, -250.0, -250.0, 350.0, -350.0, 150.0, 0.0]
            )

    @property
    def is_piecewise_constant(self) -> bool:
        return self.profile in {"random", "steps"}

    def value(self, time_s: float) -> float:
        time_s = float(min(max(time_s, 0.0), self.duration))
        if self.profile == "sine":
            return float(
                self.amplitude_scale
                * 350.0
                * math.sin(2.0 * math.pi * 0.11 * time_s)
            )
        if self.profile == "mixed":
            phase = time_s / max(self.duration, 1e-12)
            if phase < 0.25:
                raw_value = 300.0 if phase < 0.125 else -300.0
                return self.amplitude_scale * raw_value
            if phase < 0.50:
                local_phase = (phase - 0.25) / 0.25
                return float(
                    self.amplitude_scale * (300.0 - 600.0 * local_phase)
                )
            if phase < 0.75:
                local_phase = (phase - 0.50) / 0.25
                return float(
                    self.amplitude_scale
                    * 350.0
                    * math.sin(2.0 * math.pi * local_phase)
                )
            raw_value = 200.0 if phase < 0.875 else -200.0
            return self.amplitude_scale * raw_value
        index = int(np.searchsorted(self.times, time_s, side="right") - 1)
        index = min(max(index, 0), len(self.values) - 1)
        return float(self.amplitude_scale * self.values[index])


# -----------------------------------------------------------------------------
# Gymnasium environment
# -----------------------------------------------------------------------------


class DCMotorCuriosityEnv(gym.Env[np.ndarray, int]):
    """Normalized DC speed-control environment with discrete voltage actions.

    Policy observation order:

    0. normalized speed,
    1. normalized armature current,
    2. normalized applied voltage,
    3. normalized speed reference,
    4. normalized tracking error,
    5. normalized integrated tracking error.

    The ICM class below masks indices 3--5, so curiosity models only the physical
    state [speed, current, applied voltage].
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any] | None = None):
        super().__init__()
        cfg = dict(env_config or {})

        self.dt = float(cfg.get("dt", 1e-4))
        self.frame_skip = int(cfg.get("frame_skip", 20))
        self.episode_length = float(cfg.get("episode_length", 8.0))
        self.max_steps = max(
            1, int(math.ceil(self.episode_length / (self.dt * self.frame_skip)))
        )
        self.Vdc = float(cfg.get("Vdc", 24.0))
        self.tau_actuator = float(cfg.get("tau_actuator", 1e-3))
        self.nominal_load_torque = float(cfg.get("load_torque", 0.02))
        self.profile_name = str(cfg.get("profile", "random"))
        self.reference_scale = float(cfg.get("reference_scale", 1.0))
        self.training_mode = bool(cfg.get("training_mode", True))
        self.domain_randomization = bool(cfg.get("domain_randomization", True))
        self.parameter_jump = bool(cfg.get("parameter_jump", True))
        self.base_seed = int(cfg.get("seed", 0))
        self.num_voltage_levels = int(cfg.get("num_voltage_levels", 21))

        if self.num_voltage_levels < 3 or self.num_voltage_levels % 2 == 0:
            raise ValueError("num_voltage_levels must be an odd integer >= 3.")
        if self.dt <= 0.0 or self.frame_skip <= 0 or self.episode_length <= 0.0:
            raise ValueError("dt, frame_skip, and episode_length must be positive.")

        self.current_limit = float(cfg.get("current_limit", 6.0))
        self.speed_limit = float(cfg.get("speed_limit", 900.0))
        self.voltage_levels = np.linspace(
            -self.Vdc,
            self.Vdc,
            self.num_voltage_levels,
            dtype=np.float64,
        )

        self.action_space = spaces.Discrete(self.num_voltage_levels)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(self.base_seed)
        self.motor = AMax32Motor()
        self.state = MotorState()
        self.reference = SpeedReference(
            self.profile_name,
            self.episode_length,
            self.rng,
            amplitude_scale=self.reference_scale,
        )

        self.time = 0.0
        self.step_count = 0
        self.applied_voltage = 0.0
        self.voltage_command = 0.0
        self.previous_voltage = 0.0
        self.previous_error = 0.0
        self.error_integral = 0.0
        self.load_torque = self.nominal_load_torque
        self.last_reference = 0.0
        self.safety_violation = False

        self.jump_time: float | None = None
        self.jump_factor = 1.0
        self.jump_applied = False

    def _configure_episode(self) -> None:
        resistance_factor = 1.0
        inertia_factor = 1.0
        inductance_factor = 1.0
        self.load_torque = self.nominal_load_torque

        if self.training_mode and self.domain_randomization:
            resistance_factor = float(self.rng.uniform(0.85, 1.15))
            inertia_factor = float(self.rng.uniform(0.85, 1.20))
            inductance_factor = float(self.rng.uniform(0.90, 1.10))
            self.load_torque = float(
                self.rng.uniform(0.25, 1.50) * self.nominal_load_torque
            )

        self.motor = AMax32Motor()
        self.motor.set_relative_parameters(
            resistance=resistance_factor,
            inductance=inductance_factor,
            inertia=inertia_factor,
        )

        self.jump_applied = False
        self.jump_factor = 1.0
        self.jump_time = None
        if (
            self.training_mode
            and self.parameter_jump
            and self.rng.random() < 0.5
        ):
            self.jump_time = float(
                self.rng.uniform(
                    0.2 * self.episode_length,
                    0.8 * self.episode_length,
                )
            )
            self.jump_factor = float(self.rng.uniform(0.75, 1.35))

    def _apply_parameter_jump(self) -> None:
        if (
            self.jump_applied
            or self.jump_time is None
            or self.time < self.jump_time
        ):
            return
        self.motor.R *= self.jump_factor
        self.jump_applied = True

    def _observation(self) -> np.ndarray:
        reference = self.reference.value(self.time)
        error = (reference - self.state.omega) / self.speed_limit
        return np.array(
            [
                np.clip(self.state.omega / self.speed_limit, -1.0, 1.0),
                np.clip(self.state.current / self.current_limit, -1.0, 1.0),
                np.clip(self.applied_voltage / self.Vdc, -1.0, 1.0),
                np.clip(reference / self.speed_limit, -1.0, 1.0),
                np.clip(error, -1.0, 1.0),
                np.clip(self.error_integral, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._configure_episode()
        self.reference = SpeedReference(
            self.profile_name,
            self.episode_length,
            self.rng,
            amplitude_scale=self.reference_scale,
        )

        self.state = MotorState()
        self.time = 0.0
        self.step_count = 0
        self.applied_voltage = 0.0
        self.voltage_command = 0.0
        self.previous_voltage = 0.0
        self.previous_error = 0.0
        self.error_integral = 0.0
        self.last_reference = self.reference.value(0.0)
        self.safety_violation = False

        info = self._info_dict(
            reward_step=0.0,
            tracking=0.0,
            derivative_penalty=0.0,
            integral_penalty=0.0,
            effort_penalty=0.0,
            current_penalty=0.0,
            smoothness_penalty=0.0,
            settling_bonus=0.0,
        )
        return self._observation(), info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_index = int(np.asarray(action).item())
        if not self.action_space.contains(action_index):
            raise ValueError(f"Invalid discrete action: {action_index}.")

        voltage_command = float(self.voltage_levels[action_index])
        return self.step_voltage(voltage_command)

    def step_voltage(self, voltage_command: float):
        """Integrate a bounded physical voltage; legacy discrete step is unchanged."""
        voltage_command = float(np.clip(voltage_command, -self.Vdc, self.Vdc))
        self.voltage_command = voltage_command
        reward_sum = 0.0
        executed_substeps = 0
        last_terms: dict[str, float] = {}

        for _ in range(self.frame_skip):
            if hasattr(self, "before_substep_callback"):
                self.before_substep_callback()
            self._apply_parameter_jump()
            if hasattr(self, "current_control_callback"):
                voltage_command = self.current_control_callback()
                self.voltage_command = voltage_command

            alpha = min(max(self.dt / self.tau_actuator, 0.0), 1.0)
            self.applied_voltage += alpha * (
                voltage_command - self.applied_voltage
            )

            self.state = self.motor.rk4_step(
                self.state,
                self.applied_voltage,
                self.load_torque,
                self.dt,
            )
            self.time += self.dt
            executed_substeps += 1

            reference = self.reference.value(self.time)
            self.last_reference = reference
            if hasattr(self, "after_substep_callback"):
                self.after_substep_callback()
            error = (reference - self.state.omega) / self.speed_limit
            error_derivative = float(
                min(max((error - self.previous_error) / self.dt, -50.0), 50.0)
            )

            self.error_integral += error * self.dt
            self.error_integral = float(
                min(max(self.error_integral, -0.5), 0.5)
            )

            voltage_norm = self.applied_voltage / self.Vdc
            current_norm = self.state.current / self.current_limit
            voltage_change = (
                self.applied_voltage - self.previous_voltage
            ) / self.Vdc

            tracking = -20.0 * error**2
            derivative_penalty = -0.0005 * error_derivative**2
            integral_penalty = -0.1 * self.error_integral**2
            effort_penalty = -0.01 * voltage_norm**2
            current_penalty = -0.01 * current_norm**2
            smoothness_penalty = -getattr(self, "substep_smoothness_weight", 0.01) * voltage_change**2
            settling_bonus = 0.1 if abs(error) < 0.01 else 0.0

            raw_reward = (
                tracking
                + derivative_penalty
                + integral_penalty
                + effort_penalty
                + current_penalty
                + smoothness_penalty
                + settling_bonus
            )
            reward_step = (self.reward_callback() if hasattr(self, "reward_callback")
                           else float(min(max(raw_reward, -1.0), 0.5)))
            reward_sum += reward_step

            self.previous_error = error
            self.previous_voltage = self.applied_voltage

            last_terms = {
                "reward_step": reward_step,
                "tracking": tracking,
                "derivative_penalty": derivative_penalty,
                "integral_penalty": integral_penalty,
                "effort_penalty": effort_penalty,
                "current_penalty": current_penalty,
                "smoothness_penalty": smoothness_penalty,
                "settling_bonus": settling_bonus,
            }

            if (
                not np.isfinite(self.state.current)
                or not np.isfinite(self.state.omega)
                or abs(self.state.current) > 1.25 * self.current_limit
                or abs(self.state.omega) > 1.25 * self.speed_limit
            ):
                self.safety_violation = True
                break

        self.step_count += 1
        terminated = self.safety_violation
        truncated = self.step_count >= self.max_steps or self.time >= self.episode_length

        if self.safety_violation:
            reward = -5.0
        else:
            reward = reward_sum / max(1, executed_substeps)

        info = self._info_dict(**last_terms)
        return self._observation(), float(reward), terminated, truncated, info

    def _info_dict(
        self,
        *,
        reward_step: float,
        tracking: float,
        derivative_penalty: float,
        integral_penalty: float,
        effort_penalty: float,
        current_penalty: float,
        smoothness_penalty: float,
        settling_bonus: float,
    ) -> dict[str, Any]:
        return {
            "time": float(self.time),
            "omega": float(self.state.omega),
            "current": float(self.state.current),
            "voltage": float(self.applied_voltage),
            "voltage_command": float(self.voltage_command),
            "reference": float(self.last_reference),
            "load_torque": float(self.load_torque),
            "electromagnetic_torque": float(self.motor.kT * self.state.current),
            "error": float(self.last_reference - self.state.omega),
            "reward_step": float(reward_step),
            "tracking": float(tracking),
            "derivative_penalty": float(derivative_penalty),
            "integral_penalty": float(integral_penalty),
            "effort_penalty": float(effort_penalty),
            "current_penalty": float(current_penalty),
            "smoothness_penalty": float(smoothness_penalty),
            "settling_bonus": float(settling_bonus),
            "R": float(self.motor.R),
            "L": float(self.motor.L),
            "J": float(self.motor.J),
            "parameter_jump_time": (
                -1.0 if self.jump_time is None else float(self.jump_time)
            ),
            "parameter_jump_applied": bool(self.jump_applied),
            "safety_violation": bool(self.safety_violation),
        }


@dataclass
class PIController:
    """Classical speed PI controller using the same discrete voltage actions.

    The gains were selected on the nominal simulation as a transparent baseline,
    not optimized against the PPO checkpoint.  The integrator is clamped so its
    contribution cannot demand more than the available DC-link voltage.
    """

    kp: float = 0.03
    ki: float = 3.0
    integral: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def action(self, observation: np.ndarray, env: DCMotorCuriosityEnv) -> int:
        error_rad_s = float(observation[4]) * env.speed_limit
        control_period = env.dt * env.frame_skip
        self.integral += error_rad_s * control_period

        if self.ki > 0.0:
            integral_limit = env.Vdc / self.ki
            self.integral = float(
                np.clip(self.integral, -integral_limit, integral_limit)
            )

        requested_voltage = float(
            np.clip(
                self.kp * error_rad_s + self.ki * self.integral,
                -env.Vdc,
                env.Vdc,
            )
        )
        return int(np.argmin(np.abs(env.voltage_levels - requested_voltage)))


# -----------------------------------------------------------------------------
# ICM and RLlib configuration
# -----------------------------------------------------------------------------


class PhysicalStateICM(IntrinsicCuriosityModel):
    """RLlib ICM that ignores externally generated tracking quantities.

    RLlib gives the ICM the same observation as the policy. The policy needs the
    reference, error, and integrated error, but these signals are not motor dynamics.
    This class masks observation indices 3--5 before calling RLlib's official ICM,
    leaving [speed, current, applied voltage] as the physical curiosity state.
    """

    def _forward_train(self, batch: dict[str, torch.Tensor], **kwargs: Any):
        masked_batch = dict(batch)
        observations = batch[Columns.OBS].clone()
        next_observations = batch[Columns.NEXT_OBS].clone()
        observations[..., 3:] = 0.0
        next_observations[..., 3:] = 0.0
        masked_batch[Columns.OBS] = observations
        masked_batch[Columns.NEXT_OBS] = next_observations
        return super()._forward_train(masked_batch, **kwargs)


def register_dc_environment() -> None:
    register_env(ENV_NAME, lambda config: DCMotorCuriosityEnv(dict(config)))


def make_rllib_config(
    *,
    seed: int,
    episode_length: float,
    profile: str,
    train_batch_size: int,
    minibatch_size: int,
    num_epochs: int,
    num_env_runners: int,
    intrinsic_reward_coeff: float,
    forward_loss_weight: float,
    domain_randomization: bool,
    parameter_jump: bool,
    train_reference_scale: float,
    policy_hidden_sizes: tuple[int, ...],
) -> PPOConfig:
    """Create a Ray 2.56 PPO configuration with the official ICM learner."""

    if train_batch_size < minibatch_size:
        raise ValueError("train_batch_size must be >= minibatch_size.")
    if not 0.0 <= forward_loss_weight <= 1.0:
        raise ValueError("forward_loss_weight must be in [0, 1].")
    if intrinsic_reward_coeff < 0.0:
        raise ValueError("intrinsic_reward_coeff must be non-negative.")
    if not 0.0 < train_reference_scale <= 1.0:
        raise ValueError("train_reference_scale must be in (0, 1].")
    if not policy_hidden_sizes or any(size <= 0 for size in policy_hidden_sizes):
        raise ValueError("policy_hidden_sizes must contain positive integers.")

    env_config = {
        "seed": seed,
        "episode_length": episode_length,
        "profile": profile,
        "training_mode": True,
        "domain_randomization": domain_randomization,
        "parameter_jump": parameter_jump,
        "num_voltage_levels": 21,
        "reference_scale": train_reference_scale,
    }

    config = (
        PPOConfig()
        .environment(ENV_NAME, env_config=env_config)
        .framework("torch")
        .debugging(seed=seed, log_level="ERROR", log_sys_usage=False)
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",
            batch_mode="truncate_episodes",
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,
            learner_class=PPOTorchLearnerWithCuriosity,
            learner_config_dict={
                "intrinsic_reward_coeff": intrinsic_reward_coeff,
                "forward_loss_weight": forward_loss_weight,
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
            train_batch_size_per_learner=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    DEFAULT_MODULE_ID: RLModuleSpec(
                        model_config={
                            "fcnet_hiddens": list(policy_hidden_sizes),
                            "fcnet_activation": "tanh",
                            "vf_share_layers": False,
                        }
                    ),
                    ICM_MODULE_ID: RLModuleSpec(
                        module_class=PhysicalStateICM,
                        learner_only=True,
                        model_config={
                            "feature_dim": 16,
                            "feature_net_hiddens": (64, 64),
                            "feature_net_activation": "relu",
                            "inverse_net_hiddens": (64, 64),
                            "inverse_net_activation": "relu",
                            "forward_net_hiddens": (64, 64),
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
    return config


# -----------------------------------------------------------------------------
# Metrics and visualization
# -----------------------------------------------------------------------------


def plot_current_utilization_histogram(
    axis: plt.Axes,
    current_series: dict[str, np.ndarray],
    current_limit: float,
) -> None:
    """Plot the share of evaluation samples in absolute-current bins."""

    utilization = {
        label: 100.0 * np.abs(np.asarray(values, dtype=float)) / current_limit
        for label, values in current_series.items()
    }
    maximum = max(float(np.max(values)) for values in utilization.values())
    upper = max(110.0, 10.0 * math.ceil(maximum / 10.0))
    bins = np.linspace(0.0, upper, 29)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (label, values) in enumerate(utilization.items()):
        weights = np.full(len(values), 100.0 / max(1, len(values)))
        axis.hist(
            values,
            bins=bins,
            weights=weights,
            histtype="stepfilled",
            alpha=0.22,
            linewidth=1.7,
            edgecolor=colors[index % len(colors)],
            color=colors[index % len(colors)],
            label=label,
        )
        percentile_95 = float(np.percentile(values, 95.0))
        axis.axvline(
            percentile_95,
            color=colors[index % len(colors)],
            linestyle=":",
            linewidth=1.4,
        )

    axis.axvline(
        100.0,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"configured scale ({current_limit:g} A)",
    )
    axis.set_xlabel("Absolute current utilization [%]")
    axis.set_ylabel("Share of evaluation samples [%]")
    axis.set_title("Current-use histogram (dotted lines: 95th percentile)")
    axis.legend()
    axis.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.65)


def _first_numeric_by_key(data: Any, wanted_key: str) -> float | None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key == wanted_key and isinstance(value, (int, float, np.number)):
                return float(value)
        for value in data.values():
            found = _first_numeric_by_key(value, wanted_key)
            if found is not None:
                return found
    elif isinstance(data, (list, tuple)):
        for value in data:
            found = _first_numeric_by_key(value, wanted_key)
            if found is not None:
                return found
    return None


def _episode_return(result: dict[str, Any]) -> float:
    runners = result.get(ENV_RUNNER_RESULTS, {})
    value = runners.get(EPISODE_RETURN_MEAN, np.nan)
    return float(value) if value is not None else float("nan")


def find_reference_segments(reference: np.ndarray) -> list[tuple[int, int]]:
    if len(reference) < 2:
        return []
    changes = np.flatnonzero(np.abs(np.diff(reference)) > 1e-9) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes - 1, [len(reference) - 1]))
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= 3
    ]


def rise_time(
    time_s: np.ndarray,
    omega: np.ndarray,
    start_value: float,
    target_value: float,
) -> float:
    delta = target_value - start_value
    if abs(delta) < 1e-12:
        return float("nan")
    y10 = start_value + 0.1 * delta
    y90 = start_value + 0.9 * delta
    if delta > 0.0:
        indices10 = np.flatnonzero(omega >= y10)
        indices90 = np.flatnonzero(omega >= y90)
    else:
        indices10 = np.flatnonzero(omega <= y10)
        indices90 = np.flatnonzero(omega <= y90)
    if len(indices10) == 0 or len(indices90) == 0:
        return float("nan")
    index10 = int(indices10[0])
    valid90 = indices90[indices90 >= index10]
    if len(valid90) == 0:
        return float("nan")
    return float(time_s[int(valid90[0])] - time_s[index10])


def overshoot_percent(
    omega: np.ndarray,
    start_value: float,
    target_value: float,
) -> float:
    delta = target_value - start_value
    if abs(delta) < 1e-12:
        return 0.0
    if delta > 0.0:
        return float(max(0.0, (np.max(omega) - target_value) / abs(delta) * 100.0))
    return float(max(0.0, (target_value - np.min(omega)) / abs(delta) * 100.0))


def compute_segment_metrics(
    time_s: np.ndarray,
    omega: np.ndarray,
    reference: np.ndarray,
    voltage: np.ndarray,
    voltage_limit: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for segment_id, (start, end) in enumerate(
        find_reference_segments(reference), start=1
    ):
        segment_slice = slice(start, end + 1)
        t_seg = time_s[segment_slice]
        w_seg = omega[segment_slice]
        ref_seg = reference[segment_slice]
        u_seg = voltage[segment_slice]
        error = ref_seg - w_seg
        initial_value = omega[0] if start == 0 else reference[start - 1]
        target_value = float(ref_seg[0])

        rows.append(
            {
                "segment": segment_id,
                "interval": f"{t_seg[0]:.3f}-{t_seg[-1]:.3f}",
                "reference_rad_s": target_value,
                "rise_time_s": rise_time(
                    t_seg, w_seg, initial_value, target_value
                ),
                "overshoot_percent": overshoot_percent(
                    w_seg, initial_value, target_value
                ),
                "steady_state_error_rad_s": float(
                    np.mean(error[-max(1, len(error) // 5) :])
                ),
                "IAE": float(np.trapezoid(np.abs(error), t_seg)),
                "ISE": float(np.trapezoid(error**2, t_seg)),
                "RMS_voltage_V": float(np.sqrt(np.mean(u_seg**2))),
                "time_at_voltage_rail_s": float(
                    np.sum(
                        np.diff(t_seg)
                        * (np.abs(u_seg[:-1]) >= 0.98 * voltage_limit)
                    )
                ),
            }
        )
    return rows


def print_segment_metrics(rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        print("No piecewise-constant reference segments to report.")
        return

    print("\nPer-segment speed-control metrics")
    header = (
        f"{'segment':>7} | {'interval [s]':>15} | {'ref [rad/s]':>11} | "
        f"{'rise [s]':>9} | {'overshoot [%]':>13} | {'SSE':>10} | "
        f"{'IAE':>10} | {'ISE':>12} | {'RMS u [V]':>10} | {'rail [s]':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['segment']:>7d} | {row['interval']:>15} | "
            f"{row['reference_rad_s']:>11.2f} | "
            f"{row['rise_time_s']:>9.4f} | "
            f"{row['overshoot_percent']:>13.2f} | "
            f"{row['steady_state_error_rad_s']:>10.3f} | "
            f"{row['IAE']:>10.3f} | {row['ISE']:>12.3f} | "
            f"{row['RMS_voltage_V']:>10.3f} | "
            f"{row['time_at_voltage_rail_s']:>9.4f}"
        )


def save_training_history(
    history: list[dict[str, float | int | None]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    iterations = np.array([row["iteration"] for row in history], dtype=float)
    returns = np.array([row["episode_return_mean"] for row in history], dtype=float)
    intrinsic = np.array(
        [
            np.nan if row["mean_intrinsic_reward"] is None else row["mean_intrinsic_reward"]
            for row in history
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(iterations, returns, color="tab:blue")
    axes[0].set_ylabel("Mean episode return")
    axes[0].set_title("PPO + ICM training")
    axes[1].plot(iterations, intrinsic, color="tab:orange")
    axes[1].set_ylabel("Mean intrinsic reward")
    axes[1].set_xlabel("Training iteration")
    for axis in axes:
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=180)
    plt.close(fig)


def save_evaluation_outputs(
    logs: dict[str, list[float]],
    metrics: dict[str, float | int | bool],
    segment_rows: list[dict[str, Any]],
    output_dir: Path,
    current_limit: float,
    show: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "evaluation_timeseries.csv"
    keys = list(logs.keys())
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(zip(*(logs[key] for key in keys)))

    with (output_dir / "evaluation_metrics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)

    if segment_rows:
        with (output_dir / "segment_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(segment_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(segment_rows)

    time_s = np.asarray(logs["time_s"])
    omega = np.asarray(logs["omega_rad_s"])
    reference = np.asarray(logs["reference_rad_s"])
    current = np.asarray(logs["current_A"])
    voltage = np.asarray(logs["voltage_V"])
    reward = np.asarray(logs["extrinsic_reward"])

    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(time_s, omega, label=r"$\omega$")
    axes[0].plot(time_s, reference, "--", label=r"$\omega_{ref}$")
    axes[0].set_ylabel("Speed [rad/s]")
    axes[0].set_title("Curiosity-trained DC motor policy (deterministic evaluation)")
    axes[0].legend()
    axes[1].plot(time_s, current, color="tab:orange")
    axes[1].set_ylabel("Current [A]")
    axes[2].step(time_s, voltage, where="post", color="tab:green")
    axes[2].set_ylabel("Voltage [V]")
    axes[3].plot(time_s, reference - omega, color="tab:red")
    axes[3].set_ylabel("Error [rad/s]")
    axes[4].plot(time_s, reward, color="tab:purple")
    axes[4].set_ylabel("Extrinsic reward")
    axes[4].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_tracking.png", dpi=180)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
    points = axes2[0].scatter(
        omega,
        current,
        c=time_s,
        s=9,
        cmap="viridis",
        alpha=0.8,
    )
    axes2[0].set_xlabel("Speed [rad/s]")
    axes2[0].set_ylabel("Current [A]")
    axes2[0].set_title("Physical-state trajectory")
    fig2.colorbar(points, ax=axes2[0], label="Time [s]")

    plot_current_utilization_histogram(
        axes2[1], {"evaluated policy": current}, current_limit
    )
    fig2.tight_layout()
    fig2.savefig(
        output_dir / "evaluation_state_and_current_loading.png", dpi=180
    )

    saliency_keys = [f"saliency_{index}" for index in range(len(OBSERVATION_LABELS))]
    saliency = np.column_stack([np.asarray(logs[key]) for key in saliency_keys])
    mean_saliency = np.mean(np.abs(saliency), axis=0)
    normalized_saliency = mean_saliency / max(float(np.sum(mean_saliency)), 1e-12)

    with (output_dir / "observation_saliency.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                label: float(value)
                for label, value in zip(OBSERVATION_LABELS, normalized_saliency)
            },
            stream,
            indent=2,
        )

    fig3, axis3 = plt.subplots(figsize=(9, 4.5))
    axis3.bar(OBSERVATION_LABELS, normalized_saliency, color="tab:blue")
    axis3.set_ylabel("Normalized mean |gradient|")
    axis3.set_title("Policy observation sensitivity (saliency)")
    axis3.tick_params(axis="x", rotation=25)
    axis3.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    fig3.tight_layout()
    fig3.savefig(output_dir / "evaluation_observation_saliency.png", dpi=180)

    if show:
        plt.show()
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Path:
    controller_kind = (
        "curiosity_ppo" if args.intrinsic_reward_coeff > 0.0 else "ppo_baseline"
    )
    run_name = args.run_name.strip() or f"{controller_kind}_seed{args.seed}"
    model_root = (
        Path(args.model_dir).resolve()
        if args.model_dir
        else (DEFAULT_MODEL_EXPERIMENT_ROOT / run_name).resolve()
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (DEFAULT_TRAINING_OUTPUT_ROOT / run_name).resolve()
    )

    resume_checkpoint = (
        Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    )
    start_iteration = 0
    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resume_checkpoint}"
            )
        checkpoint_name = resume_checkpoint.name
        if not checkpoint_name.startswith("iteration_"):
            raise ValueError(
                "Resume checkpoint directory must be named iteration_NNNN: "
                f"{resume_checkpoint}"
            )
        try:
            start_iteration = int(checkpoint_name.removeprefix("iteration_"))
        except ValueError as error:
            raise ValueError(
                f"Cannot determine iteration from checkpoint: {resume_checkpoint}"
            ) from error
        if args.iterations <= start_iteration:
            raise ValueError(
                f"Target --iterations ({args.iterations}) must be greater than the "
                f"resume iteration ({start_iteration})."
            )

    if resume_checkpoint is None and not args.overwrite_existing:
        model_has_content = model_root.exists() and any(model_root.iterdir())
        output_has_training = (output_dir / "training_config.json").exists()
        if model_has_content or output_has_training:
            raise FileExistsError(
                "This training run already exists and will not be overwritten: "
                f"{run_name!r}. Choose a different --run-name or explicitly pass "
                "--overwrite-existing."
            )

    register_dc_environment()
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        logging_level="ERROR",
        num_cpus=max(1, args.num_env_runners + 1),
    )

    config = make_rllib_config(
        seed=args.seed,
        episode_length=args.episode_length,
        profile="random",
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        num_epochs=args.num_epochs,
        num_env_runners=args.num_env_runners,
        intrinsic_reward_coeff=args.intrinsic_reward_coeff,
        forward_loss_weight=args.forward_loss_weight,
        domain_randomization=not args.no_domain_randomization,
        parameter_jump=not args.no_parameter_jump,
        train_reference_scale=args.train_reference_scale,
        policy_hidden_sizes=tuple(args.policy_hidden_sizes),
    )

    model_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "training_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "ray_version": ray.__version__,
                "torch_version": torch.__version__,
                "motor": asdict(AMax32Params()),
                "controller_kind": controller_kind,
                "run_name": run_name,
                "model_dir": str(model_root),
                "output_dir": str(output_dir),
                "seed": args.seed,
                "iterations": args.iterations,
                "episode_length": args.episode_length,
                "train_batch_size": args.train_batch_size,
                "minibatch_size": args.minibatch_size,
                "num_epochs": args.num_epochs,
                "num_env_runners": args.num_env_runners,
                "intrinsic_reward_coeff": args.intrinsic_reward_coeff,
                "forward_loss_weight": args.forward_loss_weight,
                "domain_randomization": not args.no_domain_randomization,
                "parameter_jump": not args.no_parameter_jump,
                "train_reference_scale": args.train_reference_scale,
                "evaluation_reference_scale": args.evaluation_reference_scale,
                "policy_hidden_sizes": list(args.policy_hidden_sizes),
                "resumed_from": (
                    str(resume_checkpoint) if resume_checkpoint is not None else None
                ),
                "start_iteration": start_iteration,
            },
            stream,
            indent=2,
            sort_keys=True,
        )

    algorithm = (
        Algorithm.from_checkpoint(resume_checkpoint)
        if resume_checkpoint is not None
        else config.build_algo()
    )
    restored_intrinsic_coeff = _checkpoint_intrinsic_reward_coeff(algorithm)
    expected_intrinsic_coeff = float(args.intrinsic_reward_coeff)
    if not math.isclose(
        restored_intrinsic_coeff, expected_intrinsic_coeff, abs_tol=1e-12
    ):
        algorithm.stop()
        ray.shutdown()
        raise ValueError(
            "Resume checkpoint controller type does not match the selected training "
            f"mode: checkpoint intrinsic coefficient={restored_intrinsic_coeff:g}, "
            f"requested={expected_intrinsic_coeff:g}."
        )
    history: list[dict[str, float | int | None]] = []
    latest_checkpoint: Path | None = None

    try:
        for iteration in range(start_iteration + 1, args.iterations + 1):
            result = algorithm.train()
            episode_return = _episode_return(result)
            intrinsic_reward = _first_numeric_by_key(
                result, "mean_intrinsic_rewards"
            )
            sampled_steps = int(result.get("num_env_steps_sampled_lifetime", 0))
            history.append(
                {
                    "iteration": iteration,
                    "sampled_env_steps": sampled_steps,
                    "episode_return_mean": episode_return,
                    "mean_intrinsic_reward": intrinsic_reward,
                }
            )

            intrinsic_text = (
                "n/a" if intrinsic_reward is None else f"{intrinsic_reward:.6f}"
            )
            print(
                f"iteration={iteration:04d}  "
                f"return={episode_return: .5f}  "
                f"intrinsic={intrinsic_text}  steps={sampled_steps}"
            )

            should_checkpoint = (
                iteration % args.checkpoint_frequency == 0
                or iteration == args.iterations
            )
            if should_checkpoint:
                latest_checkpoint = model_root / f"iteration_{iteration:04d}"
                algorithm.save_to_path(latest_checkpoint)
                save_training_history(history, output_dir)

        save_training_history(history, output_dir)
    finally:
        algorithm.stop()
        ray.shutdown()

    if latest_checkpoint is None:
        raise RuntimeError("Training ended without creating a checkpoint.")
    print(f"Saved checkpoint: {latest_checkpoint}")
    print(f"Saved training outputs: {output_dir}")
    return latest_checkpoint


def _deterministic_policy_action(module: torch.nn.Module, observation: np.ndarray) -> int:
    try:
        device = next(module.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        output = module.forward_inference({Columns.OBS: tensor})
    logits = output[Columns.ACTION_DIST_INPUTS]
    return int(torch.argmax(logits, dim=-1).item())


def _observation_saliency(
    module: torch.nn.Module,
    observation: np.ndarray,
    action: int,
) -> np.ndarray:
    """Gradient saliency of the selected action logit with respect to observation."""

    try:
        device = next(module.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    tensor.requires_grad_(True)
    module.zero_grad(set_to_none=True)
    # A module restored only for evaluation is marked inference_only, therefore
    # RLlib correctly rejects forward_train(). Its generic internal forward pass
    # performs the same policy computation without the no-grad wrapper used by
    # forward_inference(), which is exactly what saliency needs.
    output = module._forward({Columns.OBS: tensor})
    selected_logit = output[Columns.ACTION_DIST_INPUTS][0, action]
    selected_logit.backward()
    if tensor.grad is None:
        raise RuntimeError("Policy saliency calculation produced no observation gradient.")
    return tensor.grad.detach().abs().squeeze(0).cpu().numpy()


def evaluate(args: argparse.Namespace) -> dict[str, float | int | bool]:
    register_dc_environment()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        logging_level="ERROR",
        num_cpus=1,
    )
    algorithm = Algorithm.from_checkpoint(checkpoint)

    env = DCMotorCuriosityEnv(
        {
            "seed": args.seed,
            "episode_length": args.episode_length,
            "profile": args.profile,
            "training_mode": False,
            "domain_randomization": False,
            "parameter_jump": False,
            "num_voltage_levels": 21,
            "reference_scale": args.evaluation_reference_scale,
        }
    )
    observation, _ = env.reset(seed=args.seed)
    policy_module = algorithm.get_module(DEFAULT_MODULE_ID)
    if policy_module is None:
        raise RuntimeError("The checkpoint does not contain the default policy module.")

    logs: dict[str, list[float]] = {
        "time_s": [],
        "reference_rad_s": [],
        "omega_rad_s": [],
        "current_A": [],
        "voltage_V": [],
        "electromagnetic_torque_Nm": [],
        "extrinsic_reward": [],
    }
    for index in range(len(OBSERVATION_LABELS)):
        logs[f"saliency_{index}"] = []
    total_return = 0.0
    terminated = False
    truncated = False

    try:
        while not (terminated or truncated):
            action = _deterministic_policy_action(policy_module, observation)
            saliency = _observation_saliency(policy_module, observation, action)
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += reward

            logs["time_s"].append(info["time"])
            logs["reference_rad_s"].append(info["reference"])
            logs["omega_rad_s"].append(info["omega"])
            logs["current_A"].append(info["current"])
            logs["voltage_V"].append(info["voltage"])
            logs["electromagnetic_torque_Nm"].append(
                info["electromagnetic_torque"]
            )
            logs["extrinsic_reward"].append(reward)
            for index, value in enumerate(saliency):
                logs[f"saliency_{index}"].append(float(value))
    finally:
        env.close()
        algorithm.stop()
        ray.shutdown()

    time_s = np.asarray(logs["time_s"], dtype=float)
    reference = np.asarray(logs["reference_rad_s"], dtype=float)
    omega = np.asarray(logs["omega_rad_s"], dtype=float)
    current = np.asarray(logs["current_A"], dtype=float)
    voltage = np.asarray(logs["voltage_V"], dtype=float)
    error = reference - omega

    if len(time_s) < 2:
        raise RuntimeError("Evaluation produced fewer than two samples.")

    metrics: dict[str, float | int | bool] = {
        "episode_return_extrinsic": float(total_return),
        "MAE_speed_rad_s": float(np.mean(np.abs(error))),
        "RMSE_speed_rad_s": float(np.sqrt(np.mean(error**2))),
        "max_abs_speed_error_rad_s": float(np.max(np.abs(error))),
        "IAE_speed": float(np.trapezoid(np.abs(error), time_s)),
        "ISE_speed": float(np.trapezoid(error**2, time_s)),
        "mean_abs_current_A": float(np.mean(np.abs(current))),
        "max_abs_current_A": float(np.max(np.abs(current))),
        "mean_abs_voltage_V": float(np.mean(np.abs(voltage))),
        "RMS_voltage_V": float(np.sqrt(np.mean(voltage**2))),
        "time_at_voltage_rail_s": float(
            np.sum(
                np.diff(time_s)
                * (np.abs(voltage[:-1]) >= 0.98 * env.Vdc)
            )
        ),
        "total_voltage_variation_V": float(np.sum(np.abs(np.diff(voltage)))),
        "absolute_electrical_energy_J": float(
            np.trapezoid(np.abs(voltage * current), time_s)
        ),
        "evaluation_steps": int(len(time_s)),
        "safety_violation": bool(terminated),
    }

    segment_rows = (
        compute_segment_metrics(time_s, omega, reference, voltage, env.Vdc)
        if env.reference.is_piecewise_constant
        else []
    )

    print("\nGlobal evaluation metrics")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:36s}: {value:.6f}")
        else:
            print(f"{key:36s}: {value}")
    print_segment_metrics(segment_rows)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (
            DEFAULT_EVALUATION_OUTPUT_ROOT
            / f"{checkpoint.parent.name}_{args.profile}_seed{args.seed}"
        ).resolve()
    )
    save_evaluation_outputs(
        logs,
        metrics,
        segment_rows,
        output_dir,
        current_limit=env.current_limit,
        show=args.show,
    )
    print(f"Saved evaluation outputs: {output_dir}")
    return metrics


def _run_comparison_episode(
    env: DCMotorCuriosityEnv,
    action_function: Any,
    seed: int,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    logs: dict[str, Any] = {
        "time_s": [],
        "reference_rad_s": [],
        "omega_rad_s": [],
        "current_A": [],
        "voltage_V": [],
        "extrinsic_reward": [],
        "action_index": [],
        "observations": [],
    }
    total_return = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = int(action_function(observation, env))
        logs["observations"].append(np.asarray(observation, dtype=np.float32).copy())
        logs["action_index"].append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        total_return += reward
        logs["time_s"].append(float(info["time"]))
        logs["reference_rad_s"].append(float(info["reference"]))
        logs["omega_rad_s"].append(float(info["omega"]))
        logs["current_A"].append(float(info["current"]))
        logs["voltage_V"].append(float(info["voltage"]))
        logs["extrinsic_reward"].append(float(reward))

    logs["episode_return"] = float(total_return)
    logs["safety_violation"] = bool(terminated)
    return logs


def _comparison_metrics(
    logs: dict[str, Any],
    *,
    voltage_limit: float,
    current_limit: float,
) -> dict[str, float | int | bool]:
    time_s = np.asarray(logs["time_s"], dtype=float)
    reference = np.asarray(logs["reference_rad_s"], dtype=float)
    omega = np.asarray(logs["omega_rad_s"], dtype=float)
    current = np.asarray(logs["current_A"], dtype=float)
    voltage = np.asarray(logs["voltage_V"], dtype=float)
    error = reference - omega
    dt_samples = np.diff(time_s)

    return {
        "episode_return_extrinsic": float(logs["episode_return"]),
        "MAE_speed_rad_s": float(np.mean(np.abs(error))),
        "RMSE_speed_rad_s": float(np.sqrt(np.mean(error**2))),
        "IAE_speed": float(np.trapezoid(np.abs(error), time_s)),
        "ISE_speed": float(np.trapezoid(error**2, time_s)),
        "mean_abs_current_A": float(np.mean(np.abs(current))),
        "RMS_current_A": float(np.sqrt(np.mean(current**2))),
        "max_abs_current_A": float(np.max(np.abs(current))),
        "current_I2t_A2s": float(np.trapezoid(current**2, time_s)),
        "peak_current_utilization_percent": float(
            100.0 * np.max(np.abs(current)) / current_limit
        ),
        "time_above_current_scale_s": float(
            np.sum(dt_samples * (np.abs(current[:-1]) > current_limit))
        ),
        "RMS_voltage_V": float(np.sqrt(np.mean(voltage**2))),
        "total_voltage_variation_V": float(np.sum(np.abs(np.diff(voltage)))),
        "time_at_voltage_rail_s": float(
            np.sum(dt_samples * (np.abs(voltage[:-1]) >= 0.98 * voltage_limit))
        ),
        "absolute_electrical_energy_J": float(
            np.trapezoid(np.abs(voltage * current), time_s)
        ),
        "evaluation_steps": int(len(time_s)),
        "safety_violation": bool(logs["safety_violation"]),
    }


def _save_comparison_outputs(
    ppo_logs: dict[str, Any],
    pi_logs: dict[str, Any],
    ppo_metrics: dict[str, Any],
    pi_metrics: dict[str, Any],
    *,
    output_dir: Path,
    ppo_label: str,
    intrinsic_reward_coeff: float,
    current_limit: float,
    pi_kp: float,
    pi_ki: float,
    tsne_samples: int,
    seed: int,
    show: bool,
) -> None:
    from sklearn.manifold import TSNE

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "PPO": {
            "label": ppo_label,
            "intrinsic_reward_coeff": intrinsic_reward_coeff,
            "metrics": ppo_metrics,
        },
        "PI": {"metrics": pi_metrics},
        "PI_parameters": {"kp": pi_kp, "ki": pi_ki},
    }
    with (output_dir / "comparison_metrics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(comparison, stream, indent=2, sort_keys=True)

    metric_names = list(ppo_metrics.keys())
    with (output_dir / "comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", ppo_label, "PI"])
        for metric_name in metric_names:
            writer.writerow(
                [metric_name, ppo_metrics[metric_name], pi_metrics[metric_name]]
            )

    time_keys = (
        "time_s",
        "reference_rad_s",
        "omega_rad_s",
        "current_A",
        "voltage_V",
        "extrinsic_reward",
        "action_index",
    )
    with (output_dir / "comparison_timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["controller", *time_keys, *[f"observation_{i}" for i in range(6)]]
        )
        for controller_name, logs in (
            (ppo_label, ppo_logs),
            ("PI", pi_logs),
        ):
            for index in range(len(logs["time_s"])):
                writer.writerow(
                    [
                        controller_name,
                        *[logs[key][index] for key in time_keys],
                        *logs["observations"][index],
                    ]
                )

    ppo_time = np.asarray(ppo_logs["time_s"])
    pi_time = np.asarray(pi_logs["time_s"])
    reference = np.asarray(ppo_logs["reference_rad_s"])
    pi_reference = np.asarray(pi_logs["reference_rad_s"])
    ppo_speed = np.asarray(ppo_logs["omega_rad_s"])
    pi_speed = np.asarray(pi_logs["omega_rad_s"])
    ppo_current = np.asarray(ppo_logs["current_A"])
    pi_current = np.asarray(pi_logs["current_A"])
    ppo_voltage = np.asarray(ppo_logs["voltage_V"])
    pi_voltage = np.asarray(pi_logs["voltage_V"])

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(ppo_time, reference, "k--", label="reference")
    axes[0].plot(ppo_time, ppo_speed, label=ppo_label)
    axes[0].plot(pi_time, pi_speed, label="PI", alpha=0.85)
    axes[0].set_ylabel("Speed [rad/s]")
    axes[0].set_title(f"{ppo_label} versus classical PI")
    axes[0].legend(ncol=3)
    axes[1].plot(ppo_time, reference - ppo_speed, label=ppo_label)
    axes[1].plot(pi_time, pi_reference - pi_speed, label="PI")
    axes[1].set_ylabel("Error [rad/s]")
    axes[2].plot(ppo_time, ppo_current, label=ppo_label)
    axes[2].plot(pi_time, pi_current, label="PI", alpha=0.8)
    axes[2].axhline(current_limit, color="tab:red", linestyle="--", linewidth=1)
    axes[2].axhline(-current_limit, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Current [A]")
    axes[3].step(ppo_time, ppo_voltage, where="post", label=ppo_label)
    axes[3].step(pi_time, pi_voltage, where="post", label="PI", alpha=0.8)
    axes[3].set_ylabel("Voltage [V]")
    axes[3].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_tracking.png", dpi=180)

    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7.5))
    axes2[0].plot(ppo_time, np.abs(ppo_current), label=ppo_label)
    axes2[0].plot(pi_time, np.abs(pi_current), label="PI", alpha=0.8)
    axes2[0].axhline(
        current_limit,
        color="tab:red",
        linestyle="--",
        label=f"configured current scale ({current_limit:g} A)",
    )
    axes2[0].set_ylabel("Absolute current [A]")
    axes2[0].set_title("Motor-current loading comparison")
    axes2[0].legend()

    plot_current_utilization_histogram(
        axes2[1], {ppo_label: ppo_current, "PI": pi_current}, current_limit
    )
    for axis in axes2:
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    fig2.tight_layout()
    fig2.savefig(output_dir / "comparison_current_loading.png", dpi=180)

    observations: list[np.ndarray] = []
    voltages: list[np.ndarray] = []
    controllers: list[np.ndarray] = []
    samples_per_controller = max(25, tsne_samples // 2)
    for controller_index, logs in enumerate((ppo_logs, pi_logs)):
        obs = np.asarray(logs["observations"], dtype=float)
        voltage = np.asarray(logs["voltage_V"], dtype=float)
        count = min(samples_per_controller, len(obs))
        indices = np.linspace(0, len(obs) - 1, count, dtype=int)
        observations.append(obs[indices])
        voltages.append(voltage[indices])
        controllers.append(np.full(count, controller_index, dtype=int))

    all_observations = np.vstack(observations)
    all_voltages = np.concatenate(voltages)
    all_controllers = np.concatenate(controllers)
    observation_std = np.std(all_observations, axis=0)
    standardized = (
        all_observations - np.mean(all_observations, axis=0)
    ) / np.where(observation_std > 1e-8, observation_std, 1.0)
    perplexity = min(30.0, max(5.0, (len(standardized) - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(standardized)

    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    scatter = None
    for controller_index, (axis, title) in enumerate(
        zip(axes3, (ppo_label, "PI"))
    ):
        mask = all_controllers == controller_index
        scatter = axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=all_voltages[mask],
            cmap="coolwarm",
            vmin=-24.0,
            vmax=24.0,
            s=12,
            alpha=0.78,
        )
        axis.set_title(title)
        axis.set_xlabel("t-SNE component 1")
        axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    axes3[0].set_ylabel("t-SNE component 2")
    if scatter is not None:
        fig3.colorbar(scatter, ax=axes3, label="Selected/applied voltage [V]")
    fig3.suptitle("Observed state and action-space map (t-SNE)")
    fig3.subplots_adjust(left=0.08, right=0.88, bottom=0.11, top=0.88, wspace=0.08)
    fig3.savefig(output_dir / "comparison_tsne_action_space.png", dpi=180)

    if show:
        plt.show()
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)


def _checkpoint_intrinsic_reward_coeff(algorithm: Algorithm) -> float:
    learner_settings = getattr(algorithm.config, "learner_config_dict", {}) or {}
    return float(learner_settings.get("intrinsic_reward_coeff", 0.0))


def _save_all_controller_comparison(
    controller_logs: dict[str, dict[str, Any]],
    controller_metrics: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    current_limit: float,
    curiosity_coeff: float,
    pi_kp: float,
    pi_ki: float,
    tsne_samples: int,
    seed: int,
    profile: str,
    show: bool,
) -> None:
    """Save joint and pairwise PPO, curiosity-PPO, and PI comparisons."""

    from sklearn.manifold import TSNE

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "profile": profile,
        "seed": seed,
        "curiosity_intrinsic_reward_coeff": curiosity_coeff,
        "PI_parameters": {"kp": pi_kp, "ki": pi_ki},
        "metrics": controller_metrics,
    }
    with (output_dir / "comparison_all_metrics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)

    labels = list(controller_metrics)
    metric_names = list(next(iter(controller_metrics.values())))
    with (output_dir / "comparison_all_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", *labels])
        for metric_name in metric_names:
            writer.writerow(
                [
                    metric_name,
                    *[
                        controller_metrics[label][metric_name]
                        for label in labels
                    ],
                ]
            )

    lower_is_better = {
        "MAE_speed_rad_s",
        "RMSE_speed_rad_s",
        "IAE_speed",
        "ISE_speed",
        "mean_abs_current_A",
        "RMS_current_A",
        "max_abs_current_A",
        "current_I2t_A2s",
        "peak_current_utilization_percent",
        "time_above_current_scale_s",
        "RMS_voltage_V",
        "total_voltage_variation_V",
        "time_at_voltage_rail_s",
        "absolute_electrical_energy_J",
    }
    higher_is_better = {"episode_return_extrinsic"}
    with (output_dir / "comparison_pairwise.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "controller_a",
                "controller_b",
                "metric",
                "value_a",
                "value_b",
                "a_minus_b",
                "relative_difference_percent_vs_b",
                "better_controller",
            ]
        )
        for label_a, label_b in combinations(labels, 2):
            for metric_name in metric_names:
                value_a = controller_metrics[label_a][metric_name]
                value_b = controller_metrics[label_b][metric_name]
                if isinstance(value_a, bool) or metric_name == "evaluation_steps":
                    continue
                numeric_a = float(value_a)
                numeric_b = float(value_b)
                difference = numeric_a - numeric_b
                relative = (
                    100.0 * difference / abs(numeric_b)
                    if abs(numeric_b) > 1e-12
                    else float("nan")
                )
                better = "equal"
                if metric_name in lower_is_better and numeric_a != numeric_b:
                    better = label_a if numeric_a < numeric_b else label_b
                elif metric_name in higher_is_better and numeric_a != numeric_b:
                    better = label_a if numeric_a > numeric_b else label_b
                writer.writerow(
                    [
                        label_a,
                        label_b,
                        metric_name,
                        numeric_a,
                        numeric_b,
                        difference,
                        relative,
                        better,
                    ]
                )

    time_keys = (
        "time_s",
        "reference_rad_s",
        "omega_rad_s",
        "current_A",
        "voltage_V",
        "extrinsic_reward",
        "action_index",
    )
    with (output_dir / "comparison_all_timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["controller", *time_keys, *[f"observation_{i}" for i in range(6)]]
        )
        for label, logs in controller_logs.items():
            for index in range(len(logs["time_s"])):
                writer.writerow(
                    [
                        label,
                        *[logs[key][index] for key in time_keys],
                        *logs["observations"][index],
                    ]
                )

    first_logs = next(iter(controller_logs.values()))
    reference = np.asarray(first_logs["reference_rad_s"])
    reference_time = np.asarray(first_logs["time_s"])
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(reference_time, reference, "k--", label="reference")
    for label, logs in controller_logs.items():
        time_s = np.asarray(logs["time_s"])
        speed = np.asarray(logs["omega_rad_s"])
        local_reference = np.asarray(logs["reference_rad_s"])
        axes[0].plot(time_s, speed, label=label, alpha=0.88)
        axes[1].plot(time_s, local_reference - speed, label=label, alpha=0.88)
        axes[2].plot(time_s, logs["current_A"], label=label, alpha=0.82)
        axes[3].step(
            time_s,
            logs["voltage_V"],
            where="post",
            label=label,
            alpha=0.78,
        )
    axes[0].set_ylabel("Speed [rad/s]")
    axes[0].set_title("PPO baseline, curiosity PPO, and PI")
    axes[0].legend(ncol=4)
    axes[1].set_ylabel("Error [rad/s]")
    axes[2].axhline(current_limit, color="tab:red", linestyle="--", linewidth=1)
    axes[2].axhline(-current_limit, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Current [A]")
    axes[3].set_ylabel("Voltage [V]")
    axes[3].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_all_tracking.png", dpi=180)

    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7.5))
    histogram_currents: dict[str, np.ndarray] = {}
    for label, logs in controller_logs.items():
        time_s = np.asarray(logs["time_s"])
        abs_current = np.abs(np.asarray(logs["current_A"]))
        histogram_currents[label] = abs_current
        axes2[0].plot(time_s, abs_current, label=label, alpha=0.85)
    axes2[0].axhline(
        current_limit,
        color="tab:red",
        linestyle="--",
        label=f"configured current scale ({current_limit:g} A)",
    )
    axes2[0].set_ylabel("Absolute current [A]")
    axes2[0].set_title("Motor-current loading: all controllers")
    axes2[0].legend()
    plot_current_utilization_histogram(
        axes2[1], histogram_currents, current_limit
    )
    for axis in axes2:
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    fig2.tight_layout()
    fig2.savefig(output_dir / "comparison_all_current_loading.png", dpi=180)

    observation_blocks: list[np.ndarray] = []
    voltage_blocks: list[np.ndarray] = []
    controller_blocks: list[np.ndarray] = []
    samples_per_controller = max(25, tsne_samples // len(controller_logs))
    for controller_index, logs in enumerate(controller_logs.values()):
        observations = np.asarray(logs["observations"], dtype=float)
        voltages = np.asarray(logs["voltage_V"], dtype=float)
        count = min(samples_per_controller, len(observations))
        indices = np.linspace(0, len(observations) - 1, count, dtype=int)
        observation_blocks.append(observations[indices])
        voltage_blocks.append(voltages[indices])
        controller_blocks.append(
            np.full(count, controller_index, dtype=int)
        )

    all_observations = np.vstack(observation_blocks)
    all_voltages = np.concatenate(voltage_blocks)
    all_controller_indices = np.concatenate(controller_blocks)
    observation_std = np.std(all_observations, axis=0)
    standardized = (
        all_observations - np.mean(all_observations, axis=0)
    ) / np.where(observation_std > 1e-8, observation_std, 1.0)
    perplexity = min(30.0, max(5.0, (len(standardized) - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(standardized)

    fig3, axes3 = plt.subplots(
        1,
        len(controller_logs),
        figsize=(15, 5),
        sharex=True,
        sharey=True,
    )
    scatter = None
    for controller_index, (axis, label) in enumerate(
        zip(np.atleast_1d(axes3), controller_logs)
    ):
        mask = all_controller_indices == controller_index
        scatter = axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=all_voltages[mask],
            cmap="coolwarm",
            vmin=-24.0,
            vmax=24.0,
            s=11,
            alpha=0.76,
        )
        axis.set_title(label)
        axis.set_xlabel("t-SNE component 1")
        axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    np.atleast_1d(axes3)[0].set_ylabel("t-SNE component 2")
    if scatter is not None:
        fig3.colorbar(
            scatter,
            ax=np.atleast_1d(axes3).tolist(),
            label="Selected/applied voltage [V]",
        )
    fig3.suptitle("Observed state and action-space map: all controllers")
    fig3.subplots_adjust(left=0.06, right=0.91, bottom=0.12, top=0.86, wspace=0.08)
    fig3.savefig(output_dir / "comparison_all_tsne_action_space.png", dpi=180)

    if show:
        plt.show()
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)


def compare_all_controllers(
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Compare PPO baseline, curiosity PPO, and PI on identical conditions."""

    register_dc_environment()
    ppo_checkpoint = Path(args.ppo_checkpoint).resolve()
    curiosity_checkpoint = Path(args.curiosity_checkpoint).resolve()
    for checkpoint in (ppo_checkpoint, curiosity_checkpoint):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    common_env_config = {
        "seed": args.seed,
        "episode_length": args.episode_length,
        "profile": args.profile,
        "training_mode": False,
        "domain_randomization": False,
        "parameter_jump": False,
        "num_voltage_levels": 21,
        "reference_scale": args.evaluation_reference_scale,
    }
    controller_logs: dict[str, dict[str, Any]] = {}
    curiosity_coeff = 0.0

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        logging_level="ERROR",
        num_cpus=1,
    )
    try:
        for label, checkpoint, expected_curiosity in (
            ("PPO baseline", ppo_checkpoint, False),
            ("PPO + curiosity", curiosity_checkpoint, True),
        ):
            algorithm = Algorithm.from_checkpoint(checkpoint)
            try:
                intrinsic_coeff = _checkpoint_intrinsic_reward_coeff(algorithm)
                if expected_curiosity and intrinsic_coeff <= 0.0:
                    raise ValueError(
                        "The --curiosity-checkpoint has no intrinsic curiosity "
                        f"reward (coefficient={intrinsic_coeff:g}): {checkpoint}"
                    )
                if not expected_curiosity and intrinsic_coeff > 0.0:
                    raise ValueError(
                        "The --ppo-checkpoint contains curiosity "
                        f"(coefficient={intrinsic_coeff:g}): {checkpoint}"
                    )
                if expected_curiosity:
                    curiosity_coeff = intrinsic_coeff
                module = algorithm.get_module(DEFAULT_MODULE_ID)
                if module is None:
                    raise RuntimeError(
                        f"Checkpoint has no default policy module: {checkpoint}"
                    )
                env = DCMotorCuriosityEnv(common_env_config)
                try:
                    controller_logs[label] = _run_comparison_episode(
                        env,
                        lambda observation, _env, policy=module: (
                            _deterministic_policy_action(policy, observation)
                        ),
                        args.seed,
                    )
                finally:
                    env.close()
            finally:
                algorithm.stop()

        pi_env = DCMotorCuriosityEnv(common_env_config)
        pi_controller = PIController(kp=args.pi_kp, ki=args.pi_ki)
        pi_controller.reset()
        try:
            controller_logs["PI"] = _run_comparison_episode(
                pi_env,
                lambda observation, env: pi_controller.action(observation, env),
                args.seed,
            )
        finally:
            pi_env.close()
    finally:
        ray.shutdown()

    controller_metrics = {
        label: _comparison_metrics(
            logs,
            voltage_limit=24.0,
            current_limit=6.0,
        )
        for label, logs in controller_logs.items()
    }

    print("\nPPO baseline vs PPO + curiosity vs PI")
    print(f"Curiosity intrinsic reward coefficient: {curiosity_coeff:g}")
    print(f"PI gains: Kp={args.pi_kp:g}, Ki={args.pi_ki:g}")
    labels = list(controller_metrics)
    print(f"{'metric':34s} | " + " | ".join(f"{label:>18s}" for label in labels))
    print("-" * (38 + 21 * len(labels)))
    for metric_name in next(iter(controller_metrics.values())):
        values = [controller_metrics[label][metric_name] for label in labels]
        if isinstance(values[0], bool):
            formatted = [f"{str(value):>18s}" for value in values]
        elif isinstance(values[0], (int, np.integer)):
            formatted = [f"{int(value):18d}" for value in values]
        else:
            formatted = [f"{float(value):18.6f}" for value in values]
        print(f"{metric_name:34s} | " + " | ".join(formatted))

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (
            DEFAULT_COMPARISON_OUTPUT_ROOT
            / (
                f"{ppo_checkpoint.parent.name}_vs_"
                f"{curiosity_checkpoint.parent.name}_vs_pi_"
                f"{args.profile}_seed{args.seed}"
            )
        ).resolve()
    )
    _save_all_controller_comparison(
        controller_logs,
        controller_metrics,
        output_dir=output_dir,
        current_limit=6.0,
        curiosity_coeff=curiosity_coeff,
        pi_kp=args.pi_kp,
        pi_ki=args.pi_ki,
        tsne_samples=args.tsne_samples,
        seed=args.seed,
        profile=args.profile,
        show=args.show,
    )
    print(f"Saved all-controller comparison outputs: {output_dir}")
    return controller_metrics


def compare_with_pi(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    register_dc_environment()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        logging_level="ERROR",
        num_cpus=1,
    )
    algorithm = Algorithm.from_checkpoint(checkpoint)
    intrinsic_reward_coeff = _checkpoint_intrinsic_reward_coeff(algorithm)
    ppo_label = (
        "PPO + curiosity"
        if intrinsic_reward_coeff > 0.0
        else "PPO (no curiosity)"
    )
    policy_module = algorithm.get_module(DEFAULT_MODULE_ID)
    if policy_module is None:
        algorithm.stop()
        ray.shutdown()
        raise RuntimeError("The checkpoint does not contain the default policy module.")

    common_env_config = {
        "seed": args.seed,
        "episode_length": args.episode_length,
        "profile": args.profile,
        "training_mode": False,
        "domain_randomization": False,
        "parameter_jump": False,
        "num_voltage_levels": 21,
        "reference_scale": args.evaluation_reference_scale,
    }
    ppo_env = DCMotorCuriosityEnv(common_env_config)
    pi_env = DCMotorCuriosityEnv(common_env_config)
    pi_controller = PIController(kp=args.pi_kp, ki=args.pi_ki)
    pi_controller.reset()

    try:
        ppo_logs = _run_comparison_episode(
            ppo_env,
            lambda observation, _env: _deterministic_policy_action(
                policy_module, observation
            ),
            args.seed,
        )
        pi_logs = _run_comparison_episode(
            pi_env,
            lambda observation, env: pi_controller.action(observation, env),
            args.seed,
        )
    finally:
        ppo_env.close()
        pi_env.close()
        algorithm.stop()
        ray.shutdown()

    ppo_metrics = _comparison_metrics(
        ppo_logs,
        voltage_limit=ppo_env.Vdc,
        current_limit=ppo_env.current_limit,
    )
    pi_metrics = _comparison_metrics(
        pi_logs,
        voltage_limit=pi_env.Vdc,
        current_limit=pi_env.current_limit,
    )

    print(f"\n{ppo_label} versus PI")
    print(f"Checkpoint intrinsic reward coefficient: {intrinsic_reward_coeff:g}")
    print(f"PI gains: Kp={args.pi_kp:g}, Ki={args.pi_ki:g}")
    print(f"{'metric':36s} | {ppo_label:>17s} | {'PI':>17s}")
    print("-" * 76)
    for key in ppo_metrics:
        ppo_value = ppo_metrics[key]
        pi_value = pi_metrics[key]
        if isinstance(ppo_value, bool):
            print(f"{key:36s} | {str(ppo_value):>17s} | {str(pi_value):>17s}")
        elif isinstance(ppo_value, (int, np.integer)):
            print(f"{key:36s} | {int(ppo_value):17d} | {int(pi_value):17d}")
        else:
            print(f"{key:36s} | {float(ppo_value):17.6f} | {float(pi_value):17.6f}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (
            DEFAULT_COMPARISON_OUTPUT_ROOT
            / f"{checkpoint.parent.name}_vs_pi_{args.profile}_seed{args.seed}"
        ).resolve()
    )
    _save_comparison_outputs(
        ppo_logs,
        pi_logs,
        ppo_metrics,
        pi_metrics,
        output_dir=output_dir,
        ppo_label=ppo_label,
        intrinsic_reward_coeff=intrinsic_reward_coeff,
        current_limit=ppo_env.current_limit,
        pi_kp=args.pi_kp,
        pi_ki=args.pi_ki,
        tsne_samples=args.tsne_samples,
        seed=args.seed,
        show=args.show,
    )
    print(f"Saved comparison outputs: {output_dir}")
    return {"PPO": ppo_metrics, "PI": pi_metrics}


def smoke_test(args: argparse.Namespace) -> None:
    """Exercise env, training, checkpoint restore, inference, and evaluation."""

    smoke_root = Path(tempfile.mkdtemp(prefix="dc_curiosity_smoke_"))
    print(f"Smoke-test directory: {smoke_root}")

    train_args = argparse.Namespace(**vars(args))
    train_args.seed = 7
    train_args.iterations = 1
    train_args.episode_length = 0.10
    train_args.train_batch_size = 64
    train_args.minibatch_size = 32
    train_args.num_epochs = 1
    train_args.num_env_runners = 0
    train_args.intrinsic_reward_coeff = 0.01
    train_args.forward_loss_weight = 0.2
    train_args.no_domain_randomization = True
    train_args.no_parameter_jump = True
    train_args.checkpoint_frequency = 1
    train_args.model_dir = str(smoke_root / "model")
    train_args.output_dir = str(smoke_root / "train_outputs")

    checkpoint = train(train_args)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.seed = 11
    eval_args.checkpoint = str(checkpoint)
    eval_args.episode_length = 0.10
    eval_args.profile = "steps"
    eval_args.output_dir = str(smoke_root / "eval_outputs")
    eval_args.show = False
    metrics = evaluate(eval_args)

    required_outputs = [
        smoke_root / "train_outputs" / "training_history.csv",
        smoke_root / "train_outputs" / "training_history.png",
        smoke_root / "eval_outputs" / "evaluation_timeseries.csv",
        smoke_root / "eval_outputs" / "evaluation_metrics.json",
        smoke_root / "eval_outputs" / "evaluation_tracking.png",
        smoke_root
        / "eval_outputs"
        / "evaluation_state_and_current_loading.png",
        smoke_root / "eval_outputs" / "evaluation_observation_saliency.png",
        smoke_root / "eval_outputs" / "observation_saliency.json",
    ]
    missing = [path for path in required_outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"Smoke test did not create: {missing}")
    if not np.isfinite(metrics["RMSE_speed_rad_s"]):
        raise RuntimeError("Smoke test produced a non-finite RMSE.")

    print("Smoke test passed: training, checkpoint restore, evaluation, and plots work.")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curiosity-driven DC speed control with RLlib PPO + ICM."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--train",
        action="store_true",
        help="Legacy configurable training mode.",
    )
    mode.add_argument(
        "--train-ppo",
        action="store_true",
        help="Train a PPO baseline without intrinsic curiosity reward.",
    )
    mode.add_argument(
        "--train-curiosity",
        action="store_true",
        help="Train PPO + ICM curiosity (defaults: intrinsic=0.02, forward=0.2).",
    )
    mode.add_argument("--eval", action="store_true", help="Evaluate a checkpoint.")
    mode.add_argument(
        "--compare-pi",
        action="store_true",
        help="Compare a PPO + ICM checkpoint with a classical PI controller.",
    )
    mode.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare PPO baseline, curiosity PPO, and PI together.",
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a very short end-to-end integration test.",
    )

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episode-length", type=float, default=8.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument(
        "--policy-hidden-sizes",
        type=int,
        nargs="+",
        default=[16, 8],
        metavar="N",
        help="Policy/value hidden-layer sizes (default: 16 8).",
    )
    parser.add_argument(
        "--train-reference-scale",
        type=float,
        default=1.0,
        help="Training reference amplitude fraction in (0, 1].",
    )
    parser.add_argument(
        "--evaluation-reference-scale",
        type=float,
        default=1.0,
        help="Evaluation reference amplitude fraction in (0, 1].",
    )
    parser.add_argument("--intrinsic-reward-coeff", type=float, default=None)
    parser.add_argument("--forward-loss-weight", type=float, default=None)
    parser.add_argument("--checkpoint-frequency", type=int, default=10)
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--no-parameter-jump", action="store_true")

    parser.add_argument(
        "--profile",
        choices=SpeedReference.SUPPORTED,
        default="mixed",
        help="Evaluation reference profile.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint directory required by --eval or --compare-pi.",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=str,
        default="",
        help="PPO baseline checkpoint required by --compare-all.",
    )
    parser.add_argument(
        "--curiosity-checkpoint",
        type=str,
        default="",
        help="Curiosity PPO checkpoint required by --compare-all.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Unique training run name used for automatic model/output folders.",
    )
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Explicitly allow a training run to reuse non-empty output paths.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help=(
            "Continue from an iteration_NNNN checkpoint; --iterations is the "
            "final target iteration."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display evaluation figures in addition to saving them.",
    )
    parser.add_argument("--pi-kp", type=float, default=0.03)
    parser.add_argument("--pi-ki", type=float, default=3.0)
    parser.add_argument(
        "--tsne-samples",
        type=int,
        default=1200,
        help="Total controller samples used by the t-SNE action-space plot.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    if args.checkpoint_frequency <= 0:
        parser.error("--checkpoint-frequency must be positive.")
    if args.episode_length <= 0.0:
        parser.error("--episode-length must be positive.")
    if not 0.0 < args.train_reference_scale <= 1.0:
        parser.error("--train-reference-scale must be in (0, 1].")
    if not 0.0 < args.evaluation_reference_scale <= 1.0:
        parser.error("--evaluation-reference-scale must be in (0, 1].")
    if not args.policy_hidden_sizes or any(
        size <= 0 for size in args.policy_hidden_sizes
    ):
        parser.error("--policy-hidden-sizes must contain positive integers.")

    if args.train_ppo:
        if args.intrinsic_reward_coeff not in (None, 0.0):
            parser.error("--train-ppo requires --intrinsic-reward-coeff 0.0.")
        if args.forward_loss_weight not in (None, 0.0):
            parser.error("--train-ppo requires --forward-loss-weight 0.0.")
        args.intrinsic_reward_coeff = 0.0
        args.forward_loss_weight = 0.0
    elif args.train_curiosity:
        if args.intrinsic_reward_coeff is None:
            args.intrinsic_reward_coeff = 0.02
        if args.forward_loss_weight is None:
            args.forward_loss_weight = 0.2
        if args.intrinsic_reward_coeff <= 0.0:
            parser.error("--train-curiosity requires a positive curiosity coefficient.")
    elif args.train:
        if args.intrinsic_reward_coeff is None:
            args.intrinsic_reward_coeff = 0.0
        if args.forward_loss_weight is None:
            args.forward_loss_weight = 0.0

    if (args.eval or args.compare_pi) and not args.checkpoint:
        parser.error("--eval and --compare-pi require --checkpoint PATH.")
    if args.compare_all and (
        not args.ppo_checkpoint or not args.curiosity_checkpoint
    ):
        parser.error(
            "--compare-all requires --ppo-checkpoint and --curiosity-checkpoint."
        )
    if args.pi_kp < 0.0 or args.pi_ki < 0.0:
        parser.error("--pi-kp and --pi-ki must be non-negative.")
    if args.tsne_samples < 50:
        parser.error("--tsne-samples must be at least 50.")

    if args.train or args.train_ppo or args.train_curiosity:
        train(args)
    elif args.eval:
        evaluate(args)
    elif args.compare_pi:
        compare_with_pi(args)
    elif args.compare_all:
        compare_all_controllers(args)
    else:
        smoke_test(args)


if __name__ == "__main__":
    main()
