"""Versioned continuous voltage environments; old checkpoint spaces are untouched."""
import gymnasium as gym
import numpy as np
from ray.tune.registry import register_env

ENV_NAMES = {m: f"{m}_continuous_voltage_v1" for m in ("dc", "pmsm")}


class ContinuousVoltageEnv(gym.Env):
    def __init__(self, config):
        from sim_DC_RLlib_Curiosity import DCMotorCuriosityEnv
        from sim_PMSM_RLlib_Curiosity import PMSMSpeedCuriosityEnv
        self.motor = config["motor"]
        self.base = (DCMotorCuriosityEnv if self.motor == "dc" else PMSMSpeedCuriosityEnv)(config)
        self.base.substep_smoothness_weight = 0.0
        self.vmax = self.base.Vdc if self.motor == "dc" else self.base.Vmax_nominal
        self.dim = 1 if self.motor == "dc" else 2
        self.control_period = self.base.dt * self.base.frame_skip
        self.slew = float(config.get("command_slew_v_per_s", 1000.0))
        self.smoothness_weight = float(config.get("command_smoothness_weight", 0.05))
        if not np.isfinite(self.slew) or self.slew < 0 or not np.isfinite(self.smoothness_weight) or self.smoothness_weight < 0:
            raise ValueError("Slew and smoothness weight must be finite and nonnegative")
        self.action_space = gym.spaces.Box(-1., 1., (self.dim,), np.float32)
        # Explicit actuator/previous-command state avoids hidden slew/penalty memory.
        self.observation_space = gym.spaces.Box(-1., 1., (self.base.observation_space.shape[0]+2*self.dim,), np.float32)
        self.requested = np.zeros(self.dim)
        self.limited = np.zeros(self.dim)

    def _obs(self, obs):
        return np.concatenate((obs, self.requested/self.vmax, self.limited/self.vmax)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.requested = np.zeros(self.dim)
        self.limited = np.zeros(self.dim)
        obs, info = self.base.reset(seed=seed, options=options)
        return self._obs(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.dim,) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite action of shape {(self.dim,)}")
        normalized = np.clip(action, -1, 1)
        normalized /= max(1., np.linalg.norm(normalized))
        requested = normalized * self.vmax
        delta_request = requested-self.requested
        delta = requested-self.limited
        if self.slew > 0:
            delta *= min(1., self.slew*self.control_period/max(np.linalg.norm(delta), 1e-12))
        self.limited = self.limited + delta
        self.requested = requested
        obs, reward, terminated, truncated, info = self.base.step_voltage(float(self.limited[0]) if self.dim == 1 else self.limited)
        # Exactly once per decision. Normalize by the physical voltage radius, not dt.
        penalty = self.smoothness_weight * float(np.sum((delta_request/self.vmax)**2))
        if not terminated:
            reward -= penalty
        info.update(requested_voltage=requested.copy(), limited_voltage=self.limited.copy(),
                    command_smoothness_penalty=penalty)
        return self._obs(obs), reward, terminated, truncated, info

    def close(self):
        self.base.close()


def register_continuous():
    from .matched import ENV_NAMES as MATCHED_NAMES, MatchedCurrentEnv, MatchedVoltageEnv
    from .current_control import ENV_NAMES as CURRENT_NAMES, CurrentReferenceEnv
    def factory(config, cls):
        values = dict(config)
        values["seed"] = int(values.get("seed", 0)) + 10000*getattr(config, "worker_index", 0) + getattr(config, "vector_index", 0)
        return cls(values)
    for name in ENV_NAMES.values():
        register_env(name, lambda config: factory(config, ContinuousVoltageEnv))
    for name in CURRENT_NAMES.values():
        register_env(name, lambda config: factory(config, CurrentReferenceEnv))
    for mode, cls in (("matched-current",MatchedCurrentEnv),("matched-voltage",MatchedVoltageEnv)):
        for name in MATCHED_NAMES[mode].values():
            register_env(name, lambda config, cls=cls: factory(config, cls))


def continuous_config(args):
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.core import DEFAULT_MODULE_ID
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.examples.learners.classes.intrinsic_curiosity_learners import ICM_MODULE_ID, PPOTorchLearnerWithCuriosity
    from .continuous_icm import ContinuousICM
    register_continuous()
    env_config = dict(motor=args.motor, seed=args.seed, episode_length=args.episode_length,
                      profile=args.train_profile, reference_scale=args.train_reference_scale, training_mode=True,
                      domain_randomization=not args.no_domain_randomization,
                      parameter_jump=not args.no_parameter_jump,
                      command_slew_v_per_s=args.command_slew_v_per_s,
                      command_smoothness_weight=args.command_smoothness_weight)
    env_name = ENV_NAMES[args.motor]
    if args.action_mode in ("current","matched-current","matched-voltage"):
        from .current_control import ENV_NAMES as CURRENT_NAMES, environment_config
        env_config.update(environment_config(args))
        env_name = CURRENT_NAMES[args.motor]
        if args.action_mode.startswith("matched-"):
            from .matched import ENV_NAMES as MATCHED_NAMES
            env_name=MATCHED_NAMES[args.action_mode][args.motor]
            env_config.update(architecture=args.action_mode+"_v3",reward_protocol="physical_cost_v3")
    config = (PPOConfig().environment(env_name, env_config=env_config,
                                    normalize_actions=False, clip_actions=True)
              .framework("torch").debugging(seed=args.seed, log_level="ERROR", log_sys_usage=False)
              .env_runners(num_env_runners=args.num_env_runners, batch_mode="truncate_episodes")
              .training(gamma=.99, lambda_=.95, lr=3e-4, clip_param=.2, vf_loss_coeff=.5,
                        vf_clip_param=args.vf_clip_param,
                        entropy_coeff=.005, grad_clip=10., train_batch_size_per_learner=args.train_batch_size,
                        minibatch_size=args.minibatch_size, num_epochs=args.num_epochs))
    specs = {DEFAULT_MODULE_ID: RLModuleSpec(model_config=dict(fcnet_hiddens=list(args.policy_hidden_sizes),
                          fcnet_activation="tanh", vf_share_layers=False))}
    from .stability import enabled, BoundedMeanPPOModule,SpeedObservationPPOModule,SpeedMemoryPPOModule,temporal_parameters
    if enabled(args):
        if args.motor != "pmsm" or args.action_mode != "current":
            raise ValueError("pmsm-v1 requires PMSM current-reference control")
        # At 1 ms: gamma=.99 sees ~0.1 s; .996 extends this to ~0.25 s.
        gamma,gae_lambda=temporal_parameters(args)
        config = config.training(gamma=gamma, lambda_=gae_lambda, lr=1e-4, entropy_coeff=0.0)
        specs[DEFAULT_MODULE_ID].module_class = BoundedMeanPPOModule
        if args.actor_observations=='speed':specs[DEFAULT_MODULE_ID].module_class=SpeedObservationPPOModule
        if args.actor_observations=='speed_memory':specs[DEFAULT_MODULE_ID].module_class=SpeedMemoryPPOModule
        specs[DEFAULT_MODULE_ID].model_config.update(policy_log_std_min=args.policy_log_std_min,
                                                     policy_log_std_max=args.policy_log_std_max)
        if args.critic_profile=='normalized':
            specs[DEFAULT_MODULE_ID].model_config.update(critic_profile='normalized',
                critic_hidden_sizes=args.critic_hidden_sizes,critic_return_scale=args.critic_return_scale,
                critic_learning_rate=args.critic_learning_rate)
    if args.controller == "icm":
        # Actuator memory is intentionally excluded: inverse learning should not
        # trivially recover the action from the stored command in next_obs.
        indices = [0,1,2] if args.motor == "dc" else [0,1,2,3,4,8]
        specs[ICM_MODULE_ID] = RLModuleSpec(module_class=ContinuousICM, learner_only=True,
                                  model_config=dict(physical_indices=indices, feature_dim=16))
        config = config.learners(num_learners=0, num_gpus_per_learner=0,
                 learner_class=PPOTorchLearnerWithCuriosity,
                 learner_config_dict=dict(intrinsic_reward_coeff=args.intrinsic_reward_coeff*args.reward_scale,
                                          forward_loss_weight=args.forward_loss_weight))
        config = config.rl_module(algorithm_config_overrides_per_module={
            ICM_MODULE_ID: AlgorithmConfig.overrides(lr=1e-3)})
    else:
        config = config.learners(num_learners=0, num_gpus_per_learner=0)
    if args.critic_profile=='normalized':
        from .normalized_critic import NormalizedCriticLearner,NormalizedCriticCuriosityLearner
        config=config.learners(learner_class=NormalizedCriticCuriosityLearner if args.controller=='icm'
                               else NormalizedCriticLearner)
    return config.rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=specs))
