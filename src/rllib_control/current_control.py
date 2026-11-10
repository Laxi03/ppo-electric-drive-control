"""V2: scalar current-reference policy above fast PI/FOC current regulation.

Nominal parameters only are used by the controller. Perturbed plant parameters
are never exposed to it. The inner loop runs at every physical integration tick.
"""
import gymnasium as gym
import numpy as np
from .continuous import ContinuousVoltageEnv

ENV_NAMES = {m: f"{m}_current_reference_v2" for m in ("dc", "pmsm")}
CONFIG_KEYS = ("current_reference_limit", "current_reference_slew", "current_bandwidth",
               "tracking_weight", "tracking_absolute_weight", "current_weight", "id_weight")


class CurrentReferenceEnv(ContinuousVoltageEnv):
    def __init__(self, config):
        super().__init__(config)
        self.reference_limit = float(config.get("current_reference_limit", .8)) * self.base.current_limit
        self.reference_slew = float(config.get("current_reference_slew", 500.))
        self.bandwidth = float(config.get("current_bandwidth", 300.))
        self.tracking_weight = float(config.get("tracking_weight", 100.))
        self.absolute_weight = float(config.get("tracking_absolute_weight", 1.))
        self.current_weight = float(config.get("current_weight", .02))
        self.id_weight = float(config.get("id_weight", .1))
        self.current_rate_weight = float(config.get("current_rate_weight", 0.))
        self.current_rate_timescale = float(config.get("current_rate_timescale_s", .02))
        self.reward_scale=float(config.get('reward_scale',1.))
        self.error_observation_gain=float(config.get('error_observation_gain',1.))
        if not np.isfinite(self.reward_scale) or self.reward_scale<=0:
            raise ValueError('Reward scale must be finite and positive')
        if not np.isfinite(self.error_observation_gain) or self.error_observation_gain<=0:
            raise ValueError('Error observation gain must be finite and positive')
        values = [self.reference_limit, self.reference_slew, self.bandwidth,
                  self.tracking_weight, self.absolute_weight, self.current_weight, self.id_weight,
                  self.current_rate_weight, self.current_rate_timescale]
        if not np.isfinite(values).all() or min(values) < 0 or self.reference_limit <= 0 or self.bandwidth <= 0:
            raise ValueError("Current-control settings must be finite and nonnegative; limit/bandwidth positive")
        if self.reference_limit > self.base.current_limit:
            raise ValueError("Current reference limit must not exceed the configured current scale")
        self.action_space = gym.spaces.Box(-1., 1., (1,), np.float32)
        size = self.base.observation_space.shape[0] + 2 + 2*self.dim
        self.observation_space = gym.spaces.Box(-1., 1., (size,), np.float32)
        self.current_request = self.current_reference = 0.
        self.integral = np.zeros(self.dim)
        self.base.current_control_callback = self._inner_voltage
        self.base.reward_callback = self._reward

    def _obs(self, obs):
        if self.error_observation_gain!=1.:
            obs=obs.copy()
            index=4 if self.motor=='dc' else 6
            obs[index]=np.clip(obs[index]*self.error_observation_gain,-1.,1.)
        return np.concatenate((obs, [self.current_request/self.reference_limit,
            self.current_reference/self.reference_limit], self.integral/self.vmax,
            self.limited/self.vmax)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.current_request = self.current_reference = 0.
        self.integral = np.zeros(self.dim)
        return super().reset(seed=seed, options=options)

    def _inner_voltage(self):
        b = self.base
        from .scenarios import feedback
        measured = feedback(b)
        if self.motor == "dc":
            nominal = b.motor.nominal
            error = np.array([self.current_reference-measured["current"]])
            kp = np.array([nominal.L*self.bandwidth])
            ki = nominal.R*self.bandwidth
            ff = np.array([nominal.Ke*measured["speed"]])
        else:
            error = np.array([-measured["id"], self.current_reference-measured["iq"]])
            kp = np.array([b.Ld_nom, b.Lq_nom])*self.bandwidth
            ki = b.R_nom*self.bandwidth
            we = b.params.pole_pairs*measured["speed"]
            ff = np.array([-we*b.Lq_nom*measured["iq"], we*(b.Ld_nom*measured["id"]+b.flux_nom)])
        candidate = self.integral + ki*error*b.dt
        raw = kp*error + candidate + ff
        self.requested = raw * min(1., self.vmax/max(np.linalg.norm(raw), 1e-12))
        delta = self.requested-self.limited
        if self.slew > 0:
            delta *= min(1., self.slew*b.dt/max(np.linalg.norm(delta), 1e-12))
        self.limited += delta
        # Back calculation sees voltage saturation AND the actual slew limiter.
        self.integral = np.clip(candidate + min(1., self.bandwidth*b.dt)*(self.limited-raw),
                                -self.vmax, self.vmax)
        return float(self.limited[0]) if self.dim == 1 else self.limited.copy()

    def _reward(self):
        b = self.base
        error = (b.last_reference-b.state.omega)/b.speed_limit
        current_sq = b.state.current**2 if self.motor == "dc" else b.state.id**2+b.state.iq**2
        id_sq = 0. if self.motor == "dc" else b.state.id**2
        # No tolerance-band bonus or clipping: even small offsets retain a gradient.
        return (-self.tracking_weight*error**2-self.absolute_weight*abs(error)
                -self.current_weight*current_sq/b.current_limit**2
                -self.id_weight*id_sq/b.current_limit**2)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("Expected one finite normalized current-reference action")
        requested = float(np.clip(action[0], -1., 1.))*self.reference_limit
        change = requested-self.current_request
        self.current_request = requested
        delta = requested-self.current_reference
        if self.reference_slew > 0:
            delta = float(np.clip(delta, -self.reference_slew*self.control_period,
                                  self.reference_slew*self.control_period))
        self.current_reference += delta
        obs, reward, terminated, truncated, info = self.base.step_voltage(
            0. if self.dim == 1 else np.zeros(2))
        penalty = self.smoothness_weight*(change/self.reference_limit)**2
        # Once per OUTER decision; normalize by a physical time scale. Unlike
        # capacity-only normalization, this does not shrink at faster sampling.
        rate_penalty = self.current_rate_weight * (
            change/self.reference_limit*self.current_rate_timescale/self.control_period)**2
        penalty += rate_penalty
        if terminated:
            # A short failed episode must not beat finishing a costly episode.
            # Bound covers normalized error <=3 and current <=2*scale before
            # termination; include the largest normalized action-change cost.
            worst_step = (9*self.tracking_weight+3*self.absolute_weight
                          +4*(self.current_weight+self.id_weight+self.smoothness_weight)
                          +4*self.current_rate_weight*(self.current_rate_timescale/self.control_period)**2+1)
            reward = -worst_step*max(1, self.base.max_steps-self.base.step_count+1)
        else:
            reward -= penalty
        info.update(requested_voltage=self.requested.copy(), limited_voltage=self.limited.copy(),
                    requested_current_A=requested, limited_current_reference_A=self.current_reference,
                    command_smoothness_penalty=penalty, current_rate_penalty=rate_penalty,
                    unscaled_reward=float(reward),reward_scale=self.reward_scale)
        return self._obs(obs), float(reward*self.reward_scale), terminated, truncated, info


class CurrentBaseline:
    """Outer speed PI sharing the exact same inner loop and constraints as RL."""
    def __init__(self, motor, gains=None):
        # Small validation grid on seeds 501/502; frozen files remain preferred
        # when changing the inner loop, current cap, or actuator constraints.
        self.gains = dict(gains or dict(kp=.015 if motor == "dc" else .03, ki=.1))
        self.integral = 0.
        self.motor = motor

    def action(self, observation, env):
        b = env.base
        error = float(observation[4 if self.motor == "dc" else 6])*b.speed_limit
        if getattr(env,'error_observation_gain',1.)!=1.:
            # Preserve the classical controller's original measurement scaling.
            # The extra feature gain belongs only to the learned representation.
            from .scenarios import feedback
            reference=b.reference.value(b.time) if self.motor=='dc' else b.last_reference
            error=float(np.clip((reference-feedback(b)['speed'])/b.speed_limit,-1.,1.))*b.speed_limit
        candidate = self.integral+self.gains["ki"]*error*env.control_period
        raw = self.gains["kp"]*error+candidate
        # Conditional integration against both current saturation and slew.
        reachable = np.clip(raw, -env.reference_limit, env.reference_limit)
        if env.reference_slew > 0:
            reachable = np.clip(reachable, env.current_reference-env.reference_slew*env.control_period,
                                env.current_reference+env.reference_slew*env.control_period)
        if (raw-reachable)*error <= 0:
            self.integral = candidate
        return np.array([np.clip(raw/env.reference_limit, -1., 1.)], dtype=np.float32)


def environment_config(args):
    from .stability import environment_fields
    result=dict(motor=args.motor, architecture="current_reference_v2",
                command_slew_v_per_s=args.command_slew_v_per_s,
                command_smoothness_weight=args.command_smoothness_weight,
                **{key: getattr(args, key) for key in CONFIG_KEYS}, **environment_fields(args))
    if args.reward_scale!=1.:result['reward_scale']=args.reward_scale
    if args.error_observation_gain!=1.:result['error_observation_gain']=args.error_observation_gain
    if getattr(args,'paired_domain_rng',False):result['paired_domain_rng']=True
    return result
