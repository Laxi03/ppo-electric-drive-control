"""Deterministic, paired evaluation disturbances; noise never changes true state."""
import hashlib
import json
import numpy as np

PRESETS = {
    "nominal": {},
    "load_step": {"load_step_scale": .5, "event_fraction": .5},
    "parameter_shift": {"resistance_scale": 1.2, "inductance_scale": .8, "inertia_scale": 1.2},
    "resistance_jump": {"jump_resistance_scale": 1.3, "event_fraction": .5},
    "sensor_noise": {"speed_noise_std": 1., "current_noise_std": .02},
}


def specification(args):
    if args.seed < 0:
        raise ValueError("Scenario seed must be nonnegative")
    values = dict(resistance_scale=1., inductance_scale=1., inertia_scale=1.,
                  load_step_scale=0., jump_resistance_scale=1., event_fraction=.5,
                  speed_noise_std=0., current_noise_std=0.)
    name = getattr(args, "scenario", "nominal")
    values.update(PRESETS[name])
    for key in values:
        override = getattr(args, key, None)
        if override is not None:
            values[key] = override
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Scenario parameters must be finite")
    if min(values[k] for k in ("resistance_scale","inductance_scale","inertia_scale","jump_resistance_scale")) <= 0:
        raise ValueError("Parameter multipliers must be positive")
    if not 0 < values["event_fraction"] < 1 or min(values["speed_noise_std"],values["current_noise_std"]) < 0:
        raise ValueError("Event fraction must be in (0,1), noise standard deviations nonnegative")
    return dict(name=name, seed=args.seed, **values)


def feedback(base):
    if hasattr(base, "measurement_callback"):
        return base.measurement_callback()
    s=base.state
    return dict(speed=s.omega, current=getattr(s,"current",0.), id=getattr(s,"id",0.), iq=getattr(s,"iq",0.))


class Scenario:
    def __init__(self, env, args):
        self.base = b = getattr(env,"base",env)
        self.motor=args.motor
        self.spec=specification(args)
        v=self.spec
        self.event_time=v["event_fraction"]*args.episode_length
        self.applied=False
        # Separate fixed stream, indexed by physical tick, not number of reads.
        rng=np.random.default_rng(np.random.SeedSequence([args.seed,98431]))
        self.noise=rng.normal(size=(int(np.ceil(args.episode_length/b.dt))+3,3))
        self.noise *= [v["speed_noise_std"],v["current_noise_std"],v["current_noise_std"]]
        if self.motor=="dc":
            b.motor.R *= v["resistance_scale"]; b.motor.L *= v["inductance_scale"]; b.motor.J *= v["inertia_scale"]
        else:
            b.R *= v["resistance_scale"]; b.Ld *= v["inductance_scale"]; b.Lq *= v["inductance_scale"]; b.J *= v["inertia_scale"]
        self.load_delta=b.load_torque*v["load_step_scale"]
        b.before_substep_callback=self.before_step
        self.original_observation=b._observation
        self.measured_integral=0.
        if v["speed_noise_std"] or v["current_noise_std"]:
            b.measurement_callback=self.measure
            b._observation=self.observation
            b.after_substep_callback=self.after_step
        self.metadata=dict(**v,event_time_s=self.event_time,load_step_Nm=self.load_delta,
                           noise_sha256=hashlib.sha256(self.noise.tobytes()).hexdigest(),
                           noise_sampling="independent Gaussian per physical tick; held for repeated reads",
                           metrics_state="true plant state")

    def before_step(self):
        b=self.base
        if not self.applied and b.time+1e-12 >= self.event_time:
            b.load_torque += self.load_delta
            if self.motor=="dc": b.motor.R *= self.spec["jump_resistance_scale"]
            else: b.R *= self.spec["jump_resistance_scale"]
            self.applied=True

    def measure(self):
        b=self.base;s=b.state
        n=self.noise[min(round(b.time/b.dt),len(self.noise)-1)]
        return dict(speed=s.omega+n[0], current=getattr(s,"current",0.)+n[1],
                    id=getattr(s,"id",0.)+n[1], iq=getattr(s,"iq",0.)+n[2])

    def after_step(self):
        b=self.base
        self.measured_integral=float(np.clip(self.measured_integral+
            (b.last_reference-self.measure()["speed"])/b.speed_limit*b.dt,-.5,.5))

    def observation(self):
        b=self.base;o=self.original_observation().copy();s=self.measure()
        o[0]=np.clip(s["speed"]/b.speed_limit,-1,1)
        if self.motor=="dc":
            o[1]=np.clip(s["current"]/b.current_limit,-1,1);error_index,integral_index=4,5
        else:
            o[1:3]=np.clip([s["id"]/b.current_limit,s["iq"]/b.current_limit],-1,1);error_index,integral_index=6,7
        o[error_index]=np.clip((b.last_reference-s["speed"])/b.speed_limit,-1,1)
        o[integral_index]=self.measured_integral
        return o

    def initial_observation(self, env):
        o=self.base._observation()
        return env._obs(o) if hasattr(env,"base") else o
