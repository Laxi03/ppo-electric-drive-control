"""Continuous classical controllers and a small, reproducible validation-only search."""
import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
import numpy as np

DEFAULTS = {"dc":dict(kp=.03,ki=3.),
            "pmsm":dict(kp=.03,ki=.03,current_kp=1.2,current_ki=600.)}


class ContinuousBaseline:
    def __init__(self,motor,gains=None):
        self.motor=motor
        self.gains=dict(DEFAULTS[motor] if gains is None else gains)
        self.integral=0.
        self.current_integral=np.zeros(2)

    def action(self,observation,env):
        b=env.base
        g=self.gains
        error=float(observation[4 if self.motor=="dc" else 6])*b.speed_limit
        if self.motor=="dc":
            self.integral=np.clip(self.integral+g["ki"]*error*env.control_period,-env.vmax,env.vmax)
            raw=g["kp"]*error+self.integral
            # Back-calculation also sees the shared slew limiter.
            self.integral += min(1.,20*env.control_period)*(env.limited[0]-raw)
            return np.array([np.clip(raw/env.vmax,-1,1)],dtype=np.float32)
        candidate=self.integral+g["ki"]*error*env.control_period
        iq_unsat=g["kp"]*error+candidate
        iq_ref=np.clip(iq_unsat,-b.current_limit,b.current_limit)
        if abs(iq_unsat)<=b.current_limit or error*iq_unsat<0:
            self.integral=candidate
        from .scenarios import feedback
        measured=feedback(b)
        e=np.array([-measured["id"],iq_ref-measured["iq"]])
        self.current_integral=np.clip(self.current_integral+g["current_ki"]*e*env.control_period,-env.vmax,env.vmax)
        we=b.params.pole_pairs*measured["speed"]
        ff=np.array([-we*b.Lq_nom*measured["iq"],we*(b.Ld_nom*measured["id"]+b.flux_nom)])
        raw=g["current_kp"]*e+self.current_integral+ff
        self.current_integral += min(1.,100*env.control_period)*(env.limited-raw)
        return (raw/max(env.vmax,np.linalg.norm(raw))).astype(np.float32)


def load_gains(args,env_config):
    if not args.baseline_config:
        return None
    data=json.loads(Path(args.baseline_config).read_text())
    if data["motor"]!=args.motor:
        raise ValueError("Baseline motor mismatch")
    if data["environment"].get("architecture") != env_config.get("architecture"):
        raise ValueError("Baseline architecture differs from checkpoint")
    for key in ("stability_profile", "current_rate_weight", "current_rate_timescale_s","reward_scale","error_observation_gain"):
        if data["environment"].get(key) != env_config.get(key):
            raise ValueError(f"Baseline tuning used a different {key}")
    from .current_control import CONFIG_KEYS
    keys = ("command_slew_v_per_s","command_smoothness_weight")
    if env_config.get("architecture") == "current_reference_v2":
        keys += CONFIG_KEYS
    for key in keys:
        if data["environment"][key]!=env_config[key]:
            raise ValueError(f"Baseline tuning used a different {key}")
    return data["gains"]


def tune(args):
    if args.action_mode not in ("continuous", "current"):
        raise ValueError("tune-baseline requires continuous or current actions")
    from .runner import ROOT,load_motor,save_json
    from .evaluation import episode
    module=load_motor(args)
    config=dict(motor=args.motor,command_slew_v_per_s=args.command_slew_v_per_s,
                command_smoothness_weight=args.command_smoothness_weight)
    if args.action_mode == "current":
        from .current_control import environment_config
        config = environment_config(args)
        candidates=[dict(kp=p,ki=i) for p,i in itertools.product((.005,.015,.03),(.1,.5,1.))]
    elif args.motor=="dc":
        candidates=[dict(kp=p,ki=i) for p,i in itertools.product((.01,.03,.06),(1.,3.))]
    else:
        candidates=[dict(kp=p,ki=.03,current_kp=1.2,current_ki=i)
                    for p,i in itertools.product((.01,.03),(100.,300.,600.))]
    # Include the original continuous counterpart explicitly for auditability.
    records=[]
    for gains in candidates:
        scores=[]
        for seed in args.tuning_seeds:
            local=argparse.Namespace(**vars(args))
            local.seed=seed
            local.profile="random"
            local.evaluation_reference_scale=args.train_reference_scale
            local.baseline_config=None
            result=episode(local,module,env_config=config,baseline_gains=gains)
            m=result["metrics"]
            scale=900 if args.motor=="dc" else 600
            ripple=np.mean([r["current_magnitude_ripple_RMS"] for r in result["plateaus"]]) if result["plateaus"] else 0.
            score=m["RMSE_speed_rad_s"]/scale+.1*ripple/result["current_limit"]
            score+=.02*m["RMS_current_A"]/result["current_limit"]
            if not m["completed_episode"] or m["safety_violation"]:
                score+=1000
            scores.append(float(score))
        records.append(dict(gains=gains,score=float(np.mean(scores)),seed_scores=scores))
        print(f"{args.motor} baseline candidate {gains}: {records[-1]['score']:.6f}",flush=True)
    best=min(records,key=lambda r:r["score"])
    folder=ROOT/"outputs"/"rllib"/args.motor/"baseline_tuning"/(args.run+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True,exist_ok=False)
    save_json(folder/"baseline.json",dict(motor=args.motor,environment=config,gains=best["gains"],
        validation_seeds=args.tuning_seeds,profile="random",reference_scale=args.train_reference_scale,episode_length=args.episode_length,
        objective="RMSE/speed_scale + 0.1*late_current_ripple/current_scale + 0.02*RMS_current/current_scale + 1000*failure",
        candidates=records))
    print(f"Frozen baseline: {folder/'baseline.json'}")
    return folder/'baseline.json'
