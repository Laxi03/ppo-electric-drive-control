"""Identical reporting for both motors; physically different quantities stay labelled."""
import csv
import json
import numpy as np


class QuantizedFOC:
    """Nominal-model cascade PI with dq feedforward, projected onto PPO's action set.

    A reproducible starting baseline, not an optimized industrial FOC design.
    Gains are intentionally explicit, and no perturbed plant parameters are read.
    """
    def __init__(self):
        self.speed_integral = 0.0
        self.current_integral = np.zeros(2)

    def action(self, observation, env):
        dt = env.control_period
        error = float(observation[6]) * env.speed_limit
        proposed = self.speed_integral + 0.03 * error * dt
        iq_unsat = 0.03 * error + proposed
        iq_ref = float(np.clip(iq_unsat, -env.current_limit, env.current_limit))
        if abs(iq_unsat) <= env.current_limit or error * iq_unsat < 0:
            self.speed_integral = proposed
        from .scenarios import feedback
        sample = feedback(env)
        measured = np.asarray([sample["id"], sample["iq"]])
        current_error = np.asarray([0.0, iq_ref]) - measured
        candidate = np.clip(self.current_integral + 600 * current_error * dt,
                            -env.Vmax_nominal, env.Vmax_nominal)
        we = env.params.pole_pairs * sample["speed"]
        feedforward = np.asarray([-we * env.Lq_nom * sample["iq"],
                                  we * (env.Ld_nom * sample["id"] + env.flux_nom)])
        voltage = 1.2 * current_error + candidate + feedforward
        magnitude = np.linalg.norm(voltage)
        if magnitude <= env.Vmax_nominal:
            self.current_integral = candidate
        voltage *= min(1.0, env.Vmax_nominal / max(magnitude, 1e-12))
        return int(np.argmin(np.sum((env.action_vectors * env.Vmax_nominal - voltage)**2, axis=1)))


def episode(args, motor_module, policy=None, env_config=None, baseline_gains=None):
    import sim_DC_RLlib_Curiosity as dc
    from .continuous import ContinuousVoltageEnv
    from .current_control import CurrentReferenceEnv, CurrentBaseline
    from .ripple import command_metrics, plateau_metrics
    continuous = env_config is not None
    cls = motor_module.DCMotorCuriosityEnv if args.motor == "dc" else motor_module.PMSMSpeedCuriosityEnv
    config = dict(env_config or {})
    config.update(dict(seed=args.seed, episode_length=args.episode_length, profile=args.profile,
                   reference_scale=args.evaluation_reference_scale, training_mode=False,
                   domain_randomization=False, parameter_jump=False,
                   num_axis_levels=args.num_axis_levels, num_voltage_levels=21))
    from .matched import MatchedCurrentEnv, MatchedVoltageEnv
    architecture=config.get("architecture")
    current_control = architecture in ("current_reference_v2","matched-current_v3")
    env_cls={"current_reference_v2":CurrentReferenceEnv,"matched-current_v3":MatchedCurrentEnv,
             "matched-voltage_v3":MatchedVoltageEnv}.get(architecture,ContinuousVoltageEnv)
    env = env_cls(config) if continuous else cls(config)
    base = env.base if continuous else env
    if continuous:
        from .baselines import ContinuousBaseline, load_gains
        baseline_cls = CurrentBaseline if current_control else ContinuousBaseline
        baseline = baseline_cls(args.motor, baseline_gains or load_gains(args,config))
    else:
        if args.baseline_config:
            raise ValueError("Continuous tuned baselines cannot be applied to legacy discrete checkpoints")
        baseline = dc.PIController() if args.motor == "dc" else QuantizedFOC()
    rows, observations, actions = [], [], []
    requested, limited = [], []
    current_references = []
    observation, _ = env.reset(seed=args.seed)
    from .scenarios import Scenario
    scenario = Scenario(env, args)
    observation = scenario.initial_observation(env)
    try:
        while True:
            observations.append(observation.copy())
            action = (continuous_action(policy,observation) if continuous else dc._deterministic_policy_action(policy, observation)) if policy is not None else baseline.action(observation, env)
            if not continuous:
                command = np.array([env.voltage_levels[int(action)],0.]) if args.motor=="dc" else env.action_vectors[int(action)]*env.Vmax_nominal
            observation, reward, terminated, truncated, info = env.step(action)
            if continuous:
                command=np.pad(info["requested_voltage"],(0,2-env.dim))
                limited.append(np.pad(info["limited_voltage"],(0,2-env.dim)))
            else:
                limited.append(command.copy())
            requested.append(command.copy())
            if current_control:
                current_references.append([info["requested_current_A"], info["limited_current_reference_A"]])
            if args.motor == "dc":
                current, voltage = info["current"], info["voltage"]
                components = [current, 0.0, voltage, 0.0]
                limit, vmax = base.current_limit, base.Vdc
            else:
                current, voltage = info["current_magnitude"], info["voltage_magnitude"]
                components = [info["id"], info["iq"], info["vd"], info["vq"]]
                limit, vmax = base.current_limit, base.Vmax_nominal
            rows.append([info["time"], info["reference"], info["omega"], current, voltage, reward, *components])
            actions.append(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    data = np.asarray(rows)
    time, reference, speed, current, voltage, reward = data[:, :6].T
    dt = np.diff(np.r_[0.0, time])
    error = reference - speed
    duration = float(time[-1])
    metrics = {
        "episode_return_extrinsic": float(reward.sum()),
        "MAE_speed_rad_s": float(np.sum(abs(error)*dt)/duration),
        "RMSE_speed_rad_s": float(np.sqrt(np.sum(error**2*dt)/duration)),
        "max_abs_speed_error_rad_s": float(max(abs(error))),
        "IAE_speed": float(np.sum(abs(error)*dt)), "ISE_speed": float(np.sum(error**2*dt)),
        "RMS_current_A": float(np.sqrt(np.sum(current**2*dt)/duration)),
        "mean_abs_current_A": float(np.sum(abs(current)*dt)/duration),
        "max_abs_current_A": float(max(abs(current))),
        "p95_current_utilization_percent": float(np.percentile(abs(current)/limit*100,95)),
        "current_I2t_A2s": float(np.sum(current**2*dt)),
        "time_above_current_scale_s": float(np.sum(dt[abs(current)>limit])),
        "RMS_voltage_V": float(np.sqrt(np.sum(voltage**2*dt)/duration)),
        "mean_abs_voltage_V": float(np.sum(abs(voltage)*dt)/duration),
        "absolute_electrical_energy_J": float(np.sum(np.abs(data[:,6]*data[:,8]+data[:,7]*data[:,9])*dt)*(1.0 if args.motor=="dc" else 1.5)),
        "total_voltage_vector_variation_V": float(np.linalg.norm(np.diff(data[:,8:10],axis=0),axis=1).sum()),
        "time_at_voltage_rail_s": float(np.sum(dt[abs(voltage)>=0.99*vmax])),
        "duration_s": duration, "evaluation_steps": len(data),
        "safety_violation": bool(info["safety_violation"]),
        "completed_episode": bool(not terminated and truncated),
    }
    segments = dc.compute_segment_metrics(time, speed, reference, voltage, vmax)
    requested,limited=np.asarray(requested),np.asarray(limited)
    metrics.update(command_metrics(time,requested,limited,data[:,8:10]))
    if continuous:
        action_array=np.asarray(actions).reshape(len(data),-1)
        metrics["policy_action_out_of_bounds_fraction"] = float(np.mean(
            np.any(np.abs(action_array)>1.,axis=1)))
    if current_control:
        references=np.asarray(current_references)
        changes=np.diff(references[:,0])
        metrics["requested_current_rms_slew_A_per_s"] = float(np.sqrt(np.mean(
            (changes/np.diff(time))**2))) if len(time)>1 else 0.
    plateaus=plateau_metrics(data,args.steady_fraction)
    if plateaus:
        for key in ("speed_ripple_RMS","speed_peak_to_peak","current_magnitude_ripple_RMS","current_magnitude_peak_to_peak"):
            metrics["mean_plateau_"+key]=float(np.mean([row[key] for row in plateaus]))
    return dict(data=data, observations=np.asarray(observations), actions=actions,
                metrics=metrics, segments=segments, current_limit=limit,plateaus=plateaus,
                requested=requested,limited=limited,continuous=continuous,scenario=scenario.metadata,
                current_references=np.asarray(current_references) if current_control else None)


def continuous_action(module,observation):
    import torch
    from ray.rllib.core import Columns
    tensor=torch.as_tensor(observation,dtype=torch.float32,device=next(module.parameters()).device)[None]
    with torch.no_grad():
        logits=module.forward_inference({Columns.OBS:tensor})[Columns.ACTION_DIST_INPUTS]
        distribution=module.get_inference_action_dist_cls().from_logits(logits)
        return distribution.to_deterministic().sample()[0].cpu().numpy()


def save_reports(args, results, output):
    import matplotlib.pyplot as plt
    import sim_DC_RLlib_Curiosity as dc
    from .runner import save_json
    save_json(output / "metrics.json", {label: result["metrics"] for label,result in results.items()})
    save_json(output / "scenarios.json", {label: result.get("scenario") for label,result in results.items()})
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", *results])
        metric_names=dict.fromkeys(key for r in results.values() for key in r["metrics"])
        for key in metric_names:
            writer.writerow([key, *[r["metrics"].get(key) for r in results.values()]])
    fig, axes = plt.subplots(4, 1, figsize=(11,10), sharex=True)
    histfig, histaxis = plt.subplots(figsize=(9,5))
    statefig, stateaxes = plt.subplots(1,len(results),figsize=(5*len(results),4),squeeze=False)
    for index,(label,result) in enumerate(results.items()):
        data = result["data"]
        time, ref, speed, current, voltage = data[:,:5].T
        with (output / f"{label}_timeseries.csv").open("w",newline="") as stream:
            writer=csv.writer(stream)
            action_data=np.asarray(result["actions"]).reshape(len(data),-1)
            action_names=[f"action_{i}" for i in range(action_data.shape[1])] if result["continuous"] else ["action_index"]
            writer.writerow(["time_s","reference_rad_s","omega_rad_s","current_A_or_dq_magnitude","voltage_V_or_dq_magnitude","reward","i_or_id_A","iq_A","u_or_vd_V","vq_V",*action_names,"requested_u_or_vd_V","requested_vq_V","limited_u_or_vd_V","limited_vq_V"])
            writer.writerows(np.column_stack([data,action_data,result["requested"],result["limited"]]))
        if result.get("current_references") is not None:
            with (output / f"{label}_current_references.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time_s", "requested_current_A", "limited_current_reference_A"])
                writer.writerows(np.column_stack([time, result["current_references"]]))
        if result["plateaus"]:
            with (output/f"{label}_ripple_by_plateau.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(result["plateaus"][0]))
                writer.writeheader()
                writer.writerows(result["plateaus"])
        if result["segments"]:
            with (output / f"{label}_segments.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(result["segments"][0]))
                writer.writeheader(); writer.writerows(result["segments"])
        if index == 0:
            axes[0].plot(time,ref,"k--",label="reference")
        for axis,values in zip(axes,[speed,ref-speed,current,voltage]):
            axis.plot(time,values,label=label,linewidth=1)
        axis=stateaxes[0,index]
        scatter=axis.scatter(speed,current,c=time,s=5,cmap="viridis")
        axis.set(xlabel="Speed [rad/s]",ylabel="Current [A]" if args.motor=="dc" else "dq current magnitude [A]",title=label)
        statefig.colorbar(scatter,ax=axis,label="Time [s]")
    axes[0].legend()
    for axis,ylabel in zip(axes,["Speed [rad/s]","Error [rad/s]","Current [A]" if args.motor=="dc" else "|i_dq| [A]","Voltage [V]" if args.motor=="dc" else "|v_dq| [V]"]):
        axis.set_ylabel(ylabel)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{args.motor.upper()} · {args.profile} · deterministic evaluation")
    dc.plot_current_utilization_histogram(histaxis,{label:r["data"][:,3] for label,r in results.items()},next(iter(results.values()))["current_limit"])
    for figure,name in [(fig,"tracking"),(histfig,"current_histogram"),(statefig,"physical_states")]:
        figure.tight_layout(); figure.savefig(output/f"{name}.png",dpi=150)
    if any(r["plateaus"] for r in results.values()):
        rf,ra=plt.subplots(3,1,figsize=(10,8),sharex=True)
        for label,r in results.items():
            rows=r["plateaus"]
            for axis,key in zip(ra,("steady_offset_rad_s","speed_ripple_RMS","current_magnitude_ripple_RMS")):
                axis.plot([row["segment"] for row in rows],[row[key] for row in rows],"o-",label=label)
        for axis,name in zip(ra,("Late-window bias [rad/s]","Speed ripple RMS [rad/s]","Current ripple RMS [A]")):
            axis.set_ylabel(name);axis.legend()
        ra[-1].set_xlabel("Reference plateau")
        rf.suptitle(f"Last {100*args.steady_fraction:g}% of each plateau; settling is not assumed")
        rf.tight_layout();rf.savefig(output/"ripple_summary.png",dpi=150)
    if getattr(args, "plot_reward", False):
        reward_fig, reward_axis = plt.subplots(figsize=(10, 4))
        for label, result in results.items():
            reward_axis.plot(result["data"][:, 0], result["data"][:, 5], label=label)
        reward_axis.set(xlabel="Time [s]", ylabel="Extrinsic reward", title="Evaluation reward")
        reward_axis.legend()
        reward_fig.tight_layout()
        reward_fig.savefig(output / "reward.png", dpi=150)
    if getattr(args, "plot_tsne", False):
        save_tsne(args, results, output)
    if args.show:
        plt.show()
    plt.close("all")


def save_tsne(args, results, output):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    # One joint fit, so controller panels share coordinates. Never compare separate fits.
    samples=[]; colors=[]; labels=[]
    for label,r in results.items():
        indices=np.linspace(0,len(r["data"])-1,min(len(r["data"]),max(1,args.tsne_samples//len(results))),dtype=int)
        samples.append(r["observations"][indices])
        colors.extend(r["data"][indices,4] if args.motor=="dc" else r["data"][indices,9])
        labels.extend([label]*len(indices))
    if sum(map(len,samples)) >= 5:
        x=StandardScaler().fit_transform(np.concatenate(samples))
        embedding=TSNE(n_components=2,perplexity=min(30.,(len(x)-1)/3),random_state=args.seed,init="random",learning_rate="auto").fit_transform(x)
        tf,ta=plt.subplots(1,len(results),figsize=(5*len(results),4),squeeze=False,sharex=True,sharey=True)
        labels=np.asarray(labels); colors=np.asarray(colors); bound=max(float(max(abs(colors))),1e-6)
        for index,label in enumerate(results):
            mask=labels==label
            sc=ta[0,index].scatter(*embedding[mask].T,c=colors[mask],cmap="coolwarm",vmin=-bound,vmax=bound,s=8)
            ta[0,index].set(title=label,xlabel="t-SNE 1",ylabel="t-SNE 2")
        tf.colorbar(sc,ax=list(ta.flat),label="Applied u [V]" if args.motor=="dc" else "Applied vq [V]")
        tf.savefig(output/"tsne.png",dpi=150,bbox_inches="tight")
        with (output/"tsne.csv").open("w",newline="") as stream:
            writer=csv.writer(stream); writer.writerow(["controller","component1","component2","color_voltage_V"])
            writer.writerows(zip(labels,*embedding.T,colors))


def evaluate_checkpoints(args, module, selected, output):
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    results={}
    common_config=None
    common_env=None
    try:
        ray.init(include_dashboard=False,log_to_driver=False,logging_level="ERROR")
        for label,path in selected.items():
            algorithm=Algorithm.from_checkpoint(str(path))
            try:
                from .continuous import ENV_NAMES
                from .current_control import ENV_NAMES as CURRENT_NAMES
                from .matched import ENV_NAMES as MATCHED_NAMES
                continuous_names=[ENV_NAMES[args.motor],CURRENT_NAMES[args.motor],
                                  *[names[args.motor] for names in MATCHED_NAMES.values()]]
                expected_name=algorithm.config.env
                if expected_name not in [module.ENV_NAME,*continuous_names]:
                    raise ValueError(f"Checkpoint motor mismatch: {expected_name}")
                coefficient=float((algorithm.config.learner_config_dict or {}).get("intrinsic_reward_coeff",0))
                if (label=="icm") != (coefficient>0):
                    raise ValueError(f"Checkpoint is not the requested {label}: intrinsic coefficient={coefficient}")
                config=algorithm.config.env_config
                continuous=expected_name in continuous_names
                effective=dict(config) if continuous else None
                # Compare using identical saved actuation settings, not CLI defaults.
                if common_env is not None and (expected_name!=common_env or effective!=common_config):
                    raise ValueError("Comparison checkpoints use different environments/settings")
                common_env,common_config=expected_name,effective
                if not continuous and args.motor=="pmsm" and config.get("num_axis_levels",7)!=args.num_axis_levels:
                    raise ValueError("--num-axis-levels differs from checkpoint")
                results[label]=episode(args,module,algorithm.get_module(module.DEFAULT_MODULE_ID),env_config=effective)
            finally:
                algorithm.stop()
        if args.command=="compare":
            name="pi" if args.motor=="dc" else ("foc" if common_config is not None else "quantized_foc")
            results[name]=episode(args,module,env_config=common_config)
    finally:
        ray.shutdown()
    save_reports(args,results,output)
    from .runner import save_json
    save_json(output/"effective_environment.json",dict(environment=common_env,config=common_config,
        steady_fraction=args.steady_fraction,command_change_tolerance_V=.001,baseline_config=args.baseline_config))
    if args.baseline_config:
        from pathlib import Path
        save_json(output/"frozen_baseline.json",json.loads(Path(args.baseline_config).read_text()))
    print(json.dumps({label:r["metrics"] for label,r in results.items()},indent=2))
    print(f"Saved evaluation: {output}")
