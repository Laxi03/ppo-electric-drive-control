"""One CLI for isolated, reproducible DC/PMSM PPO experiments.

Motor physics and checkpoint class paths intentionally stay in the legacy modules.
No existing results are moved or overwritten by this entry point.
"""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import time
import math
from datetime import datetime, timezone

from .workspace import workspace_root

ROOT = workspace_root()
SCRIPTS = {"dc": "sim_DC_RLlib_Curiosity", "pmsm": "sim_PMSM_RLlib_Curiosity"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "eval", "compare", "list", "tune-baseline"])
    p.add_argument("--motor", choices=SCRIPTS, default="dc")
    p.add_argument("--controller", choices=["ppo", "icm"], default="ppo")
    p.add_argument("--run", default="current_v2_seed123")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--episode-length", type=float, default=8.0)
    p.add_argument("--policy-hidden-sizes", nargs="+", type=int, default=[16, 8])
    p.add_argument("--train-reference-scale", type=float, default=1.0)
    p.add_argument('--train-profile',choices=['random','random_full'],default='random')
    p.add_argument("--evaluation-reference-scale", type=float, default=1.0)
    p.add_argument("--train-batch-size", type=int, default=4096)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--num-epochs", type=int, default=6)
    p.add_argument("--checkpoint-frequency", type=int, default=50)
    p.add_argument("--num-env-runners", type=int, default=2, help="Parallel sampling processes; 0 uses local sampling")
    p.add_argument("--num-axis-levels", type=int, default=7)
    p.add_argument("--intrinsic-reward-coeff", type=float, default=0.02)
    p.add_argument("--forward-loss-weight", type=float, default=0.2)
    p.add_argument("--no-domain-randomization", action="store_true")
    p.add_argument("--paired-domain-rng",action="store_true",
                   help="PMSM ablation: preserve reference/jump RNG draws when reset parameter randomization is disabled")
    p.add_argument("--no-parameter-jump", action="store_true")
    p.add_argument("--resume", action="store_true", help="Continue latest checkpoint to --iterations")
    p.add_argument("--checkpoint", help="Explicit checkpoint, including legacy checkpoints")
    p.add_argument("--ppo-checkpoint")
    p.add_argument("--curiosity-checkpoint")
    p.add_argument("--profile", choices=["steps", "random", "sine", "mixed"], default="mixed")
    p.add_argument("--show", action="store_true")
    p.add_argument("--plot-reward", action="store_true", help="Generate reward plots (training history or evaluation reward)")
    p.add_argument("--plot-tsne", action="store_true", help="Compute and plot t-SNE during evaluation; off by default")
    p.add_argument("--tsne-samples", type=int, default=1200)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--action-mode", choices=["discrete", "continuous", "current","matched-current","matched-voltage"], default="current")
    p.add_argument("--stability-profile", choices=["legacy", "pmsm-v1"], default="legacy",
                   help="Opt-in bounded-mean PMSM current-control training; use a NEW run")
    p.add_argument("--current-rate-weight",type=float,default=None,
                   help="Override PMSM stability-profile current-command rate penalty; default 0.02")
    p.add_argument('--actor-observations',choices=['full','speed','speed_memory'],default='full',
                   help='PMSM actor feature set; critic always retains the full observation')
    p.add_argument('--discount-horizon-s',type=float,default=None,
                   help='Opt-in PMSM PPO discount e-folding horizon in physical seconds; pair with --advantage-horizon-s')
    p.add_argument('--advantage-horizon-s',type=float,default=None,
                   help='Opt-in PMSM GAE e-folding horizon including gamma*lambda, in seconds')
    p.add_argument('--critic-profile',choices=['legacy','normalized'],default='legacy')
    p.add_argument('--critic-hidden-sizes',nargs='+',type=int,default=[64,64])
    p.add_argument('--critic-return-scale',type=float,default=500.,
                   help='Normalized critic units only; never scales physical reward or ICM bonus')
    p.add_argument('--critic-learning-rate',type=float,default=1e-3)
    p.add_argument('--policy-log-std-min',type=float,default=-4.)
    p.add_argument('--policy-log-std-max',type=float,default=-1.)
    p.add_argument("--current-reference-limit", type=float, default=.8, help="Fraction of current scale available as current reference")
    p.add_argument("--current-reference-slew", type=float, default=500., help="Current reference slew A/s; 0 disables")
    p.add_argument("--current-bandwidth", type=float, default=300., help="Nominal current-loop bandwidth in rad/s")
    p.add_argument("--tracking-weight", type=float, default=100.)
    p.add_argument("--tracking-absolute-weight", type=float, default=1.)
    p.add_argument("--current-weight", type=float, default=.02)
    p.add_argument("--id-weight", type=float, default=.1)
    p.add_argument("--reward-scale",type=float,default=1.,help="Uniform current-mode training reward scale; physical metrics stay in SI units")
    p.add_argument("--vf-clip-param",type=float,default=10.,help="RLlib critic squared-loss cap; must match the reward/return scale")
    p.add_argument("--error-observation-gain",type=float,default=1.,help="Extra gain on normalized tracking-error input; does not change reference range or reward")
    p.add_argument("--command-slew-v-per-s", type=float, default=1000., help="Continuous command-vector slew limit; 0 disables")
    p.add_argument("--command-smoothness-weight", type=float, default=.05)
    p.add_argument("--intrinsic-final-coeff", type=float, default=None)
    p.add_argument("--intrinsic-decay-iterations", type=int, default=1000)
    p.add_argument("--steady-fraction", type=float, default=.3, help="Last fraction of each plateau used for offset/ripple")
    p.add_argument("--baseline-config", help="Frozen JSON from tune-baseline")
    p.add_argument("--tuning-seeds", nargs="+", type=int, default=[501, 502])
    from .scenarios import PRESETS
    p.add_argument("--scenario", choices=list(PRESETS), default="nominal")
    for name in ("resistance-scale","inductance-scale","inertia-scale","load-step-scale",
                 "jump-resistance-scale","event-fraction","speed-noise-std","current-noise-std"):
        p.add_argument("--"+name, type=float, default=None, help="Override evaluation scenario (SI units for noise)")
    return p


def run_dir(args, controller=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run):
        raise ValueError("--run must be a simple directory name (letters, digits, _, -, .)")
    return ROOT / "RL_models" / args.motor / (controller or args.controller) / args.run


def checkpoints(folder):
    return sorted(p for p in folder.glob("iteration_*")
                  if p.is_dir() and re.fullmatch(r"iteration_\d+", p.name))


def latest(folder):
    found = checkpoints(folder)
    if not found:
        raise FileNotFoundError(f"No checkpoint in {folder}. Train this run first or pass an explicit checkpoint.")
    return max(found, key=lambda p: int(p.name.split("_")[-1]))


def save_json(path, data):
    # Atomic replacement protects manifests from interruption during a write.
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def settings(args):
    names = ("motor", "controller", "seed", "episode_length", "policy_hidden_sizes",
             "train_reference_scale", "train_batch_size", "minibatch_size", "num_epochs",
             "num_axis_levels", "no_domain_randomization", "no_parameter_jump")
    result = {name: getattr(args, name) for name in names}
    result.update(intrinsic_reward_coeff=args.intrinsic_reward_coeff if args.controller == "icm" else 0.0,
                  forward_loss_weight=args.forward_loss_weight if args.controller == "icm" else 0.0,
                  icm_hidden_sizes=[64, 64], train_profile=args.train_profile)
    if args.action_mode != "discrete":
        result.update(action_mode=args.action_mode, environment_version=2 if args.action_mode == "current" else 1,
                      command_slew_v_per_s=args.command_slew_v_per_s,
                      command_smoothness_weight=args.command_smoothness_weight,
                      intrinsic_final_coeff=args.intrinsic_final_coeff,
                      intrinsic_decay_iterations=args.intrinsic_decay_iterations)
    if args.action_mode in ("current","matched-current","matched-voltage"):
        from .current_control import CONFIG_KEYS
        result.update({key: getattr(args, key) for key in CONFIG_KEYS})
        if args.action_mode.startswith("matched-"):
            result.update(environment_version=3,reward_protocol="physical_cost_v3")
    result["num_env_runners"] = args.num_env_runners
    if args.reward_scale!=1.:result['reward_scale']=args.reward_scale
    if args.vf_clip_param!=10.:result['vf_clip_param']=args.vf_clip_param
    if args.error_observation_gain!=1.:result['error_observation_gain']=args.error_observation_gain
    if args.paired_domain_rng:result['paired_domain_rng']=True
    if args.actor_observations!='full':result['actor_observations']=args.actor_observations
    if args.policy_log_std_min!=-4.:result['policy_log_std_min']=args.policy_log_std_min
    if args.policy_log_std_max!=-1.:result['policy_log_std_max']=args.policy_log_std_max
    if args.critic_profile!='legacy':
        result.update(critic_profile=args.critic_profile,critic_hidden_sizes=args.critic_hidden_sizes,
                      critic_return_scale=args.critic_return_scale,critic_learning_rate=args.critic_learning_rate,
                      critic_loss='normalized_twice_huber',optimizer_protocol='separate_actor_critic_v1')
    from .stability import training_fields
    result.update(training_fields(args))
    return result


def load_motor(args):
    from .continuous import register_continuous
    register_continuous()
    module = importlib.import_module(SCRIPTS[args.motor])
    if args.motor == "dc":
        module.register_dc_environment()
    else:
        module.register_pmsm_environment()
    return module


def config_for(args, module):
    if args.action_mode != "discrete":
        from .continuous import continuous_config
        return continuous_config(args)
    values = settings(args)
    if args.motor == "dc":
        config = module.make_rllib_config(
            seed=args.seed, episode_length=args.episode_length, profile="random",
            train_batch_size=args.train_batch_size, minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs, num_env_runners=args.num_env_runners,
            intrinsic_reward_coeff=values["intrinsic_reward_coeff"],
            forward_loss_weight=values["forward_loss_weight"],
            domain_randomization=not args.no_domain_randomization,
            parameter_jump=not args.no_parameter_jump,
            train_reference_scale=args.train_reference_scale,
            policy_hidden_sizes=tuple(args.policy_hidden_sizes))
    else:
        copied = argparse.Namespace(**vars(args))
        copied.intrinsic_reward_coeff = values["intrinsic_reward_coeff"]
        copied.forward_loss_weight = values["forward_loss_weight"]
        config = module.make_config(copied)
    # Identical ICM capacity in the new workflow, without changing legacy defaults.
    module_specs = config.rl_module_spec
    spec = module_specs.rl_module_specs[module.ICM_MODULE_ID]
    for key in ("feature_net_hiddens", "inverse_net_hiddens", "forward_net_hiddens"):
        spec.model_config[key] = (64, 64)
    return config.rl_module(rl_module_spec=module_specs).env_runners(num_env_runners=args.num_env_runners)


def train(args):
    folder = run_dir(args)
    values = settings(args)
    if args.dry_run:
        print(json.dumps({"model_directory": str(folder), "target_iterations": args.iterations,
                          "settings": values}, indent=2))
        return
    checkpoint = None
    if args.resume:
        stored = json.loads((folder / "manifest.json").read_text())
        stored["settings"].setdefault("num_env_runners", 0)
        if stored["settings"] != values:
            raise ValueError("Resume configuration differs from manifest. Use the original settings.")
        checkpoint = latest(folder)
    elif folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Run already exists: {folder}. Use --resume or another --run.")
    start = int(checkpoint.name.split("_")[-1]) if checkpoint else 0
    if start >= args.iterations:
        print(f"Already complete: {start} >= {args.iterations}")
        return
    module = load_motor(args)
    import ray
    import torch
    from ray.rllib.algorithms.algorithm import Algorithm
    history_path = folder / "training_history.csv"
    history = []
    if checkpoint and history_path.exists():
        with history_path.open() as stream:
            history = [row for row in csv.DictReader(stream) if int(row["iteration"]) <= start]
    source = Path(module.__file__).resolve()
    manifest = {"settings": values, "target_iterations": args.iterations,
                "ray": ray.__version__, "torch": torch.__version__,
                "python": sys.version, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "workflow_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                    for name in ("runner.py", "evaluation.py", "continuous.py", "continuous_icm.py", "current_control.py", "matched.py", "scenarios.py", "ripple.py", "baselines.py", "stability.py", "normalized_critic.py")},
                "execution": dict(num_env_runners=args.num_env_runners, checkpoint_frequency=args.checkpoint_frequency),
                "updated_at": datetime.now(timezone.utc).isoformat(), "status": "starting"}
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "manifest.json", manifest)
    algo = None
    try:
        ray.init(include_dashboard=False, log_to_driver=False, num_cpus=max(1,args.num_env_runners+1), logging_level="ERROR")
        algo = Algorithm.from_checkpoint(str(checkpoint)) if checkpoint else config_for(args, module).build_algo()
        for iteration in range(start + 1, args.iterations + 1):
            coefficient = None
            if args.action_mode != "discrete":
                from .continuous_icm import coefficient_at, set_coefficient
                coefficient = coefficient_at(args, iteration)
                coefficient *= args.reward_scale
                if args.controller == "icm":
                    algo.learner_group.foreach_learner(lambda learner: set_coefficient(learner, coefficient))
            tick = time.perf_counter()
            result = algo.train()
            elapsed = time.perf_counter()-tick
            finder = module._first_numeric_by_key if args.motor == "dc" else module._find_numeric
            episode_return = module._episode_return(result)
            if episode_return is not None and not math.isfinite(episode_return):
                # Fragmented parallel sampling may finish no episode this iteration.
                episode_return = None
            history.append({"iteration": iteration, "extrinsic_return": episode_return,
                            "intrinsic_reward": finder(result, "mean_intrinsic_rewards"),
                            "steps": result.get("num_env_steps_sampled_lifetime", 0),
                            "iteration_wall_s": elapsed})
            # Retain numerical-learning diagnostics, not just episode returns.
            # Read the policy learner only (never the ICM auxiliary learner).
            policy_result=result.get('learners',{}).get('default_policy',{})
            for key in ('vf_explained_var','vf_loss','vf_loss_unclipped','policy_loss','entropy','mean_kl_loss',
                        'vf_normalized_loss','value_target_std','value_prediction_std',
                        'value_target_mean','value_prediction_mean',
                        'gradients_actor_global_norm','gradients_critic_global_norm'):
                value=finder(policy_result,key)
                history[-1][key]=value if value is not None and math.isfinite(value) else None
            if coefficient is not None:
                history[-1]["intrinsic_coefficient"] = coefficient
            print(f"{args.motor}/{args.controller} iteration {iteration}/{args.iterations}: {history[-1]}", flush=True)
            if iteration % args.checkpoint_frequency == 0 or iteration == args.iterations:
                destination = folder / f"iteration_{iteration:04d}"
                if destination.exists():
                    raise FileExistsError(destination)
                pending = folder / f".iteration_{iteration:04d}_pending"
                if pending.exists():
                    raise FileExistsError(f"Incomplete checkpoint needs inspection: {pending}")
                algo.save_to_path(str(pending))
                pending.rename(destination)
                temp = history_path.with_suffix(".tmp")
                with temp.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(history[-1]))
                    writer.writeheader()
                    writer.writerows(history)
                temp.replace(history_path)
                if args.plot_reward:
                    import matplotlib.pyplot as plt
                    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
                    for axis, key in zip(axes, ["extrinsic_return", "intrinsic_reward"]):
                        axis.plot([int(row["iteration"]) for row in history],
                                  [float(row[key]) if row[key] not in (None, "", "None") else float("nan") for row in history])
                        axis.set_ylabel(key)
                    axes[-1].set_xlabel("Iteration")
                    figure.suptitle(f"{args.motor.upper()} / {args.controller.upper()}")
                    figure.tight_layout()
                    figure.savefig(folder / "training_history.png", dpi=150)
                    plt.close(figure)
                manifest.update(status="running", completed_iteration=iteration)
                save_json(folder / "manifest.json", manifest)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="interrupted_or_failed", error=str(error))
        raise
    finally:
        save_json(folder / "manifest.json", manifest)
        if algo is not None:
            algo.stop()
        ray.shutdown()


def evaluate(args):
    if args.command == "compare":
        selected = {"ppo": Path(args.ppo_checkpoint).resolve() if args.ppo_checkpoint else latest(run_dir(args, "ppo")),
                    "icm": Path(args.curiosity_checkpoint).resolve() if args.curiosity_checkpoint else latest(run_dir(args, "icm"))}
    else:
        selected = {args.controller: Path(args.checkpoint).resolve() if args.checkpoint else latest(run_dir(args))}
    for path in selected.values():
        if not path.is_dir():
            raise FileNotFoundError(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "outputs" / "rllib" / args.motor / args.command / args.run / f"{args.profile}_seed{args.seed}_{stamp}"
    if args.dry_run:
        print(json.dumps({"checkpoints": selected, "output": output}, indent=2, default=str))
        return
    module = load_motor(args)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "evaluation_manifest.json", {"arguments": vars(args), "checkpoints": selected})
    from .evaluation import evaluate_checkpoints
    evaluate_checkpoints(args, module, selected, output)
    return output


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.stability_profile != "legacy" and (args.motor != "pmsm" or args.action_mode != "current"):
        p.error("pmsm-v1 stability profile requires --motor pmsm --action-mode current")
    if args.paired_domain_rng and (args.motor!='pmsm' or args.action_mode!='current'):
        p.error('--paired-domain-rng requires PMSM current-reference control')
    if args.train_profile!='random' and (args.motor!='pmsm' or args.action_mode!='current'):
        p.error('random_full training profile currently requires PMSM current-reference control')
    if args.current_rate_weight is not None:
        if args.stability_profile!='pmsm-v1' or not math.isfinite(args.current_rate_weight) or args.current_rate_weight<0:
            p.error('--current-rate-weight requires pmsm-v1 and a finite nonnegative value')
    if not (math.isfinite(args.policy_log_std_min) and math.isfinite(args.policy_log_std_max)
            and -10<=args.policy_log_std_min<args.policy_log_std_max<=0):
        p.error('Policy log-std bounds must satisfy -10 <= min < max <= 0')
    if (args.actor_observations!='full' or args.policy_log_std_min!=-4. or args.policy_log_std_max!=-1.) and args.stability_profile!='pmsm-v1':
        p.error('Actor feature/noise overrides require the PMSM stability profile')
    if args.discount_horizon_s is not None or args.advantage_horizon_s is not None:
        if args.stability_profile!='pmsm-v1':p.error('Physical training horizons require pmsm-v1')
        from .stability import temporal_parameters
        try:temporal_parameters(args)
        except ValueError as error:p.error(str(error))
    if args.critic_profile!='legacy' and args.stability_profile!='pmsm-v1':
        p.error('Normalized critic requires PMSM pmsm-v1 current control')
    if (not args.critic_hidden_sizes or min(args.critic_hidden_sizes)<1 or
            not all(math.isfinite(v) and v>0 for v in (args.critic_return_scale,args.critic_learning_rate))):
        p.error('Critic sizes, scale and learning rate must be positive and finite')
    from .scenarios import specification
    try:
        specification(args)
    except ValueError as error:
        p.error(str(error))
    if min(args.iterations, args.checkpoint_frequency, args.train_batch_size, args.minibatch_size, args.num_epochs) <= 0:
        p.error("Iterations, batch sizes, epochs and checkpoint frequency must be positive.")
    if args.episode_length <= 0 or any(n <= 0 for n in args.policy_hidden_sizes):
        p.error("Episode length and hidden sizes must be positive.")
    if not 0 < args.train_reference_scale <= 1 or not 0 < args.evaluation_reference_scale <= 1:
        p.error("Reference scales must be in (0, 1].")
    if args.minibatch_size > args.train_batch_size:
        p.error("Minibatch cannot exceed train batch.")
    if not all(math.isfinite(v) and v>0 for v in (args.reward_scale,args.vf_clip_param,args.error_observation_gain)):
        p.error('Reward scale and critic loss cap must be finite and positive')
    if args.reward_scale!=1. and args.action_mode!='current':
        p.error('Reward scaling is currently supported for current-reference mode')
    if args.error_observation_gain!=1. and args.action_mode!='current':
        p.error('Error observation gain is currently supported for current-reference mode')
    if args.num_axis_levels < 3 or args.num_axis_levels % 2 == 0:
        p.error("Axis levels must be odd and >=3.")
    if args.intrinsic_reward_coeff <= 0 or not 0 <= args.forward_loss_weight <= 1:
        p.error("ICM coefficient must be positive; forward loss weight must be in [0,1].")
    if args.tsne_samples < 50:
        p.error("--tsne-samples must be >=50.")
    if args.num_env_runners < 0:
        p.error("--num-env-runners must be nonnegative")
    if not 0 < args.current_reference_limit <= 1 or not math.isfinite(args.current_reference_limit):
        p.error("--current-reference-limit must be in (0,1]")
    if not all(math.isfinite(x) and x >= 0 for x in (args.current_reference_slew, args.current_bandwidth,
            args.tracking_weight, args.tracking_absolute_weight, args.current_weight, args.id_weight)) or args.current_bandwidth == 0:
        p.error("Current/reward parameters must be finite and nonnegative; bandwidth positive")
    if not all(math.isfinite(x) and x >= 0 for x in (args.command_slew_v_per_s, args.command_smoothness_weight)):
        p.error("Slew and smoothness weight must be finite and nonnegative.")
    if not 0 < args.steady_fraction < 1:
        p.error("--steady-fraction must be in (0,1).")
    if args.intrinsic_final_coeff is not None and (not math.isfinite(args.intrinsic_final_coeff) or args.intrinsic_final_coeff < 0 or args.action_mode == "discrete"):
        p.error("Decay requires continuous/current mode and a nonnegative finite final coefficient.")
    if args.intrinsic_decay_iterations < 2:
        p.error("--intrinsic-decay-iterations must be >=2.")
    try:
        run_dir(args)
        if args.command == "list":
            for motor in SCRIPTS:
                for manifest in sorted((ROOT / "RL_models" / motor).glob("*/*/manifest.json")):
                    data = json.loads(manifest.read_text())
                    print(manifest.parent.relative_to(ROOT), data.get("status"), data.get("completed_iteration", 0))
        elif args.command == "train":
            train(args)
        elif args.command == "tune-baseline":
            from .baselines import tune
            tune(args)
        else:
            return evaluate(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        p.error(str(error))
