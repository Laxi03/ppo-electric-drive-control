"""Benchmark sampling, screen PPO settings, then train paired current-control trials."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from . import runner

CANDIDATES = {
    "balanced": dict(command_smoothness_weight=.05, tracking_weight=100., current_reference_limit=.8),
    "smooth": dict(command_smoothness_weight=.1, tracking_weight=100., current_reference_limit=.7),
    "tracking": dict(command_smoothness_weight=.02, tracking_weight=200., current_reference_limit=.9),
}


def options(values):
    return [part for key,value in values.items() for part in ("--"+key.replace("_", "-"), str(value))]


def validate(args, motor, checkpoint):
    """Fixed validation seeds and profiles; never rank using the final seed 123."""
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from .evaluation import episode
    local = runner.parser().parse_args(["eval", "--motor", motor])
    module = runner.load_motor(local)
    records = []
    ray.init(include_dashboard=False, log_to_driver=False, logging_level="ERROR")
    algo = None
    try:
        algo = Algorithm.from_checkpoint(str(checkpoint))
        for seed in (501,502):
            for profile in ("steps","random"):
                local.seed, local.profile = seed, profile
                result = episode(local, module, algo.get_module(module.DEFAULT_MODULE_ID), algo.config.env_config)
                m, rows = result["metrics"], result["plateaus"]
                offset = max((abs(r["steady_offset_rad_s"]) for r in rows), default=float("inf"))
                ripple = max((r["speed_ripple_RMS"] for r in rows), default=float("inf"))
                settled = [r["settling_time_s"] for r in rows]
                passes = (m["completed_episode"] and not m["safety_violation"] and offset <= args.max_offset
                          and ripple <= args.max_speed_ripple and bool(settled)
                          and all(np.isfinite(s) and s <= args.max_settling for s in settled))
                score = (m["RMSE_speed_rad_s"]/(900 if motor=="dc" else 600)
                         + offset/100 + ripple/10 + .02*m["RMS_current_A"]/(6 if motor=="dc" else 3))
                if not m["completed_episode"] or m["safety_violation"]:
                    score += 1000
                records.append(dict(seed=seed, profile=profile, score=score, targets_passed=bool(passes),metrics=m))
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()
    return dict(score=float(np.mean([r["score"] for r in records])), validation=records)


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["benchmark","screen","final"])
    p.add_argument("--run", default="current_v2_seed123")
    p.add_argument("--motor", choices=["dc","pmsm","both"], default="both")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-env-runners", type=int, default=2)
    p.add_argument("--screen-iterations", type=int, default=150)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--benchmark-iterations", type=int, default=3)
    p.add_argument("--max-speed-ripple", type=float, default=1., help="Validation target in rad/s RMS, every plateau")
    p.add_argument("--max-offset", type=float, default=5., help="Validation target in rad/s, every plateau")
    p.add_argument("--max-settling", type=float, default=.2, help="Validation target in seconds, 5 rad/s band")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--use-screen", action="store_true", help="Explicitly promote a completed screening winner; default final uses fixed settings")
    p.add_argument("--profiles", nargs="+", choices=["steps","random","sine","mixed"],
                   default=["steps","random","sine","mixed"], help="Final held-out test profiles")
    args=p.parse_args()
    if min(args.screen_iterations,args.iterations,args.benchmark_iterations) < 1 or args.num_env_runners < 0:
        p.error("Iteration budgets must be positive; worker count nonnegative")
    if args.seed in (501,502):
        p.error("Seeds 501/502 are reserved for validation; choose another training/test seed")
    if not all(np.isfinite(x) and x > 0 for x in (args.max_offset,args.max_speed_ripple,args.max_settling)):
        p.error("Validation targets must be positive and finite")
    # Validate the user-controlled directory name before using it anywhere.
    runner.run_dir(runner.parser().parse_args(["train","--run",args.run]))
    root=runner.ROOT/"outputs"/"rllib"/"current_suite"/args.run
    root.mkdir(parents=True,exist_ok=True)
    motors=("dc","pmsm") if args.motor=="both" else (args.motor,)
    for motor in motors:
        common=["--motor",motor,"--action-mode","current","--seed",str(args.seed),
                "--train-reference-scale","1","--num-env-runners",str(args.num_env_runners)]
        if args.stage=="benchmark":
            records=[]
            for workers in (0,2,4):
                name=f"{args.run}_bench{workers}"
                start=time.perf_counter()
                runner.main(["train",*common,"--run",name,"--num-env-runners",str(workers),
                             "--iterations",str(args.benchmark_iterations)])
                history=runner.ROOT/"RL_models"/motor/"ppo"/name/"training_history.csv"
                with history.open() as stream:
                    rows=list(csv.DictReader(stream))
                # Exclude the cold first training iteration when possible.
                times=[float(r["iteration_wall_s"]) for r in rows[1:] or rows]
                records.append(dict(workers=workers, median_iteration_s=float(np.median(times)),
                                    total_wall_s=time.perf_counter()-start))
            runner.save_json(root/f"{motor}_benchmark.json",records)
            print(json.dumps(records,indent=2))
        elif args.stage=="screen":
            records=[]
            for name,values in CANDIDATES.items():
                run=f"{args.run}_{name}"
                command=["train",*common,"--run",run,"--iterations",str(args.screen_iterations),*options(values)]
                folder=runner.ROOT/"RL_models"/motor/"ppo"/run
                if args.resume and (folder/"manifest.json").exists():
                    command.append("--resume")
                runner.main(command)
                records.append(dict(run=run,settings=values,**validate(args,motor,runner.latest(folder))))
                runner.save_json(root/f"{motor}_screen.json",dict(candidates=records))
            winner=min(records,key=lambda r:r["score"])
            runner.save_json(root/f"{motor}_screen.json",dict(selected=winner,candidates=records,
                num_env_runners=args.num_env_runners,seed=args.seed,
                targets=dict(speed_ripple_RMS=args.max_speed_ripple,absolute_offset=args.max_offset,settling_s=args.max_settling)))
            print(f"Selected {winner['run']}; validation targets passed: {all(r['targets_passed'] for r in winner['validation'])}")
        else:
            run, extra = args.run, []
            if args.use_screen:
                selection_path=root/f"{motor}_screen.json"
                if not selection_path.is_file():
                    p.error(f"Missing screening: {selection_path}")
                selection=json.loads(selection_path.read_text())
                if "selected" not in selection:
                    p.error("Screening is incomplete")
                if selection["seed"] != args.seed or selection["num_env_runners"] != args.num_env_runners:
                    p.error("Use the screen's seed and worker count")
                run=selection["selected"]["run"]
                extra=options(selection["selected"]["settings"])
            ppo_folder=runner.ROOT/"RL_models"/motor/"ppo"/run
            if runner.checkpoints(ppo_folder) and int(runner.latest(ppo_folder).name.split("_")[-1]) > args.iterations:
                p.error("Final iteration budget is below the selected PPO checkpoint")
            for controller, suffix in (("ppo",""),("icm",""),("icm","_decay")):
                name=run+suffix
                command=["train",*common,*extra,"--controller",controller,"--run",name,"--iterations",str(args.iterations)]
                if controller=="icm":
                    command += ["--intrinsic-reward-coeff", ".02" if suffix else ".005"]
                    if suffix:
                        command += ["--intrinsic-final-coeff",".001","--intrinsic-decay-iterations",str(args.iterations)]
                folder=runner.ROOT/"RL_models"/motor/controller/name
                if (controller=="ppo" and args.use_screen) or (args.resume and (folder/"manifest.json").exists()):
                    command.append("--resume")
                runner.main(command)
            # Fit the classical outer PI using validation seeds, same inner loop.
            from .baselines import tune
            baseline=tune(runner.parser().parse_args(["tune-baseline",*common,*extra,"--run",run]))
            artifacts=[]
            for suffix in ("","_decay"):
                for profile in args.profiles:
                    output=runner.main(["compare",*common,"--run",run+suffix,"--profile",profile,
                        "--ppo-checkpoint",str(runner.latest(ppo_folder)),
                        "--curiosity-checkpoint",str(runner.latest(runner.ROOT/"RL_models"/motor/"icm"/(run+suffix))),
                        "--baseline-config",str(baseline)])
                    artifacts.append(dict(variant="decay" if suffix else "reduced",profile=profile,output=str(output)))
            runner.save_json(root/f"{motor}_final.json",dict(selected_run=run,iterations=args.iterations,
                baseline=str(baseline),evaluations=artifacts))
    print(f"Suite reports: {root}")


if __name__=="__main__":
    run()
