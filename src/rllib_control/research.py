"""Frozen multi-seed experiments, paired robustness evaluation and aggregation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from . import runner
from .scenarios import PRESETS

STUDIES={
    "stability": [("ppo","ppo","current",["--stability-profile","pmsm-v1"]),
                  ("icm_reduced","icm","current",["--stability-profile","pmsm-v1",
                     "--intrinsic-reward-coeff",".005"])],
    "main": [("ppo","ppo","current",[]),
             ("icm_reduced","icm","current",["--intrinsic-reward-coeff",".005"]),
             ("icm_decay","icm","current",["--intrinsic-reward-coeff",".02","--intrinsic-final-coeff",".001"])],
    "architecture": [("current","ppo","matched-current",[]),("voltage","ppo","matched-voltage",[])],
    "reward": [("smooth","ppo","current",["--command-smoothness-weight",".05"]),
               ("no_smooth","ppo","current",["--command-smoothness-weight","0"])],
}


def fingerprints():
    from importlib.util import find_spec

    package = Path(__file__).resolve().parent
    files = {f"rllib_control/{p.name}": p for p in package.glob("*.py")}
    for name in runner.SCRIPTS.values():
        spec = find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Cannot locate simulator source: {name}")
        files[f"{name}.py"] = Path(spec.origin)
    return {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in sorted(files.items())}


def invoke(command):
    try:
        return runner.main(command)
    except SystemExit as error:
        raise RuntimeError(f"CLI rejected job (exit {error.code}): {' '.join(command)}") from error


def make_protocol(args):
    if args.study == "stability" and args.motors != ["pmsm"]:
        raise ValueError("The stability study requires --motors pmsm")
    jobs=[]
    for motor in args.motors:
        for variant,controller,mode,extra in STUDIES[args.study]:
            for seed in args.seeds:
                name=f"{args.experiment}_{args.study}_{variant}_seed{seed}"
                command=["train","--run",name,"--motor",motor,"--controller",controller,
                    "--action-mode",mode,"--seed",str(seed),"--iterations",str(args.iterations),
                    "--intrinsic-decay-iterations",str(max(2,args.iterations)),"--train-reference-scale","1",
                    "--num-env-runners",str(args.num_env_runners),"--episode-length",str(args.episode_length),
                    "--train-batch-size",str(args.train_batch_size),"--minibatch-size",str(args.minibatch_size),
                    "--num-epochs",str(args.num_epochs),*extra]
                parsed=runner.parser().parse_args(command)
                runner.run_dir(parsed)
                jobs.append(dict(id=f"{motor}_{variant}_{seed}",motor=motor,variant=variant,
                    controller=controller,train_seed=seed,run=name,command=command,settings=runner.settings(parsed)))
        if args.study in ("main","stability"):
            jobs.append(dict(id=motor+"_baseline",motor=motor,variant="baseline",controller="baseline",
                             train_seed=None,run=args.experiment+"_baseline"))
    return dict(version=1,study=args.study,iterations=args.iterations,episode_length=args.episode_length,
        test_seeds=args.test_seeds,profiles=args.profiles,scenarios=args.scenarios,
        jobs=jobs,source_sha256=fingerprints(),evaluation_scale=1.,validation_seeds=[501,502])


def case_id(job, profile, scenario, seed):
    return f"{job['id']}__{profile}__{scenario}__test{seed}"


def cases(protocol):
    for job in protocol["jobs"]:
        for profile in protocol["profiles"]:
            for scenario in protocol["scenarios"]:
                for seed in protocol["test_seeds"]:
                    yield job,profile,scenario,seed


def train(protocol, root, resume):
    milestones=protocol.get('training_milestones')
    if milestones is not None:
        if (not milestones or any(type(v) is not int or v<=0 for v in milestones)
                or milestones!=sorted(set(milestones)) or milestones[-1]!=protocol['iterations']):
            raise ValueError('Training milestones must increase strictly to the final iteration budget')
    for job in protocol["jobs"]:
        if job["controller"]=="baseline" or job.get('reuse_checkpoint'):continue
        args=runner.parser().parse_args(job["command"])
        folder=runner.run_dir(args)
        command=list(job["command"])
        try:
            if milestones:
                if (folder/'manifest.json').exists() and not resume:
                    raise FileExistsError(f'Fresh confirmation run already exists: {folder}')
                for target in milestones:
                    stage_command=list(command)
                    stage_command[stage_command.index('--iterations')+1]=str(target)
                    path=folder/'manifest.json'
                    if path.exists():
                        stored=json.loads(path.read_text())
                        if stored['settings']!=job['settings']:raise ValueError('Milestone settings differ from frozen plan')
                        if stored.get('status') not in ('complete','interrupted_or_failed'):
                            raise ValueError('Refusing a possibly active milestone trainer')
                        if stored.get('completed_iteration',0)>=target:continue
                        stage_command.append('--resume')
                    print(f"MILESTONE {job['id']}: {target}/{protocol['iterations']}",flush=True)
                    invoke(stage_command)
                    runner.save_json(root/(job['id']+f'_milestone{target}.json'),dict(status='complete',iteration=target))
                stored=json.loads((folder/'manifest.json').read_text())
                if stored.get('status')!='complete' or stored.get('completed_iteration')!=protocol['iterations']:
                    raise ValueError('Milestone training did not complete the final budget')
                if any(not (folder/f'iteration_{v:04d}').is_dir() for v in milestones):
                    raise ValueError('Missing required milestone checkpoint')
                stored['training_protocol']=dict(milestones=milestones,fresh_start=True,
                    source_protocol=str((root/'protocol.json').resolve()))
                runner.save_json(folder/'manifest.json',stored)
            else:
                if resume and (folder/"manifest.json").exists():command.append("--resume")
                invoke(command)
            runner.save_json(root/(job["id"]+"_train.json"),dict(status="complete"))
        except Exception as error:
            runner.save_json(root/(job["id"]+"_train.json"),dict(status="error",error=repr(error)))
            print(f"Training failed: {job['id']}: {error}",flush=True)


def freeze_baselines(protocol,root):
    """Tune on validation only, snapshot gains, and refuse changed frozen files."""
    from .baselines import tune
    paths={}
    for job in protocol['jobs']:
        if job['controller']!='baseline':continue
        index=root/(job['motor']+'_baseline.json')
        if index.exists():
            locked=json.loads(index.read_text());path=Path(locked['path'])
            if not locked.get('sha256') or hashlib.sha256(path.read_bytes()).hexdigest()!=locked['sha256']:
                raise ValueError('Frozen baseline missing or modified; do not retune on test data')
        else:
            if job.get('baseline_config'):
                path=Path(job['baseline_config'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=job['baseline_sha256']:
                    raise ValueError('Parent frozen baseline changed')
            else:
                command=['tune-baseline',*job['environment_command'][1:]]
                path=tune(runner.parser().parse_args(command))
            data=json.loads(path.read_text())
            if set(data['validation_seeds'])!={501,502}:raise ValueError('Baseline must be tuned on validation seeds 501/502 only')
            from .current_control import environment_config
            args=runner.parser().parse_args(job['environment_command'])
            if data['motor']!=job['motor'] or data['environment']!=environment_config(args):
                raise ValueError('Frozen baseline environment differs from the planned baseline')
            snapshot=root/(job['motor']+'_frozen_baseline.json')
            runner.save_json(snapshot,data);path=snapshot
            runner.save_json(index,dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                       tuning_seeds=[501,502],frozen_before_evaluation=True))
        paths[job['motor']]=path
    return paths


def freeze_models(protocol,root):
    """Do not consume held-out cases until EVERY planned checkpoint is ready."""
    frozen={}
    for job in protocol['jobs']:
        if job['controller']=='baseline':continue
        args=runner.parser().parse_args(job['command']);folder=runner.run_dir(args)
        manifest_path=folder/'manifest.json';manifest=json.loads(manifest_path.read_text())
        checkpoint=folder/f"iteration_{protocol['iterations']:04d}"
        if (manifest.get('status')!='complete' or manifest.get('completed_iteration',0)<protocol['iterations']
                or manifest['settings']!=job['settings'] or not checkpoint.is_dir()):
            raise ValueError(f"All final models must be complete before held-out tests: {job['id']}")
        if protocol.get('training_milestones'):
            if manifest.get('training_protocol',{}).get('milestones')!=protocol['training_milestones']:
                raise ValueError('Model does not have the frozen matched training schedule')
        files=sorted(p for p in checkpoint.rglob('*') if p.is_file())
        if not files:raise ValueError(f'Empty checkpoint: {checkpoint}')
        frozen[job['id']]=dict(checkpoint=str(checkpoint.resolve()),
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            files_sha256={str(p.relative_to(checkpoint)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    path=root/'frozen_models.json'
    if path.exists():
        if json.loads(path.read_text())!=frozen:raise ValueError('Models changed after the held-out evaluation freeze')
    else:runner.save_json(path,frozen)


def evaluate(protocol, root, resume):
    from .evaluation import episode,save_reports
    from .current_control import environment_config
    from .baselines import tune
    if protocol.get('require_frozen_baseline'):freeze_models(protocol,root)
    case_root=root/"cases";case_root.mkdir(exist_ok=True)
    baseline_paths=freeze_baselines(protocol,root) if protocol.get('require_frozen_baseline') else {}
    for job,profile,scenario,seed in cases(protocol):
        key=case_id(job,profile,scenario,seed);record=case_root/(key+".json")
        if resume and record.exists() and json.loads(record.read_text()).get("status")=="complete":continue
        common=["--motor",job["motor"],"--run",job["run"],"--seed",str(seed),
                "--profile",profile,"--scenario",scenario,"--episode-length",str(protocol["episode_length"])]
        stability_args=["--stability-profile","pmsm-v1"] if protocol['study']=='stability' else []
        try:
            if job["controller"]=="baseline":
                if job["motor"] not in baseline_paths:
                    index=root/(job["motor"]+"_baseline.json")
                    if index.exists():path=Path(json.loads(index.read_text())["path"])
                    else:
                        tune_command=(["tune-baseline",*job['environment_command'][1:]]
                                      if job.get('environment_command') else
                                      ["tune-baseline","--motor",job["motor"],"--run",job["run"],
                                       "--episode-length",str(protocol["episode_length"]),*stability_args])
                        tune_args=runner.parser().parse_args(tune_command)
                        path=tune(tune_args)
                        runner.save_json(index,dict(path=str(path)))
                    baseline_paths[job["motor"]]=path
                baseline_command=(["eval",*job['environment_command'][1:],*common]
                                  if job.get('environment_command') else ["eval",*common,*stability_args])
                args=runner.parser().parse_args([*baseline_command,"--baseline-config",str(baseline_paths[job["motor"]])])
                result=episode(args,runner.load_motor(args),env_config=environment_config(args))
                output=case_root/(key+"_baseline")
                output.mkdir(exist_ok=True)
                save_reports(args,{"baseline":result},output)
                metrics=result["metrics"]
                scenario_meta=result["scenario"]
            else:
                args=runner.parser().parse_args(job["command"])
                folder=runner.run_dir(args)
                checkpoint=folder/f"iteration_{protocol['iterations']:04d}"
                stored=json.loads((folder/"manifest.json").read_text())
                if stored["settings"]!=job["settings"]:raise ValueError("Training settings differ from frozen protocol")
                output=invoke(["eval",*common,"--controller",job["controller"],"--checkpoint",str(checkpoint)])
                metrics=json.loads((output/"metrics.json").read_text())[job["controller"]]
                scenario_meta=json.loads((output/"scenarios.json").read_text())[job["controller"]]
            status="complete" if metrics["completed_episode"] and not metrics["safety_violation"] else "failed_episode"
            paired_reference=None
            measurement_error=None
            if protocol.get('extended_metrics'):
                from .tdk_extension import enrich_metrics
                try:
                    metrics,paired_reference=enrich_metrics(output,metrics,job['motor'],profile)
                except ValueError as error:
                    if status=='complete':raise
                    measurement_error=str(error)
                    metrics=dict(metrics,engineering_pass=0.,largest_step_rise_s=None)
                nominal_path=case_root/(case_id(job,profile,'nominal',seed)+'.json')
                nominal=json.loads(nominal_path.read_text()) if nominal_path.exists() else {}
                nominal_mae=metrics['MAE_speed_rad_s'] if scenario=='nominal' else nominal.get('metrics',{}).get('MAE_speed_rad_s')
                metrics['MAE_degradation_vs_nominal_rad_s']=(metrics['MAE_speed_rad_s']-nominal_mae
                    if isinstance(nominal_mae,(int,float)) and math.isfinite(nominal_mae) else None)
            runner.save_json(record,dict(status=status,job=job["id"],profile=profile,scenario=scenario,
                test_seed=seed,metrics=metrics,scenario_metadata=scenario_meta,output=str(output),
                paired_reference_sha256=paired_reference,measurement_error=measurement_error))
        except Exception as error:
            runner.save_json(record,dict(status="error",job=job["id"],profile=profile,scenario=scenario,
                test_seed=seed,error=repr(error)))
            print(f"Evaluation failed: {key}: {error}",flush=True)


METRICS=("MAE_speed_rad_s","RMSE_speed_rad_s","mean_plateau_speed_ripple_RMS",
         "mean_plateau_current_magnitude_ripple_RMS","RMS_current_A","current_I2t_A2s","time_at_voltage_rail_s")


def write_csv(path,rows,fields):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def aggregate(protocol,root):
    # Enumerate EXPECTED cases, so missing/error runs cannot disappear from means.
    metrics=protocol.get('metrics',METRICS)
    raw=[];groups={}
    for job,profile,scenario,seed in cases(protocol):
        path=root/"cases"/(case_id(job,profile,scenario,seed)+".json")
        record=json.loads(path.read_text()) if path.exists() else dict(status="missing")
        row=dict(motor=job["motor"],variant=job["variant"],train_seed=job["train_seed"],
                 test_seed=seed,profile=profile,scenario=scenario,status=record["status"],error=record.get("error",""))
        row.update({m:record.get("metrics",{}).get(m) for m in metrics})
        raw.append(row);groups.setdefault((job["motor"],job["variant"],profile,scenario),[]).append(row)
    write_csv(root/"cases.csv",raw,list(raw[0]) if raw else ["status"])
    summaries=[]
    for (motor,variant,profile,scenario),rows in groups.items():
        for metric in metrics:
            seeds=sorted(set(r["train_seed"] for r in rows),key=str)
            values=[]
            for seed in seeds:
                group=[r for r in rows if r["train_seed"]==seed]
                # A seed contributes only if ALL planned test cases succeeded.
                if all(r["status"]=="complete" and isinstance(r[metric],(float,int)) and math.isfinite(r[metric]) for r in group):
                    values.append(float(np.mean([r[metric] for r in group])))
            summaries.append(dict(motor=motor,variant=variant,profile=profile,scenario=scenario,metric=metric,
                expected_cases=len(rows),complete_cases=sum(r["status"]=="complete" for r in rows),
                failed_cases=sum(r["status"]=="failed_episode" for r in rows),error_cases=sum(r["status"]=="error" for r in rows),
                missing_cases=sum(r["status"]=="missing" for r in rows),
                valid_seed_groups=len(values),mean=float(np.mean(values)) if values else None,
                std_across_training_seeds=float(np.std(values,ddof=1)) if len(values)>1 and variant!="baseline" else None))
    write_csv(root/"summary.csv",summaries,list(summaries[0]) if summaries else ["metric"])
    lines=["# Experiment summary", "", "Test cases are averaged within each training seed before computing seed mean/sample SD.",
           "Only complete seed groups enter each metric; failed, error and missing cases remain counted. Baseline has no training-seed SD.",
           "", "|Motor|Variant|Profile|Scenario|MAE mean ± SD (rad/s)|Valid seed groups|Complete / expected|Failed / error / missing|",
           "|---|---|---|---|---|---:|---|---|"]
    for row in summaries:
        if row["metric"]!="MAE_speed_rad_s":continue
        mean="—" if row["mean"] is None else f"{row['mean']:.4g}"
        sd="—" if row["std_across_training_seeds"] is None else f"{row['std_across_training_seeds']:.3g}"
        lines.append(f"|{row['motor']}|{row['variant']}|{row['profile']}|{row['scenario']}|{mean} ± {sd}|{row['valid_seed_groups']}|{row['complete_cases']} / {row['expected_cases']}|{row['failed_cases']} / {row['error_cases']} / {row['missing_cases']}|")
    (root/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(f"Summary: {root/'SUMMARY.md'}")
    return summaries


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["plan","train","evaluate","aggregate","all"])
    p.add_argument("--experiment",default="tdk_v1")
    p.add_argument("--study",choices=list(STUDIES),default="main")
    p.add_argument("--motors",nargs="+",choices=["dc","pmsm"],default=["dc","pmsm"])
    p.add_argument("--seeds",nargs="+",type=int,default=[123,234,345])
    p.add_argument("--test-seeds",nargs="+",type=int,default=[901,902,903])
    p.add_argument("--profiles",nargs="+",choices=["steps","random","sine","mixed"],default=["steps","random"])
    p.add_argument("--scenarios",nargs="+",choices=list(PRESETS),default=list(PRESETS))
    p.add_argument("--iterations",type=int,default=1000)
    p.add_argument("--num-env-runners",type=int,default=4)
    p.add_argument("--episode-length",type=float,default=8.)
    p.add_argument("--train-batch-size",type=int,default=4096)
    p.add_argument("--minibatch-size",type=int,default=256)
    p.add_argument("--num-epochs",type=int,default=6)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args(argv)
    if set(args.seeds)&(set(args.test_seeds)|{501,502}) or set(args.test_seeds)&{501,502}:
        p.error("Training, validation (501/502) and test seeds must be disjoint")
    if min(args.seeds+args.test_seeds)<0:p.error("Seeds must be nonnegative")
    for values in (args.motors,args.seeds,args.test_seeds,args.profiles,args.scenarios):
        if len(values)!=len(set(values)):p.error("Duplicate motors/seeds/profiles/scenarios are not allowed")
    if min(args.iterations,args.train_batch_size,args.minibatch_size,args.num_epochs)<=0 or args.minibatch_size>args.train_batch_size or args.num_env_runners<0 or not math.isfinite(args.episode_length) or args.episode_length<=0:
        p.error("Invalid training budget or episode length")
    runner.run_dir(runner.parser().parse_args(["train","--run",args.experiment]))
    root=runner.ROOT/"outputs"/"rllib"/"research"/args.experiment/args.study
    path=root/"protocol.json"
    requested=make_protocol(args)
    if path.exists():
        protocol=json.loads(path.read_text())
        if args.stage in ("plan","train","all") and protocol!=requested:
            p.error("Frozen protocol/source differs. Use a new experiment name or the original configuration and code")
        if args.stage=="evaluate" and protocol["source_sha256"]!=fingerprints():
            p.error("Source changed since the frozen protocol; use its original code or create a new experiment")
    else:
        if args.stage not in ("plan","train","all"):p.error("Create the experiment with plan/train/all first")
        root.mkdir(parents=True,exist_ok=True);protocol=requested;runner.save_json(path,protocol)
    if args.stage=="plan":print(f"Frozen plan: {path}; {len(protocol['jobs'])} jobs including baselines")
    if args.stage in ("train","all"):train(protocol,root,args.resume)
    if args.stage in ("evaluate","all"):evaluate(protocol,root,args.resume)
    if args.stage in ("aggregate","all"):aggregate(protocol,root)


if __name__=="__main__":main()
