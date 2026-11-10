"""Generate held-out PPO/ICM/baseline figures only after validation passes.

This command never trains, silently changes checkpoints, or smooths the signals.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from . import runner


def selected_report(motor, controller, run):
    root=runner.ROOT/'outputs/rllib'/motor/'validation'/run
    reports=sorted(root.glob('*/validation.json'))
    if not reports:
        raise ValueError(f'Run validation first: {motor}/{controller}/{run}')
    matching=[]
    for path in reports:
        report=json.loads(path.read_text()); args=report['arguments']
        if (args['motor'],args['controller'],args['run'])==(motor,controller,run):
            matching.append((path,report))
    if not matching: raise ValueError(f'No validation for {controller}/{run}')
    path,report=matching[-1]; args=report['arguments']
    checkpoint=report.get('selected_checkpoint')
    records=[r for r in report['records'] if r['checkpoint']==checkpoint]
    if not checkpoint or not records or not records[0].get('quality_pass',False):
        raise ValueError(f'No acceptable checkpoint in the latest validation: {path}')
    expected=runner.ROOT/'RL_models'/motor/controller/run
    if Path(checkpoint).resolve().parent!=expected.resolve():
        raise ValueError('Checkpoint is outside the requested run')
    if not {501,502}.issubset(args['seeds']) or not {'steps','random'}.issubset(args['profiles']):
        raise ValueError('Presentation requires steps/random validation on both 501 and 502')
    return path,report,checkpoint


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motor',choices=['dc','pmsm'],default='pmsm')
    p.add_argument('--run',required=True)
    p.add_argument('--show',action='store_true')
    opts=p.parse_args(argv)
    from .current_control import environment_config
    from .evaluation import episode,save_reports
    from .validate import quality
    from .baselines import tune,load_gains
    from ray.rllib.core.rl_module.rl_module import RLModule
    selected={c:selected_report(opts.motor,c,opts.run) for c in ('ppo','icm')}
    configs={}; policies={}; report_args={}
    args=runner.parser().parse_args(['eval','--motor',opts.motor,'--run',opts.run])
    for controller,(_,report,checkpoint) in selected.items():
        manifest=json.loads((Path(checkpoint).parent/'manifest.json').read_text())
        restored=runner.parser().parse_args(['eval','--motor',opts.motor,'--run',opts.run])
        for key,value in manifest['settings'].items():
            if hasattr(restored,key): setattr(restored,key,value)
        if restored.action_mode!='current': raise ValueError('Only current-reference runs are supported')
        configs[controller]=environment_config(restored)
        report_args[controller]=SimpleNamespace(**report['arguments'])
        policies[controller]=RLModule.from_checkpoint(str(Path(checkpoint)/'learner_group/learner/rl_module/default_policy'))
        args=restored
    if configs['ppo']!=configs['icm']: raise ValueError('Controllers use different environments')
    keys=('max_ripple','max_mae','max_speed_pp','max_offset','max_current_ripple_fraction')
    if any(getattr(report_args['ppo'],k)!=getattr(report_args['icm'],k) for k in keys):
        raise ValueError('Controllers were selected with different validation thresholds')
    args.command='tune-baseline'; args.show=False
    baseline=tune(args)
    args.baseline_config=str(baseline)
    config=configs['ppo']; gains=load_gains(args,config)
    output=runner.ROOT/'outputs/rllib'/opts.motor/'showcase'/opts.run/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    module=runner.load_motor(args); cases=[]
    for profile,seed in (('steps',901),('random',902)):
        args.profile=profile; args.seed=seed; args.show=False
        results={c:episode(args,module,policy,config) for c,policy in policies.items()}
        results['pi' if opts.motor=='dc' else 'foc']=episode(args,module,env_config=config,baseline_gains=gains)
        case_dir=output/f'{profile}_seed{seed}'; case_dir.mkdir()
        save_reports(args,results,case_dir)
        for name,result in results.items():
            assessment=quality(result,report_args['ppo'])
            cases.append(dict(profile=profile,seed=seed,controller=name,quality=assessment,metrics=result['metrics']))
        import matplotlib.pyplot as plt
        if opts.show: plt.show()
        plt.close('all')
    for name in {c['controller'] for c in cases}:
        group=[c for c in cases if c['controller']==name]
        mean_mae=sum(c['metrics']['MAE_speed_rad_s'] for c in group)/len(group)
        for c in group:
            c['quality']['mean_profile_MAE']=mean_mae
            c['quality']['passed']=c['quality']['passed'] and mean_mae<=report_args['ppo'].max_mae
    ready=all(c['quality']['passed'] for c in cases)
    runner.save_json(output/'readiness.json',dict(ready=ready,cases=cases,baseline=str(baseline),
        validations={c:str(v[0]) for c,v in selected.items()},checkpoints={c:v[2] for c,v in selected.items()}))
    lines=['# TDK demonstration report','',f"Quality gate: {'PASS' if ready else 'FAIL — inspect failed cases'}",'',
           'Nominal held-out demonstration only; not a proof of global stability or robustness.',
           '', '| Profile | Controller | MAE [rad/s] | Quality passed |','|---|---|---:|---|']
    for c in cases: lines.append(f"| {c['profile']} | {c['controller']} | {c['metrics']['MAE_speed_rad_s']:.3f} | {c['quality']['passed']} |")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'Report: {output / "REPORT.md"}')


if __name__=='__main__': main()
