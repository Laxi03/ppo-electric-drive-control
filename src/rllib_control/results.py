"""Curated, manifest-restored DC/PMSM evaluation; never a held-out qualification."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from . import runner

PRESETS = {
    'dc': dict(ppo='tdk_v1_main_ppo_seed123', icm='tdk_v1_main_icm_reduced_seed123'),
    'pmsm': dict(ppo='tdk_confirm700_v1_ppo_seed456', icm='tdk_confirm700_v1_icm_fixed_seed456'),
}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motor', choices=['dc','pmsm'], required=True)
    p.add_argument('--controller', choices=['all','ppo','icm','pi','foc'], default='all')
    p.add_argument('--profile', choices=['mixed','steps','random','sine'], default='mixed')
    p.add_argument('--seed', type=int, default=123, help='Demonstration seed, not a held-out confirmation')
    p.add_argument('--scenario', choices=['nominal','load_step','parameter_shift','resistance_jump','sensor_noise'], default='nominal')
    p.add_argument('--show', action='store_true')
    p.add_argument('--plot-reward', action='store_true')
    p.add_argument('--plot-tsne', action='store_true')
    p.add_argument('--ppo-run')
    p.add_argument('--icm-run')
    p.add_argument('--baseline-config')
    p.add_argument('--train', action='store_true', help='Train a NEW named run using the selected manifest settings')
    p.add_argument('--run', help='Required NEW destination name with --train')
    p.add_argument('--iterations', type=int, help='Training budget; defaults to the selected template budget')
    p.add_argument('--dry-run', action='store_true')
    return p


def restored(motor, controller, run):
    args=runner.parser().parse_args(['eval','--motor',motor,'--controller',controller,'--run',run])
    folder=runner.run_dir(args)
    manifest_path = folder/'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'No model manifest at {manifest_path}. Train this run first, place trusted checkpoints '
            'in RL_models/, or use --controller pi (DC) / foc (PMSM) without checkpoints.'
        )
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='complete':
        raise ValueError(f'Incomplete run: {folder}')
    for key,value in manifest['settings'].items():
        if hasattr(args,key): setattr(args,key,value)
    if (args.motor,args.controller)!=(motor,controller) or args.action_mode!='current':
        raise ValueError('Wrong motor/controller or unsupported non-current checkpoint')
    checkpoint=folder/f"iteration_{manifest['completed_iteration']:04d}"
    if not checkpoint.is_dir(): raise FileNotFoundError(checkpoint)
    return args,manifest,checkpoint


def baseline_settings(motor, override=None):
    path = (Path(override).expanduser().resolve() if override else
            Path(__file__).resolve().parent/'data'/f'{motor}_baseline.json')
    data = json.loads(path.read_text())
    if 'path' in data:
        path = (path.parent/Path(data['path'])).resolve()
        data = json.loads(path.read_text())
    if data.get('motor') != motor or data.get('environment', {}).get('architecture') != 'current_reference_v2':
        raise ValueError('Baseline must match the selected motor and current-reference architecture')
    return path, data


def training_command(args):
    """Serialize known parser fields, leaving derived manifest metadata untouched."""
    command=['train']
    for action in runner.parser()._actions:
        if not action.option_strings or action.dest in ('help','checkpoint','ppo_checkpoint','curiosity_checkpoint'):
            continue
        value=getattr(args,action.dest)
        if value is None: continue
        if isinstance(action,argparse._StoreTrueAction):
            if value: command.append(action.option_strings[0])
        else:
            command.append(action.option_strings[0])
            if isinstance(value, list):
                command.extend(str(v) for v in value)
            else:
                command.append(str(value))
    return command


def main(argv=None):
    p=parser(); opts=p.parse_args(argv)
    baseline_name='pi' if opts.motor=='dc' else 'foc'
    if opts.controller in ('pi','foc') and opts.controller!=baseline_name:
        p.error(f'{opts.motor} baseline is {baseline_name}')
    preset=PRESETS[opts.motor]
    names={'ppo':opts.ppo_run or preset['ppo'],'icm':opts.icm_run or preset['icm']}
    selected=['ppo','icm'] if opts.controller=='all' else [opts.controller] if opts.controller in names else []
    if opts.train:
        if opts.controller not in ('ppo','icm') or not opts.run or not opts.run.strip():
            p.error('--train requires --controller ppo/icm and a NEW --run')
        args,manifest,_=restored(opts.motor,opts.controller,names[opts.controller])
        args.command='train'; args.run=opts.run; args.seed=opts.seed
        args.iterations=manifest['completed_iteration'] if opts.iterations is None else opts.iterations
        args.dry_run=opts.dry_run; args.resume=False
        return runner.main(training_command(args))
    if opts.run or opts.iterations is not None: p.error('--run/--iterations are training options; use --ppo-run/--icm-run for evaluation')
    from .current_control import environment_config
    include_baseline = opts.controller in ('all', baseline_name)
    baseline_path, baseline = baseline_settings(opts.motor, opts.baseline_config) if include_baseline else (None, None)
    loaded={c:restored(opts.motor,c,names[c]) for c in selected or (['ppo'] if opts.ppo_run else [])}
    configs={c:environment_config(v[0]) for c,v in loaded.items()}
    if loaded:
        config=next(iter(configs.values()))
        if any(c!=config for c in configs.values()): raise ValueError('PPO/ICM environments differ')
        if len({(v[0].episode_length,v[0].evaluation_reference_scale) for v in loaded.values()})!=1:
            raise ValueError('Episode lengths or reference scales differ')
        args=next(iter(loaded.values()))[0]
    else:
        args=runner.parser().parse_args(['eval','--motor',opts.motor])
        config=baseline['environment']
        args.episode_length=baseline.get('episode_length',args.episode_length)
    args.profile=opts.profile; args.seed=opts.seed; args.scenario=opts.scenario
    args.show=opts.show; args.plot_reward=opts.plot_reward; args.plot_tsne=opts.plot_tsne
    args.baseline_config=str(baseline_path) if baseline_path else None
    from .baselines import load_gains
    gains=load_gains(args,config)
    if selected and all(names[c]==preset[c] for c in selected):
        selection='Saved demonstration checkpoints; not a multi-seed result.'
    elif selected:
        selection='User-selected runs; not a multi-seed result.'
    else:
        selection='Frozen classical baseline; no model checkpoint loaded.'
    metadata=dict(motor=opts.motor,profile=opts.profile,seed=opts.seed,scenario=opts.scenario,
        purpose='Demonstration/diagnostic, not held-out qualification; no thresholds changed',
        checkpoints={c:str(loaded[c][2]) for c in selected},baseline=args.baseline_config,
        selection=selection,
        environment=config)
    if opts.dry_run:
        print(json.dumps(metadata,indent=2)); return
    import matplotlib
    if not opts.show: matplotlib.use('Agg')
    import numpy as np
    from .evaluation import episode,save_reports
    from .response import step_response_metrics
    module=runner.load_motor(args); results={}
    if selected:
        from ray.rllib.core.rl_module.rl_module import RLModule
    for controller in selected:
        policy=RLModule.from_checkpoint(str(loaded[controller][2]/'learner_group/learner/rl_module/default_policy'))
        results[controller]=episode(args,module,policy,config)
    if include_baseline:
        results[baseline_name]=episode(args,module,env_config=config,baseline_gains=gains)
    reference=next(iter(results.values()))['data'][:,1]
    for result in results.values():
        trace = result['data'][:,1]
        overlap = min(len(reference), len(trace))
        np.testing.assert_array_equal(reference[:overlap], trace[:overlap])
    output=runner.ROOT/'outputs/final_results'/opts.motor/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    runner.save_json(output/'evaluation_manifest.json',metadata)
    runner.save_json(output/'model_manifests.json',{c:loaded[c][1] for c in selected})
    if baseline is not None:
        runner.save_json(output/'frozen_baseline.json',baseline)
    runner.save_json(output/'responses.json',{c:step_response_metrics(r['data']) for c,r in results.items()})
    save_reports(args,results,output)
    lines=['# '+opts.motor.upper()+' comparison', '',f'Profile: {opts.profile}; seed: {opts.seed}; scenario: {opts.scenario}.',
           '',metadata['purpose'],metadata['selection'],'',
           '| Controller | MAE [rad/s] | RMSE [rad/s] | Current RMS [A] | Peak current [A] | Completed | Safety violation |',
           '|---|---:|---:|---:|---:|---|---|']
    for c,r in results.items():
        m=r['metrics']; lines.append(f"| {c} | {m['MAE_speed_rad_s']:.4f} | {m['RMSE_speed_rad_s']:.4f} | {m['RMS_current_A']:.4f} | {m['max_abs_current_A']:.4f} | {m['completed_episode']} | {m['safety_violation']} |")
    lines+=['','![Tracking](tracking.png)','','![Current histogram](current_histogram.png)',
            '', 'Full plateau metrics: *_ripple_by_plateau.csv; raw signals: *_timeseries.csv; step responses: responses.json.',
            'Mixed includes dynamic references: plateau metrics only characterize its constant segments, not the entire episode.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if opts.controller == 'all':
        runner.save_json(output.parent/'latest.json',dict(directory=str(output)))
    print('\n'.join(lines[:10+len(results)])); print(f'Outputs: {output}',flush=True)


if __name__=='__main__': main()
