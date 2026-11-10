"""Select current-control checkpoints on validation seeds, never the test suite.

python -m rllib_control.validate --motor pmsm --controller ppo --run NAME
Only checkpoints passing every requested case's ripple threshold are eligible.
Selection does not rename checkpoints or change the final-iteration benchmark.
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np
from . import runner


def choose(records, max_ripple, max_mae=30.):
    eligible=[r for r in records if r['complete'] and math.isfinite(r['mean_mae'])
              and r.get('quality_pass', False)
              and r['mean_mae']<=max_mae
              and math.isfinite(r['worst_ripple']) and r['worst_ripple']<=max_ripple]
    return min(eligible,key=lambda r:r['mean_mae'])['checkpoint'] if eligible else None


def quality(result, opts):
    """Use unsmoothed, late-plateau measurements; zero ripple alone is insufficient."""
    rows=result['plateaus']
    if not rows:
        return dict(passed=False, reason='No measurable reference plateaus')
    required=('speed_ripple_RMS','speed_peak_to_peak','steady_offset_rad_s',
              'i_or_id_ripple_RMS','iq_ripple_RMS')
    if (not math.isfinite(result['current_limit']) or result['current_limit']<=0
            or any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k])
                   for row in rows for k in required)):
        return dict(passed=False,reason='Missing or nonfinite plateau measurements')
    values=dict(speed_ripple=max(r['speed_ripple_RMS'] for r in rows),
                speed_peak_to_peak=max(r['speed_peak_to_peak'] for r in rows),
                absolute_offset=max(abs(r['steady_offset_rad_s']) for r in rows),
                current_ripple_fraction=max(max(r['i_or_id_ripple_RMS'],r['iq_ripple_RMS'])
                                           for r in rows)/result['current_limit'])
    limits=dict(speed_ripple=opts.max_ripple,speed_peak_to_peak=opts.max_speed_pp,
                absolute_offset=opts.max_offset,current_ripple_fraction=opts.max_current_ripple_fraction)
    metrics=result['metrics']
    passed=(metrics['completed_episode'] and not metrics['safety_violation']
            and metrics['time_above_current_scale_s']==0
            and all(math.isfinite(v) and v<=limits[k] for k,v in values.items()))
    return dict(passed=bool(passed),values=values,limits=limits)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motor',choices=['dc','pmsm'],default='pmsm')
    p.add_argument('--controller',choices=['ppo','icm'],default='ppo')
    p.add_argument('--run',required=True)
    p.add_argument('--iterations',type=int,nargs='+',default=[250,500])
    p.add_argument('--seeds',type=int,nargs='+',default=[501,502])
    p.add_argument('--profiles',nargs='+',choices=['steps','random'],default=['steps','random'])
    p.add_argument('--max-ripple',type=float,default=1.,help='Maximum plateau speed ripple RMS, rad/s')
    p.add_argument('--max-mae',type=float,default=30.,help='Maximum mean tracking MAE across validation cases, rad/s')
    p.add_argument('--max-speed-pp',type=float,default=5.,help='Maximum late-plateau speed peak-to-peak, rad/s')
    p.add_argument('--max-offset',type=float,default=10.,help='Maximum absolute late-plateau speed offset, rad/s')
    p.add_argument('--max-current-ripple-fraction',type=float,default=.05,
                   help='Maximum per-axis current ripple RMS as fraction of current scale')
    p.add_argument('--max-step-rise',type=float,default=None,
                   help='Optional largest steps-profile reversal 10-90 rise ceiling, seconds; both validation seeds')
    opts=p.parse_args(argv)
    if not all(math.isfinite(v) and v>0 for v in (opts.max_ripple,opts.max_mae,opts.max_speed_pp,
            opts.max_offset,opts.max_current_ripple_fraction)) or min(opts.iterations)<1:
        p.error('Positive finite ripple threshold and positive iterations required')
    if set(opts.seeds)&{901,902,903}:
        p.error('901/902/903 are held-out research test seeds, not validation seeds')
    if opts.max_step_rise is not None and (not math.isfinite(opts.max_step_rise) or opts.max_step_rise<=0 or 'steps' not in opts.profiles):
        p.error('--max-step-rise requires a positive finite ceiling and the steps profile')
    args=runner.parser().parse_args(['eval','--motor',opts.motor,'--controller',opts.controller,'--run',opts.run])
    folder=runner.run_dir(args)
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['settings'].get('action_mode')!='current':
        p.error('This validator supports current-reference checkpoints only')
    for key,value in manifest['settings'].items():
        if hasattr(args,key): setattr(args,key,value)
    if args.motor!=opts.motor or args.controller!=opts.controller:
        p.error('Manifest motor/controller mismatch')
    from .current_control import environment_config
    from .evaluation import episode
    from .response import step_response_metrics
    from ray.rllib.core.rl_module.rl_module import RLModule
    config=environment_config(args)
    module=runner.load_motor(args)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output=runner.ROOT/'outputs/rllib'/opts.motor/'validation'/opts.run/stamp
    output.mkdir(parents=True)
    records=[]
    for iteration in sorted(set(opts.iterations)):
        checkpoint=folder/f'iteration_{iteration:04d}'
        record=dict(checkpoint=str(checkpoint),iteration=iteration,cases=[],complete=False,
                    mean_mae=None,worst_ripple=None)
        try:
            policy=RLModule.from_checkpoint(str(checkpoint/'learner_group/learner/rl_module/default_policy'))
            for profile in opts.profiles:
                for seed in opts.seeds:
                    args.profile=profile; args.seed=seed
                    result=episode(args,module,policy,config)
                    metrics=result['metrics']
                    ripple=max((r['speed_ripple_RMS'] for r in result['plateaus']),default=float('inf'))
                    record['cases'].append(dict(profile=profile,seed=seed,metrics=metrics,
                                               max_plateau_ripple=ripple,quality=quality(result,opts),
                                               response=step_response_metrics(result['data']) if profile=='steps' else []))
            record.update(complete=all(r['metrics']['completed_episode'] and not r['metrics']['safety_violation']
                                       for r in record['cases']),
                          mean_mae=float(np.mean([r['metrics']['MAE_speed_rad_s'] for r in record['cases']])),
                          worst_ripple=max(r['max_plateau_ripple'] for r in record['cases']),
                          quality_pass=all(r['quality']['passed'] for r in record['cases']))
            if opts.max_step_rise is not None:
                largest=[max(c['response'],key=lambda r:abs(r['step_rad_s']))['rise_time_10_90_s']
                         for c in record['cases'] if c['profile']=='steps' and c['response']]
                record['response_pass']=(len(largest)==len(opts.seeds) and all(
                    math.isfinite(v) and v<=opts.max_step_rise for v in largest))
                record['largest_reversal_rise_s']=largest
                record['quality_pass']=record['quality_pass'] and record['response_pass']
        except Exception as error:
            record['error']=repr(error)
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='cases'}),flush=True)
        runner.save_json(output/'validation.json',dict(arguments=vars(opts),records=records,
                          training_settings=manifest['settings'],
                          selected_checkpoint=choose(records,opts.max_ripple,opts.max_mae)))
    selected=choose(records,opts.max_ripple,opts.max_mae)
    lines=['# Checkpoint validation','',f'Ripple ceiling: {opts.max_ripple} rad/s (every plateau).',
           f'Mean tracking MAE ceiling: {opts.max_mae} rad/s.',
           f'Peak-to-peak ceiling: {opts.max_speed_pp} rad/s; offset ceiling: {opts.max_offset} rad/s.',
           f'Per-axis current ripple RMS ceiling: {100*opts.max_current_ripple_fraction}% of current scale.',
           'Validation only; the research test set was not used.','',
           '| Iteration | MAE [rad/s] | Worst ripple RMS [rad/s] | Completed | Quality passed |',
           '|---|---:|---:|---|---|']
    for r in records: lines.append(f"| {r['iteration']} | {r['mean_mae']} | {r['worst_ripple']} | {r['complete']} | {r.get('quality_pass',False)} |")
    lines+=['',f'Selected checkpoint: {selected}' if selected else 'NO checkpoint passed. Do not claim a stable trained model.']
    if opts.max_step_rise is not None:
        lines.append(f'Largest reversal 10-90 rise ceiling: {opts.max_step_rise} s, required on every steps validation seed.')
    (output/'VALIDATION.md').write_text('\n'.join(lines)+'\n')
    print(f'Validation: {output / "VALIDATION.md"}',flush=True)


if __name__=='__main__': main()
