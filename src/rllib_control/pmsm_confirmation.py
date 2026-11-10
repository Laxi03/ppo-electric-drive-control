"""Freeze a qualified PMSM configuration; six matched PPO/ICM runs."""
import argparse
import hashlib
import json
import csv
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import runner,research


def matched_command(settings,run,controller,seed,iterations,workers):
    command=['train']
    omitted={'command','run','controller','seed','iterations','num_env_runners',
             'intrinsic_reward_coeff','forward_loss_weight','intrinsic_final_coeff','intrinsic_decay_iterations'}
    for action in runner.parser()._actions:
        key=action.dest
        if key in omitted or key not in settings or settings[key] is None or not action.option_strings:continue
        flag=action.option_strings[0];value=settings[key]
        if isinstance(action,argparse._StoreTrueAction):
            if value:command.append(flag)
        else:
            command.extend([flag,*[str(v) for v in value]] if isinstance(value,list) else [flag,str(value)])
    command+=['--run',run,'--controller',controller,'--seed',str(seed),'--iterations',str(iterations),
              '--num-env-runners',str(workers),'--intrinsic-decay-iterations',str(iterations)]
    if controller=='icm':command+=['--intrinsic-reward-coeff','.005','--forward-loss-weight','.2']
    return command


def make_protocol(experiment,report_path,seeds,iterations=500,workers=2):
    if len(seeds)!=3 or len(set(seeds))!=3 or min(seeds)<0 or set(seeds)&{501,502,1901,1902,1903}:
        raise ValueError('Exactly three distinct training seeds, disjoint from validation/test seeds, required')
    report=json.loads(report_path.read_text());opts=report['arguments']
    required={'max_mae':30.,'max_ripple':1.,'max_speed_pp':5.,'max_offset':10.,'max_current_ripple_fraction':.05,'max_step_rise':.24}
    if (set(opts.get('seeds',[]))!={501,502} or set(opts.get('profiles',[]))!={'steps','random'}
            or any(opts.get(k) is None or not 0<float(opts[k])<=v for k,v in required.items())):
        raise ValueError('Qualification must use both profiles, seeds 501/502, and all fixed accuracy/ripple/speed gates')
    selected=report.get('selected_checkpoint')
    if not selected:raise ValueError('No qualified candidate: the six-run comparison will NOT start')
    checkpoint=Path(selected).resolve()
    if not checkpoint.is_dir():raise FileNotFoundError(checkpoint)
    if checkpoint.name!=f'iteration_{iterations:04d}':
        raise ValueError('Confirmation budget must match the qualified final checkpoint')
    manifest_path=checkpoint.parent/'manifest.json';manifest=json.loads(manifest_path.read_text());settings=manifest['settings']
    if settings.get('motor')!='pmsm' or settings.get('controller')!='ppo' or settings.get('stability_profile')!='pmsm-v1':
        raise ValueError('A pure PMSM pmsm-v1 PPO candidate is required')
    if settings!=report.get('training_settings'):raise ValueError('Qualification and candidate manifest settings differ')
    record=next((r for r in report['records'] if r['checkpoint']==selected),None)
    if not record or not all(record.get(k) for k in ('complete','quality_pass','response_pass')):
        raise ValueError('Candidate did not pass the complete quality and response qualification')
    if settings['seed'] in seeds:
        raise ValueError('Use three fresh training seeds, excluding the tuning-candidate seed')
    jobs=[]
    for controller,variant in [('ppo','ppo'),('icm','icm_fixed')]:
        for seed in seeds:
            run=f'{experiment}_{variant}_seed{seed}'
            command=matched_command(settings,run,controller,seed,iterations,workers)
            parsed=runner.parser().parse_args(command);runner.run_dir(parsed);actual=runner.settings(parsed)
            allowed={'controller','seed','intrinsic_reward_coeff','forward_loss_weight','intrinsic_final_coeff',
                     'intrinsic_decay_iterations','num_env_runners'}
            differences={k for k in settings.keys()|actual.keys() if settings.get(k)!=actual.get(k)}
            if differences-allowed:raise ValueError(f'Candidate settings not reproduced: {differences-allowed}')
            jobs.append(dict(id=f'pmsm_{variant}_{seed}',motor='pmsm',variant=variant,controller=controller,
                             train_seed=seed,run=run,command=command,settings=actual))
    baseline_command=list(jobs[0]['command']);baseline_run=experiment+'_baseline'
    baseline_command[baseline_command.index('--run')+1]=baseline_run
    jobs.append(dict(id='pmsm_baseline',motor='pmsm',variant='baseline',controller='baseline',
                     train_seed=None,run=baseline_run,environment_command=baseline_command))
    protocol=dict(version=1,study='pmsm_confirmation',iterations=iterations,episode_length=settings['episode_length'],require_frozen_baseline=True,
        test_seeds=[1901,1902,1903],profiles=['steps','random'],scenarios=['nominal'],jobs=jobs,
        source_sha256=research.fingerprints(),evaluation_scale=1.,validation_seeds=[501,502],
        qualification_report=str(report_path.resolve()),qualification_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        candidate_checkpoint=str(checkpoint),candidate_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        selection_rule='Qualified tuning seed; all six confirmation runs start from scratch; final checkpoint only')
    if iterations==700:
        lineage=manifest.get('lineage',{})
        if (lineage.get('pretraining_iterations')!=500 or lineage.get('additional_training_iterations')!=200
                or lineage.get('total_iterations')!=700):
            raise ValueError('700-iteration confirmation requires the explicit qualified 500+200 lineage')
        protocol.update(training_milestones=[100,500,700],
            training_schedule_note='Fresh seeds, identical 100 -> 500 -> 700 checkpoint/restore schedule on PPO and ICM; no tuning or early stopping at milestones')
    return protocol


def summarize(protocol,root):
    """Never drop poor completed runs; average test cases before training-seed SD."""
    from .tdk_results import GATE
    from .validate import quality
    from .response import step_response_metrics
    rows=[];traces={};references={}
    for job,profile,scenario,seed in research.cases(protocol):
        path=root/'cases'/(research.case_id(job,profile,scenario,seed)+'.json')
        stored=json.loads(path.read_text()) if path.exists() else dict(status='missing')
        row=dict(variant=job['variant'],train_seed=job['train_seed'],test_seed=seed,
                 profile=profile,status=stored['status'],engineering_pass=False)
        if stored['status']!='complete':
            row['error']=stored.get('error','Incomplete case')
            rows.append(row)
            continue
        try:
            output=Path(stored['output']);files=list(output.glob('*_timeseries.csv'))
            plateaus=list(output.glob('*_ripple_by_plateau.csv'))
            if len(files)!=1 or len(plateaus)!=1:raise ValueError('Expected exactly one controller trace/plateau file')
            data=np.loadtxt(files[0],delimiter=',',skiprows=1,usecols=range(10))
            with plateaus[0].open() as stream:
                plateau_rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
            assessment=quality(dict(metrics=stored['metrics'],plateaus=plateau_rows,current_limit=3.),SimpleNamespace(**GATE))
            response=step_response_metrics(data) if profile=='steps' else []
            largest=max(response,key=lambda r:abs(r['step_rad_s']))['rise_time_10_90_s'] if response else None
            row.update(MAE_speed_rad_s=stored['metrics']['MAE_speed_rad_s'],
                RMS_current_A=stored['metrics']['RMS_current_A'],peak_current_A=stored['metrics']['max_abs_current_A'],
                worst_offset_rad_s=max(abs(r['steady_offset_rad_s']) for r in plateau_rows),
                worst_ripple_RMS_rad_s=max(r['speed_ripple_RMS'] for r in plateau_rows),
                largest_step_rise_s=largest,engineering_pass=assessment['passed'],
                response_pass=bool(math.isfinite(largest) and largest<=.24) if largest is not None else None)
            key=(profile,scenario,seed)
            if key in references:np.testing.assert_array_equal(data[:,1],references[key])
            else:references[key]=data[:,1].copy()
            if seed==protocol['test_seeds'][0]:traces[job['variant'],job['train_seed'],profile]=data
        except (ValueError,KeyError,OSError) as error:
            row.update(status='missing' if stored['status']=='missing' else 'error',error=str(error))
        rows.append(row)
    summary=[]
    metrics=['MAE_speed_rad_s','worst_offset_rad_s','worst_ripple_RMS_rad_s','RMS_current_A','peak_current_A','largest_step_rise_s']
    for variant in ['baseline','ppo','icm_fixed']:
        for profile in protocol['profiles']:
            group=[r for r in rows if r['variant']==variant and r['profile']==profile]
            for metric in metrics:
                if profile!='steps' and metric=='largest_step_rise_s':continue
                values=[]
                for seed in sorted({r['train_seed'] for r in group},key=str):
                    local=[r for r in group if r['train_seed']==seed]
                    if all(r['status']=='complete' and isinstance(r.get(metric),(int,float)) and math.isfinite(r[metric]) for r in local):
                        values.append(float(np.mean([r[metric] for r in local])))
                summary.append(dict(variant=variant,profile=profile,metric=metric,expected_cases=len(group),
                    complete_cases=sum(r['status']=='complete' for r in group),
                    engineering_pass_cases=sum(bool(r['engineering_pass']) for r in group),
                    valid_seed_groups=len(values),mean=float(np.mean(values)) if values else None,
                    sd=float(np.std(values,ddof=1)) if len(values)>1 else None))
    fields=list(dict.fromkeys(k for r in rows for k in r))
    research.write_csv(root/'confirmation_cases.csv',rows,fields)
    research.write_csv(root/'confirmation_summary.csv',summary,list(summary[0]))
    effort=[]
    for job in protocol['jobs']:
        if job['controller']=='baseline':continue
        args=runner.parser().parse_args(job['command']);folder=runner.run_dir(args)
        path=folder/'training_history.csv';history=[]
        if path.exists():
            with path.open() as stream:history=list(csv.DictReader(stream))
        manifest=json.loads((folder/'manifest.json').read_text()) if (folder/'manifest.json').exists() else {}
        effort.append(dict(variant=job['variant'],train_seed=job['train_seed'],status=manifest.get('status','missing'),
            completed_iterations=manifest.get('completed_iteration',0),
            environment_steps=max((float(r['steps']) for r in history),default=0.),
            training_loop_wall_s=sum(float(r['iteration_wall_s']) for r in history)))
    paired=[]
    for profile in protocol['profiles']:
        for seed in sorted({r['train_seed'] for r in rows if r['variant']=='ppo'}):
            groups=[[r for r in rows if r['profile']==profile and r['train_seed']==seed and r['variant']==v]
                    for v in ['ppo','icm_fixed']]
            if all(len(g)==len(protocol['test_seeds']) and all(r['status']=='complete' and math.isfinite(r['MAE_speed_rad_s']) for r in g) for g in groups):
                a,b=[float(np.mean([r['MAE_speed_rad_s'] for r in g])) for g in groups]
                paired.append(dict(profile=profile,train_seed=seed,ppo_mae=a,icm_mae=b,icm_minus_ppo_mae=b-a))
    runner.save_json(root/'confirmation.json',dict(cases=rows,summary=summary,training_effort=effort,
        paired_differences=paired,protocol=str(root/'protocol.json')))
    lines=[f"# PMSM: matched {protocol.get('iterations','planned')}-iteration PPO / ICM confirmation",'',
        'All three fresh training seeds are retained. Test cases are averaged within each training seed before mean / sample SD.',
        'Baseline: separately tuned on validation seeds 501/502; test seeds 1901/1902/1903.',
        'The engineering case gate excludes global MAE; the separate validation mean-MAE ceiling is 30 rad/s.',
        '', '| Variant | Profile | Metric | Mean | Training-seed SD | Valid seed groups | Complete / expected | Engineering pass cases |',
        '|---|---|---|---:|---:|---:|---|---:|']
    for r in summary:
        lines.append(f"| {r['variant']} | {r['profile']} | {r['metric']} | {r['mean']} | {r['sd']} | {r['valid_seed_groups']} | {r['complete_cases']}/{r['expected_cases']} | {r['engineering_pass_cases']} |")
    lines+=['','## Training effort','',
            'Loop wall time excludes process initialization and checkpoint writing; it is not total end-to-end runtime.',
            '', '| Variant | Training seed | Status | Iterations | Environment steps | Training loop wall [s] |',
            '|---|---:|---|---:|---:|---:|']
    for r in effort:
        lines.append(f"| {r['variant']} | {r['train_seed']} | {r['status']} | {r['completed_iterations']} | {r['environment_steps']} | {r['training_loop_wall_s']:.1f} |")
    lines+=['','## Paired seed differences','',
            'Positive ICM-minus-PPO MAE means ICM was worse. Test seeds are averaged within each training seed; no significance claim is made from three training seeds.',
            '', '| Profile | Training seed | PPO MAE | ICM MAE | ICM minus PPO |','|---|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f"| {r['profile']} | {r['train_seed']} | {r['ppo_mae']:.4f} | {r['icm_mae']:.4f} | {r['icm_minus_ppo_mae']:.4f} |")
    (root/'CONFIRMATION.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'baseline':'#555555','ppo':'#0072B2','icm_fixed':'#D55E00'}
    for profile in protocol['profiles']:
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
        for variant in ['baseline','ppo','icm_fixed']:
            local=[d for (v,s,p),d in traces.items() if v==variant and p==profile]
            if not local:continue
            for d in local:
                axes[0].plot(d[:,0],d[:,2],color=colors[variant],alpha=.35,lw=.8)
            d=local[0];mean=np.mean([d[:,2] for d in local],axis=0)
            axes[0].plot(d[:,0],mean,color=colors[variant],label=variant,lw=1.5)
            axes[1].plot(d[:,0],d[:,1]-mean,color=colors[variant],lw=1.3)
        if any(p==profile for v,s,p in traces):
            axes[0].plot(d[:,0],d[:,1],'--',color='black',label='Reference')
        axes[0].set_ylabel('Speed [rad/s]');axes[1].set_ylabel('Mean tracking error [rad/s]')
        axes[1].set_xlabel('Time [s]')
        if axes[0].get_legend_handles_labels()[0]:axes[0].legend()
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle(f"PMSM {profile}; test seed {protocol['test_seeds'][0]}\nThin curves: individual training seeds; thick curves: mean, no temporal smoothing")
        fig.savefig(root/f'{profile}_tracking.png',dpi=150);plt.close(fig)
    print('CONFIRMATION REPORT',root/'CONFIRMATION.md',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['plan','freeze-baseline','train','evaluate','aggregate','all'])
    p.add_argument('--experiment',required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--qualification',type=Path,help='Exact validation.json qualification report')
    source.add_argument('--candidate-run',help='Use the LATEST validation report for this PMSM PPO run, never an older passing report')
    p.add_argument('--seeds',type=int,nargs=3,default=[123,234,456])
    p.add_argument('--iterations',type=int,default=500)
    p.add_argument('--num-env-runners',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if args.iterations not in (500,700) or args.num_env_runners<0:p.error('Use the qualified 500 or 700 iteration budget; workers must be nonnegative')
    runner.run_dir(runner.parser().parse_args(['train','--run',args.experiment]))
    root=runner.ROOT/'outputs/rllib/research'/args.experiment/'confirmation'
    try:
        if args.candidate_run:
            runner.run_dir(runner.parser().parse_args(['eval','--run',args.candidate_run]))
            reports=sorted((runner.ROOT/'outputs/rllib/pmsm/validation'/args.candidate_run).glob('*/validation.json'))
            if not reports:raise ValueError('Validate the candidate first')
            args.qualification=reports[-1]
        requested=make_protocol(args.experiment,args.qualification,args.seeds,args.iterations,args.num_env_runners)
        path=root/'protocol.json'
        if path.exists():
            protocol=json.loads(path.read_text())
            if args.stage!='aggregate' and protocol!=requested:raise ValueError('Frozen protocol/source changed; use a new experiment')
        else:
            if args.stage in ('evaluate','aggregate'):raise ValueError('Create the frozen plan first')
            root.mkdir(parents=True);protocol=requested;runner.save_json(path,protocol)
        print('FROZEN PROTOCOL',path,flush=True)
        if args.stage in ('freeze-baseline','all'):research.freeze_baselines(protocol,root)
        if args.stage in ('train','all'):research.train(protocol,root,args.resume)
        if args.stage in ('evaluate','all'):research.evaluate(protocol,root,args.resume)
        if args.stage in ('aggregate','all'):
            research.aggregate(protocol,root)
            summarize(protocol,root)
    except (ValueError,FileNotFoundError) as error:p.error(str(error))


if __name__=='__main__':main()
