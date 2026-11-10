"""Frozen-baseline robustness and single-factor PMSM ablations after confirmation."""
import argparse
import csv
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import runner,research
from .pmsm_confirmation import matched_command

TEST_SEEDS=[2901,2902,2903]
EXTRA_METRICS=['worst_absolute_offset_rad_s','worst_speed_ripple_RMS_rad_s',
               'worst_current_ripple_fraction','engineering_pass','largest_step_rise_s',
               'max_abs_current_A','applied_rms_slew_V_per_s','requested_current_rms_slew_A_per_s',
               'MAE_degradation_vs_nominal_rad_s']


def make_protocol(parent_path,study,experiment):
    parent=json.loads(parent_path.read_text())
    if parent.get('study')!='pmsm_confirmation' or not parent.get('require_frozen_baseline'):
        raise ValueError('A qualified, frozen PMSM confirmation protocol is required')
    if parent['source_sha256']!=research.fingerprints():
        raise ValueError('Source differs from parent protocol; do not silently mix experiments')
    index=parent_path.parent/'pmsm_baseline.json'
    locked=json.loads(index.read_text());baseline=Path(locked['path'])
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=locked['sha256']:
        raise ValueError('Parent frozen baseline changed')
    jobs=[]
    if study=='robustness':
        jobs=copy.deepcopy(parent['jobs'])
        for job in jobs:
            if job['controller']!='baseline':job['reuse_checkpoint']=True
        contrast='PPO / ICM / validation-frozen PI/FOC under identical held-out disturbances; no retraining'
    else:
        reference=[j for j in parent['jobs'] if j['controller']=='ppo']
        if len(reference)!=3:raise ValueError('Three matched parent PPO training seeds are required')
        for original in reference:
            full=copy.deepcopy(original);full.update(variant='full',id=f"pmsm_full_{original['train_seed']}",reuse_checkpoint=True)
            jobs.append(full)
            settings=copy.deepcopy(original['settings']);seed=original['train_seed']
            if study=='environment':
                if settings['no_domain_randomization']:raise ValueError('Parent already disables reset randomization')
                settings.update(no_domain_randomization=True,paired_domain_rng=True)
                variant='reset_randomization_off';allowed={'no_domain_randomization','paired_domain_rng'}
            elif study=='reward':
                if settings.get('current_rate_weight',0.)<=0:raise ValueError('Parent has no rate penalty to ablate')
                settings['current_rate_weight']=0.;variant='rate_penalty_off';allowed={'current_rate_weight'}
            else:raise ValueError('Unknown study')
            run=f'{experiment}_{variant}_seed{seed}'
            command=matched_command(settings,run,'ppo',seed,parent['iterations'],settings['num_env_runners'])
            actual=runner.settings(runner.parser().parse_args(command))
            differences={k for k in actual.keys()|original['settings'].keys() if actual.get(k)!=original['settings'].get(k)}
            if differences!=allowed:raise ValueError(f'Ablation changes unexpected settings: {differences}')
            jobs.append(dict(id=f'pmsm_{variant}_{seed}',variant=variant,motor='pmsm',controller='ppo',
                             train_seed=seed,run=run,command=command,settings=actual))
        jobs.append(copy.deepcopy(next(j for j in parent['jobs'] if j['controller']=='baseline')))
        contrast=('Only reset R/J/load randomization removed; reference and training-jump RNG draws retained'
                  if study=='environment' else
                  'Only timescale-normalized current-reference rate penalty removed; ordinary command-change penalty retained')
    for job in jobs:
        if job['controller']=='baseline':job.update(baseline_config=str(baseline.resolve()),baseline_sha256=locked['sha256'])
    protocol=dict(version=1,study='pmsm_'+study,iterations=parent['iterations'],episode_length=parent['episode_length'],
        jobs=jobs,profiles=['steps','random'],scenarios=list(research.PRESETS),test_seeds=TEST_SEEDS,
        validation_seeds=[501,502],evaluation_scale=1.,require_frozen_baseline=True,extended_metrics=True,
        metrics=[*research.METRICS,*EXTRA_METRICS],source_sha256=research.fingerprints(),
        parent_protocol=str(parent_path.resolve()),parent_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        contrast=contrast,selection_rule='Final iteration only, all training seeds and failed cases retained; no test-based tuning')
    if parent.get('training_milestones'):protocol['training_milestones']=parent['training_milestones']
    return protocol


def enrich_metrics(output,metrics,motor,profile):
    from .validate import quality
    from .tdk_results import GATE
    from .response import step_response_metrics
    files=list(output.glob('*_timeseries.csv'));plateaus=list(output.glob('*_ripple_by_plateau.csv'))
    if len(files)!=1 or len(plateaus)!=1:raise ValueError('Missing unambiguous trace / plateau measurements')
    data=np.loadtxt(files[0],delimiter=',',skiprows=1,usecols=range(10))
    with plateaus[0].open() as stream:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    assessed=quality(dict(metrics=metrics,plateaus=rows,current_limit=3. if motor=='pmsm' else 6.),SimpleNamespace(**GATE))
    values=assessed.get('values',{})
    result=dict(metrics,worst_absolute_offset_rad_s=values.get('absolute_offset',float('inf')),
        worst_speed_ripple_RMS_rad_s=values.get('speed_ripple',float('inf')),
        worst_current_ripple_fraction=values.get('current_ripple_fraction',float('inf')),
        engineering_pass=float(assessed['passed']))
    response=step_response_metrics(data) if profile=='steps' else []
    result['largest_step_rise_s']=max(response,key=lambda r:abs(r['step_rad_s']))['rise_time_10_90_s'] if response else None
    return result,hashlib.sha256(data[:,1].tobytes()).hexdigest()


def verify_pairing(protocol,root):
    seen={}
    for job,profile,scenario,seed in research.cases(protocol):
        path=root/'cases'/(research.case_id(job,profile,scenario,seed)+'.json')
        if not path.exists():continue
        record=json.loads(path.read_text())
        if record['status']!='complete':continue
        value=(record.get('paired_reference_sha256'),record.get('scenario_metadata',{}).get('noise_sha256'))
        if None in value:raise ValueError('Missing paired-reference/noise provenance')
        key=profile,scenario,seed
        if key in seen and seen[key]!=value:raise ValueError(f'Non-identical reference/noise streams: {key}')
        seen[key]=value


def report(protocol,root):
    verify_pairing(protocol,root)
    summary=research.aggregate(protocol,root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants=list(dict.fromkeys(j['variant'] for j in protocol['jobs']))
    scenarios=protocol['scenarios'];colors=['#0072B2','#D55E00','#555555']
    for profile in protocol['profiles']:
        fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
        for ax,metric,title in zip(axes,['MAE_speed_rad_s','worst_absolute_offset_rad_s','engineering_pass'],
                ['Speed MAE [rad/s]','Mean worst-plateau offset [rad/s]','Cases meeting engineering gate [fraction]']):
            for v,color in zip(variants,colors):
                rows=[next(r for r in summary if r['variant']==v and r['profile']==profile and r['scenario']==s and r['metric']==metric) for s in scenarios]
                values=[r['mean'] if r['mean'] is not None else np.nan for r in rows]
                ax.plot(range(len(scenarios)),values,'o-',color=color,label=v,lw=1.3)
            ax.set_xticks(range(len(scenarios)),scenarios,rotation=40,ha='right');ax.set_title(title);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{protocol['study']}: {profile}; averages over training seeds (not best-seed results)")
        fig.savefig(root/f'{profile}_overview.png',dpi=160);plt.close(fig)
    lines=['# Robustness / ablation report','',protocol['contrast'],'',
        'PI/FOC gains are inherited unchanged from validation-only parent tuning. No test-scenario retuning.',
        f"Held-out seeds: {protocol['test_seeds']}. Expected cases: {sum(1 for _ in research.cases(protocol))}.",
        'Steps are deterministic in this environment: different test seeds do not provide three independent step-reference realizations.',
        'Compare physical metrics across reward variants, not differently defined reward returns.',
        'engineering_pass covers plateau offset/ripple, sampled current scale and episode completion; it does not include the validation mean-MAE or 240 ms speed gates.',
        'Plateau gate is descriptive under disturbances: a late disturbance recovery can increase ripple without sustained oscillation.',
        'See SUMMARY.md / summary.csv for seed-group means and sample SD, and cases.csv for every missing or failed case.']
    (root/'ROBUSTNESS_ABLATION.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['plan','train','evaluate','aggregate','all'])
    p.add_argument('--study',choices=['robustness','environment','reward'],required=True)
    p.add_argument('--experiment',required=True)
    p.add_argument('--parent-protocol',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    runner.run_dir(runner.parser().parse_args(['train','--run',args.experiment]))
    root=runner.ROOT/'outputs/rllib/research'/args.experiment/args.study
    try:
        requested=make_protocol(args.parent_protocol,args.study,args.experiment)
        path=root/'protocol.json'
        if path.exists():
            protocol=json.loads(path.read_text())
            if args.stage!='aggregate' and protocol!=requested:raise ValueError('Frozen protocol or source changed')
        else:
            if args.stage not in ('plan','train','all'):raise ValueError('Freeze the plan first')
            root.mkdir(parents=True);protocol=requested;runner.save_json(path,protocol)
        research.freeze_baselines(protocol,root)
        print('FROZEN PLAN',path,'cases',sum(1 for _ in research.cases(protocol)),flush=True)
        if args.stage in ('train','all'):research.train(protocol,root,args.resume)
        if args.stage in ('evaluate','all'):research.evaluate(protocol,root,args.resume)
        if args.stage in ('aggregate','all'):report(protocol,root)
    except (ValueError,FileNotFoundError) as error:p.error(str(error))


if __name__=='__main__':main()
