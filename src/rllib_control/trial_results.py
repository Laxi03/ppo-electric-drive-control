"""Paired nominal validation and plots for pure PPO reward/network experiments."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import runner
from .tdk_results import GATE

TRIALS=[('stability_pilot100_seed345','16:8, absolute weight 1'),
        ('tdk_pure_reward20_16x8_seed345','16:8, absolute weight 20'),
        ('tdk_pure_base_32x16_seed345','32:16, absolute weight 1')]
EXTRA_LABELS={'tdk_pure_conditioned_reward20_16x8_seed345':'Reward 20 + critic conditioning',
              'tdk_pure_abs100_rate002_seed345':'Absolute error 100, rate penalty 0.002',
              'tdk_pure_rate002_reward20_seed345':'Reward 20, current-rate penalty 0.002',
              'tdk_pure_fast_conditioned_reward20_16x8_seed345':'Reward 20 + conditioning + quadratic 300',
              'tdk_pure_scale001_error1_seed345':'Reward 20, scale 0.001, error gain 1',
              'tdk_pure_scale001_error10_seed345':'Reward 20, scale 0.001, error gain 10'}
COLORS={'stability_pilot100_seed345':'#777777','tdk_pure_reward20_16x8_seed345':'#0072B2',
        'tdk_pure_abs100_rate002_seed345':'#009E73',
        'tdk_pure_rate002_reward20_seed345':'#D55E00',
        'tdk_pure_base_32x16_seed345':'#D55E00','tdk_pure_conditioned_reward20_16x8_seed345':'#CC79A7',
        'tdk_pure_fast_conditioned_reward20_16x8_seed345':'#E69F00',
        'tdk_pure_scale001_error1_seed345':'#777777','tdk_pure_scale001_error10_seed345':'#009E73'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iterations',type=int,default=500)
    p.add_argument('--runs',nargs='+',help='Explicit pure PPO run folders to compare (up to 3)')
    args=p.parse_args(argv)
    labels=dict(TRIALS);labels.update(EXTRA_LABELS)
    if args.runs and (len(args.runs)>3 or len(set(args.runs))!=len(args.runs)):
        p.error('Choose 1 to 3 distinct run names')
    trials=[(run,labels.get(run,run)) for run in args.runs] if args.runs else TRIALS
    from .evaluation import episode
    from .current_control import environment_config
    from .validate import quality
    from .response import step_response_metrics
    from ray.rllib.core.rl_module.rl_module import RLModule
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42})
    output=runner.ROOT/'outputs/tdk_results_20261016/pure_trials'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    traces={};records=[];references={}
    physical_settings=None
    for trial_index,(run,label) in enumerate(trials):
        a=runner.parser().parse_args(['eval','--motor','pmsm','--controller','ppo','--run',run])
        folder=runner.run_dir(a)
        manifest=json.loads((folder/'manifest.json').read_text())
        if run not in labels and manifest['settings'].get('actor_observations')=='speed':
            s=manifest['settings']
            label=f"Speed actor; log-std [{s.get('policy_log_std_min',-4):g}, {s.get('policy_log_std_max',-1):g}]"
            trials[trial_index]=(run,label)
        for key,value in manifest['settings'].items():
            if hasattr(a,key):setattr(a,key,value)
        assert a.action_mode=='current', 'Only pure current-reference PPO allowed'
        checkpoint=folder/f'iteration_{args.iterations:04d}'
        policy=RLModule.from_checkpoint(str(checkpoint/'learner_group/learner/rl_module/default_policy'))
        module=runner.load_motor(a);config=environment_config(a);cases=[]
        physical={k:config.get(k) for k in ('architecture','current_reference_limit','current_reference_slew',
                  'current_bandwidth','command_slew_v_per_s')}
        physical.update(episode_length=a.episode_length,evaluation_reference_scale=a.evaluation_reference_scale)
        if physical_settings is None:physical_settings=physical
        elif physical!=physical_settings:raise ValueError('Trials use different physical conditions')
        for profile in ['steps','random']:
            for seed in [501,502]:
                a.profile=profile;a.seed=seed
                result=episode(a,module,policy,config)
                key=(profile,seed)
                reference=result['data'][:,1]
                if key in references:np.testing.assert_array_equal(reference,references[key])
                else:references[key]=reference.copy()
                assessment=quality(result,SimpleNamespace(**GATE))
                cases.append(dict(profile=profile,seed=seed,metrics=result['metrics'],quality=assessment,
                                  plateaus=result['plateaus'],response=step_response_metrics(result['data'])))
                name=f'{run}_{profile}_seed{seed}'
                np.savetxt(output/(name+'.csv'),result['data'],delimiter=',',
                    header='time,reference,speed,current_magnitude,voltage_magnitude,reward,id,iq,vd,vq',comments='')
                if seed==501:traces[run,profile]=result
        mean_mae=float(np.mean([c['metrics']['MAE_speed_rad_s'] for c in cases]))
        finite_values=[c['quality'].get('values',{}) for c in cases]
        record=dict(run=run,label=label,checkpoint=str(checkpoint),settings=manifest['settings'],
            training_seed=a.seed,mean_mae=mean_mae,
            worst_ripple=max(c.get('speed_ripple',float('inf')) for c in finite_values),
            worst_offset=max(c.get('absolute_offset',float('inf')) for c in finite_values),
            worst_current_ripple_fraction=max(c.get('current_ripple_fraction',float('inf')) for c in finite_values),
            passed=all(c['quality']['passed'] for c in cases) and mean_mae<=GATE['max_mae'],cases=cases)
        record['training_seed']=manifest['settings']['seed']
        responses=[r for c in cases if c['profile']=='steps' and c['seed']==501 for r in c['response']]
        rise=[r['rise_time_10_90_s'] for r in responses if np.isfinite(r['rise_time_10_90_s'])]
        record['mean_step_rise_time_s']=float(np.mean(rise)) if rise else None
        record['steps_not_reaching_90_percent']=len(responses)-len(rise)
        record['largest_step_rise_time_s']=max(responses,key=lambda r:abs(r['step_rad_s']))['rise_time_10_90_s'] if responses else None
        record['worst_step_overshoot_percent']=max((r['overshoot_percent'] for r in responses),default=None)
        record['peak_current_A']=max(c['metrics']['max_abs_current_A'] for c in cases)
        records.append(record)
        print(json.dumps({k:record[k] for k in ['run','mean_mae','worst_ripple','worst_offset','passed']}),flush=True)
        runner.save_json(output/'results.json',dict(criteria=GATE,protocol='Nominal validation, steps/random, seeds 501/502; pure PPO only',records=records))
    colors=[COLORS.get(run,['#0072B2','#D55E00','#009E73'][i]) for i,(run,label) in enumerate(trials)]
    for profile in ['steps','random']:
        fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
        for (run,label),color in zip(trials,colors):
            result=traces[run,profile];d=result['data'];t=d[:,0]
            axes[0].plot(t,d[:,2],label=label,color=color,lw=1.25)
            axes[1].plot(t,d[:,1]-d[:,2],color=color,lw=1.1)
            axes[2].plot(t,d[:,7],color=color,lw=1.1)
        axes[0].plot(t,d[:,1],'--',color='black',lw=1.4,label='Reference')
        for ax,name in zip(axes,['Speed [rad/s]','Tracking error [rad/s]','q-axis current [A]']):ax.set_ylabel(name)
        axes[-1].set_xlabel('Time [s]');axes[0].legend(ncol=2,fontsize=10)
        fig.suptitle(f'Pure PMSM PPO, {args.iterations} iterations; {profile}, validation seed 501\nRaw sampled signals, no smoothing')
        fig.savefig(output/f'{profile}_comparison.png',dpi=160);plt.close(fig)
    reference=references['steps',501]
    step_index=int(np.argmax(abs(np.diff(reference))))+1
    first=traces[trials[0][0],'steps']['data']
    step_time=float(first[step_index,0])
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,layout='constrained')
    for (run,label),color in zip(trials,colors):
        d=traces[run,'steps']['data']
        mask=(d[:,0]>=step_time-.05)&(d[:,0]<=step_time+.6)
        axes[0].plot(d[mask,0]-step_time,d[mask,2],label=label,color=color,lw=1.4)
        axes[1].plot(d[mask,0]-step_time,d[mask,7],color=color,lw=1.4)
    axes[0].plot(d[mask,0]-step_time,d[mask,1],'--',color='black',label='Reference')
    axes[0].set_ylabel('Speed [rad/s]');axes[0].legend(fontsize=9)
    axes[1].set_ylabel('q-axis current [A]');axes[1].set_xlabel('Time from largest reference step [s]')
    fig.suptitle('Largest reversal: raw response and acceleration current')
    fig.savefig(output/'largest_step_zoom.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,len(trials),figsize=(4*len(trials),4),layout='constrained',squeeze=False)
    for ax,(run,label),color in zip(axes.flat,trials,colors):
        result=traces[run,'steps'];d=result['data']
        row=max(result['plateaus'],key=lambda r:abs(r['steady_offset_rad_s']))
        mask=(d[:,0]>=row['steady_start_s'])&(d[:,0]<=row['end_s'])
        ax.plot(d[mask,0],d[mask,1]-d[mask,2],color=color)
        ax.axhline(0,color='black',lw=.8)
        ax.axhspan(-10,10,color='#009E73',alpha=.08)
        ax.set(title=label+'\n'+f"Offset {row['steady_offset_rad_s']:.2f} rad/s",xlabel='Time [s]',ylabel='Late-window error [rad/s]')
    fig.suptitle('Worst plateau offset per trial; green band: +/-10 rad/s')
    fig.savefig(output/'offset_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key,title,limit in zip(axes,['mean_mae','worst_offset','worst_ripple'],
                                 ['Mean MAE [rad/s]','Worst absolute offset [rad/s]','Worst ripple RMS [rad/s]'],[30,10,1]):
        values=[r[key] for r in records]
        ax.bar(range(len(trials)),values,color=colors[:len(trials)])
        ax.axhline(limit,color='#ba3434',ls='--',lw=1.,label=f'Limit {limit:g}')
        tick_labels=[]
        fields=[('policy_hidden_sizes','net',[16,8]),('tracking_absolute_weight','w1',1.),
                ('tracking_weight','w2',100.),('current_rate_weight','rate',0.),
                ('reward_scale','scale',1.),('error_observation_gain','e-gain',1.)]
        fields.extend([('actor_observations','actor','full'),('policy_log_std_min','logstd min',-4.),
                       ('policy_log_std_max','logstd max',-1.)])
        varying=[(key,name,default) for key,name,default in fields if
                 len({json.dumps(r['settings'].get(key,default)) for r in records})>1]
        for i,r in enumerate(records):
            parts=[f'Trial {i+1}']
            for key,name,default in varying:
                value=r['settings'].get(key,default)
                value=':'.join(str(v) for v in value) if isinstance(value,list) else value if isinstance(value,str) else f'{value:g}'
                parts.append(f'{name}={value}')
            tick_labels.append('\n'.join(parts))
        ax.set_xticks(range(len(trials)),tick_labels);ax.set_title(title);ax.legend(fontsize=9)
    fig.suptitle(f'Pure PPO validation: same motor, seed and {args.iterations}-iteration budget')
    fig.savefig(output/'metrics_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key,title,multiplier in zip(axes,
            ['largest_step_rise_time_s','worst_step_overshoot_percent','peak_current_A'],
            ['Largest-step rise time [ms]','Worst step overshoot [%]','Peak current [A]'],[1000.,1.,1.]):
        values=[r[key]*multiplier if r[key] is not None else float('nan') for r in records]
        ax.bar(range(len(trials)),values,color=colors)
        ax.set_xticks(range(len(trials)),[f'Trial {i+1}' for i in range(len(trials))]);ax.set_title(title)
        for i,value in enumerate(values):
            if np.isfinite(value):ax.text(i,value,f'{value:.2f}',ha='center',va='bottom',fontsize=10)
    fig.suptitle('Response speed and physical demand; trial order matches the tracking figure')
    fig.savefig(output/'response_comparison.png',dpi=160);plt.close(fig)
    lines=['# Pure PPO: reward and network results','',
           '| Network / reward | Mean MAE [rad/s] | Worst offset [rad/s] | Worst ripple RMS [rad/s] | Mean rise [s] | Largest-step rise [s] | Steps not reaching 90% | Pass |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in records:lines.append(f"| {r['label']} | {r['mean_mae']:.3f} | {r['worst_offset']:.3f} | {r['worst_ripple']:.3f} | {r['mean_step_rise_time_s']} | {r['largest_step_rise_time_s']} | {r['steps_not_reaching_90_percent']} | {r['passed']} |")
    seeds=sorted({r['training_seed'] for r in records})
    lines+=['',f'Validation only, training seeds: {seeds}. Every failed case is retained.']
    if any(run=='stability_pilot100_seed345' for run,label in trials):
        lines.append('The stability_pilot100 reference was resumed from iteration 100; the newer trials started from scratch.')
    lines.append('This is an exploratory comparison, not proof of a causal or statistically significant improvement.')
    lines.append('Response times are from steps / validation seed 501. Mean rise time excludes unreached responses; the failure count is reported explicitly. Accuracy and engineering gates cover all four validation cases.')
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('RESULTS',output,flush=True)


if __name__=='__main__':main()
