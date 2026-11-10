"""Paired blue-PPO / separate default PI diagnostic, not a final tuned benchmark."""
import json
from datetime import datetime
import numpy as np
from . import runner
from .current_control import environment_config
from .evaluation import episode
from .response import step_response_metrics


def main():
    from ray.rllib.core.rl_module.rl_module import RLModule
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run='tdk_pure_reward20_16x8_seed345'
    args=runner.parser().parse_args(['eval','--motor','pmsm','--controller','ppo','--run',run])
    folder=runner.run_dir(args)
    manifest=json.loads((folder/'manifest.json').read_text())
    for key,value in manifest['settings'].items():
        if hasattr(args,key):setattr(args,key,value)
    args.seed=501;args.profile='steps';args.baseline_config=None
    config=environment_config(args);module=runner.load_motor(args)
    checkpoint=folder/'iteration_0500'
    policy=RLModule.from_checkpoint(str(checkpoint/'learner_group/learner/rl_module/default_policy'))
    output=runner.ROOT/'outputs/tdk_results_20261016/response_diagnosis'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    results={};summary={}
    for label,model in [('PPO (blue)',policy),('Separate PI/FOC (default gains)',None)]:
        result=episode(args,module,model,config)
        results[label]=result
        responses=step_response_metrics(result['data'])
        finite=[r['rise_time_10_90_s'] for r in responses if np.isfinite(r['rise_time_10_90_s'])]
        summary[label]=dict(metrics=result['metrics'],responses=responses,
            mean_rise_s=float(np.mean(finite)) if finite else None,
            unreached=len(responses)-len(finite),
            largest_step_rise_s=max(responses,key=lambda r:abs(r['step_rad_s']))['rise_time_10_90_s'])
        refs=result['current_references'];data=result['data']
        np.savetxt(output/('ppo.csv' if model is not None else 'pi.csv'),
            np.column_stack((data,refs)),delimiter=',',
            header='time,reference,speed,current_magnitude,voltage_magnitude,reward,id,iq,vd,vq,requested_current,limited_current',comments='')
    np.testing.assert_array_equal(results['PPO (blue)']['data'][:,1],results['Separate PI/FOC (default gains)']['data'][:,1])
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for (label,result),color in zip(results.items(),['#0072B2','#555555']):
        d=result['data'];t=d[:,0]
        axes[0].plot(t,d[:,2],label=label,color=color,lw=1.2)
        axes[1].plot(t,result['current_references'][:,0],color=color,lw=1.2)
        axes[2].plot(t,d[:,7],color=color,lw=1.2)
    axes[0].plot(t,d[:,1],'--',color='black',label='Reference',lw=1.2)
    for ax,label in zip(axes,['Speed [rad/s]','Requested q current [A]','Measured q current [A]']):
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=9);axes[-1].set_xlabel('Time [s]')
    fig.suptitle('PMSM diagnostic: identical plant and actuator constraints\nSeparate default PI, not a PI added to PPO; raw signals')
    fig.savefig(output/'tracking_and_current.png',dpi=150);plt.close(fig)
    runner.save_json(output/'diagnosis.json',dict(diagnostic_only=True,checkpoint=str(checkpoint),
        config=config,seed=501,profile='steps',baseline_gains=dict(kp=.03,ki=.1),results=summary))
    print(json.dumps(summary,indent=2),flush=True)
    print('OUTPUT',output,flush=True)


if __name__=='__main__':main()
