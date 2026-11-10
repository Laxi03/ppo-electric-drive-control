"""Validation-only late-training response trend; no selection or test-set access."""
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
    args=runner.parser().parse_args(['eval','--motor','pmsm','--run','tdk_normalized_critic_seed345'])
    folder=runner.run_dir(args);settings=json.loads((folder/'manifest.json').read_text())['settings']
    for key,value in settings.items():
        if hasattr(args,key):setattr(args,key,value)
    args.seed=501;args.profile='steps'
    output=runner.ROOT/'outputs/pmsm_late_learning'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    records=[];traces={};reference=None
    for iteration in [350,400,450,500]:
        checkpoint=folder/f'iteration_{iteration:04d}'
        policy=RLModule.from_checkpoint(str(checkpoint/'learner_group/learner/rl_module/default_policy'))
        result=episode(args,runner.load_motor(args),policy,environment_config(args))
        data=result['data'];current=result['current_references'];traces[iteration]=result
        if reference is not None:np.testing.assert_array_equal(reference,data[:,1])
        reference=data[:,1]
        response=max(step_response_metrics(data),key=lambda r:abs(r['step_rad_s']))
        segment=(data[:,0]>=response['start_s'])&(data[:,0]<=response['end_s'])
        sample=data[segment];progress=(sample[:,2]-response['initial_speed_rad_s'])/response['step_rad_s']
        currents={}
        for level in [.1,.5,.9]:
            indices=np.flatnonzero(progress>=level)
            currents[str(level)]=float(current[segment][indices[0],0]) if len(indices) else None
        record=dict(iteration=iteration,response=response,current_request_at_progress_A=currents,
                    max_request_A=float(current[segment,0].max()),
                    worst_offset=max(abs(r['steady_offset_rad_s']) for r in result['plateaus']),
                    worst_ripple=max(r['speed_ripple_RMS'] for r in result['plateaus']),metrics=result['metrics'])
        records.append(record);print(json.dumps(record),flush=True)
        np.savetxt(output/f'iteration_{iteration}.csv',np.column_stack((data,current)),delimiter=',',
            header='time,reference,speed,current_magnitude,voltage_magnitude,reward,id,iq,vd,vq,requested_current,limited_current',comments='')
        runner.save_json(output/'diagnosis.json',dict(diagnostic_only=True,source_settings=settings,
            not_a_checkpoint_selection=True,profile='steps',validation_seed=501,records=records))
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,layout='constrained')
    start=records[-1]['response']['start_s']
    for iteration,result in traces.items():
        d=result['data'];mask=(d[:,0]>=start-.02)&(d[:,0]<=start+.5);time=d[mask,0]-start
        axes[0].plot(time,d[mask,2],label=f'{iteration} iterations')
        axes[1].plot(time,result['current_references'][mask,0])
        axes[2].plot(time,d[mask,7])
    axes[0].plot(time,d[mask,1],'k--',label='Reference');axes[0].legend(fontsize=9)
    for ax,label in zip(axes,['Speed [rad/s]','Requested q current [A]','Measured q current [A]']):
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time from largest reversal [s]')
    fig.suptitle('Late-training response trend: unchanged pure PPO and physical constraints')
    fig.savefig(output/'late_response.png',dpi=160);plt.close(fig)
    print('DIAGNOSIS',output,flush=True)


if __name__=='__main__':main()
