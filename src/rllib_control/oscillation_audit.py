"""Validation-only feedback audit; no checkpoint, physics or gate is modified."""
import argparse
import json
from datetime import datetime
import numpy as np
from . import runner

FEATURES = ['speed', 'id', 'iq', 'vd', 'vq', 'reference', 'error',
            'error_integral', 'load', 'previous_request', 'limited_current_reference',
            'inner_integral_d', 'inner_integral_q', 'limited_vd', 'limited_vq']


def dominant_frequency(values, dt):
    values = np.asarray(values)
    spectrum = np.abs(np.fft.rfft((values-values.mean())*np.hanning(len(values))))**2
    spectrum[0] = 0
    return float(np.fft.rfftfreq(len(values), dt)[int(np.argmax(spectrum))])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', default=['tdk_pure_rate002_reward20_seed345',
                                               'tdk_fullrange_seed345'])
    opts = p.parse_args(argv)
    import torch
    from ray.rllib.core import Columns
    from ray.rllib.core.rl_module.rl_module import RLModule
    from .current_control import environment_config
    from .evaluation import episode
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = runner.ROOT/'outputs/pmsm_oscillation_audit'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    records=[]
    fig, axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    for run in opts.runs:
        args=runner.parser().parse_args(['eval','--motor','pmsm','--run',run])
        folder=runner.run_dir(args)
        manifest=json.loads((folder/'manifest.json').read_text())
        for key,value in manifest['settings'].items():
            if hasattr(args,key):setattr(args,key,value)
        args.profile='steps';args.seed=501
        policy=RLModule.from_checkpoint(str(folder/'iteration_0500/learner_group/learner/rl_module/default_policy'))
        result=episode(args,runner.load_motor(args),policy,environment_config(args))
        obs=torch.tensor(result['observations'],dtype=torch.float32,requires_grad=True)
        mean=policy.forward_train({Columns.OBS:obs})[Columns.ACTION_DIST_INPUTS][:,0]
        gradient=torch.autograd.grad(mean.sum(),obs)[0].detach().numpy()
        data=result['data'];t=data[:,0]
        plateaus=[]
        for row in result['plateaus']:
            mask=(t>=row['steady_start_s'])&(t<=row['end_s'])
            g=gradient[mask]
            plateaus.append(dict(segment=row['segment'],reference=row['reference_rad_s'],
                speed_ripple_RMS=row['speed_ripple_RMS'],offset=row['steady_offset_rad_s'],
                dominant_speed_frequency_Hz=dominant_frequency(data[mask,2],float(np.median(np.diff(t)))),
                signed_mean_action_gradient=dict(zip(FEATURES,g.mean(axis=0).tolist())),
                mean_absolute_action_gradient=dict(zip(FEATURES,abs(g).mean(axis=0).tolist())),
                direct_command_memory_gain_mean=float((g[:,9]+g[:,10]).mean()),
                direct_speed_feedback_A_per_rad_s=float(((g[:,0]-g[:,6]*args.error_observation_gain)
                    *args.current_reference_limit*3./600.).mean())))
        record=dict(run=run,settings=manifest['settings'],plateaus=plateaus,
            caveat='Local partial derivatives are not closed-loop eigenvalues or a stability proof.')
        records.append(record)
        np.savetxt(output/(run+'.csv'),np.column_stack((data,result['current_references'])),delimiter=',',
            header='time,reference,speed,current_magnitude,voltage_magnitude,reward,id,iq,vd,vq,requested_current,limited_current',comments='')
        axes[0].plot(t,data[:,2],label=run,lw=1.)
        axes[1].plot(t,result['current_references'][:,0],lw=1.)
        axes[2].plot(t,data[:,7],lw=1.)
        print(json.dumps(dict(run=run,worst=max(plateaus,key=lambda r:r['speed_ripple_RMS']))),flush=True)
    axes[0].plot(t,data[:,1],'k--',label='Reference');axes[0].legend(fontsize=8)
    for ax,label in zip(axes,['Speed [rad/s]','Requested q current [A]','Measured q current [A]']):
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Pure PPO feedback audit: identical nominal plant, validation steps seed 501')
    fig.savefig(output/'feedback_comparison.png',dpi=160);plt.close(fig)
    runner.save_json(output/'audit.json',dict(validation_only=True,records=records))
    print('OUTPUT',output,flush=True)


if __name__=='__main__':main()
