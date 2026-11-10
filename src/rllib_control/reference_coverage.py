"""Reference-generator coverage only; not a trained-controller benchmark."""
from datetime import datetime
import numpy as np
from . import runner


def main():
    from sim_PMSM_RLlib_Curiosity import PMSMSpeedReference
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=runner.ROOT/'outputs/tdk_results_20261016/reference_coverage'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True);records={};arrays={}
    for profile in ['random','random_full']:
        values=[];jumps=[]
        for seed in range(6000,6500):
            r=PMSMSpeedReference(profile,8.,np.random.default_rng(seed))
            values.extend(r.values[:-1]);jumps.extend(abs(np.diff(r.values[:-1])))
        a=np.asarray(values);d=np.asarray(jumps);arrays[profile]=(a,d)
        records[profile]=dict(plateaus=len(a),jumps=len(d),fraction_abs_reference_gt300=float(np.mean(abs(a)>300)),
            fraction_jumps_gt500=float(np.mean(d>500)),max_jump=float(d.max()))
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for profile,color in [('random','#0072B2'),('random_full','#D55E00')]:
        a,d=arrays[profile]
        axes[0].hist(a,bins=np.linspace(-400,400,25),density=True,histtype='step',lw=2,color=color,label=profile)
        axes[1].hist(d,bins=np.linspace(0,800,25),density=True,histtype='step',lw=2,color=color,label=profile)
    axes[0].set(xlabel='Reference plateau [rad/s]',ylabel='Probability density')
    axes[1].set(xlabel='Absolute reference jump [rad/s]',ylabel='Probability density')
    axes[1].axvline(650,color='black',ls='--',label='Largest validation reversal')
    for ax in axes:ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.suptitle('Training-reference coverage: 500 generated episodes\nNo motor rollout and no validation/held-out seeds used')
    fig.savefig(output/'coverage.png',dpi=160);plt.close(fig)
    runner.save_json(output/'coverage.json',dict(generator_only=True,seeds=list(range(6000,6500)),records=records))
    print('COVERAGE',output,records,flush=True)


if __name__=='__main__':main()
