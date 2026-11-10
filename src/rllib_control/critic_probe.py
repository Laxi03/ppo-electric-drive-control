"""Frozen-actor critic diagnosis on fresh training/probe seeds, not TDK test data.

Fixed discounted Monte Carlo returns diagnose value representability. This is
not an on-policy PPO performance test or a controller imitation experiment.
"""
import argparse
import json
from datetime import datetime
import numpy as np
import torch
from torch import nn
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from . import runner
from .current_control import CurrentReferenceEnv, environment_config


def explained(target,predicted):
    return float(1.-np.var(target-predicted)/max(np.var(target),1e-12))


def collect(policy,args,seeds):
    arrays=[]
    gamma=args.gamma if hasattr(args,'gamma') else .996
    for seed in seeds:
        config=environment_config(args)
        config.update(seed=seed,profile=args.train_profile,episode_length=args.episode_length,
                      training_mode=True,domain_randomization=not args.no_domain_randomization,
                      parameter_jump=not args.no_parameter_jump,reference_scale=args.train_reference_scale)
        env=CurrentReferenceEnv(config);obs,_=env.reset(seed=seed)
        observations=[];rewards=[];values=[];terminated=False
        with torch.no_grad():
            while True:
                batch={Columns.OBS:torch.tensor(obs,dtype=torch.float32)[None]}
                logits=policy.forward_exploration(batch)[Columns.ACTION_DIST_INPUTS]
                action=policy.get_exploration_action_dist_cls().from_logits(logits).sample()[0].numpy()
                observations.append(obs);values.append(float(policy.compute_values(batch)[0]))
                obs,r,terminated,truncated,_=env.step(action)
                rewards.append(r)
                if terminated or truncated:break
            bootstrap=0. if terminated else float(policy.compute_values(
                {Columns.OBS:torch.tensor(obs,dtype=torch.float32)[None]})[0])
        returns=np.empty(len(rewards));gae=np.empty(len(rewards));acc=bootstrap;adv=0.
        for t in range(len(rewards)-1,-1,-1):
            acc=rewards[t]+gamma*acc;returns[t]=acc
            next_value=bootstrap if t==len(rewards)-1 else values[t+1]
            delta=rewards[t]+gamma*next_value-values[t]
            adv=delta+gamma*.98*adv;gae[t]=adv+values[t]
        arrays.append((np.array(observations),returns,np.array(values),gae))
        print('COLLECT',seed,'steps',len(rewards),'terminated',terminated,flush=True)
        env.close()
    return [np.concatenate([a[i] for a in arrays]) for i in range(4)]


def fit(network,train,test,scale,steps,target_kind):
    x,y,_,gae=train;xt,yt,_,gaet=test
    if target_kind=='gae':y,yt=gae,gaet
    mean=float(y.mean());std=float(y.std())
    # Normalization is estimated from training trajectories only.
    divisor=std if scale=='standardized' else 1.
    center=mean if scale=='standardized' else 0.
    x=torch.tensor(x,dtype=torch.float32);target=torch.tensor((y-center)/divisor,dtype=torch.float32)
    xt=torch.tensor(xt,dtype=torch.float32)
    optimizer=torch.optim.Adam(network.parameters(),lr=1e-3)
    history=[]
    for step in range(steps+1):
        if step%100==0 or step==steps:
            with torch.no_grad():pred=network(xt).squeeze(-1).numpy()*divisor+center
            with torch.no_grad():train_pred=network(x).squeeze(-1).numpy()*divisor+center
            history.append(dict(step=step,test_explained_variance=explained(yt,pred),
                train_explained_variance=explained(y,train_pred),
                test_RMSE=float(np.sqrt(np.mean((yt-pred)**2))),prediction_std=float(pred.std())))
        if step==steps:break
        indices=torch.randint(len(x),(256,))
        loss=(network(x[indices]).squeeze(-1)-target[indices]).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),10.)
        optimizer.step()
    return dict(center=center,scale=divisor,history=history)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='tdk_pure_rate002_reward20_seed345')
    p.add_argument('--fit-steps',type=int,default=2000)
    p.add_argument('--target',choices=['gae','mc'],default='gae')
    p.add_argument('--train-episodes',type=int,default=16)
    p.add_argument('--probe-episodes',type=int,default=8)
    opts=p.parse_args(argv)
    torch.set_num_threads(1);torch.manual_seed(6345);np.random.seed(6345)
    a=runner.parser().parse_args(['eval','--motor','pmsm','--run',opts.run])
    folder=runner.run_dir(a);manifest=json.loads((folder/'manifest.json').read_text())
    for key,value in manifest['settings'].items():setattr(a,key,value)
    a.gamma=manifest['settings'].get('gamma',.996)
    policy=RLModule.from_checkpoint(str(folder/'iteration_0500/learner_group/learner/rl_module/default_policy'))
    before={k:v.clone() for k,v in policy.state_dict().items()}
    if not 1<=opts.train_episodes<=50 or not 1<=opts.probe_episodes<=50:
        p.error('Choose 1 to 50 episodes per disjoint seed stream')
    train_seeds=list(range(6000,6000+opts.train_episodes));test_seeds=list(range(6100,6100+opts.probe_episodes))
    train=collect(policy,a,train_seeds)
    test=collect(policy,a,test_seeds)
    output=runner.ROOT/'outputs/pmsm_critic_probe'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    x,y,pred,gae=train;xt,yt,pt,gt=test
    original=dict(train_value_std=float(pred.std()),train_target_std=float(y.std()),
        train_value_mean=float(pred.mean()),train_target_mean=float(y.mean()),
        test_explained_variance_MC=explained(yt,pt),test_explained_variance_GAE=explained(gt,pt),
        GAE_fraction_beyond_old_squared_loss_cap=float(np.mean((gae-pred)**2>10)),
        train_steps=len(x),test_steps=len(xt),train_target_abs_p95=float(np.percentile(abs(y),95)))
    records={}
    for dims,activation in [([16,8],nn.Tanh),([64,64],nn.SiLU)]:
        for scaling in ['raw','standardized']:
            torch.manual_seed(6345)
            layers=[];size=15
            for dim in dims:layers.extend([nn.Linear(size,dim),activation()]);size=dim
            layers.append(nn.Linear(size,1));net=nn.Sequential(*layers)
            name=f'{dims}_{activation.__name__}_{scaling}'
            records[name]=fit(net,train,test,scaling,opts.fit_steps,opts.target)
            print('FIT',name,records[name]['history'][-1],flush=True)
    assert all(torch.equal(v,policy.state_dict()[k]) for k,v in before.items()),'Frozen actor/critic was mutated'
    result=dict(diagnostic_only=True,source_run=opts.run,original=original,fits=records,
        training_seeds=train_seeds,probe_seeds=test_seeds,actor_changed=False,target=opts.target,
        caveat='Fixed Monte Carlo targets of a frozen stochastic actor; successful fitting is necessary evidence, not proof of PPO control performance.')
    runner.save_json(output/'probe.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for name,r in records.items():
        ax.plot([h['step'] for h in r['history']],[h['test_explained_variance'] for h in r['history']],label=name)
    ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Critic-only gradient updates',ylabel='Held-out probe explained variance',
        title=f'Frozen pure PPO actor: fitting {opts.target.upper()} targets on independent probe episodes')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(output/'critic_fit.png',dpi=160);plt.close(fig)
    print('ORIGINAL',json.dumps(original),flush=True);print('OUTPUT',output,flush=True)


if __name__=='__main__':main()
