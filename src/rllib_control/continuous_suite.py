"""Paired PPO, constant-ICM and decaying-ICM trials; deliberately sequential."""
import argparse
import csv
from pathlib import Path
from .runner import main, ROOT


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='continuous_fullrange_seed123')
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--train-batch-size',type=int,default=4096)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--train-reference-scale',type=float,default=1.)
    p.add_argument('--num-env-runners',type=int,default=2)
    p.add_argument('--dc-baseline',required=True)
    p.add_argument('--pmsm-baseline',required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    for path in (args.dc_baseline,args.pmsm_baseline):
        if not Path(path).is_file():
            p.error(f'Missing frozen baseline: {path}')
    for motor in ('dc','pmsm'):
        for variant,controller in (('constant','ppo'),('constant','icm'),('decay','icm')):
            name=args.run if variant=='constant' else args.run+'_decay'
            command=['train','--motor',motor,'--controller',controller,'--action-mode','continuous',
                     '--run',name,'--iterations',str(args.iterations),'--train-batch-size',str(args.train_batch_size),
                     '--seed',str(args.seed),'--train-reference-scale',str(args.train_reference_scale),
                     '--num-env-runners',str(args.num_env_runners)]
            if variant=='decay':
                command+=['--intrinsic-final-coeff','0.001','--intrinsic-decay-iterations',str(max(2,args.iterations))]
            if args.resume:
                folder=ROOT/'RL_models'/motor/controller/name
                if (folder/'manifest.json').exists():
                    command+=['--resume']
            main(command)
        for variant in ('constant','decay'):
            icm_name=args.run if variant=='constant' else args.run+'_decay'
            ppo=ROOT/'RL_models'/motor/'ppo'/args.run/f'iteration_{args.iterations:04d}'
            icm=ROOT/'RL_models'/motor/'icm'/icm_name/f'iteration_{args.iterations:04d}'
            main(['compare','--motor',motor,'--run',icm_name,'--ppo-checkpoint',str(ppo),
                  '--curiosity-checkpoint',str(icm),'--profile','steps','--seed',str(args.seed),
                  '--baseline-config',getattr(args,motor+'_baseline')])


if __name__=='__main__':
    run()
