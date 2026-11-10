"""Validation-only counterfactual actor-gain diagnostic; NOT a trained benchmark."""
import json
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import torch
from ray.rllib.core import Columns
from . import runner
from .current_control import environment_config
from .evaluation import episode
from .response import step_response_metrics
from .validate import quality
from .tdk_results import GATE


class GainProbe(torch.nn.Module):
    def __init__(self,policy,gain):
        super().__init__();self.policy=policy;self.gain=gain

    def get_inference_action_dist_cls(self):return self.policy.get_inference_action_dist_cls()

    def forward_inference(self,batch):
        output=dict(self.policy.forward_inference(batch))
        mean,log_std=output[Columns.ACTION_DIST_INPUTS].chunk(2,dim=-1)
        mean=torch.tanh(self.gain*torch.atanh(mean.clamp(-1.+1e-6,1.-1e-6)))
        output[Columns.ACTION_DIST_INPUTS]=torch.cat((mean,log_std),dim=-1)
        return output


def main():
    from ray.rllib.core.rl_module.rl_module import RLModule
    run='tdk_pure_reward20_16x8_seed345'
    args=runner.parser().parse_args(['eval','--motor','pmsm','--controller','ppo','--run',run])
    folder=runner.run_dir(args);manifest=json.loads((folder/'manifest.json').read_text())
    for key,value in manifest['settings'].items():
        if hasattr(args,key):setattr(args,key,value)
    module=runner.load_motor(args);config=environment_config(args)
    checkpoint=folder/'iteration_0500'
    policy=RLModule.from_checkpoint(str(checkpoint/'learner_group/learner/rl_module/default_policy'))
    output=runner.ROOT/'outputs/tdk_results_20261016/policy_gain_diagnosis'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True);records=[]
    for gain in [1.,1.5,2.,3.]:
        cases=[]
        for profile in ['steps','random']:
            for seed in [501,502]:
                args.profile=profile;args.seed=seed
                result=episode(args,module,GainProbe(policy,gain),config)
                response=step_response_metrics(result['data']) if profile=='steps' else []
                cases.append(dict(profile=profile,seed=seed,quality=quality(result,SimpleNamespace(**GATE)),
                    metrics=result['metrics'],response=response))
                np.savetxt(output/f'gain{gain}_{profile}_seed{seed}.csv',result['data'],delimiter=',')
        times=[max(c['response'],key=lambda r:abs(r['step_rad_s']))['rise_time_10_90_s'] for c in cases if c['response']]
        record=dict(gain=gain,mean_mae=float(np.mean([c['metrics']['MAE_speed_rad_s'] for c in cases])),
            worst_offset=max(c['quality'].get('values',{}).get('absolute_offset',float('inf')) for c in cases),
            worst_ripple=max(c['quality'].get('values',{}).get('speed_ripple',float('inf')) for c in cases),
            largest_reversal_rise_s=times,quality_pass=all(c['quality']['passed'] for c in cases),cases=cases)
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='cases'}),flush=True)
        runner.save_json(output/'diagnosis.json',dict(diagnostic_only=True,checkpoint=str(checkpoint),
            no_training_or_test_seeds_used=True,not_eligible_for_confirmation=True,records=records))
    print('DIAGNOSIS',output,flush=True)


if __name__=='__main__':main()
