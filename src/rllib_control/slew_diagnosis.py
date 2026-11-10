"""Counterfactual validation diagnostic, NOT a trained-controller comparison.

The saved blue policy is held fixed while actuator limits are varied. The result
only diagnoses a possible bottleneck. Changed limits require retraining before
they can be used in the final benchmark.
"""
import json
from datetime import datetime
from . import runner
from .current_control import environment_config
from .evaluation import episode
from .response import step_response_metrics


def main():
    from ray.rllib.core.rl_module.rl_module import RLModule
    run='tdk_pure_reward20_16x8_seed345'
    args=runner.parser().parse_args(['eval','--motor','pmsm','--controller','ppo','--run',run])
    folder=runner.run_dir(args)
    saved=json.loads((folder/'manifest.json').read_text())
    for key,value in saved['settings'].items():
        if hasattr(args,key):setattr(args,key,value)
    args.seed=501;args.profile='steps'
    config=environment_config(args)
    policy=RLModule.from_checkpoint(str(folder/'iteration_0500/learner_group/learner/rl_module/default_policy'))
    module=runner.load_motor(args)
    out=runner.ROOT/'outputs/tdk_results_20261016/slew_diagnosis'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    results=[]
    for label,voltage,current in [('Trained limits',1000.,500.),('Voltage slew x5',5000.,500.),('Both slew limits raised',5000.,1000.)]:
        result=episode(args,module,policy,dict(config,command_slew_v_per_s=voltage,current_reference_slew=current))
        rows=step_response_metrics(result['data'])
        record=dict(label=label,voltage_slew_V_per_s=voltage,current_reference_slew_A_per_s=current,
                    response=rows,metrics=result['metrics'])
        results.append(record)
        print(label,[(round(r['reference_rad_s']),round(r['rise_time_10_90_s'],4)) for r in rows],flush=True)
    runner.save_json(out/'diagnosis.json',dict(diagnostic_only=True,policy_run=run,iterations=500,
        changed_limits_not_used_for_final_test=True,results=results))
    print('DIAGNOSIS',out,flush=True)


if __name__=='__main__':main()
