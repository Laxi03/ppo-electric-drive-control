"""Regenerate paired DC mixed results using the documented step-design baseline."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from . import results, runner


def main():
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    from .current_control import environment_config
    from .baselines import load_gains
    from .evaluation import episode, save_reports
    from .response import step_response_metrics
    from ray.rllib.core.rl_module.rl_module import RLModule
    root=runner.ROOT/'outputs/dc_design_comparison'/datetime.now().strftime('%Y%m%d_%H%M%S')
    baseline_path=runner.ROOT/'outputs/overshoot_design/dc_current_baseline.json'
    loaded={c:results.restored('dc',c,results.PRESETS['dc'][c]) for c in ['ppo','icm']}
    args=loaded['ppo'][0];config=environment_config(args)
    assert config==environment_config(loaded['icm'][0])
    args.profile='mixed';args.seed=501;args.scenario='nominal';args.episode_length=8.
    args.show=False;args.plot_reward=False;args.plot_tsne=False;args.baseline_config=str(baseline_path)
    gains=load_gains(args,config);module=runner.load_motor(args)
    policies={c:RLModule.from_checkpoint(str(v[2]/'learner_group/learner/rl_module/default_policy')) for c,v in loaded.items()}
    baseline=json.loads(baseline_path.read_text())
    for case in ['nominal','parameter_mismatch']:
        args.resistance_scale=1. if case=='nominal' else 1.3
        args.inductance_scale=1. if case=='nominal' else .8;args.inertia_scale=1.
        folder=root/'evidence/dc'/case;folder.mkdir(parents=True)
        records={}
        for c in ['ppo','icm','pi']:
            records[c]=episode(args,module,policies.get(c),config,gains if c=='pi' else None)
            print(case,c,records[c]['metrics']['MAE_speed_rad_s'],flush=True)
        ref=records['ppo']['data'][:,:2]
        for r in records.values():np.testing.assert_array_equal(ref,r['data'][:,:2])
        save_reports(args,records,folder)
        runner.save_json(folder/'responses.json',{c:step_response_metrics(r['data']) for c,r in records.items()})
        runner.save_json(folder/'frozen_baseline.json',baseline)
        runner.save_json(folder/'model_manifests.json',{c:v[1] for c,v in loaded.items()})
        runner.save_json(folder/'evaluation_manifest.json',dict(motor='dc',case=case,profile='mixed',seed=501,
            duration_s=8.,resistance_scale=args.resistance_scale,inductance_scale=args.inductance_scale,
            inertia_scale=1.,mismatch_timing='present from reset, not a mid-episode jump',
            baseline=str(baseline_path),baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            baseline_tuning=baseline['tuning'],environment=config,
            checkpoints={c:str(v[2]) for c,v in loaded.items()},
            purpose='Selected-model comparison using an explicit nominal step-design baseline; not held-out qualification'))
    runner.save_json(root.parent/'latest.json',dict(directory=str(root)))
    print('DC RESULTS:',root,flush=True)


if __name__=='__main__':main()
