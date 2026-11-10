"""Gated 100-iteration critic pilot; continue to 500 only on numerical evidence."""
import argparse
import contextlib
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime,timezone
import numpy as np
from . import runner,research,validate
from .pmsm_confirmation import matched_command

RUN='tdk_normalized_critic_seed345'


def learning_health(history,record):
    """Progression gate only; never replaces final engineering acceptance."""
    tail=history[-20:];ev=[];ratios=[]
    try:
        for row in tail:
            values=[float(row[k]) for k in ('vf_explained_var','value_prediction_std',
                                           'value_target_std','vf_normalized_loss')]
            if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite metrics')
            ev.append(values[0]);ratios.append(values[1]/max(values[2],1e-12))
        evidence=dict(iterations=len(tail),median_explained_variance=float(np.median(ev)),
                      median_prediction_target_std_ratio=float(np.median(ratios)))
        cases=record.get('cases',[])
        safe=(len(cases)==4 and all(c['metrics']['completed_episode'] and
              not c['metrics']['safety_violation'] and c['metrics']['time_above_current_scale_s']==0 for c in cases))
        evidence['passed']=bool(len(tail)==20 and evidence['median_explained_variance']>=.2 and
            .1<=evidence['median_prediction_target_std_ratio']<=3. and safe and record['mean_mae']<=60.)
        return evidence
    except (KeyError,ValueError,TypeError) as error:return dict(passed=False,error=str(error))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detach',action='store_true');p.add_argument('--resume',action='store_true')
    opts=p.parse_args(argv)
    root=runner.ROOT/'outputs/pmsm_critic_repair_v1';root.mkdir(parents=True,exist_ok=True)
    if opts.detach:
        arguments=list(sys.argv[1:] if argv is None else argv);arguments.remove('--detach')
        with (root/'driver.log').open('a') as stream:
            child=subprocess.Popen(['caffeinate','-i',sys.executable,'-m','rllib_control.critic_repair',*arguments],
                cwd=runner.ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print('BATCH PID',child.pid,'LOG',root/'driver.log',flush=True);return
    source=runner.ROOT/'RL_models/pmsm/ppo/tdk_pure_rate002_reward20_seed345/manifest.json'
    old=json.loads(source.read_text())['settings']
    command=matched_command(old,RUN,'ppo',345,100,2)
    command+=['--intrinsic-decay-iterations',str(old['intrinsic_decay_iterations']),
              '--critic-profile','normalized','--critic-return-scale','100']
    settings=runner.settings(runner.parser().parse_args(command))
    allowed={'critic_profile','critic_hidden_sizes','critic_return_scale','critic_learning_rate',
             'critic_loss','optimizer_protocol'}
    if {k for k in old.keys()|settings.keys() if old.get(k)!=settings.get(k)}-allowed:
        raise ValueError('Pilot changes more than the critic protocol')
    plan=dict(source_manifest=str(source),command=command,settings=settings,source_sha256=research.fingerprints(),
        pilot_iterations=100,final_iterations=500,held_out_tests_allowed=False,
        progression_gate=dict(last_iterations=20,min_median_explained_variance=.2,
            prediction_target_std_ratio=[.1,3.],max_pilot_MAE=60.,all_four_validation_cases_safe=True),
        final_gates=dict(max_mae=30.,max_ripple=1.,max_offset=10.,max_speed_pp=5.,
                         max_current_ripple_fraction=.05,max_step_rise=.24))
    with (root/'workflow.lock').open('a') as lease:
        try:fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Critic repair already running')
        if (root/'plan.json').exists():
            if not opts.resume:raise FileExistsError('Existing critic pilot: use --resume')
            if json.loads((root/'plan.json').read_text())!=plan:raise ValueError('Frozen plan/source changed')
        else:runner.save_json(root/'plan.json',plan)
        state=dict(pid=os.getpid(),run=RUN,held_out_tests_started=False)
        def save(**kw):
            state.update(kw,updated_at=datetime.now(timezone.utc).isoformat());runner.save_json(root/'state.json',state)
        def training(iterations):
            cmd=command.copy();cmd[cmd.index('--iterations')+1]=str(iterations)
            folder=runner.run_dir(runner.parser().parse_args(cmd))
            if (folder/'manifest.json').exists():
                stored=json.loads((folder/'manifest.json').read_text())
                if stored['settings']!=settings:raise ValueError('Pilot settings changed')
                if stored.get('completed_iteration',0)>=iterations:return
                if stored.get('status') not in ('complete','interrupted_or_failed'):
                    raise ValueError('Refusing a possibly active trainer')
                cmd.append('--resume')
            save(status='running',stage=f'train_{iterations}')
            with (root/'training.log').open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):runner.main(cmd)
        def validation(iterations):
            save(status='running',stage=f'validate_{iterations}')
            with (root/'validation.log').open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                validate.main(['--run',RUN,'--iterations',str(iterations),'--max-step-rise','.24'])
            path=sorted((runner.ROOT/'outputs/rllib/pmsm/validation'/RUN).glob('*/validation.json'))[-1]
            return path,json.loads(path.read_text())
        try:
            training(100);path,report=validation(100)
            folder=runner.run_dir(runner.parser().parse_args(command))
            with (folder/'training_history.csv').open() as stream:history=list(csv.DictReader(stream))
            health=learning_health([r for r in history if int(r['iteration'])<=100],report['records'][0])
            save(pilot_report=str(path),learning_health=health)
            if not health['passed']:
                save(status='pilot_failed',stage='stopped_before_long_training');return
            training(500);path,report=validation(500)
            save(status='qualified' if report.get('selected_checkpoint') else 'no_qualified_candidate',
                 stage='finished',final_report=str(path))
        except BaseException as error:
            save(status='error',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
