"""Two predeclared, pure PPO validation pilots; never evaluates held-out seeds."""
import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from . import runner, research, validate, trial_results
from .pmsm_confirmation import matched_command


def commands(source=None):
    from pathlib import Path

    if source is None:
        source=runner.ROOT/'RL_models/pmsm/ppo/tdk_pure_conditioned_reward20_16x8_seed345/manifest.json'
        if not source.exists():
            source=runner.ROOT/'archive/cleanup_20261020'/source.relative_to(runner.ROOT)
    else:
        source=Path(source)
    settings=json.loads(source.read_text())['settings']
    jobs=[]
    for suffix,actor in [('full','full'),('memory','speed_memory')]:
        run=f'tdk_horizon1s_{suffix}_seed345'
        command=matched_command(settings,run,'ppo',345,500,2)
        # Preserve the original coefficient schedule metadata, though PPO has no ICM.
        command+=['--intrinsic-decay-iterations',str(settings['intrinsic_decay_iterations']),
                  '--discount-horizon-s','1.0','--advantage-horizon-s','.2',
                  '--actor-observations',actor]
        actual=runner.settings(runner.parser().parse_args(command))
        allowed={'gamma','gae_lambda','discount_horizon_s','advantage_horizon_s','actor_observations'}
        differences={k for k in actual.keys()|settings.keys() if actual.get(k)!=settings.get(k)}
        if differences-allowed:raise ValueError(f'Unexpected pilot changes: {differences-allowed}')
        jobs.append(dict(run=run,command=command,settings=actual))
    return source,jobs


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detach',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    root=runner.ROOT/'outputs/pmsm_repair_v1'
    root.mkdir(parents=True,exist_ok=True)
    if args.detach:
        arguments=list(sys.argv[1:] if argv is None else argv);arguments.remove('--detach')
        with (root/'driver.log').open('a') as stream:
            child=subprocess.Popen(['caffeinate','-i',sys.executable,'-m','rllib_control.pmsm_repair',*arguments],
                cwd=runner.ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print('BATCH PID',child.pid,'LOG',root/'driver.log',flush=True);return
    source,jobs=commands()
    plan=dict(source_manifest=str(source),jobs=jobs,source_sha256=research.fingerprints(),
              iterations=500,validation_seeds=[501,502],validation_profiles=['steps','random'],
              held_out_tests_allowed=False,stopping_rule='Exactly two pilots; all fixed gates, final checkpoint only')
    with (root/'workflow.lock').open('a') as lease:
        try:fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Repair pilots already running')
        path=root/'plan.json'
        if path.exists():
            if not args.resume:raise FileExistsError('Use --resume for the existing repair plan')
            if json.loads(path.read_text())!=plan:raise ValueError('Frozen repair plan/source changed')
        else:runner.save_json(path,plan)
        state=dict(pid=os.getpid(),history=[],held_out_tests_started=False)
        def save(**values):
            state.update(values,updated_at=datetime.now(timezone.utc).isoformat())
            runner.save_json(root/'state.json',state)
        try:
            for job in jobs:
                run=job['run'];command=job['command'].copy()
                folder=runner.run_dir(runner.parser().parse_args(command))
                needs_training=True
                if (folder/'manifest.json').exists():
                    manifest=json.loads((folder/'manifest.json').read_text())
                    if manifest['settings']!=job['settings']:raise ValueError('Existing pilot settings differ')
                    if manifest.get('status')=='complete' and manifest.get('completed_iteration')==500:
                        needs_training=False
                    elif args.resume and manifest.get('status')=='interrupted_or_failed':command.append('--resume')
                    else:raise ValueError('Refusing to overwrite or duplicate a pilot trainer')
                if needs_training:
                    save(status='running',stage='train',run=run)
                    with (root/(run+'_train.log')).open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                        runner.main(command)
                save(status='running',stage='validate',run=run)
                with (root/(run+'_validation.log')).open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                    validate.main(['--run',run,'--iterations','500','--max-step-rise','.24'])
                reports=sorted((runner.ROOT/'outputs/rllib/pmsm/validation'/run).glob('*/validation.json'))
                report=json.loads(reports[-1].read_text())
                state['history'].append(dict(run=run,report=str(reports[-1]),passed=bool(report['selected_checkpoint'])))
                save()
            save(status='running',stage='comparison')
            with (root/'comparison.log').open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                trial_results.main(['--runs','tdk_pure_conditioned_reward20_16x8_seed345',*[job['run'] for job in jobs]])
            save(status='qualified' if any(r['passed'] for r in state['history']) else 'no_qualified_candidate',stage='finished')
        except BaseException as error:
            save(status='error',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
