"""One-factor reward trial: tracking weight 100 -> 200, gated 100 -> 500."""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime,timezone
from . import runner,research,validate,trial_results
from .critic_repair import learning_health
from .pmsm_confirmation import matched_command

SOURCE='tdk_normalized_critic_seed345'
RUN='tdk_tracking200_critic_seed345'


def comparison_values(record):
    cases=record['cases']
    rises=[max(c['response'],key=lambda r:abs(r['step_rad_s']))['rise_time_10_90_s']
           for c in cases if c['profile']=='steps']
    if len(cases)!=4 or len(rises)!=2 or not all(math.isfinite(v) for v in rises):
        raise ValueError('Four complete cases and two finite largest-reversal rise times required')
    names=['speed_ripple','speed_peak_to_peak','absolute_offset','current_ripple_fraction']
    result={name:max(c['quality']['values'][name] for c in cases) for name in names}
    result.update(mean_mae=record['mean_mae'],largest_rise=max(rises))
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite comparison measurements')
    return result


def progression(history,record,control):
    health=learning_health(history,record)
    result=dict(learning_health=health,passed=False)
    try:
        measured=comparison_values(record);reference=comparison_values(control)
        limits=dict(mean_mae=reference['mean_mae'],
                    largest_rise=.95*reference['largest_rise'],speed_ripple=1.,speed_peak_to_peak=5.,
                    absolute_offset=reference['absolute_offset'],
                    current_ripple_fraction=.05)
        result.update(values=measured,control_values=reference,progression_limits=limits,
            passed=bool(health['passed'] and all(measured[k]<=v for k,v in limits.items())))
    except (ValueError,KeyError,TypeError) as error:result['error']=str(error)
    return result


def make_command(settings):
    command=matched_command(settings,RUN,'ppo',settings['seed'],100,settings['num_env_runners'])
    command+=['--intrinsic-decay-iterations',str(settings['intrinsic_decay_iterations']),
              '--tracking-weight','200']
    actual=runner.settings(runner.parser().parse_args(command))
    differences={k for k in settings.keys()|actual.keys() if settings.get(k)!=actual.get(k)}
    if differences!={'tracking_weight'} or settings['tracking_weight']!=100.:
        raise ValueError(f'Reward trial must change ONLY tracking_weight 100 -> 200: {differences}')
    return command,actual


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detach',action='store_true');p.add_argument('--resume',action='store_true')
    opts=p.parse_args(argv)
    root=runner.ROOT/'outputs/pmsm_tracking200_trial_v1';root.mkdir(parents=True,exist_ok=True)
    if opts.detach:
        arguments=list(sys.argv[1:] if argv is None else argv);arguments.remove('--detach')
        with (root/'driver.log').open('a') as stream:
            child=subprocess.Popen(['caffeinate','-i',sys.executable,'-m','rllib_control.tracking_trial',*arguments],
                cwd=runner.ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print('BATCH PID',child.pid,'LOG',root/'driver.log',flush=True);return
    source=runner.ROOT/'RL_models/pmsm/ppo'/SOURCE/'manifest.json'
    manifest=json.loads(source.read_text())
    if manifest.get('status')!='complete' or manifest.get('completed_iteration')!=500:
        raise ValueError('Control training must be complete at 500 before starting a matched trial')
    state_path=runner.ROOT/'outputs/pmsm_critic_repair_v1/state.json'
    control_state=json.loads(state_path.read_text())
    control_path=runner.ROOT/str(control_state['pilot_report'])
    control_report=json.loads(control_path.read_text());control=control_report['records'][0]
    if (control_report['training_settings']!=manifest['settings'] or control['iteration']!=100
            or set(control_report['arguments']['seeds'])!={501,502}
            or set(control_report['arguments']['profiles'])!={'steps','random'}):
        raise ValueError('Control must be the matched 100-iteration four-case validation')
    comparison_values(control)
    command,settings=make_command(manifest['settings'])
    plan=dict(source_manifest=str(source),source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        control_report=str(control_path),control_report_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),
        command=command,settings=settings,source_sha256=research.fingerprints(),
        pilot_iterations=100,final_iterations=500,held_out_tests_allowed=False,
        progression_rule='Critic health and safe completion; >=5% faster than matched control at 100; MAE/offset no worse; fixed ripple/peak-to-peak/current-ripple ceilings',
        final_rule='All original quality gates plus <=240 ms largest reversal, final checkpoint only')
    with (root/'workflow.lock').open('a') as lease:
        try:fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Tracking trial already running')
        if (root/'plan.json').exists():
            if not opts.resume:raise FileExistsError('Existing trial: use --resume')
            if json.loads((root/'plan.json').read_text())!=plan:raise ValueError('Frozen trial/source changed')
        else:runner.save_json(root/'plan.json',plan)
        state=dict(pid=os.getpid(),run=RUN,held_out_tests_started=False)
        def save(**kw):
            state.update(kw,updated_at=datetime.now(timezone.utc).isoformat());runner.save_json(root/'state.json',state)
        def train(iterations):
            cmd=command.copy();cmd[cmd.index('--iterations')+1]=str(iterations)
            folder=runner.run_dir(runner.parser().parse_args(cmd))
            if (folder/'manifest.json').exists():
                stored=json.loads((folder/'manifest.json').read_text())
                if stored['settings']!=settings:raise ValueError('Existing model settings differ')
                if stored.get('completed_iteration',0)>=iterations:return
                if stored.get('status') not in ('complete','interrupted_or_failed'):
                    raise ValueError('Refusing a possibly active trainer')
                cmd.append('--resume')
            save(status='running',stage=f'train_{iterations}')
            with (root/'training.log').open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):runner.main(cmd)
        def evaluate(iterations):
            save(status='running',stage=f'validate_{iterations}')
            with (root/'validation.log').open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                validate.main(['--run',RUN,'--iterations',str(iterations),'--max-step-rise','.24'])
            path=sorted((runner.ROOT/'outputs/rllib/pmsm/validation'/RUN).glob('*/validation.json'))[-1]
            return path,json.loads(path.read_text())
        try:
            train(100);path,report=evaluate(100)
            folder=runner.run_dir(runner.parser().parse_args(command))
            with (folder/'training_history.csv').open() as stream:history=list(csv.DictReader(stream))
            gate=progression([r for r in history if int(r['iteration'])<=100],report['records'][0],control)
            save(pilot_report=str(path),progression=gate)
            if not gate['passed']:
                save(status='pilot_failed',stage='stopped_before_long_training');return
            train(500);path,report=evaluate(500)
            save(final_report=str(path),status='running',stage='comparison')
            with (root/'comparison.log').open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                trial_results.main(['--runs',SOURCE,RUN])
            save(status='qualified' if report.get('selected_checkpoint') else 'no_qualified_candidate',stage='finished')
        except BaseException as error:
            save(status='error',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
