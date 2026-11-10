"""Execute the four remaining TDK stages, stopping before held-out data on failed validation."""
import argparse
import contextlib
import hashlib
import json
import math
import time
import os
import sys
import subprocess
import fcntl
import traceback
from datetime import datetime,timezone
from pathlib import Path
from . import runner,research,validate,pmsm_confirmation,tdk_extension,trial_results


def latest_validation(run,controller='ppo'):
    reports=sorted((runner.ROOT/'outputs/rllib/pmsm/validation'/run).glob('*/validation.json'))
    if not reports:raise ValueError(f'No validation report for {controller}/{run}')
    path=reports[-1];report=json.loads(path.read_text())
    if report['arguments']['controller']!=controller:raise ValueError('Validation controller mismatch')
    return path,report


def pick_candidate(reports):
    eligible=[]
    for run,path,report in reports:
        selected=report.get('selected_checkpoint')
        record=next((r for r in report['records'] if r['checkpoint']==selected),None)
        if (selected and record and record.get('complete') and record.get('iteration')==500
                and record.get('quality_pass') and record.get('response_pass')
                and isinstance(record.get('mean_mae'),(int,float)) and math.isfinite(record['mean_mae']) and record['mean_mae']<=30.):
            eligible.append((record['mean_mae'],run,path,selected))
    return min(eligible,key=lambda r:r[0]) if eligible else None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',default='tdk_final_v2')
    p.add_argument('--candidates',nargs='+',default=['tdk_speed_actor_seed345','tdk_speed_actor_lownoise_seed345'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--wait-for-candidates',action='store_true',help='Wait for already-running candidate training before qualifying; never starts a duplicate trainer')
    p.add_argument('--train-candidates',action='store_true',help='Resume explicitly interrupted candidate training from its existing checkpoint before qualification; refuses running trainers')
    p.add_argument('--detach',action='store_true',help='Launch one independent local batch process with caffeinate and persistent driver log')
    args=p.parse_args(argv)
    if args.wait_for_candidates and args.train_candidates:p.error('Choose waiting OR ownership of interrupted candidate training')
    if not 1<=len(args.candidates)<=2 or len(set(args.candidates))!=len(args.candidates):p.error('Choose 1 or 2 predeclared candidates')
    for name in [args.experiment,*args.candidates]:runner.run_dir(runner.parser().parse_args(['train','--run',name]))
    if args.detach:
        arguments=list(sys.argv[1:] if argv is None else argv);arguments.remove('--detach')
        log=runner.ROOT/'outputs'/(args.experiment+'_driver.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            child=subprocess.Popen(['caffeinate','-i',sys.executable,'-m','rllib_control.tdk_finish',*arguments],
                cwd=runner.ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print('BATCH PID',child.pid,'DRIVER LOG',log,flush=True);return 0
    root=runner.ROOT/'outputs/rllib/research'/args.experiment/'workflow'
    plan=dict(experiment=args.experiment,candidates=args.candidates,source_sha256=research.fingerprints(),
        final_iterations=500,training_seeds=[123,234,456],validation_seeds=[501,502],
        wait_for_candidates=args.wait_for_candidates,train_candidates=args.train_candidates,
        nominal_test_seeds=[1901,1902,1903],robustness_test_seeds=[2901,2902,2903],
        candidate_rule='Final checkpoint only; fixed quality gates and largest-step rise <= 240 ms; lowest validation MAE')
    if root.exists() and not args.resume:p.error('Workflow exists; use --resume with unchanged code and arguments')
    root.mkdir(parents=True,exist_ok=True)
    lease=(root/'workflow.lock').open('a')
    try:fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('This experiment already has a running batch process')
    plan_path=root/'plan.json';state_path=root/'state.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        lease.close();p.error('Frozen workflow/code changed; use a new experiment')
    runner.save_json(plan_path,plan)
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(history=[])
    state['pid']=os.getpid()
    def save(**values):
        state.update(values,updated_at=datetime.now(timezone.utc).isoformat());runner.save_json(state_path,state)
    def step(name,function):
        save(status='running',stage=name)
        state['history'].append(dict(stage=name,status='started',time=state['updated_at']))
        save()
        print('STAGE',name,flush=True)
        with (root/(name+'.log')).open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            result=function()
        state['history'].append(dict(stage=name,status='complete',time=datetime.now(timezone.utc).isoformat()))
        save();return result
    try:
        if args.train_candidates:
            for run in args.candidates:
                folder=runner.run_dir(runner.parser().parse_args(['train','--motor','pmsm','--run',run]))
                manifest=json.loads((folder/'manifest.json').read_text())
                if manifest.get('status')=='complete':continue
                if manifest.get('status')!='interrupted_or_failed':
                    raise ValueError(f'Refusing to duplicate a possibly running candidate trainer: {run}')
                settings=manifest['settings']
                command=pmsm_confirmation.matched_command(settings,run,'ppo',settings['seed'],500,settings['num_env_runners'])
                command+=['--intrinsic-decay-iterations',str(settings['intrinsic_decay_iterations']),'--resume']
                if runner.settings(runner.parser().parse_args(command))!=settings:raise ValueError('Candidate resume settings differ')
                step('resume_candidate_'+run,lambda command=command:runner.main(command))
        deadline=time.monotonic()+30*60
        while True:
            pending=[]
            for run in args.candidates:
                folder=runner.run_dir(runner.parser().parse_args(['train','--motor','pmsm','--run',run]))
                manifest=json.loads((folder/'manifest.json').read_text())
                if manifest.get('status')=='interrupted_or_failed':raise ValueError(f'Candidate training stopped: {run}')
                if manifest.get('status')!='complete' or not (folder/'iteration_0500').is_dir():pending.append(run)
            if not pending:break
            if not args.wait_for_candidates:raise ValueError(f'Candidate training must finish first: {pending}')
            if time.monotonic()>deadline:raise TimeoutError('Candidate training did not finish within the 30-minute dependency window')
            save(status='running',stage='awaiting_candidate_training',pending_candidates=pending,held_out_tests_started=False)
            time.sleep(10)
        if not state.get('qualification'):
            reports=[]
            for run in args.candidates:
                step('qualify_'+run,lambda run=run:validate.main(['--motor','pmsm','--controller','ppo','--run',run,
                                                               '--iterations','500','--max-step-rise','.24']))
                path,report=latest_validation(run);reports.append((run,path,report))
            step('candidate_comparison',lambda:trial_results.main(['--iterations','500','--runs',*args.candidates]))
            selected=pick_candidate(reports)
            if selected is None:
                save(status='candidate_failed',stage='candidate_selection',held_out_tests_started=False,
                     message='No final candidate passed all fixed validation gates; no six-run study or held-out evaluation started')
                print('NO QUALIFIED CANDIDATE. Reports:',root,flush=True);return 2
            _,run,path,checkpoint=selected
            save(candidate=run,checkpoint=checkpoint,qualification=str(path.resolve()),
                 qualification_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        qualification=Path(state['qualification'])
        if hashlib.sha256(qualification.read_bytes()).hexdigest()!=state['qualification_sha256']:
            raise ValueError('Qualification report changed')
        name=args.experiment+'_confirm'
        common=['--experiment',name,'--qualification',str(qualification)]
        step('confirmation_plan',lambda:pmsm_confirmation.main(['plan',*common]))
        step('freeze_baseline',lambda:pmsm_confirmation.main(['freeze-baseline',*common]))
        parent=runner.ROOT/'outputs/rllib/research'/name/'confirmation/protocol.json'
        extensions={s:[ '--study',s,'--experiment',args.experiment+'_'+s,'--parent-protocol',str(parent)]
                    for s in ['robustness','environment','reward']}
        for study,command in extensions.items():step('plan_'+study,lambda command=command:tdk_extension.main(['plan',*command]))
        step('train_six_confirmation_models',lambda:pmsm_confirmation.main(['train',*common,*(['--resume'] if args.resume else [])]))
        protocol=json.loads(parent.read_text());validation=[]
        for job in protocol['jobs']:
            if job['controller']=='baseline':continue
            step('validate_'+job['id'],lambda job=job:validate.main(['--motor','pmsm','--controller',job['controller'],
                '--run',job['run'],'--iterations','500','--max-step-rise','.24']))
            path,report=latest_validation(job['run'],job['controller'])
            validation.append(dict(run=job['run'],controller=job['controller'],passed=bool(report.get('selected_checkpoint')),report=str(path)))
        save(confirmation_validation=validation)
        if any(not r['passed'] for r in validation if r['controller']=='ppo'):
            save(status='confirmation_validation_failed',held_out_tests_started=False,
                 message='Fresh PPO seeds did not all meet the predeclared engineering goals. ICM failures are retained, not filtered.')
            print('CONFIRMATION VALIDATION FAILED. Reports:',root,flush=True);return 3
        save(held_out_tests_started=True)
        step('nominal_confirmation_tests',lambda:pmsm_confirmation.main(['evaluate',*common,*(['--resume'] if args.resume else [])]))
        step('nominal_confirmation_report',lambda:pmsm_confirmation.main(['aggregate',*common]))
        for study,command in extensions.items():
            step(study+'_study',lambda command=command:tdk_extension.main(['all',*command,*(['--resume'] if args.resume else [])]))
        save(status='complete',stage='all_experiments_finished',
             message='Execution complete; consult reports for engineering failures and tradeoffs, not every model is guaranteed to pass')
        print('WORKFLOW COMPLETE',root,flush=True);return 0
    except BaseException as error:
        save(status='error',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        lease.close()


if __name__=='__main__':raise SystemExit(main())
