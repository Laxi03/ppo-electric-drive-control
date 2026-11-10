"""Six fresh matched PMSM runs, frozen baseline, nominal and robustness tests."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime,timezone
from . import runner,research,validate,pmsm_confirmation,tdk_extension

QUALIFICATION=runner.ROOT/'outputs/rllib/pmsm/validation/tdk_normalized_critic_continue700_seed345/20261016_164900_242713/validation.json'


def completed_models(protocol):
    records=[]
    for job in protocol['jobs']:
        if job['controller']=='baseline':continue
        args=runner.parser().parse_args(job['command']);folder=runner.run_dir(args)
        try:
            manifest=json.loads((folder/'manifest.json').read_text())
            complete=(manifest.get('status')=='complete' and manifest.get('completed_iteration')==700
                and manifest['settings']==job['settings'] and (folder/'iteration_0700').is_dir()
                and manifest.get('training_protocol',{}).get('milestones')==[100,500,700])
            records.append(dict(run=job['run'],controller=job['controller'],seed=job['train_seed'],complete=complete))
        except (OSError,ValueError,KeyError) as error:
            records.append(dict(run=job['run'],controller=job['controller'],seed=job['train_seed'],complete=False,error=str(error)))
    return records


def ppo_validation_passed(records):
    ppo=[r for r in records if r['controller']=='ppo']
    return len(ppo)==3 and {r['seed'] for r in ppo}=={123,234,456} and all(r['passed'] for r in ppo)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',default='tdk_confirm700_v1')
    p.add_argument('--qualification',type=Path,default=QUALIFICATION)
    p.add_argument('--resume',action='store_true');p.add_argument('--detach',action='store_true')
    args=p.parse_args(argv)
    runner.run_dir(runner.parser().parse_args(['train','--run',args.experiment]))
    if args.detach:
        arguments=list(sys.argv[1:] if argv is None else argv);arguments.remove('--detach')
        log=runner.ROOT/'outputs'/(args.experiment+'_driver.log')
        with log.open('a') as stream:
            child=subprocess.Popen(['caffeinate','-i',sys.executable,'-m','rllib_control.confirmation700_batch',*arguments],
                cwd=runner.ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print('BATCH PID',child.pid,'DRIVER LOG',log,flush=True);return
    qualification=args.qualification.resolve()
    requested=pmsm_confirmation.make_protocol(args.experiment,qualification,[123,234,456],700,2)
    root=runner.ROOT/'outputs/rllib/research'/args.experiment/'workflow'
    root.mkdir(parents=True,exist_ok=True)
    parent=root.parent/'confirmation/protocol.json'
    plan=dict(experiment=args.experiment,qualification=str(qualification),
        qualification_sha256=hashlib.sha256(qualification.read_bytes()).hexdigest(),
        source_sha256=research.fingerprints(),iterations=700,training_milestones=[100,500,700],
        training_seeds=[123,234,456],validation_seeds=[501,502],nominal_test_seeds=[1901,1902,1903],
        robustness_test_seeds=[2901,2902,2903],training_jobs=6,nominal_cases=42,robustness_cases=210,
        test_opening_rule='All six complete final models; all three PPO seeds pass unchanged validation goals; ICM failures retained')
    with (root/'workflow.lock').open('a') as lease:
        try:fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This confirmation batch is already running')
        if (root/'plan.json').exists():
            if not args.resume:raise FileExistsError('Existing batch: use --resume with unchanged code')
            if json.loads((root/'plan.json').read_text())!=plan:raise ValueError('Frozen batch/code changed')
        else:runner.save_json(root/'plan.json',plan)
        state_path=root/'state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else dict(history=[],held_out_tests_started=False)
        state['pid']=os.getpid()
        def save(**kw):
            state.update(kw,updated_at=datetime.now(timezone.utc).isoformat());runner.save_json(state_path,state)
        def step(name,function):
            save(status='running',stage=name)
            state['history'].append(dict(stage=name,status='started',time=state['updated_at']));save()
            print('STAGE',name,flush=True)
            with (root/(name+'.log')).open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                result=function()
            state['history'].append(dict(stage=name,status='complete',time=datetime.now(timezone.utc).isoformat()));save()
            return result
        common=['--experiment',args.experiment,'--qualification',str(qualification),'--iterations','700']
        robust=['--study','robustness','--experiment',args.experiment+'_robustness','--parent-protocol',str(parent)]
        try:
            step('freeze_confirmation_plan',lambda:pmsm_confirmation.main(['plan',*common]))
            if json.loads(parent.read_text())!=requested:raise ValueError('Unexpected confirmation protocol')
            step('freeze_pi_foc_baseline',lambda:pmsm_confirmation.main(['freeze-baseline',*common]))
            step('freeze_robustness_plan',lambda:tdk_extension.main(['plan',*robust]))
            step('train_six_matched_models',lambda:pmsm_confirmation.main(['train',*common,*(['--resume'] if args.resume else [])]))
            protocol=json.loads(parent.read_text());models=completed_models(protocol);save(models=models)
            if len(models)!=6 or not all(r['complete'] for r in models):
                save(status='training_failed',stage='stopped_before_tests',held_out_tests_started=False);return
            records=[]
            for job in protocol['jobs']:
                if job['controller']=='baseline':continue
                step('validate_'+job['id'],lambda job=job:validate.main(['--run',job['run'],'--controller',job['controller'],
                    '--iterations','700','--max-step-rise','.24']))
                path=sorted((runner.ROOT/'outputs/rllib/pmsm/validation'/job['run']).glob('*/validation.json'))[-1]
                report=json.loads(path.read_text());record=report['records'][0]
                records.append(dict(run=job['run'],controller=job['controller'],seed=job['train_seed'],
                    report=str(path),passed=bool(report.get('selected_checkpoint')),
                    mean_mae=record.get('mean_mae'),worst_ripple=record.get('worst_ripple'),
                    largest_reversal_rise_s=record.get('largest_reversal_rise_s')))
                save(validation=records)
            research.write_csv(root/'validation_summary.csv',records,list(records[0]))
            lines=['# Fresh three-seed validation before held-out tests','',
                'All final models retained, including failed ICM/PPO seeds. No best-seed selection.',
                '', '| Controller | Seed | Passed all goals | Mean MAE | Worst ripple RMS | Largest reversal rise [s] |',
                '|---|---:|---|---:|---:|---|']
            for r in records:lines.append(f"| {r['controller']} | {r['seed']} | {r['passed']} | {r['mean_mae']} | {r['worst_ripple']} | {r['largest_reversal_rise_s']} |")
            (root/'VALIDATION_SUMMARY.md').write_text('\n'.join(lines)+'\n')
            if not ppo_validation_passed(records):
                save(status='confirmation_validation_failed',stage='stopped_before_tests',held_out_tests_started=False,
                     message='Fresh PPO seeds did not all pass. All failures retained; held-out sets untouched.');return
            research.freeze_models(protocol,parent.parent)
            save(held_out_tests_started=True)
            step('nominal_held_out_tests',lambda:pmsm_confirmation.main(['evaluate',*common,*(['--resume'] if args.resume else [])]))
            step('nominal_report',lambda:pmsm_confirmation.main(['aggregate',*common]))
            step('robustness_held_out_tests',lambda:tdk_extension.main(['all',*robust,*(['--resume'] if args.resume else [])]))
            errors=[]
            for protocol_path in [parent,runner.ROOT/'outputs/rllib/research'/(args.experiment+'_robustness')/'robustness/protocol.json']:
                planned=json.loads(protocol_path.read_text())
                for job,profile,scenario,seed in research.cases(planned):
                    key=research.case_id(job,profile,scenario,seed)
                    path=protocol_path.parent/'cases'/(key+'.json')
                    case=json.loads(path.read_text()) if path.exists() else dict(status='missing')
                    if case['status'] not in ('complete','failed_episode'):errors.append(dict(case=key,status=case['status']))
            save(status='complete_with_errors' if errors else 'complete',stage='finished',evaluation_errors=errors,
                message='Execution finished. Engineering failures and seed variability remain in the reports.')
        except BaseException as error:
            save(status='error',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
