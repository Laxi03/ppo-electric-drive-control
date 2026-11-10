"""Bounded 500 -> 700 continuation on a copied run; never mutates the source."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from datetime import datetime,timezone
from . import runner,research,validate
from .pmsm_confirmation import matched_command

SOURCE='tdk_normalized_critic_seed345'
RUN='tdk_normalized_critic_continue700_seed345'


def file_hashes(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file()}


def clone_run(source,destination,expected):
    """Atomic owned-run copy; reject overwrites and incomplete prior copies."""
    if destination.exists():raise FileExistsError(destination)
    pending=destination.with_name('.'+destination.name+'_clone_pending')
    if pending.exists():raise FileExistsError(f'Inspect incomplete prior clone: {pending}')
    pending.mkdir(parents=True)
    checkpoint=source/'iteration_0500'
    if file_hashes(checkpoint)!=expected:raise ValueError('Source checkpoint changed')
    shutil.copytree(checkpoint,pending/'iteration_0500')
    if file_hashes(pending/'iteration_0500')!=expected:raise ValueError('Copied checkpoint differs')
    shutil.copy2(source/'training_history.csv',pending/'training_history.csv')
    manifest=json.loads((source/'manifest.json').read_text())
    manifest.update(status='interrupted_or_failed',target_iterations=700,
        lineage=dict(source_checkpoint=str(checkpoint.resolve()),pretraining_iterations=500,
                     additional_training_iterations=200,total_iterations=700))
    runner.save_json(pending/'manifest.json',manifest)
    pending.rename(destination)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detach',action='store_true');p.add_argument('--resume',action='store_true')
    opts=p.parse_args(argv)
    root=runner.ROOT/'outputs/pmsm_continuation700_v1';root.mkdir(parents=True,exist_ok=True)
    if opts.detach:
        arguments=list(sys.argv[1:] if argv is None else argv);arguments.remove('--detach')
        with (root/'driver.log').open('a') as stream:
            child=subprocess.Popen(['caffeinate','-i',sys.executable,'-m','rllib_control.continuation_trial',*arguments],
                cwd=runner.ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print('BATCH PID',child.pid,'LOG',root/'driver.log',flush=True);return
    source=runner.ROOT/'RL_models/pmsm/ppo'/SOURCE
    manifest=json.loads((source/'manifest.json').read_text());settings=manifest['settings']
    if manifest.get('status')!='complete' or manifest.get('completed_iteration')!=500:
        raise ValueError('A completed 500-iteration source is required')
    command=matched_command(settings,RUN,'ppo',settings['seed'],700,settings['num_env_runners'])
    command+=['--intrinsic-decay-iterations',str(settings['intrinsic_decay_iterations']),'--resume']
    parsed=runner.parser().parse_args(command);destination=runner.run_dir(parsed)
    if runner.settings(parsed)!=settings:raise ValueError('Continuation settings differ from source')
    originals=dict(manifest=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                   history=hashlib.sha256((source/'training_history.csv').read_bytes()).hexdigest(),
                   checkpoint=file_hashes(source/'iteration_0500'))
    plan=dict(source=str(source),destination=str(destination),originals=originals,command=command,
        source_sha256=research.fingerprints(),pretraining_iterations=500,additional_iterations=200,
        final_iterations=700,final_checkpoint_only=True,held_out_tests_allowed=False,
        selection_rule='No tuning of parameters or limits; evaluate final 700 only on all original validation gates')
    with (root/'workflow.lock').open('a') as lease:
        try:fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Continuation already running')
        if (root/'plan.json').exists():
            if not opts.resume:raise FileExistsError('Existing continuation plan: use --resume')
            if json.loads((root/'plan.json').read_text())!=plan:raise ValueError('Frozen plan or source changed')
        else:
            if destination.exists():raise FileExistsError('Destination exists without this plan')
            runner.save_json(root/'plan.json',plan)
        state=dict(pid=os.getpid(),run=RUN,held_out_tests_started=False,
                   pretraining_iterations=500,additional_iterations=200,total_iterations=700)
        def save(**kw):
            state.update(kw,updated_at=datetime.now(timezone.utc).isoformat());runner.save_json(root/'state.json',state)
        try:
            if not destination.exists():clone_run(source,destination,originals['checkpoint'])
            stored=json.loads((destination/'manifest.json').read_text())
            if stored['settings']!=settings:raise ValueError('Copied run settings changed')
            if stored.get('completed_iteration',0)<700:
                if stored.get('status') not in ('complete','interrupted_or_failed'):
                    raise ValueError('Refusing to duplicate a possibly active trainer')
                save(status='running',stage='train_501_to_700')
                with (root/'training.log').open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                    runner.main(command)
            current=json.loads((destination/'manifest.json').read_text())
            current['lineage']=dict(source_checkpoint=str(source/'iteration_0500'),
                pretraining_iterations=500,additional_training_iterations=200,total_iterations=700,
                original_checkpoint_sha256=originals['checkpoint'])
            runner.save_json(destination/'manifest.json',current)
            save(status='running',stage='validate_final700')
            with (root/'validation.log').open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                validate.main(['--run',RUN,'--iterations','700','--max-step-rise','.24'])
            path=sorted((runner.ROOT/'outputs/rllib/pmsm/validation'/RUN).glob('*/validation.json'))[-1]
            report=json.loads(path.read_text())
            if (hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()!=originals['manifest'] or
                hashlib.sha256((source/'training_history.csv').read_bytes()).hexdigest()!=originals['history'] or
                file_hashes(source/'iteration_0500')!=originals['checkpoint']):
                raise ValueError('Original source run changed unexpectedly')
            save(status='qualified' if report.get('selected_checkpoint') else 'no_qualified_candidate',
                 stage='finished',final_report=str(path),source_preserved=True)
        except BaseException as error:
            save(status='error',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
