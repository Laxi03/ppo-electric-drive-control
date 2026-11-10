import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rllib_control import runner
from rllib_control.tdk_finish import main,pick_candidate


class FinishTests(unittest.TestCase):
    def test_only_complete_final_qualified_candidates_enter_selection(self):
        def item(run,mae,**changes):
            record=dict(checkpoint=run,iteration=500,complete=True,quality_pass=True,response_pass=True,mean_mae=mae)
            record.update(changes)
            return run,Path(run),dict(selected_checkpoint=run,records=[record])
        records=[item('ripple_failure',1.,quality_pass=False),item('slow',2.,response_pass=False),
                 item('unfinished',2.,complete=False),item('earlier',3.,iteration=250),
                 item('large_error',40.),item('nonfinite',float('nan'))]
        self.assertIsNone(pick_candidate(records))
        records+=[item('ok',25.),item('better',20.)]
        self.assertEqual(pick_candidate(records)[1],'better')

    @patch('rllib_control.research.fingerprints',return_value={'code':'hash'})
    def test_failed_candidate_never_starts_confirmation_or_held_out_tests(self,_):
        with tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
            root=Path(directory)
            for run in ['a','b']:
                folder=root/'RL_models/pmsm/ppo'/run
                (folder/'iteration_0500').mkdir(parents=True)
                runner.save_json(folder/'manifest.json',dict(status='complete'))
            def fake_validate(command):
                run=command[command.index('--run')+1]
                folder=root/'outputs/rllib/pmsm/validation'/run/'stamp';folder.mkdir(parents=True)
                runner.save_json(folder/'validation.json',dict(arguments=dict(controller='ppo'),records=[],selected_checkpoint=None))
            with patch('rllib_control.validate.main',side_effect=fake_validate),\
                 patch('rllib_control.trial_results.main'),\
                 patch('rllib_control.pmsm_confirmation.main') as confirmation,\
                 patch('rllib_control.tdk_extension.main') as extensions:
                code=main(['--experiment','test','--candidates','a','b'])
                self.assertEqual(code,2);confirmation.assert_not_called();extensions.assert_not_called()
            state=json.loads((root/'outputs/rllib/research/test/workflow/state.json').read_text())
            self.assertEqual(state['status'],'candidate_failed')
            self.assertFalse(state['held_out_tests_started'])

    @patch('rllib_control.research.fingerprints',return_value={'code':'hash'})
    def test_all_plans_precede_tests_and_icm_failures_are_not_filtered(self,_):
        with tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
            root=Path(directory);events=[]
            folder=root/'RL_models/pmsm/ppo/a';(folder/'iteration_0500').mkdir(parents=True)
            runner.save_json(folder/'manifest.json',dict(status='complete'))
            def fake_validate(command):
                run=command[command.index('--run')+1];controller=command[command.index('--controller')+1]
                folder=root/'outputs/rllib/pmsm/validation'/run/'stamp';folder.mkdir(parents=True)
                selected='checkpoint_'+run if controller=='ppo' else None
                record=dict(checkpoint=selected,iteration=500,complete=True,quality_pass=True,response_pass=True,mean_mae=10.)
                runner.save_json(folder/'validation.json',dict(arguments=dict(controller=controller),records=[record],selected_checkpoint=selected))
            def fake_confirm(command):
                events.append(('confirmation',command[0]))
                if command[0]=='plan':
                    name=command[command.index('--experiment')+1]
                    folder=root/'outputs/rllib/research'/name/'confirmation';folder.mkdir(parents=True)
                    jobs=[dict(id=f'{c}_{s}',controller=c,run=f'{c}_{s}') for c in ['ppo','icm'] for s in [123,234,456]]
                    runner.save_json(folder/'protocol.json',dict(jobs=jobs))
            def fake_extension(command):events.append((command[command.index('--study')+1],command[0]))
            with patch('rllib_control.validate.main',side_effect=fake_validate),\
                 patch('rllib_control.trial_results.main'),\
                 patch('rllib_control.pmsm_confirmation.main',side_effect=fake_confirm),\
                 patch('rllib_control.tdk_extension.main',side_effect=fake_extension):
                self.assertEqual(main(['--experiment','test','--candidates','a']),0)
            state=json.loads((root/'outputs/rllib/research/test/workflow/state.json').read_text())
            self.assertEqual(state['status'],'complete')
            self.assertEqual(sum(not r['passed'] for r in state['confirmation_validation']),3)
            first_test=events.index(('confirmation','evaluate'))
            for study in ['robustness','environment','reward']:
                self.assertLess(events.index((study,'plan')),first_test)
                self.assertIn((study,'all'),events)

    @patch('rllib_control.research.fingerprints',return_value={'code':'hash'})
    def test_owned_candidate_resume_preserves_settings_and_refuses_running_job(self,_):
        for status in ['running','interrupted_or_failed']:
            with tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
                root=Path(directory)
                original=runner.parser().parse_args(['train','--motor','pmsm','--run','a','--seed','345',
                    '--stability-profile','pmsm-v1','--tracking-absolute-weight','20','--current-rate-weight','.002',
                    '--train-profile','random_full'])
                settings=runner.settings(original);folder=runner.run_dir(original);folder.mkdir(parents=True)
                runner.save_json(folder/'manifest.json',dict(status=status,settings=settings))
                def fake_train(command):
                    self.assertIn('--resume',command)
                    self.assertEqual(runner.settings(runner.parser().parse_args(command)),settings)
                    (folder/'iteration_0500').mkdir()
                    runner.save_json(folder/'manifest.json',dict(status='complete',settings=settings))
                def fake_validate(command):
                    output=root/'outputs/rllib/pmsm/validation/a/stamp';output.mkdir(parents=True)
                    runner.save_json(output/'validation.json',dict(arguments=dict(controller='ppo'),records=[],selected_checkpoint=None))
                with patch('rllib_control.runner.main',side_effect=fake_train) as training,\
                     patch('rllib_control.validate.main',side_effect=fake_validate),\
                     patch('rllib_control.trial_results.main'):
                    command=['--experiment','test','--candidates','a','--train-candidates']
                    if status=='running':
                        with self.assertRaises(ValueError):main(command)
                        training.assert_not_called()
                    else:
                        self.assertEqual(main(command),2)
                        training.assert_called_once()
