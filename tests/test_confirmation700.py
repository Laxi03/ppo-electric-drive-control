import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rllib_control import runner,research
from rllib_control.pmsm_confirmation import make_protocol
from rllib_control.confirmation700_batch import ppo_validation_passed


class Confirmation700Tests(unittest.TestCase):
    def fixture(self,root):
        checkpoint=root/'candidate/iteration_0700';checkpoint.mkdir(parents=True)
        args=runner.parser().parse_args(['train','--motor','pmsm','--stability-profile','pmsm-v1',
            '--critic-profile','normalized','--critic-return-scale','100','--seed','345'])
        settings=runner.settings(args)
        runner.save_json(checkpoint.parent/'manifest.json',dict(settings=settings,lineage=dict(
            pretraining_iterations=500,additional_training_iterations=200,total_iterations=700)))
        path=root/'qualification.json'
        runner.save_json(path,dict(arguments=dict(seeds=[501,502],profiles=['steps','random'],max_mae=30.,
            max_ripple=1.,max_speed_pp=5.,max_offset=10.,max_current_ripple_fraction=.05,max_step_rise=.24),
            training_settings=settings,selected_checkpoint=str(checkpoint),
            records=[dict(checkpoint=str(checkpoint),iteration=700,complete=True,quality_pass=True,response_pass=True)]))
        return path

    @patch('rllib_control.research.fingerprints',return_value={})
    def test_budget_must_match_qualification_and_all_six_share_schedule(self,_):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=self.fixture(root)
            with self.assertRaises(ValueError):make_protocol('x',path,[123,234,456],500)
            protocol=make_protocol('x',path,[123,234,456],700)
            self.assertEqual(protocol['training_milestones'],[100,500,700])
            self.assertEqual(len(protocol['jobs']),7)
            for job in protocol['jobs'][:-1]:
                self.assertEqual(job['settings']['critic_profile'],'normalized')
                self.assertEqual(runner.parser().parse_args(job['command']).iterations,700)
                self.assertNotIn('--resume',job['command'])

    def test_milestones_restore_without_changing_settings_and_legacy_not_affected(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(runner,'ROOT',Path(directory)):
            root=Path(directory)/'study';root.mkdir()
            command=['train','--motor','pmsm','--run','scheduletest','--iterations','700']
            args=runner.parser().parse_args(command);folder=runner.run_dir(args);settings=runner.settings(args)
            protocol=dict(iterations=700,training_milestones=[100,500,700],jobs=[dict(
                id='job',command=command,controller='ppo',settings=settings)])
            runner.save_json(root/'protocol.json',protocol)
            calls=[]
            def invoke(cmd):
                parsed=runner.parser().parse_args(cmd);calls.append((parsed.iterations,parsed.resume))
                self.assertEqual(runner.settings(parsed),settings)
                (folder/f'iteration_{parsed.iterations:04d}').mkdir(parents=True)
                runner.save_json(folder/'manifest.json',dict(settings=settings,status='complete',completed_iteration=parsed.iterations))
            with patch.object(research,'invoke',side_effect=invoke):
                research.train(protocol,root,False)
                self.assertEqual(calls,[(100,False),(500,True),(700,True)])
                research.train(protocol,root,True)
                self.assertEqual(len(calls),3)
            stored=json.loads((folder/'manifest.json').read_text())
            self.assertEqual(stored['training_protocol']['milestones'],[100,500,700])

    def test_ppo_gate_requires_all_three_fresh_seeds_but_does_not_filter_icm_failures(self):
        records=[dict(controller='ppo',seed=s,passed=True) for s in [123,234,456]]
        records.extend(dict(controller='icm',seed=s,passed=False) for s in [123,234,456])
        self.assertTrue(ppo_validation_passed(records))
        records[0]['passed']=False;self.assertFalse(ppo_validation_passed(records))
        self.assertFalse(ppo_validation_passed(records[1:]))


if __name__=='__main__':unittest.main()
