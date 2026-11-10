import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rllib_control import runner
from rllib_control.pmsm_confirmation import make_protocol


class ConfirmationTests(unittest.TestCase):
    def fixture(self,root):
        checkpoint=root/'candidate/iteration_0500';checkpoint.mkdir(parents=True)
        args=runner.parser().parse_args(['train','--motor','pmsm','--controller','ppo',
            '--seed','345','--stability-profile','pmsm-v1','--tracking-absolute-weight','20',
            '--current-rate-weight','.002'])
        settings=runner.settings(args)
        (checkpoint.parent/'manifest.json').write_text(json.dumps(dict(settings=settings)))
        report=dict(arguments=dict(seeds=[501,502],profiles=['steps','random'],max_mae=30.,
            max_ripple=1.,max_speed_pp=5.,max_offset=10.,max_current_ripple_fraction=.05,max_step_rise=.24),
            training_settings=settings,selected_checkpoint=str(checkpoint),
            records=[dict(checkpoint=str(checkpoint),complete=True,quality_pass=True,response_pass=True)])
        path=root/'qualification.json';path.write_text(json.dumps(report))
        return path,report

    @patch('rllib_control.research.fingerprints',return_value={'test':'hash'})
    def test_six_fresh_matched_jobs_and_baseline(self,_):
        with tempfile.TemporaryDirectory() as folder:
            path,report=self.fixture(Path(folder))
            protocol=make_protocol('confirm',path,[123,234,456])
            self.assertEqual(len(protocol['jobs']),7)
            self.assertEqual(protocol['test_seeds'],[1901,1902,1903])
            jobs=protocol['jobs'][:-1]
            for job in jobs:
                self.assertEqual(job['settings']['current_rate_weight'],.002)
                self.assertEqual(job['settings']['tracking_absolute_weight'],20.)
                self.assertEqual(job['settings']['policy_hidden_sizes'],[16,8])
                self.assertNotIn('--resume',job['command'])
            self.assertEqual({j['settings']['intrinsic_reward_coeff'] for j in jobs if j['controller']=='icm'},{.005})
            self.assertEqual({j['settings']['intrinsic_reward_coeff'] for j in jobs if j['controller']=='ppo'},{0.})
            baseline=runner.parser().parse_args(protocol['jobs'][-1]['environment_command'])
            self.assertEqual(baseline.current_rate_weight,.002)
            self.assertEqual(baseline.tracking_absolute_weight,20.)

    @patch('rllib_control.research.fingerprints',return_value={})
    def test_rejects_unqualified_or_changed_candidate_and_reused_seed(self,_):
        with tempfile.TemporaryDirectory() as folder:
            path,report=self.fixture(Path(folder))
            with self.assertRaises(ValueError):make_protocol('x',path,[123,234,345])
            report['selected_checkpoint']=None;path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):make_protocol('x',path,[123,234,456])
            path,report=self.fixture(Path(folder)/'second')
            report['arguments']['max_offset']=20.;path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):make_protocol('x',path,[123,234,456])
            report['arguments']['max_offset']=10.;report['training_settings']['current_rate_weight']=.02
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):make_protocol('x',path,[123,234,456])

    @patch('rllib_control.research.fingerprints',return_value={})
    def test_actor_features_and_noise_settings_propagate_to_both_algorithms(self,_):
        with tempfile.TemporaryDirectory() as directory:
            path,report=self.fixture(Path(directory))
            report['training_settings'].update(actor_observations='speed',policy_log_std_min=-6.,policy_log_std_max=-2.,train_profile='random_full')
            checkpoint=Path(report['selected_checkpoint'])
            runner.save_json(checkpoint.parent/'manifest.json',dict(settings=report['training_settings']))
            runner.save_json(path,report)
            protocol=make_protocol('x',path,[123,234,456])
            for job in protocol['jobs'][:-1]:
                self.assertEqual(job['settings']['actor_observations'],'speed')
                self.assertEqual(job['settings']['policy_log_std_min'],-6.)
                self.assertEqual(job['settings']['policy_log_std_max'],-2.)
                self.assertEqual(job['settings']['train_profile'],'random_full')

    def test_summary_preserves_missing_and_failed_cases(self):
        from rllib_control.pmsm_confirmation import summarize
        from rllib_control import research
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'cases').mkdir()
            protocol=dict(jobs=[dict(id=str(s),variant='ppo',train_seed=s,controller='ppo',
                command=['train','--motor','pmsm','--run',f'unittest_missing_{s}']) for s in [1,2,3]],
                          profiles=['steps'],scenarios=['nominal'],test_seeds=[1901,1902])
            job=protocol['jobs'][0]
            runner.save_json(root/'cases'/(research.case_id(job,'steps','nominal',1901)+'.json'),
                             dict(status='failed_episode',error='test failure'))
            summarize(protocol,root)
            report=json.loads((root/'confirmation.json').read_text())
            self.assertEqual(len(report['cases']),6)
            self.assertEqual(sum(r['status']=='failed_episode' for r in report['cases']),1)
            self.assertEqual(sum(r['status']=='missing' for r in report['cases']),5)
            self.assertTrue(all(r['mean'] is None for r in report['summary']))


if __name__=='__main__':unittest.main()
