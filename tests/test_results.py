import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from rllib_control import runner, results


class ResultsTests(unittest.TestCase):
    def setUp(self):
        workspace = self.enterContext(TemporaryDirectory())
        self.enterContext(patch.object(runner, 'ROOT', Path(workspace)))

    def create_checkpoint_manifest(self, motor, controller):
        run = results.PRESETS[motor][controller]
        args = runner.parser().parse_args(['train', '--motor', motor, '--controller', controller,
                                           '--run', run, '--iterations', '1'])
        folder = runner.run_dir(args)
        (folder/'iteration_0001').mkdir(parents=True)
        runner.save_json(folder/'manifest.json', dict(status='complete', completed_iteration=1,
                                                      settings=runner.settings(args)))
        return results.restored(motor, controller, run)

    def test_mixed_defaults(self):
        self.assertEqual(results.parser().parse_args(['--motor','dc']).profile,'mixed')
        self.assertEqual(runner.parser().parse_args(['compare']).profile,'mixed')

    def test_templates_and_training_roundtrip(self):
        for motor in ('dc','pmsm'):
            for controller in ('ppo','icm'):
                args,manifest,path=self.create_checkpoint_manifest(motor,controller)
                self.assertTrue(path.is_dir())
                args.command='train'; args.run='never_started_test'; args.dry_run=True
                restored=runner.parser().parse_args(results.training_command(args))
                self.assertEqual(runner.settings(args),runner.settings(restored))
                self.assertEqual(restored.run,'never_started_test')

    def test_pair_environment(self):
        from rllib_control.current_control import environment_config
        for motor in ('dc','pmsm'):
            configs=[environment_config(self.create_checkpoint_manifest(motor,c)[0]) for c in ('ppo','icm')]
            self.assertEqual(*configs)

    def test_baselines_run_without_checkpoints(self):
        for motor, controller in [('dc','pi'), ('pmsm','foc')]:
            with self.subTest(motor=motor):
                results.main(['--motor',motor,'--controller',controller])
                parent=runner.ROOT/'outputs'/'final_results'/motor
                report,=parent.glob('*/metrics.json')
                metrics=json.loads(report.read_text())
                self.assertEqual(set(metrics), {controller})
                self.assertTrue(metrics[controller]['completed_episode'])
                self.assertFalse(metrics[controller]['safety_violation'])
                self.assertTrue((report.parent/'tracking.png').is_file())
                self.assertFalse((parent/'latest.json').exists())
                self.assertFalse((runner.ROOT/'RL_models').exists())

    def test_missing_checkpoint_explains_baseline_fallback(self):
        with self.assertRaisesRegex(FileNotFoundError, '--controller pi'):
            results.main(['--motor','dc','--controller','ppo','--dry-run'])

    def test_no_wrong_baseline_or_implicit_training(self):
        for command in (['--motor','dc','--controller','foc'],
                        ['--motor','pmsm','--train'],
                        ['--motor','dc','--train','--controller','ppo']):
            with self.assertRaises(SystemExit): results.main(command)
