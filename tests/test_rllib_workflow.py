"""Fast regression tests, no Ray service or long training required."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rllib_control import runner


class WorkflowTests(unittest.TestCase):
    def args(self, *extra):
        return runner.parser().parse_args(["train", *extra])

    def test_shared_defaults(self):
        args = self.args()
        self.assertEqual(args.iterations, 1000)
        self.assertEqual(args.policy_hidden_sizes, [16, 8])
        self.assertEqual(args.train_reference_scale, 1.)
        self.assertEqual(args.action_mode, "current")

    def test_four_separate_paths(self):
        paths = {runner.run_dir(self.args("--motor", motor, "--controller", controller))
                 for motor in ("dc", "pmsm") for controller in ("ppo", "icm")}
        self.assertEqual(len(paths), 4)

    def test_reject_path_traversal(self):
        with self.assertRaises(ValueError):
            runner.run_dir(self.args("--run", "../other"))

    def test_latest_numerical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("iteration_9999", "iteration_10000", ".iteration_10001_pending"):
                (root / name).mkdir()
            self.assertEqual(runner.latest(root).name, "iteration_10000")

    def test_existing_run_protected_before_loading_ray(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "ROOT", Path(directory)):
            args = self.args()
            folder = runner.run_dir(args)
            folder.mkdir(parents=True)
            (folder / "existing").touch()
            with self.assertRaises(FileExistsError):
                runner.train(args)

    def test_resume_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "ROOT", Path(directory)):
            args = self.args("--resume")
            folder = runner.run_dir(args)
            folder.mkdir(parents=True)
            values = runner.settings(args)
            values["seed"] = 999
            runner.save_json(folder / "manifest.json", {"settings": values})
            with self.assertRaisesRegex(ValueError, "differs"):
                runner.train(args)

    def test_baselines_and_common_metrics(self):
        from rllib_control.evaluation import episode
        for motor in ("dc", "pmsm"):
            args = self.args("--motor", motor, "--episode-length", ".02")
            module = runner.load_motor(args)
            first = episode(args, module)
            second = episode(args, module)
            import numpy as np
            np.testing.assert_array_equal(first["data"], second["data"])
            self.assertGreater(first["metrics"]["evaluation_steps"], 0)
            self.assertTrue(np.isfinite(first["data"]).all())
            data = first["data"]
            dt = np.diff(np.r_[0, data[:,0]])
            expected = np.sum(np.abs(data[:,1]-data[:,2])*dt)
            self.assertAlmostEqual(first["metrics"]["IAE_speed"], expected)

    def test_equal_icm_capacity(self):
        for motor in ("dc", "pmsm"):
            args = self.args("--motor", motor, "--action-mode", "discrete")
            module = runner.load_motor(args)
            config = runner.config_for(args, module)
            model = config.rl_module_spec.rl_module_specs[module.ICM_MODULE_ID].model_config
            self.assertEqual(model["forward_net_hiddens"], (64, 64))
            self.assertEqual(config.learner_config_dict["intrinsic_reward_coeff"], 0)


if __name__ == "__main__":
    unittest.main()
