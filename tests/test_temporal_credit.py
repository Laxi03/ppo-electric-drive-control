import math
import unittest
import numpy as np
import torch
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from rllib_control import runner
from rllib_control.stability import temporal_parameters, SpeedMemoryPPOModule
from rllib_control.current_control import CurrentReferenceEnv, environment_config
from rllib_control.pmsm_confirmation import matched_command


class TemporalCreditTests(unittest.TestCase):
    def args(self, extra=()):
        return runner.parser().parse_args(['train','--motor','pmsm','--stability-profile','pmsm-v1',*extra])

    def test_horizons_are_physical_efolding_times_and_saved(self):
        args=self.args(['--discount-horizon-s','1','--advantage-horizon-s','.2'])
        gamma,lam=temporal_parameters(args)
        self.assertAlmostEqual(gamma**1000,math.exp(-1.))
        self.assertAlmostEqual((gamma*lam)**200,math.exp(-1.))
        config=runner.config_for(args,runner.load_motor(args))
        self.assertEqual(config.gamma,gamma);self.assertEqual(config.lambda_,lam)
        settings=runner.settings(args)
        self.assertEqual(settings['discount_horizon_s'],1.)
        self.assertEqual(settings['advantage_horizon_s'],.2)
        copied=runner.parser().parse_args(matched_command(settings,'test','icm',123,500,2))
        self.assertEqual(temporal_parameters(copied),(gamma,lam))
        self.assertEqual(environment_config(args),environment_config(self.args()))

    def test_legacy_exact_and_invalid_horizons_rejected(self):
        args=self.args()
        self.assertEqual(temporal_parameters(args),(.996,.98))
        self.assertNotIn('discount_horizon_s',runner.settings(args))
        for discount,advantage in [(None,.2),(1.,None),(-1.,.2),(1.,2.),(float('nan'),.2),(1.,0.)]:
            args.discount_horizon_s=discount;args.advantage_horizon_s=advantage
            with self.assertRaises(ValueError):temporal_parameters(args)

    def test_memory_actor_keeps_commands_masks_fast_inputs_and_preserves_critic(self):
        env=CurrentReferenceEnv(dict(motor='pmsm'))
        model=RLModuleSpec(module_class=SpeedMemoryPPOModule,observation_space=env.observation_space,
            action_space=env.action_space,model_config=dict(fcnet_hiddens=[16,8],
            fcnet_activation='tanh',vf_share_layers=False)).build()
        obs=torch.randn(12,15,requires_grad=True)
        first=model.forward_train({Columns.OBS:obs})
        ignored=[i for i in range(15) if i not in model.indices]
        other=obs.detach().clone();other[:,ignored]+=2.
        second=model.forward_train({Columns.OBS:other})
        torch.testing.assert_close(first[Columns.ACTION_DIST_INPUTS],second[Columns.ACTION_DIST_INPUTS])
        self.assertFalse(torch.allclose(first[Columns.EMBEDDINGS],second[Columns.EMBEDDINGS]))
        logits=first[Columns.ACTION_DIST_INPUTS]
        torch.testing.assert_close(logits,model.forward_inference({Columns.OBS:obs})[Columns.ACTION_DIST_INPUTS])
        torch.testing.assert_close(logits,model.forward_exploration({Columns.OBS:obs})[Columns.ACTION_DIST_INPUTS])
        logits[:,0].sum().backward()
        np.testing.assert_array_equal(obs.grad[:,ignored].numpy(),0.)
        self.assertGreater(float(obs.grad[:,9:11].abs().sum()),0.)
        env.close()

    def test_repair_plan_changes_only_declared_training_factors(self):
        from rllib_control.pmsm_repair import commands
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        old=runner.settings(self.args(['--seed','345','--num-env-runners','2']))
        with TemporaryDirectory() as directory:
            source=Path(directory)/'manifest.json'
            source.write_text(json.dumps({'settings':old}))
            _,jobs=commands(source)
        self.assertEqual(len(jobs),2)
        for job in jobs:
            changed={k for k in old.keys()|job['settings'].keys() if old.get(k)!=job['settings'].get(k)}
            self.assertLessEqual(changed,{'gamma','gae_lambda','discount_horizon_s','advantage_horizon_s','actor_observations'})
            self.assertEqual(job['settings']['seed'],345)
            self.assertNotIn('--resume',job['command'])
        first,second=[job['settings'] for job in jobs]
        self.assertEqual({k for k in first.keys()|second.keys() if first.get(k)!=second.get(k)},
                         {'actor_observations'})


if __name__=='__main__':unittest.main()
