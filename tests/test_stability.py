import unittest
import numpy as np
import torch
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from rllib_control.stability import BoundedMeanPPOModule
from rllib_control.stability import SpeedObservationPPOModule
from rllib_control.current_control import CurrentReferenceEnv, environment_config
from rllib_control.runner import parser, config_for, load_motor, settings


class StabilityTests(unittest.TestCase):
    def test_validation_rejects_failed_or_oscillating_checkpoints(self):
        from rllib_control.validate import choose
        records=[dict(checkpoint='failed',complete=False,mean_mae=None,worst_ripple=None),
                 dict(checkpoint='oscillating',complete=True,mean_mae=1.,worst_ripple=10.),
                 dict(checkpoint='stable',complete=True,mean_mae=2.,worst_ripple=.1,quality_pass=True)]
        self.assertEqual(choose(records,1.),'stable')
        self.assertIsNone(choose(records[:2],1.))
        self.assertIsNone(choose([dict(checkpoint='smooth_but_wrong',complete=True,
                                      mean_mae=100.,worst_ripple=.01)],1.))

    def test_legacy_unchanged_and_profile_saved(self):
        args=parser().parse_args(['train','--motor','pmsm'])
        self.assertNotIn('stability_profile', settings(args))
        self.assertNotIn('current_rate_weight', environment_config(args))
        args.stability_profile='pmsm-v1'
        cfg=config_for(args,load_motor(args))
        self.assertEqual(cfg.gamma,.996)
        self.assertEqual(cfg.lambda_,.98)
        self.assertEqual(cfg.env_config['current_rate_weight'],.02)
        self.assertEqual(settings(args)['policy_parameterization'],'bounded_gaussian_mean_v1')

    def test_rate_cost_once_and_no_extra_cost_at_constant_request(self):
        config=dict(motor='pmsm',training_mode=False,domain_randomization=False)
        a=CurrentReferenceEnv(dict(config,current_rate_weight=.02,current_rate_timescale_s=.02))
        b=CurrentReferenceEnv(config)
        a.reset(seed=1); b.reset(seed=1)
        _,ra,_,_,info=a.step([.1]); _,rb,_,_,_=b.step([.1])
        self.assertAlmostEqual(rb-ra,.02*(.1*.02/a.control_period)**2)
        self.assertGreater(info['current_rate_penalty'],0.)
        self.assertEqual(a.step([.1])[-1]['current_rate_penalty'],0.)
        for key in ('current_rate_weight','current_rate_timescale_s'):
            with self.assertRaises(ValueError): CurrentReferenceEnv(dict(config,**{key:float('nan')}))

    def test_rate_override_is_saved_and_only_changes_reward(self):
        original=parser().parse_args(['train','--motor','pmsm','--stability-profile','pmsm-v1'])
        changed=parser().parse_args(['train','--motor','pmsm','--stability-profile','pmsm-v1',
                                    '--current-rate-weight','.002'])
        a,b=environment_config(original),environment_config(changed)
        self.assertEqual(settings(changed)['current_rate_weight'],.002)
        self.assertEqual(a.pop('current_rate_weight'),.02)
        self.assertEqual(b.pop('current_rate_weight'),.002)
        self.assertEqual(a,b)
        a=CurrentReferenceEnv(environment_config(original));b=CurrentReferenceEnv(environment_config(changed))
        a.reset(seed=501);b.reset(seed=501)
        for command in ([.5],[-.5],[.1]):
            oa,ra,ta,ua,ia=a.step(command);ob,rb,tb,ub,ib=b.step(command)
            np.testing.assert_array_equal(oa,ob)
            self.assertAlmostEqual(rb-ra,ia['current_rate_penalty']-ib['current_rate_penalty'])
        a.close();b.close()

    def test_policy_same_parameterization_and_gradients(self):
        env=CurrentReferenceEnv(dict(motor='pmsm'))
        module=RLModuleSpec(module_class=BoundedMeanPPOModule,
            observation_space=env.observation_space,action_space=env.action_space,
            model_config=dict(fcnet_hiddens=[16,8],fcnet_activation='tanh',vf_share_layers=False)).build()
        batch={Columns.OBS:torch.randn(20,15)}
        inference=module.forward_inference(batch)[Columns.ACTION_DIST_INPUTS]
        train=module.forward_train(batch)[Columns.ACTION_DIST_INPUTS]
        torch.testing.assert_close(inference,train)
        torch.testing.assert_close(module.forward_exploration(batch)[Columns.ACTION_DIST_INPUTS],train)
        self.assertTrue(torch.all(abs(train[:,:1])<=1))
        self.assertTrue(torch.all((train[:,1:]>=-4)&(train[:,1:]<=-1)))
        train.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad is not None and torch.any(p.grad!=0) for p in module.pi.parameters()))

    def test_speed_actor_mask_does_not_mask_critic_and_noise_is_consistent(self):
        env=CurrentReferenceEnv(dict(motor='pmsm'))
        module=RLModuleSpec(module_class=SpeedObservationPPOModule,
            observation_space=env.observation_space,action_space=env.action_space,
            model_config=dict(fcnet_hiddens=[16,8],fcnet_activation='tanh',vf_share_layers=False,
                              policy_log_std_min=-6.,policy_log_std_max=-2.)).build()
        obs=torch.randn(8,15,requires_grad=True);batch={Columns.OBS:obs}
        original=module.forward_train(batch)
        torch.testing.assert_close(original[Columns.ACTION_DIST_INPUTS],module.forward_inference(batch)[Columns.ACTION_DIST_INPUTS])
        torch.testing.assert_close(original[Columns.ACTION_DIST_INPUTS],module.forward_exploration(batch)[Columns.ACTION_DIST_INPUTS])
        ignored=[i for i in range(15) if i not in module.indices]
        changed=obs.detach().clone();changed[:,ignored]+=3.
        after=module.forward_train({Columns.OBS:changed})
        torch.testing.assert_close(original[Columns.ACTION_DIST_INPUTS],after[Columns.ACTION_DIST_INPUTS])
        self.assertFalse(torch.allclose(original[Columns.EMBEDDINGS],after[Columns.EMBEDDINGS]))
        logits=original[Columns.ACTION_DIST_INPUTS]
        self.assertTrue(torch.all((logits[:,1:]>=-6.)&(logits[:,1:]<=-2.)))
        logits.sum().backward()
        self.assertTrue(torch.all(obs.grad[:,ignored]==0))
        self.assertGreater(float(abs(obs.grad[:,module.indices]).sum()),0.)
        env.close()


    def test_full_range_training_reference_exposes_large_reversals_and_keeps_old_default(self):
        from sim_PMSM_RLlib_Curiosity import PMSMSpeedReference
        old_jumps=[];new_jumps=[]
        for seed in range(6000,6100):
            old=PMSMSpeedReference('random',8.,np.random.default_rng(seed))
            new=PMSMSpeedReference('random_full',8.,np.random.default_rng(seed))
            old_jumps.extend(abs(np.diff(old.values[:-1])))
            new_jumps.extend(abs(np.diff(new.values[:-1])))
            self.assertTrue(np.all(abs(new.values)<=400.))
            self.assertEqual(new.values[0],0.)
            self.assertEqual(new.values[-1],new.values[-2])
        self.assertLessEqual(max(old_jumps),320.)
        self.assertGreater(max(new_jumps),650.)
        self.assertEqual(parser().parse_args(['train']) .train_profile,'random')
        args=parser().parse_args(['train','--motor','pmsm','--stability-profile','pmsm-v1','--train-profile','random_full'])
        self.assertEqual(settings(args)['train_profile'],'random_full')
        self.assertEqual(config_for(args,load_motor(args)).env_config['profile'],'random_full')


if __name__=='__main__': unittest.main()
