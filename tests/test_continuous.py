import unittest
from types import SimpleNamespace
import numpy as np
from rllib_control.continuous import ContinuousVoltageEnv
from rllib_control.ripple import plateau_metrics,command_metrics
from rllib_control.runner import parser,config_for,load_motor


class ContinuousTests(unittest.TestCase):
    def test_spaces_limits_and_reset(self):
        for motor,dim,size in (("dc",1,8),("pmsm",2,13)):
            env=ContinuousVoltageEnv(dict(motor=motor,episode_length=.03,training_mode=False,
                command_slew_v_per_s=100.,command_smoothness_weight=.05))
            obs,_=env.reset(seed=1)
            self.assertEqual(obs.shape,(size,))
            previous=np.zeros(dim)
            for action in (np.ones(dim),-np.ones(dim),np.zeros(dim)):
                obs,_,_,_,info=env.step(action)
                self.assertTrue(env.observation_space.contains(obs))
                self.assertLessEqual(np.linalg.norm(info['limited_voltage']-previous),100*env.control_period+1e-9)
                self.assertLessEqual(np.linalg.norm(info['requested_voltage']),env.vmax+1e-9)
                previous=info['limited_voltage'].copy()
            env.reset(seed=1)
            np.testing.assert_array_equal(env.requested,np.zeros(dim))
            with self.assertRaises(ValueError):
                env.step(np.full(dim,np.nan))

    def test_penalty_once_per_decision(self):
        for motor,dim in (("dc",1),("pmsm",2)):
            common=dict(motor=motor,episode_length=.02,training_mode=False,command_slew_v_per_s=1000.)
            first=ContinuousVoltageEnv(dict(common,command_smoothness_weight=.1))
            second=ContinuousVoltageEnv(dict(common,command_smoothness_weight=0.))
            first.reset(seed=2);second.reset(seed=2)
            action=np.full(dim,.2)
            _,r1,_,_,info=first.step(action)
            _,r2,_,_,_=second.step(action)
            self.assertAlmostEqual(r2-r1,.1*dim*.2**2)
            _,_,_,_,info=first.step(action)
            self.assertAlmostEqual(info['command_smoothness_penalty'],0.)

    def test_measurements(self):
        data=np.zeros((100,10));data[:,0]=np.arange(1,101)*.01
        data[:,1]=10.;data[:,2]=8.+np.tile([-1.,1.],50)
        data[:,3]=np.tile([1.,3.],50);data[:,6]=data[:,3]
        row=plateau_metrics(data,.4)[0]
        self.assertAlmostEqual(row['steady_offset_rad_s'],2.)
        self.assertAlmostEqual(row['speed_peak_to_peak'],2.)
        self.assertAlmostEqual(row['speed_ripple_RMS'],1.)
        self.assertAlmostEqual(row['current_magnitude_ripple_RMS'],1.)
        command=np.column_stack((np.arange(1,101)*.1,np.zeros(100)))
        metrics=command_metrics(data[:,0],command,command,command)
        self.assertAlmostEqual(metrics['limited_max_slew_V_per_s'],10.)
        self.assertAlmostEqual(metrics['requested_command_change_rate_per_s'],100.)

    def test_continuous_icm_gradient(self):
        import torch
        from ray.rllib.core import Columns
        from rllib_control.continuous_icm import ContinuousICM
        env=ContinuousVoltageEnv(dict(motor="pmsm"))
        module=ContinuousICM(observation_space=env.observation_space,action_space=env.action_space,
                model_config=dict(physical_indices=[0,1,2,3,4,8],feature_dim=16))
        batch={Columns.OBS:torch.randn(8,13),Columns.NEXT_OBS:torch.randn(8,13),Columns.ACTIONS:torch.randn(8,2)}
        out=module.forward_train(batch)
        self.assertEqual(out[Columns.INTRINSIC_REWARDS].shape,(8,))
        out[Columns.INTRINSIC_REWARDS].mean().backward()
        self.assertTrue(any(p.grad is not None for p in module.parameters()))

    def test_decay_resume_formula(self):
        from rllib_control.continuous_icm import coefficient_at
        args=SimpleNamespace(controller="icm",intrinsic_reward_coeff=.02,intrinsic_final_coeff=.001,intrinsic_decay_iterations=1000)
        self.assertAlmostEqual(coefficient_at(args,1),.02)
        self.assertAlmostEqual(coefficient_at(args,1000),.001)
        self.assertAlmostEqual(coefficient_at(args,1200),.001)

    def test_ppo_has_no_icm_module(self):
        args=parser().parse_args(['train','--action-mode','continuous'])
        config=config_for(args,load_motor(args))
        self.assertEqual(len(config.rl_module_spec.rl_module_specs),1)


if __name__=='__main__':
    unittest.main()
