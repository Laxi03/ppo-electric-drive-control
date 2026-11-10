import unittest
import numpy as np
from rllib_control.current_control import CurrentReferenceEnv, CurrentBaseline
from rllib_control.runner import parser, config_for, load_motor


class CurrentControlTests(unittest.TestCase):
    def test_inner_loop_limits_memory_reset(self):
        for motor,size in (("dc",10),("pmsm",15)):
            env=CurrentReferenceEnv(dict(motor=motor,episode_length=.1,training_mode=False,
                domain_randomization=False,command_slew_v_per_s=500.))
            obs,_=env.reset(seed=12)
            self.assertEqual(obs.shape,(size,))
            self.assertEqual(env.action_space.shape,(1,))
            previous=np.zeros(env.dim)
            for a in (1.,-1.,.5,0.):
                obs,reward,_,_,info=env.step([a])
                self.assertTrue(env.observation_space.contains(obs))
                self.assertTrue(np.isfinite(reward))
                self.assertLessEqual(abs(env.current_reference),env.reference_limit)
                self.assertLessEqual(np.linalg.norm(env.limited-previous),500*env.control_period+1e-8)
                self.assertLessEqual(np.linalg.norm(env.requested),env.vmax+1e-8)
                previous=env.limited.copy()
            env.reset(seed=12)
            np.testing.assert_array_equal(env.integral,np.zeros(env.dim))
            self.assertEqual(env.current_reference,0.)
            for bad in ([float('nan')],[1,2]):
                with self.assertRaises(ValueError):env.step(bad)

    def test_inner_loop_runs_at_physical_rate(self):
        env=CurrentReferenceEnv(dict(motor='dc',training_mode=False))
        env.reset(seed=1)
        original=env.base.current_control_callback
        calls=[]
        def counted():
            calls.append(1)
            return original()
        env.base.current_control_callback=counted
        env.step([.1])
        self.assertEqual(len(calls),env.base.frame_skip)

    def test_reward_penalty_once(self):
        config=dict(motor='pmsm',training_mode=False,domain_randomization=False)
        a=CurrentReferenceEnv(dict(config,command_smoothness_weight=.1))
        b=CurrentReferenceEnv(dict(config,command_smoothness_weight=0.))
        a.reset(seed=2);b.reset(seed=2)
        _,ra,_,_,_=a.step([.2]);_,rb,_,_,_=b.step([.2])
        self.assertAlmostEqual(rb-ra,.1*.2**2)
        self.assertEqual(a.step([.2])[-1]['command_smoothness_penalty'],0.)

    def test_current_architecture_config(self):
        for motor in ('dc','pmsm'):
            args=parser().parse_args(['train','--motor',motor])
            cfg=config_for(args,load_motor(args))
            self.assertEqual(cfg.env,f'{motor}_current_reference_v2')
            self.assertEqual(cfg.env_config['reference_scale'],1.)
            self.assertEqual(cfg.num_env_runners,2)
            self.assertEqual(len(cfg.rl_module_spec.rl_module_specs),1)

    def test_failure_cannot_avoid_remaining_cost(self):
        env=CurrentReferenceEnv(dict(motor='dc',episode_length=1.,training_mode=False))
        env.reset(seed=1)
        env.base.state.current=100.
        _,reward,terminated,_,_=env.step([0.])
        self.assertTrue(terminated)
        self.assertLess(reward,-env.base.max_steps)

    def test_baseline_zero_error_no_windup(self):
        env=CurrentReferenceEnv(dict(motor='dc',training_mode=False,profile='steps'))
        obs,_=env.reset(seed=1)
        pi=CurrentBaseline('dc')
        env.base.last_reference=1e6
        for _ in range(100):pi.action(obs,env)
        self.assertEqual(pi.integral,0.)


if __name__=='__main__':unittest.main()
