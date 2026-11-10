import unittest
import torch
import numpy as np
from rllib_control.current_control import CurrentReferenceEnv
from rllib_control.runner import parser,settings,config_for,load_motor


class RewardConditioningTests(unittest.TestCase):
    def test_error_feature_gain_preserves_physics_reward_and_classical_action(self):
        from rllib_control.current_control import CurrentBaseline
        for motor,index in [('dc',4),('pmsm',6)]:
            config=dict(motor=motor,training_mode=False,domain_randomization=False)
            a=CurrentReferenceEnv(config);b=CurrentReferenceEnv(dict(config,error_observation_gain=10.))
            a.reset(seed=3);b.reset(seed=3)
            a.base.last_reference=10.;b.base.last_reference=10.
            a.base.reference.value=lambda t:10.
            b.base.reference.value=lambda t:10.
            oa=a._obs(a.base._observation());ob=b._obs(b.base._observation())
            self.assertAlmostEqual(ob[index],10*oa[index],places=6)
            pa=CurrentBaseline(motor);pb=CurrentBaseline(motor)
            np.testing.assert_allclose(pa.action(oa,a),pb.action(ob,b),atol=1e-7)
            for action in [.1,-.2,.1]:
                _,ra,_,_,ia=a.step([action]);_,rb,_,_,ib=b.step([action])
                self.assertEqual(ra,rb);self.assertEqual(ia['omega'],ib['omega'])

    def test_scale_all_costs_without_changing_physical_trajectory(self):
        cfg=dict(motor='pmsm',training_mode=False,domain_randomization=False)
        a=CurrentReferenceEnv(cfg);b=CurrentReferenceEnv(dict(cfg,reward_scale=.01))
        a.reset(seed=2);b.reset(seed=2)
        for action in [.1,.4,-.2,0.]:
            oa,ra,*_=a.step([action]);ob,rb,*_=b.step([action])
            np.testing.assert_array_equal(oa,ob)
            self.assertAlmostEqual(rb,.01*ra)
        a.base.state.iq=100.;b.base.state.iq=100.
        _,ra,ta,_,_=a.step([0.]);_,rb,tb,_,_=b.step([0.])
        self.assertTrue(ta and tb);self.assertAlmostEqual(rb,.01*ra)

    def test_config_preserves_icm_relative_reward_weight(self):
        args=parser().parse_args(['train','--motor','pmsm','--controller','icm',
            '--reward-scale','.01','--vf-clip-param','100000000','--intrinsic-reward-coeff','.005'])
        cfg=config_for(args,load_motor(args))
        self.assertEqual(cfg.env_config['reward_scale'],.01)
        self.assertEqual(cfg.vf_clip_param,1e8)
        self.assertAlmostEqual(cfg.learner_config_dict['intrinsic_reward_coeff'],.00005)
        old=parser().parse_args(['train'])
        self.assertNotIn('reward_scale',settings(old));self.assertNotIn('vf_clip_param',settings(old))

    def test_critic_squared_loss_cap_can_remove_the_gradient(self):
        prediction=torch.tensor(0.,requires_grad=True)
        loss=(prediction-torch.tensor(100.)).square().clamp(0,10.)
        loss.backward();self.assertEqual(prediction.grad.item(),0.)
        prediction=torch.tensor(0.,requires_grad=True)
        loss=(prediction-torch.tensor(1.)).square().clamp(0,1e8)
        loss.backward();self.assertNotEqual(prediction.grad.item(),0.)


if __name__=='__main__':unittest.main()
