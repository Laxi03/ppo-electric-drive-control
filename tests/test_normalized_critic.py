import unittest
from types import SimpleNamespace
import torch
from ray.rllib.core import Columns,DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from rllib_control.stability import BoundedMeanPPOModule
from rllib_control.current_control import CurrentReferenceEnv,environment_config
from rllib_control.normalized_critic import robust_value_loss,NormalizedCriticLearner,actor_only_config
from rllib_control import runner
from rllib_control.pmsm_confirmation import matched_command


class NormalizedCriticTests(unittest.TestCase):
    def model(self,profile='legacy'):
        env=CurrentReferenceEnv(dict(motor='pmsm'))
        config=dict(fcnet_hiddens=[16,8],fcnet_activation='tanh',vf_share_layers=False)
        if profile=='normalized':config.update(critic_profile=profile,critic_hidden_sizes=[64,64],
            critic_return_scale=100.,critic_learning_rate=1e-3)
        torch.manual_seed(6345)
        return RLModuleSpec(module_class=BoundedMeanPPOModule,observation_space=env.observation_space,
                            action_space=env.action_space,model_config=config).build()

    def test_actor_initialization_and_outputs_are_exactly_preserved(self):
        old=self.model();new=self.model('normalized')
        for prefix in ['encoder.actor_encoder','pi']:
            a={k:v for k,v in old.state_dict().items() if k.startswith(prefix)}
            b={k:v for k,v in new.state_dict().items() if k.startswith(prefix)}
            self.assertEqual(set(a),set(b))
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        batch={Columns.OBS:torch.randn(20,15)}
        torch.testing.assert_close(old.forward_train(batch)[Columns.ACTION_DIST_INPUTS],
                                   new.forward_train(batch)[Columns.ACTION_DIST_INPUTS],rtol=0,atol=0)

    def test_inference_only_worker_prunes_critic_and_keeps_actor_output(self):
        model=self.model('normalized')
        inference=RLModuleSpec(module_class=BoundedMeanPPOModule,
            observation_space=model.observation_space,action_space=model.action_space,
            model_config=model.model_config,inference_only=True).build()
        state={k:v for k,v in model.state_dict().items() if k in inference.state_dict()}
        inference.set_state(state)
        self.assertFalse(hasattr(inference,'vf'))
        batch={Columns.OBS:torch.randn(12,15)}
        torch.testing.assert_close(inference.forward_inference(batch)[Columns.ACTION_DIST_INPUTS],
                                   model.forward_inference(batch)[Columns.ACTION_DIST_INPUTS])

    def test_frozen_global_ppo_config_is_not_mutated(self):
        from ray.rllib.algorithms.ppo import PPOConfig
        original=PPOConfig().training(vf_loss_coeff=.5);original.freeze()
        copied=actor_only_config(original)
        self.assertEqual(original.vf_loss_coeff,.5);self.assertEqual(copied.vf_loss_coeff,0.)
        self.assertTrue(original._is_frozen);self.assertTrue(copied._is_frozen)

    def test_repaired_loss_preserves_the_exact_ppo_actor_gradient(self):
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
        from ray.rllib.evaluation.postprocessing import Postprocessing
        model=self.model('normalized');obs=torch.randn(24,15)
        fwd=model.forward_train({Columns.OBS:obs});logits=fwd[Columns.ACTION_DIST_INPUTS]
        dist=model.get_train_action_dist_cls().from_logits(logits);actions=dist.sample().detach()
        batch={Columns.OBS:obs,Columns.ACTIONS:actions,Columns.ACTION_DIST_INPUTS:logits.detach(),
            Columns.ACTION_LOGP:dist.logp(actions).detach(),Postprocessing.ADVANTAGES:torch.randn(24),
            Postprocessing.VALUE_TARGETS:torch.randn(24)*1e6}
        class LossOnlyLearner(NormalizedCriticLearner):
            def __init__(self):pass
        fake=LossOnlyLearner()
        fake._module={DEFAULT_MODULE_ID:SimpleNamespace(unwrapped=lambda:model)}
        fake.entropy_coeff_schedulers_per_module={DEFAULT_MODULE_ID:SimpleNamespace(get_current_value=lambda:0.)}
        fake.curr_kl_coeffs_per_module={DEFAULT_MODULE_ID:torch.tensor(.2)}
        fake.metrics=SimpleNamespace(log_dict=lambda *a,**k:None)
        cfg=PPOConfig().training(vf_loss_coeff=.5)
        original=PPOTorchLearner.compute_loss_for_module(fake,module_id=DEFAULT_MODULE_ID,
            config=actor_only_config(cfg),batch=batch,fwd_out=fwd)
        original.backward()
        params=list(model.encoder.actor_encoder.parameters())+list(model.pi.parameters())
        gradients=[p.grad.clone() for p in params]
        model.zero_grad()
        corrected=NormalizedCriticLearner.compute_loss_for_module(fake,module_id=DEFAULT_MODULE_ID,
            config=cfg,batch=batch,fwd_out=model.forward_train({Columns.OBS:obs}))
        corrected.backward()
        for before,after in zip(gradients,params):torch.testing.assert_close(before,after.grad,rtol=0,atol=0)

    def test_value_units_and_gradients_are_separate_from_actor(self):
        new=self.model('normalized');batch={Columns.OBS:torch.randn(20,15)}
        embeddings=new.encoder.critic_encoder(batch)['encoder_out']
        value=new.compute_values(batch)
        torch.testing.assert_close(value,100.*new.vf(embeddings).squeeze(-1))
        value.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in new.pi.parameters()))
        self.assertTrue(all(p.grad is None for p in new.encoder.actor_encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in new.vf.parameters()))

    def test_large_errors_keep_bounded_nonzero_gradients_and_padding_is_ignored(self):
        prediction=torch.tensor([0.,0.,0.],requires_grad=True)
        target=torch.tensor([50.,1e6,-1e6])
        loss=robust_value_loss(prediction,target,100.,torch.tensor([True,True,False]))
        loss.backward()
        self.assertNotEqual(prediction.grad[0].item(),0.)
        self.assertNotEqual(prediction.grad[1].item(),0.)
        self.assertLessEqual(abs(prediction.grad[1].item()),.0100001)
        self.assertEqual(prediction.grad[2].item(),0.)
        a=torch.tensor([20.,-30.]);b=torch.tensor([0.,10.])
        torch.testing.assert_close(robust_value_loss(a,b,100.),((a-b)/100.).square().mean())

    def test_optimizers_cover_disjoint_actor_critic_parameters(self):
        model=self.model('normalized');registrations=[]
        fake=SimpleNamespace(module={DEFAULT_MODULE_ID:SimpleNamespace(unwrapped=lambda:model)},
            register_optimizer=lambda **kw:registrations.append(kw))
        NormalizedCriticLearner.configure_optimizers_for_module(fake,DEFAULT_MODULE_ID,SimpleNamespace(lr=1e-4))
        self.assertEqual([x['optimizer_name'] for x in registrations],['actor','critic'])
        a,b=[set(map(id,x['params'])) for x in registrations]
        self.assertFalse(a&b);self.assertEqual(a|b,set(map(id,model.parameters())))
        self.assertEqual([x['lr_or_lr_schedule'] for x in registrations],[1e-4,1e-3])

    def test_matched_ppo_icm_settings_and_reward_environment_unchanged(self):
        for controller in ['ppo','icm']:
            args=runner.parser().parse_args(['train','--motor','pmsm','--controller',controller,
                '--stability-profile','pmsm-v1','--critic-profile','normalized','--critic-return-scale','100'])
            baseline=runner.parser().parse_args(['train','--motor','pmsm','--controller',controller,
                '--stability-profile','pmsm-v1'])
            self.assertEqual(environment_config(args),environment_config(baseline))
            config=runner.config_for(args,runner.load_motor(args))
            self.assertIn('NormalizedCritic',config.learner_class.__name__)
            old=runner.settings(baseline);self.assertNotIn('critic_profile',old)
            settings=runner.settings(args)
            copied=runner.parser().parse_args(matched_command(settings,'copy',controller,123,500,2))
            self.assertEqual(runner.settings(copied)['critic_return_scale'],100.)
            self.assertEqual(runner.settings(copied)['critic_hidden_sizes'],[64,64])

    def test_long_training_gate_rejects_flat_critic_or_unsafe_bad_controller(self):
        from rllib_control.critic_repair import learning_health
        history=[dict(vf_explained_var=.5,value_prediction_std=50.,value_target_std=60.,
                      vf_normalized_loss=.1) for _ in range(20)]
        metrics=dict(completed_episode=True,safety_violation=False,time_above_current_scale_s=0.)
        record=dict(mean_mae=40.,cases=[dict(metrics=dict(metrics)) for _ in range(4)])
        self.assertTrue(learning_health(history,record)['passed'])
        self.assertFalse(learning_health(history[:10],record)['passed'])
        flat=[dict(h,vf_explained_var=0.) for h in history]
        self.assertFalse(learning_health(flat,record)['passed'])
        record['mean_mae']=100.;self.assertFalse(learning_health(history,record)['passed'])
        record['mean_mae']=40.;record['cases'][0]['metrics']['safety_violation']=True
        self.assertFalse(learning_health(history,record)['passed'])


if __name__=='__main__':unittest.main()
