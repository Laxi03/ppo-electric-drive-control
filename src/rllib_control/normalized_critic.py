"""Opt-in numerical critic repair; PPO actor objective and SI rewards unchanged.

Values and bootstrap predictions remain in original reward units. Only the
critic regression residual is normalized. Separate optimizers prevent critic
gradient clipping from suppressing actor gradients. No teacher/controller law.
"""
import copy
import torch
from ray.rllib.core import DEFAULT_MODULE_ID, Columns
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.examples.learners.classes.intrinsic_curiosity_learners import add_intrinsic_curiosity_connectors


def robust_value_loss(prediction,target,scale,mask=None):
    """Twice Huber: squared error near zero, nonzero bounded slope outside."""
    residual=(prediction-target)/scale
    loss=torch.where(residual.abs()<=1.,residual.square(),2.*residual.abs()-1.)
    if mask is not None:loss=loss[mask]
    return loss.mean()


def actor_only_config(config):
    # RLlib freezes live configs. Override only an independent shallow copy;
    # retain the global config, its freeze status, and all other PPO settings.
    result=copy.copy(config)
    object.__setattr__(result,'vf_loss_coeff',0.)
    return result


class NormalizedCriticLearner(PPOTorchLearner):
    def configure_optimizers_for_module(self,module_id,config=None):
        if module_id!=DEFAULT_MODULE_ID:
            return super().configure_optimizers_for_module(module_id,config)
        module=self.module[module_id].unwrapped()
        actor=list(module.encoder.actor_encoder.parameters())+list(module.pi.parameters())
        critic=list(module.encoder.critic_encoder.parameters())+list(module.vf.parameters())
        if set(map(id,actor))&set(map(id,critic)):
            raise ValueError('The normalized critic requires genuinely separate actor/critic parameters')
        if set(map(id,actor+critic))!=set(map(id,module.parameters())):
            raise ValueError('Unaccounted trainable parameters in normalized critic module')
        for name,params,lr in [('actor',actor,config.lr),
                               ('critic',critic,module.model_config['critic_learning_rate'])]:
            self.register_optimizer(module_id=module_id,optimizer_name=name,
                optimizer=torch.optim.Adam(params),params=params,lr_or_lr_schedule=lr)

    def compute_loss_for_module(self,*,module_id,config,batch,fwd_out):
        # Delegate the complete PPO surrogate, entropy, KL, masking and logging
        # to installed RLlib, but remove its gradient-killing squared-loss cap.
        actor_config=actor_only_config(config)
        actor_loss=super().compute_loss_for_module(module_id=module_id,config=actor_config,
                                                  batch=batch,fwd_out=fwd_out)
        module=self.module[module_id].unwrapped()
        prediction=module.compute_values(batch,embeddings=fwd_out.get(Columns.EMBEDDINGS))
        target=batch[Postprocessing.VALUE_TARGETS]
        scale=module.model_config['critic_return_scale']
        mask=batch.get(Columns.LOSS_MASK)
        value_loss=robust_value_loss(prediction,target,scale,mask)
        valid_prediction=prediction if mask is None else prediction[mask]
        valid_target=target if mask is None else target[mask]
        self.metrics.log_dict(dict(vf_loss=value_loss,vf_normalized_loss=value_loss,
            value_target_std=valid_target.std(unbiased=False),
            value_prediction_std=valid_prediction.std(unbiased=False),
            value_target_mean=valid_target.mean(),value_prediction_mean=valid_prediction.mean()),
            key=module_id,window=1)
        return actor_loss+config.vf_loss_coeff*value_loss


class NormalizedCriticCuriosityLearner(NormalizedCriticLearner):
    def build(self):
        super().build()
        add_intrinsic_curiosity_connectors(self)
