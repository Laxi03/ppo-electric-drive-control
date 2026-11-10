"""ICM with continuous inverse-action regression, not discrete classification."""
import torch
from torch import nn
from ray.rllib.core import Columns
from ray.rllib.examples.rl_modules.classes.intrinsic_curiosity_model_rlm import IntrinsicCuriosityModel


def network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,outputs))


class ContinuousICM(IntrinsicCuriosityModel):
    def setup(self):
        self.indices = self.model_config["physical_indices"]
        self.dim = self.action_space.shape[0]
        features = self.model_config.get("feature_dim",16)
        self._feature_net = network(len(self.indices),features)
        self._forward_net = network(features+self.dim,features)
        self._inverse_net = network(2*features,self.dim)

    def _actions(self,batch):
        # Same clipping and disk projection as the environment.
        action = batch[Columns.ACTIONS].float().clamp(-1,1)
        return action / torch.linalg.vector_norm(action,dim=-1,keepdim=True).clamp(min=1.)

    def _forward_train(self,batch,**kwargs):
        phi = self._feature_net(batch[Columns.OBS][...,self.indices])
        next_phi = self._feature_net(batch[Columns.NEXT_OBS][...,self.indices])
        predicted = self._forward_net(torch.cat((phi,self._actions(batch)),dim=-1))
        # Do not let the forward target move to reduce its own prediction error.
        intrinsic = .5*((predicted-next_phi.detach())**2).sum(-1)
        return {Columns.INTRINSIC_REWARDS:intrinsic,"phi":phi,"next_phi":next_phi}

    def compute_self_supervised_loss(self,*,learner,module_id,config,batch,fwd_out):
        module = learner.module[module_id].unwrapped()
        forward = fwd_out[Columns.INTRINSIC_REWARDS].mean()
        predicted = module._inverse_net(torch.cat((fwd_out["phi"],fwd_out["next_phi"]),dim=-1))
        inverse = ((predicted-module._actions(batch))**2).mean()
        weight = config.learner_config_dict["forward_loss_weight"]
        learner.metrics.log_dict(dict(mean_intrinsic_rewards=forward,forward_loss=forward,inverse_loss=inverse),key=module_id,window=1)
        return weight*forward+(1-weight)*inverse


def coefficient_at(args, iteration):
    if args.controller != "icm":
        return 0.
    end = args.intrinsic_reward_coeff if args.intrinsic_final_coeff is None else args.intrinsic_final_coeff
    fraction = min(1., max(0., (iteration-1)/max(1,args.intrinsic_decay_iterations-1)))
    return args.intrinsic_reward_coeff+(end-args.intrinsic_reward_coeff)*fraction


def set_coefficient(learner, coefficient):
    from ray.rllib.examples.learners.classes.intrinsic_curiosity_learners import IntrinsicCuriosityModelConnector
    matches = 0
    for connector in learner._learner_connector.connectors:
        if isinstance(connector,IntrinsicCuriosityModelConnector):
            connector.intrinsic_reward_coeff = coefficient
            matches += 1
    if matches != 1:
        raise RuntimeError("Expected exactly one curiosity reward connector")
    return coefficient
