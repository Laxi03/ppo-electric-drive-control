"""Opt-in PMSM training protocol. Never retrofit it onto an old checkpoint."""
import math
import torch
from ray.rllib.core import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule


class BoundedMeanPPOModule(DefaultPPOTorchRLModule):
    """Bound the Gaussian mean, NOT sampled actions or their PPO likelihoods.

    Exploration is still Gaussian and the environment still clips samples. This
    is not a squashed-Gaussian distribution. Train/explore/inference use exactly
    the same parameterization so likelihood ratios remain consistent.
    """
    def setup(self):
        super().setup()
        if self.model_config.get('critic_profile')=='normalized' and not self.inference_only:
            from ray.rllib.core.models.configs import MLPEncoderConfig, MLPHeadConfig
            dims=self.model_config['critic_hidden_sizes']
            # Actor is built first by the original catalog and is not replaced.
            # Workers may prune the separate critic as usual for inference.
            self.encoder.critic_encoder=MLPEncoderConfig(input_dims=self.observation_space.shape,
                hidden_layer_dims=dims,hidden_layer_activation='silu').build('torch')
            self.vf=MLPHeadConfig(input_dims=(dims[-1],),hidden_layer_dims=[],output_layer_dim=1).build('torch')

    def compute_values(self,batch,embeddings=None):
        values=super().compute_values(batch,embeddings=embeddings)
        return values*self.model_config.get('critic_return_scale',1.)

    def bound(self,output):
        mean, log_std = output[Columns.ACTION_DIST_INPUTS].chunk(2, dim=-1)
        lower=self.model_config.get('policy_log_std_min',-4.)
        upper=self.model_config.get('policy_log_std_max',-1.)
        # Smooth scale bound avoids a dead gradient at an initial log_std of 0.
        output[Columns.ACTION_DIST_INPUTS] = torch.cat(
            (mean.tanh(), lower+(upper-lower)*log_std.sigmoid()), dim=-1)
        return output

    def _forward(self, batch, **kwargs):
        return self.bound(super()._forward(batch, **kwargs))

    def _forward_train(self, batch, **kwargs):
        return self.bound(super()._forward_train(batch, **kwargs))


class SpeedObservationPPOModule(BoundedMeanPPOModule):
    """Pure learned speed policy; critic keeps all 15 Markov-state features.

    Actor retains speed, reference, speed error, error integral and load from
    the existing observation. Fast current/voltage/command-memory channels are
    masked identically during learning, exploration and deterministic inference.
    No analytical feedback law, extra PI, or imitation target is introduced.
    """
    indices=(0,5,6,7,8)

    def _actor(self,batch):
        masked=dict(batch);obs=torch.zeros_like(batch[Columns.OBS])
        obs[...,self.indices]=batch[Columns.OBS][...,self.indices]
        masked[Columns.OBS]=obs
        embeddings=self.encoder.actor_encoder(masked)[ENCODER_OUT]
        return self.bound({Columns.ACTION_DIST_INPUTS:self.pi(embeddings)})

    def _forward(self,batch,**kwargs):return self._actor(batch)

    def _forward_train(self,batch,**kwargs):
        output=self._actor(batch)
        output[Columns.EMBEDDINGS]=self.encoder.critic_encoder(batch)[ENCODER_OUT]
        return output


class SpeedMemoryPPOModule(SpeedObservationPPOModule):
    """Learned speed policy with command memory, without fast inner-loop feedback.

    Unlike the speed-only pilot, actuator memory needed by the slew limiter and
    action-change cost is retained. The critic still sees every original input.
    """
    indices=(0,5,6,7,8,9,10)


def temporal_parameters(args):
    """Optional physical-time horizons, with exp(-t/tau) decay conventions.

    GAE weights decay as (gamma*lambda)**k, not lambda**k alone.
    Legacy numerical defaults must remain exactly unchanged for old manifests.
    """
    discount=getattr(args,'discount_horizon_s',None)
    advantage=getattr(args,'advantage_horizon_s',None)
    if discount is None and advantage is None:
        return .996,.98
    if (discount is None or advantage is None or not math.isfinite(discount)
            or not math.isfinite(advantage) or not 0<advantage<=discount):
        raise ValueError('Both horizons required: 0 < advantage horizon <= discount horizon')
    dt=.001  # pmsm-v1: 20 physical ticks of 50 microseconds; no plant change.
    gamma=math.exp(-dt/discount)
    gae_lambda=math.exp(-dt/advantage)/gamma
    if not 0<gamma<1 or not 0<gae_lambda<=1:
        raise ValueError('Horizons are outside the supported floating-point range')
    return gamma,gae_lambda


def enabled(args):
    return getattr(args, "stability_profile", "legacy") == "pmsm-v1"


def environment_fields(args):
    if not enabled(args):
        return {}
    weight=getattr(args,'current_rate_weight',None)
    return dict(stability_profile="pmsm-v1", current_rate_weight=.02 if weight is None else weight,
                current_rate_timescale_s=.02)


def training_fields(args):
    if not enabled(args):
        return {}
    gamma,gae_lambda=temporal_parameters(args)
    fields=dict(policy_parameterization="bounded_gaussian_mean_v1",
                gamma=gamma, gae_lambda=gae_lambda, learning_rate=1e-4,
                entropy_coefficient=0.0, **environment_fields(args))
    if getattr(args,'discount_horizon_s',None) is not None:
        fields.update(discount_horizon_s=args.discount_horizon_s,
                      advantage_horizon_s=args.advantage_horizon_s)
    return fields
