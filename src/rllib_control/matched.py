"""New paired environments with identical physical cost and voltage-slew penalty.

Architecture is a bundle (action semantics, memory, inner current loop and its
reference limits); this does not identify an isolated PI-gain causal effect.
"""
import gymnasium as gym
import numpy as np
from .current_control import CurrentReferenceEnv
from .continuous import ContinuousVoltageEnv

ENV_NAMES={mode:{m:f"{m}_{mode.replace('-','_')}_v3" for m in ("dc","pmsm")}
           for mode in ("matched-current","matched-voltage")}


class MatchedCurrentEnv(CurrentReferenceEnv):
    def step(self, action):
        old=self.limited.copy()
        weight=self.smoothness_weight
        self.smoothness_weight=0.
        try:
            obs,reward,terminated,truncated,info=super().step(action)
        finally:
            self.smoothness_weight=weight
        penalty=weight*float(np.sum(((self.limited-old)/self.vmax)**2))
        if terminated:
            bound=9*self.tracking_weight+3*self.absolute_weight+4*(self.current_weight+self.id_weight+weight)+1
            reward=-bound*max(1,self.base.max_steps-self.base.step_count+1)
        else:
            reward-=penalty
        info["command_smoothness_penalty"]=penalty
        return obs,float(reward),terminated,truncated,info


class MatchedVoltageEnv(MatchedCurrentEnv):
    def __init__(self,config):
        super().__init__(config)
        del self.base.current_control_callback
        self.action_space=gym.spaces.Box(-1.,1.,(self.dim,),np.float32)
        self.observation_space=gym.spaces.Box(-1.,1.,(self.base.observation_space.shape[0]+2*self.dim,),np.float32)

    _obs=ContinuousVoltageEnv._obs

    def step(self,action):
        old=self.limited.copy();weight=self.smoothness_weight
        self.smoothness_weight=0.
        try:
            obs,reward,terminated,truncated,info=ContinuousVoltageEnv.step(self,action)
        finally:
            self.smoothness_weight=weight
        penalty=weight*float(np.sum(((self.limited-old)/self.vmax)**2))
        if terminated:
            bound=9*self.tracking_weight+3*self.absolute_weight+4*(self.current_weight+self.id_weight+weight)+1
            reward=-bound*max(1,self.base.max_steps-self.base.step_count+1)
        else:
            reward-=penalty
        info["command_smoothness_penalty"]=penalty
        return obs,float(reward),terminated,truncated,info
