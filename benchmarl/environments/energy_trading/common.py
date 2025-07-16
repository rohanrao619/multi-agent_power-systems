#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

import torch
from tensordict import TensorDictBase

from torchrl.data import CompositeSpec, Unbounded, Composite
from torchrl.envs import EnvBase
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.utils import MarlGroupMapType

from env.energy_trading import EnergyTradingEnv

class EnergyTradingTask(Task):
    # Your task names.
    # Their config will be loaded from conf/task/customenv

    SIMPLE_P2P = None  # Loaded automatically from conf/task/customenv/task_1

    @staticmethod
    def associated_class():
        return EnergyTradingClass


class EnergyTradingClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:    
        return lambda: PettingZooWrapper(
            env=EnergyTradingEnv(self.config),
            return_state=False if self.config["use_single_group"] else True,
            group_map=MarlGroupMapType.ALL_IN_ONE_GROUP if self.config["use_single_group"] else MarlGroupMapType.ONE_GROUP_PER_AGENT,
            use_mask=False,
            categorical_actions=False,
            seed=None,
            done_on_any=True)

    def supports_continuous_actions(self) -> bool:
        # Does the environment support continuous actions?
        return True

    def supports_discrete_actions(self) -> bool:
        # Does the environment support discrete actions?
        return False

    def has_render(self, env: EnvBase) -> bool:
        # Does the env have a env.render(mode="rgb_array") or env.render() function?
        return False

    def max_steps(self, env: EnvBase) -> int:
        # Maximum number of steps for a rollout during evaluation
        return 100

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # The group map mapping group names to agent names
        # The data in the tensordict will havebe presented this way
        return env.group_map

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the observation.
        # Must be a CompositeSpec with one (group_name, observation_key) entry per group.
        
        observation_spec = env.observation_spec.clone()
        
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the action.
        # If provided, must be a CompositeSpec with one (group_name, "action") entry per group.
        return env.full_action_spec

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the state.
        # If provided, must be a CompositeSpec with one "state" entry

        if self.config["use_single_group"]:
            specs = None      
        else:
            n_agents = len(env.group_map)
            specs = Composite(
                state = Unbounded(shape=(n_agents, (n_agents - 1)*2),
                            dtype=torch.float32,
                            device=env.device))

        return specs
    
    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the action mask.
        # If provided, must be a CompositeSpec with one (group_name, "action_mask") entry per group.
        return None

    # Used nowhere!?
    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the info.
        # If provided, must be a CompositeSpec with one (group_name, "info") entry per group (this entry can be composite).
        return None

    @staticmethod
    def env_name() -> str:
        # The name of the environment in the benchmarl/conf/task folder
        return "energy_trading"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        # Optionally return a str->float dict with extra things to log
        # This function has access to the collected batch and is optional
        return {}