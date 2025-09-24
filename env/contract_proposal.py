import numpy as np

from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import ParallelEnv
from .energy_trading import EnergyTradingEnv

from torch import Tensor
from tensordict import TensorDict
from benchmarl.experiment import Experiment

# Level 2: Environment for Contract Proposal
## Assume frozen downstream environment, already trained on the entire contract space

class ContractProposalEnv(ParallelEnv):

    def __init__(self, config, render_mode=None):

        # Zoo requirement
        self.render_mode = render_mode

        # Base experminent data
        base_exp = Experiment.reload_from_file(config['base_exp_path'])
        self.base_config = base_exp.task.config
        self.group_map = base_exp.group_map
        self.base_policy = base_exp.policy

        # Clear memory
        del base_exp

        # Base environment
        self.trading_env = EnergyTradingEnv(self.base_config, render_mode=render_mode)

        # Setup agents
        self.possible_agents = self.trading_env.possible_agents

        # Episode length, Max contract quantity and type, same as base environment
        self.eps_len = self.base_config['eps_len']
        self.max_contract_qnt = self.base_config['max_contract_qnt']
        self.use_absolute_contracts = self.base_config['use_absolute_contracts']
                                  
    def observation_space(self, agent):
        
        # Same for all
        # Going in Bandit Style
        # Observation Space: [ToU, FiT, sin(t), cos(t), is_weekend, is_summer, is_winter]
        # Optionally add max_contract_qnt
        return spaces.Box(low=-32, high=32, shape=(7 if not self.use_absolute_contracts else 8,), dtype=np.float32)
    
    def action_space(self, agent):
        
        # Same for all
        # Action Space: Contract quantity
        return spaces.Box(low=-1 if self.use_absolute_contracts else 0, 
                          high=1, shape=(1,), dtype=np.float32)

    # Zoo requirement
    def render(self):
        pass

    def reset(self, seed=None, options=None):

        # Reset base environment, everything runs through it
        self.base_obs, _ = self.trading_env.reset(options=options,
                                                  seed = seed) # Seeding happens only once
        
        self.day = self.trading_env.day  # Sync days
        self.timestep = 0
        self.agents = self.possible_agents.copy()

        obs = {aid: self._get_obs(aid) for aid in self.agents}
        infos = {aid: {} for aid in self.agents}

        return obs, infos
    
    def _get_obs(self, aid):

        t = self.timestep/self.eps_len

        obs = [self.trading_env.ToU[self.trading_env._timestep_to_ToU_period(self.timestep)],
               self.trading_env.FiT,
               np.sin(2 * np.pi * t),
               np.cos(2 * np.pi * t),
               int(self.trading_env.is_weekend[self.day]),
               int(self.trading_env.is_summer[self.day]),
               int(self.trading_env.is_winter[self.day])]
        
        if self.use_absolute_contracts:
            obs.append(self.max_contract_qnt)

        return np.array(obs, dtype=np.float32)
    
    # Use Global State, same as base Critics
    def state(self):

        t = self.timestep/self.eps_len

        global_state = [self.trading_env.ToU[self.trading_env._timestep_to_ToU_period(self.timestep)],
                        self.trading_env.FiT,
                        np.sin(2 * np.pi * t),
                        np.cos(2 * np.pi * t),
                        int(self.trading_env.is_weekend[self.day]),
                        int(self.trading_env.is_summer[self.day]),
                        int(self.trading_env.is_winter[self.day])]
        
        for aid in self.agents:
            soc = self.trading_env.state_vars[aid]["soc"]/self.trading_env.es_capacity[1]  # Assume min soc is 0
            demand, pv = self.trading_env._get_load(aid, scaled=True)
            global_state.extend([demand, pv if self.trading_env._is_prosumer(aid) else 0.0, soc])

        return np.array(global_state, dtype=np.float32)
    
    # Apply base policy to trade, assuming single group
    def _trade_forward(self, obs):

        stacked_obs = Tensor(np.array([obs[aid] for aid in self.group_map['agents']], dtype=np.float32))
        obs_tdict = TensorDict(agents=TensorDict(observation=stacked_obs)).to(self.base_policy.device)

        actions = self.base_policy.forward(obs_tdict)['agents']['action'].detach().cpu().numpy()
        action_dict = {aid: actions[i] for i, aid in enumerate(self.group_map['agents'])}

        # Clear memory
        del obs_tdict
        del actions
        
        return action_dict
    
    def step(self, action_dict):
        
        for aid in self.agents:
            if self.use_absolute_contracts:
                # Rescale action from [-1, 1] to [-max_contract_qnt, max_contract_qnt]
                contract_qnt = action_dict[aid][0] * self.max_contract_qnt
            else:
                contract_qnt = action_dict[aid][0]
            
            self.trading_env.contracts[self.timestep][aid] = contract_qnt
        
        # Init rewards
        rewards = {aid: 0.0 for aid in self.agents}
        
        # Step base environment
        base_actions = self._trade_forward(self.base_obs)
        
        self.base_obs, base_rewards, _, _, _ = self.trading_env.step(base_actions)
        for aid in self.agents:
            rewards[aid] += base_rewards[aid]
        
        # Time Controls
        self.timestep += 1
        done = (self.timestep >= self.eps_len)

        # Finishing Touches
        obs = {aid: self._get_obs(aid) for aid in self.agents}
        terminations = {aid: done for aid in self.agents}
        truncations = {aid: False for aid in self.agents}
        infos = {aid: {} for aid in self.agents}
        
        return obs, rewards, terminations, truncations, infos