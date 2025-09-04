import json
import math
import itertools
import numpy as np

import torch

from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import ParallelEnv

# Basic environment, inspired from https://doi.org/10.24963/ijcai.2021/401
## Environment config items: dt, eps_len, es_P, es_capacity, es_efficiency, ToU, FiT, data_path, use_contracts, max_contract_qnt

class EnergyTradingEnv(ParallelEnv):
    
    def __init__(self, config, render_mode=None):

        # Zoo requirement
        self.render_mode = render_mode

        # Decision every hour, episode runs over a day
        self.eps_len = config.get("eps_len", 24)
        self.dt = config.get("dt", 1)  # Time step in hours

        # PV and Demand Data
        self.data = json.load(open(config.get("data_path"), "r"))

        # Setup Agents, use the agent mapping later to access data
        self._setup_agents()

        # 3 year of data
        self.n_days = int(len(self.data[self.aid_mapping[self.possible_agents[0]]]["pv"])/24)

        # Battery (ES) Config
        self.es_P = config.get("es_P", 0.5)  # Power rating of the battery
        self.es_capacity = config.get("es_capacity", [0, 2]) # [min soc, max soc]
        self.es_efficiency = config.get("es_efficiency", [0.95, 0.95])  # [charge efficiency, discharge efficiency]

        # Grid Config, Time of Use
        self.ToU = config.get("ToU", [0.08, # Off-Peak
                                      0.13, # Shoulder
                                      0.18]) # Peak
        
        self.FiT = config.get("FiT", 0.04)  # Grid Config, Feed-in Tariff

        # Timestep (hours) to ToU period, AER 2014 + Seed Prices
        self.timemap = config.get("timemap", [0, 0, 0, 0, 0, 0, 0, # 12 AM - 7 AM
                                              1, 1, 1, 1, 1, 1, 1, 1, # 7 AM - 3 PM
                                              2, 2, 2, 2, 2, 2, 2, # 3 PM - 10 PM
                                              1, # 10 PM - 11 PM
                                              0]) # 11 PM - 12 AM

        # Contracts
        self.use_contracts = config.get("use_contracts", False) # Bool
        self.use_absolute_contracts = config.get("use_absolute_contracts", False) # Bool, absolute or relative to load
        self.max_contract_qnt = config.get("max_contract_qnt", None) # Max contract quantity

        if self.max_contract_qnt is None:
            self.max_contract_qnt = self.max_pv

        # Clearing Mechanisms
        self.use_double_auction = config.get("use_double_auction", True)
        self.use_pooling = config.get("use_pooling", False)

        # Double auction takes priority, if both are true, DA first then pool
        assert self.use_double_auction or self.use_pooling, "At least one clearing mechanism must be used!"
    
    def _setup_agents(self):

        prosumer_idx = 1
        consumer_idx = 1

        # For better agent names
        self.aid_mapping = {}

        # Max PV for contract quantity
        self.max_pv = 0

        for aid in self.data.keys():

            if self.data[aid]["prosumer"]:
                self.aid_mapping[f"prosumer_{prosumer_idx}"] = aid
                self.max_pv = max(self.max_pv, max(self.data[aid]["pv"]))
                prosumer_idx += 1
            else:
                self.aid_mapping[f"consumer_{consumer_idx}"] = aid
                consumer_idx += 1
        
        self.possible_agents = sorted(self.aid_mapping.keys())
    
    def _timestep_to_ToU_period(self, timestep):
        return self.timemap[timestep % len(self.timemap)]
    
    def observation_space(self, agent):
        
        # Same for all
        if not self.use_contracts:
            # Observation Space: [load, soc, ToU, FiT, sin(t), cos(t)]
            return spaces.Box(low=-32, high=32, shape=(6,), dtype=np.float32)
        else:
            # Add contract: commited qnt
            return spaces.Box(low=-32, high=32, shape=(7,), dtype=np.float32)
    
    def action_space(self, agent):
        
        # Same for all
        if self.use_double_auction:
            # price and soc control
            return spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), shape=(2,), dtype=np.float32)
        else:
            # Only soc control
            return spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    # Zoo requirement
    def render(self):
        pass

    def reset(self, seed=None, options=None):

        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        
        self.timestep = 0
        self.day = options.get("day") if options is not None and "day" in options else np.random.randint(0, self.n_days-1)

        self.agents = self.possible_agents.copy()

        self.state_vars = {
            aid: {
                "soc": options.get("soc")[aid] if (options is not None and "soc" in options) else
                self._gaussian_init(self.es_capacity[0] + (self.es_capacity[1] - self.es_capacity[0]) / 2,
                                    self.es_capacity[1] - self.es_capacity[0],
                                    self.es_capacity[0], self.es_capacity[1]),
                "grid_reliance": 0.0,
                "p2p_participation": 0.0
            } for aid in self.agents}

        if self.use_contracts:

            if options is not None and "contracts" in options:
                self.contracts = options["contracts"]         
            else:
                # Randomly generate contracts (better exploration possible?)
                self.contracts = list(dict() for _ in range(len(self.timemap)))

                for t in range(len(self.timemap)):
                    for aid in self.agents:
                        if self.use_absolute_contracts:
                            # Bit crude, 0 centered Gaussian also possible?
                            self.contracts[t][aid] = np.random.uniform(-self.max_contract_qnt, self.max_contract_qnt)
                            
                            # # Better exploration maybe? Ensure consumers buy, prosumers sell
                            # if "prosumer" in aid:
                            #     self.contracts[t][aid] = np.random.uniform(-self.max_contract_qnt, 0)
                            # else:
                            #     self.contracts[t][aid] = np.random.uniform(0, self.max_contract_qnt)
                        
                        else:
                            # Relative to load, commit percentage of expected load
                            self.contracts[t][aid] = np.random.uniform(0, 1)
        
        obs = {aid: self._get_obs(aid) for aid in self.agents}
        infos = {aid: {"grid_reliance": 0.0,
                       "p2p_participation": 0.0,
                       "soc": self.state_vars[aid]["soc"]} for aid in self.agents}

        return obs, infos
    
    def step(self, action_dict):
        
        # Rewards
        rewards = {aid: 0.0 for aid in self.agents}

        # Total load, including ES actions
        total_load = dict()

        # Update State
        for aid, action in action_dict.items():

            if self.use_double_auction:
                price, soc_control = action
                price = self.FiT + price * (self.ToU[self._timestep_to_ToU_period(self.timestep)] - self.FiT)  # Scale Price
            else:
                soc_control = action[0]
                price = 0.5 # Dummy price for pooling, unused

            soc = self.state_vars[aid]["soc"]
            load = self._get_load(aid)

            # Chip in contracts
            if self.use_contracts and self.use_absolute_contracts:
                load -= self.contracts[self.timestep][aid] 

            if soc_control >= 0:
                
                # Charging
                charge_P = min(self.es_P * soc_control,
                               (self.es_capacity[1] - soc)/(self.es_efficiency[0]*self.dt))
                
                total_load[aid] = load + charge_P * self.dt
                self.state_vars[aid]["soc"] = soc + charge_P * self.dt * self.es_efficiency[0]

            else:

                # Discharging
                discharge_P = max(self.es_P * soc_control,
                                  ((self.es_capacity[0] - soc)*self.es_efficiency[1])/self.dt)
                
                total_load[aid] = load + discharge_P * self.dt
                self.state_vars[aid]["soc"] = soc + (discharge_P * self.dt)/self.es_efficiency[1]

            # Ensure SOC is within bounds
            self.state_vars[aid]["soc"] = np.clip(self.state_vars[aid]["soc"], self.es_capacity[0], self.es_capacity[1])

        # Run the double auction
        if self.use_double_auction:

            # DA quotes
            if self.use_double_auction:
                quotes = dict()

            # Prepare trade action
            for aid in self.agents:
                qnt = total_load[aid]

                # Chip in contracts
                if self.use_contracts and not self.use_absolute_contracts:
                    qnt *= (1 - self.contracts[self.timestep][aid])
                
                quotes[aid] = (qnt, price)
                self.orderbook = quotes
            
            matches, trades, open_book = self._run_double_auction(quotes)

            # Costs/Earnings from successful trades
            for aid in self.agents:
                rewards[aid] -= (trades[aid]["price"] * trades[aid]["qnt"])
            
            # Update p2p participation based on local trades
            for aid in self.agents:
                self.state_vars[aid]["p2p_participation"] += abs(trades[aid]["qnt"])
            
            # Store unmet quotes for pooling
            if self.use_pooling:
                pool_qnts = dict()
            
            for buyer in open_book["buyers"]:
                aid, _, qnt = buyer

                # Settle unmet quotes with ToU and FiT
                if not self.use_pooling:
                    rewards[aid] -= qnt * self.ToU[self._timestep_to_ToU_period(self.timestep)]
                    self.state_vars[aid]["grid_reliance"] += qnt
                else:
                    pool_qnts[aid] = qnt

            for seller in open_book["sellers"]:
                aid, _, qnt = seller

                # Settle unmet quotes with ToU and FiT
                if not self.use_pooling:
                    rewards[aid] += qnt * self.FiT
                    self.state_vars[aid]["grid_reliance"] += qnt
                else:
                    pool_qnts[aid] = -qnt
        
        # Direct pooling (skip DA)
        else:
            pool_qnts = dict()
            for aid in self.agents:
                qnt = total_load[aid]

                # Chip in contracts
                if self.use_contracts and not self.use_absolute_contracts:
                    qnt *= (1 - self.contracts[self.timestep][aid])

                pool_qnts[aid] = qnt

        # Run Shapley Pooling
        if self.use_pooling:
            shapley_values, grid_reliance = self._get_shapley_values(pool_qnts)

            for aid in self.agents:
                rewards[aid] += shapley_values[aid]
                
                # Trick, not sure of individual contributions to grid reliance
                self.state_vars[aid]["grid_reliance"] += (grid_reliance/len(self.agents))

        # Settle Contracts
        if self.use_contracts:

            contract_qnts = dict()
            
            for aid in self.agents:
                if self.use_absolute_contracts:
                    qnt = self.contracts[self.timestep][aid]
                else:
                    qnt = self.contracts[self.timestep][aid] * total_load[aid]
                contract_qnts[aid] = qnt
            
            shapley_values, grid_reliance = self._get_shapley_values(contract_qnts)

            for aid in self.agents:
                rewards[aid] += shapley_values[aid]

                # Trick, not sure of individual contributions to grid reliance
                self.state_vars[aid]["grid_reliance"] += (grid_reliance/len(self.agents))
        
        # Time Controls
        self.timestep += 1
        done = (self.timestep >= self.eps_len)

        # Finishing Touches
        obs = {aid: self._get_obs(aid) for aid in self.agents}
        terminations = {aid: done for aid in self.agents}
        truncations = {aid: False for aid in self.agents}
        infos = {aid: {"grid_reliance": self.state_vars[aid]["grid_reliance"],
                       "p2p_participation": self.state_vars[aid]["p2p_participation"],
                       "soc": self.state_vars[aid]["soc"]} for aid in self.agents}
        
        return obs, rewards, terminations, truncations, infos
    
    def _get_obs(self, aid):

        soc = self.state_vars[aid]["soc"]

        # Should be forecasted? How do we know beforehand?
        load = self._get_load(aid)

        t = self.timestep/self.eps_len

        obs = [load,
               soc,
               self.ToU[self._timestep_to_ToU_period(self.timestep)],
               self.FiT,
               np.sin(2 * np.pi * t),
               np.cos(2 * np.pi * t)]
        
        if self.use_contracts:
            obs.append(self.contracts[self.timestep % len(self.timemap)][aid])

        return np.array(obs, dtype=np.float32)
    
    # Use Global State for Critic
    # Removes redundant info like FiT, ToU, sin(t), cos(t) for all agents
    def state(self):

        t = self.timestep/self.eps_len

        global_state = [self.ToU[self._timestep_to_ToU_period(self.timestep)],
                        self.FiT,
                        np.sin(2 * np.pi * t),
                        np.cos(2 * np.pi * t)]
        
        for aid in self.agents:
            soc = self.state_vars[aid]["soc"]
            load = self._get_load(aid)
            global_state.extend([load, soc])

            if self.use_contracts:
                global_state.append(self.contracts[self.timestep % len(self.timemap)][aid])

        return np.array(global_state, dtype=np.float32)
    
    def _get_load(self, aid):

        aid = self.aid_mapping[aid]

        idx = self.day*24 + self.timestep
        is_prosumer = self.data[aid]["prosumer"]
        
        demand = self.data[aid]["demand"][idx]
        pv = self.data[aid]["pv"][idx]
        load = demand - pv if is_prosumer else demand

        return load
    
    def _gaussian_init(self, mean, bucket, v_min, v_max):

        # Gaussian (99.7%) on the bucket [mean-bucket/2, mean + bucket/2], clipped appropriately
        return np.clip(np.random.normal(mean, bucket/6), max(v_min, mean - bucket/2), min(v_max, mean + bucket/2))
    
    def _run_double_auction(self, quotes):
       
        buyers = []
        sellers = []

        # Parse actions
        for aid, action in quotes.items():
            
            qnt, price = action
            if qnt > 0:
                buyers.append([aid, price, qnt])  # buy quantity
            elif qnt < 0:
                sellers.append([aid, price, -qnt])  # sell quantity (as positive)
            # quantity == 0 → no trade, ignored

        # Sort by willingness: buyers (high price first), sellers (low price first)
        buyers.sort(key=lambda x: -x[1])
        sellers.sort(key=lambda x: x[1])

        matches = []
        trades = {aid: {"qnt": 0, "price": 0} for aid in self.agents}

        buyer_idx = 0
        seller_idx = 0

        while buyer_idx < len(buyers) and seller_idx < len(sellers):
            buyer_id, bid_price, bid_qnt = buyers[buyer_idx]
            seller_id, ask_price, ask_qnt = sellers[seller_idx]

            if bid_price >= ask_price:
                trade_qnt = min(bid_qnt, ask_qnt)
                clearing_price = (bid_price + ask_price) / 2.0

                matches.append({
                    "buyer": buyer_id,
                    "seller": seller_id,
                    "price": clearing_price,
                    "qnt": trade_qnt,
                })

                trades[buyer_id]["price"] = (trades[buyer_id]["price"] * trades[buyer_id]["qnt"] + clearing_price * trade_qnt) / (trades[buyer_id]["qnt"] + trade_qnt)
                trades[buyer_id]["qnt"] += trade_qnt

                trades[seller_id]["price"] = (trades[seller_id]["price"] * abs(trades[seller_id]["qnt"]) + clearing_price * trade_qnt) / (abs(trades[seller_id]["qnt"]) + trade_qnt)
                trades[seller_id]["qnt"] -= trade_qnt

                buyers[buyer_idx] = [buyer_id, bid_price, bid_qnt - trade_qnt]
                sellers[seller_idx] = [seller_id, ask_price, ask_qnt - trade_qnt]

                if buyers[buyer_idx][2] == 0:
                    buyer_idx += 1
                if sellers[seller_idx][2] == 0:
                    seller_idx += 1
            else:
                break  # No more compatible matches
        
        open_book = {
            "buyers": buyers[buyer_idx:] if buyer_idx < len(buyers) else [],
            "sellers": sellers[seller_idx:] if seller_idx < len(sellers) else []}

        return matches, trades, open_book
    
    def _get_coalition_value(self, S, qnts, ToU):

        tot = 0
        for p in S:
            tot += qnts[p]
        
        return -(ToU * tot) if tot >= 0 else -(self.FiT * tot)

    def _get_shapley_values(self, qnts):
        
        n = len(self.agents)
        values = {p: 0 for p in self.agents}

        ToU = self.ToU[self._timestep_to_ToU_period(self.timestep)]
        
        for p in self.agents:
            for S in itertools.chain.from_iterable(itertools.combinations([x for x in self.agents if x != p], r) 
                                                    for r in range(len(self.agents))):
                S = set(S)
                marginal_contribution = self._get_coalition_value(S | {p}, qnts, ToU) - self._get_coalition_value(S, qnts, ToU)
                weight = (math.factorial(len(S)) * math.factorial(n - len(S) - 1)) / math.factorial(n)
                values[p] += weight * marginal_contribution

        grid_reliance = abs(sum(qnts[p] for p in self.agents))
        
        return values, grid_reliance
    
    @staticmethod
    def decode_actions_for_critic(obs, action, task_config):

        es_P = task_config.get("es_P")
        es_efficiency = task_config.get("es_efficiency")
        es_capacity = task_config.get("es_capacity")
        dt = task_config.get("dt")

        price = action[..., 0:1]
        soc_control = action[..., 1:2]

        FiT = obs[..., -1:]
        ToU = obs[..., -2:-1]
        soc = obs[..., -3:-2]
        load = obs[..., -4:-3]

        # Scale Price
        price = FiT + price * (ToU - FiT)

        # Charging mask: soc_control >= 0
        charging_mask = soc_control >= 0

        # Charging calculation
        charge_P = torch.minimum(
            es_P * soc_control,
            (es_capacity[1] - soc) / (es_efficiency[0] * dt))
        
        charge_P = torch.where(charging_mask, charge_P, torch.zeros_like(charge_P))

        # Discharging calculation
        discharge_P = torch.maximum(
            es_P * soc_control,
            ((es_capacity[0] - soc) * es_efficiency[1]) / dt)
        
        discharge_P = torch.where(~charging_mask, discharge_P, torch.zeros_like(discharge_P))

        # qnt: load + charge/discharge
        qnt = load + charge_P * dt + discharge_P * dt

        # Decoded actions as quotes
        quotes = torch.cat((price, qnt), dim=-1).reshape(*price.shape[:-2], -1)

        # Expand quotes to match the number of agents
        quotes = quotes.unsqueeze(-2).expand(*quotes.shape[:-1], obs.shape[-2], quotes.shape[-1])
        
        return quotes
