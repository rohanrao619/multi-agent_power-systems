import json
import numpy as np

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
        self.es_P = config.get("es_P", 2)  # Power rating of the battery
        self.es_capacity = config.get("es_capacity", [2, 10]) # [min soc, max soc]
        self.es_efficiency = config.get("es_efficiency", [0.95,0.95])  # [charge efficiency, discharge efficiency]

        # Grid Config, Time of Use
        self.ToU = config.get("ToU", [0.08, # 9 PM - 8 AM (Overnight)
                                      0.13, # 9 AM - 4 PM (Off-Peak)
                                      0.18]) # 5 PM - 8 PM (Peak)
        
        self.FiT = config.get("FiT", 0.04)  # Grid Config, Feed-in Tariff

        # Timestep (hours) to ToU period
        self.timemap = [0, 0, 0, 0, 0, 0, 0, 0, 0, # 12 AM - 8 AM
                        1, 1, 1, 1, 1, 1, 1, 1, # 9 AM - 4 PM
                        2, 2, 2, 2, # 5 PM - 8 PM
                        0, 0, 0] # 9 PM - 11 PM

        # Contracts
        self.use_contracts = config.get("use_contracts", False) # Bool
        self.max_contract_qnt = config.get("max_contract_qnt", None) # Max contract quantity

        if self.max_contract_qnt is None:
            self.max_contract_qnt = self.max_pv

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
            # Observation Space: [load, soc, ToU, FiT]
            return spaces.Box(low=0, high=128, shape=(4,), dtype=np.float32)
        else:
            # Add contract: [commited qnt and price]
            return spaces.Box(low=0, high=128, shape=(6,), dtype=np.float32)
    
    def action_space(self, agent):
        
        # Same for all
        # Action Space: [price and soc control]
        return spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), shape=(2,), dtype=np.float32)
    

    # Zoo requirement
    def render(self):
        pass


    def reset(self, seed=None, options=None):

        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        
        self.timestep = 0
        self.day = options.get("day") if options is not None and "day" in options else np.random.randint(0, self.n_days-7)

        self.agents = self.possible_agents.copy()

        self.state_vars = {
            aid: {
                "soc": self._gaussian_init(self.es_capacity[0] + (self.es_capacity[1] - self.es_capacity[0]) / 2,
                                           self.es_capacity[1] - self.es_capacity[0],
                                           self.es_capacity[0], self.es_capacity[1]),
                "grid_reliance": 0.0,
                "p2p_participation": 0.0,
                "contracted_qnt": 0.0
            } for aid in self.agents
        } if (options is None or "state_vars" not in options) else options["state_vars"]

        # Initialize orderbook with zero quotes
        self.orderbook = {aid: (0, 0) for aid in self.agents}  # (quantity, price)

        if self.use_contracts:
            
            if options is not None and "contract_bids" in options:
                # Use provided bids
                contract_bids = options["contract_bids"]
            else:
                contract_bids = list(dict() for _ in range(len(self.ToU)))
                for aid in self.agents:
                    for period in range(len(self.ToU)):
                        
                        # Randomly generate bids (better exploration possible?)
                        price = self.FiT + np.random.uniform(0, 1) * (self.ToU[period] - self.FiT)
                        qnt = np.random.uniform(-self.max_contract_qnt, self.max_contract_qnt)
                        
                        contract_bids[period][aid] = (qnt, price)

            # Sign contracts using Double Auction, Just an idea for now
            self.contracts = list(dict() for _ in range(len(self.ToU)))
            for period in range(len(self.ToU)):
                matches, trades, open_book = self._run_double_auction(contract_bids[period])
                self.contracts[period] = {"matches": matches,
                                          "trades": trades,
                                          "open_book": open_book,
                                          "bids": contract_bids[period]}
        
        obs = {aid: self._get_obs(aid) for aid in self.agents}
        infos = {aid: {"grid_reliance": 0.0,
                       "p2p_participation": 0.0,
                       "contracted_qnt": 0.0,
                       "soc": self.state_vars[aid]["soc"]} for aid in self.agents}

        return obs, infos

    
    def step(self, action_dict):
        
        # P2P quotes
        quotes = {}

        # Update State
        for aid, action in action_dict.items():

            price, soc_control = action
            price = self.FiT + price * (self.ToU[self._timestep_to_ToU_period(self.timestep)] - self.FiT)  # Scale Price

            soc = self.state_vars[aid]["soc"]
            load = self._get_load(aid)

            # Chip in contracts
            if self.use_contracts:
                load -= self.contracts[self._timestep_to_ToU_period(self.timestep)]["trades"][aid]["qnt"]

            if soc_control >= 0:
                
                # Charging
                charge_P = min(self.es_P * soc_control,
                               (self.es_capacity[1] - soc)/(self.es_efficiency[0]*self.dt))
                
                self.state_vars[aid]["soc"] = soc + charge_P * self.dt * self.es_efficiency[0]
                qnt = load + charge_P * self.dt

            else:

                # Discharging
                discharge_P = max(self.es_P * soc_control,
                                  ((self.es_capacity[0] - soc)*self.es_efficiency[1])/self.dt)
                
                self.state_vars[aid]["soc"] = soc + (discharge_P * self.dt)/self.es_efficiency[1]
                qnt = load + discharge_P * self.dt

            # Ensure SOC is within bounds
            self.state_vars[aid]["soc"] = np.clip(self.state_vars[aid]["soc"], self.es_capacity[0], self.es_capacity[1])

            # Prepare trade action
            quotes[aid] = (qnt, price)

        # State Update
        self.orderbook = quotes
        
        # Run the double auction
        matches, trades, open_book = self._run_double_auction(quotes)

        # Costs/Earnings from successful trades
        rewards = {aid: -(trades[aid]["price"] * trades[aid]["qnt"]) for aid in self.agents}
        # Update p2p participation based on local trades
        for aid in self.agents:
            self.state_vars[aid]["p2p_participation"] += abs(trades[aid]["qnt"])

        # Settle unmet quotes with ToU and FiT
        for buyer in open_book["buyers"]:
            aid, _, qnt = buyer
            rewards[aid] -= qnt * self.ToU[self._timestep_to_ToU_period(self.timestep)]
            self.state_vars[aid]["grid_reliance"] += qnt

        for seller in open_book["sellers"]:
            aid, _, qnt = seller
            rewards[aid] += qnt * self.FiT
            self.state_vars[aid]["grid_reliance"] += qnt

        # Settle contracts
        if self.use_contracts:
            for aid in self.agents:
                contract_trade = self.contracts[self._timestep_to_ToU_period(self.timestep)]["trades"][aid]
                rewards[aid] -= (contract_trade["price"] * contract_trade["qnt"])
                # Update contracted quantity
                self.state_vars[aid]["contracted_qnt"] += abs(contract_trade["qnt"])
        
        # Time Controls
        self.timestep += 1
        done = (self.timestep >= self.eps_len)

        # Finishing Touches
        obs_len = 6 if self.use_contracts else 4
        obs = {aid: self._get_obs(aid) if not done else np.zeros((obs_len,), dtype=np.float32) for aid in self.agents} # Gibberish at the end, does not matter
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: done for aid in self.agents}
        infos = {aid: {"grid_reliance": self.state_vars[aid]["grid_reliance"],
                       "p2p_participation": self.state_vars[aid]["p2p_participation"],
                       "contracted_qnt": self.state_vars[aid]["contracted_qnt"],
                       "soc": self.state_vars[aid]["soc"]} for aid in self.agents}
        
        return obs, rewards, terminations, truncations, infos
    
    # Centralised training
    def state(self):

        state_vals = []

        for aid in sorted(self.agents):
            state_vals.append([self.orderbook[k] for k in sorted(self.agents) if k != aid])

        state_vals = np.array(state_vals, dtype=np.float32).reshape(len(self.agents), -1)
        
        return state_vals
    
    def _get_obs(self, aid):

        soc = self.state_vars[aid]["soc"]

        # Should be forecasted? How do we know beforehand?
        load = self._get_load(aid)

        obs = [load,
               soc,
               self.ToU[self._timestep_to_ToU_period(self.timestep)],
               self.FiT]
        
        if self.use_contracts:
            contract_trade = self.contracts[self._timestep_to_ToU_period(self.timestep)]["trades"][aid]
            obs.append(contract_trade["qnt"])
            obs.append(contract_trade["price"])

        return np.array(obs, dtype=np.float32)
    
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
                buyers.append((aid, price, qnt))  # buy quantity
            elif qnt < 0:
                sellers.append((aid, price, -qnt))  # sell quantity (as positive)
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

                buyers[buyer_idx] = (buyer_id, bid_price, bid_qnt - trade_qnt)
                sellers[seller_idx] = (seller_id, ask_price, ask_qnt - trade_qnt)

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
