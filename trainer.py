import sys
sys.path.append(".")  # Adjust the path to import "local" benchmarl

from benchmarl.algorithms import MaddpgConfig, MappoConfig
from benchmarl.environments import EnergyTradingTask, ContractProposalTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    # Loads from "benchmarl/conf/task/energy_trading/simple_p2p.yaml"
    task = EnergyTradingTask.SIMPLE_P2P.get_from_yaml()

    # # Loads from "benchmarl/conf/task/contract_proposal/tou_proposal.yaml"
    # task = ContractProposalTask.TOU_PROPOSAL.get_from_yaml()
    # task.config["base_exp_path"] = "results/mappo_simple_p2p_mlp__97375b87_25_07_26-19_17_31/checkpoints/checkpoint_32768.pt"

    # Modify as needed
    algorithm_config = MaddpgConfig.get_from_yaml()
    experiment_config = ExperimentConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Experiment Config
    experiment_config.share_policy_params = False
    experiment_config.max_n_frames = 262144
    experiment_config.off_policy_collected_frames_per_batch = 8192
    experiment_config.off_policy_n_envs_per_worker = 8
    
    # Logging Config
    experiment_config.loggers = ['csv'] # No WandB for now
    experiment_config.save_folder = 'results'
    experiment_config.checkpoint_interval = 8192
    
    # Evaluation Config
    experiment_config.evaluation = True
    experiment_config.evaluation_interval = 8192
    experiment_config.evaluation_episodes = 32
    experiment_config.evaluation_static = True

    # Actor Config
    model_config.num_cells = [128,128]
    # model_config.activation_class = torch.nn.ReLU

    # Critic Config
    critic_model_config.num_cells = [128,128]
    # critic_model_config.activation_class = torch.nn.ReLU

    # Algorithm Config
    algorithm_config.use_double_auction_critic = False # Only implemented for MADDPG
    algorithm_config.share_param_critic = False

    # Task Config
    task.config["use_single_group"] = True
    task.config["use_contracts"] = False
    task.config["eps_len"] = 24

    experiment = Experiment(task=task,
                            algorithm_config=algorithm_config,
                            model_config=model_config,
                            critic_model_config=critic_model_config,
                            seed=42,
                            config=experiment_config)
    
    experiment.run()