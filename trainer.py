import sys
sys.path.append(".")  # Adjust the path to import "local" benchmarl

from benchmarl.algorithms import MaddpgConfig, MappoConfig
from benchmarl.environments import EnergyTradingTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    # Loads from "benchmarl/conf/task/energy_trading/simple_p2p.yaml"
    task = EnergyTradingTask.SIMPLE_P2P.get_from_yaml()

    # Modify as needed
    algorithm_config = MaddpgConfig.get_from_yaml()
    experiment_config = ExperimentConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Experiment Config
    experiment_config.share_policy_params = False
    experiment_config.max_n_frames = 131072
    experiment_config.off_policy_collected_frames_per_batch = 1024
    experiment_config.off_policy_n_envs_per_worker = 8
    experiment_config.evaluation = False
    experiment_config.loggers = ['csv'] # No WandB for now
    experiment_config.save_folder = 'results'

    # Actor Config
    model_config.num_cells = [64,64]

    # Critic Config
    critic_model_config.num_cells = [64,64]

    # Algorithm Config
    algorithm_config.use_double_auction_critic = False

    # Task Config
    task.config["use_single_group"] = True
    task.config["use_contracts"] = True

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=42,
        config=experiment_config)
    
    experiment.run()