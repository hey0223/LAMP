# coding=utf-8

import numpy as np
import os
import argparse
import logging
from omegaconf import OmegaConf
from env.env_core import economic_society
from agents.rule_based import rule_agent
from agents.calibration import calibration_agent
from agents.MADDPG_block.MAAC import maddpg_agent
from utils.seeds import set_seeds
import asyncio

# Setup logging configuration
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "Test.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),   # File handler to write logs to file
        logging.StreamHandler()                    # Stream handler to output logs to console
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='default')
    parser.add_argument("--alg", type=str, default='maddpg', help="ppo, rule_based, independent")
    parser.add_argument("--task", type=str, default='gdp', help="gini, social_welfare, gdp_gini")
    parser.add_argument('--device-num', type=int, default=0, help='CUDA device number')
    parser.add_argument('--n_households', type=int, default=10, help='Total number of households')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--hidden_size', type=int, default=128, help='[64, 128, 256]')
    parser.add_argument('--q_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--p_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--batch_size', type=int, default=64, help='[32, 64, 128, 256]')
    parser.add_argument('--update_cycles', type=int, default=10, help='[10, 100, 1000]')
    parser.add_argument('--update_freq', type=int, default=10, help='[10, 20, 30]')
    parser.add_argument('--initial_train', type=int, default=10, help='[10, 100, 200]')
    parser.add_argument('--news_interval', type=int, default=10, help='Interval for generating news')
    parser.add_argument(
        '--data_sources', type=str, nargs='+',
        default=["economic_observations", "agent_actions"],
        help='Data sources for the Intelligence Bureau'
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Set environment variables to limit threading and avoid performance overhead
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    args = parse_args()
    path = args.config

    try:
        # Load and configure the YAML configuration file
        yaml_cfg = OmegaConf.load(f'./cfg/{path}.yaml')
        yaml_cfg.Trainer["n_households"] = args.n_households
        yaml_cfg.Environment.Entities[1]["entity_args"].n = args.n_households
        yaml_cfg.Environment.env_core["env_args"].gov_task = args.task
        yaml_cfg.seed = args.seed

        # Apply hyperparameter overrides from command line
        yaml_cfg.Trainer["hidden_size"] = args.hidden_size
        yaml_cfg.Trainer["q_lr"] = args.q_lr
        yaml_cfg.Trainer["p_lr"] = args.p_lr
        yaml_cfg.Trainer["batch_size"] = args.batch_size

        # Set random seed and specify GPU device
        set_seeds(yaml_cfg.seed, cuda=yaml_cfg.Trainer["cuda"])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)

        # Initialize the economic society environment
        env = economic_society(yaml_cfg.Environment)

        # Instantiate the trainer based on the selected algorithm
        if args.alg == "rule_based":
            trainer = rule_agent(env, yaml_cfg.Trainer)
        elif args.alg == "maddpg":
            trainer = maddpg_agent(env, yaml_cfg.Trainer)

        # Log system configuration and active hyperparameters
        logger.info(f"Selected Algorithm: {args.alg}")
        logger.info(f"Active Task: {args.task}")
        logger.info(f"Total Households: {yaml_cfg.Trainer['n_households']}")
        logger.info(
            f"Hyperparameters -> Hidden Size: {args.hidden_size}, "
            f"Q_LR: {args.q_lr}, P_LR: {args.p_lr}, "
            f"Batch Size: {args.batch_size}"
        )

        # Execute training phase
        logger.info("Starting training process...")
        trainer.learn()
        logger.info("Training process completed.")

        # Execute testing/evaluation phase
        logger.info("Starting testing process...")
        trainer.test()
        logger.info("Testing process completed.")

        # Cleanup and shutdown
        env.close()
        logger.info("Environment successfully closed.")

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)
