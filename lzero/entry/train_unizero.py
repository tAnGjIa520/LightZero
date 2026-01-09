import logging
import os
from functools import partial
from typing import Tuple, Optional

import numpy as np
import torch
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.worker import MuZeroCollector as Collector
from .utils import random_collect, calculate_update_per_collect
import torch.distributed as dist
from ding.utils import set_pkg_seed, get_rank, get_world_size


def train_unizero(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':
    """
    Overview:
        This function serves as the training entry point for UniZero, as proposed in our paper "UniZero: Generalized and Efficient Planning with Scalable Latent World Models".
        UniZero aims to enhance the planning capabilities of reinforcement learning agents by addressing the limitations found in MuZero-style algorithms,
        particularly in environments that require capturing long-term dependencies. More details can be found in https://arxiv.org/abs/2406.10667.
    
    Arguments:
        - input_cfg (:obj:`Tuple[dict, dict]`): Configuration in dictionary format.
            ``Tuple[dict, dict]`` indicates [user_config, create_cfg].
        - seed (:obj:`int`): Random seed for reproducibility.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of a PyTorch model.
        - model_path (:obj:`Optional[str]`): Path to the pretrained model, which should
            point to the checkpoint file of the pretrained model. An absolute path is recommended.
            In LightZero, the path typically resembles ``exp_name/ckpt/ckpt_best.pth.tar``.
        - max_train_iter (:obj:`Optional[int]`): Maximum number of policy update iterations during training.
        - max_env_step (:obj:`Optional[int]`): Maximum number of environment interaction steps to collect.
    
    Returns:
        - policy (:obj:`Policy`): The converged policy after training.
    """

    cfg, create_cfg = input_cfg

    # Ensure the specified policy type is supported
    assert create_cfg.policy.type in ['unizero', 'sampled_unizero'], "train_unizero only supports the following algorithms: 'unizero', 'sampled_unizero'"
    logging.info(f"Using policy type: {create_cfg.policy.type}")

    # Import the appropriate GameBuffer class based on the policy type
    game_buffer_classes = {'unizero': 'UniZeroGameBuffer', 'sampled_unizero': 'SampledUniZeroGameBuffer'}
    GameBuffer = getattr(__import__('lzero.mcts', fromlist=[game_buffer_classes[create_cfg.policy.type]]),
                         game_buffer_classes[create_cfg.policy.type])
    
    # Check for GPU availability and set the device accordingly
    cfg.policy.device = cfg.policy.model.world_model_cfg.device if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device set to: {cfg.policy.device}")

    # Compile the configuration file
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Create environment manager
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    # Initialize environment and random seed
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=torch.cuda.is_available())

    # Initialize wandb if specified
    if cfg.policy.use_wandb:
        logging.info("Initializing wandb...")
        wandb.init(
            project="LightZero",
            config=cfg,
            sync_tensorboard=False,
            monitor_gym=False,
            save_code=True,
        )
        logging.info("wandb initialization completed!")

    # Create policy
    logging.info("Creating policy...")
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
    logging.info("Policy created successfully!")

    # Load pretrained model if specified
    if model_path is not None:
        logging.info(f"Loading pretrained model from {model_path}...")
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))
        logging.info("Pretrained model loaded successfully!")

    # Create core components for training
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = GameBuffer(cfg.policy)
    collector = Collector(env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=cfg.exp_name,
                          policy_config=cfg.policy)
    evaluator = Evaluator(eval_freq=cfg.policy.eval_freq, n_evaluator_episode=cfg.env.n_evaluator_episode,
                          stop_value=cfg.env.stop_value, env=evaluator_env, policy=policy.eval_mode,
                          tb_logger=tb_logger, exp_name=cfg.exp_name, policy_config=cfg.policy)

    # Execute the learner's before_run hook
    learner.call_hook('before_run')

    if cfg.policy.use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    # Randomly collect data if specified
    if cfg.policy.random_collect_episode_num > 0:
        logging.info("Collecting random data...")
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)
        logging.info("Random data collection completed!")

    batch_size = policy._cfg.batch_size

    if cfg.policy.multi_gpu:
        # Get current world size and rank
        world_size = get_world_size()
        rank = get_rank()
    else:
        world_size = 1
        rank = 0

    while True:
        # Log memory usage of the replay buffer
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)

        # Set temperature parameter for data collection
        collect_kwargs = {
            'temperature': visit_count_temperature(
                cfg.policy.manual_temperature_decay,
                cfg.policy.fixed_temperature_value,
                cfg.policy.threshold_training_steps_for_final_temperature,
                trained_steps=learner.train_iter
            ),
            'epsilon': 0.0  # Default epsilon value
        }

        # Configure epsilon-greedy exploration
        if cfg.policy.eps.eps_greedy_exploration_in_collect:
            epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=cfg.policy.eps.start,
                end=cfg.policy.eps.end,
                decay=cfg.policy.eps.decay,
                type_=cfg.policy.eps.type
            )
            collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)

        # Evaluate policy performance
        # # if learner.train_iter == 0 or evaluator.should_eval(learner.train_iter):
        logging.info(f"Training iteration {learner.train_iter}: Starting evaluation...")
        stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
        logging.info(f"Training iteration {learner.train_iter}: Evaluation completed, stop condition: {stop}, current reward: {reward}")
        if stop:
            logging.info("Stopping condition met, training ends!")
            break

        # Collect new data
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        
        # # Print new_data structure and advantage information
        # data_segments, meta_data = new_data
        # logging.info(f"\n{'='*80}")
        # logging.info(f"Rank {rank}, Training iteration {learner.train_iter}: New data collection completed!")
        # logging.info(f"Number of segments collected: {len(data_segments)}")
        
        # for i, segment in enumerate(data_segments):
        #     logging.info(f"\n--- Segment {i} ---")
        #     logging.info(f"  episode_id: {segment.episode_id}")
        #     logging.info(f"  action_segment shape: {segment.action_segment.shape}")
        #     logging.info(f"  reward_segment shape: {segment.reward_segment.shape}")
        #     logging.info(f"  root_value_segment shape: {segment.root_value_segment.shape}")
            
        #     if hasattr(segment, 'advantage_segment') and len(segment.advantage_segment) > 0:
        #         logging.info(f"  advantage_segment shape: {segment.advantage_segment.shape}")
        #         logging.info(f"  advantage_segment (first 5): {segment.advantage_segment[:5]}")
        #         logging.info(f"  advantage mean: {segment.advantage_segment.mean():.4f}, std: {segment.advantage_segment.std():.4f}")
        #     else:
        #         logging.info(f"  advantage_segment: NOT COMPUTED or EMPTY")
            
        #     logging.info(f"  meta_data - done: {meta_data[i]['done']}, priorities: {meta_data[i]['priorities']}")
        
        # logging.info(f"{'='*80}\n")
        
        # exit()

        # Determine updates per collection
        update_per_collect = cfg.policy.update_per_collect
        if update_per_collect is None:
            update_per_collect = calculate_update_per_collect(cfg, new_data, world_size)

        # Update replay buffer
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        if world_size > 1:
            # Synchronize all ranks before training
            try:
                dist.barrier()
            except Exception as e:
                logging.error(f'Rank {rank}: Synchronization barrier failed, error: {e}')
                break

        # Check if there is sufficient data for training
        if collector.envstep > cfg.policy.train_start_after_envsteps:
            if cfg.policy.sample_type == 'episode':
                data_sufficient = replay_buffer.get_num_of_game_segments() > batch_size
            else:
                data_sufficient = replay_buffer.get_num_of_transitions() > batch_size
            
            if not data_sufficient:
                logging.warning(
                    f'Rank {rank}: The data in replay_buffer is not sufficient to sample a mini-batch: '
                    f'batch_size: {batch_size}, replay_buffer: {replay_buffer}. Continue to collect now ....'
                )
                continue

            # Execute multiple training rounds
            for i in range(update_per_collect):
                # 
                train_data = replay_buffer.sample(batch_size, policy)
                
                # Print train_data structure (only first iteration)
                # if i == 0 and learner.train_iter == 0:
                #     logging.info(f"\n{'='*80}")
                #     logging.info(f"train_data structure from replay_buffer.sample():")
                #     logging.info(f"train_data type: {type(train_data)}")
                #     logging.info(f"train_data length: {len(train_data)}")
                    
                #     if len(train_data) >= 2:
                #         current_batch, target_batch = train_data[0], train_data[1]
                        
                #         logging.info(f"\n--- current_batch (input) ---")
                #         logging.info(f"current_batch type: {type(current_batch)}")
                #         logging.info(f"current_batch length: {len(current_batch)}")
                        
                #         current_batch_names = [
                #             "obs_list (stacked observations)",
                #             "action_list (actions at t)",
                #             "bootstrap_action_list (actions at t+td_steps, UniZero)",
                #             "mask_list (validity mask)",
                #             "batch_index_list (sample indices)",
                #             "weights_list (priority weights)",
                #             "make_time_list (creation time)",
                #             "timestep_list (timesteps for transformer, UniZero)",
                #             "advantage_list (GAE advantages, PPO)",
                #             "old_log_prob_list (old policy log probs, PPO)",
                #             "return_list (GAE returns = advantage + value, PPO)"
                #         ]
                        
                #         for idx, item in enumerate(current_batch):
                #             name = current_batch_names[idx] if idx < len(current_batch_names) else f"unknown_{idx}"
                #             if hasattr(item, 'shape'):
                #                 logging.info(f"  current_batch[{idx}] ({name}): shape={item.shape}, dtype={item.dtype}")
                #                 # Extra info for PPO-related metrics
                #                 if idx == 8 and "advantage" in name.lower():  # advantage_list is index 8
                #                     logging.info(f"    -> advantage mean: {item.mean():.4f}, std: {item.std():.4f}, min: {item.min():.4f}, max: {item.max():.4f}")
                #                 elif idx == 9 and "log_prob" in name.lower():  # old_log_prob_list is index 9
                #                     logging.info(f"    -> old_log_prob mean: {item.mean():.4f}, std: {item.std():.4f}, min: {item.min():.4f}, max: {item.max():.4f}")
                #                 elif idx == 10 and "return" in name.lower():  # return_list is index 10
                #                     logging.info(f"    -> return mean: {item.mean():.4f}, std: {item.std():.4f}, min: {item.min():.4f}, max: {item.max():.4f}")
                #             else:
                #                 logging.info(f"  current_batch[{idx}] ({name}): type={type(item)}, len={len(item) if hasattr(item, '__len__') else 'N/A'}")
                        
                #         logging.info(f"\n--- target_batch (labels) ---")
                #         logging.info(f"target_batch type: {type(target_batch)}")
                #         logging.info(f"target_batch length: {len(target_batch)}")
                        
                #         target_batch_names = [
                #             "batch_rewards (target rewards)",
                #             "batch_target_values (target values)",
                #             "batch_target_policies (target policy distributions)"
                #         ]
                        
                #         for idx, item in enumerate(target_batch):
                #             name = target_batch_names[idx] if idx < len(target_batch_names) else f"unknown_{idx}"
                #             if hasattr(item, 'shape'):
                #                 logging.info(f"  target_batch[{idx}] ({name}): shape={item.shape}, dtype={item.dtype}")
                #             else:
                #                 logging.info(f"  target_batch[{idx}] ({name}): type={type(item)}, len={len(item) if hasattr(item, '__len__') else 'N/A'}")
                    
                #     logging.info(f"{'='*80}\n")
                # # exit()
                
                if replay_buffer._cfg.reanalyze_ratio > 0 and i % 20 == 0:
                    policy.recompute_pos_emb_diff_and_clear_cache()
                
                if cfg.policy.use_wandb:
                    policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

                train_data.append(learner.train_iter)

                log_vars = learner.train(train_data, collector.envstep)
                if cfg.policy.use_priority:
                    replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        # Clear replay buffer after training for online learning
        if cfg.policy.get('online_learning', False):
            replay_buffer.clear()

        policy.recompute_pos_emb_diff_and_clear_cache()

        # Check stopping criteria
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            logging.info("Stopping condition met, training ends!")
            break

    learner.call_hook('after_run')
    if cfg.policy.use_wandb:
        wandb.finish()
    logging.info("===== Training Completed =====")
    return policy