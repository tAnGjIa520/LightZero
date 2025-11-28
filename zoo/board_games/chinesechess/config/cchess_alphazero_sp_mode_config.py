from easydict import EasyDict

# ==============================================================
# 最常修改的配置参数
# ==============================================================
# 多GPU配置
use_multi_gpu = True  # 开启多GPU训练
gpu_num = 8  # 使用的GPU数量，根据实际情况修改
batch_size = 128

collector_env_num = 4
n_episode = 64
evaluator_env_num = 10
num_simulations = 20  # MCTS模拟次数
update_per_collect = 10
max_env_step = int(1e8)  # 中国象棋需要更多训练步数
mcts_ctree = True  # 使用C树优化的MCTS
# ==============================================================
# 配置参数结束
# ==============================================================

cchess_alphazero_config = dict(
    exp_name=f'data_alphazero/cchess_alphazero_sp-mode_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        battle_mode='self_play_mode',
        channel_last=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        alphazero_mcts_ctree=mcts_ctree,
        # UCI引擎配置（可选，用于eval_mode评估）
        # uci_engine_path='pikafish',  # UCI引擎路径，如 'pikafish' 或 '/path/to/pikafish'
        # engine_depth=10,  # 引擎搜索深度，1-20，越大越强（5=弱，10=中，15=强，20=很强）
        # render_mode='human',  # 渲染模式: 'human'打印棋盘, 'svg'生成SVG
    ),
    policy=dict(
        mcts_ctree=mcts_ctree,
        # ==============================================================
        # for the creation of simulation env
        simulation_env_id='cchess',
        simulation_env_config_type='self_play',
        # ==============================================================
        torch_compile=False,
        tensor_float_32=False,
        model=dict(
            # 15层 * 4帧 + 1层颜色 = 57层
            # 14层(7己+7敌) * 4历史 + 1颜色
            observation_shape=(57, 10, 9),
            action_space_size=90 * 90,  # 8100 个可能的动作
            # image_channel=57,  # 匹配 observation_shape
            num_res_blocks=9,  # 增加到9个残差块，匹配中国象棋复杂度
            num_channels=128,  # 增加通道数
        ),
        cuda=True,
        multi_gpu=use_multi_gpu,  # 开启多GPU数据并行
        env_type='board_games',
        action_type='varied_action_space',
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,  # AlphaZero标准学习率
        manual_temperature_decay=True,  # 温度衰减
        grad_clip_value=0.5,
        value_weight=1.0,  # 价值权重
        entropy_weight=0.0,  # 熵权重
        n_episode=n_episode,
        eval_freq=int(200),
        mcts=dict(num_simulations=num_simulations),
        replay_buffer_size=int(2e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
cchess_alphazero_config = EasyDict(cchess_alphazero_config)
main_config = cchess_alphazero_config

cchess_alphazero_create_config = dict(
    env=dict(
        type='cchess',
        import_names=['zoo.board_games.chinesechess.envs.cchess_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero',
        import_names=['lzero.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        import_names=['lzero.worker.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.alphazero_evaluator'],
    )
)
cchess_alphazero_create_config = EasyDict(cchess_alphazero_create_config)
create_config = cchess_alphazero_create_config

if __name__ == '__main__':
    if main_config.policy.tensor_float_32:
        import torch

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
