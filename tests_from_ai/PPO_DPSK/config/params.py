# config/params.py
training_config = {
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'ppo_epochs': 4,
    'batch_size': 64,
    'memory_size': 1000,
    'hidden_dim': 128,
    'num_episodes': 1000,
    'max_steps': 1000,
    'log_interval': 10
}

env_config = {
    'base_search_speed': 1.0,
    'move_penalty_factor': 0.3,
    'search_reward_factor': 1.0,
    'move_to_search_bonus': 0.3, # 获取部分的搜索奖励作为移动激励
    'idle_penalty_factor': 0.5,
    'idle_timeout': 5.0,
    'max_time': 1000.0,
    'max_steps': 1000,
    'max_task_time': 20,
    'max_area': 100,
}

model_config = {
    'node_feature_in': 8,
    'drone_feature_in': 6,
    'feature_out': 128,
    'hidden_dim': 128,
    'num_heads': 4,
}