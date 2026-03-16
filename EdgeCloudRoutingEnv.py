import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EdgeCloudRoutingEnv(gym.Env):
    def __init__(self, vd_path="real_vd_features.npy", diff_path="real_difficulties.npy"):
        super(EdgeCloudRoutingEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        # 注意：真实网络输出的 v_d 范围可能不是严丝合缝的 [-1, 1]，所以这里放宽 Box 的边界
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(35,), dtype=np.float32)
        
        # 加载真实的特征库
        self.real_vds = np.load(vd_path)
        self.real_diffs = np.load(diff_path)
        self.num_samples = len(self.real_vds)
        
        self.w_acc = 10.0
        self.w_lat = 5.0
        self.w_cost = 2.0
        
        self.current_state = None
        self.current_real_difficulty = 0.0 # 用于准确计算 Reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 🎯 从真实特征库中随机抽取一个任务
        sample_idx = self.np_random.integers(0, self.num_samples)
        task_features = self.real_vds[sample_idx]
        self.current_real_difficulty = self.real_diffs[sample_idx] # 记录该任务的真实归一化难度
        
        # 物理状态依然使用随机生成，因为我们需要模拟各种极端的网速和负载
        network_congestion = self.np_random.uniform(0.0, 1.0) 
        edge_load = self.np_random.uniform(0.0, 1.0)          
        cloud_load = self.np_random.uniform(0.0, 1.0)         
        sys_features = np.array([network_congestion, edge_load, cloud_load])
        
        self.current_state = np.concatenate((task_features, sys_features)).astype(np.float32)
        return self.current_state, {}

    def step(self, action):
        net_congestion = self.current_state[32]
        edge_load = self.current_state[33]
        cloud_load = self.current_state[34]
        
        # 真实难度 (0.0 ~ 1.0)
        task_difficulty = self.current_real_difficulty 
        
        acc_score, lat_score, cost_score = 0.0, 0.0, 0.0
        
        # ================= 重新设计的工业级物理反馈 =================
        if action == 0:  # 【动作 0：本地直出】
            # 甜头：难度低于 0.3 时，端侧完美解决，拿满分！
            # 惩罚：难度高于 0.5 时，端侧胡言乱语，精度呈指数崩盘
            # 只要难度超过 0.5，端侧的精度得分将呈断崖式下跌，甚至扣成负分！
            acc_score = 1.0 if task_difficulty < 0.3 else max(-1.0, 1.0 - 3.5 * (task_difficulty - 0.3)**2)
            lat_score = 0.5 * edge_load 
            cost_score = 0.0 # 绝对省钱
            
        elif action == 1:  # 【动作 1：云端全量】
            acc_score = 1.0 
            # 惩罚：网络拥塞大于 0.7 时，时延惩罚呈指数级爆炸 (模拟 API 请求超时)
            lat_score = (np.exp(net_congestion * 2) / 7.3) + 0.5 * cloud_load
            cost_score = 1.0 
            
        elif action == 2:  # 【动作 2：任务分解】
            acc_score = 0.85 
            # 平摊风险，都不会发生指数爆炸
            lat_score = 0.8 * edge_load + 0.8 * net_congestion
            cost_score = 0.3
            
        elif action == 3:  # 【动作 3：端侧解码+云侧验证 (削弱神坛)】
            acc_score = 0.95 
            # 削弱：如果题目太难 (难度高)，端侧极大概率猜错。
            # 猜错的代价是：既吃了端侧的时延，又吃了云端的网络时延，属于“双输”！
            guess_wrong_prob = task_difficulty
            # 惩罚加倍：猜错时的网络惩罚比直接上云还惨
            lat_score = (0.5 * edge_load) + guess_wrong_prob * (1.5 * np.exp(net_congestion))
            cost_score = 0.6
            
        # ========================================================
        
        reward = (self.w_acc * acc_score) - (self.w_lat * lat_score) - (self.w_cost * cost_score)
        
        info = {
            "real_difficulty": task_difficulty,
            "acc_score": acc_score,
            "lat_penalty": lat_score,
            "cost_penalty": cost_score
        }
        
        next_state, _ = self.reset()
        return next_state, float(reward), True, False, info

# 测试环境是否符合 Gymnasium 规范
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    env = EdgeCloudRoutingEnv()
    check_env(env, warn=True)
    print("✅ 环境检查通过！完美符合 Gymnasium 标准。")