import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# 导入你刚刚写好的环境 (假设你保存在了 EdgeCloudRoutingEnv.py 中)
# 如果是写在同一个文件，直接把类的代码贴在这上面即可
from EdgeCloudRoutingEnv import EdgeCloudRoutingEnv

def train_ppo_router():
    print("🌍 正在初始化端云协同路由环境...")
    # 使用 make_vec_env 将环境向量化，SB3 训练需要这种格式
    env = make_vec_env(lambda: EdgeCloudRoutingEnv(), n_envs=1)

    print("🧠 正在构建 PPO 强化学习大脑...")
    # 初始化 PPO 模型
    # MlpPolicy: 使用多层感知机作为策略网络 (Actor-Critic)
    # learning_rate: 学习率，RL通常设置得比较小
    # n_steps: 每次更新前收集的经验步数
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99 # 折扣因子，因为我们的路由是单步决策，这个影响不大
    )

    print("🔥 开始在虚拟世界中进行毒打与试错 (Training)...")
    # 让 Agent 在环境里玩 50,000 步
    # 你可以在控制台看到 ep_rew_mean (平均奖励) 在稳步上升
    model.learn(total_timesteps=50000)

    print("✅ 训练完成！正在保存模型...")
    model.save("ppo_dynamic_router")

    # ================= 评估模型 =================
    print("\n📊 正在评估训练后的策略表现...")
    # evaluate_policy 会跑 100 个回合，看看平均能拿多少分
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"100次随机测试的平均 Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model

def test_router_in_action(model):
    print("\n🔮 【实战演练】观察路由器的真实决策逻辑：")
    env = EdgeCloudRoutingEnv()
    
    action_map = {
        0: "📱 本地直出 (Edge Only)",
        1: "☁️ 云端全量 (Cloud Only)",
        2: "🧩 任务分解 (Decompose)",
        3: "🤝 端云验证 (Edge+Cloud Verify)"
    }

    for i in range(5):
        obs, _ = env.reset()
        
        # 1. AI 做决策
        action, _states = model.predict(obs, deterministic=True)
        
        # 2. 环境推演
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. 提取真实状态
        net_congestion = obs[32]
        edge_load = obs[33]
        real_difficulty = info['real_difficulty']
        
        # 4. 打印完整战报
        print(f"\n--- 场景 {i+1} ---")
        print(f"📥 状态观察 | 真实认知难度: {real_difficulty:.2f} | 网络拥塞: {net_congestion:.2f} | 手机负载: {edge_load:.2f}")
        print(f"🎯 智能体决策: {action_map[action.item()]}")
        print(f"💰 决策结果 | 精度得分: {info['acc_score']:.2f} | 时延惩罚: -{info['lat_penalty']:.2f} | 总收益: {reward:.2f}")
if __name__ == "__main__":
    # 1. 训练模型
    trained_model = train_ppo_router()
    
    # 2. 观察决策行为
    test_router_in_action(trained_model)