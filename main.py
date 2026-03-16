import torch
import numpy as np
from stable_baselines3 import PPO

# 导入阶段一写好的多任务评估器网络结构
from train_evaluator import MultiTaskDifficultyEvaluator

class EndToEndRouter:
    def __init__(self, evaluator_path=None, ppo_path="ppo_dynamic_router"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ 系统初始化中... (设备: {self.device})")
        
        # ================= 1. 加载阶段一：多维难度评估器 (雷达) =================
        print("📡 加载多任务特征提取网络...")
        # 注意：这里的 num_classes 要和你训练时 CSV 里的分类数量一致 (比如 2)
        self.evaluator = MultiTaskDifficultyEvaluator(num_classes=2).to(self.device)
        if evaluator_path:
            # 如果你之前保存了微调后的权重，就在这里加载
            self.evaluator.load_state_dict(torch.load(evaluator_path, map_location=self.device))
        self.evaluator.eval() # 切换到推理模式
        
        # ================= 2. 加载阶段二：动态自适应路由器 (大脑) =================
        print("🧠 加载 PPO 强化学习路由决策模型...")
        self.router = PPO.load(ppo_path)
        
        # 定义动作的语义映射
        self.action_map = {
            0: "📱 本地直出 (Edge Only) - 适合简单任务，零网络延迟",
            1: "☁️ 云端全量 (Cloud Only) - 适合高难度任务，需全额 API 成本",
            2: "🧩 任务分解 (Decompose) - 适合复杂长文本，端云协同分担",
            3: "🤝 端侧推测+云侧验证 (Edge+Cloud Verify) - 适合中高难度，防范端侧幻觉"
        }
        print("✅ 端云协同智能网关已就绪！\n")

    def route_query(self, query: str, net_congestion: float, edge_load: float, cloud_load: float):
        """
        核心处理管道：接收文本和物理状态，输出最终路由决策
        """
        print(f"[{'='*40}]")
        print(f"📥 接收到新请求: '{query}'")
        print(f"📊 当前系统物理状态 | 网络拥塞: {net_congestion:.2f} | 手机负载: {edge_load:.2f} | 云端队列: {cloud_load:.2f}")
        
        # --- Step 1: 提取 32 维任务特征 (v_d) ---
        with torch.no_grad():
            v_d, pred_ent, pred_len, pred_dom = self.evaluator([query])
            v_d_np = v_d.cpu().numpy().flatten()
            
            # 获取绝对原始难度
            raw_difficulty = pred_ent.item()
            
            # ✨ 新增：映射到 0~1 的人类直觉区间 (假设你数据集里的真实极值是 0.04 和 0.15)
            min_ent, max_ent = 0.07, 0.1100 # 这里的范围需要根据数据集里的真实难度分布来调整
            human_readable_diff = (raw_difficulty - min_ent) / (max_ent - min_ent + 1e-8)
            human_readable_diff = max(0.0, min(1.0, human_readable_diff)) # 截断
            
            # 打印映射后的分数
            print(f"🔍 评估器深度透视 | 预测认知难度: {human_readable_diff:.2f} (绝对底层值: {raw_difficulty:.4f})")
            
        # --- Step 2: 组装 35 维 RL 状态空间 ---
        sys_features = np.array([net_congestion, edge_load, cloud_load], dtype=np.float32)
        rl_state = np.concatenate((v_d_np, sys_features))
        
        # --- Step 3: PPO 智能体进行最终决策 ---
        # deterministic=True 确保在部署阶段输出概率最大的最优动作
        action, _states = self.router.predict(rl_state, deterministic=True)
        final_decision = self.action_map[action.item()]
        
        print(f"🚀 最终路由调度策略: \n>> {final_decision}")
        print(f"[{'='*40}]\n")
        
        return action.item()

# ================= 测试实战场景 =================
if __name__ == "__main__":
    # 初始化网关 (如果有保存的 evaluator 权重，传入 evaluator_path="...")
    gateway = EndToEndRouter(evaluator_path="multitask_evaluator_finetuned.pth",ppo_path="ppo_dynamic_router")
    
    # --- 场景 A：网络极好，且任务是极高难度的算法题 ---
    query_a = "给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请使用双指针法实现并分析时间复杂度。"
    gateway.route_query(query=query_a, net_congestion=0.1, edge_load=0.2, cloud_load=0.3)
    
    # --- 场景 B：在极度弱网（地铁里）且手机卡顿，问一个简单的闲聊问题 ---
    query_b = "今天天气真好，我想出去散步，你能给我写一句发朋友圈的文案吗？"
    gateway.route_query(query=query_b, net_congestion=0.9, edge_load=0.8, cloud_load=0.2)
    
    # --- 场景 C：经典的 RAG 冲突处理（中高难度），网络一般 ---
    query_c = "在检索增强生成 (RAG) 的流程中，如果通过 BGE-Reranker 重排后，排名前两位的上下文片段存在严重的信息冲突，系统应该如何进行自适应纠错？"
    gateway.route_query(query=query_c, net_congestion=0.4, edge_load=0.3, cloud_load=0.5)