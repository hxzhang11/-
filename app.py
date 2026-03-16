import streamlit as st
import torch
import numpy as np
from stable_baselines3 import PPO

# 导入你的评估器网络
from train_evaluator import MultiTaskDifficultyEvaluator

# ================= 1. 缓存加载模型 (避免拖动滑块时重复加载) =================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载阶段一：难度评估器 (注意 num_classes=2 要和训练时一致)
    evaluator = MultiTaskDifficultyEvaluator(num_classes=2).to(device)
    evaluator.load_state_dict(torch.load("multitask_evaluator_finetuned.pth", map_location=device))
    evaluator.eval()
    
    # 2. 加载阶段二：PPO 路由器
    router = PPO.load("ppo_dynamic_router")
    
    return evaluator, router, device

evaluator, router, device = load_models()

# 完美贴合你数据的真实包络线
MIN_ENT = 0.0700
MAX_ENT = 0.1000  # 把 0.11 改成 0.10

# ================= 2. 页面 UI 设计 =================
st.set_page_config(page_title="端云协同智能路由", page_icon="🧠", layout="wide")

st.title("🧠 大模型端云协同智能路由网关")
st.markdown("基于 **多维语义特征 (BGE-small)** 与 **PPO 强化学习** 的动态自适应算力分发系统。")
st.divider()

# 左侧控制面板：物理状态模拟
with st.sidebar:
    st.header("🎛️ 物理环境模拟器")
    st.caption("拖动滑块模拟极端的物理约束条件")
    net_congestion = st.slider("🌐 网络拥塞度 (越近1越卡)", 0.0, 1.0, 0.10, 0.05)
    edge_load = st.slider("📱 端侧负载 (越近1手机越卡)", 0.0, 1.0, 0.20, 0.05)
    cloud_load = st.slider("☁️ 云端队列 (越近1 API越慢)", 0.0, 1.0, 0.30, 0.05)
    
    st.divider()
    st.markdown("### 💡 预设场景测试")
    st.markdown("- **算法题 + 网好** 👉 预期: 云端全量")
    st.markdown("- **闲聊 + 网卡** 👉 预期: 本地直出")
    st.markdown("- **长文本知识 + 适中状态** 👉 预期: 任务分解 / 端云协同")

# 主面板：用户提问区
query = st.text_area("✍️ 请输入用户的 Prompt 提问：", height=120, 
                     value="给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请使用双指针法实现并分析时间复杂度。")

# 动作字典 (带详细解释和 UI 颜色标记)
action_map = {
    0: ("📱 本地直出 (Edge Only)", "端侧独立完成计算。通常用于简单任务，或在极端弱网下被逼无奈的保底。", "success"),
    1: ("☁️ 云端全量 (Cloud Only)", "高难度任务，为防止端侧严重幻觉，全量卸载至云端。", "error"),
    2: ("🧩 任务分解 (Decompose)", "弱网下的复杂任务，端侧与云侧切分协同计算以平摊风险。", "warning"),
    3: ("🤝 端侧推测+云侧验证", "平衡方案，端侧生成草稿云端把关，防范幻觉的兜底策略。", "info")
}

col_btn, _ = st.columns([1, 4])
with col_btn:
    submit = st.button("🚀 执行智能路由决策", type="primary", use_container_width=True)

if submit:
    if not query.strip():
        st.warning("⚠️ 请输入有效的问题！")
    else:
        with st.spinner("系统正在进行高维特征提取与 RL 纳什博弈..."):
            
            # --- 1. 提取任务特征 (Phase 1) ---
            with torch.no_grad():
                v_d, pred_ent, _, _ = evaluator([query])
                v_d_np = v_d.cpu().numpy().flatten()
                
                # 获取真实底层熵值，并映射到人类易读的 0~1 区间
                raw_difficulty = pred_ent.item()
                norm_difficulty = (raw_difficulty - MIN_ENT) / (MAX_ENT - MIN_ENT + 1e-8)
                norm_difficulty = max(0.0, min(1.0, norm_difficulty))
            
            # --- 2. 组装 35 维 RL 状态 (Phase 2) ---
            sys_features = np.array([net_congestion, edge_load, cloud_load], dtype=np.float32)
            rl_state = np.concatenate((v_d_np, sys_features))
            
           # --- 3. PPO 决策 ---
            action, _ = router.predict(rl_state, deterministic=True)
            final_action = action.item()
            
            # 🛡️ 工业级护栏 (Guardrails)：红线严格对齐！
            if norm_difficulty > 0.65 and final_action == 0:
                st.toast("🛡️ 触发安全护栏：任务难度超纲，拦截端侧裸跑，强制端云协同！", icon="🚨")
                final_action = 3 
                
            elif norm_difficulty < 0.20 and final_action == 1:
                st.toast("🛡️ 触发安全护栏：拦截不必要的云端开销，强制本地消化！", icon="🚨")
                final_action = 0
                
            decision_name, decision_desc, msg_type = action_map[final_action]
            
            # --- 4. 结果炫酷展示 ---
            st.divider()
            st.markdown("### 📊 决策透视面板")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔍 阶段一：多维特征感知雷达")
                st.metric(label="任务认知难度指数", value=f"{norm_difficulty:.2f}", delta=f"底层绝对熵值: {raw_difficulty:.4f}", delta_color="off")
                st.progress(norm_difficulty)
                if norm_difficulty > 0.7:
                    st.error("⚠️ 警告：该任务对端侧 SLM 极易产生严重幻觉！")
                elif norm_difficulty < 0.3:
                    st.success("✅ 安全：该任务处在端侧 SLM 的舒适区。")
                else:
                    st.info("ℹ️ 中等：端侧 SLM 可能出现逻辑断层，建议协同。")
                
            with col2:
                st.markdown("#### 🎯 阶段二：RL 动态调度策略")
                
                # 根据不同的动作给出不同的颜色提示框
                if msg_type == "success": st.success(decision_name)
                elif msg_type == "error": st.error(decision_name)
                elif msg_type == "warning": st.warning(decision_name)
                else: st.info(decision_name)
                
                st.markdown(f"**💡 策略解析：** {decision_desc}")
                
                # 展示 RL 眼中的环境参数
                st.caption(f"环境参数回传 ➔ 网络拥塞: {net_congestion:.2f} | 端侧负载: {edge_load:.2f} | 云端队列: {cloud_load:.2f}")