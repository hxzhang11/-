import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
# 导入你的评估器网络
from train_evaluator import MultiTaskDifficultyEvaluator

def extract_and_save_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ 加载真实评估器模型 (设备: {device})...")
    
    # 初始化模型并加载你微调好的权重 (注意 num_classes 要对应)
    model = MultiTaskDifficultyEvaluator(num_classes=2).to(device)
    model.load_state_dict(torch.load("multitask_evaluator_finetuned.pth", map_location=device))
    model.eval()

    print("📂 加载原始数据集...")
    df = pd.read_csv("large_scale_difficulty_dataset_transformers.csv")
    prompts = df['Prompt'].tolist()

    all_vd = []
    all_difficulties = []

    print("🧠 开始提取真实的 32 维特征向量...")
    with torch.no_grad():
        for prompt in tqdm(prompts):
            v_d, pred_ent, _, _ = model([prompt])
            
            all_vd.append(v_d.cpu().numpy().flatten())
            all_difficulties.append(pred_ent.item())

    # 将 List 转换为 Numpy 数组
    all_vd = np.array(all_vd)
    all_difficulties = np.array(all_difficulties)

    # 🛠️ 关键一步：将极小的真实熵归一化到 0~1 之间，方便 PPO 的 Reward 计算
    min_ent, max_ent = all_difficulties.min(), all_difficulties.max()
    normalized_difficulties = (all_difficulties - min_ent) / (max_ent - min_ent + 1e-8)

    # 保存到本地硬盘
    np.save("real_vd_features.npy", all_vd)
    np.save("real_difficulties.npy", normalized_difficulties)
    
    print(f"\n✅ 提取完成！")
    print(f"保存了 {len(all_vd)} 条真实特征向量，维度: {all_vd.shape}")
    print(f"真实难度范围: {min_ent:.4f} ~ {max_ent:.4f} (已归一化到 0~1)")

if __name__ == "__main__":
    extract_and_save_features()