import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd

# ================= 1. 真实数据加载与处理 (读取 CSV) =================
class RealMultiTaskDataset(Dataset):
    def __init__(self, csv_file, max_length_norm=1000.0):
        print(f"📂 正在加载真实数据集: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # 1. 读取 Prompt
        self.prompts = df['Prompt'].astype(str).tolist()
        
        # 2. 读取难度得分 (认知熵)
        self.entropies = df['Difficulty_Score'].tolist()
        
        # 3. 动态计算长度并归一化 (防止 Length Loss 爆炸)
        self.lengths = [min(len(p) / max_length_norm, 1.0) for p in self.prompts]
        
        # 4. 动态映射领域分类 (Domain)
        # 将 CSV 里的文字标签 (如 "Math_Reasoning") 映射为数字 (0, 1, 2...)
        self.domains = []
        
        # 自动扫描 CSV 中有多少种任务类型，并生成映射表
        if 'Task_Type' in df.columns:
            unique_tasks = df['Task_Type'].unique().tolist()
            self.task2id = {task: idx for idx, task in enumerate(unique_tasks)}
            self.id2task = {idx: task for task, idx in self.task2id.items()}
            print(f"🏷️ 发现 {len(unique_tasks)} 种任务类型: {self.task2id}")
            
            for task in df['Task_Type']:
                self.domains.append(self.task2id[task])
        else:
            print("⚠️ 警告: CSV 中没有找到 'Task_Type' 列，默认全部归为一类。")
            self.domains = [0] * len(df)
            self.id2task = {0: "Unknown"}

        # 记录分类的总数，用于构建网络的 Domain Head
        self.num_classes = max(1, len(self.id2task))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return (
            self.prompts[idx], 
            torch.tensor(self.entropies[idx], dtype=torch.float32),
            torch.tensor(self.lengths[idx], dtype=torch.float32),
            torch.tensor(self.domains[idx], dtype=torch.long)
        )

# ================= 2. 核心网络 (自动适应分类数量) =================
class MultiTaskDifficultyEvaluator(nn.Module):
    def __init__(self, num_classes=3, embedding_model="BAAI/bge-small-zh-v1.5", structural_dim=3, hidden_dim=64, v_d_dim=32):
        super(MultiTaskDifficultyEvaluator, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.encoder = AutoModel.from_pretrained(embedding_model)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        semantic_dim = self.encoder.config.hidden_size
        input_dim = semantic_dim + structural_dim
        
        # 32 维强壮特征空间 (v_d)
        self.fc_fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, v_d_dim)
        )
        
        # 三个监督头
        self.cognitive_head = nn.Sequential(nn.Linear(v_d_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.compute_head = nn.Sequential(nn.Linear(v_d_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        # 根据 CSV 中真实的任务类型数量动态调整分类头
        self.domain_head = nn.Sequential(nn.Linear(v_d_dim, 16), nn.ReLU(), nn.Linear(16, num_classes)) 

    def extract_structural_features(self, texts):
        features = []
        for text in texts:
            length = len(text) / 500.0 
            punct_density = sum(text.count(p) for p in ["?", "：", ","]) / (len(text) + 1)
            logic_words = sum(text.count(w) for w in ["因为", "推导", "求", "已知"]) / (len(text) + 1)
            features.append([length, punct_density, logic_words])
        return torch.tensor(features, dtype=torch.float32)

    def forward(self, texts):
        device = next(self.parameters()).device
        
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        with torch.no_grad():
            semantic_embeds = self.encoder(**encoded_input)[0][:, 0, :]
            
        struct_embeds = self.extract_structural_features(texts).to(device)
        combined_features = torch.cat((semantic_embeds, struct_embeds), dim=1)
        
        v_d = self.fc_fusion(combined_features)
        
        pred_entropy = self.cognitive_head(v_d).squeeze()
        pred_length = self.compute_head(v_d).squeeze()
        pred_domain = self.domain_head(v_d)
        
        return v_d, pred_entropy, pred_length, pred_domain

# ================= 3. 训练循环 =================
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_mtl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用计算设备: {device}")

    # ... (保持原有的数据加载部分不变) ...
    csv_filename = "large_scale_difficulty_dataset_transformers.csv" 
    dataset = RealMultiTaskDataset(csv_filename)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = MultiTaskDifficultyEvaluator(num_classes=dataset.num_classes).to(device)
    
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    
    # 【微调 1】：稍微降低初始学习率，避免震荡
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    # 【微调 2】：引入学习率调度器。如果总 Loss 连续 3 个 Epoch 不降，学习率减半
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 【微调 3】：重新分配多任务权重（逼迫网络死磕“认知熵”）
    lambda_entropy = 3.0   # 权重拉满，这是核心！
    lambda_length = 0.5    # 保持适中
    lambda_domain = 0.1    # 权重降到极低，因为它太容易学了

    epochs = 20 # 引入 scheduler 后可以多跑几轮
    print("\n🚀 开始微调后的多任务联合训练 (Focusing on Entropy)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_ent_loss, total_len_loss, total_dom_loss = 0, 0, 0, 0
        
        for texts, true_entropies, true_lengths, true_domains in dataloader:
            true_entropies = true_entropies.to(device)
            true_lengths = true_lengths.to(device)
            true_domains = true_domains.to(device)
            
            optimizer.zero_grad()
            v_d, pred_entropy, pred_length, pred_domain = model(texts)
            
            if pred_entropy.dim() == 0:
                pred_entropy = pred_entropy.unsqueeze(0)
                pred_length = pred_length.unsqueeze(0)

            loss_ent = criterion_mse(pred_entropy, true_entropies)
            loss_len = criterion_mse(pred_length, true_lengths)
            loss_dom = criterion_ce(pred_domain, true_domains)
            
            # 使用新的权重计算总 Loss
            loss = (lambda_entropy * loss_ent) + (lambda_length * loss_len) + (lambda_domain * loss_dom)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_ent_loss += loss_ent.item()
            total_len_loss += loss_len.item()
            total_dom_loss += loss_dom.item()
            
        avg_loss = total_loss / len(dataloader)
        avg_ent = total_ent_loss / len(dataloader)
        avg_len = total_len_loss / len(dataloader)
        avg_dom = total_dom_loss / len(dataloader)
        
        # 调度器根据当前 Epoch 的平均熵 Loss 进行学习率衰减
        scheduler.step(avg_ent)
        
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}] | 总Loss: {avg_loss:.4f} | "
                  f"熵: {avg_ent:.4f} | 长: {avg_len:.4f} | 类: {avg_dom:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.5f}")

    print("✅ 微调完成！模型已强制对齐最重要的认知难度维度。")
    torch.save(model.state_dict(), "multitask_evaluator_finetuned.pth")

if __name__ == "__main__":
    train_mtl()