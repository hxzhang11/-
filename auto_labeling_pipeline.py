import os
import torch
import numpy as np
import os
import torch
import numpy as np
import pandas as pd
from itertools import combinations
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ================= 1. 配置参数 =================
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat" 
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
N_SAMPLES = 5
TEMPERATURE = 0.8     
MAX_NEW_TOKENS = 128  
OUTPUT_FILE = "large_scale_difficulty_dataset_transformers.csv"
SAVE_INTERVAL = 50    

# ================= 2. 准备混合数据集 =================
def prepare_mixed_dataset(num_samples_per_task=100):
    print("📦 正在从 HuggingFace 加载混合数据集...")
    prompts = []
    labels = []

    try:
        gsm8k = load_dataset("gsm8k", "main", split=f"train[:{num_samples_per_task}]")
        for item in gsm8k:
            prompts.append(f"请解答以下数学题，并给出推导过程：\n{item['question']}")
            labels.append("Math_Reasoning")
    except Exception as e:
        print(f"GSM8K 加载失败，请检查网络: {e}")

    try:
        squad = load_dataset("squad", split=f"train[:{num_samples_per_task}]")
        for item in squad:
            prompts.append(f"根据以下背景回答问题：\n背景：{item['context']}\n问题：{item['question']}")
            labels.append("Fact_Extraction")
    except Exception as e:
        print(f"SQuAD 加载失败，请检查网络: {e}")
        
    return prompts, labels

# ================= 3. 主程序 =================
def main():
    prompts, task_labels = prepare_mixed_dataset(num_samples_per_task=100)
    if not prompts:
        print("❌ 没有加载到任何数据，程序退出。")
        return
        
    print(f"✅ 成功加载 {len(prompts)} 条混合测试数据。")

    # ================= 【内存优化：仅加载历史 Prompt 用于查重】 =================
    processed_prompts = set()
    
    if os.path.exists(OUTPUT_FILE):
        try:
            # 只读取 Prompt 列以节省内存
            existing_df = pd.read_csv(OUTPUT_FILE, usecols=['Prompt'])
            processed_prompts = set(existing_df['Prompt'].tolist())
            print(f"🔄 检测到历史文件，已跳过 {len(processed_prompts)} 条已处理数据。")
        except Exception as e:
            print(f"⚠️ 读取历史文件失败，将从头开始: {e}")
            # 如果文件损坏，可能需要重命名或删除老文件
    # =========================================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 初始化 Transformers 引擎 (设备: {device})...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)

    print("📏 初始化语义向量模型...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"🧠 开始评估，目标总量: {len(prompts)} ...")
    
    # 临时存放新生成的数据，每次保存后都会清空
    batch_results = [] 
    
    for i, prompt in enumerate(tqdm(prompts, desc="计算语义熵与难度")):
        
        # 1. 跳过已处理数据
        if prompt in processed_prompts:
            continue

        # 2. 构造模型输入
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 3. 生成 N 个采样回答
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                num_return_sequences=N_SAMPLES,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # 4. 截取新生成的部分
        input_length = model_inputs.input_ids.shape[1]
        generated_ids = [output_ids[input_length:] for output_ids in generated_ids]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 5. 计算一致性与难度得分
        embeddings = embedder.encode(responses, convert_to_tensor=True)
        similarities = [util.cos_sim(embeddings[x], embeddings[y]).item() 
                        for x, y in combinations(range(N_SAMPLES), 2)]
            
        difficulty_score = 1.0 - np.mean(similarities)
        
        # 6. 将当前新跑出的结果放入 batch 列表
        batch_results.append({
            "Task_Type": task_labels[i],
            "Prompt": prompt,
            "Difficulty_Score": round(difficulty_score, 4)
        })

        # ================= 【核心优化：追加模式保存并清空内存】 =================
        # 当攒够 SAVE_INTERVAL 条，或者到了最后一条数据时，触发保存
        if len(batch_results) >= SAVE_INTERVAL or (i + 1) == len(prompts):
            if batch_results: # 确保列表不为空
                df = pd.DataFrame(batch_results)
                # 判断是否需要写表头：如果文件不存在，说明是第一次写，需要表头
                write_header = not os.path.exists(OUTPUT_FILE)
                # 使用 mode='a' 追加写入
                df.to_csv(OUTPUT_FILE, mode='a', index=False, header=write_header)
                
                # 保存完毕，清空 batch_results 释放内存
                batch_results = [] 
        # =======================================================================

    print(f"\n💾 增量处理完成！所有数据已安全落盘至 {OUTPUT_FILE}")
    
    # 任务结束后，完整读取一次用于最终的统计打印
    try:
        final_df = pd.read_csv(OUTPUT_FILE)
        print("\n📊 各类任务的平均难度得分统计：")
        print(final_df.groupby('Task_Type')['Difficulty_Score'].mean())
    except Exception as e:
        print(f"统计打印失败: {e}")

if __name__ == "__main__":
    main()