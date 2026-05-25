import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score

def analyze_ASRO_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 假设数据格式是 [{'true': 12.0, 'pred': 10.5, 'prob': 0.8}, ...]
    df = pd.DataFrame(data['details'])
    
    # 1. 计算 QWK
    y_true = (df['true'] * 2).astype(int)
    y_pred = (df['pred'] * 2).astype(int)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    print(f"📊 Current QWK: {qwk:.4f}")

    # 2. 绘制混淆矩阵热力图
    plt.figure(figsize=(12, 10))
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 归一化，看每个真分档位的预测分布
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=[l/2 for l in labels], yticklabels=[l/2 for l in labels])
    plt.title(f"Confusion Matrix (QWK: {qwk:.4f})")
    plt.xlabel("Predicted Score")
    plt.ylabel("True Score")
    plt.savefig("confusion_heatmap.png")
    
    # 3. 置信度分析 (ASRO 核心)
    plt.figure(figsize=(10, 6))
    df['is_correct'] = df['true'] == df['pred']
    sns.kdeplot(data=df, x='prob', hue='is_correct', fill=True)
    plt.title("Confidence (Logprobs) Distribution: Correct vs Incorrect")
    plt.savefig("confidence_dist.png")
    
    print("✅ 诊断图表已生成：confusion_heatmap.png, confidence_dist.png")

