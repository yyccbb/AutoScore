import numpy as np

def calculate_psi(logprobs_list, true_label):
    """
    计算误置信度 psi: max(log P_wrong) / log P_correct
    """
    # 查找正确标签和错误标签中概率最高的 logprob
    log_p_correct = -np.inf
    log_p_wrong_max = -np.inf
    
    # 遍历 token 级 logprobs (逻辑需匹配你的评分数字 token)
    for entry in logprobs_list:
        # 寻找代表分数的 token (如 '12', '13')
        token = entry.token.strip()
        lp = entry.logprob
        
        if token == str(true_label):
            log_p_correct = lp
        else:
            if lp > log_p_wrong_max:
                log_p_wrong_max = lp

    # 公式: psi = max(log P_wrong) / log P_correct
    # 注意: logprob 是负值，值越大(越接近0)概率越高
    return log_p_wrong_max / (log_p_correct + 1e-9)

def calculate_zeta(logprobs_results):
    """计算测试集置信度指标 zeta"""
    avg_conf = np.mean([res['max_logprob'] for res in logprobs_results])
    return avg_conf