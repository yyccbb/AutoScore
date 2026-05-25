import numpy as np
import random
from sklearn.metrics import cohen_kappa_score
from utils_asro.utils import _score_to_tier

class ASROEvaluator:
    def __init__(self, client, max_score=15.0, tier_count=5, misconf_tier_weight=10.0):
        self.client = client
        self.max_score = float(max_score)
        self.tier_count = int(tier_count)
        self.misconf_tier_weight = float(misconf_tier_weight)

    def calculate_kappa(self, guideline, dataset):
        y_true, y_pred = [], []
        for sample in dataset:
            score, _, _, _ = self.client.get_ordinal_score(sample['text'], guideline)
            y_true.append(int(round(sample['true_score'] * 2)))
            y_pred.append(int(round(score * 2)))
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')

    def evaluate_minibatch(self, minibatch, guideline):
        results = []
        for sample in minibatch:
            pred_score, lp_pred, lp_true, reasoning = self.client.get_ordinal_score(
                sample['text'], guideline, sample['true_score']
            )
            is_correct = (int(round(pred_score * 2)) == int(round(sample['true_score'] * 2)))
            true_tier = _score_to_tier(sample['true_score'], self.max_score, self.tier_count)
            pred_tier = _score_to_tier(pred_score, self.max_score, self.tier_count)
            fallback_misconf = ((sample['true_score'] - pred_score) ** 2) + self.misconf_tier_weight * (abs(true_tier - pred_tier) ** 2)
            if lp_pred == -1.0 and lp_true == -1.0:
                m_val = fallback_misconf
            else:
                m_val = -lp_pred if is_correct else max(fallback_misconf, (sample['true_score'] - pred_score)**2 * (lp_pred - lp_true))
            
            results.append({
                "true": sample['true_score'], "pred": pred_score,
                "prob": np.exp(lp_pred), "misconf": m_val,
                "text": sample['text'], "reasoning": reasoning
            })
        return results

    def get_top_k_modes(self, results, k):
        y_true_indices = [int(round(r['true'] * 2)) for r in results]
        y_pred_indices = [int(round(r['pred'] * 2)) for r in results]
        labels = sorted(list(set(y_true_indices + y_pred_indices)))
        if not labels: return []
        
        matrix_size = max(int(round(self.max_score * 2)) + 1, max(labels) + 1)
        weighted_cm = np.zeros((matrix_size, matrix_size))
        for r in results:
            t_idx, p_idx = int(round(r['true'] * 2)), int(round(r['pred'] * 2))
            if t_idx != p_idx and 0 <= t_idx < matrix_size and 0 <= p_idx < matrix_size:
                weighted_cm[t_idx][p_idx] += r['misconf']
        
        flat_weighted = weighted_cm.flatten()
        top_indices = np.argsort(flat_weighted)[::-1][:k]
        return [divmod(idx, matrix_size) for idx in top_indices if flat_weighted[idx] > 0]
