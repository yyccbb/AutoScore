from concurrent.futures import ThreadPoolExecutor

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from utils_asro.progress import log_progress
from utils_asro.utils import _score_to_tier


class ASROSampler:
    def __init__(self, model_name="all-MiniLM-L6-v2", max_score=15.0, tier_count=5, misconf_tier_weight=10.0, max_workers=10):
        self.max_score = float(max_score)
        self.tier_count = int(tier_count)
        self.misconf_tier_weight = float(misconf_tier_weight)
        self.max_workers = max(1, int(max_workers))
        log_progress("sampler", "loading SentenceTransformer", model=model_name)
        self.model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            log_progress("sampler", "SentenceTransformer moved to CUDA", model=model_name)
        log_progress("sampler", "SentenceTransformer ready", model=model_name)

    def _score_samples(self, D_train, current_g, client):
        log_progress("sampler", "scoring training samples for sampling", samples=len(D_train), workers=self.max_workers)
        if hasattr(client, "get_ordinal_score_batch"):
            results = client.get_ordinal_score_batch(
                D_train,
                current_g,
                max_workers=self.max_workers,
                use_multithread=self.max_workers > 1,
            )
            log_progress("sampler", "training sample scoring finished", samples=len(results))
            return results

        score_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(client.get_ordinal_score, s["text"], current_g, s["true_score"]) for s in D_train]
            for future in tqdm(futures, desc="Sampling", leave=False, ncols=70):
                score_results.append(future.result())
        log_progress("sampler", "training sample scoring finished", samples=len(score_results))
        return score_results

    def sample_minibatch(self, D_train, current_g, client, k=5, batch_size=40):
        if len(D_train) <= batch_size:
            log_progress("sampler", "using full training set as minibatch", train=len(D_train), batch_size=batch_size)
            return D_train

        scored_samples = []
        score_results = self._score_samples(D_train, current_g, client)
        for i, (score, lp_pred, lp_true, _) in enumerate(tqdm(score_results, desc="Sampling", leave=False, ncols=70)):
            is_correct = int(round(score * 2)) == int(round(D_train[i]["true_score"] * 2))
            true_tier = _score_to_tier(D_train[i]["true_score"], self.max_score, self.tier_count)
            pred_tier = _score_to_tier(score, self.max_score, self.tier_count)
            fallback_misconf = ((D_train[i]["true_score"] - score) ** 2) + self.misconf_tier_weight * (abs(true_tier - pred_tier) ** 2)
            if lp_pred == -1.0 and lp_true == -1.0:
                m_val = fallback_misconf
            else:
                m_val = -lp_pred if is_correct else max(fallback_misconf, (D_train[i]["true_score"] - score) ** 2 * (lp_pred - lp_true))
            scored_samples.append({"sample": D_train[i], "misconf": m_val, "text": D_train[i]["text"]})

        seeds = sorted(scored_samples, key=lambda x: x["misconf"], reverse=True)[:k]
        log_progress("sampler", "encoding seed and sample embeddings", seeds=len(seeds), samples=len(scored_samples))
        seed_emb = self.model.encode([s["text"] for s in seeds], convert_to_tensor=True)
        all_emb = self.model.encode([s["text"] for s in scored_samples], convert_to_tensor=True)

        cosine_scores = util.cos_sim(seed_emb, all_emb) # cos similarities (K, #all)
        neighbor_indices = torch.topk(cosine_scores, k=min(batch_size // k, len(scored_samples)), dim=1).indices.flatten().tolist() # sample #batch_size high similarity samples for each of the top k samples. The top K samples are usually included inside all_emb hence chosen here.
        minibatch = [scored_samples[i]["sample"] for i in set(neighbor_indices)]
        log_progress("sampler", "minibatch neighbors selected", minibatch=len(minibatch), requested=batch_size)
        return minibatch
