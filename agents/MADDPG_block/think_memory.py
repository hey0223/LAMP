import logging
import numpy as np

from utils.faiss_memory import FaissIndexManager

class ThinkMemoryManager:
    def __init__(self, n_households, short_top_k, long_top_k, retrieval_top_k, faiss_dim=9):
        self.short_top_k = short_top_k
        self.long_top_k = long_top_k
        self.retrieval_top_k = retrieval_top_k
        self.faiss_dim = faiss_dim

        self.short_term_memory = {
            f"Household{i+1}": [] for i in range(n_households)
        }

        self.faiss_manager_long_term = FaissIndexManager(dim=self.faiss_dim, use_gpu=False)

    def reset_short_term_memory(self):
        for hid in self.short_term_memory:
            self.short_term_memory[hid].clear()

    def build_think_query_vector(self, global_obs, private_obs):
        """
        Build the retrieval query vector from the current observation.
        Keep the same dimensionality as FAISS (global + private = 9).
        """
        g = np.array(global_obs, dtype=np.float32).copy()
        p = np.array(private_obs, dtype=np.float32).copy()

        # use the same normalization convention as observation_wrapper
        if g.shape[0] > 0:
            g[0] /= 1e7
        if g.shape[0] > 1:
            g[1] /= 1e5
        if g.shape[0] > 3:
            g[3] /= 1e5
        if g.shape[0] > 4:
            g[4] /= 1e5
        if p.shape[0] > 1:
            p[1] /= 1e5

        return np.concatenate([g, p], axis=0).astype(np.float32)

    def push_to_short_term_memory(self, household_id, experience):
        """
        Update H_short for one household.
        Keep only top-k1 trajectories by reward.
        """
        if experience is None:
            return
        if not experience.get("reasoning_text", ""):
            return

        self.short_term_memory[household_id].append(experience)
        self.short_term_memory[household_id] = sorted(
            self.short_term_memory[household_id],
            key=lambda e: e["reward"],
            reverse=True
        )[: self.short_top_k]

    def harvest_long_term_memory_from_short(self):
        """
        Update H_long only at long-term checkpoints:
        collect top-k2 trajectories across all households' H_short and add to FAISS.
        """
        candidates = []
        for household_id, exps in self.short_term_memory.items():
            for exp in exps:
                if not exp.get("added_to_long_term", False):
                    candidates.append(exp)

        if not candidates:
            logging.info("No fresh short-term experiences to harvest into H_long.")
            return

        top_exps = sorted(candidates, key=lambda e: e["reward"], reverse=True)[: self.long_top_k]
        vectors = [exp["query_vector"] for exp in top_exps if "query_vector" in exp]

        if len(vectors) == 0:
            logging.info("No valid query vectors found for H_long harvest.")
            return

        arr = np.stack(vectors).astype(np.float32)
        self.faiss_manager_long_term.add(arr, top_exps)

        for exp in top_exps:
            exp["added_to_long_term"] = True

        logging.info(f"Harvested {len(top_exps)} top short-term experiences into H_long.")

    def format_experience_for_prompt(self, exp, tag):
        reward = float(exp.get("reward", 0.0))
        reasoning = exp.get("reasoning_text", "")
        priv_obs = exp.get("private_obs", [0.0, 0.0])
        action = exp.get("action", [0.0, 0.0])

        e_val = float(priv_obs[0]) if len(priv_obs) > 0 else 0.0
        wealth_val = float(priv_obs[1]) if len(priv_obs) > 1 else 0.0
        save_ratio = float(action[0]) if len(action) > 0 else 0.0
        work_time = float(action[1]) if len(action) > 1 else 0.0

        return (
            f"\n[{tag}] "
            f"Reward={reward:.2f}, "
            f"E={e_val:.2f}, Wealth={wealth_val:.2f}, "
            f"SavingsRatio={save_ratio:.2f}, WorkTime={work_time:.2f}, "
            f"Reasoning={reasoning}"
        )

    def build_similar_experience_text(self, household_id, raw_global_obs, raw_private_obs):
        """
        At the start of a long-term reasoning phase:
        retrieve from H_long and merge with current H_short.
        """
        pieces = []

        # (A) Current H_short for this household
        for idx, exp in enumerate(self.short_term_memory[household_id]):
            pieces.append(self.format_experience_for_prompt(exp, f"ShortExp#{idx+1}"))

        # (B) Retrieved H_long neighbors
        if self.faiss_manager_long_term.current_count > 0:
            query_vec = self.build_think_query_vector(raw_global_obs, raw_private_obs)
            query_vec = query_vec.reshape(1, -1).astype(np.float32)

            _, faiss_results = self.faiss_manager_long_term.search(
                query_vec,
                top_k=self.retrieval_top_k
            )

            for idx, exp in enumerate(faiss_results[0]):
                if exp is None:
                    continue
                pieces.append(self.format_experience_for_prompt(exp, f"LongExp#{idx+1}"))

        return "".join(pieces)

    def record_reasoning_trajectories(
        self,
        action,
        rewards,
        raw_global_obs,
        raw_private_obs,
        llm_results,
        household_name,
        n_households,
        last_news_type,
    ):
        """
        After a new reasoning is produced and reward is observed,
        update H_short; if the news type is long, harvest into H_long.
        """
        for i in range(n_households):
            hid = f"Household{i+1}"
            a = action[household_name][i]

            reasoning = ""
            if isinstance(llm_results, list) and i < len(llm_results):
                reasoning = llm_results[i].get("reasoning", "")

            exp = {
                "reasoning_text": reasoning,
                "reward": float(rewards[i]),
                "action": a.tolist() if hasattr(a, "tolist") else list(a),
                "global_obs": raw_global_obs.copy(),
                "private_obs": raw_private_obs[i].copy(),
                "query_vector": self.build_think_query_vector(raw_global_obs, raw_private_obs[i]),
                "added_to_long_term": False,
            }

            self.push_to_short_term_memory(hid, exp)

        if last_news_type == "long":
            self.harvest_long_term_memory_from_short()