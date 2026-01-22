import copy 
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
import csv
import time 
import os, sys
import json
import wandb
from .maddpg import MADDPG
sys.path.append(os.path.abspath('../..'))
from agents.log_path import make_logpath
from utils.experience_replay import replay_buffer
from agents.utils import get_action_info
from datetime import datetime
from env.evaluation import save_parameters
import logging
from ..intelligence_bureau import IntelligenceBureauAgent
from utils.dialogue import DialogueManager
from utils.embedding import Household_embed  
import asyncio
import shutil
from utils.llm_driver import LLMCall

torch.autograd.set_detect_anomaly(True)

import faiss
import pickle  


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
        
class FaissIndexManager:
    """
    Manage a FAISS index for nearest-neighbor search of experience vectors.

    Attributes:
        dim (int): Dimension of the input vectors.
        use_gpu (bool): If True, use FAISS GPU resources (requires faiss-gpu).
        index: FAISS index object for storing and querying vectors.
        id_to_experience (dict): Maps vector IDs to corresponding experience metadata.
        current_count (int): Total number of vectors added so far.
    """
    def __init__(self, dim: int, use_gpu: bool = False):
        self.dim = dim
        self.use_gpu = use_gpu

        # Create an exact L2 index (O(N) search).
        # For approximate search, consider replacing with an IVF or HNSW index.
        self.index = faiss.IndexFlatL2(dim)

        # If GPU mode is enabled, move the index to GPU
        if self.use_gpu:
            gpu_res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)

        # Initialize mapping from FAISS vector ID to experience data
        self.id_to_experience = {}
        self.current_count = 0

    def add(self, vectors: np.ndarray, experiences: list):
        """
        Add a batch of vectors with associated experience metadata to the index.

        Args:
            vectors (np.ndarray): Float32 array of shape (batch_size, dim).
            experiences (list): List of dicts, one per vector, containing metadata.
        """
        batch_size = vectors.shape[0]
        # Add vectors to the FAISS index
        self.index.add(vectors)
        # Record the mapping from new vector IDs to experiences
        for i in range(batch_size):
            vec_id = self.current_count + i
            self.id_to_experience[vec_id] = experiences[i]
        # Update the running count of stored vectors
        self.current_count += batch_size

    def search(self, query_vectors: np.ndarray, top_k: int = 5):
        """
        Query the index for the nearest neighbors of each input vector.

        Args:
            query_vectors (np.ndarray): Float32 array, shape (num_queries, dim).
            top_k (int): Number of nearest neighbors to return per query.

        Returns:
            distances (np.ndarray): Array of shape (num_queries, top_k) containing L2 distances.
            results (list of list): A nested list where results[i][j] is the experience
                                    associated with the j-th nearest vector to query i,
                                    or None if no mapping exists.
        """
        distances, indices = self.index.search(query_vectors, top_k)
        results = []
        # Retrieve the stored experiences for each returned index
        for idx_row in indices:
            row_exps = [self.id_to_experience.get(vec_id) for vec_id in idx_row]
            results.append(row_exps)
        return distances, results


'''
maddpg
'''
class maddpg_agent:
    """
    High-level controller for MADDPG training and evaluation in the economic simulation.
    Integrates environment interaction, experience replay, FAISS-based retrieval,
    and LLM-driven embedding updates.
    """
    def __init__(self, envs, args):
        # Reference to the training environment
        self.envs = envs  
        # Configuration and hyperparameters
        self.args = args  
        # Copy of the environment for evaluation purposes
        self.eval_env = copy.copy(envs)  
        
        # Exploration noise parameters
        self.noise = args.noise_rate  
        self.epsilon = args.epsilon    
        # Interval (in steps) for generating long-term analysis
        self.long_term_analysis_interval = 40
        
        # Storage for LLM outputs and embeddings between updates
        self.llm_results = []
        self.embeddings = None
        self.last_long_term_llm_results = None
        self.previous_evaluations = {}
        self.previous_statements = {}
        
        # Per-epoch experience pools for each household (opaque to one another)
        n_households = self.envs.households.n_households
        self.current_epoch_experiences = {
            f"Household{i+1}": [] for i in range(n_households)
        }
        # Temporary buffer for experiences when no news is generated
        self.pending_experiences = {
            f"Household{i+1}": [] for i in range(n_households)
        }
        
        # Total number of agents (households + government)
        self.args.n_agents = n_households + 1  
        
        # Determine observation/action dimensions for government and households
        self.args.gov_obs_dim = self.envs.government.observation_space.shape[0]
        self.args.gov_action_dim = self.envs.government.action_space.shape[0]
        # Note: household observation dimension is augmented by embed_dim
        # Actual obs construction: [global_obs, private_obs, embedding]
        # households.observation_space already includes global_obs, so private_obs_dim = households.observation_space.shape[0] - gov_obs_dim
        private_obs_dim = self.envs.households.observation_space.shape[0] - self.args.gov_obs_dim
        self.args.house_obs_dim = (
            self.args.gov_obs_dim      # global obs
            + private_obs_dim          # private obs (e, a)
            + self.args.embed_dim      # LLM embedding
        )
        self.args.house_action_dim = self.envs.households.action_space.shape[1]
        # Number of agent blocks: low-, mid-, high-income households + government
        self.args.agent_block_num = 4  
        
        # Instantiate the core MADDPG modules for each agent
        self.agents = self._init_agents()

        # Experience replay buffer for training
        self.buffer = replay_buffer(self.args.buffer_size)
        
        # FAISS indices for short- and long-term experience retrieval
        self.faiss_dim = 9
        self.faiss_manager = FaissIndexManager(dim=self.faiss_dim, use_gpu=False)
        self.faiss_manager_long_term = FaissIndexManager(dim=self.faiss_dim, use_gpu=False)
        
        # Create logging/checkpoint directory and save initial args
        self.model_path, _ = make_logpath(algo="maddpg", n=n_households)
        save_args(path=self.model_path, args=self.args)
        
        # Names of the economic indicators tracked during training/evaluation
        self.indicators_name = [
            "gov_reward", "mean_utility", "years", "total_income", "income_10", "income_50",
            "income_100", "total_tax", "income_tax", "income_tax_10", "income_tax_50",
            "income_tax_100", "total_wealth", "wealth_10", "wealth_50", "wealth_100", "wealth_tax",
            "wealth_tax_10", "wealth_tax_50", "wealth_tax_100",
            "per_gdp", "income_gini", "wealth_gini", "wage", "total_labor", "labor_10", "labor_50",
            "labor_100", "sw_10", "sw_50", "sw_100",
            "total_consumption", "consumption_10", "consumption_50", "consumption_100", "Bt", "Kt",
            "Gt_prob", "income_tau", "income_xi", "wealth_tau", "wealth_xi"
        ]
        
        # Optionally initialize Weights & Biases logging
        self.wandb = False
        if self.wandb:
            wandb.init(
                config=self.args,
                project="TaxAI",
                entity="taxai",
                name=f"{self.model_path.parent.parent.name}-{self.model_path.name}_n={n_households}",
                dir=str(self.model_path),
                job_type="training",
                reinit=True
            )
        
        # Agents for generating economic news and orchestrating dialogue
        self.intelligence_bureau = IntelligenceBureauAgent(
            llm_model=LLMCall,
            data_sources=["economic_observations", "agent_actions"]
        )
        self.dialogue_manager = DialogueManager(self.envs)
        
        # Buffer timing logs in memory to reduce file I/O
        self.step_time_log = []

    def _init_agents(self):
        """
        Create and return a list of MADDPG core objects,
        one per agent index (household groups + government).
        """
        return [MADDPG(self.args, i) for i in range(self.args.agent_block_num)]
    
    def save_faiss_index(self, save_dir):
        """
        Persist the FAISS index and experience mapping to disk.
        
        Args:
          save_dir: Directory in which to write the index and mapping files.
        """
        # Convert GPU index to CPU if needed
        if self.faiss_manager_long_term.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.faiss_manager_long_term.index)
        else:
            cpu_index = self.faiss_manager_long_term.index
        
        # Write out the FAISS index file
        faiss.write_index(cpu_index, os.path.join(save_dir, "faiss_index.index"))
        
        # Serialize the id->experience mapping
        with open(os.path.join(save_dir, "id_to_experience.pkl"), "wb") as f:
            pickle.dump(self.faiss_manager_long_term.id_to_experience, f)
        
        print(f"✅ FAISS index and experience mapping saved to {save_dir}")
        
    def load_faiss_index(self, load_dir):
        """
        Load FAISS index and experience mapping from disk.
        
        Args:
          load_dir: Directory from which to read index and mapping files.
        """
        # Read CPU index file
        index_cpu = faiss.read_index(os.path.join(load_dir, "faiss_index.index"))
        
        # Move to GPU if configured
        if self.faiss_manager_long_term.use_gpu:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            self.faiss_manager_long_term.index = gpu_index
        else:
            self.faiss_manager_long_term.index = index_cpu
        
        # Load the serialized id->experience dictionary
        with open(os.path.join(load_dir, "id_to_experience.pkl"), "rb") as f:
            exp_dict = pickle.load(f)
        
        self.faiss_manager_long_term.id_to_experience = exp_dict
        self.faiss_manager_long_term.current_count = len(exp_dict)
        
        print(f"✅ Loaded FAISS index and experiences from {load_dir}")


    def log_step_time(self, epoch, step, duration):
        self.step_time_log.append([epoch, step, duration])
        if len(self.step_time_log) >= 100:
            self.flush_log()

    def flush_log(self):
        log_file_path = os.path.join(self.model_path, "step_time_log.csv")
        file_exists = os.path.isfile(log_file_path)
        try:
            with open(log_file_path, "a", newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Epoch", "Step", "Duration (seconds)"])
                writer.writerows(self.step_time_log)
            self.step_time_log = []
        except Exception as e:
            logging.error(f"Error writing step time log: {e}")

    def observation_wrapper(self, global_obs, private_obs):  
        # Make a copy to avoid in-place modification of raw observations
        global_obs_wrapped = global_obs.copy()
        private_obs_wrapped = private_obs.copy()
        
        # global
        global_obs[0] /= 1e7
        global_obs[1] /= 1e5
        global_obs[3] /= 1e5
        global_obs[4] /= 1e5
        private_obs[:, 1] /= 1e5
        return global_obs, private_obs
    
    def _update_experience_pool(self, top_n=5):
        all_exps = []

        for household_id, exps_list in self.current_epoch_experiences.items():
            logging.info(f"Processing experiences for household {household_id}:{exps_list}")
            all_exps.extend(exps_list)

        if len(all_exps) == 0:
            logging.warning("No experiences collected in this epoch.")
            return
        
        sorted_exps = sorted(all_exps, key=lambda e: e["reward"], reverse=True)
        top_exps = sorted_exps[:top_n]

        vectors_long = []
        for exp in top_exps:
            g = exp["global_obs"]
            p = exp["private_obs"]
            vec = np.concatenate([g, p], axis=0)
            vectors_long.append(vec)
        if len(vectors_long) > 0:
            arr_long = np.array(vectors_long, dtype=np.float32)
            logging.info(f"Adding {arr_long.shape} vectors to long_term Faiss index.")
            self.faiss_manager_long_term.add(arr_long, top_exps)

        logging.info(f"Epoch ended. Added top {top_n} experiences to long_term.")

            
    def compute_key_indicators(self):
        indicators = {}
        if hasattr(self.envs, "per_household_gdp"):
            indicators["per_gdp"] = self.envs.per_household_gdp
        else:
            print("per_gdp not found in training environment.")
        if hasattr(self, "last_house_reward"):
            indicators["social_welfare"] = np.mean(self.last_house_reward)
        else:
            print("last_house_reward not found in training environment.")
        #if hasattr(self.envs, "income_gini"):
            #indicators["income_gini"] = self.envs.income_gini
        #else:
            #print("income_gini not found in training environment.")
        if hasattr(self.envs, "wealth_gini"):
            indicators["wealth_gini"] = self.envs.wealth_gini
        else:
            print("wealth_gini not found in training environment.")
        logging.info("Key indicators computed: " + str(indicators))
        return indicators

    async def generate_short_term_news(self, step, short_term_news, private_obs):
        n_households = self.envs.households.n_households
        tasks = []
        
        for i in range(n_households):
            recent_result_str = self.get_recent_long_term_result(i)
            
            tasks.append(
                self.envs.households.derive_reasoning_text(
                    short_term_news=short_term_news,
                    recent_long_term_result=recent_result_str,
                    private_observation=private_obs[i],
                    temperature=0.6,
                    max_retries=15
                )
            )
        derived_jsons = await asyncio.gather(*tasks)
        return derived_jsons

    def get_recent_long_term_result(self, household_idx):
        if self.last_long_term_llm_results is not None:
            for prev_result in self.last_long_term_llm_results:
                if prev_result.get("agent_id", "") == f"Household{household_idx+1}":
                    return json.dumps(prev_result)
        return ""


    def should_generate_short_term(self, current_key_indicators, step, threshold, epsilon=1e-6):
        if not hasattr(self, "prev_key_indicators"):
            self.prev_key_indicators = None
        if self.prev_key_indicators is None or step == 0:
            return False
        for key, current_value in current_key_indicators.items():
            prev_value = self.prev_key_indicators.get(key, 0)
            if abs(current_value - prev_value) / (abs(prev_value) + epsilon) > threshold:
                return True
        return False

    def should_generate_long_term(self, step):
        return step % self.long_term_analysis_interval == 0 and step != 0 or step == -1

    def llm_generate_embeddings(self, step, global_obs, private_obs):
        """
        Generate or update agent embeddings using LLM-driven economic news and past experiences.

        1. Log the new global observation.
        2. Compute key economic indicators.
        3. Decide whether to generate short-term or long-term news.
        4. If no news is needed, reuse existing embeddings.
        5. Otherwise, generate the appropriate news and retrieve similar experiences:
           a. For long-term news, query the FAISS index across epochs.
           b. Also retrieve top experiences from the current epoch.
        6. Run the dialogue manager to produce new LLM results.
        7. Update and return the reconstructed embeddings.
        """
        # Step 1: Record the latest global observation
        self.intelligence_bureau.collect_observations(global_obs)
        current_key_indicators = self.compute_key_indicators()

        # Step 2: Decide if we need to generate short-term or long-term news
        generate_short = self.should_generate_short_term(current_key_indicators, step, self.args.threshold)
        self.prev_key_indicators = current_key_indicators
        generate_long = self.should_generate_long_term(step)

        # If no news generation is triggered, reuse the existing embeddings
        if not generate_short and not generate_long:
            logging.info(f"No news update at step {step+1}; reusing existing embeddings.")
            return None

        # Step 3: Generate news as needed
        short_term_news = (self.intelligence_bureau.generate_news(recent_only=True)
                           if generate_short else None)
        long_term_news  = (self.intelligence_bureau.generate_news(recent_only=False)
                           if generate_long  else None)

        # Ensure an asyncio event loop is available
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Step 4: If we have long-term news, retrieve similar experiences
        if long_term_news:
            similar_experience_text = ""
            retrieved_exps = []
            top_k = 2
            n_households = self.envs.households.n_households

            # (A) Cross-epoch retrieval from the long-term FAISS index
            for i in range(n_households):
                if self.faiss_manager_long_term.current_count == 0:
                    continue
                query_vec = np.concatenate([global_obs, private_obs[i]], axis=0)
                query_vec = query_vec.reshape(1, -1).astype(np.float32)
                distances, faiss_results = self.faiss_manager_long_term.search(query_vec, top_k=top_k)
                for exp_item in faiss_results[0]:
                    if exp_item is not None:
                        retrieved_exps.append(exp_item)
            retrieved_exps = retrieved_exps[:top_k]

            # Build a summary string of the retrieved cross-epoch experiences
            for idx, exp in enumerate(retrieved_exps):
                reward = exp.get("reward", 0)
                reasoning = exp.get("reasoning_text", "")
                priv_obs = exp.get("private_obs", [])
                action   = exp.get("action", [])
                similar_experience_text += (
                    f"\n[CrossEpochExp#{idx+1}] "
                    f"Reward={reward:.2f}, "
                    f"E={priv_obs[0]:.2f}, Wealth={priv_obs[1]:.2f}, "
                    f"SavingsRatio={action[0]:.2f}, WorkTime={action[1]:.2f}, "
                    f"Reasoning={reasoning}"
                )

            # (B) In-epoch retrieval from the current epoch's own experiences
            for i in range(n_households):
                h_id = f"Household{i+1}"
                personal_list = self.current_epoch_experiences[h_id]
                if not personal_list:
                    continue
                # Select the top 2 by reward
                top_personal = sorted(personal_list, key=lambda e: e["reward"], reverse=True)[:2]
                for exp in top_personal:
                    reward = exp.get("reward", 0)
                    reasoning = exp.get("reasoning_text", "")
                    priv_obs = exp.get("private_obs", [])
                    action   = exp.get("action", [])
                    similar_experience_text += (
                        f"\n[CurrentEpochExp h={h_id}] "
                        f"Reward={reward:.2f}, "
                        f"E={priv_obs[0]:.2f}, Wealth={priv_obs[1]:.2f}, "
                        f"SavingsRatio={action[0]:.2f}, WorkTime={action[1]:.2f}, "
                        f"Reasoning={reasoning}"
                    )

            logging.info(f"Retrieved similar experiences for long-term news at step {step+1}:"
                         f"{similar_experience_text}")

            # Step 5: Run the dialogue manager for long-term news
            new_results = loop.run_until_complete(
                self.dialogue_manager.run_dialogue_round(
                    step=step,
                    long_term_news=long_term_news,
                    similar_experience=similar_experience_text,
                    private_obs=private_obs
                )
            )
            # Cache for possible future reference
            self.last_long_term_llm_results = new_results

        else:
            # Step 5 (alternative): Generate short-term reasoning updates
            derived_jsons = loop.run_until_complete(
                self.generate_short_term_news(
                    step=step,
                    short_term_news=short_term_news,
                    private_obs=private_obs
                )
            )
            new_results = self.process_derived_jsons(derived_jsons)

        logging.info(f"LLM results at step {step+1}: {new_results}")

        # Step 6: Update stored LLM results and reconstruct embeddings
        self.llm_results = new_results
        embeddings = self.construct_embeddings()
        self.embeddings = embeddings
        return embeddings

    def process_derived_jsons(self, derived_jsons):
        llm_results = []
        for i, derived_json in enumerate(derived_jsons):
            new_economic_status = derived_json.get("economic_status", 1)
            new_reasoning = derived_json.get("reasoning", "No valid reasoning response from LLM after retries.")
            default_evaluation = self.get_default_evaluation(i)
            
            result = {
                "agent_id": f"Household{i+1}",
                "evaluation": default_evaluation,
                "economic_status": new_economic_status,
                "reasoning": new_reasoning
            }
            llm_results.append(result)
        return llm_results

    def get_default_evaluation(self, household_idx):
        default_evaluation = {
            "wealth_guesses": [1] * 10,
            "trust_levels": [5.0] * 10,
            "reflection_text": ""
        }
        if hasattr(self, "last_long_term_llm_results"):
            for prev_result in self.last_long_term_llm_results:
                if prev_result.get("agent_id", "") == f"Household{household_idx+1}":
                    return prev_result.get("evaluation", default_evaluation)
        return default_evaluation

    def construct_embeddings(self):
        embeddings = {}
        n_households = self.envs.households.n_households
        for house_idx in range(n_households):
            evaluation_str, obs_text = self.construct_evaluation_text(house_idx)
            embed_t = Household_embed(evaluation=evaluation_str, obs_text=obs_text)
            embeddings[f"Household{house_idx+1}"] = embed_t
        return embeddings

    def construct_evaluation_text(self, house_idx):
        evaluation_str = ""
        obs_text = ""
        target_agent_id = f"Household{house_idx + 1}"
        for result in self.llm_results:
            if result.get("agent_id", "") != target_agent_id:
                continue
            evaluation = result.get("evaluation", {})
            economic_status = result.get("economic_status", 1)
            
            wealth_guesses = evaluation.get("wealth_guesses", [1] * 10)
            trust_levels = evaluation.get("trust_levels", [5] * 10)
            
            reflection_text = evaluation.get("reflection_text", "")
            evaluation_str += (
                #f"\n--- Evaluation Summary ---\n"
                #f"🔹 Estimated Wealth Levels (0=Low, 1=Medium, 2=High): {wealth_guesses}\n"
                #f"🔹 Trust Levels for Each Statement (0-10 scale): {trust_levels}\n"
                f"🔹 Overall Economic Condition (0=Bad, 1=Neutral, 2=Good): {economic_status}\n"
                f"🔹 Reflection & Key Insights:\n{reflection_text}\n"
            )
            obs_text = result.get("reasoning", obs_text)
            break
        return evaluation_str, obs_text

    def learn(self):
        """
        Main training loop for the MADDPG agents.  
        - Resets the environment and normalizes observations.  
        - Iterates over epochs and steps per epoch:  
        (A) Select actions for government and households  
        (B) Step the environment  
        (C) Store transitions in replay buffer  
        (D) Potentially update LLM-based embeddings  
        (E) Periodically sample from buffer to train networks  
        - Logs performance metrics and saves models/indices at specified intervals.
        """
        n_households = self.envs.households.n_households
        # --- Initialize environment and observations ---
        global_obs, private_obs = self.envs.reset()
        global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)

        # Prepare logging of rewards and losses
        gov_rew = []
        house_rew = []
        epochs = []
        sum_actor_loss = 0
        sum_critic_loss = 0
        log_file = f"{self.model_path}/loss_and_years.csv"
        log_columns = [
            "timestamp", "epoch", "total_epochs", "frames",
            "gov_reward", "house_reward", "years", "actor_loss", "critic_loss"
        ]
        pd.DataFrame(columns=log_columns).to_csv(log_file, index=False)
        print(f"📊 Initialized loss log at: {log_file}")

        print(f"n_epochs: {self.args.n_epochs}, epoch_length: {self.args.epoch_length}")

        # --- Epoch loop ---
        for epoch in range(self.args.n_epochs):
            # Reset epoch-specific loss accumulators
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            epoch_update_count = 0

            # (Re-)generate embeddings at the start of each epoch if needed
            new_embeddings = self.llm_generate_embeddings(-1, global_obs, private_obs)
            if new_embeddings is not None:
                self.embeddings = new_embeddings
            # Distribute the latest embeddings to all agents
            for agent in self.agents:
                agent.embeddings = self.embeddings

            print(f"Epoch {epoch+1} start:")

            # --- Step loop ---
            for t in range(self.args.epoch_length):
                start_time = time.time()
                global_step = epoch * self.args.epoch_length + t

                # (A) --- Select actions ---
                global_obs_tensor = self._get_tensor_inputs(global_obs)
                private_obs_tensor = self._get_tensor_inputs(private_obs)

                house_embeddings = []
                for i in range(n_households):
                    hid = f"Household{i+1}"
                    emb = self.embeddings.get(hid, torch.zeros(self.args.embed_dim, device=private_obs_tensor.device))
                    house_embeddings.append(emb)
                embedding_tensor = torch.stack(house_embeddings, dim=0)
                combined_private_obs = torch.cat([private_obs_tensor, embedding_tensor], dim=-1)

                n_global_obs = global_obs_tensor.repeat(n_households, 1)
                obs = torch.cat([n_global_obs, combined_private_obs], dim=-1)

                sorted_indices = torch.argsort(private_obs_tensor[:, 1], descending=True)
                sorted_obs = obs[sorted_indices]

                hou_action = np.zeros((n_households, self.args.house_action_dim))
                gov_action = self.agents[-1].select_action(global_obs_tensor, self.noise, self.epsilon)

                num_set = range(n_households)
                for group_idx in range(self.args.agent_block_num - 1):
                    if group_idx == 0:
                        idxs = num_set[: int(0.1 * n_households)]
                    elif group_idx == 1:
                        idxs = num_set[int(0.1 * n_households): int(0.5 * n_households)]
                    else:
                        idxs = num_set[int(0.5 * n_households):]
                    hou_action[idxs] = self.agents[group_idx].select_action(sorted_obs[list(idxs)], self.noise, self.epsilon)

                orig_order = sorted_indices.cpu().numpy()
                hou_action = hou_action[np.argsort(orig_order)]

                action = {self.envs.government.name: gov_action, self.envs.households.name: hou_action}

                # Prepare current embeddings for buffer storage
                # self.embeddings is a dict of tensors
                def get_emb_array(emb_dict):
                    if emb_dict is None:
                        return np.zeros((n_households, self.args.embed_dim))
                    return np.array([emb_dict[f"Household{i+1}"].detach().cpu().numpy() for i in range(n_households)])

                current_embeddings_arr = get_emb_array(self.embeddings)

                # (B) --- Step the environment ---
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)
                next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                # Store last rewards for logging
                self.last_gov_reward = gov_reward
                self.last_house_reward = house_reward

                # (D) --- Potential LLM-driven embedding update ---
                new_embeddings = self.llm_generate_embeddings(global_step, next_global_obs, next_private_obs)
                
                # If new_embeddings is None, it means we reuse the old ones
                if new_embeddings is None:
                    actual_new_embeddings = self.embeddings
                else:
                    actual_new_embeddings = new_embeddings
                    # Save new reasoning experiences for each household
                    for i in range(n_households):
                        hid = f"Household{i+1}"
                        r = house_reward[i]
                        a = action[self.envs.households.name][i]
                        
                        current_res = next((res for res in self.llm_results if res["agent_id"] == hid), {})
                        reasoning = current_res.get("reasoning", "No reasoning")
                        
                        self.current_epoch_experiences[hid].append({
                            "reasoning_text": reasoning,
                            "reward": float(r),
                            "action": a.tolist() if hasattr(a, "tolist") else a,
                            "global_obs": next_global_obs.copy(),
                            "private_obs": next_private_obs[i].copy()
                        })
                    # Update global embeddings for next steps
                    self.embeddings = new_embeddings
                    for ag in self.agents:
                        ag.embeddings = new_embeddings

                next_embeddings_arr = get_emb_array(actual_new_embeddings)

                # (C) --- Store transition in replay buffer ---
                self.buffer.add(
                    global_obs, private_obs,
                    gov_action, hou_action,
                    gov_reward, house_reward,
                    next_global_obs, next_private_obs,
                    float(done),
                    current_embeddings_arr,
                    next_embeddings_arr
                )

                # Advance to next state
                global_obs, private_obs = next_global_obs, next_private_obs

                # If the episode ends, reset environment
                if done:
                    global_obs, private_obs = self.envs.reset()
                    global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)

                # (E) --- Train networks every 10 steps ---
                if t % 10 == 0:
                    transitions = self.buffer.sample(self.args.batch_size)
                    step_actor_loss = 0
                    step_critic_loss = 0
                    for agent in self.agents:
                        actor_loss, critic_loss = agent.train(transitions, self.agents)
                        step_actor_loss += actor_loss
                        step_critic_loss += critic_loss
                    
                    epoch_actor_loss += step_actor_loss
                    epoch_critic_loss += step_critic_loss
                    epoch_update_count += 1

                # Log step duration
                duration = time.time() - start_time
                print(f"Step {t+1} completed in {duration:.2f}s.")
                self.log_step_time(epoch, t, duration)

            # End of epoch: update FAISS pools and clear per-epoch experiences
            self._update_experience_pool()
            for hid in self.current_epoch_experiences:
                self.current_epoch_experiences[hid].clear()

            # Periodic logging, evaluation, and checkpointing
            if epoch % self.args.display_interval == 0:
                print(f"Displaying metrics at epoch {epoch+1}")
                stats = self._evaluate_agent()
                frames = (epoch + 1) * self.args.epoch_length
                gov_rew.append(stats["gov_rew"])
                house_rew.append(stats["social_welfare"])
                epochs.append(frames)
                np.savetxt(f"{self.model_path}/gov_reward.txt", gov_rew)
                np.savetxt(f"{self.model_path}/house_reward.txt", house_rew)
                np.savetxt(f"{self.model_path}/steps.txt", epochs)
                if self.wandb:
                    wandb.log(stats)
                
                # Compute average loss over the epoch for logging
                avg_actor_loss = epoch_actor_loss / epoch_update_count if epoch_update_count > 0 else 0
                avg_critic_loss = epoch_critic_loss / epoch_update_count if epoch_update_count > 0 else 0

                log_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch,
                    "total_epochs": self.args.n_epochs,
                    "frames": frames,
                    "gov_reward": stats["gov_rew"],
                    "house_reward": stats["social_welfare"],
                    "years": stats["years"],
                    "actor_loss": avg_actor_loss,
                    "critic_loss": avg_critic_loss
                }
                pd.DataFrame([log_entry]).to_csv(log_file, mode="a", header=False, index=False)
                logging.info(f"Epoch {epoch+1} stats: {log_entry}")
                print(f"✅ Saved epoch {epoch+1} metrics to {log_file}")

            # Increase analysis interval and optionally save checkpoints
            if epoch % 15 == 0:
                self.long_term_analysis_interval += 5
            if epoch % self.args.save_interval == 0:
                save_dir = os.path.join(self.model_path, f"epoch_{epoch}")
                os.makedirs(save_dir, exist_ok=True)
                for i, agent in enumerate(self.agents):
                    logging.info(f"Saving agent {i} checkpoint at epoch {epoch+1}")
                    torch.save(agent.actor_network.state_dict(), f"{save_dir}/agent_{i}.pt")
                self.save_faiss_index(save_dir)

            # Decay exploration parameters
            self.noise = max(0.05, self.noise - 5e-7)
            self.epsilon = max(0.05, self.epsilon - 5e-7)

        # After all epochs, save final FAISS index and optionally finish W&B
        self.save_faiss_index(str(self.model_path))
        logging.info("Training complete. FAISS index and experiences saved.")
        if self.wandb:
            wandb.finish()


    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        return obs_tensor
    
    def test(self):
        
        path = "./epoch_80/"
        for agent_i in range(len(self.agents)):
            self.agents[agent_i].actor_network.load_state_dict(
            torch.load(path + f'/agent_{agent_i}.pt', weights_only=True))
        # total_gov_reward,total_house_reward,total_steps,tax,income_gini,wealth_gini,gdp,income,income_tax,wealth,wealth_tax,labor,consumption,wage = self.episode_evolution()
        #self.episode_evolution()
        self.light_episode_evolution()


    def _evaluate_agent(self):
        """
        Run a set number of evaluation episodes using the current policies.
        Collects economic indicators at each step and computes the average over all episodes.

        Returns:
            dict: A mapping from indicator names to their average values.
        """
        economic_indicators = []
        column_names = [
            "epoch", "gov_rew", "social_welfare", "years", "total_income", "total_tax", "income_tax",
            "total_wealth", "wealth_tax", "per_gdp", "income_gini", "wealth_gini", "wage",
            "total_labor", "total_consumption", "Bt", "Kt", "Gt_prob", "income_tau",
            "income_xi", "wealth_tau", "wealth_xi"
        ]

        for episode_idx in range(self.args.eval_episodes):
            step_count = 0
            # Reset evaluation environment and normalize observations
            global_obs, private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)

            episode_data = []

            # Play until the episode terminates
            while True:
                with torch.no_grad():
                    action, sort_idx = self._evaluate_get_action(global_obs, private_obs)
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = \
                        self.eval_env.step(action)
                    next_global_obs, next_private_obs = self.observation_wrapper(
                        next_global_obs, next_private_obs
                    )

                step_count += 1

                # Compute key economic indicators
                total_income   = np.mean(self.eval_env.post_income)
                total_tax      = np.mean(self.eval_env.tax_array)
                income_tax     = np.mean(self.eval_env.income_tax)
                total_wealth   = np.mean(self.eval_env.households.at_next)
                wealth_tax     = np.mean(self.eval_env.asset_tax)
                per_gdp        = self.eval_env.per_household_gdp
                income_gini    = self.eval_env.income_gini
                wealth_gini    = self.eval_env.wealth_gini
                wage           = self.eval_env.WageRate
                total_labor    = self.eval_env.Lt
                total_consumption = np.mean(self.eval_env.consumption)
                Bt             = self.eval_env.Bt
                Kt             = self.eval_env.Kt
                Gt_prob        = self.eval_env.Gt_prob
                income_tau     = self.eval_env.government.tau
                income_xi      = self.eval_env.government.xi
                wealth_tau     = self.eval_env.government.tau_a
                wealth_xi      = self.eval_env.government.xi_a
                social_welfare = np.mean(house_reward)

                # Record this step's data
                episode_data.append([
                    episode_idx, gov_reward, social_welfare, 1,
                    total_income, total_tax, income_tax, total_wealth,
                    wealth_tax, per_gdp, income_gini, wealth_gini, wage,
                    total_labor, total_consumption, Bt, Kt, Gt_prob,
                    income_tau, income_xi, wealth_tau, wealth_xi
                ])

                if done:
                    break

                # Advance to next state
                global_obs, private_obs = next_global_obs, next_private_obs

            # Compute averages for this episode
            years = len(episode_data)  # each step counts as one time unit
            avg_data = np.mean(episode_data, axis=0)
            # Scale the 3rd–5th entries (indices 2:5) by the number of time steps
            avg_data[2:5] *= years

            economic_indicators.append(avg_data)

        # Save all episodes' indicators to CSV
        log_file = f"{self.model_path}/economic_indicators.csv"
        df = pd.DataFrame(economic_indicators, columns=column_names)
        df.to_csv(log_file, mode="a", header=not pd.io.common.file_exists(log_file), index=False)
        print(f"Appended evaluation data to {log_file}")

        # Return the overall average across episodes
        overall_avg = np.mean(economic_indicators, axis=0)
        return dict(zip(column_names, overall_avg))


    def episode_evolution(self):
        """
        Run detailed episode simulations and record per-step indicators plus LLM-derived context.
        Saves a CSV with one row per step, including text summaries.
        """
        # Define the columns, including a final 'text' column for LLM output
        self.indicators_name = [
            "gov_reward", "mean_utility", "years", "total_income", "income_10", "income_50",
            "income_100", "total_tax", "income_tax", "income_tax_10", "income_tax_50",
            "income_tax_100", "total_wealth", "wealth_10", "wealth_50", "wealth_100",
            "wealth_tax", "wealth_tax_10", "wealth_tax_50", "wealth_tax_100",
            "per_gdp", "income_gini", "wealth_gini", "wage", "total_labor",
            "labor_10", "labor_50", "labor_100", "sw_10", "sw_50", "sw_100",
            "total_consumption", "consumption_10", "consumption_50", "consumption_100",
            "text"
        ]

        step_records = []

        for episode_idx in range(self.args.eval_episodes):
            step_count = 0
            global_obs, private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)

            # Synchronize LLM embeddings if needed
            new_embeddings = self.llm_generate_embeddings(-1, global_obs, private_obs)
            if new_embeddings is not None:
                self.embeddings = new_embeddings
            for agent in self.agents:
                agent.embeddings = self.embeddings

            while True:
                with torch.no_grad():
                    action, sort_idx = self._evaluate_get_action(global_obs, private_obs)
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = \
                        self.eval_env.step(action)
                    next_global_obs, next_private_obs = self.observation_wrapper(
                        next_global_obs, next_private_obs
                    )
                    self.last_gov_reward = gov_reward
                    self.last_house_reward = house_reward

                step_count += 1

                # Compute grouped and overall indicators just as in the base code
                total_income   = np.mean(self.eval_env.post_income)
                income_10      = np.mean(self.eval_env.post_income[sort_idx[:1]])
                income_50      = np.mean(self.eval_env.post_income[sort_idx[1:5]])
                income_100     = np.mean(self.eval_env.post_income[sort_idx[5:]])
                total_tax      = np.mean(self.eval_env.tax_array)
                income_tax     = np.mean(self.eval_env.income_tax)
                income_tax_10  = np.mean(self.eval_env.income_tax[sort_idx[:1]])
                income_tax_50  = np.mean(self.eval_env.income_tax[sort_idx[1:5]])
                income_tax_100 = np.mean(self.eval_env.income_tax[sort_idx[5:]])
                total_wealth   = np.mean(self.eval_env.households.at_next)
                wealth_10      = np.mean(self.eval_env.households.at_next[sort_idx[:1]])
                wealth_50      = np.mean(self.eval_env.households.at_next[sort_idx[1:5]])
                wealth_100     = np.mean(self.eval_env.households.at_next[sort_idx[5:]])
                wealth_tax     = np.mean(self.eval_env.asset_tax)
                wealth_tax_10  = np.mean(self.eval_env.asset_tax[sort_idx[:1]])
                wealth_tax_50  = np.mean(self.eval_env.asset_tax[sort_idx[1:5]])
                wealth_tax_100 = np.mean(self.eval_env.asset_tax[sort_idx[5:]])
                per_gdp        = self.eval_env.per_household_gdp
                income_gini    = self.eval_env.income_gini
                wealth_gini    = self.eval_env.wealth_gini
                wage           = self.eval_env.WageRate
                total_labor    = self.eval_env.Lt
                labor          = self.eval_env.households.e * self.eval_env.ht
                labor_10       = np.mean(labor[sort_idx[:1]])
                labor_50       = np.mean(labor[sort_idx[1:5]])
                labor_100      = np.mean(labor[sort_idx[5:]])
                sw             = np.mean(house_reward)
                sw_10          = np.mean(house_reward[sort_idx[:1]])
                sw_50          = np.mean(house_reward[sort_idx[1:5]])
                sw_100         = np.mean(house_reward[sort_idx[5:]])
                total_consumption = np.mean(self.eval_env.consumption)
                consumption_10    = np.mean(self.eval_env.consumption[sort_idx[:1]])
                consumption_50    = np.mean(self.eval_env.consumption[sort_idx[1:5]])
                consumption_100   = np.mean(self.eval_env.consumption[sort_idx[5:]])

                # Include the latest LLM reasoning as JSON text
                text = ""
                if new_embeddings is not None:
                    text = json.dumps(self.llm_results, indent=2, ensure_ascii=False)

                # Record this step
                step_records.append([
                    episode_idx, step_count,
                    gov_reward, sw, 1,
                    total_income, income_10, income_50, income_100,
                    total_tax, income_tax, income_tax_10, income_tax_50, income_tax_100,
                    total_wealth, wealth_10, wealth_50, wealth_100,
                    wealth_tax, wealth_tax_10, wealth_tax_50, wealth_tax_100,
                    per_gdp, income_gini, wealth_gini, wage,
                    total_labor, labor_10, labor_50, labor_100,
                    sw_10, sw_50, sw_100,
                    total_consumption, consumption_10, consumption_50, consumption_100,
                    text
                ])

                if done:
                    break

                global_obs, private_obs = next_global_obs, next_private_obs

                # Potentially update embeddings mid-episode
                new_embeddings = self.llm_generate_embeddings(step_count, global_obs, private_obs)
                if new_embeddings is not None:
                    for i in range(self.envs.households.n_households):
                        hid = f"Household{i+1}"
                        r = house_reward[i]
                        a = action[self.envs.households.name][i]
                        
                        current_res = next((res for res in self.llm_results if res["agent_id"] == hid), {})
                        reasoning = current_res.get("reasoning", "")
                        
                        self.current_epoch_experiences[hid].append({
                            "reasoning_text": reasoning,
                            "reward": float(r),
                            "action": a.tolist() if hasattr(a, "tolist") else a,
                            "global_obs": next_global_obs.copy(),
                            "private_obs": next_private_obs[i].copy()
                        })
                    self.embeddings = new_embeddings
                    for ag in self.agents:
                        ag.embeddings = new_embeddings

            # Save per-step data to CSV
            columns = ["epoch", "step"] + self.indicators_name
            df = pd.DataFrame(step_records, columns=columns)
            df.to_csv(f"{self.model_path}/episode_evolution_step.csv", index=False, float_format="%.6f")


    def light_episode_evolution(self):
        """
        In evaluation mode, compare three strategies—MADDPG, random, and fixed—
        by tracking both overall and group-level indicators at each step.
        Outputs each series to separate CSV files.
        """
        for episode_idx in range(self.args.eval_episodes):
            step_count = 0
            global_obs, private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)

            # Containers for overall indicators: [total_tax, house_reward, total_labor, total_consumption, gov_reward, gdp, income_gini, wealth_gini]
            maddpg_data = [[] for _ in range(8)]
            random_data = [[] for _ in range(8)]
            fixed_data  = [[] for _ in range(8)]

            # Containers for group-level indicators: 6 variables × 3 groups
            maddpg_group = [[[] for _ in range(3)] for _ in range(6)]
            random_group = [[[] for _ in range(3)] for _ in range(6)]
            fixed_group  = [[[] for _ in range(3)] for _ in range(6)]

            # Sync embeddings at start
            new_embeddings = self.llm_generate_embeddings(-1, global_obs, private_obs)
            if new_embeddings is not None:
                self.embeddings = new_embeddings
            for agent in self.agents:
                agent.embeddings = self.embeddings

            print(f"Light evaluation epoch {episode_idx+1}:")

            while True:
                with torch.no_grad():
                    action, sort_idx = self._evaluate_get_action(global_obs, private_obs)

                    # Copy env for random and fixed
                    self.random_env = copy.copy(self.eval_env)
                    self.fixed_env  = copy.copy(self.eval_env)

                    # MADDPG strategy
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    self.last_gov_reward = gov_reward
                    self.last_house_reward = house_reward
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                    # Random strategy
                    random_action = self.random_evaluate_get_action(action)
                    _, _, random_gov, random_house, _ = self.random_env.step(random_action)

                    # Fixed strategy
                    fixed_action = self.fixed_evaluate_get_action(self.fixed_env, action)
                    _, _, fixed_gov, fixed_house, _ = self.fixed_env.step(fixed_action)

                step_count += 1

                # --- Record overall indicators ---
                # MADDPG
                maddpg_data[0].append(np.mean(self.eval_env.tax_array))
                maddpg_data[1].append(np.mean(house_reward))
                maddpg_data[2].append(self.eval_env.Lt)
                maddpg_data[3].append(np.mean(self.eval_env.consumption))
                maddpg_data[4].append(gov_reward)
                maddpg_data[5].append(self.eval_env.per_household_gdp)
                maddpg_data[6].append(self.eval_env.income_gini)
                maddpg_data[7].append(self.eval_env.wealth_gini)

                # Random
                random_data[0].append(np.mean(self.random_env.tax_array))
                random_data[1].append(np.mean(random_house))
                random_data[2].append(self.random_env.Lt)
                random_data[3].append(np.mean(self.random_env.consumption))
                random_data[4].append(random_gov)
                random_data[5].append(self.random_env.per_household_gdp)
                random_data[6].append(self.random_env.income_gini)
                random_data[7].append(self.random_env.wealth_gini)

                # Fixed
                fixed_data[0].append(np.mean(self.fixed_env.tax_array))
                fixed_data[1].append(np.mean(fixed_house))
                fixed_data[2].append(self.fixed_env.Lt)
                fixed_data[3].append(np.mean(self.fixed_env.consumption))
                fixed_data[4].append(fixed_gov)
                fixed_data[5].append(self.fixed_env.per_household_gdp)
                fixed_data[6].append(self.fixed_env.income_gini)
                fixed_data[7].append(self.fixed_env.wealth_gini)

                # --- Record group-level indicators (3 groups × 6 variables) ---
                group_idxs = [
                    sort_idx[:1],    # top 10%
                    sort_idx[1:5],   # middle 40%
                    sort_idx[5:]     # bottom 50%
                ]
                for g, idxs in enumerate(group_idxs):
                    # Variables: total_income, total_tax, total_wealth, labor, social_welfare, consumption
                    maddpg_group[0][g].append(np.mean(self.eval_env.post_income[idxs]))
                    maddpg_group[1][g].append(np.mean(self.eval_env.tax_array[idxs]))
                    maddpg_group[2][g].append(np.mean(self.eval_env.households.at_next[idxs]))
                    maddpg_group[3][g].append(np.mean(self.eval_env.households.e[idxs] * self.eval_env.ht[idxs]))
                    maddpg_group[4][g].append(np.mean(house_reward[idxs]))
                    maddpg_group[5][g].append(np.mean(self.eval_env.consumption[idxs]))

                    random_group[0][g].append(np.mean(self.random_env.post_income[idxs]))
                    random_group[1][g].append(np.mean(self.random_env.tax_array[idxs]))
                    random_group[2][g].append(np.mean(self.random_env.households.at_next[idxs]))
                    random_group[3][g].append(np.mean(self.random_env.households.e[idxs] * self.random_env.ht[idxs]))
                    random_group[4][g].append(np.mean(random_house[idxs]))
                    random_group[5][g].append(np.mean(self.random_env.consumption[idxs]))

                    fixed_group[0][g].append(np.mean(self.fixed_env.post_income[idxs]))
                    fixed_group[1][g].append(np.mean(self.fixed_env.tax_array[idxs]))
                    fixed_group[2][g].append(np.mean(self.fixed_env.households.at_next[idxs]))
                    fixed_group[3][g].append(np.mean(self.fixed_env.households.e[idxs] * self.fixed_env.ht[idxs]))
                    fixed_group[4][g].append(np.mean(fixed_house[idxs]))
                    fixed_group[5][g].append(np.mean(self.fixed_env.consumption[idxs]))

                if done:
                    break
                global_obs, private_obs = next_global_obs, next_private_obs

            # --- Save overall series ---
            np.savetxt(f"{self.model_path}/maddpg_episode_{episode_idx}.csv", maddpg_data, delimiter=",")
            np.savetxt(f"{self.model_path}/random_episode_{episode_idx}.csv", random_data, delimiter=",")
            np.savetxt(f"{self.model_path}/fixed_episode_{episode_idx}.csv", fixed_data, delimiter=",")

            # --- Save group series (one file per group) ---
            for g in range(3):
                group_name = f"group_{g}"
                np.savetxt(
                    f"{self.model_path}/maddpg_{group_name}_{episode_idx}.csv",
                    maddpg_group[g], delimiter=","
                )
                np.savetxt(
                    f"{self.model_path}/random_{group_name}_{episode_idx}.csv",
                    random_group[g], delimiter=","
                )
                np.savetxt(
                    f"{self.model_path}/fixed_{group_name}_{episode_idx}.csv",
                    fixed_group[g], delimiter=","
                )

    def sav_list(self, file_name, data_list):
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in data_list:
                writer.writerow(row)

    def _evaluate_get_action(self, global_obs, private_obs):
        if self.embeddings is None:
            self.embeddings = {}
        global_obs_tensor = self._get_tensor_inputs(global_obs)
        private_obs_tensor = self._get_tensor_inputs(private_obs)
        n_households = self.envs.households.n_households
        
        house_embeddings = []
        for i in range(n_households):
            hid = f"Household{i+1}"
            emb = self.embeddings.get(hid, torch.zeros(self.args.embed_dim, device=private_obs_tensor.device))
            house_embeddings.append(emb)
        embedding_tensor = torch.stack(house_embeddings, dim=0)
        combined_private_obs = torch.cat([private_obs_tensor, embedding_tensor], dim=-1)
        
        n_global_obs = global_obs_tensor.repeat(n_households, 1)
        obs = torch.cat([n_global_obs, combined_private_obs], dim=-1)
        
        sorted_indices = torch.argsort(private_obs_tensor[:, 1], descending=True)
        sorted_obs = obs[sorted_indices]
        
        hou_action = np.zeros((n_households, self.args.house_action_dim))
        gov_action = self.agents[-1].select_action(global_obs_tensor, self.noise, self.epsilon)
        
        num_set = range(n_households)
        for i in range(self.args.agent_block_num - 1):
            if i == 0:
                num = num_set[:int(0.1 * n_households)]
            elif i == 1:
                num = num_set[int(0.1 * n_households):int(0.5 * n_households)]
            else:
                num = num_set[int(0.5 * n_households):]
            hou_action[num] = self.agents[i].select_action(sorted_obs[num], self.noise, self.epsilon)
            
        house_sort_index = sorted_indices.cpu().numpy()
        hou_action = hou_action[np.argsort(house_sort_index)]
        action = {self.envs.government.name: gov_action, self.envs.households.name: hou_action}
        return action, house_sort_index
    
    def random_evaluate_get_action(self, original_action):
        gov_action = original_action[self.envs.government.name]
        temp = np.random.random((self.args.n_households, self.envs.households.action_space.shape[1]))
        hou_action = temp * 2 - 1
        action = {
            self.envs.government.name: gov_action,
            self.envs.households.name: hou_action
        }
        return action
    
    def fixed_evaluate_get_action(self, env, original_action):
        gov_action = original_action[self.envs.government.name]
        IFE = 2
        CRRA = 1
        m = (IFE / (IFE + CRRA)) * np.log((IFE / (IFE + CRRA)) * np.exp(0.045))
        e = 0.200 * np.random.random((self.args.n_households, 1))
        h = 1/2 * e - 1/2 * m
        c = np.log(env.households.e) - e + m
        h = np.exp(h) / 2
        c = np.exp(c)
        temp = np.random.random((self.args.n_households, self.envs.households.action_space.shape[1]))
        temp[:, 0] = 1 - c.flatten()
        temp[:, 1] = h.flatten()
        hou_action = temp * 2 - 1
        action = {self.envs.government.name: gov_action,
                  self.envs.households.name: hou_action}
        return action
