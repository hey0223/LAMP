import copy 
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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

from .think_memory import ThinkMemoryManager

torch.autograd.set_detect_anomaly(True)


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

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
        # Previous indicators used for short-term shock detection
        self.prev_key_indicators = None
        
        # Storage for LLM outputs and embeddings between updates
        self.llm_results = []
        self.embeddings = None
        self.last_long_term_llm_results = None
        self.previous_evaluations = {}
        self.previous_statements = {}
        self.last_long_term_news = ""
        
        # Paper-aligned Think memories
        n_households = self.envs.households.n_households
        
        self.short_top_k = int(getattr(self.args, "short_top_k", 2))
        self.long_top_k = int(getattr(self.args, "long_top_k", 5))
        self.retrieval_top_k = int(getattr(self.args, "retrieval_top_k", 2))

        # FIX 1: initialize long-term checkpoint interval
        self.long_term_analysis_interval = int(getattr(self.args, "long_term_step_size", 50))
        if self.long_term_analysis_interval <= 0:
            raise ValueError("long_term_step_size must be a positive integer.")

        self.memory_manager = ThinkMemoryManager(
            n_households=n_households,
            short_top_k=self.short_top_k,
            long_top_k=self.long_top_k,
            retrieval_top_k=self.retrieval_top_k,
            faiss_dim=9,
        )
        
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
          
        # Create logging/checkpoint directory and save initial args
        self.model_path, _ = make_logpath(algo="maddpg", n=n_households)
        save_args(path=self.model_path, args=self.args)
        os.environ["LAMP_MODEL_PATH"] = str(self.model_path)
        self.current_epoch = -1
        
        self.fix_gov = True  # Flag to freeze government policy during certain phases
        
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
        self.current_episode_index = 0
        self.current_episode_step = 0
        self.device = torch.device('cuda' if self.args.cuda else 'cpu')

    def _init_agents(self):
        """
        Create and return a list of MADDPG core objects,
        one per agent index (household groups + government).
        """
        return [MADDPG(self.args, i) for i in range(self.args.agent_block_num)]

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
        # global
        global_obs[0] /= 1e7
        global_obs[1] /= 1e5
        global_obs[3] /= 1e5
        global_obs[4] /= 1e5
        private_obs[:, 1] /= 1e5
        return global_obs, private_obs

    def _buffer_size(self):
        return len(getattr(self.buffer, "storge", []))

    def _safe_preview(self, text, limit=160):
        if text is None:
            return "[None]"
        flat = str(text).replace("\n", " ").strip()
        if len(flat) <= limit:
            return flat
        return flat[:limit] + "..."

    def _module_grad_norm(self, module):
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def _log_array_stats(self, name, arr, step=None, prefix=""):
        try:
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            arr = np.asarray(arr)
            if arr.size == 0:
                logging.info(f"{prefix}[{name}] empty")
                return
            logging.info(
                f"{prefix}[{name}]"
                + (f" step={step}" if step is not None else "")
                + f" shape={arr.shape}, mean={np.mean(arr):.6f}, std={np.std(arr):.6f}, "
                  f"min={np.min(arr):.6f}, max={np.max(arr):.6f}"
            )
        except Exception as e:
            logging.warning(f"Failed to log stats for {name}: {e}")

    def _log_step_summary(self, global_step, gov_action, hou_action, gov_reward, house_reward, done, duration):
        logging.info(
            f"[Train Step {global_step}] gov_reward={float(gov_reward):.6f}, "
            f"house_reward_mean={float(np.mean(house_reward)):.6f}, "
            f"house_reward_std={float(np.std(house_reward)):.6f}, done={done}, duration={duration:.2f}s"
        )
        self._log_array_stats("gov_action", gov_action, step=global_step, prefix="[Train]")
        self._log_array_stats("house_action", hou_action, step=global_step, prefix="[Train]")
        self._log_array_stats("house_reward", house_reward, step=global_step, prefix="[Train]")

    def compute_key_indicators(self):
        indicators = {}
        if hasattr(self.envs, "per_household_gdp"):
            indicators["per_gdp"] = self.envs.per_household_gdp
        else:
            logging.warning("per_gdp not found in training environment.")
        if hasattr(self, "last_house_reward"):
            indicators["social_welfare"] = np.mean(self.last_house_reward)
        else:
            logging.warning("last_house_reward not found in training environment.")
        #if hasattr(self.envs, "income_gini"):
            #indicators["income_gini"] = self.envs.income_gini
        #else:
            #print("income_gini not found in training environment.")
        if hasattr(self.envs, "wealth_gini"):
            indicators["wealth_gini"] = self.envs.wealth_gini
        else:
            logging.warning("wealth_gini not found in training environment.")
        logging.info("Key indicators computed: " + str(indicators))
        return indicators

    async def generate_short_term_news(self, step, short_term_news, private_obs, episode_index=None, episode_step=None):
        n_households = self.envs.households.n_households
        tasks = []
        
        for i in range(n_households):
            recent_result_str = self.get_recent_long_term_news(i)
            
            tasks.append(
                self.envs.households.derive_reasoning_text(
                    short_term_news=short_term_news,
                    recent_long_term_news=recent_result_str,
                    private_observation=private_obs[i],
                    temperature=0.6,
                    max_retries=8,
                )
            )
        derived_jsons = await asyncio.gather(*tasks)
        return derived_jsons
    
    def get_recent_long_term_news(self, household_idx):
        return self.last_long_term_news if self.last_long_term_news else ""

    def _normalized_change(self, key, current_value, prev_value):
        current_value = float(current_value)
        prev_value = float(prev_value)

        floor_map = {
            "per_gdp": 1e5,
            "social_welfare": 1.0,
            "wealth_gini": 1e-3,
        }

        denom = max(abs(prev_value), floor_map.get(key, 1e-8))
        return abs(current_value - prev_value) / denom

    def should_generate_short_term(self, current_key_indicators, step, threshold):
        """
        Normalized short-term trigger:
        - only evaluate on non-checkpoint intermediate steps
        - trigger when NORMALIZED change of any key indicator exceeds threshold
        """
        if self.prev_key_indicators is None or step <= 0:
            return False

        monitored_keys = ("wealth_gini", "social_welfare", "per_gdp")

        max_norm_delta = 0.0
        max_key = None
        max_prev = None
        max_curr = None

        for key in monitored_keys:
            if key not in current_key_indicators or key not in self.prev_key_indicators:
                continue

            current_value = float(current_key_indicators[key])
            prev_value = float(self.prev_key_indicators[key])

            norm_delta = self._normalized_change(
                key=key,
                current_value=current_value,
                prev_value=prev_value,
            )

            if norm_delta > max_norm_delta:
                max_norm_delta = norm_delta
                max_key = key
                max_prev = prev_value
                max_curr = current_value

        if max_key is not None and max_norm_delta > threshold:
            logging.info(
                f"[ShortTerm Trigger] key={max_key}, "
                f"prev={max_prev:.6f}, curr={max_curr:.6f}, "
                f"normalized_change={max_norm_delta:.6f} > threshold={threshold:.6f}"
            )
            return True

        return False 
    
    def should_generate_long_term(self, step):
        """
        Paper-aligned long-term trigger:
        - fixed checkpoints only
        - keep step == -1 as an optional manual force-refresh hook
        """
        if step == -1:
            return True
        if step <= 0:
            return False
        return (step % self.long_term_analysis_interval) == 0

    def llm_generate_embeddings(self, step, raw_global_obs, raw_private_obs):
        episode_index = int(self.current_episode_index)
        episode_step = int(getattr(self.envs, "step_cnt", self.current_episode_step))
        self.current_episode_step = episode_step

        # Step 1: Record the latest raw global observation for LLM/news
        self.intelligence_bureau.collect_observations(raw_global_obs.copy())
        current_key_indicators = self.compute_key_indicators()

        # Step 2: Decide news type with paper-aligned precedence:
        # long-term checkpoint > short-term shock > no news
        generate_long = self.should_generate_long_term(step)

        if generate_long:
            generate_short = False
        else:
            generate_short = self.should_generate_short_term(
                current_key_indicators=current_key_indicators,
                step=step,
                threshold=self.args.threshold
            )

        if generate_long:
            self.last_news_type = "long"
        elif generate_short:
            self.last_news_type = "short"
        else:
            self.last_news_type = None

        logging.info(
            f"[LLM Trigger] step={step+1}, generate_long={generate_long}, "
            f"generate_short={generate_short}, last_news_type={self.last_news_type}, "
            f"indicators={current_key_indicators}"
        )

        # update previous indicators after decision
        self.prev_key_indicators = dict(current_key_indicators)

        # If no news generation is triggered, reuse the existing embeddings
        if not generate_short and not generate_long:
            logging.info(f"No news update at step {step+1}; reusing existing embeddings.")
            return None

        # Step 3: Generate news as needed
        recent_long_term_news = None
        if self.intelligence_bureau.long_term_memory:
            recent_long_term_news = self.intelligence_bureau.long_term_memory[-1].content

        short_term_news = (
            self.intelligence_bureau.generate_news(
                recent_only=True,
                recent_long_term_news=recent_long_term_news,
            )
            if generate_short else None
        )
        long_term_news  = (
            self.intelligence_bureau.generate_news(
                recent_only=False,
            )
            if generate_long else None
        )
        if long_term_news:
            self.last_long_term_news = long_term_news
        
        # Ensure an asyncio event loop is available
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Step 4: Prepare short-term reasoning if needed
        short_term_jsons = None
        if short_term_news:
            short_term_jsons = loop.run_until_complete(
                self.generate_short_term_news(
                    step=step,
                    short_term_news=short_term_news,
                    private_obs=raw_private_obs,
                    episode_index=episode_index,
                    episode_step=episode_step,
                )
            )

        # Step 5: If we have long-term news, run dialogue round as the backbone
        if long_term_news:
            n_households = self.envs.households.n_households
            similar_experience_map = {}

            for i in range(n_households):
                h_id = f"Household{i+1}"
                similar_experience_map[h_id] = self.memory_manager.build_similar_experience_text(
                    household_id=h_id,
                    raw_global_obs=raw_global_obs,
                    raw_private_obs=raw_private_obs[i]
                )

            logging.info(f"Retrieved paper-aligned H_short + H_long context at step {step+1}:")
            for h_id, exp_text in similar_experience_map.items():
                logging.info(
                    f"[Retrieval] step={step+1}, agent={h_id}, "
                    f"text_len={len(exp_text) if exp_text else 0}, preview={self._safe_preview(exp_text, limit=200)}"
                )

            long_term_results = loop.run_until_complete(
                self.dialogue_manager.run_dialogue_round(
                    step=step,
                    long_term_news=long_term_news,
                    private_obs=raw_private_obs,
                    similar_experience_map=similar_experience_map,
                    episode_index=episode_index,
                    episode_step=episode_step,
                )
            )
            self.last_long_term_llm_results = long_term_results

            # Step 6: If short-term reasoning also exists, merge it into long-term results
            if short_term_jsons is not None:
                new_results = self.merge_short_and_long_term_results(
                    long_term_results=long_term_results,
                    short_term_jsons=short_term_jsons
                )
            else:
                new_results = long_term_results

        else:
            # No long-term news: fall back to short-term-only reasoning updates
            if short_term_jsons is None:
                logging.warning(f"Short-term trigger fired but no short-term reasoning generated at step {step+1}.")
                return None
            new_results = self.process_derived_jsons(short_term_jsons)
        logging.info(f"[LLM Results] step={step+1}, count={len(new_results) if new_results is not None else 0}, news_type={self.last_news_type}")
        for item in (new_results or []):
            logging.info(
                f"[LLM Result] step={step+1}, agent={item.get('agent_id')}, "
                f"economic_status={item.get('economic_status')}, reasoning_len={len(item.get('reasoning', ''))}, "
                f"has_evaluation={item.get('evaluation') is not None}"
            )

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

    def merge_short_and_long_term_results(self, long_term_results, short_term_jsons):
        """
        Merge short-term reasoning into long-term dialogue results agent by agent.

        Rules:
        - Keep long-term dialogue outputs (statements / evaluation) as the backbone.
        - If short-term reasoning exists, prepend/append it into the final reasoning text.
        - Use short-term economic_status only when long-term result is missing or invalid.
        """
        merged_results = []
        n_households = self.envs.households.n_households

        for i in range(n_households):
            agent_id = f"Household{i+1}"

            long_result = None
            if isinstance(long_term_results, list):
                for item in long_term_results:
                    if item.get("agent_id") == agent_id:
                        long_result = item
                        break

            short_json = short_term_jsons[i] if i < len(short_term_jsons) else {}

            short_reasoning = short_json.get("reasoning", "") if isinstance(short_json, dict) else ""
            short_status = short_json.get("economic_status", None) if isinstance(short_json, dict) else None

            if long_result is None:
                # fallback: if somehow long-term result missing, build from short-term path
                merged_results.append({
                    "agent_id": agent_id,
                    "evaluation": self.get_default_evaluation(i),
                    "economic_status": short_status if short_status is not None else 1,
                    "reasoning": short_reasoning or "No valid reasoning response from LLM after retries."
                })
                continue

            long_reasoning = long_result.get("reasoning", "")
            long_status = long_result.get("economic_status", 1)

            if short_reasoning:
                merged_reasoning = (
                    f"[Short-Term Shock Analysis]\n{short_reasoning}\n\n"
                    f"[Long-Term Structural Analysis]\n{long_reasoning}"
                )
            else:
                merged_reasoning = long_reasoning

            merged_result = dict(long_result)
            merged_result["reasoning"] = merged_reasoning
            merged_result["economic_status"] = long_status if long_status is not None else (
                short_status if short_status is not None else 1
            )

            merged_results.append(merged_result)

        return merged_results

    def get_default_evaluation(self, household_idx):
        n_households = self.envs.households.n_households
        ordered_agent_ids = [f"Household{i+1}" for i in range(n_households)]
        current_agent_id = f"Household{household_idx+1}"

        default_evaluation = {
            "wealth_guesses_by_agent": {agent_id: 1 for agent_id in ordered_agent_ids},
            "trust_levels_by_agent": {
                agent_id: (10 if agent_id == current_agent_id else 5)
                for agent_id in ordered_agent_ids
            },
            "reflection_text": ""
        }
        if hasattr(self, "last_long_term_llm_results"):
            for prev_result in self.last_long_term_llm_results:
                if prev_result.get("agent_id", "") == f"Household{household_idx+1}":
                    return prev_result.get("evaluation", default_evaluation)
        return default_evaluation

    def construct_embeddings(self):
        embeddings = {}

        for house_idx in range(self.envs.households.n_households):
            hid = f"Household{house_idx+1}"
            evaluation_str, obs_text = self.construct_evaluation_text(house_idx)
            embed_t = Household_embed(evaluation=evaluation_str, obs_text=obs_text)
            embeddings[hid] = embed_t

        self.embeddings = embeddings
        for ag in self.agents:
            ag.embeddings = embeddings
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

            reflection_text = evaluation.get("reflection_text", "")
            evaluation_str += (
                f"🔹 Overall Economic Condition (0=Bad, 1=Neutral, 2=Good): {economic_status}\n"
                f"🔹 Reflection & Key Insights:\n{reflection_text}\n"
            )
            obs_text = result.get("reasoning", obs_text)
            break
        return evaluation_str, obs_text

    def learn(self):
        n_households = self.envs.households.n_households

        # --- Initialize environment and observations ---
        raw_global_obs, raw_private_obs = self.envs.reset()
        self.memory_manager.reset_short_term_memory()
        self.prev_key_indicators = None
        self.last_news_type = None
        self.current_episode_index = 0
        self.current_episode_step = int(getattr(self.envs, "step_cnt", 0))

        global_obs, private_obs = self.observation_wrapper(
            raw_global_obs.copy(),
            raw_private_obs.copy()
        )

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
        logging.info(f"Initialized loss log at: {log_file}")

        logging.info(f"n_epochs: {self.args.n_epochs}, epoch_length: {self.args.epoch_length}")

        # --- Epoch loop ---
        for epoch in range(self.args.n_epochs):
            self.current_epoch = epoch
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            epoch_update_count = 0

            # (Re-)generate embeddings at the start of each epoch if needed
            new_embeddings = self.llm_generate_embeddings(-1, raw_global_obs, raw_private_obs)
            if new_embeddings is not None:
                self.embeddings = new_embeddings

            # Distribute the latest embeddings to all agents
            for agent in self.agents:
                agent.embeddings = self.embeddings

            logging.info(f"[Epoch {epoch+1}/{self.args.n_epochs}] start | noise={self.noise:.4f}, epsilon={self.epsilon:.4f}, buffer_size={self._buffer_size()}, has_embeddings={self.embeddings is not None}")

            # --- Step loop ---
            for t in range(self.args.epoch_length):
                start_time = time.time()
                global_step = epoch * self.args.epoch_length + t
                self.current_episode_step = int(getattr(self.envs, "step_cnt", self.current_episode_step))
                episode_step = self.current_episode_step

                pre_action_raw_global_obs = raw_global_obs.copy()
                pre_action_raw_private_obs = raw_private_obs.copy()
                pre_action_llm_results = self.llm_results
                pre_action_news_type = self.last_news_type

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
                    hou_action[idxs] = self.agents[group_idx].select_action(
                        sorted_obs[list(idxs)],
                        self.noise,
                        self.epsilon
                    )

                orig_order = sorted_indices.cpu().numpy()
                hou_action = hou_action[np.argsort(orig_order)]

                action = {
                    self.envs.government.name: gov_action,
                    self.envs.households.name: hou_action
                }

                # Prepare current embeddings for buffer storage
                def get_projected_emb_array(emb_dict):
                    if emb_dict is None:
                        return np.zeros((n_households, self.args.embed_dim), dtype=np.float32)
                    return np.stack([
                        emb_dict[f"Household{i+1}"].detach().cpu().numpy().astype(np.float32)
                        for i in range(n_households)
                    ], axis=0)

                current_embeddings_arr = get_projected_emb_array(self.embeddings)

                # (B) --- Step the environment ---
                raw_next_global_obs, raw_next_private_obs, gov_reward, house_reward, done = self.envs.step(action)

                next_global_obs, next_private_obs = self.observation_wrapper(
                    raw_next_global_obs.copy(),
                    raw_next_private_obs.copy()
                )

                # Store last rewards for logging
                self.last_gov_reward = gov_reward
                self.last_house_reward = house_reward

                if pre_action_news_type is not None and pre_action_llm_results:
                    self.memory_manager.record_reasoning_trajectories(
                        action=action,
                        rewards=house_reward,
                        raw_global_obs=pre_action_raw_global_obs,
                        raw_private_obs=pre_action_raw_private_obs,
                        llm_results=pre_action_llm_results,
                        household_name=self.envs.households.name,
                        n_households=self.envs.households.n_households,
                        last_news_type=pre_action_news_type,
                    )

                # (D) --- Generate / refresh embeddings for NEXT state ---
                new_embeddings = self.llm_generate_embeddings(
                    global_step,
                    raw_next_global_obs,
                    raw_next_private_obs
                )

                # If new_embeddings is None, it means we reuse the old ones
                if new_embeddings is not None:
                    self.embeddings = new_embeddings
                    for ag in self.agents:
                        ag.embeddings = new_embeddings

                next_embeddings_arr = get_projected_emb_array(self.embeddings)

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
                raw_global_obs, raw_private_obs = raw_next_global_obs, raw_next_private_obs
                global_obs, private_obs = next_global_obs, next_private_obs

                # If the episode ends, reset environment
                if done:
                    raw_global_obs, raw_private_obs = self.envs.reset()
                    self.memory_manager.reset_short_term_memory()
                    self.prev_key_indicators = None
                    self.last_news_type = None
                    self.current_episode_index += 1
                    self.current_episode_step = int(getattr(self.envs, "step_cnt", 0))

                    global_obs, private_obs = self.observation_wrapper(
                        raw_global_obs.copy(),
                        raw_private_obs.copy()
                    )

                # (E) --- Train networks every 10 steps ---
                if t % 10 == 0:
                    transitions = self.buffer.sample(self.args.batch_size)
                    step_actor_loss = 0
                    step_critic_loss = 0

                    for agent in self.agents:
                        actor_loss, critic_loss = agent.train(
                            transitions,
                            self.agents
                        )
                        step_actor_loss += actor_loss
                        step_critic_loss += critic_loss

                    has_projector_grad = False
                    projector_grad_norm = 0.0

                    logging.info(
                        f"[Update] step={global_step}, actor_loss={step_actor_loss:.6f}, "
                        f"critic_loss={step_critic_loss:.6f}, projector_has_grad={has_projector_grad}, "
                        f"projector_grad_norm={projector_grad_norm:.6f}, buffer_size={self._buffer_size()}"
                    )
                    epoch_actor_loss += step_actor_loss
                    epoch_critic_loss += step_critic_loss
                    epoch_update_count += 1

                elif global_step % 20 == 0:
                    logging.info(
                        f"[Update Skipped] step={global_step}, buffer_size={self._buffer_size()}, "
                        f"batch_size={self.args.batch_size}, initial_train={self.args.initial_train}, "
                        f"update_freq={self.args.update_freq}"
                    )

                # Log step duration
                duration = time.time() - start_time
                if global_step % 20 == 0 or done:
                    self._log_step_summary(global_step, gov_action, hou_action, gov_reward, house_reward, done, duration)
                    logging.info(
                        f"[Env Summary] step={global_step}, per_gdp={getattr(self.envs, 'per_household_gdp', float('nan')):.6f}, "
                        f"income_gini={getattr(self.envs, 'income_gini', float('nan')):.6f}, "
                        f"wealth_gini={getattr(self.envs, 'wealth_gini', float('nan')):.6f}, "
                        f"wage={getattr(self.envs, 'WageRate', float('nan')):.6f}, Bt={getattr(self.envs, 'Bt', float('nan')):.6f}, "
                        f"Kt={getattr(self.envs, 'Kt', float('nan')):.6f}"
                    )
                self.log_step_time(epoch, t, duration)

            # Periodic logging, evaluation, and checkpointing
            if epoch % self.args.display_interval == 0:
                logging.info(f"Displaying metrics at epoch {epoch+1}")
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
                logging.info(f"Saved epoch {epoch+1} metrics to {log_file}")

            # Save checkpoints
            if epoch % self.args.save_interval == 0:
                save_dir = os.path.join(self.model_path, f"epoch_{epoch}")
                os.makedirs(save_dir, exist_ok=True)
                for i, agent in enumerate(self.agents):
                    logging.info(f"Saving agent {i} checkpoint at epoch {epoch+1}")
                    torch.save(agent.actor_network.state_dict(), f"{save_dir}/agent_{i}.pt")
                self.memory_manager.faiss_manager_long_term.save(save_dir)

            # Decay exploration parameters
            self.noise = max(0.05, self.noise - 5e-7)
            self.epsilon = max(0.05, self.epsilon - 5e-7)

        # After all epochs, save final FAISS index and optionally finish W&B
        self.memory_manager.faiss_manager_long_term.save(str(self.model_path))
        logging.info("Training complete. FAISS index and experiences saved.")
        if self.wandb:
            wandb.finish()

    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        return obs_tensor
    
    def test(self):
        path = ""
        for agent_i in range(len(self.agents)):
            self.agents[agent_i].actor_network.load_state_dict(
            torch.load(path + f'/agent_{agent_i}.pt', weights_only=True))
        #self.episode_evolution()
        self.light_episode_evolution()


    def _evaluate_agent(self):
        economic_indicators = []
        column_names = [
            "epoch", "gov_rew", "social_welfare", "years", "total_income", "total_tax", "income_tax",
            "total_wealth", "wealth_tax", "per_gdp", "income_gini", "wealth_gini", "wage",
            "total_labor", "total_consumption", "Bt", "Kt", "Gt_prob", "income_tau",
            "income_xi", "wealth_tau", "wealth_xi"
        ]

        old_embeddings = self.embeddings
        old_llm_results = copy.deepcopy(self.llm_results)
        old_prev_key_indicators = copy.deepcopy(self.prev_key_indicators)
        old_last_news_type = self.last_news_type
        old_last_long_term_llm_results = copy.deepcopy(self.last_long_term_llm_results)
        old_last_long_term_news = self.last_long_term_news

        old_short_term_memory = None
        if hasattr(self.memory_manager, "short_term_memory"):
            old_short_term_memory = copy.deepcopy(self.memory_manager.short_term_memory)

        try:
            self.embeddings = None
            for agent in self.agents:
                agent.embeddings = None

            for episode_idx in range(self.args.eval_episodes):
                step_count = 0
                raw_global_obs, raw_private_obs = self.eval_env.reset()
                global_obs, private_obs = self.observation_wrapper(
                    raw_global_obs.copy(),
                    raw_private_obs.copy()
                )

                episode_data = []

                self.memory_manager.reset_short_term_memory()
                self.prev_key_indicators = None
                self.last_news_type = None
                self.llm_results = None
                self.last_long_term_llm_results = None
                self.last_long_term_news = None

                while True:
                    with torch.no_grad():
                        action, sort_idx = self._evaluate_get_action(global_obs, private_obs)
                        raw_next_global_obs, raw_next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                        next_global_obs, next_private_obs = self.observation_wrapper(
                            raw_next_global_obs.copy(),
                            raw_next_private_obs.copy()
                        )

                    step_count += 1
                    steps = 1

                    total_income = np.mean(self.eval_env.post_income)
                    total_tax = np.mean(self.eval_env.tax_array)
                    income_tax = np.mean(self.eval_env.income_tax)
                    total_wealth = np.mean(self.eval_env.households.at_next)
                    wealth_tax = np.mean(self.eval_env.asset_tax)
                    per_gdp = self.eval_env.per_household_gdp
                    income_gini = self.eval_env.income_gini
                    wealth_gini = self.eval_env.wealth_gini
                    wage = self.eval_env.WageRate
                    total_labor = self.eval_env.Lt
                    total_consumption = np.mean(self.eval_env.consumption)
                    Bt = self.eval_env.Bt
                    Kt = self.eval_env.Kt
                    Gt_prob = self.eval_env.Gt_prob
                    income_tau = self.eval_env.government.tau
                    income_xi = self.eval_env.government.xi
                    wealth_tau = self.eval_env.government.tau_a
                    wealth_xi = self.eval_env.government.xi_a
                    social_welfare = np.mean(house_reward)

                    episode_data.append([
                        episode_idx,
                        gov_reward,
                        social_welfare,
                        steps,
                        total_income,
                        total_tax,
                        income_tax,
                        total_wealth,
                        wealth_tax,
                        per_gdp,
                        income_gini,
                        wealth_gini,
                        wage,
                        total_labor,
                        total_consumption,
                        Bt,
                        Kt,
                        Gt_prob,
                        income_tau,
                        income_xi,
                        wealth_tau,
                        wealth_xi,
                    ])

                    if done:
                        break

                    raw_global_obs, raw_private_obs = raw_next_global_obs, raw_next_private_obs
                    global_obs, private_obs = next_global_obs, next_private_obs

                years = len(episode_data)
                avg_data = np.mean(episode_data, axis=0)
                avg_dict = dict(zip(column_names, avg_data))
                avg_dict["epoch"] = episode_idx
                avg_dict["gov_rew"] *= years
                avg_dict["social_welfare"] *= years
                avg_dict["years"] = years

                economic_indicators.append([avg_dict[col] for col in column_names])

            log_file = f"{self.model_path}/economic_indicators.csv"
            df = pd.DataFrame(economic_indicators, columns=column_names)
            df.to_csv(log_file, mode="a", header=not pd.io.common.file_exists(log_file), index=False)

            overall_avg = np.mean(economic_indicators, axis=0)
            eval_summary = dict(zip(column_names, overall_avg))
            logging.info(
                f"[Eval Summary] gov_rew={eval_summary['gov_rew']:.6f}, social_welfare={eval_summary['social_welfare']:.6f}, "
                f"per_gdp={eval_summary['per_gdp']:.6f}, income_gini={eval_summary['income_gini']:.6f}, "
                f"wealth_gini={eval_summary['wealth_gini']:.6f}, years={eval_summary['years']:.2f}"
            )
            return eval_summary

        finally:
            self.embeddings = old_embeddings
            self.llm_results = old_llm_results
            self.prev_key_indicators = old_prev_key_indicators
            self.last_news_type = old_last_news_type
            self.last_long_term_llm_results = old_last_long_term_llm_results
            self.last_long_term_news = old_last_long_term_news

            if old_short_term_memory is not None:
                self.memory_manager.short_term_memory = old_short_term_memory

            for agent in self.agents:
                agent.embeddings = self.embeddings

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
            raw_global_obs, raw_private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(
                raw_global_obs.copy(),
                raw_private_obs.copy()
            )
            self.memory_manager.reset_short_term_memory()
            self.prev_key_indicators = None
            self.last_news_type = None
            new_embeddings = self.llm_generate_embeddings(-1, raw_global_obs, raw_private_obs)
            if new_embeddings is not None:
                self.embeddings = new_embeddings
            for agent in self.agents:
                agent.embeddings = self.embeddings

            while True:
                with torch.no_grad():
                    action, sort_idx = self._evaluate_get_action(global_obs, private_obs)
                    raw_next_global_obs, raw_next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    next_global_obs, next_private_obs = self.observation_wrapper(
                        raw_next_global_obs.copy(),
                        raw_next_private_obs.copy()
                    )

                    # Store last rewards for logging
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

                raw_global_obs, raw_private_obs = raw_next_global_obs, raw_next_private_obs
                global_obs, private_obs = next_global_obs, next_private_obs

                # Potentially update embeddings mid-episode
                new_embeddings = self.llm_generate_embeddings(step_count, raw_global_obs, raw_private_obs)
                if new_embeddings is not None:
                    self.memory_manager.record_reasoning_trajectories(
                        action=action,
                        rewards=house_reward,
                        raw_global_obs=raw_next_global_obs,
                        raw_private_obs=raw_next_private_obs,
                        llm_results=self.llm_results,
                        household_name=self.envs.households.name,
                        n_households=self.envs.households.n_households,
                        last_news_type=self.last_news_type,
                    )
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
            raw_global_obs, raw_private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(
                raw_global_obs.copy(),
                raw_private_obs.copy()
            )
            self.memory_manager.reset_short_term_memory()
            self.prev_key_indicators = None
            self.last_news_type = None
            # Containers for overall indicators: [total_tax, house_reward, total_labor, total_consumption, gov_reward, gdp, income_gini, wealth_gini]
            maddpg_data = [[] for _ in range(8)]
            random_data = [[] for _ in range(8)]
            fixed_data  = [[] for _ in range(8)]

            # Sync embeddings at start
            new_embeddings = self.llm_generate_embeddings(-1, raw_global_obs, raw_private_obs)
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
                    raw_next_global_obs, raw_next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    self.last_gov_reward = gov_reward
                    self.last_house_reward = house_reward
                    next_global_obs, next_private_obs = self.observation_wrapper(
                        raw_next_global_obs.copy(),
                        raw_next_private_obs.copy()
                    )
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

                if done:
                    break
                raw_global_obs, raw_private_obs = raw_next_global_obs, raw_next_private_obs
                global_obs, private_obs = next_global_obs, next_private_obs

            # --- Save overall series ---
            np.savetxt(f"{self.model_path}/maddpg_episode_{episode_idx}.csv", maddpg_data, delimiter=",")
            np.savetxt(f"{self.model_path}/random_episode_{episode_idx}.csv", random_data, delimiter=",")
            np.savetxt(f"{self.model_path}/fixed_episode_{episode_idx}.csv", fixed_data, delimiter=",")

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
    