import asyncio
import logging
import random
import torch
from typing import List, Dict, Any
from .embedding import run_policy_inference

class DialogueManager:
    def __init__(self, env):
        self.env = env
        self.previous_statements = {}   # Stores each agent's last statement
        self.previous_reasoning = {}    # Stores each agent's last reasoning
        self.previous_evaluations = {}  # Stores each agent's last evaluation results
    def _ordered_agent_ids(self):
        n_households = self.env.households.n_households
        return [f"Household{i+1}" for i in range(n_households)]

    def _map_to_ordered_list(self, agent_value_map, default_value):
        ordered_ids = self._ordered_agent_ids()
        return [agent_value_map.get(agent_id, default_value) for agent_id in ordered_ids]

    async def run_dialogue_round(self, step: int, long_term_news: str, similar_experience_map: dict, private_obs: list, episode_index=None, episode_step=None) -> list:
        """
        Asynchronously run one dialogue round.
        Parallelizes LLM calls, policy inference, and evaluations across agents
        to reduce overall runtime.
        """
        n_households = self.env.households.n_households

        # Phase 1: Concurrently invoke each agent's LLM processing
        tasks = []
        for household_id in range(n_households):
            agent_id = f"Household{household_id + 1}"
            personal_similar_experience = ""
            if similar_experience_map is not None:
                personal_similar_experience = similar_experience_map.get(agent_id, "")

            tasks.append(
                self.process_household_llm(
                    household_id,
                    step,
                    long_term_news,
                    personal_similar_experience,
                    private_obs[household_id],
                    episode_index=episode_index,
                    episode_step=episode_step,
                )
            )
        llm_results = await asyncio.gather(*tasks)
        llm_results = [res for res in llm_results if res is not None]

        # Phase 2: Concurrent policy inference using run_in_executor for sync calls
        loop = asyncio.get_running_loop()
        tasks = []
        for result in llm_results:
            tasks.append(loop.run_in_executor(
                None,
                self.process_policy_inference,
                result,
                step,
                private_obs,
                episode_index,
                episode_step,
            ))
        updated_results = await asyncio.gather(*tasks)

        for result in updated_results:
            if result is not None:
                self.previous_statements[result["agent_id"]] = result.get("chosen_statement") or ""

        # Phase 3: Concurrently evaluate each agent's final statement
        tasks = [self.process_evaluation(result, updated_results, private_obs, episode_index=episode_index, episode_step=episode_step) for result in updated_results]
        final_results = await asyncio.gather(*tasks)

        for res in final_results:
            if res is not None:
                logging.debug(
                    f"[DialogueRound] {res['agent_id']} "
                    f"chosen_statement={res.get('chosen_statement')!r}, "
                    f"candidate_count={len(res.get('statements', []))}"
                )
        return final_results
    
    async def process_household_llm(self, household_id, step, long_term_news, similar_experience, private_obs_item, episode_index=None, episode_step=None):
        """
        Asynchronously perform one agent's LLM call.
        Calls the environment's analyze_and_communicate method, constructs the
        dialogue data for this round, and saves info for the next round.
        """
        agent_id = f"Household{household_id + 1}"
        try:
            # Await the asynchronous LLM call for analysis and communication
            llm_result = await self.env.households.analyze_and_communicate(
                long_term_news,
                similar_experience,
                private_obs_item,
            )

            # Extract returned fields
            personal_statements = llm_result.get("statements", [])
            personal_analysis   = llm_result.get("analysis", "")
            economic_status     = llm_result.get("economic_status", 1)
            personal_reasoning  = llm_result.get("reasoning", "")

            self.previous_reasoning[agent_id] = personal_analysis + personal_reasoning
            logging.info(
                f"[Dialogue LLM] agent={agent_id}, step={step}, statements={len(personal_statements)}, "
                f"economic_status={economic_status}, reasoning_len={len(personal_analysis + personal_reasoning)}"
            )
        
            return {
                "step": step,
                "episode_index": episode_index,
                "episode_step": episode_step,
                "agent_id": agent_id,
                "private_obs": private_obs_item.tolist() if hasattr(private_obs_item, "tolist") else private_obs_item,
                "statements": personal_statements,
                "reasoning": personal_analysis + personal_reasoning,
                "economic_status": economic_status,
                "chosen_statement": None,
                "evaluation": None
            }
        except Exception as e:
            logging.error(f"Error in dialogue generation for {agent_id}: {repr(e)}", exc_info=True)
            return None
        
    def process_policy_inference(self, result, step, private_obs, episode_index=None, episode_step=None):
        """
        Perform synchronous policy inference.
        Samples one statement based on the probability distribution
        and updates the result dict.
        """
        try:
            if not result.get("statements") or len(result["statements"]) == 0:
                logging.warning(f"No statements available for {result['agent_id']}, skipping policy inference")
                result["probabilities"] = []
                result["chosen_statement"] = None
                return result

            evaluation = self.previous_evaluations.get(result["agent_id"], {})
            household_index = int(result['agent_id'].replace("Household", "")) - 1
            n_households = self.env.households.n_households

            personal_labor_productivity = private_obs[household_index][0]
            personal_wealth = private_obs[household_index][1]


            wealth_guesses_by_agent = evaluation.get("wealth_guesses_by_agent", {})
            trust_levels_by_agent = evaluation.get("trust_levels_by_agent", {})

            if wealth_guesses_by_agent:
                wealth_guesses = self._map_to_ordered_list(wealth_guesses_by_agent, 1)
            else:
                wealth_guesses = evaluation.get("wealth_guesses", [1] * n_households)

            if trust_levels_by_agent:
                trust_levels = self._map_to_ordered_list(trust_levels_by_agent, 5.0)
            else:
                trust_levels = evaluation.get("trust_levels", [5.0] * n_households)

            if 0 <= household_index < len(trust_levels):
                trust_levels[household_index] = 10.0

            statement_probs, statement_scores, statement_attn = run_policy_inference(
                player_id=household_index,
                economic_status=result.get("economic_status", 1),
                personal_labor_productivity=personal_labor_productivity,
                personal_wealth=personal_wealth,
                step=step,
                wealth_guesses=wealth_guesses,
                trust_levels=trust_levels,
                obs_text=result["reasoning"],
                candidate_action_texts=result["statements"],
                temperature=1.0,
            )

            if isinstance(statement_probs, torch.Tensor):
                statement_probs = statement_probs.detach().cpu().tolist()

            if isinstance(statement_scores, torch.Tensor):
                result["scores"] = statement_scores.detach().cpu().tolist()
            else:
                result["scores"] = statement_scores if statement_scores is not None else []

            if isinstance(statement_attn, torch.Tensor):
                if statement_attn.dim() != 2:
                    raise ValueError(
                        f"Expected 2D attention matrix, got shape={tuple(statement_attn.shape)}"
                    )
                result["attention_weights"] = statement_attn.detach().cpu().tolist()
            else:
                result["attention_weights"] = statement_attn if statement_attn is not None else []

            result["probabilities"] = statement_probs if statement_probs is not None else []

            if len(result["statements"]) == len(result["probabilities"]) and len(result["probabilities"]) > 0:
                chosen_index = random.choices(
                    range(len(result["probabilities"])),
                    weights=result["probabilities"],
                    k=1
                )[0]
                result["chosen_statement"] = result["statements"][chosen_index]
                result["chosen_statement_index"] = chosen_index
            else:
                result["chosen_statement"] = None
                result["chosen_statement_index"] = None

            logging.info(
                f"[PolicyInference] agent={result['agent_id']}, step={step}, num_candidates={len(result.get('statements', []))}, "
                f"chosen_idx={result.get('chosen_statement_index')}, max_prob={max(result.get('probabilities', [0])) if result.get('probabilities') else 0:.6f}"
            )
            return result

        except Exception as e:
            logging.error(f"Error in probability generation for {result['agent_id']}: {repr(e)}", exc_info=True)
            result["probabilities"] = []
            result["scores"] = []
            result["attention_weights"] = []
            result["chosen_statement"] = None
            result["chosen_statement_index"] = None
            return result
                   
    async def process_evaluation(self, result, all_results, private_obs, episode_index=None, episode_step=None):
        """
        Asynchronously evaluate the agent's chosen statement.
        Calls the environment's evaluate_agents_statements method
        and updates the evaluation field.
        """
        agent_id = result["agent_id"]

        personal_statement = result.get("chosen_statement") or ""
        personal_reasoning = self.previous_reasoning.get(agent_id, "")

        try:
            other_agents_statements = [
                {
                    "agent_id": r["agent_id"],
                    "statement": r.get("chosen_statement")
                }
                for r in all_results
                if r["agent_id"] != agent_id and r.get("chosen_statement")
            ]

            household_index = int(agent_id.replace("Household", "")) - 1
            analysis_result = await self.env.households.evaluate_agents_statements(
                current_agent_id=agent_id,
                personal_statement=personal_statement,
                personal_reasoning=personal_reasoning,
                other_agents_statements=other_agents_statements,
                private_observation=private_obs[household_index],
            )

            result["evaluation"] = analysis_result
            self.previous_evaluations[agent_id] = analysis_result
            logging.info(
                f"[Dialogue Evaluation] agent={agent_id}, other_agents={len(other_agents_statements)}, "
                f"evaluation_keys={list(analysis_result.keys()) if isinstance(analysis_result, dict) else type(analysis_result).__name__}"
            )
            return result

        except Exception as e:
            logging.error(f"Error in evaluation for {agent_id}: {repr(e)}", exc_info=True)
            result["evaluation"] = {}
            return result