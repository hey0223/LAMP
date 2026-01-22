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

    async def run_dialogue_round(self, step: int, long_term_news: str, similar_experience: str, private_obs: list) -> list:
        """
        Asynchronously run one dialogue round.
        Parallelizes LLM calls, policy inference, and evaluations across agents
        to reduce overall runtime.
        """
        n_households = self.env.households.n_households

        # Phase 1: Concurrently invoke each agent's LLM processing
        tasks = []
        for household_id in range(n_households):
            tasks.append(
                self.process_household_llm(
                    household_id,
                    step,
                    long_term_news,
                    similar_experience,
                    private_obs[household_id]
                )
            )
        llm_results = await asyncio.gather(*tasks)
        # Filter out any None results
        llm_results = [res for res in llm_results if res is not None]

        # Phase 2: Concurrent policy inference using run_in_executor for sync calls
        loop = asyncio.get_event_loop()
        tasks = []
        for result in llm_results:
            tasks.append(loop.run_in_executor(
                None,
                self.process_policy_inference,
                result,
                step,
                private_obs
            ))
        updated_results = await asyncio.gather(*tasks)

        # Phase 3: Concurrently evaluate each agent's final statement
        tasks = [self.process_evaluation(result, updated_results, private_obs) for result in updated_results]
        final_results = await asyncio.gather(*tasks)

        return final_results

    async def process_household_llm(self, household_id, step, long_term_news, similar_experience, private_obs_item):
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
                private_obs_item
            )

            # Extract returned fields
            personal_statements = llm_result.get("statements", [])
            personal_analysis   = llm_result.get("analysis", "")
            economic_status     = llm_result.get("economic_status", 1)
            personal_reasoning  = llm_result.get("reasoning", "")

            # Save last statement and reasoning for next round
            self.previous_statements[agent_id] = personal_statements[-1] if personal_statements else ""
            self.previous_reasoning[agent_id]  = personal_analysis + personal_reasoning

            return {
                "step": step,
                "agent_id": agent_id,
                "private_obs": private_obs_item.tolist() if hasattr(private_obs_item, "tolist") else private_obs_item,
                "statements": personal_statements,
                "reasoning": personal_analysis + personal_reasoning,
                "economic_status": economic_status,
                "evaluation": None
            }
        except Exception as e:
            logging.error(f"Error in dialogue generation for {agent_id}: {repr(e)}", exc_info=True)
            return None

    def process_policy_inference(self, result, step, private_obs):
        """
        Perform synchronous policy inference.
        Samples one statement based on the probability distribution
        and updates the result dict.
        """
        try:
            # Early check: if statements list is empty, skip policy inference
            if not result.get("statements") or len(result["statements"]) == 0:
                logging.warning(f"No statements available for {result['agent_id']}, skipping policy inference")
                result["probabilities"] = []
                result["chosen_statement"] = None
                return result
            
            evaluation = self.previous_evaluations.get(result["agent_id"], {})
            household_index = int(result['agent_id'].replace("Household", "")) - 1
            n_households = self.env.households.n_households

            # Extract features from private observations (tensor or list)
            personal_labor_productivity = private_obs[household_index][0]
            personal_wealth             = private_obs[household_index][1]

            # Prepare wealth guesses and trust levels
            wealth_guesses = evaluation.get("wealth_guesses", [1] * n_households)
            trust_levels   = evaluation.get("trust_levels",   [5.0] * n_households)
            if 0 <= household_index < len(trust_levels):
                trust_levels[household_index] = 10.0  # full trust in self (10/10)

            # Log statements information before policy inference
            logging.info(f"process_policy_inference for {result['agent_id']}: num_statements={len(result['statements'])}, step={step}")
            if result["statements"]:
                logging.info(f"process_policy_inference for {result['agent_id']}: statements={result['statements']}")
                logging.debug(f"process_policy_inference for {result['agent_id']}: reasoning (first 300 chars)={result.get('reasoning', '')[:300]}")
            else:
                logging.warning(f"process_policy_inference for {result['agent_id']}: statements list is empty")
            
            # Run the policy inference to get statement probabilities
            statement_probs, _ = run_policy_inference(
                player_id=household_index,
                economic_status=result.get("economic_status", 1),
                personal_labor_productivity=personal_labor_productivity,
                personal_wealth=personal_wealth,
                step=step,
                wealth_guesses=wealth_guesses,
                trust_levels=trust_levels,
                obs_text=f"Agent {result['reasoning']}",
                candidate_action_texts=result["statements"]
            )
            # Convert tensor to list if necessary
            if isinstance(statement_probs, torch.Tensor):
                statement_probs = statement_probs.tolist()

            result["probabilities"] = statement_probs
            # Choose a statement based on the computed probabilities
            if len(result["statements"]) == len(statement_probs) and statement_probs:
                chosen_index = random.choices(
                    range(len(statement_probs)),
                    weights=statement_probs,
                    k=1
                )[0]
                result["chosen_statement"] = result["statements"][chosen_index]
                # Update self.previous_statements with the actual chosen statement
                self.previous_statements[result["agent_id"]] = result["chosen_statement"]
            else:
                result["chosen_statement"] = None
                self.previous_statements[result["agent_id"]] = ""

            return result
        except Exception as e:
            logging.error(f"Error in probability generation for {result['agent_id']}: {repr(e)}", exc_info=True)
            result["probabilities"] = []
            result["chosen_statement"] = None
            return result

    async def process_evaluation(self, result, all_results, private_obs):
        """
        Asynchronously evaluate the agent's chosen statement.
        Calls the environment's evaluate_agents_statements method
        and updates the evaluation field.
        """
        agent_id = result["agent_id"]
        personal_statement = self.previous_statements.get(agent_id, "")
        personal_reasoning = self.previous_reasoning.get(agent_id, "")
        try:
            # Gather chosen statements of other agents
            other_agents_statements = [
                r["chosen_statement"]
                for r in all_results
                if r["agent_id"] != agent_id and r.get("chosen_statement")
            ]
            # Await the evaluation call
            household_index = int(agent_id.replace("Household", "")) - 1
            analysis_result = await self.env.households.evaluate_agents_statements(
                personal_statement=personal_statement,
                personal_reasoning=personal_reasoning,
                other_agents_statements=other_agents_statements,
                private_observation=private_obs[household_index]
            )
            result["evaluation"] = analysis_result
            # Save evaluation for the next round
            self.previous_evaluations[agent_id] = analysis_result
            return result
        except Exception as e:
            logging.error(f"Error in evaluation for {agent_id}: {repr(e)}", exc_info=True)
            result["evaluation"] = {}
            return result
