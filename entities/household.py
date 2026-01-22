from entities.base import BaseEntity  
import numpy as np  
import copy  
import pandas as pd  
import random  
import quantecon as qe  
import matplotlib.pyplot as plt  
import json  
import re  
import logging  
import gymnasium as gym
from gymnasium.spaces import Box  
from utils.llm_driver import async_llm_call
import os
from datetime import datetime  

class Household(BaseEntity):  
    """
    Household agent that simulates household behavior in an economic environment.
    Inherits common functionality from BaseEntity.
    """
    name = 'Household'

    def __init__(self, entity_args):  
        super().__init__()
        # Unique identifier for this household agent  
        self.agent_id = entity_args.get("agent_id", "anonymous")  
        # Total number of households in the simulation  
        self.n_households = entity_args['n']  
        # Risk aversion coefficient (CRRA)  
        self.CRRA = entity_args['CRRA']  
        # Intertemporal elasticity parameter  
        self.IFE = entity_args['IFE']  
        # Elasticity parameter for utility  
        self.eta = entity_args['eta']  
        # Probability of transitioning from normal to superstar state  
        self.e_p = entity_args['e_p']  
        # Probability of remaining or returning to superstar state  
        self.e_q = entity_args['e_q']  
        # Persistence coefficient for ability shock  
        self.rho_e = entity_args['rho_e']  
        # Volatility of ability shock  
        self.sigma_e = entity_args['sigma_e']  
        # Multiplier applied when in superstar state  
        self.super_e = entity_args['super_e']  
        # Dimension of the action space for each household  
        self.action_dim = entity_args['action_shape']  
        # Load real-world asset and education data  
        self.real_asset, self.real_e = self.get_real_data()  
        # Initialize households with sampled data  
        self.households_init()  
        # Reset dynamic state variables to initial values  
        self.reset()  

        # Define continuous action space across all households  
        self.action_space = Box(
            low=-1, high=1,
            shape=(self.n_households, self.action_dim),
            dtype=np.float32
        )  
        # Store reasoning history for each household  
        self.reasoning_history = {}  

    def e_initial(self, n):  
        """
        Initialize the ability state array for n households.
        Creates an n×2 array for normal and superstar ability levels.
        """
        self.e_array = np.zeros((n, 2))  
        random_set = np.random.rand(n)  
        # Set initial superstar ability based on probability e_p  
        self.e_array[:, 0] = (
            (random_set > self.e_p).astype(int) * self.e_init.flatten()
        )  
        # Indicator for normal state  
        self.e_array[:, 1] = (random_set < self.e_p).astype(int)  
        # Total ability is sum of both components  
        self.e = np.sum(self.e_array, axis=1, keepdims=True)  
        # Save original state for resets  
        self.e_0 = copy.copy(self.e)  
        self.e_array_0 = copy.copy(self.e_array)  

    def generate_e_ability(self):  
        """
        Simulate evolution of ability for each household.
        Transitions between normal and superstar states occur based on probabilities.
        """
        # Store previous state for computing dynamics  
        self.e_past = copy.copy(self.e_array)  
        # Mean of past superstar ability for scaling  
        e_past_mean = (
            np.sum(self.e_past[:, 0]) / np.count_nonzero(self.e_past[:, 0])
        )  
        for i in range(self.n_households):  
            is_superstar = int(self.e_array[i, 1] > 0)  
            if is_superstar == 0:  
                # Currently in normal state  
                if np.random.rand() < self.e_p:  
                    # Transition to superstar state  
                    self.e_array[i, 0] = 0  
                    self.e_array[i, 1] = self.super_e * e_past_mean  
                else:  
                    # Remain normal, AR(1) update  
                    self.e_array[i, 1] = 0  
                    self.e_array[i, 0] = np.exp(
                        self.rho_e * np.log(self.e_past[i, 0])
                        + self.sigma_e * np.random.randn()
                    )  
            else:  
                # Currently in superstar state  
                if np.random.rand() < self.e_q:  
                    # Persist in superstar state  
                    self.e_array[i, 0] = 0  
                    self.e_array[i, 1] = self.super_e * e_past_mean  
                else:  
                    # Revert to normal at random level  
                    self.e_array[i, 1] = 0  
                    self.e_array[i, 0] = random.uniform(
                        self.e_array[:, 0].min(), self.e_array[:, 0].max()
                    )  
        # Update total ability  
        self.e = np.sum(self.e_array, axis=1, keepdims=True)  

    def reset(self, **custom_cfg):  
        """
        Reset dynamic variables to their initial values and generate new abilities.
        """
        self.e = copy.copy(self.e_0)  
        self.e_array = copy.copy(self.e_array_0)  
        self.generate_e_ability()  
        # Reset asset levels to initial state  
        self.at = copy.copy(self.at_init)  
        self.at_next = copy.copy(self.at)  

    def households_init(self):  
        """
        Sample initial asset and education data for all households.
        Uses real-world data for initialization.
        """
        self.at_init, self.e_init = self.sample_real_data()  
        self.e_initial(self.n_households)  

    def lorenz_curve(self, wealths):  
        """
        Plot the Lorenz curve for a given array of wealth values.
        """
        f_vals, l_vals = qe.lorenz_curve(wealths)  
        fig, ax = plt.subplots()  
        ax.plot(f_vals, l_vals, label='Lorenz curve')  
        ax.plot(f_vals, f_vals, label='Line of equality')  
        ax.legend()  
        plt.show()  

    def get_real_data(self):  
        """
        Load real asset and education data from CSV file.
        Returns arrays of asset and education levels.
        """
        df = pd.read_csv('agents/data/advanced_scfp2022.csv')  
        return df['ASSET'].values, df['EDUC'].values  

    def sample_real_data(self):  
        """
        Randomly sample real-world data for each household without replacement.
        Returns asset and education arrays of shape (n_households, 1).
        """
        indices = np.random.choice(
            len(self.real_asset), self.n_households, replace=False
        )  
        return (
            self.real_asset[indices].reshape(self.n_households, 1),
            self.real_e[indices].reshape(self.n_households, 1)
        )  

    def close(self):  
        """
        Placeholder for any cleanup operations when environment closes.
        """
        pass  

    @staticmethod
    def extract_json_from_text(text: str) -> str:  
        """
        Extract JSON object from text, supporting code fences or inline braces.
        Returns the JSON string or None if not found.
        """
        match = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL
        )  
        if match:  
            return match.group(1)  
        start, end = text.find('{'), text.rfind('}')  
        return (text[start:end+1] if start != -1 and end != -1 else None)

    @staticmethod
    def _save_failed_attempts(function_name: str, failed_attempts: list, context: dict):
        """
        Save the results of failed attempts to a file for debugging purposes.
        
        Args:
            function_name: The name of the function that failed.
            failed_attempts: A list of failed attempts.
            context: Contextual information (input parameters, etc.).
        """
        if not failed_attempts:
            return
        
        # Create the save directory
        save_dir = "debug_failures"
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate a filename including a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_dir, f"{function_name}_{timestamp}.json")
        
        # Prepare the data to be saved
        save_data = {
            "function_name": function_name,
            "timestamp": datetime.now().isoformat(),
            "total_failed_attempts": len(failed_attempts),
            "context": context,
            "failed_attempts": failed_attempts
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Save failed results to file: {filename}")
        except Exception as e:
            logging.error(f"Error saving failed results: {str(e)}")  

    @staticmethod
    async def analyze_and_communicate(
        long_term_news, similar_experience, private_observation,
        temperature=0.7, max_retries=20
    ):  
        """
        Use LLM to analyze long-term news and private observation.
        Returns JSON with keys: analysis, economic_status, reasoning, statements.
        """
        # Format private observation for better readability
        private_obs_formatted = f"""
        • Personal productivity (e): {private_observation[0]:.4f}
        • Personal wealth: {private_observation[1]:.4f}"""
                
        similar_exp_text = similar_experience if similar_experience else "No similar experiences found."
                
        prompt = f"""You are a family decision inferent. Analyze the given data and provide insights.

        Long-Term News: {long_term_news}

        Private Observation:
        {private_obs_formatted}

        Similar Experiences: {similar_exp_text}

        Your final goal is to improve the self-utility of the current family, where increased labor time reduces utility and increased consumption improves utility, under the Bewley–Aiyagari model.

        **Tasks:**
        1. Summarize key economic insights in "analysis".
        2. Rate the economic condition as:
        • 0 = Bad
        • 1 = Neutral
        • 2 = Good
        Store this as "economic_status".
        3. Based on the current situation and private observation, give suggestions in "reasoning".
        4. Generate 3 unique public statements in "statements".

        **Output Format:**
        Return exactly this JSON (no extra keys or commentary):
        {{
        "analysis": "...",
        "economic_status": 0,
        "reasoning": "...",
        "statements": ["statement1", "statement2", "statement3"]
        }}"""
        required_keys = {"analysis", "economic_status", "reasoning", "statements"}
        failed_attempts = []
        
        for attempt in range(max_retries):
            llm_output = await async_llm_call(prompt, temperature=temperature)
            failure_info = {
                "attempt": attempt + 1,
                "llm_output": llm_output,
                "error": None,
                "parsed_result": None
            }
            
            try:
                result = json.loads(llm_output)
                failure_info["parsed_result"] = result
            except json.JSONDecodeError as e:
                failure_info["error"] = f"JSONDecodeError: {str(e)}"
                extracted = Household.extract_json_from_text(llm_output)
                if extracted:
                    try:
                        result = json.loads(extracted)
                        failure_info["parsed_result"] = result
                        failure_info["extracted_json"] = extracted
                    except Exception as e2:
                        failure_info["error"] = f"JSONDecodeError: {str(e)}, ExtractionError: {str(e2)}"
                        failed_attempts.append(failure_info)
                        continue
                else:
                    failure_info["error"] = f"JSONDecodeError: {str(e)}, No JSON found in text"
                    failed_attempts.append(failure_info)
                    continue
            
            # Check if the result meets the requirements
            if set(result.keys()) == required_keys and isinstance(result.get('statements'), list) and len(result.get('statements', [])) == 3:
                return result
            else:
                # Record the reasons for non-compliance
                missing_keys = required_keys - set(result.keys())
                extra_keys = set(result.keys()) - required_keys
                statements_info = {
                    "is_list": isinstance(result.get('statements'), list),
                    "length": len(result.get('statements', [])) if isinstance(result.get('statements'), list) else "N/A"
                }
                failure_info["error"] = f"Validation failed: missing_keys={missing_keys}, extra_keys={extra_keys}, statements_info={statements_info}"
                failed_attempts.append(failure_info)
        
        # Save failed results to file
        Household._save_failed_attempts("analyze_and_communicate", failed_attempts, {
            "long_term_news": long_term_news,
            "similar_experience": similar_experience,
            "private_observation": str(private_observation),
            "temperature": temperature,
            "max_retries": max_retries,
            "prompt": prompt
        })
        
        logging.error("Exceeded max retries for analyze_and_communicate")
        return {"analysis":"","economic_status":1,"reasoning":"","statements":[]}

    @staticmethod
    async def derive_reasoning_text(
        short_term_news, recent_long_term_result,
        private_observation, temperature, max_retries
    ):  
        """
        Use LLM to derive reasoning based on short-term news and recent results.
        Returns JSON with keys: economic_status and reasoning.
        """
        # Format private observation for better readability
        private_obs_formatted = f"""
  • Personal productivity (e): {private_observation[0]:.4f}
  • Personal wealth: {private_observation[1]:.4f}"""
        
        recent_long_term_text = recent_long_term_result if recent_long_term_result else "None"
        
        prompt = f"""You are a family decision inferent. Your goal is to improve the family's self-utility under the Bewley–Aiyagari model (more labor ↓ utility, more consumption ↑ utility).

        **Inputs:**
        • Short-Term News: {short_term_news}
        • Recent Long-Term News: {recent_long_term_text}
        • Private Observation:
        {private_obs_formatted}

        **Tasks:**
        1. Provide a detailed analysis of current economic conditions, considering savings rate and working hours.
        2. Rate the economic condition:
        • 0 = Bad
        • 1 = Neutral
        • 2 = Good

        **Output Format:**
        Return exactly this JSON (no extra keys or commentary):
        {{
        "economic_status": 0,
        "reasoning": "..."
        }}"""
        required_keys = {"economic_status", "reasoning"}
        for attempt in range(max_retries):
            llm_output = await async_llm_call(prompt, temperature=temperature)
            try:
                result = json.loads(llm_output)
            except json.JSONDecodeError:
                extracted = Household.extract_json_from_text(llm_output)
                if extracted:
                    try:
                        result = json.loads(extracted)
                    except Exception:
                        continue
                else:
                    continue
            if set(result.keys()) == required_keys:
                return result
        logging.error("Exceeded max retries for derive_reasoning_text")
        return {"economic_status":1,"reasoning":""}

    async def evaluate_agents_statements(
        self, personal_statement, personal_reasoning,
        other_agents_statements, private_observation,
        temp=0.7, max_retries=15
    ):  
        """
        Evaluate statements from other households and generate private feedback.
        Returns JSON with keys: wealth_guesses, trust_levels, reflection_text.
        """
        expected_num = self.n_households
        # Format private observation for better readability
        private_obs_formatted = f"""
  • Personal productivity (e): {private_observation[0]:.4f}
  • Personal wealth: {private_observation[1]:.4f}"""
        
        # Format other agents' statements
        if isinstance(other_agents_statements, list) and len(other_agents_statements) > 0:
            other_statements_formatted = "\n".join([f"- {stmt}" for stmt in other_agents_statements])
        else:
            other_statements_formatted = "No other statements available."
        
        prompt = f"""You are a family decision inferent. Analyze the given other households' statements and provide private insights.

        Private Observation:
        {private_obs_formatted}

        Internal Reasoning: {personal_reasoning}

        Public Personal Statement: {personal_statement}

        Other Households' Statements:
        {other_statements_formatted}

        Your final goal is to improve the self-utility of the current family, where increased labor time reduces utility and increased consumption improves utility, under the Bewley–Aiyagari model.

        **Tasks:**
        1. Classify each household's wealth level as "wealth_guesses" (0=Low, 1=Medium, 2=High) with exactly {expected_num} elements. You must provide a guess for ALL {expected_num} households, including yourself. Base your guesses on the statements provided and your understanding of the economic context.
        2. Rate each statement's trustworthiness from 0 (not trustworthy) to 10 (highly trustworthy) as "trust_levels" with exactly {expected_num} elements. You must provide a trust level for ALL {expected_num} households, including yourself (you should trust yourself highly).
        3. Provide a brief reflection in "reflection_text", focusing on yourself, others' statements, and ensuing economic decisions.

        **Output Format:**
        Return exactly this JSON (no extra keys or commentary):
        {{
        "wealth_guesses": [0, 1, 2, ...],
        "trust_levels": [5, 7, 3, ...],
        "reflection_text": "..."
        }}

        Note: Both "wealth_guesses" and "trust_levels" must be lists of exactly {expected_num} integers."""
        required_keys = {"wealth_guesses","trust_levels","reflection_text"}
        for attempt in range(max_retries):
            response = await async_llm_call(prompt, temperature=temp)
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                extracted = Household.extract_json_from_text(response)
                if extracted:
                    try:
                        result = json.loads(extracted)
                    except Exception:
                        continue
                else:
                    continue
            if set(result.keys()) != required_keys:
                continue
            # Ensure correct lengths
            wg = result['wealth_guesses']
            tl = result['trust_levels']
            if len(wg)!=expected_num or len(tl)!=expected_num:
                continue
            return result
        logging.error("Exceeded max retries for evaluate_agents_statements")
        return {
            "wealth_guesses":[1]*expected_num,
            "trust_levels":[5]*expected_num,
            "reflection_text":""
        }
