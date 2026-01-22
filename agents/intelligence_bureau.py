from utils.llm_driver import LLMCall  
import numpy as np  
import json  
import pickle  
import os  
from typing import List, Dict, Union, Optional, Any  
from dataclasses import dataclass  
from datetime import datetime, timedelta  
from pathlib import Path  
import logging  
from copy import deepcopy  

@dataclass
class NewsMemory:
    """
    Data structure for storing a news item with timestamp, metric values, and computed changes.
    """
    content: str                           # Full text of the news report
    timestamp: datetime                   # Time when the news was generated
    metrics: Dict[str, float]             # Current values of tracked economic metrics
    changes: Optional[Dict[str, float]]   # Percent changes relative to previous observation

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the NewsMemory instance to a JSON-serializable dictionary.
        """
        return {
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'changes': self.changes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsMemory':
        """
        Construct a NewsMemory instance from a dictionary, parsing timestamp.
        """
        return cls(
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metrics=data['metrics'],
            changes=data.get('changes')
        )

class IntelligenceBureauAgent:
    """
    Agent responsible for gathering economic observations, computing changes,
    and generating short- or long-term news summaries via an LLM.
    """
    # List of metrics tracked by this agent
    METRICS = [
        "Top 10% Wealth", "Top 10% Income", "Top 10% Productivity",
        "Bottom 50% Wealth", "Bottom 50% Income", "Bottom 50% Productivity",
        "Wage Rate"
    ]

    def __init__(
        self,
        llm_model: Any,
        data_sources: List[str],
        memory_dir: str = './llm_results',
        short_term_capacity: int = 10,
        long_term_capacity: int = 20
    ):
        """
        Initialize the Intelligence Bureau agent.
        - llm_model: callable for generating text from prompts
        - data_sources: names of data feeds (for bookkeeping)
        - memory_dir: directory to persist memory files
        - short_term_capacity / long_term_capacity: memory size limits
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('IntelligenceBureau')

        self.llm_model = llm_model
        self.data_sources = data_sources
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Capacity limits for short- and long-term news memories
        self.memory_capacity = {
            "short_term": short_term_capacity,
            "long_term": long_term_capacity
        }

        # Logs and memory lists
        self.observation_log: List[List[float]] = []
        self.short_term_memory: List[NewsMemory] = []
        self.long_term_memory: List[NewsMemory] = []
        self.last_observation: Optional[List[float]] = None
        self.last_long_term_news_index: Optional[int] = None

        # Statistics tracking
        self.stats = {
            "total_news_generated": 0,
            "total_observations": 0,
            "last_update": None
        }

        # Load any previously saved memories from disk
        self.load_memories()

    def reset(self) -> None:
        """
        Reset internal state (except for stored memories), update timestamp.
        """
        self.last_observation = None
        self.stats["last_update"] = datetime.now()
        self.logger.info("Agent state reset")

    def calculate_changes(self, current_obs: List[float]) -> Optional[Dict[str, float]]:
        """
        Compute percent changes from last observation. Returns None on first call.
        """
        if self.last_observation is None:
            self.last_observation = current_obs
            return None
        return self.compute_changes(current_obs, self.last_observation)

    def compute_changes(
        self,
        current_obs: List[float],
        baseline: Optional[List[float]]
    ) -> Optional[Dict[str, float]]:
        """
        Vectorized percent change calculation using NumPy, skipping near-zero baselines.
        """
        if baseline is None:
            return None
        try:
            curr = np.array(current_obs, dtype=float)
            prev = np.array(baseline, dtype=float)
            valid = np.abs(prev) > 1e-10
            changes = np.zeros_like(curr)
            changes[valid] = ((curr[valid] - prev[valid]) / np.abs(prev[valid])) * 100
            # Zip metrics with computed changes only where valid
            return {
                metric: float(ch)
                for metric, ch, ok in zip(self.METRICS, changes, valid)
                if ok
            }
        except Exception as e:
            self.logger.error("Error computing changes", exc_info=True)
            return None

    def validate_observation(self, observation: List[float]) -> bool:
        """
        Check that the observation is a list/ndarray of the correct length and types.
        """
        if not isinstance(observation, (list, np.ndarray)):
            return False
        if len(observation) != len(self.METRICS):
            return False
        return all(isinstance(x, (int, float)) for x in observation)

    def collect_observations(
        self,
        observations: Union[np.ndarray, List[float], List[List[float]]]
    ) -> bool:
        """
        Accepts a single observation or batch (1D/2D array or list),
        validates and appends to log, updates stats.
        """
        try:
            # Normalize input to a list of lists
            if isinstance(observations, np.ndarray):
                if observations.ndim == 1:
                    observations = [observations.tolist()]
                elif observations.ndim == 2:
                    observations = observations.tolist()
                else:
                    self.logger.error("Invalid ndarray shape: ndim > 2 not supported.")
                    return False
            elif isinstance(observations, list) and observations and isinstance(observations[0], (int, float)):
                observations = [observations]

            # Ensure it's a list of lists
            if not all(isinstance(row, list) for row in observations):
                self.logger.error("Invalid observation format: expected list of lists.")
                return False

            # Validate each row
            for row in observations:
                float_row = [float(x) for x in row]
                if not self.validate_observation(float_row):
                    self.logger.error(f"Invalid observation: {float_row}")
                    return False
                self.observation_log.append(float_row)

            # Update stats and timestamp
            self.stats["total_observations"] += len(observations)
            self.stats["last_update"] = datetime.now()
            return True
        except Exception as e:
            self.logger.error("Error collecting observations", exc_info=True)
            return False

    def _create_metrics_dict(self, observation: List[float]) -> Dict[str, float]:
        """
        Map metric names to values for the given observation row.
        """
        return dict(zip(self.METRICS, observation))

    def _create_prompt(
        self,
        recent_only: bool = True,
        changes: Optional[Dict[str, Any]] = None,
        start_index: Optional[int] = None
    ) -> str:
        """
        Build a prompt string for the LLM based on recent or multi-period data.
        Includes metric values and optionally formatted changes.
        """
        # Select representative observations for the period
        if recent_only:
            recent_obs = [self.observation_log[-1]]
        else:
            # For long-term news, sample observations from the start of the period to provide context
            if start_index is not None and 0 <= start_index < len(self.observation_log):
                full_period_obs = self.observation_log[start_index:]
                # Sample up to 10 observations to represent the trend without overwhelming the LLM
                if len(full_period_obs) > 10:
                    indices = np.linspace(0, len(full_period_obs) - 1, 10, dtype=int)
                    recent_obs = [full_period_obs[i] for i in indices]
                else:
                    recent_obs = full_period_obs
            else:
                # Fallback to last 10 if start_index is not provided or invalid
                recent_obs = self.observation_log[-10:]

        prompt = (
            "You are a professional economic journalist. "
            "Write a concise financial news report (≤300 words) highlighting key trends."
        )
        # Include change information if provided
        if changes:
            prompt += "\nRecent Trends:\n"
            for metric, pct in changes.items():
                # Handle both float (from compute_changes) and str (from _compute_long_term_changes)
                if isinstance(pct, (int, float)):
                    direction = "increase" if pct > 0 else "decrease"
                    prompt += f"- {metric}: {abs(pct):.2f}% {direction}\n"
                else:
                    # pct is already a formatted string from _compute_long_term_changes
                    prompt += f"- {metric}: {pct}\n"
            prompt += "\n"

        # Append metric values
        prompt += "Economic Metrics:\n"
        for obs in recent_obs:
            prompt += (
                f"Top10% Wealth: {obs[0]:,.2f}, Income: {obs[1]:,.2f}, Productivity: {obs[2]:.2f}\n"
                f"Bot50% Wealth: {obs[3]:,.2f}, Income: {obs[4]:,.2f}, Productivity: {obs[5]:.2f}, Wage: {obs[6]:.2f}\n"
            )

        # Add writing guidelines
        if recent_only:
            prompt += (
                "\nGuidelines:\n1. Craft an engaging headline.\n"
                "2. Start with a strong lead.\n3. Highlight immediate implications.\n4. Conclude with short-term outlook."
            )
        else:
            prompt += (
                "\nGuidelines:\n1. Summarize 5-period trends.\n"
                "2. Provide executive summary.\n3. Discuss long-term impacts.\n4. Suggest future developments."
            )
        return prompt

    def _store_news_memory(
        self,
        news_content: str,
        metrics: Dict[str, float],
        changes: Optional[Dict[str, float]],
        is_short_term: bool
    ) -> None:
        """
        Persist a generated news report into short- or long-term memory.
        Enforce capacity limits and update stats.
        """
        news_mem = NewsMemory(
            content=news_content,
            timestamp=datetime.now(),
            metrics=metrics,
            changes=changes
        )
        mem_list = self.short_term_memory if is_short_term else self.long_term_memory
        mem_list.append(news_mem)
        # Trim excess
        cap = self.memory_capacity["short_term" if is_short_term else "long_term"]
        if len(mem_list) > cap:
            mem_list.pop(0)
        self.stats["total_news_generated"] += 1
        self.save_memories()

    def generate_news(self, recent_only: bool = True) -> str:
        """
        Generate either short- or long-term news based on observations.
        Returns the generated news text, or error message.
        """
        if not self.observation_log:
            return "No observations available"

        current = self.observation_log[-1]
        try:
            # Capture the start index for long-term prompt generation before it potentially updates
            start_index = self.last_long_term_news_index if not recent_only else None

            changes = (
                self.compute_changes(current, self.last_observation) if recent_only
                else self._compute_long_term_changes()
            )
            self.last_observation = current
            prompt = self._create_prompt(recent_only, changes, start_index=start_index)
            news_text = self.llm_model(prompt)
            metrics_dict = self._create_metrics_dict(current)
            self._store_news_memory(news_text, metrics_dict, changes, recent_only)
            return news_text
        except Exception as e:
            self.logger.error("Error generating news", exc_info=True)
            return f"Error generating news: {e}"

    def _compute_long_term_changes(self) -> Optional[Dict[str, str]]:
        """
        Calculate multi-period percent changes and format them as strings.
        """
        if self.last_long_term_news_index is None or len(self.observation_log) < 2:
            self.last_long_term_news_index = len(self.observation_log) - 1
            return None
        data = np.array(self.observation_log[self.last_long_term_news_index:], dtype=float)
        diffs = np.diff(data, axis=0)
        base = data[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = np.where(np.abs(base) > 1e-10, (diffs/base)*100, 0)
        changes = {}
        for idx, metric in enumerate(self.METRICS):
            seq = [f"{p:.2f}%" for p in pct[:, idx] if abs(p) > 1e-3]
            if seq:
                changes[metric] = " -> ".join(seq)
        self.last_long_term_news_index = len(self.observation_log) - 1
        return changes or None

    def save_memories(self) -> None:
        """
        Save short-term and long-term memories, observation log, and stats to disk.
        """
        try:
            # Short-term
            short_path = self.memory_dir / 'short_term_memory.json'
            json.dump([m.to_dict() for m in self.short_term_memory], open(short_path, 'w', encoding='utf-8'), indent=2)
            # Long-term
            long_path = self.memory_dir / 'long_term_memory.json'
            json.dump([m.to_dict() for m in self.long_term_memory], open(long_path, 'w', encoding='utf-8'), indent=2)
            # Observations
            pickle.dump(self.observation_log, open(self.memory_dir/'observation_log.pkl', 'wb'))
            # Stats
            stats_copy = deepcopy(self.stats)
            stats_copy['last_update'] = stats_copy['last_update'].isoformat() if stats_copy['last_update'] else None
            json.dump(stats_copy, open(self.memory_dir/'stats.json', 'w', encoding='utf-8'), indent=2)
            self.logger.info(f"Memories saved to {self.memory_dir}")
        except Exception as e:
            self.logger.error("Error saving memories", exc_info=True)

    def load_memories(self) -> None:
        """
        Load any existing memories, observations, and stats from disk.
        Initializes empty lists on failure.
        """
        try:
            # Load short-term
            short_path = self.memory_dir / 'short_term_memory.json'
            if short_path.exists():
                self.short_term_memory = [NewsMemory.from_dict(item) for item in json.load(open(short_path))]
            # Load long-term
            long_path = self.memory_dir / 'long_term_memory.json'
            if long_path.exists():
                self.long_term_memory = [NewsMemory.from_dict(item) for item in json.load(open(long_path))]
            # Load observations
            obs_path = self.memory_dir / 'observation_log.pkl'
            if obs_path.exists():
                self.observation_log = pickle.load(open(obs_path, 'rb'))
            # Load stats
            stats_path = self.memory_dir / 'stats.json'
            if stats_path.exists():
                stats = json.load(open(stats_path))
                stats['last_update'] = datetime.fromisoformat(stats['last_update']) if stats.get('last_update') else None
                self.stats = stats
            self.logger.info(f"Loaded memories from {self.memory_dir}")
        except Exception:
            self.logger.error("Error loading memories", exc_info=True)
            self.observation_log = []
            self.short_term_memory = []
            self.long_term_memory = []
            self.stats = {"total_news_generated":0,"total_observations":0,"last_update":None}

    def get_recent_news(
        self,
        count: int = 5,
        is_short_term: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Return the most recent 'count' news items from short- or long-term memory.
        """
        mem = self.short_term_memory if is_short_term else self.long_term_memory
        return [m.to_dict() for m in mem[-count:]]

    def merge_memories(self) -> List[Dict[str, Any]]:
        """
        Merge and sort short- and long-term memories by timestamp (newest first).
        """
        combined = self.short_term_memory + self.long_term_memory
        combined.sort(key=lambda x: x.timestamp, reverse=True)
        return [m.to_dict() for m in combined]

    def summarize_long_term_insights(self) -> str:
        """
        Compute average values for each metric across long-term memory.
        Returns a multiline summary string.
        """
        if not self.long_term_memory:
            return "No long-term insights available."
        summary = "Long-term Economic Insights:\n"
        agg: Dict[str, List[float]] = {}
        for mem in self.long_term_memory:
            for metric, val in mem.metrics.items():
                agg.setdefault(metric, []).append(val)
        for metric, vals in agg.items():
            summary += f"- {metric}: Average {sum(vals)/len(vals):.2f}\n"
        return summary

    def delete_old_memories(self, threshold_days: int = 365) -> None:
        """
        Remove memories older than threshold_days from both memory lists.
        """
        cutoff = datetime.now() - timedelta(days=threshold_days)
        self.short_term_memory = [m for m in self.short_term_memory if m.timestamp >= cutoff]
        self.long_term_memory  = [m for m in self.long_term_memory if m.timestamp >= cutoff]
        self.logger.info(f"Deleted memories older than {threshold_days} days")