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
    content: str
    timestamp: datetime
    metrics: Dict[str, float]
    changes: Optional[Dict[str, Union[float, str]]]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the NewsMemory instance to a JSON-serializable dictionary.
        """
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "changes": self.changes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NewsMemory":
        """
        Construct a NewsMemory instance from a dictionary, parsing timestamp.
        """
        return cls(
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metrics=data["metrics"],
            changes=data.get("changes"),
        )


class IntelligenceBureauAgent:
    """
    Agent responsible for gathering economic observations, computing changes,
    and generating short- or long-term news summaries via an LLM.
    """
    METRICS = [
        "Top 10% Wealth", "Top 10% Income", "Top 10% Productivity",
        "Bottom 50% Wealth", "Bottom 50% Income", "Bottom 50% Productivity",
        "Wage Rate"
    ]

    def __init__(
        self,
        llm_model: Any,
        data_sources: List[str],
        memory_dir: str = "./llm_results",
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
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger("IntelligenceBureau")

        self.llm_model = llm_model
        self.data_sources = data_sources
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.memory_capacity = {
            "short_term": short_term_capacity,
            "long_term": long_term_capacity
        }

        self.observation_log: List[List[float]] = []
        self.short_term_memory: List[NewsMemory] = []
        self.long_term_memory: List[NewsMemory] = []
        self.last_observation: Optional[List[float]] = None
        self.last_long_term_news_index: Optional[int] = None

        self.stats = {
            "total_news_generated": 0,
            "total_observations": 0,
            "last_update": None
        }

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
            return {
                metric: float(ch)
                for metric, ch, ok in zip(self.METRICS, changes, valid)
                if ok
            }
        except Exception:
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

            if not all(isinstance(row, list) for row in observations):
                self.logger.error("Invalid observation format: expected list of lists.")
                return False

            for row in observations:
                float_row = [float(x) for x in row]
                if not self.validate_observation(float_row):
                    self.logger.error(f"Invalid observation: {float_row}")
                    return False
                self.observation_log.append(float_row)

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

    def _get_long_term_window(self) -> Optional[np.ndarray]:
        """
        Return the observation window from the previous long-term checkpoint
        to the current step (inclusive).
        """
        if len(self.observation_log) < 2:
            return None

        if self.last_long_term_news_index is None:
            start_idx = 0
        else:
            start_idx = self.last_long_term_news_index

        window = np.asarray(self.observation_log[start_idx:], dtype=float)
        if len(window) < 2:
            return None
        return window

    def _create_prompt(
        self,
        recent_only: bool = True,
        changes: Optional[Dict[str, Any]] = None,
        recent_long_term_news: Optional[str] = None
    ) -> str:
        """
        Build a prompt string for the LLM based on recent or multi-period data.
        Includes metric values and optionally formatted changes.
        """
        if recent_only:
            recent_obs = [self.observation_log[-1]]
        else:
            window = self._get_long_term_window()
            if window is None:
                recent_obs = [self.observation_log[-1]]
            else:
                if len(window) >= 5:
                    mid_idx = len(window) // 2
                    recent_obs = [window[0].tolist(), window[mid_idx].tolist(), window[-1].tolist()]
                elif len(window) >= 2:
                    recent_obs = [window[0].tolist(), window[-1].tolist()]
                else:
                    recent_obs = [window[-1].tolist()]

        prompt = (
            "You are a professional economic journalist. "
            "Write a concise financial news report (≤300 words) highlighting key trends."
        )

        if changes:
            prompt += "\nRecent Trends:\n"
            for metric, pct in changes.items():
                if isinstance(pct, (int, float)):
                    direction = "increase" if pct > 0 else "decrease"
                    prompt += f"- {metric}: {abs(pct):.2f}% {direction}\n"
                else:
                    prompt += f"- {metric}: {pct}\n"
            prompt += "\n"

        if recent_only and recent_long_term_news:
            prompt += "\nMost Recent Long-Term Context:\n"
            prompt += recent_long_term_news.strip() + "\n"

        prompt += "Economic Metrics:\n"
        for obs in recent_obs:
            prompt += (
                f"Top10% Wealth: {obs[0]:,.2f}, Income: {obs[1]:,.2f}, Productivity: {obs[2]:.2f}\n"
                f"Bot50% Wealth: {obs[3]:,.2f}, Income: {obs[4]:,.2f}, Productivity: {obs[5]:.2f}, Wage: {obs[6]:.2f}\n"
            )

        if recent_only:
            prompt += (
                "\nGuidelines:\n1. Craft an engaging headline.\n"
                "2. Start with a strong lead.\n3. Highlight immediate implications.\n4. Conclude with short-term outlook."
            )
        else:
            prompt += (
                "\nGuidelines:\n1. Summarize the structural trends across the checkpoint interval.\n"
                "2. Focus on persistent changes rather than one-step noise.\n"
                "3. Provide an executive summary and discuss long-term implications.\n"
                "4. Suggest future developments."
            )
        return prompt

    def _store_news_memory(
        self,
        news_content: str,
        metrics: Dict[str, float],
        changes: Optional[Dict[str, Union[float, str]]],
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

        cap = self.memory_capacity["short_term" if is_short_term else "long_term"]
        if len(mem_list) > cap:
            mem_list.pop(0)

        self.stats["total_news_generated"] += 1
        self.save_memories()

    def _is_error_response_text(self, text: Any) -> bool:
        if not isinstance(text, str):
            return False
        stripped = text.strip()
        if stripped.startswith("Error generating news:"):
            return True
        if not stripped:
            return False
        try:
            parsed = json.loads(stripped)
            return isinstance(parsed, dict) and "error" in parsed
        except Exception:
            return False

    def _fallback_news_text(self, recent_only: bool, changes: Optional[Dict[str, Any]], reason: str = "") -> str:
        horizon = "short-term" if recent_only else "long-term"
        if not changes:
            base = f"Fallback {horizon} news: no reliable LLM output was available. Economic signals look mixed, so households should stay cautious and adjust labor and savings gradually."
        else:
            highlights = []
            for metric, val in list(changes.items())[:4]:
                highlights.append(f"{metric}: {val}")
            highlights_text = "; ".join(highlights)
            base = f"Fallback {horizon} news: LLM output was unavailable. Key observed changes -> {highlights_text}. The outlook is uncertain, so gradual and defensive adjustments are preferred."
        if reason:
            base += f" [fallback_reason={reason}]"
        return base

    def generate_news(self, recent_only: bool = True, recent_long_term_news: Optional[str] = None) -> str:
        """
        Generate either short- or long-term news based on observations.
        Returns the generated news text, or error message.
        """
        if not self.observation_log:
            return "No observations available"

        current = self.observation_log[-1]
        try:
            changes = (
                self.compute_changes(current, self.last_observation) if recent_only
                else self._compute_long_term_changes()
            )
            self.last_observation = current
            prompt = self._create_prompt(
                recent_only=recent_only,
                changes=changes,
                recent_long_term_news=recent_long_term_news
            )
            news_text = self.llm_model(
                prompt,
                max_tokens=512,
                max_retries=2,
            )
            if self._is_error_response_text(news_text):
                self.logger.warning("LLM returned error-like news output; using fallback news text.")
                news_text = self._fallback_news_text(recent_only=recent_only, changes=changes, reason="llm_error_response")
            metrics_dict = self._create_metrics_dict(current)
            self._store_news_memory(news_text, metrics_dict, changes, recent_only)
            return news_text
        except Exception as e:
            self.logger.error("Error generating news", exc_info=True)
            return self._fallback_news_text(recent_only=recent_only, changes=changes if 'changes' in locals() else None, reason=type(e).__name__)

    def _compute_long_term_changes(self) -> Optional[Dict[str, str]]:
        """
        Compute long-horizon trend summaries over the whole checkpoint interval,
        but keep the output compact so the prompt does not get too long.
        """
        window = self._get_long_term_window()
        if window is None:
            return None

        start = window[0]
        end = window[-1]
        mean = window.mean(axis=0)
        std = window.std(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            total_pct = np.where(np.abs(start) > 1e-10, (end - start) / np.abs(start) * 100.0, 0.0)

        changes = {}
        for idx, metric in enumerate(self.METRICS):
            if abs(total_pct[idx]) > 1e-3 or std[idx] > 1e-6:
                direction = "up" if total_pct[idx] > 0 else "down" if total_pct[idx] < 0 else "flat"
                changes[metric] = (
                    f"start={start[idx]:.2f}, end={end[idx]:.2f}, "
                    f"total={total_pct[idx]:+.2f}%, mean={mean[idx]:.2f}, "
                    f"volatility={std[idx]:.2f}, trend={direction}"
                )

        self.last_long_term_news_index = len(self.observation_log) - 1
        return changes or None

    def _normalize_for_key(self, obj: Any) -> Any:
        """
        Recursively convert objects into stable JSON-serializable values
        so memory entries can be deduplicated safely.
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, datetime):
            return obj.isoformat()

        if isinstance(obj, np.generic):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, dict):
            return {
                str(k): self._normalize_for_key(v)
                for k, v in sorted(obj.items(), key=lambda x: str(x[0]))
            }

        if isinstance(obj, (list, tuple, set)):
            return [self._normalize_for_key(v) for v in obj]

        if isinstance(obj, NewsMemory):
            return self._normalize_for_key(obj.to_dict())

        return str(obj)

    def _memory_key(self, memory: NewsMemory) -> str:
        """
        Build a stable string key for de-duplication.
        This avoids 'unhashable type: dict' errors when metrics/changes
        contain nested dict/list structures.
        """
        normalized = self._normalize_for_key(memory)
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)

    def _deduplicate_memories(self, memories: List[NewsMemory]) -> List[NewsMemory]:
        deduped = []
        seen = set()

        for memory in memories:
            try:
                key = self._memory_key(memory)
            except Exception as e:
                self.logger.warning(f"Failed to build memory key, keeping item as-is: {e}")
                deduped.append(memory)
                continue

            if key in seen:
                continue

            seen.add(key)
            deduped.append(memory)

        return deduped

    def save_memories(self) -> None:
        """
        Save short-term and long-term memories, observation log, and stats to disk.
        Overwrite on-disk state with current in-memory state to avoid duplication.
        """
        try:
            self.short_term_memory = self._deduplicate_memories(self.short_term_memory)
            self.long_term_memory = self._deduplicate_memories(self.long_term_memory)

            self.short_term_memory = self.short_term_memory[-self.memory_capacity["short_term"]:]
            self.long_term_memory = self.long_term_memory[-self.memory_capacity["long_term"]:]

            short_path = self.memory_dir / "short_term_memory.json"
            long_path = self.memory_dir / "long_term_memory.json"
            obs_path = self.memory_dir / "observation_log.pkl"
            stats_path = self.memory_dir / "stats.json"

            with open(short_path, "w", encoding="utf-8") as f:
                json.dump([m.to_dict() for m in self.short_term_memory], f, indent=2, ensure_ascii=False)

            with open(long_path, "w", encoding="utf-8") as f:
                json.dump([m.to_dict() for m in self.long_term_memory], f, indent=2, ensure_ascii=False)

            with open(obs_path, "wb") as f:
                pickle.dump(self.observation_log, f)

            stats_copy = deepcopy(self.stats)
            stats_copy["last_long_term_news_index"] = self.last_long_term_news_index
            stats_copy["last_update"] = (
                stats_copy["last_update"].isoformat()
                if stats_copy["last_update"] else None
            )
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats_copy, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Memories saved to {self.memory_dir}")

        except Exception as e:
            self.logger.error("Error saving memories", exc_info=True)

    def load_memories(self) -> None:
        """
        Load any existing memories, observations, and stats from disk.
        Initializes empty lists on failure.
        """
        try:
            short_path = self.memory_dir / "short_term_memory.json"
            if short_path.exists():
                with open(short_path, "r", encoding="utf-8") as f:
                    loaded_short = [NewsMemory.from_dict(item) for item in json.load(f)]
                self.short_term_memory = self._deduplicate_memories(loaded_short)
                self.short_term_memory = self.short_term_memory[-self.memory_capacity["short_term"]:]
            else:
                self.short_term_memory = []

            long_path = self.memory_dir / "long_term_memory.json"
            if long_path.exists():
                with open(long_path, "r", encoding="utf-8") as f:
                    loaded_long = [NewsMemory.from_dict(item) for item in json.load(f)]
                self.long_term_memory = self._deduplicate_memories(loaded_long)
                self.long_term_memory = self.long_term_memory[-self.memory_capacity["long_term"]:]
            else:
                self.long_term_memory = []

            obs_path = self.memory_dir / "observation_log.pkl"
            if obs_path.exists():
                with open(obs_path, "rb") as f:
                    self.observation_log = pickle.load(f)
            else:
                self.observation_log = []

            stats_path = self.memory_dir / "stats.json"
            if stats_path.exists():
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                stats["last_update"] = (
                    datetime.fromisoformat(stats["last_update"])
                    if stats.get("last_update") else None
                )
                self.stats = stats
                self.last_long_term_news_index = self.stats.get("last_long_term_news_index")
            else:
                self.stats = {
                    "total_news_generated": 0,
                    "total_observations": 0,
                    "last_update": None
                }
                self.last_long_term_news_index = None

            self.logger.info(f"Loaded memories from {self.memory_dir}")

        except Exception:
            self.logger.error("Error loading memories", exc_info=True)
            self.observation_log = []
            self.short_term_memory = []
            self.long_term_memory = []
            self.stats = {
                "total_news_generated": 0,
                "total_observations": 0,
                "last_update": None
            }
            self.last_long_term_news_index = None

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
            summary += f"- {metric}: Average {sum(vals) / len(vals):.2f}\n"
        return summary

    def delete_old_memories(self, threshold_days: int = 365) -> None:
        """
        Remove memories older than threshold_days from both memory lists.
        """
        cutoff = datetime.now() - timedelta(days=threshold_days)
        self.short_term_memory = [m for m in self.short_term_memory if m.timestamp >= cutoff]
        self.long_term_memory = [m for m in self.long_term_memory if m.timestamp >= cutoff]
        self.logger.info(f"Deleted memories older than {threshold_days} days")