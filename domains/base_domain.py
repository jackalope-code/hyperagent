"""Abstract base class for HyperAgents evaluation domains.

A Domain defines:
  - A distribution of task samples
  - A scoring function for predictions
  - An evaluate() method that runs an agent and returns aggregate results
"""

from abc import ABC, abstractmethod


class Domain(ABC):
    """Base class for all evaluation domains."""

    name: str = "base"

    @abstractmethod
    def get_samples(self, split: str = "train", n: int = -1) -> list[dict]:
        """Return a list of sample dicts.

        Each sample must have at minimum:
            - 'id':     unique identifier
            - 'domain': this domain's name
        Plus task-specific fields.
        """

    @abstractmethod
    def score(self, sample: dict, prediction: str) -> float:
        """Score a single prediction against ground truth. Returns float in [0, 1]."""

    def evaluate(self, agent, split: str = "train", n: int = -1) -> dict:
        """Run agent on all samples and return aggregated results."""
        samples = self.get_samples(split=split, n=n)
        scores = []
        results = []
        for sample in samples:
            try:
                prediction, _ = agent.forward(sample)
                s = self.score(sample, prediction)
            except Exception as exc:
                prediction = f"ERROR: {exc}"
                s = 0.0
            scores.append(s)
            results.append({"id": sample.get("id"), "prediction": prediction, "score": s})

        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {"score": mean_score, "results": results, "n": len(scores)}
