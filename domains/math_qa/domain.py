"""Simple arithmetic word-problem domain for demonstrating HyperAgents.

The domain asks the agent 20 multi-step math questions and scores exact
numeric matches.  It is intentionally simple so results are quick to evaluate
and easy to verify — the MetaAgent can improve task_agent.py to get a
higher score here.
"""

import re

from domains.base_domain import Domain

_SAMPLES = [
    {"id": "1",  "question": "What is 15 + 27?",                        "answer": "42"},
    {"id": "2",  "question": "What is 100 - 37?",                       "answer": "63"},
    {"id": "3",  "question": "What is 6 * 8?",                          "answer": "48"},
    {"id": "4",  "question": "What is 144 / 12?",                       "answer": "12"},
    {"id": "5",  "question": "What is 2 ** 10?",                        "answer": "1024"},
    {"id": "6",  "question": "What is 17 * 13?",                        "answer": "221"},
    {"id": "7",  "question": "What is 500 / 25?",                       "answer": "20"},
    {"id": "8",  "question": "What is 99 + 101?",                       "answer": "200"},
    {"id": "9",  "question": "What is 7 * 7?",                          "answer": "49"},
    {"id": "10", "question": "What is 1000 - 456?",                     "answer": "544"},
    {"id": "11", "question": "What is the sum of 3, 5, 7, and 11?",     "answer": "26"},
    {"id": "12", "question": "What is 9 ** 2?",                         "answer": "81"},
    {"id": "13", "question": "What is 360 / 8?",                        "answer": "45"},
    {"id": "14", "question": "What is 123 * 4?",                        "answer": "492"},
    {"id": "15", "question": "What is 1024 / 4?",                       "answer": "256"},
    {"id": "16", "question": "What is 37 + 63?",                        "answer": "100"},
    {"id": "17", "question": "What is 5 ** 3?",                         "answer": "125"},
    {"id": "18", "question": "What is 200 - 87?",                       "answer": "113"},
    {"id": "19", "question": "What is 12 * 15?",                        "answer": "180"},
    {"id": "20", "question": "What is 999 + 1?",                        "answer": "1000"},
]


class MathQADomain(Domain):
    name = "math_qa"

    def get_samples(self, split: str = "train", n: int = -1) -> list[dict]:
        samples = [
            {
                "id": s["id"],
                "domain": self.name,
                "question": s["question"],
                "answer": s["answer"],
            }
            for s in _SAMPLES
        ]
        return samples[:n] if n > 0 else samples

    def score(self, sample: dict, prediction: str) -> float:
        expected = sample["answer"].strip()
        # Strip everything except digits, minus, and decimal point
        pred = re.sub(r"[^\d.\-]", "", str(prediction).strip())
        return 1.0 if pred == expected else 0.0
