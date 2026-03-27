"""Competition Math domain for HyperAgents.

20 problems drawn from AIME/AMC-style benchmarks across three difficulty tiers:
  - Easy   (6 problems, ~75% expected gpt-4o-mini baseline) — algebra, direct computation
  - Medium (8 problems, ~50% expected baseline) — combinatorics, number theory, sequences
  - Hard   (6 problems, ~30-35% expected baseline) — AIME-style multi-step reasoning

All answers are integers.  The scorer accepts exact string matches, float-equivalent
strings (e.g. "6.0" for 6), and simple fraction notation (e.g. "4/2" for 2).
"""

import re
from fractions import Fraction

from domains.base_domain import Domain

_SAMPLES = [
    # ── Easy tier ────────────────────────────────────────────────────────────
    {
        "id": "E1",
        "tier": "easy",
        "question": (
            "Three consecutive integers have a sum of 57. "
            "What is the smallest of the three integers?"
        ),
        "answer": "18",
    },
    {
        "id": "E2",
        "tier": "easy",
        "question": (
            "Solve the system of equations: 4x − 3y = 17 and 2x + y = 11. "
            "What is the value of x?"
        ),
        "answer": "5",
    },
    {
        "id": "E3",
        "tier": "easy",
        "question": (
            "A right triangle has legs of length 5 cm and 12 cm. "
            "What is the area of the triangle in square centimetres?"
        ),
        "answer": "30",
    },
    {
        "id": "E4",
        "tier": "easy",
        "question": "What is the units digit of 7^2026?",
        "answer": "9",
    },
    {
        "id": "E5",
        "tier": "easy",
        "question": (
            "How many integers from 1 to 200 inclusive are divisible by 6 "
            "but not divisible by 9?"
        ),
        "answer": "22",
    },
    {
        "id": "E6",
        "tier": "easy",
        "question": (
            "Two positive integers have a product of 180 and a greatest common "
            "divisor of 6. What is their least common multiple?"
        ),
        "answer": "30",
    },
    # ── Medium tier ───────────────────────────────────────────────────────────
    {
        "id": "M1",
        "tier": "medium",
        "question": (
            "How many 3-digit positive integers have digits that are "
            "in strictly increasing order from left to right?"
        ),
        "answer": "84",
    },
    {
        "id": "M2",
        "tier": "medium",
        "question": (
            "What is the largest prime factor of 12! ÷ 10! "
            "(where n! denotes n factorial)?"
        ),
        "answer": "11",
    },
    {
        "id": "M3",
        "tier": "medium",
        "question": (
            "A geometric sequence has first term 3 and common ratio 2. "
            "What is the sum of the first 8 terms?"
        ),
        "answer": "765",
    },
    {
        "id": "M4",
        "tier": "medium",
        "question": (
            "In how many distinct ways can the letters of the word ARRANGE "
            "be rearranged (including the original arrangement)?"
        ),
        "answer": "1260",
    },
    {
        "id": "M5",
        "tier": "medium",
        "question": "What is the remainder when 2^50 is divided by 7?",
        "answer": "4",
    },
    {
        "id": "M6",
        "tier": "medium",
        "question": (
            "A ball is dropped from a height of 64 metres. After each bounce "
            "it rises to exactly half the height from which it fell. "
            "What is the total distance travelled by the ball before it comes "
            "to rest (in metres)?"
        ),
        "answer": "192",
    },
    {
        "id": "M7",
        "tier": "medium",
        "question": (
            "How many positive integers less than 1000 are divisible by 4 or "
            "by 6 (or both)?"
        ),
        "answer": "332",
    },
    {
        "id": "M8",
        "tier": "medium",
        "question": (
            "A regular hexagon has an area of 54√3 square units. "
            "What is the side length of the hexagon?"
        ),
        "answer": "6",
    },
    # ── Hard tier ─────────────────────────────────────────────────────────────
    {
        "id": "H1",
        "tier": "hard",
        "question": (
            "How many ordered pairs (x, y) of positive integers satisfy the "
            "equation x + y + xy = 98?"
        ),
        "answer": "4",
    },
    {
        "id": "H2",
        "tier": "hard",
        "question": (
            "The sum of the first n terms of a sequence is given by S(n) = 3n² + 5n. "
            "What is the 15th term of the sequence?"
        ),
        "answer": "92",
    },
    {
        "id": "H3",
        "tier": "hard",
        "question": (
            "A right triangle has legs of length 8 and 15 (and hypotenuse 17). "
            "What is the radius of the inscribed circle (incircle) of the triangle?"
        ),
        "answer": "3",
    },
    {
        "id": "H4",
        "tier": "hard",
        "question": (
            "A fair coin is tossed 10 times. In how many of the 1024 equally "
            "likely outcomes do exactly 4 heads appear?"
        ),
        "answer": "210",
    },
    {
        "id": "H5",
        "tier": "hard",
        "question": (
            "What is the sum of all positive integers less than 100 that are "
            "coprime to 100 (i.e. share no common factor greater than 1 with 100)?"
        ),
        "answer": "2000",
    },
    {
        "id": "H6",
        "tier": "hard",
        "question": (
            "Let f(n) = n² − 4n + 7. For how many integers n with 1 ≤ n ≤ 20 "
            "is f(n) divisible by 3?"
        ),
        "answer": "7",
    },
]


def _parse_number(s: str):
    """Return a Fraction parsed from s, or None on failure.

    Accepts:
      - plain integers / decimals: "42", "6.0", "-3"
      - simple fractions:          "3/4", "22/7"
    """
    s = s.strip()
    # fraction notation p/q
    if re.fullmatch(r"-?\d+\s*/\s*\d+", s):
        try:
            return Fraction(s)
        except (ValueError, ZeroDivisionError):
            return None
    # decimal / integer
    try:
        return Fraction(s).limit_denominator(10**9)
    except ValueError:
        pass
    # strip trailing non-numeric noise (e.g. "42 metres")
    m = re.match(r"(-?\d+(?:\.\d+)?)", s)
    if m:
        try:
            return Fraction(m.group(1)).limit_denominator(10**9)
        except ValueError:
            return None
    return None


class CompetitionMathDomain(Domain):
    name = "competition_math"

    def get_samples(self, split: str = "train", n: int = -1) -> list[dict]:
        samples = [
            {
                "id": s["id"],
                "tier": s["tier"],
                "domain": self.name,
                "question": s["question"],
                "answer": s["answer"],
            }
            for s in _SAMPLES
        ]
        return samples[:n] if n > 0 else samples

    def score(self, sample: dict, prediction: str) -> float:
        expected_str = sample["answer"].strip()

        # 1. Exact string match
        if str(prediction).strip() == expected_str:
            return 1.0

        # 2. Numeric match via Fraction (handles "6.0", "765.0", "3/1", etc.)
        # Extract the last number-like token from the prediction to be robust
        # against the model embedding the answer in a sentence.
        tokens = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", str(prediction))
        expected_frac = _parse_number(expected_str)
        if expected_frac is not None:
            for tok in reversed(tokens):  # last numeric token is usually the answer
                pred_frac = _parse_number(tok)
                if pred_frac is not None and pred_frac == expected_frac:
                    return 1.0

        return 0.0
