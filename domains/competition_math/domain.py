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
    # ── Easy tier  (~65% expected gpt-4o-mini baseline) ──────────────────────
    # These require careful multi-step counting/arithmetic — not single-formula lookups.
    {
        "id": "E1",
        "tier": "easy",
        "question": (
            "How many 3-digit positive integers are divisible by both 4 and 6?"
        ),
        # LCM(4,6)=12; multiples of 12 in [100,999]: floor(999/12)-floor(99/12) = 83-8 = 75
        "answer": "75",
    },
    {
        "id": "E2",
        "tier": "easy",
        "question": (
            "A bag contains 3 red marbles and 5 blue marbles. Two marbles are "
            "drawn without replacement. The probability that both are blue can be "
            "written as p/q in lowest terms. What is p + q?"
        ),
        # P = C(5,2)/C(8,2) = 10/28 = 5/14; p+q = 19
        "answer": "19",
    },
    {
        "id": "E3",
        "tier": "easy",
        "question": (
            "The first term of a geometric sequence is 2 and the fifth term is 162. "
            "What is the common ratio? (Assume the ratio is a positive integer.)"
        ),
        # 2·r^4 = 162 → r^4 = 81 → r = 3
        "answer": "3",
    },
    {
        "id": "E4",
        "tier": "easy",
        "question": (
            "Find the sum of all integers from 1 to 99 inclusive that leave a "
            "remainder of 3 when divided by 7."
        ),
        # Terms: 3,10,17,...,94 (count=14); sum = 14*(3+94)/2 = 679
        "answer": "679",
    },
    {
        "id": "E5",
        "tier": "easy",
        "question": (
            "What is the remainder when 1! + 2! + 3! + 4! + 5! + 6! + 7! + 8! + 9! + 10! "
            "is divided by 9?"
        ),
        # For n≥6, n! is divisible by 9; sum of 1!..5! = 1+2+6+24+120 = 153 = 17*9 → rem 0
        "answer": "0",
    },
    {
        "id": "E6",
        "tier": "easy",
        "question": (
            "How many integers between 1 and 1000 inclusive are divisible by "
            "at least one of 3, 5, or 7?"
        ),
        # inclusion-exclusion: 333+200+142-66-47-28+9 = 543
        "answer": "543",
    },
    # ── Medium tier  (~40% expected baseline) ────────────────────────────────
    {
        "id": "M1",
        "tier": "medium",
        "question": (
            "How many 6-digit positive integers have digits in non-decreasing order "
            "from left to right, using only the digits 1 through 9 (no zeros)?"
        ),
        # Non-decreasing sequences of length 6 from {1..9}: C(9+6-1,6) = C(14,6) = 3003
        "answer": "3003",
    },
    {
        "id": "M2",
        "tier": "medium",
        "question": (
            "For how many integers n with 1 ≤ n ≤ 200 is n³ − n² divisible by 4?"
        ),
        # n^2(n-1) div by 4: n even (100 values) OR n≡1(mod 4) (50 values) → 150
        "answer": "150",
    },
    {
        "id": "M3",
        "tier": "medium",
        "question": (
            "A circle with centre O and radius 5 is given. From an external point P "
            "with OP = 13, two tangent lines are drawn, touching the circle at A and B. "
            "What is the area of quadrilateral OAPB?"
        ),
        # PA = PB = sqrt(169-25) = 12; area = OA*PA = 5*12 = 60
        "answer": "60",
    },
    {
        "id": "M4",
        "tier": "medium",
        "question": (
            "Compute the sum 1·2 + 2·3 + 3·4 + … + 50·51."
        ),
        # Sum = n(n+1)(n+2)/3 for n=50 = 50*51*52/3 = 44200
        "answer": "44200",
    },
    {
        "id": "M5",
        "tier": "medium",
        "question": (
            "The polynomial x³ − 6x² + 11x − 6 has three integer roots. "
            "What is the sum of the squares of its roots?"
        ),
        # Roots 1,2,3; sum of squares = 1+4+9 = 14
        "answer": "14",
    },
    {
        "id": "M6",
        "tier": "medium",
        "question": (
            "What is the remainder when 17^2026 is divided by 13?"
        ),
        # 17≡4(mod13); 4^6≡1(mod13); 2026=6*337+4; 4^4≡(4^3)*4=(-1)*4=-4≡9(mod13)
        "answer": "9",
    },
    {
        "id": "M7",
        "tier": "medium",
        "question": (
            "What is the largest prime factor of 2^8 − 1?"
        ),
        # 2^8-1 = 255 = 3*5*17; largest prime factor = 17
        "answer": "17",
    },
    {
        "id": "M8",
        "tier": "medium",
        "question": (
            "How many ordered pairs of non-negative integers (a, b) satisfy "
            "a + 2b ≤ 20?"
        ),
        # For b=0..10: (21-2b) values of a; sum = 21+19+...+1 = 11^2 = 121
        "answer": "121",
    },
    # ── Hard tier  (~25% expected baseline) ──────────────────────────────────
    {
        "id": "H1",
        "tier": "hard",
        "question": (
            "What is the least common multiple of all integers from 1 to 15 inclusive?"
        ),
        # LCM = 2^3 * 3^2 * 5 * 7 * 11 * 13 = 360360
        "answer": "360360",
    },
    {
        "id": "H2",
        "tier": "hard",
        "question": (
            "A 3×3 grid of cells is to be coloured with three colours (Red, Blue, Green) "
            "so that each row and each column contains exactly one cell of each colour. "
            "How many valid colourings are there?"
        ),
        # 3x3 Latin squares = 12
        "answer": "12",
    },
    {
        "id": "H3",
        "tier": "hard",
        "question": (
            "A sequence satisfies a₁ = 1 and aₙ₊₁ = aₙ / (1 + aₙ) for all n ≥ 1. "
            "What is 1 / a₁₀₀?"
        ),
        # bₙ = 1/aₙ satisfies bₙ₊₁ = bₙ + 1, b₁ = 1 → bₙ = n → 1/a₁₀₀ = 100
        "answer": "100",
    },
    {
        "id": "H4",
        "tier": "hard",
        "question": (
            "In how many ways can a 2×10 rectangle be completely tiled "
            "using 1×2 dominoes (which may be placed horizontally or vertically)?"
        ),
        # T(n): T(1)=1,T(2)=2,T(n)=T(n-1)+T(n-2); T(10)=89
        "answer": "89",
    },
    {
        "id": "H5",
        "tier": "hard",
        "question": (
            "A committee of 3 people is chosen at random from a group of 7 boys "
            "and 5 girls. How many committees include at least one girl?"
        ),
        # C(12,3) - C(7,3) = 220 - 35 = 185
        "answer": "185",
    },
    {
        "id": "H6",
        "tier": "hard",
        "question": (
            "What is the greatest common divisor of 3^2026 − 1 and 3^2021 − 1?"
        ),
        # GCD(3^a-1, 3^b-1) = 3^gcd(a,b)-1; gcd(2026,2021)=gcd(2021,5)=gcd(5,1)=1; answer=2
        "answer": "2",
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
        all_samples = [
            {
                "id": s["id"],
                "tier": s["tier"],
                "domain": self.name,
                "question": s["question"],
                "answer": s["answer"],
            }
            for s in _SAMPLES
        ]
        # Interleave tiers so that any prefix of n samples contains a mix of
        # Easy / Medium / Hard problems.  This ensures --eval_samples 14 tests
        # the Hard tier rather than only Easy + Medium.
        easy   = [s for s in all_samples if s["tier"] == "easy"]
        medium = [s for s in all_samples if s["tier"] == "medium"]
        hard   = [s for s in all_samples if s["tier"] == "hard"]
        interleaved: list[dict] = []
        for i in range(max(len(easy), len(medium), len(hard))):
            if i < len(easy):   interleaved.append(easy[i])
            if i < len(medium): interleaved.append(medium[i])
            if i < len(hard):   interleaved.append(hard[i])
        return interleaved[:n] if n > 0 else interleaved

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
