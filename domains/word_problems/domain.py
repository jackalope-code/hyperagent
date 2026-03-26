"""Multi-step word problem domain for HyperAgents.

20 problems requiring 2–4 arithmetic operations with integer answers.
A bare gpt-4o-mini prompt scores ~0.50–0.70, leaving clear room for the
MetaAgent to improve the task-solving strategy.
"""

import re

from domains.base_domain import Domain

_SAMPLES = [
    # id, question, answer
    {"id": "1",  "question": "A baker bakes 12 loaves per hour for 3 hours, then gives away 15 loaves. How many loaves remain?", "answer": "21"},
    {"id": "2",  "question": "A train travels 60 km/h for 2 hours, then 80 km/h for 3 hours. How many km did it travel in total?", "answer": "360"},
    {"id": "3",  "question": "A shop has 200 apples. It sells 45 in the morning and 38 in the afternoon, then receives a delivery of 60 more. How many apples does the shop have now?", "answer": "177"},
    {"id": "4",  "question": "Tom earns $15 per hour. He works 8 hours on Monday and 6 hours on Tuesday. How much does he earn in total?", "answer": "210"},
    {"id": "5",  "question": "A rectangle has a length of 14 cm and a width of 9 cm. What is its perimeter?", "answer": "46"},
    {"id": "6",  "question": "A school has 4 classes of 28 students each. After 12 students leave the school, how many students remain?", "answer": "100"},
    {"id": "7",  "question": "A tank holds 500 litres. It is 3/5 full. If 80 litres are removed, how many litres are left in the tank?", "answer": "220"},
    {"id": "8",  "question": "Alice has 3 times as many marbles as Bob. Bob has 24 marbles. If Alice gives 18 marbles to Bob, how many does Alice have left?", "answer": "54"},
    {"id": "9",  "question": "A cinema has 15 rows of 22 seats. 187 seats are occupied. How many seats are empty?", "answer": "143"},
    {"id": "10", "question": "A car uses 8 litres of petrol per 100 km. How many litres does it use on a 350 km trip?", "answer": "28"},
    {"id": "11", "question": "A farmer plants 6 rows of corn with 25 plants per row, and 4 rows of wheat with 30 plants per row. How many plants are there in total?", "answer": "270"},
    {"id": "12", "question": "A bucket is filled at 9 litres per minute and drained at 4 litres per minute. How many litres are in the bucket after 12 minutes if it starts empty?", "answer": "60"},
    {"id": "13", "question": "A bookshelf has 5 shelves. Each shelf holds 32 books. The owner removes 47 books and donates 15 more. How many books remain on the shelf?", "answer": "98"},
    {"id": "14", "question": "A recipe uses 250 g of flour per cake. If you have 2 kg of flour and want to make 6 cakes, how many grams of flour will you have left over?", "answer": "500"},
    {"id": "15", "question": "A runner completes 3 laps of a 400 m track, rests, then runs 2 more laps. How many metres did the runner cover in total?", "answer": "2000"},
    {"id": "16", "question": "Three friends split a restaurant bill of $126 equally. Each leaves a $7 tip. What is the total amount each person pays?", "answer": "49"},
    {"id": "17", "question": "A warehouse receives 4 pallets each holding 150 boxes. Workers unload 220 boxes on the first day and 175 on the second. How many boxes remain?", "answer": "205"},
    {"id": "18", "question": "A worker earns $18 per hour for normal time and $27 per hour for overtime. She works 40 normal hours and 5 overtime hours. What are her total earnings?", "answer": "855"},
    {"id": "19", "question": "A garden centre sells small plants for $4 each and large plants for $9 each. A customer buys 7 small and 3 large plants. How much does the customer pay?", "answer": "55"},
    {"id": "20", "question": "A coach drives 5 passengers to the airport. The one-way fare is $23 per person. On the return journey the coach is empty. What is the total fare collected for the round trip?", "answer": "115"},
]


class WordProblemsDomain(Domain):
    name = "word_problems"

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
        # Accept exact match or number with optional trailing .0
        pred = re.sub(r"[^\d.\-]", "", str(prediction).strip())
        # Normalise "21.0" → "21"
        try:
            if "." in pred:
                pred = str(int(float(pred)))
        except ValueError:
            pass
        return 1.0 if pred == expected else 0.0
