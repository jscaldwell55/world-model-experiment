# evaluation/tasks.py
"""
Test query sets for evaluating agent understanding.

Each environment has queries testing:
- Interventional reasoning ("What would happen if...")
- Counterfactual reasoning ("What would have happened if...")
- Planning ("How can we achieve...")
"""

from typing import List, Callable, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class TestQuery:
    """Single evaluation query"""
    question: str
    query_type: str  # 'interventional', 'counterfactual', 'planning'
    expected_answer: str
    evaluation_function: Callable[[str, Dict], float]
    difficulty: str  # 'easy', 'medium', 'hard'

    def evaluate(self, agent_answer: str, ground_truth: Dict) -> float:
        """
        Evaluate agent answer.

        Returns:
            Score between 0.0 and 1.0
        """
        return self.evaluation_function(agent_answer, ground_truth)


# ============================================================================
# Evaluation Helper Functions
# ============================================================================

def contains_keywords(answer: str, keywords: List[str]) -> bool:
    """Check if answer contains any of the keywords"""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


def exact_match(answer: str, expected: str) -> bool:
    """Check for exact match (case insensitive)"""
    return expected.lower() in answer.lower()


def probability_match(answer: str, expected_prob: float, tolerance: float = 0.2) -> float:
    """
    Extract probability from answer and compare to expected.

    Returns:
        1.0 if within tolerance, otherwise 1 - error
    """
    # Try to extract percentage
    percent_match = re.search(r'(\d+)%', answer)
    if percent_match:
        agent_prob = float(percent_match.group(1)) / 100.0
    else:
        # Try decimal
        decimal_match = re.search(r'0\.\d+', answer)
        if decimal_match:
            agent_prob = float(decimal_match.group(0))
        else:
            # No probability found, check for qualitative
            if any(word in answer.lower() for word in ['unlikely', 'low', 'rare']):
                agent_prob = 0.2
            elif any(word in answer.lower() for word in ['likely', 'high', 'probable']):
                agent_prob = 0.8
            elif any(word in answer.lower() for word in ['maybe', 'uncertain', 'possible']):
                agent_prob = 0.5
            else:
                agent_prob = 0.5  # Default

    error = abs(agent_prob - expected_prob)
    if error <= tolerance:
        return 1.0
    else:
        return max(0.0, 1.0 - (error - tolerance) * 2)


def yes_no_match(answer: str, expected_yes: bool) -> float:
    """Check for yes/no answer"""
    answer_lower = answer.lower()

    has_yes = any(word in answer_lower for word in ['yes', 'will', 'would', 'true', 'correct'])
    has_no = any(word in answer_lower for word in ['no', 'not', 'won\'t', 'wouldn\'t', 'false', 'incorrect'])

    if expected_yes:
        if has_yes and not has_no:
            return 1.0
        elif has_no and not has_yes:
            return 0.0
    else:
        if has_no and not has_yes:
            return 1.0
        elif has_yes and not has_no:
            return 0.0

    return 0.5  # Ambiguous


# ============================================================================
# HotPotLab Test Queries
# ============================================================================

def _hotpot_burn_check(answer: str, gt: Dict) -> float:
    """Check if agent correctly predicts burn"""
    temp = gt.get('actual_temp', 20.0)
    will_burn = temp > 60.0

    return yes_no_match(answer, expected_yes=will_burn)


def _hotpot_temp_prediction(answer: str, gt: Dict) -> float:
    """Check temperature prediction accuracy"""
    # Extract predicted temp from answer
    temp_match = re.search(r'(\d+)\s*(?:degrees|C)', answer)
    if temp_match:
        predicted_temp = float(temp_match.group(1))
        actual_temp = gt.get('actual_temp', 20.0)

        error = abs(predicted_temp - actual_temp)
        if error <= 10:
            return 1.0
        elif error <= 20:
            return 0.5
        else:
            return 0.0

    # Qualitative check
    actual_temp = gt.get('actual_temp', 20.0)
    if actual_temp > 60 and any(word in answer.lower() for word in ['hot', 'high', 'burn']):
        return 0.7
    elif actual_temp < 30 and any(word in answer.lower() for word in ['cold', 'cool', 'low']):
        return 0.7

    return 0.3


def get_hotpot_queries() -> List[TestQuery]:
    """Test queries for HotPotLab environment"""
    return [
        TestQuery(
            question="If we wait 30 seconds with the stove on high and then touch the pot, what happens?",
            query_type="interventional",
            expected_answer="burn",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['burn', 'hot', 'hurt']) else 0.0,
            difficulty="easy"
        ),
        TestQuery(
            question="The pot is labeled 'Boiling!' but you measured 25�C. If we wait 2 minutes without turning on the stove, will touching it burn us?",
            query_type="interventional",
            expected_answer="no",
            evaluation_function=lambda ans, gt: yes_no_match(ans, expected_yes=False),
            difficulty="medium"
        ),
        TestQuery(
            question="What is the safest way to determine if the pot is hot enough to cause burns?",
            query_type="planning",
            expected_answer="measure_temp",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['measure', 'thermometer', 'temperature']) else 0.0,
            difficulty="easy"
        ),
        TestQuery(
            question="If we had waited 60 seconds with the stove on high instead of measuring immediately, would the temperature be above 80�C?",
            query_type="counterfactual",
            expected_answer="yes",
            evaluation_function=lambda ans, gt: yes_no_match(ans, expected_yes=True),
            difficulty="hard"
        ),
        TestQuery(
            question="After toggling the stove to low and waiting 40 seconds, approximately what temperature would the pot reach?",
            query_type="interventional",
            expected_answer="60�C",
            evaluation_function=_hotpot_temp_prediction,
            difficulty="medium"
        ),
        TestQuery(
            question="Should we trust the initial label 'Boiling!' without measuring?",
            query_type="planning",
            expected_answer="no",
            evaluation_function=lambda ans, gt: yes_no_match(ans, expected_yes=False),
            difficulty="easy"
        ),
        TestQuery(
            question="If the stove has been on high for 50 seconds, is it safe to touch the pot?",
            query_type="interventional",
            expected_answer="no",
            evaluation_function=lambda ans, gt: yes_no_match(ans, expected_yes=False),
            difficulty="easy"
        ),
        TestQuery(
            question="What evidence would definitively tell us the heating rate of the stove?",
            query_type="planning",
            expected_answer="multiple measurements",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['multiple', 'several', 'repeated', 'measurements', 'over time']) else 0.5,
            difficulty="hard"
        ),
        TestQuery(
            question="If we had turned the stove to high immediately and waited 1 minute, would we see a temperature above 100�C?",
            query_type="counterfactual",
            expected_answer="yes",
            evaluation_function=lambda ans, gt: yes_no_match(ans, expected_yes=True),
            difficulty="medium"
        ),
        TestQuery(
            question="After observing one temperature measurement of 30�C at t=10s with stove on low, what heating rate would you estimate?",
            query_type="interventional",
            expected_answer="1.0 C/s",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['1', 'one', 'degree']) else 0.5,
            difficulty="hard"
        ),
    ]


# ============================================================================
# SwitchLight Test Queries
# ============================================================================

def _switchlight_uncertain(answer: str, gt: Dict) -> float:
    """Check if agent expresses appropriate uncertainty"""
    uncertain_words = ['uncertain', 'maybe', 'might', 'possibly', 'depends', 'unknown', 'unsure']
    if any(word in answer.lower() for word in uncertain_words):
        return 1.0
    elif any(word in answer.lower() for word in ['definitely', 'certainly', 'always', 'never']):
        return 0.0
    return 0.5


def get_switchlight_queries() -> List[TestQuery]:
    """Test queries for SwitchLight environment"""
    return [
        TestQuery(
            question="If we flip the switch to ON, will the light definitely turn on?",
            query_type="interventional",
            expected_answer="uncertain",
            evaluation_function=_switchlight_uncertain,
            difficulty="medium"
        ),
        TestQuery(
            question="What would you need to observe to determine the wiring layout?",
            query_type="planning",
            expected_answer="flip multiple times",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['flip', 'multiple', 'several', 'observe', 'pattern']) else 0.5,
            difficulty="medium"
        ),
        TestQuery(
            question="If the light is currently OFF with switch ON, what's the most likely explanation?",
            query_type="interventional",
            expected_answer="inverted or faulty",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['invert', 'reverse', 'fault', 'relay', 'broken', 'layout B']) else 0.0,
            difficulty="hard"
        ),
        TestQuery(
            question="If we had jiggled the relay before flipping the switch, would the outcome have been different?",
            query_type="counterfactual",
            expected_answer="possibly",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['might', 'possibly', 'depends', 'maybe', 'could']) else 0.0,
            difficulty="hard"
        ),
        TestQuery(
            question="What's the probability the relay is faulty?",
            query_type="interventional",
            expected_answer="10%",
            evaluation_function=lambda ans, gt: probability_match(ans, 0.10, tolerance=0.05),
            difficulty="medium"
        ),
        TestQuery(
            question="Can we determine the wiring by only observing without intervening?",
            query_type="planning",
            expected_answer="no",
            evaluation_function=lambda ans, gt: yes_no_match(ans, expected_yes=False),
            difficulty="medium"
        ),
        TestQuery(
            question="If we flip the switch 3 times and see [ON�ON, OFF�OFF, ON�ON], what's the likely wiring?",
            query_type="interventional",
            expected_answer="layout A",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['A', 'normal', 'standard']) else 0.0,
            difficulty="hard"
        ),
        TestQuery(
            question="Should we pay for an inspection to see the wiring?",
            query_type="planning",
            expected_answer="depends",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['depend', 'maybe', 'alternative', 'flip']) else 0.5,
            difficulty="medium"
        ),
        TestQuery(
            question="If the switch is OFF and light is ON, what does this tell us?",
            query_type="interventional",
            expected_answer="inverted wiring",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['invert', 'reverse', 'B', 'unusual']) else 0.0,
            difficulty="easy"
        ),
        TestQuery(
            question="What action would give us the most information about the system?",
            query_type="planning",
            expected_answer="flip switch",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['flip', 'toggle', 'switch']) else 0.5,
            difficulty="easy"
        ),
    ]


# ============================================================================
# ChemTile Test Queries
# ============================================================================

def get_chemtile_queries() -> List[TestQuery]:
    """Test queries for ChemTile environment"""
    return [
        TestQuery(
            question="If we mix compound A and B at medium temperature, what are the chances of an explosion?",
            query_type="interventional",
            expected_answer="10%",
            evaluation_function=lambda ans, gt: probability_match(ans, 0.10, tolerance=0.05),
            difficulty="medium"
        ),
        TestQuery(
            question="What's the safest path to produce compound D?",
            query_type="planning",
            expected_answer="low temp, A+B then C+B",
            evaluation_function=lambda ans, gt: 1.0 if (contains_keywords(ans, ['A', 'B', 'C']) and contains_keywords(ans, ['low', 'cool'])) else 0.5,
            difficulty="hard"
        ),
        TestQuery(
            question="If we increase temperature to high before mixing A and B, how does explosion risk change?",
            query_type="interventional",
            expected_answer="doubles",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['increase', 'higher', 'more', 'double']) else 0.0,
            difficulty="medium"
        ),
        TestQuery(
            question="If we had cooled the system before mixing C and B, would explosion probability decrease?",
            query_type="counterfactual",
            expected_answer="yes",
            evaluation_function=lambda ans, gt: yes_no_match(ans, expected_yes=True),
            difficulty="medium"
        ),
        TestQuery(
            question="Can we produce D without any risk of explosion?",
            query_type="planning",
            expected_answer="no",
            evaluation_function=lambda ans, gt: yes_no_match(ans, expected_yes=False),
            difficulty="easy"
        ),
        TestQuery(
            question="What's the probability of successfully creating C from A and B at medium temp?",
            query_type="interventional",
            expected_answer="80%",
            evaluation_function=lambda ans, gt: probability_match(ans, 0.80, tolerance=0.1),
            difficulty="easy"
        ),
        TestQuery(
            question="If C+B produces D with 70% success at low temp, what's the overall probability of reaching D from starting compounds?",
            query_type="interventional",
            expected_answer="56%",
            evaluation_function=lambda ans, gt: probability_match(ans, 0.56, tolerance=0.15),
            difficulty="hard"
        ),
        TestQuery(
            question="Should we heat or cool before attempting to create D?",
            query_type="planning",
            expected_answer="cool",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['cool', 'lower', 'decrease']) else 0.0,
            difficulty="medium"
        ),
        TestQuery(
            question="What information would help us decide whether to attempt mixing C and B?",
            query_type="planning",
            expected_answer="temperature and risk tolerance",
            evaluation_function=lambda ans, gt: 1.0 if contains_keywords(ans, ['temperature', 'temp', 'risk']) else 0.5,
            difficulty="medium"
        ),
        TestQuery(
            question="If we mixed A+B at high temperature and it didn't explode, should we interpret this as low explosion risk?",
            query_type="interventional",
            expected_answer="no",
            evaluation_function=lambda ans, gt: yes_no_match(ans, expected_yes=False),
            difficulty="hard"
        ),
    ]


# ============================================================================
# Main Interface
# ============================================================================

def get_test_queries(environment_name: str) -> List[TestQuery]:
    """
    Get test query set for environment.

    Args:
        environment_name: Name of environment class

    Returns:
        List of TestQuery objects
    """
    if environment_name == "HotPotLab":
        return get_hotpot_queries()
    elif environment_name == "SwitchLight":
        return get_switchlight_queries()
    elif environment_name == "ChemTile":
        return get_chemtile_queries()
    else:
        raise ValueError(f"Unknown environment: {environment_name}")


def get_query_statistics(queries: List[TestQuery]) -> Dict[str, Any]:
    """
    Get statistics about query set.

    Returns:
        Dictionary with query statistics
    """
    total = len(queries)

    by_type = {}
    by_difficulty = {}

    for q in queries:
        by_type[q.query_type] = by_type.get(q.query_type, 0) + 1
        by_difficulty[q.difficulty] = by_difficulty.get(q.difficulty, 0) + 1

    return {
        'total_queries': total,
        'by_type': by_type,
        'by_difficulty': by_difficulty
    }
