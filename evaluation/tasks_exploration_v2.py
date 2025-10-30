# evaluation/tasks_exploration_v2.py
"""
Exploration-dependent test queries (V2).

CRITICAL REQUIREMENT: All questions must be UNANSWERABLE from environment
descriptions alone. They must require actual exploration data (measurements,
observations, action results).

Design Principles:
1. Questions reference SPECIFIC measurements agent could have taken
2. Questions ask about ENVIRONMENT-SPECIFIC dynamics (not general knowledge)
3. Questions require TEMPORAL data or STATE TRANSITIONS
4. Observer (passive agent) should score <40% on these questions
"""

from typing import List, Callable, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class TestQuery:
    """Single evaluation query"""
    question: str
    query_type: str  # 'interventional', 'counterfactual', 'planning'
    expected_answer: Any  # Can be string, number, or dict
    evaluation_function: Callable[[str, Dict], float]
    difficulty: str  # 'easy', 'medium', 'hard'
    requires_exploration: bool = True  # All V2 questions require exploration

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

def extract_number(text: str) -> float:
    """Extract first number from text"""
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return float(numbers[0])
    return None


def numeric_tolerance_check(answer: str, expected: float, tolerance: float = 2.0) -> float:
    """
    Check if numeric answer is within tolerance of expected value.

    Args:
        answer: Agent's text answer
        expected: Expected numeric value
        tolerance: Absolute tolerance (default: ±2 for temperatures)

    Returns:
        1.0 if within tolerance, partial credit for close, 0.0 otherwise
    """
    agent_value = extract_number(answer)

    if agent_value is None:
        return 0.0

    error = abs(agent_value - expected)

    if error <= tolerance:
        return 1.0
    elif error <= tolerance * 2:
        return 0.5
    else:
        return 0.0


def temporal_reference_check(answer: str, gt: Dict, time_key: str, value_key: str, tolerance: float) -> float:
    """
    Check if agent correctly reports a value at a specific time.

    Args:
        answer: Agent's answer
        gt: Ground truth dict containing trajectory data
        time_key: Key in gt for time (e.g., 'temperature_at_t30')
        value_key: Key within that for the actual value
        tolerance: Tolerance for numeric comparison
    """
    expected = gt.get(time_key, {}).get(value_key)

    if expected is None:
        # Ground truth not available, can't evaluate
        return 0.5  # Neutral score

    return numeric_tolerance_check(answer, expected, tolerance)


# ============================================================================
# HotPotLab Exploration-Dependent Queries
# ============================================================================

def _hotpot_exact_temp_at_time(answer: str, gt: Dict, time_seconds: int) -> float:
    """
    Check if agent reports correct temperature at specific time.
    Requires agent to have actually measured at that time.
    """
    # Ground truth should contain temperature trajectory
    # Format: gt['temperature_trajectory'] = [(time, temp), ...]
    trajectory = gt.get('temperature_trajectory', [])

    # Find closest measurement to requested time
    target_temp = None
    for t, temp in trajectory:
        if abs(t - time_seconds) <= 1.0:  # Within 1 second
            target_temp = temp
            break

    if target_temp is None:
        # No measurement at that time, agent can't possibly know
        # Check if they admit uncertainty
        if any(word in answer.lower() for word in ['uncertain', "don't know", 'not measured', 'did not measure']):
            return 1.0  # Correct to admit uncertainty
        else:
            return 0.0  # Claimed knowledge they don't have

    return numeric_tolerance_check(answer, target_temp, tolerance=2.0)


def _hotpot_heating_rate(answer: str, gt: Dict) -> float:
    """
    Check if agent calculated correct heating rate from measurements.
    Requires multiple temperature measurements over time.
    """
    # Ground truth heating rate
    actual_rate = gt.get('heating_rate', 0.0)

    if actual_rate == 0.0:
        # No heating occurred
        return 0.5  # Can't test this

    return numeric_tolerance_check(answer, actual_rate, tolerance=0.5)


def _hotpot_when_exceeded_threshold(answer: str, gt: Dict, threshold: float) -> float:
    """
    Check if agent correctly identifies when temperature exceeded threshold.
    Requires temporal tracking of measurements.
    """
    # Ground truth: time when threshold was exceeded
    trajectory = gt.get('temperature_trajectory', [])

    time_exceeded = None
    for t, temp in trajectory:
        if temp >= threshold:
            time_exceeded = t
            break

    if time_exceeded is None:
        # Never exceeded threshold
        if any(word in answer.lower() for word in ['never', 'did not', 'not exceed']):
            return 1.0
        else:
            return 0.0

    # Check if agent's answer is close to actual time
    agent_time = extract_number(answer)
    if agent_time is None:
        return 0.0

    error = abs(agent_time - time_exceeded)
    if error <= 2.0:  # Within 2 seconds
        return 1.0
    elif error <= 5.0:  # Within 5 seconds
        return 0.5
    else:
        return 0.0


def get_hotpot_queries_v2() -> List[TestQuery]:
    """
    Exploration-dependent test queries for HotPotLab.

    These questions CANNOT be answered from environment description.
    They require actual measurements during exploration.
    """
    return [
        # EASY - Single specific measurement
        TestQuery(
            question="What exact temperature (in Celsius) did you measure at time t=10 seconds?",
            query_type="interventional",
            expected_answer="[from actual measurement]",
            evaluation_function=lambda ans, gt: _hotpot_exact_temp_at_time(ans, gt, time_seconds=10),
            difficulty="easy",
            requires_exploration=True
        ),

        TestQuery(
            question="What exact temperature (in Celsius) did you measure at time t=20 seconds?",
            query_type="interventional",
            expected_answer="[from actual measurement]",
            evaluation_function=lambda ans, gt: _hotpot_exact_temp_at_time(ans, gt, time_seconds=20),
            difficulty="easy",
            requires_exploration=True
        ),

        TestQuery(
            question="What was the FIRST temperature measurement you took in this episode?",
            query_type="interventional",
            expected_answer="[from first measurement]",
            evaluation_function=lambda ans, gt: _hotpot_exact_temp_at_time(ans, gt, time_seconds=gt.get('first_measurement_time', 0)),
            difficulty="easy",
            requires_exploration=True
        ),

        # MEDIUM - Requires understanding dynamics from multiple measurements
        TestQuery(
            question="What is the heating rate in degrees Celsius per second when the stove is on high?",
            query_type="interventional",
            expected_answer="[calculated from measurements]",
            evaluation_function=_hotpot_heating_rate,
            difficulty="medium",
            requires_exploration=True
        ),

        TestQuery(
            question="At what time (in seconds) did the temperature first exceed 80 degrees Celsius?",
            query_type="interventional",
            expected_answer="[from temporal tracking]",
            evaluation_function=lambda ans, gt: _hotpot_when_exceeded_threshold(ans, gt, threshold=80.0),
            difficulty="medium",
            requires_exploration=True
        ),

        TestQuery(
            question="What was the temperature difference between your first and last measurement?",
            query_type="interventional",
            expected_answer="[from measurements]",
            evaluation_function=lambda ans, gt: numeric_tolerance_check(
                ans,
                gt.get('final_temp', 0) - gt.get('initial_temp', 0),
                tolerance=3.0
            ),
            difficulty="medium",
            requires_exploration=True
        ),

        TestQuery(
            question="What was the pot's temperature immediately AFTER you toggled the stove?",
            query_type="interventional",
            expected_answer="[from measurement after action]",
            evaluation_function=lambda ans, gt: numeric_tolerance_check(
                ans,
                gt.get('temp_after_toggle', 20.0),
                tolerance=2.0
            ),
            difficulty="medium",
            requires_exploration=True
        ),

        # HARD - Requires extrapolation or counterfactual reasoning from measurements
        TestQuery(
            question="Based on the heating rate you observed, if you had left the stove on for 60 seconds total, what temperature would the pot reach?",
            query_type="counterfactual",
            expected_answer="[extrapolated from rate]",
            evaluation_function=lambda ans, gt: numeric_tolerance_check(
                ans,
                gt.get('initial_temp', 20) + gt.get('heating_rate', 0) * 60,
                tolerance=10.0
            ),
            difficulty="hard",
            requires_exploration=True
        ),

        TestQuery(
            question="At what time would the pot reach 100 degrees Celsius at the observed heating rate?",
            query_type="counterfactual",
            expected_answer="[calculated from rate]",
            evaluation_function=lambda ans, gt: numeric_tolerance_check(
                ans,
                (100 - gt.get('initial_temp', 20)) / max(gt.get('heating_rate', 1), 0.1),
                tolerance=5.0
            ),
            difficulty="hard",
            requires_exploration=True
        ),

        TestQuery(
            question="How many times did you measure the temperature during this episode?",
            query_type="interventional",
            expected_answer="[count from actions]",
            evaluation_function=lambda ans, gt: 1.0 if extract_number(ans) == gt.get('num_measurements', 0) else 0.0,
            difficulty="easy",
            requires_exploration=True
        ),
    ]


# ============================================================================
# SwitchLight Exploration-Dependent Queries
# ============================================================================

def _switchlight_state_at_configuration(answer: str, gt: Dict, config: str) -> float:
    """
    Check if agent correctly reports light state for a specific switch configuration.
    Requires agent to have tested that configuration.
    """
    # Ground truth should contain tested configurations
    # Format: gt['tested_configs'] = {'on': True, 'off': False, ...}
    tested_configs = gt.get('tested_configurations', {})

    if config not in tested_configs:
        # Agent didn't test this configuration
        if any(word in answer.lower() for word in ['uncertain', "don't know", 'not tested', 'did not test']):
            return 1.0  # Correct to admit uncertainty
        else:
            return 0.0  # Claimed knowledge they don't have

    expected_light_on = tested_configs[config]

    # Check if answer matches
    answer_lower = answer.lower()
    has_on = any(word in answer_lower for word in ['on', 'lit', 'illuminated', 'true', 'yes'])
    has_off = any(word in answer_lower for word in ['off', 'dark', 'not lit', 'false', 'no'])

    if expected_light_on and has_on and not has_off:
        return 1.0
    elif not expected_light_on and has_off and not has_on:
        return 1.0
    else:
        return 0.0


def _switchlight_wiring_from_observations(answer: str, gt: Dict) -> float:
    """
    Check if agent correctly identified wiring layout from observations.
    """
    actual_layout = gt.get('wire_layout', 'unknown')

    answer_lower = answer.lower()

    if 'layout_a' in actual_layout.lower() or actual_layout.lower() == 'a':
        return 1.0 if 'a' in answer_lower or 'normal' in answer_lower or 'standard' in answer_lower else 0.0
    elif 'layout_b' in actual_layout.lower() or actual_layout.lower() == 'b':
        return 1.0 if 'b' in answer_lower or 'inverted' in answer_lower or 'reverse' in answer_lower else 0.0
    else:
        return 0.5


def get_switchlight_queries_v2() -> List[TestQuery]:
    """
    Exploration-dependent test queries for SwitchLight.

    These questions CANNOT be answered from environment description.
    They require actual switch flip observations.
    """
    return [
        # EASY - Direct observation from specific test
        TestQuery(
            question="When the switch was in the ON position, was the light on or off?",
            query_type="interventional",
            expected_answer="[from observation]",
            evaluation_function=lambda ans, gt: _switchlight_state_at_configuration(ans, gt, 'on'),
            difficulty="easy",
            requires_exploration=True
        ),

        TestQuery(
            question="When the switch was in the OFF position, was the light on or off?",
            query_type="interventional",
            expected_answer="[from observation]",
            evaluation_function=lambda ans, gt: _switchlight_state_at_configuration(ans, gt, 'off'),
            difficulty="easy",
            requires_exploration=True
        ),

        TestQuery(
            question="How many times did you flip the switch during this episode?",
            query_type="interventional",
            expected_answer="[count from actions]",
            evaluation_function=lambda ans, gt: 1.0 if extract_number(ans) == gt.get('num_flips', 0) else 0.0,
            difficulty="easy",
            requires_exploration=True
        ),

        # MEDIUM - Requires pattern recognition from multiple observations
        TestQuery(
            question="Based on your observations, what is the wiring layout: A (normal) or B (inverted)?",
            query_type="interventional",
            expected_answer="[inferred from tests]",
            evaluation_function=_switchlight_wiring_from_observations,
            difficulty="medium",
            requires_exploration=True
        ),

        TestQuery(
            question="Out of all the switch positions you tested, how many resulted in the light being ON?",
            query_type="interventional",
            expected_answer="[count from observations]",
            evaluation_function=lambda ans, gt: 1.0 if extract_number(ans) == gt.get('num_on_states', 0) else 0.0,
            difficulty="medium",
            requires_exploration=True
        ),

        TestQuery(
            question="What was the FIRST switch position you tested, and was the light on or off?",
            query_type="interventional",
            expected_answer="[from first observation]",
            evaluation_function=lambda ans, gt: 0.5 if gt.get('first_switch_pos') and gt.get('first_switch_pos') in answer.lower() else 0.0,
            difficulty="medium",
            requires_exploration=True
        ),

        # HARD - Requires understanding of system dynamics
        TestQuery(
            question="Based on your observations, what is the probability that the relay is faulty?",
            query_type="interventional",
            expected_answer="[inferred from consistency of observations]",
            evaluation_function=lambda ans, gt: numeric_tolerance_check(
                ans,
                gt.get('inferred_fault_probability', 0.1),
                tolerance=0.05
            ),
            difficulty="hard",
            requires_exploration=True
        ),

        TestQuery(
            question="If you flip the switch one more time from its current position, what will happen to the light?",
            query_type="counterfactual",
            expected_answer="[predicted from wiring pattern]",
            evaluation_function=lambda ans, gt: 1.0 if gt.get('predicted_next_state', '').lower() in ans.lower() else 0.0,
            difficulty="hard",
            requires_exploration=True
        ),

        TestQuery(
            question="Did you ever observe the light state change when you flipped the switch? Answer yes or no.",
            query_type="interventional",
            expected_answer="[from observations]",
            evaluation_function=lambda ans, gt: 1.0 if (gt.get('observed_state_change', False) and 'yes' in ans.lower()) or (not gt.get('observed_state_change', True) and 'no' in ans.lower()) else 0.0,
            difficulty="medium",
            requires_exploration=True
        ),

        TestQuery(
            question="Did you use the jiggle_relay action during this episode? Answer yes or no.",
            query_type="interventional",
            expected_answer="[from action history]",
            evaluation_function=lambda ans, gt: 1.0 if (gt.get('jiggled_relay', False) and 'yes' in ans.lower()) or (not gt.get('jiggled_relay', False) and 'no' in ans.lower()) else 0.0,
            difficulty="easy",
            requires_exploration=True
        ),
    ]


# ============================================================================
# Main Interface
# ============================================================================

def get_test_queries_v2(environment_name: str) -> List[TestQuery]:
    """
    Get exploration-dependent test query set for environment.

    Args:
        environment_name: Name of environment class

    Returns:
        List of TestQuery objects (all requiring exploration)
    """
    if environment_name == "HotPotLab":
        return get_hotpot_queries_v2()
    elif environment_name == "SwitchLight":
        return get_switchlight_queries_v2()
    else:
        raise ValueError(f"Unknown environment: {environment_name}")


def get_query_statistics_v2(queries: List[TestQuery]) -> Dict[str, Any]:
    """
    Get statistics about query set.

    Returns:
        Dictionary with query statistics
    """
    total = len(queries)

    by_type = {}
    by_difficulty = {}
    requires_exploration_count = 0

    for q in queries:
        by_type[q.query_type] = by_type.get(q.query_type, 0) + 1
        by_difficulty[q.difficulty] = by_difficulty.get(q.difficulty, 0) + 1
        if q.requires_exploration:
            requires_exploration_count += 1

    return {
        'total_queries': total,
        'by_type': by_type,
        'by_difficulty': by_difficulty,
        'requires_exploration': requires_exploration_count,
        'exploration_percentage': 100 * requires_exploration_count / total if total > 0 else 0
    }
