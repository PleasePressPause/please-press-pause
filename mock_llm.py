"""
Mock forecasting for testing the bot without making real API calls.
Provides mock predictions that match the expected output format.
"""

import logging
from forecasting_tools import (
    ReasonedPrediction,
    NumericDistribution,
    PredictedOptionList,
    PredictedOption,
    Percentile,
    BinaryQuestion,
    NumericQuestion,
    MultipleChoiceQuestion,
    DateQuestion,
    MetaculusQuestion,
)

logger = logging.getLogger(__name__)


def get_mock_binary_prediction() -> ReasonedPrediction[float]:
    """Return a mock binary prediction."""
    reasoning = """
(a) Time left: Approximately 6 months until resolution.
(b) Status quo outcome: Based on current trends, the status quo would suggest No.
(c) No scenario: Current conditions continue unchanged.
(d) Yes scenario: A major unexpected event shifts the outcome.

Rationale: Given the status quo bias and current information, I weight toward the base rate.
However, there's meaningful uncertainty given the time horizon.

Probability: 35%
"""
    return ReasonedPrediction(prediction_value=0.35, reasoning=reasoning)


def get_mock_numeric_prediction(question: NumericQuestion) -> ReasonedPrediction[NumericDistribution]:
    """Return a mock numeric prediction."""
    reasoning = """
(a) Time left: Several months until the outcome is determined.
(b) Outcome if nothing changed: Around 100 based on current levels.
(c) Outcome if trend continued: Slight increase to approximately 110.
(d) Expert expectations: Analysts generally expect values between 80-150.
(e) Low scenario: Economic downturn could push values to 50.
(f) High scenario: Strong positive developments could push to 200.

Rationale: Setting wide confidence intervals to account for unknowns.

Percentile 10: 60
Percentile 20: 75
Percentile 40: 95
Percentile 60: 115
Percentile 80: 140
Percentile 90: 180
"""
    # Create percentiles within question bounds
    lower = question.lower_bound if question.lower_bound is not None else 0
    upper = question.upper_bound if question.upper_bound is not None else 1000
    range_size = upper - lower

    percentiles = [
        Percentile(percentile=0.10, value=lower + 0.1 * range_size),
        Percentile(percentile=0.20, value=lower + 0.2 * range_size),
        Percentile(percentile=0.40, value=lower + 0.4 * range_size),
        Percentile(percentile=0.60, value=lower + 0.6 * range_size),
        Percentile(percentile=0.80, value=lower + 0.75 * range_size),
        Percentile(percentile=0.90, value=lower + 0.85 * range_size),
    ]

    distribution = NumericDistribution.from_question(percentiles, question)
    return ReasonedPrediction(prediction_value=distribution, reasoning=reasoning)


def get_mock_multiple_choice_prediction(question: MultipleChoiceQuestion) -> ReasonedPrediction[PredictedOptionList]:
    """Return a mock multiple choice prediction."""
    reasoning = """
(a) Time left: Several months until resolution.
(b) Status quo outcome: Current trends favor the first option.
(c) Unexpected scenario: External factors could shift probabilities.

Rationale: Weighting status quo while leaving probability mass on alternatives.
"""
    options = question.options
    n = len(options)

    # Distribute probabilities - first option gets most weight
    if n == 2:
        probs = [0.60, 0.40]
    elif n == 3:
        probs = [0.50, 0.30, 0.20]
    elif n == 4:
        probs = [0.40, 0.30, 0.20, 0.10]
    else:
        # Distribute roughly evenly with slight preference for first
        base = 1.0 / n
        probs = [base] * n
        probs[0] += 0.1
        probs[-1] -= 0.1

    predicted_options = [
        PredictedOption(option_name=opt, probability=prob)
        for opt, prob in zip(options, probs)
    ]

    option_list = PredictedOptionList(predicted_options=predicted_options)
    return ReasonedPrediction(prediction_value=option_list, reasoning=reasoning)


def get_mock_date_prediction(question: DateQuestion) -> ReasonedPrediction[NumericDistribution]:
    """Return a mock date prediction."""
    reasoning = """
(a) Time left: The resolution timeframe spans several years.
(b) Outcome if nothing changed: Based on current trajectory, mid-2027.
(c) Outcome if trend continued: Accelerating trends suggest early 2027.
(d) Expert expectations: Most analysts expect 2026-2028 range.
(e) Low scenario (early): Rapid progress could lead to late 2025.
(f) High scenario (late): Delays could push to 2029.

Rationale: Accounting for uncertainty in complex developments.

Percentile 10: 2025-09-15
Percentile 20: 2026-03-01
Percentile 40: 2026-10-15
Percentile 60: 2027-06-01
Percentile 80: 2028-02-15
Percentile 90: 2028-11-01
"""
    # Use question bounds for date range
    lower_ts = question.lower_bound.timestamp()
    upper_ts = question.upper_bound.timestamp()
    range_ts = upper_ts - lower_ts

    percentiles = [
        Percentile(percentile=0.10, value=lower_ts + 0.1 * range_ts),
        Percentile(percentile=0.20, value=lower_ts + 0.2 * range_ts),
        Percentile(percentile=0.40, value=lower_ts + 0.4 * range_ts),
        Percentile(percentile=0.60, value=lower_ts + 0.6 * range_ts),
        Percentile(percentile=0.80, value=lower_ts + 0.75 * range_ts),
        Percentile(percentile=0.90, value=lower_ts + 0.85 * range_ts),
    ]

    distribution = NumericDistribution.from_question(percentiles, question)
    return ReasonedPrediction(prediction_value=distribution, reasoning=reasoning)


def get_mock_research() -> str:
    """Return mock research output."""
    return """
## Research Summary

Based on available information:

1. **Current Status**: The situation remains relatively stable with no major recent developments.

2. **Recent News**: No significant breaking news that would dramatically shift expectations.

3. **Expert Opinion**: Analysts remain divided, with most expecting continuation of current trends.

4. **Key Factors to Watch**:
   - Policy developments
   - Economic indicators
   - Technological progress

5. **Historical Context**: Similar situations in the past have resolved in various ways.

This research provides context for the forecaster's analysis.
"""


# Keep these for backwards compatibility
def get_mock_llms() -> dict:
    """Return a config that will be replaced by MockForecastBot."""
    return {
        "default": "mock/default",
        "summarizer": "mock/summarizer",
        "researcher": "no_research",
        "parser": "mock/parser",
    }


def get_mock_llms_with_research() -> dict:
    """Return a config that will be replaced by MockForecastBot."""
    return {
        "default": "mock/default",
        "summarizer": "mock/summarizer",
        "researcher": "mock/researcher",
        "parser": "mock/parser",
    }


def get_alphabetical_mock_prediction(question: MultipleChoiceQuestion) -> ReasonedPrediction[PredictedOptionList]:
    """
    Return a mock multiple choice prediction based on alphabetical order of option names.

    This assigns probabilities deterministically by option name (not position):
    - First alphabetically: ~40%
    - Second alphabetically: ~30%
    - Third alphabetically: ~20%
    - Fourth alphabetically: ~10%

    This is useful for testing permutation logic - the same option should get
    the same probability regardless of presentation order.
    """
    options = question.options
    n = len(options)

    # Sort options alphabetically to determine their "rank"
    sorted_options = sorted(options)
    rank = {opt: i for i, opt in enumerate(sorted_options)}

    # Base probabilities (for first 4 options by alphabetical rank)
    base_probs = [0.40, 0.30, 0.20, 0.10]

    if n <= 4:
        # Use first n probabilities, normalized
        probs = base_probs[:n]
        total = sum(probs)
        probs = [p / total for p in probs]
    else:
        # First 4 get base probs, rest split the remaining
        remaining = 0.10
        each_remaining = remaining / (n - 4)
        probs = base_probs + [each_remaining] * (n - 4)
        total = sum(probs)
        probs = [p / total for p in probs]

    # Create predictions - assign prob based on alphabetical rank
    predicted_options = []
    for opt in options:  # Keep original order
        r = rank[opt]
        prob = probs[r] if r < len(probs) else probs[-1]
        predicted_options.append(PredictedOption(option_name=opt, probability=prob))

    reasoning = f"Mock prediction based on alphabetical order. Sorted: {sorted_options}"

    return ReasonedPrediction(
        prediction_value=PredictedOptionList(predicted_options=predicted_options),
        reasoning=reasoning
    )
