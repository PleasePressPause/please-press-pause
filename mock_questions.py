"""
Mock questions for fully offline testing without any API calls.
"""

from datetime import datetime, timezone, timedelta
from forecasting_tools import BinaryQuestion, NumericQuestion, MultipleChoiceQuestion


def get_mock_binary_question() -> BinaryQuestion:
    """Create a mock binary question for testing."""
    return BinaryQuestion(
        question_text="Will the test pass successfully?",
        id_of_post=99999,
        id_of_question=99999,
        page_url="https://example.com/mock-question",
        background_info="This is a mock question used for offline testing of the forecasting bot.",
        resolution_criteria="Resolves YES if the test passes, NO otherwise.",
        fine_print="This question is not real and is used only for testing purposes.",
        created_at=datetime.now(timezone.utc),
        scheduled_close_time=datetime.now(timezone.utc) + timedelta(days=30),
        scheduled_resolve_time=datetime.now(timezone.utc) + timedelta(days=31),
        open_time=datetime.now(timezone.utc) - timedelta(days=1),
        my_forecasts=None,
    )


def get_mock_numeric_question() -> NumericQuestion:
    """Create a mock numeric question for testing."""
    return NumericQuestion(
        question_text="How many widgets will be produced in 2026?",
        id_of_post=99998,
        id_of_question=99998,
        page_url="https://example.com/mock-numeric-question",
        background_info="This is a mock numeric question for testing.",
        resolution_criteria="Resolves to the number of widgets produced.",
        fine_print="Mock question for testing purposes only.",
        created_at=datetime.now(timezone.utc),
        scheduled_close_time=datetime.now(timezone.utc) + timedelta(days=30),
        scheduled_resolve_time=datetime.now(timezone.utc) + timedelta(days=31),
        open_time=datetime.now(timezone.utc) - timedelta(days=1),
        my_forecasts=None,
        lower_bound=0,
        upper_bound=1000,
        open_lower_bound=False,
        open_upper_bound=True,
        unit_of_measure="widgets",
    )


def get_mock_multiple_choice_question() -> MultipleChoiceQuestion:
    """Create a mock multiple choice question for testing."""
    return MultipleChoiceQuestion(
        question_text="Which color will win the contest?",
        id_of_post=99997,
        id_of_question=99997,
        page_url="https://example.com/mock-mc-question",
        background_info="This is a mock multiple choice question for testing.",
        resolution_criteria="Resolves to the winning color.",
        fine_print="Mock question for testing purposes only.",
        created_at=datetime.now(timezone.utc),
        scheduled_close_time=datetime.now(timezone.utc) + timedelta(days=30),
        scheduled_resolve_time=datetime.now(timezone.utc) + timedelta(days=31),
        open_time=datetime.now(timezone.utc) - timedelta(days=1),
        my_forecasts=None,
        options=["Red", "Blue", "Green"],
    )


def get_all_mock_questions() -> list:
    """Get all mock questions for comprehensive testing."""
    return [
        get_mock_binary_question(),
        get_mock_numeric_question(),
        get_mock_multiple_choice_question(),
    ]
