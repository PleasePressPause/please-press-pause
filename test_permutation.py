"""
Tests for the permutation module.
"""

import pytest
from forecasting_tools import PredictedOption, PredictedOptionList

from permutation import (
    Permutation,
    permute_options,
    unpermute_predictions,
    generate_diverse_permutations,
    detect_ordered_options,
    average_predictions,
    run_with_permutations,
)


class TestPermutation:
    """Tests for the Permutation class."""

    def test_identity(self):
        """Identity permutation should not change anything."""
        perm = Permutation.identity(4)
        assert perm.forward == [0, 1, 2, 3]
        assert perm.inverse == [0, 1, 2, 3]
        assert perm.is_identity()
        assert perm.is_self_inverse()

    def test_reversed(self):
        """Reversal permutation should reverse the order."""
        perm = Permutation.reversed(4)
        assert perm.forward == [3, 2, 1, 0]
        assert perm.is_self_inverse()  # Reversal is self-inverse
        assert not perm.is_identity()

    def test_rotation(self):
        """Rotation should shift elements."""
        perm = Permutation.rotate(4, k=1)
        items = ['A', 'B', 'C', 'D']
        result = perm.apply(items)
        assert result == ['D', 'A', 'B', 'C']  # Each element shifts right
        assert not perm.is_self_inverse()  # Rotation by 1 is NOT self-inverse

    def test_swap(self):
        """Swap should exchange two elements."""
        perm = Permutation.swap(4, 0, 3)
        items = ['A', 'B', 'C', 'D']
        result = perm.apply(items)
        assert result == ['D', 'B', 'C', 'A']
        assert perm.is_self_inverse()  # Swaps are self-inverse

    def test_apply_and_unapply(self):
        """Applying and then unapplying should give original."""
        perm = Permutation.from_forward([2, 0, 3, 1])
        items = ['A', 'B', 'C', 'D']

        permuted = perm.apply(items)
        restored = perm.unapply(permuted)

        assert restored == items
        assert permuted != items  # Make sure it actually changed something

    def test_random_permutation_is_valid(self):
        """Random permutation should be a valid permutation."""
        perm = Permutation.random(5, seed=42)

        # Should contain each index exactly once
        assert sorted(perm.forward) == [0, 1, 2, 3, 4]
        assert sorted(perm.inverse) == [0, 1, 2, 3, 4]

        # Apply and unapply should restore
        items = ['A', 'B', 'C', 'D', 'E']
        assert perm.unapply(perm.apply(items)) == items

    def test_non_self_inverse_permutation(self):
        """Test a permutation that is NOT its own inverse."""
        # [2, 0, 1] is a 3-cycle: 0->2->1->0
        # This is NOT self-inverse
        perm = Permutation.from_forward([2, 0, 1])

        assert not perm.is_self_inverse()
        assert not perm.is_identity()

        items = ['A', 'B', 'C']
        once = perm.apply(items)
        twice = perm.apply(once)

        assert once == ['B', 'C', 'A']
        assert twice == ['C', 'A', 'B']
        assert twice != items  # Applying twice doesn't give identity


class TestPermuteOptions:
    """Tests for permute_options function."""

    def test_permute_with_reversal(self):
        """Test permuting options with reversal."""
        options = ['Red', 'Blue', 'Green', 'Yellow']
        perm = Permutation.reversed(4)

        result = permute_options(options, perm)

        assert result == ['Yellow', 'Green', 'Blue', 'Red']

    def test_permute_length_mismatch(self):
        """Should raise error if lengths don't match."""
        options = ['A', 'B', 'C']
        perm = Permutation.identity(4)

        with pytest.raises(ValueError):
            permute_options(options, perm)


class TestUnpermutePredictions:
    """Tests for unpermute_predictions function."""

    def test_unpermute_restores_order(self):
        """Predictions should be restored to original option order."""
        original_options = ['Red', 'Blue', 'Green', 'Yellow']
        perm = Permutation.reversed(4)

        # Simulate predictions made with permuted options
        # Permuted order was: Yellow, Green, Blue, Red
        # LLM returned predictions keyed by name
        predictions = PredictedOptionList(predicted_options=[
            PredictedOption(option_name='Yellow', probability=0.4),
            PredictedOption(option_name='Green', probability=0.3),
            PredictedOption(option_name='Blue', probability=0.2),
            PredictedOption(option_name='Red', probability=0.1),
        ])

        result = unpermute_predictions(predictions, original_options, perm)

        # Should be in original order: Red, Blue, Green, Yellow
        assert [p.option_name for p in result.predicted_options] == original_options
        assert result.predicted_options[0].option_name == 'Red'
        assert result.predicted_options[0].probability == pytest.approx(0.1, rel=0.01)
        assert result.predicted_options[3].option_name == 'Yellow'
        assert result.predicted_options[3].probability == pytest.approx(0.4, rel=0.01)

    def test_unpermute_handles_missing_option(self):
        """Missing options should get 0 probability."""
        original_options = ['A', 'B', 'C']
        perm = Permutation.identity(3)

        predictions = PredictedOptionList(predicted_options=[
            PredictedOption(option_name='A', probability=0.6),
            PredictedOption(option_name='C', probability=0.4),
            # B is missing
        ])

        result = unpermute_predictions(predictions, original_options, perm)

        assert result.predicted_options[1].option_name == 'B'
        # Note: PredictedOptionList normalizes, so 0.0 becomes a small value
        assert result.predicted_options[1].probability < 0.02  # Should be close to 0


class TestGenerateDiversePermutations:
    """Tests for generate_diverse_permutations function."""

    def test_includes_identity(self):
        """Should always include identity first."""
        perms = generate_diverse_permutations(4, 3, seed=42)

        assert perms[0].is_identity()

    def test_includes_reversal(self):
        """Should include reversal when count >= 2."""
        perms = generate_diverse_permutations(4, 3, seed=42)

        assert any(p.forward == [3, 2, 1, 0] for p in perms)

    def test_includes_non_self_inverse(self):
        """Should include at least one non-self-inverse permutation."""
        perms = generate_diverse_permutations(4, 5, seed=42)

        non_self_inverse = [p for p in perms if not p.is_self_inverse()]
        assert len(non_self_inverse) >= 1, "Should have at least one non-self-inverse permutation"

    def test_no_duplicates(self):
        """Should not have duplicate permutations."""
        perms = generate_diverse_permutations(4, 5, seed=42)

        forwards = [tuple(p.forward) for p in perms]
        assert len(forwards) == len(set(forwards)), "Should have no duplicates"

    def test_respects_count(self):
        """Should return requested number of permutations."""
        perms = generate_diverse_permutations(4, 3, seed=42)
        assert len(perms) == 3

        perms = generate_diverse_permutations(4, 5, seed=42)
        assert len(perms) == 5


class TestDetectOrderedOptions:
    """Tests for detect_ordered_options function."""

    def test_numeric_ordering(self):
        """Should detect numeric ordering."""
        assert detect_ordered_options(['1', '2', '3', '4'])
        assert detect_ordered_options(['10', '20', '30'])
        assert detect_ordered_options(['1.5', '2.5', '3.5'])

    def test_alphabetic_ordering(self):
        """Should detect single-letter alphabetic ordering."""
        assert detect_ordered_options(['A', 'B', 'C', 'D'])
        assert detect_ordered_options(['a', 'b', 'c'])

    def test_numbered_prefixes(self):
        """Should detect numbered prefixes."""
        assert detect_ordered_options(['1. First', '2. Second', '3. Third'])
        assert detect_ordered_options(['1) Option A', '2) Option B', '3) Option C'])

    def test_unordered_options(self):
        """Should not detect ordering in normal options."""
        assert not detect_ordered_options(['Red', 'Blue', 'Green'])
        assert not detect_ordered_options(['Trump', 'Biden', 'Other'])
        assert not detect_ordered_options(['Yes', 'No'])

    def test_reverse_numeric_ordering(self):
        """Should detect reverse numeric ordering too."""
        assert detect_ordered_options(['3', '2', '1'])

    def test_empty_and_single(self):
        """Should handle edge cases."""
        assert not detect_ordered_options([])
        assert not detect_ordered_options(['Single'])


class TestEndToEndPermutation:
    """End-to-end tests for the full permutation workflow."""

    def test_full_workflow(self):
        """Test the complete permutation workflow."""
        # Original options
        original_options = ['Trump', 'Biden', 'Other', 'Neither']

        # Generate permutations
        perms = generate_diverse_permutations(4, 4, seed=123)

        # Simulate predictions with different orderings
        all_predictions = []

        for perm in perms:
            # Permute options
            permuted_options = permute_options(original_options, perm)

            # Simulate LLM prediction (in real use, this would call the LLM)
            # Here we just create mock predictions with the permuted names
            mock_probs = {'Trump': 0.45, 'Biden': 0.35, 'Other': 0.15, 'Neither': 0.05}
            predictions = PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=mock_probs[opt])
                for opt in permuted_options
            ])

            # Unpermute back to original order
            restored = unpermute_predictions(predictions, original_options, perm)
            all_predictions.append(restored)

        # All restored predictions should have same order
        for pred in all_predictions:
            names = [p.option_name for p in pred.predicted_options]
            assert names == original_options

        # All predictions should have same probabilities (since we used same mock)
        for pred in all_predictions:
            probs = {p.option_name: p.probability for p in pred.predicted_options}
            assert probs['Trump'] == 0.45
            assert probs['Biden'] == 0.35


class TestAveragePredictions:
    """Tests for average_predictions function."""

    def test_average_two_predictions(self):
        """Average of two predictions should give midpoint probabilities."""
        pred1 = PredictedOptionList(predicted_options=[
            PredictedOption(option_name='A', probability=0.6),
            PredictedOption(option_name='B', probability=0.4),
        ])
        pred2 = PredictedOptionList(predicted_options=[
            PredictedOption(option_name='A', probability=0.4),
            PredictedOption(option_name='B', probability=0.6),
        ])

        result = average_predictions([pred1, pred2], ['A', 'B'])

        probs = {p.option_name: p.probability for p in result.predicted_options}
        assert probs['A'] == pytest.approx(0.5, rel=0.01)
        assert probs['B'] == pytest.approx(0.5, rel=0.01)

    def test_average_preserves_order(self):
        """Averaged predictions should be in original option order."""
        pred1 = PredictedOptionList(predicted_options=[
            PredictedOption(option_name='X', probability=0.3),
            PredictedOption(option_name='Y', probability=0.7),
        ])

        result = average_predictions([pred1], ['Y', 'X'])  # Different order

        names = [p.option_name for p in result.predicted_options]
        assert names == ['Y', 'X']

    def test_average_many_predictions(self):
        """Averaging many predictions should reduce variance."""
        predictions = [
            PredictedOptionList(predicted_options=[
                PredictedOption(option_name='A', probability=0.9),
                PredictedOption(option_name='B', probability=0.1),
            ]),
            PredictedOptionList(predicted_options=[
                PredictedOption(option_name='A', probability=0.1),
                PredictedOption(option_name='B', probability=0.9),
            ]),
            PredictedOptionList(predicted_options=[
                PredictedOption(option_name='A', probability=0.5),
                PredictedOption(option_name='B', probability=0.5),
            ]),
        ]

        result = average_predictions(predictions, ['A', 'B'])

        probs = {p.option_name: p.probability for p in result.predicted_options}
        assert probs['A'] == pytest.approx(0.5, rel=0.01)
        assert probs['B'] == pytest.approx(0.5, rel=0.01)


class TestAlphabeticalMockBot:
    """Tests for AlphabeticalMockBot."""

    @pytest.mark.asyncio
    async def test_alphabetical_bot_assigns_by_name(self):
        """Bot should assign consistent probabilities based on alphabetical order."""
        from main import AlphabeticalMockBot
        from mock_questions import get_mock_multiple_choice_question

        bot = AlphabeticalMockBot(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=False,
        )

        # Create a question with options in non-alphabetical order
        question = get_mock_multiple_choice_question()
        # The mock question has ["Red", "Blue", "Green"]
        # Alphabetically: Blue, Green, Red
        # So Blue should get ~40%, Green ~30%, Red ~20%

        result = await bot._run_forecast_on_multiple_choice(question, "test research")

        probs = {p.option_name: p.probability for p in result.prediction_value.predicted_options}

        # Blue is first alphabetically -> highest prob
        # Red is last alphabetically -> lowest prob
        assert probs['Blue'] > probs['Green']
        assert probs['Green'] > probs['Red']

    @pytest.mark.asyncio
    async def test_alphabetical_bot_consistent_across_permutations(self):
        """Same option should get same probability regardless of presentation order."""
        from main import AlphabeticalMockBot
        from forecasting_tools import MultipleChoiceQuestion

        bot = AlphabeticalMockBot(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=False,
        )

        # Create two questions with same options in different orders
        options1 = ["Alpha", "Beta", "Gamma", "Delta"]
        options2 = ["Delta", "Gamma", "Beta", "Alpha"]  # Reversed

        question1 = MultipleChoiceQuestion(
            question_text="Test question",
            id_of_post=1,
            id_of_question=1,
            page_url="https://example.com/1",
            options=options1,
        )
        question2 = MultipleChoiceQuestion(
            question_text="Test question",
            id_of_post=2,
            id_of_question=2,
            page_url="https://example.com/2",
            options=options2,
        )

        result1 = await bot._run_forecast_on_multiple_choice(question1, "")
        result2 = await bot._run_forecast_on_multiple_choice(question2, "")

        probs1 = {p.option_name: p.probability for p in result1.prediction_value.predicted_options}
        probs2 = {p.option_name: p.probability for p in result2.prediction_value.predicted_options}

        # Same option should get same probability regardless of presentation order
        for opt in options1:
            assert probs1[opt] == pytest.approx(probs2[opt], rel=0.01), f"{opt} has different probs"


class TestPermutationWorkflow:
    """Tests for the complete permutation workflow using AlphabeticalMockBot."""

    @pytest.mark.asyncio
    async def test_permute_unpermute_workflow(self):
        """Test that permuting options and unpermuting predictions works correctly."""
        from main import AlphabeticalMockBot
        from forecasting_tools import MultipleChoiceQuestion

        bot = AlphabeticalMockBot(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=False,
        )

        original_options = ["Delta", "Alpha", "Gamma", "Beta"]

        # Generate a permutation
        perm = Permutation.from_forward([2, 0, 3, 1])  # Not self-inverse
        permuted_options = permute_options(original_options, perm)

        # Create question with permuted options
        question = MultipleChoiceQuestion(
            question_text="Test question",
            id_of_post=1,
            id_of_question=1,
            page_url="https://example.com/1",
            options=permuted_options,
        )

        # Get prediction
        result = await bot._run_forecast_on_multiple_choice(question, "")

        # Unpermute the predictions
        unpermuted = unpermute_predictions(result.prediction_value, original_options, perm)

        # The unpermuted predictions should be in original order
        names = [p.option_name for p in unpermuted.predicted_options]
        assert names == original_options

        # And probabilities should match what we'd get with original order
        question_orig = MultipleChoiceQuestion(
            question_text="Test question",
            id_of_post=1,
            id_of_question=1,
            page_url="https://example.com/1",
            options=original_options,
        )
        result_orig = await bot._run_forecast_on_multiple_choice(question_orig, "")

        probs_unpermuted = {p.option_name: p.probability for p in unpermuted.predicted_options}
        probs_orig = {p.option_name: p.probability for p in result_orig.prediction_value.predicted_options}

        for opt in original_options:
            assert probs_unpermuted[opt] == pytest.approx(probs_orig[opt], rel=0.01)


class TestRunWithPermutations:
    """Tests for the run_with_permutations orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_basic(self):
        """Test basic orchestrator functionality."""
        original_options = ["Alpha", "Beta", "Gamma", "Delta"]

        # Create a simple predict function that assigns probs by alphabetical order
        async def mock_predict(options: list[str]) -> PredictedOptionList:
            sorted_opts = sorted(options)
            probs = [0.4, 0.3, 0.2, 0.1]
            rank = {opt: i for i, opt in enumerate(sorted_opts)}
            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=probs[rank[opt]])
                for opt in options
            ])

        result, log = await run_with_permutations(
            mock_predict,
            original_options,
            num_permutations=3,
            seed=42
        )

        # Result should be in original order
        names = [p.option_name for p in result.predicted_options]
        assert names == original_options

        # Should have 3 log entries
        assert len(log) == 3

        # Alpha should have highest prob (first alphabetically)
        probs = {p.option_name: p.probability for p in result.predicted_options}
        assert probs['Alpha'] > probs['Beta']
        assert probs['Beta'] > probs['Gamma']

    @pytest.mark.asyncio
    async def test_orchestrator_with_alphabetical_bot(self):
        """Test orchestrator with AlphabeticalMockBot."""
        from main import AlphabeticalMockBot
        from forecasting_tools import MultipleChoiceQuestion

        bot = AlphabeticalMockBot(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=False,
        )

        original_options = ["Delta", "Alpha", "Gamma", "Beta"]

        # Create a predict function that uses the bot
        async def predict_with_bot(options: list[str]) -> PredictedOptionList:
            question = MultipleChoiceQuestion(
                question_text="Test question",
                id_of_post=1,
                id_of_question=1,
                page_url="https://example.com/1",
                options=options,
            )
            result = await bot._run_forecast_on_multiple_choice(question, "")
            return result.prediction_value

        result, log = await run_with_permutations(
            predict_with_bot,
            original_options,
            num_permutations=5,
            seed=123
        )

        # Result should be in original order
        names = [p.option_name for p in result.predicted_options]
        assert names == original_options

        # Should have 5 log entries
        assert len(log) == 5

        # Since AlphabeticalMockBot assigns by name (not position),
        # all permutations should give the same probabilities,
        # so the average should match a single run
        probs = {p.option_name: p.probability for p in result.predicted_options}
        assert probs['Alpha'] == pytest.approx(0.4, rel=0.05)  # First alphabetically
        assert probs['Beta'] == pytest.approx(0.3, rel=0.05)   # Second
        assert probs['Delta'] == pytest.approx(0.2, rel=0.05)  # Third
        assert probs['Gamma'] == pytest.approx(0.1, rel=0.05)  # Fourth

    @pytest.mark.asyncio
    async def test_orchestrator_skips_ordered_options(self):
        """Orchestrator should skip permutation for ordered options."""
        call_count = 0

        async def counting_predict(options: list[str]) -> PredictedOptionList:
            nonlocal call_count
            call_count += 1
            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=1/len(options))
                for opt in options
            ])

        # Ordered options (1, 2, 3) should not be permuted
        result, log = await run_with_permutations(
            counting_predict,
            ["1", "2", "3"],
            num_permutations=5
        )

        # Should only call once (no permutation)
        assert call_count == 1
        assert log[0].get("ordered_options") is True

    @pytest.mark.asyncio
    async def test_orchestrator_includes_non_self_inverse(self):
        """Orchestrator should use diverse permutations including non-self-inverse."""
        permutations_used = []

        async def tracking_predict(options: list[str]) -> PredictedOptionList:
            permutations_used.append(options.copy())
            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=1/len(options))
                for opt in options
            ])

        # Use non-ordered options (not A, B, C, D which is detected as alphabetic)
        original_options = ["Trump", "Biden", "Other", "Neither"]

        result, log = await run_with_permutations(
            tracking_predict,
            original_options,
            num_permutations=5,
            seed=42
        )

        # Should have 5 different orderings
        assert len(permutations_used) == 5

        # Check that we have diverse permutations (not all the same)
        unique_orderings = set(tuple(p) for p in permutations_used)
        assert len(unique_orderings) == 5

        # At least one should be a rotation (non-self-inverse)
        # The log should show this
        has_rotation = any(
            log_entry["permutation"] == [3, 0, 1, 2]  # rotation by 1
            for log_entry in log
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
