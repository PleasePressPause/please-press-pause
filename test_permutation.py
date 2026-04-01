"""
Tests for the permutation module.
"""

import pytest
from forecasting_tools import PredictedOption, PredictedOptionList, MultipleChoiceQuestion

from permutation import (
    Permutation,
    PermutationSets,
    permute_options,
    unpermute_predictions,
    get_standard_permutations,
    average_predictions,
    run_with_permutations,
)
from mock_llm import get_alphabetical_mock_prediction


class TestPermutation:
    """Tests for the Permutation class."""

    def test_from_forward(self):
        """from_forward should create valid permutation with inverse."""
        perm = Permutation.from_forward([2, 0, 3, 1])
        assert perm.forward == [2, 0, 3, 1]
        # inverse[j] = i where forward[i] = j
        # forward[0] = 2, so inverse[2] = 0
        # forward[1] = 0, so inverse[0] = 1
        # forward[2] = 3, so inverse[3] = 2
        # forward[3] = 1, so inverse[1] = 3
        assert perm.inverse == [1, 3, 0, 2]

    def test_apply(self):
        """Apply should reorder items according to permutation."""
        perm = Permutation.from_forward([2, 0, 1])
        items = ['A', 'B', 'C']
        result = perm.apply(items)
        # apply uses inverse: result[i] = items[inverse[i]]
        # forward = [2, 0, 1] -> inverse = [1, 2, 0]
        # result[0] = items[1] = 'B'
        # result[1] = items[2] = 'C'
        # result[2] = items[0] = 'A'
        assert result == ['B', 'C', 'A']

    def test_is_identity(self):
        """is_identity should detect identity permutation."""
        identity = Permutation.from_forward([0, 1, 2, 3])
        assert identity.is_identity()

        non_identity = Permutation.from_forward([1, 0, 2, 3])
        assert not non_identity.is_identity()

    def test_is_self_inverse(self):
        """is_self_inverse should detect self-inverse permutations."""
        # Reversal is self-inverse
        reversal = Permutation.from_forward([3, 2, 1, 0])
        assert reversal.is_self_inverse()

        # Single swap is self-inverse
        swap = Permutation.from_forward([1, 0, 2, 3])
        assert swap.is_self_inverse()

        # 3-cycle is NOT self-inverse
        cycle = Permutation.from_forward([1, 2, 0])
        assert not cycle.is_self_inverse()

    def test_len(self):
        """len should return number of elements."""
        perm = Permutation.from_forward([2, 0, 1])
        assert len(perm) == 3

    def test_non_self_inverse_permutation(self):
        """Test a permutation that is NOT its own inverse."""
        # [1, 2, 0] is a 3-cycle: 0->1->2->0
        perm = Permutation.from_forward([1, 2, 0])

        assert not perm.is_self_inverse()
        assert not perm.is_identity()

        items = ['A', 'B', 'C']
        once = perm.apply(items)
        twice = perm.apply(once)

        assert once == ['C', 'A', 'B']
        assert twice == ['B', 'C', 'A']
        assert twice != items  # Applying twice doesn't give identity


class TestPermutationSets:
    """Tests for PermutationSets class."""

    def test_for_size_n_equals_1(self):
        """Single option should return identity."""
        perms = PermutationSets.for_size(1)
        assert len(perms) == 1
        assert perms[0].forward == [0]

    def test_for_size_n_equals_2(self):
        """Two options should return identity and reversal."""
        perms = PermutationSets.for_size(2)
        assert len(perms) == 2
        assert perms[0].forward == [0, 1]  # identity
        assert perms[1].forward == [1, 0]  # reversal

    def test_for_size_n_equals_3(self):
        """Three options should return all 6 permutations."""
        perms = PermutationSets.for_size(3)
        assert len(perms) == 6

        # Should include identity first
        assert perms[0].is_identity()

        # Should include reversal
        assert any(p.forward == [2, 1, 0] for p in perms)

        # Should include non-self-inverse permutations (rotations)
        non_self_inverse = [p for p in perms if not p.is_self_inverse()]
        assert len(non_self_inverse) >= 2

        # All should be unique
        forwards = [tuple(p.forward) for p in perms]
        assert len(set(forwards)) == 6

    def test_for_size_n_equals_4(self):
        """Four options should return the specified fixed set."""
        perms = PermutationSets.for_size(4)
        assert len(perms) == 4

        # Check specific permutations as specified by reviewer
        # (1,2,3,4), (2,4,1,3), (3,1,4,2), (4,3,2,1) in 1-indexed
        # (0,1,2,3), (1,3,0,2), (2,0,3,1), (3,2,1,0) in 0-indexed
        expected = [
            [0, 1, 2, 3],  # identity
            [1, 3, 0, 2],  # not self-inverse
            [2, 0, 3, 1],  # not self-inverse
            [3, 2, 1, 0],  # reversal
        ]
        for i, perm in enumerate(perms):
            assert perm.forward == expected[i]

        # Should include non-self-inverse permutations
        non_self_inverse = [p for p in perms if not p.is_self_inverse()]
        assert len(non_self_inverse) >= 1

    def test_for_size_n_greater_than_4(self):
        """n > 4 should return identity, reversal, and rotations."""
        perms = PermutationSets.for_size(5)
        assert len(perms) == 4

        # First should be identity
        assert perms[0].is_identity()

        # Second should be reversal
        assert perms[1].forward == [4, 3, 2, 1, 0]

        # Should include rotations (non-self-inverse)
        non_self_inverse = [p for p in perms if not p.is_self_inverse()]
        assert len(non_self_inverse) >= 1

    def test_convenience_function_matches_class(self):
        """get_standard_permutations should match PermutationSets.for_size."""
        for n in [1, 2, 3, 4, 5]:
            from_function = get_standard_permutations(n)
            from_class = PermutationSets.for_size(n)
            assert len(from_function) == len(from_class)
            for i in range(len(from_function)):
                assert from_function[i].forward == from_class[i].forward


class TestPermuteOptions:
    """Tests for permute_options function."""

    def test_permute_with_reversal(self):
        """Test permuting options with reversal."""
        options = ['Red', 'Blue', 'Green', 'Yellow']
        perm = Permutation.from_forward([3, 2, 1, 0])  # reversal

        result = permute_options(options, perm)

        assert result == ['Yellow', 'Green', 'Blue', 'Red']

    def test_permute_length_mismatch(self):
        """Should raise error if lengths don't match."""
        options = ['A', 'B', 'C']
        perm = Permutation.from_forward([0, 1, 2, 3])

        with pytest.raises(ValueError):
            permute_options(options, perm)


class TestUnpermutePredictions:
    """Tests for unpermute_predictions function."""

    def test_unpermute_restores_order(self):
        """Predictions should be restored to original option order."""
        original_options = ['Red', 'Blue', 'Green', 'Yellow']
        perm = Permutation.from_forward([3, 2, 1, 0])  # reversal

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
        perm = Permutation.from_forward([0, 1, 2])  # identity

        predictions = PredictedOptionList(predicted_options=[
            PredictedOption(option_name='A', probability=0.6),
            PredictedOption(option_name='C', probability=0.4),
            # B is missing
        ])

        result = unpermute_predictions(predictions, original_options, perm)

        assert result.predicted_options[1].option_name == 'B'
        # Note: PredictedOptionList normalizes, so 0.0 becomes a small value
        assert result.predicted_options[1].probability < 0.02  # Should be close to 0


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


class TestAlphabeticalMockPrediction:
    """Tests for the alphabetical mock prediction function."""

    def test_assigns_by_name(self):
        """Should assign consistent probabilities based on alphabetical order."""
        question = MultipleChoiceQuestion(
            question_text="Test question",
            id_of_post=1,
            id_of_question=1,
            page_url="https://example.com/1",
            options=["Red", "Blue", "Green"],
        )
        # Alphabetically: Blue, Green, Red
        # So Blue should get ~40%, Green ~30%, Red ~20%

        result = get_alphabetical_mock_prediction(question)

        probs = {p.option_name: p.probability for p in result.prediction_value.predicted_options}

        # Blue is first alphabetically -> highest prob
        # Red is last alphabetically -> lowest prob
        assert probs['Blue'] > probs['Green']
        assert probs['Green'] > probs['Red']

    def test_consistent_across_option_orders(self):
        """Same option should get same probability regardless of presentation order."""
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

        result1 = get_alphabetical_mock_prediction(question1)
        result2 = get_alphabetical_mock_prediction(question2)

        probs1 = {p.option_name: p.probability for p in result1.prediction_value.predicted_options}
        probs2 = {p.option_name: p.probability for p in result2.prediction_value.predicted_options}

        # Same option should get same probability regardless of presentation order
        for opt in options1:
            assert probs1[opt] == pytest.approx(probs2[opt], rel=0.01), f"{opt} has different probs"


class TestEndToEndPermutation:
    """End-to-end tests for the full permutation workflow."""

    def test_full_workflow(self):
        """Test the complete permutation workflow."""
        # Original options
        original_options = ['Trump', 'Biden', 'Other', 'Neither']

        # Get standard permutations for 4 options
        perms = PermutationSets.for_size(4)

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
        )

        # Result should be in original order
        names = [p.option_name for p in result.predicted_options]
        assert names == original_options

        # Should have 4 log entries (standard permutations for n=4)
        assert len(log) == 4

        # Alpha should have highest prob (first alphabetically)
        probs = {p.option_name: p.probability for p in result.predicted_options}
        assert probs['Alpha'] > probs['Beta']
        assert probs['Beta'] > probs['Gamma']

    @pytest.mark.asyncio
    async def test_orchestrator_with_custom_permutations(self):
        """Test orchestrator with custom permutation list."""
        original_options = ["A", "B", "C"]

        call_count = 0

        async def counting_predict(options: list[str]) -> PredictedOptionList:
            nonlocal call_count
            call_count += 1
            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=1/len(options))
                for opt in options
            ])

        # Use only 2 custom permutations
        custom_perms = [
            Permutation.from_forward([0, 1, 2]),  # identity
            Permutation.from_forward([2, 1, 0]),  # reversal
        ]

        result, log = await run_with_permutations(
            counting_predict,
            original_options,
            permutations=custom_perms,
        )

        # Should have called predict function twice
        assert call_count == 2
        assert len(log) == 2

    @pytest.mark.asyncio
    async def test_orchestrator_single_option(self):
        """Orchestrator should skip permutation for single option."""
        call_count = 0

        async def counting_predict(options: list[str]) -> PredictedOptionList:
            nonlocal call_count
            call_count += 1
            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=1.0)
                for opt in options
            ])

        result, log = await run_with_permutations(
            counting_predict,
            ["Only"],
        )

        # Should only call once (no point permuting single option)
        assert call_count == 1
        assert log[0].get("skipped") is True

    @pytest.mark.asyncio
    async def test_orchestrator_uses_diverse_permutations(self):
        """Orchestrator should use diverse permutations including non-self-inverse."""
        permutations_used = []

        async def tracking_predict(options: list[str]) -> PredictedOptionList:
            permutations_used.append(options.copy())
            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=1/len(options))
                for opt in options
            ])

        original_options = ["Trump", "Biden", "Other", "Neither"]

        result, log = await run_with_permutations(
            tracking_predict,
            original_options,
        )

        # Should have 4 different orderings (standard permutations for n=4)
        assert len(permutations_used) == 4

        # Check that we have diverse permutations (not all the same)
        unique_orderings = set(tuple(p) for p in permutations_used)
        assert len(unique_orderings) == 4

        # Should include identity (first)
        assert permutations_used[0] == original_options

    @pytest.mark.asyncio
    async def test_orchestrator_averages_position_bias(self):
        """Orchestrator should average out position bias."""
        # Simulate a predictor with position bias (always favors first option)
        async def biased_predict(options: list[str]) -> PredictedOptionList:
            probs = [0.7, 0.2, 0.05, 0.05]  # Heavy bias to first position
            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=probs[i])
                for i, opt in enumerate(options)
            ])

        original_options = ["A", "B", "C", "D"]

        result, log = await run_with_permutations(
            biased_predict,
            original_options,
        )

        probs = {p.option_name: p.probability for p in result.predicted_options}

        # After averaging across permutations, probabilities should be more balanced
        # Not testing exact values, just that no option dominates
        assert max(probs.values()) < 0.6  # Should be less extreme than 0.7


class TestPermutationWorkflow:
    """Tests for the complete permutation workflow."""

    def test_permute_unpermute_workflow(self):
        """Test that permuting options and unpermuting predictions works correctly."""
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

        # Get prediction using alphabetical mock
        result = get_alphabetical_mock_prediction(question)

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
        result_orig = get_alphabetical_mock_prediction(question_orig)

        probs_unpermuted = {p.option_name: p.probability for p in unpermuted.predicted_options}
        probs_orig = {p.option_name: p.probability for p in result_orig.prediction_value.predicted_options}

        for opt in original_options:
            assert probs_unpermuted[opt] == pytest.approx(probs_orig[opt], rel=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
