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
        """Two options: each of 2 permutations repeated 10 times = 20."""
        perms = PermutationSets.for_size(2)
        assert len(perms) == 20

        # All should be valid permutations of [0, 1]
        for p in perms:
            assert sorted(p.forward) == [0, 1]

        # Should have 10 copies of identity and 10 copies of reversal
        identity_count = sum(1 for p in perms if p.forward == [0, 1])
        reversal_count = sum(1 for p in perms if p.forward == [1, 0])
        assert identity_count == 10
        assert reversal_count == 10

    def test_for_size_n_equals_3(self):
        """Three options: each of 6 permutations repeated 3 times = 18."""
        perms = PermutationSets.for_size(3)
        assert len(perms) == 18

        # All should be valid permutations of [0, 1, 2]
        for p in perms:
            assert sorted(p.forward) == [0, 1, 2]

        # Should have 6 unique permutations, each appearing 3 times
        forwards = [tuple(p.forward) for p in perms]
        assert len(set(forwards)) == 6
        for fwd in set(forwards):
            assert forwards.count(fwd) == 3

    def test_for_size_n_equals_4(self):
        """Four options: all 24 permutations, each used once."""
        perms = PermutationSets.for_size(4)
        assert len(perms) == 24

        # All should be unique
        forwards = [tuple(p.forward) for p in perms]
        assert len(set(forwards)) == 24

        # All should be valid permutations of [0,1,2,3]
        for p in perms:
            assert sorted(p.forward) == [0, 1, 2, 3]

    def test_for_size_n_equals_5(self):
        """Five options: 20 permutations from i -> i*a+b (mod 5)."""
        perms = PermutationSets.for_size(5)
        assert len(perms) == 20

        # All should be valid permutations of [0,1,2,3,4]
        for p in perms:
            assert sorted(p.forward) == [0, 1, 2, 3, 4]

        # All should be unique
        forwards = [tuple(p.forward) for p in perms]
        assert len(set(forwards)) == 20

        # Verify the formula: i -> i*a+b (mod 5) for a=1..4, b=0..4
        idx = 0
        for a in range(1, 5):
            for b in range(5):
                expected = [(i * a + b) % 5 for i in range(5)]
                assert perms[idx].forward == expected
                idx += 1

    def test_for_size_n_equals_6(self):
        """Six options: 18 permutations from i -> ((i+1)*a mod 7 + b) mod 6."""
        perms = PermutationSets.for_size(6)
        assert len(perms) == 18

        # All should be valid permutations of [0,1,2,3,4,5]
        for p in perms:
            assert sorted(p.forward) == [0, 1, 2, 3, 4, 5]

        # All should be unique
        forwards = [tuple(p.forward) for p in perms]
        assert len(set(forwards)) == 18

        # Verify the formula
        idx = 0
        for a in range(1, 7):
            for b in (0, 2, 4):
                expected = [((i + 1) * a % 7 + b) % 6 for i in range(6)]
                assert perms[idx].forward == expected
                idx += 1

    def test_for_size_n_greater_than_6(self):
        """n > 6: 10 random permutations and their reverses = 20."""
        perms = PermutationSets.for_size(8)
        assert len(perms) == 20

        # All should be valid permutations of [0,...,7]
        for p in perms:
            assert sorted(p.forward) == list(range(8))

        # Every even-indexed perm should have its reverse at the next odd index
        for i in range(0, 20, 2):
            assert perms[i + 1].forward == perms[i].forward[::-1]

        # Should be deterministic (seeded RNG)
        perms2 = PermutationSets.for_size(8)
        for i in range(20):
            assert perms[i].forward == perms2[i].forward

    def test_convenience_function_matches_class(self):
        """get_standard_permutations should match PermutationSets.for_size."""
        for n in [1, 2, 3, 4, 5, 6, 7]:
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

        # Should have 24 log entries (all permutations for n=4)
        assert len(log) == 24

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

        # Should have 24 different orderings (all permutations for n=4)
        assert len(permutations_used) == 24

        # Check that we have diverse permutations (not all the same)
        unique_orderings = set(tuple(p) for p in permutations_used)
        assert len(unique_orderings) == 24

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


class TestIntegrationPermutationWorkflow:
    """
    Integration tests for the complete permutation workflow.

    These tests verify that:
    - The correct number of prediction calls are made for each n
    - The final averaged result is correct
    - The workflow handles different option counts correctly
    """

    @pytest.mark.asyncio
    async def test_three_options_eighteen_calls(self):
        """For 3 options, should make 18 prediction calls (6 permutations x 3)."""
        original_options = ["Red", "Blue", "Green"]
        call_count = 0
        options_seen = []

        async def tracking_predict(options: list[str]) -> PredictedOptionList:
            nonlocal call_count
            call_count += 1
            options_seen.append(options.copy())

            # Use alphabetical mock logic inline
            sorted_opts = sorted(options)
            rank = {opt: i for i, opt in enumerate(sorted_opts)}
            base_probs = [0.40, 0.30, 0.20]
            total = sum(base_probs)
            probs = [p / total for p in base_probs]

            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=probs[rank[opt]])
                for opt in options
            ])

        result, log = await run_with_permutations(
            tracking_predict,
            original_options,
        )

        # Should make exactly 18 calls (6 permutations x 3 repetitions)
        assert call_count == 18
        assert len(log) == 18

        # All 6 unique orderings should appear
        unique_orderings = set(tuple(opts) for opts in options_seen)
        assert len(unique_orderings) == 6

        # Result should be in original order
        names = [p.option_name for p in result.predicted_options]
        assert names == original_options

        # Since alphabetical mock assigns by name (not position),
        # all permutations give same probabilities, so average equals single run
        probs = {p.option_name: p.probability for p in result.predicted_options}
        assert probs['Blue'] == pytest.approx(0.444, rel=0.01)
        assert probs['Green'] == pytest.approx(0.333, rel=0.01)
        assert probs['Red'] == pytest.approx(0.222, rel=0.01)

    @pytest.mark.asyncio
    async def test_four_options_twentyfour_calls(self):
        """For 4 options, should make 24 prediction calls (all permutations)."""
        original_options = ["Alpha", "Beta", "Gamma", "Delta"]
        call_count = 0

        async def counting_predict(options: list[str]) -> PredictedOptionList:
            nonlocal call_count
            call_count += 1

            # Alphabetical mock logic
            sorted_opts = sorted(options)
            rank = {opt: i for i, opt in enumerate(sorted_opts)}
            probs = [0.40, 0.30, 0.20, 0.10]

            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=probs[rank[opt]])
                for opt in options
            ])

        result, log = await run_with_permutations(
            counting_predict,
            original_options,
        )

        # Should make exactly 24 calls (all permutations of 4)
        assert call_count == 24
        assert len(log) == 24

        # Verify expected probabilities (alphabetically: Alpha, Beta, Delta, Gamma)
        probs = {p.option_name: p.probability for p in result.predicted_options}
        assert probs['Alpha'] == pytest.approx(0.40, rel=0.01)
        assert probs['Beta'] == pytest.approx(0.30, rel=0.01)
        assert probs['Delta'] == pytest.approx(0.20, rel=0.01)
        assert probs['Gamma'] == pytest.approx(0.10, rel=0.01)

    @pytest.mark.asyncio
    async def test_two_options_twenty_calls(self):
        """For 2 options, should make 20 prediction calls (2 permutations x 10)."""
        original_options = ["Yes", "No"]
        call_count = 0
        options_seen = []

        async def tracking_predict(options: list[str]) -> PredictedOptionList:
            nonlocal call_count
            call_count += 1
            options_seen.append(options.copy())

            # Alphabetical: No, Yes -> No gets 0.571, Yes gets 0.429
            sorted_opts = sorted(options)
            rank = {opt: i for i, opt in enumerate(sorted_opts)}
            base_probs = [0.40, 0.30]
            total = sum(base_probs)
            probs = [p / total for p in base_probs]

            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=probs[rank[opt]])
                for opt in options
            ])

        result, log = await run_with_permutations(
            tracking_predict,
            original_options,
        )

        # Should make exactly 20 calls (2 permutations x 10 repetitions)
        assert call_count == 20
        assert len(log) == 20

        # Should see both orderings
        assert ["Yes", "No"] in options_seen
        assert ["No", "Yes"] in options_seen

        # Verify expected probabilities
        probs = {p.option_name: p.probability for p in result.predicted_options}
        assert probs['No'] == pytest.approx(0.571, rel=0.01)
        assert probs['Yes'] == pytest.approx(0.429, rel=0.01)

    @pytest.mark.asyncio
    async def test_position_bias_correction(self):
        """
        Test that permutation averaging corrects for position bias.

        Uses a predictor with strong position bias (first option always gets 80%)
        and verifies that averaging across permutations balances the probabilities.
        """
        original_options = ["A", "B", "C"]

        async def biased_predict(options: list[str]) -> PredictedOptionList:
            # Strong position bias: first option gets 80%
            probs = [0.80, 0.15, 0.05]
            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=probs[i])
                for i, opt in enumerate(options)
            ])

        result, log = await run_with_permutations(
            biased_predict,
            original_options,
        )

        # With 18 calls (6 unique permutations x 3 reps) and position bias:
        # Each option appears in each position twice per 6 unique permutations
        # So each option gets: (0.80 + 0.80 + 0.15 + 0.15 + 0.05 + 0.05) / 6 = 0.333
        probs = {p.option_name: p.probability for p in result.predicted_options}

        # All options should be approximately equal after averaging
        assert probs['A'] == pytest.approx(0.333, rel=0.02)
        assert probs['B'] == pytest.approx(0.333, rel=0.02)
        assert probs['C'] == pytest.approx(0.333, rel=0.02)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
