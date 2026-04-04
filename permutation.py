"""
Permutation utilities for multiple choice options.

This module provides tools for permuting options in multiple choice questions
to reduce position bias in LLM predictions.
"""

import random
from dataclasses import dataclass
from itertools import permutations as itertools_permutations
from typing import TypeVar, Sequence, Callable, Awaitable
from forecasting_tools import PredictedOption, PredictedOptionList

T = TypeVar('T')


@dataclass
class Permutation:
    """
    Represents a permutation of indices.

    Attributes:
        forward: Maps original index -> permuted index
        inverse: Maps permuted index -> original index
    """
    forward: list[int]
    inverse: list[int]

    @classmethod
    def from_forward(cls, forward: list[int]) -> "Permutation":
        """Create a Permutation from a forward mapping."""
        n = len(forward)
        inverse = [0] * n
        for i, j in enumerate(forward):
            inverse[j] = i
        return cls(forward=forward, inverse=inverse)

    def apply(self, items: Sequence[T]) -> list[T]:
        """Apply the permutation to a sequence of items."""
        return [items[self.inverse[i]] for i in range(len(items))]

    def is_identity(self) -> bool:
        """Check if this is the identity permutation."""
        return self.forward == list(range(len(self.forward)))

    def is_self_inverse(self) -> bool:
        """Check if applying this permutation twice gives identity."""
        return self.forward == self.inverse

    def __len__(self) -> int:
        return len(self.forward)

    def __repr__(self) -> str:
        return f"Permutation({self.forward})"


class PermutationSets:
    """
    Factory class for generating standard sets of permutations.

    Provides fixed permutation sets designed to reduce position bias
    in multiple choice predictions.
    """

    @staticmethod
    def for_size(n: int) -> list[Permutation]:
        """
        Get a set of permutations for n options designed to reduce position bias.

        Args:
            n: Number of options

        Returns:
            List of Permutation objects
        """
        if n < 2:
            return [Permutation.from_forward(list(range(n)))]

        if n == 2:
            return PermutationSets._for_2()
        if n == 3:
            return PermutationSets._for_3()
        if n == 4:
            return PermutationSets._for_4()
        if n == 5:
            return PermutationSets._for_5()
        if n == 6:
            return PermutationSets._for_6()
        return PermutationSets._for_large(n)

    @staticmethod
    def _for_2() -> list[Permutation]:
        """Each of the 2 permutations repeated 10 times = 20 total."""
        base = [
            Permutation.from_forward([0, 1]),
            Permutation.from_forward([1, 0]),
        ]
        return [p for p in base for _ in range(10)]

    @staticmethod
    def _for_3() -> list[Permutation]:
        """Each of the 6 permutations repeated 3 times = 18 total."""
        base = [
            Permutation.from_forward([0, 1, 2]),
            Permutation.from_forward([2, 1, 0]),
            Permutation.from_forward([1, 2, 0]),
            Permutation.from_forward([2, 0, 1]),
            Permutation.from_forward([0, 2, 1]),
            Permutation.from_forward([1, 0, 2]),
        ]
        return [p for p in base for _ in range(3)]

    @staticmethod
    def _for_4() -> list[Permutation]:
        """All 24 permutations of [0,1,2,3], each used once."""
        return [
            Permutation.from_forward(list(p))
            for p in itertools_permutations(range(4))
        ]

    @staticmethod
    def _for_5() -> list[Permutation]:
        """
        20 permutations defined by i -> i*a + b (mod 5).
        a in {1,2,3,4}, b in {0,1,2,3,4}. Valid because 5 is prime.
        """
        perms = []
        for a in range(1, 5):
            for b in range(5):
                forward = [(i * a + b) % 5 for i in range(5)]
                perms.append(Permutation.from_forward(forward))
        return perms

    @staticmethod
    def _for_6() -> list[Permutation]:
        """
        18 permutations defined by i -> ((i+1)*a mod 7) + b (mod 6).
        a in {1,...,6}, b in {0,2,4}. Valid because 7 is prime.
        """
        perms = []
        for a in range(1, 7):
            for b in (0, 2, 4):
                forward = [((i + 1) * a % 7 + b) % 6 for i in range(6)]
                perms.append(Permutation.from_forward(forward))
        return perms

    @staticmethod
    def _for_large(n: int) -> list[Permutation]:
        """10 random permutations and their reverses = 20 total for n > 6."""
        rng = random.Random(42)
        perms = []
        for _ in range(10):
            forward = list(range(n))
            rng.shuffle(forward)
            perms.append(Permutation.from_forward(forward))
            perms.append(Permutation.from_forward(forward[::-1]))
        return perms


# Convenience function for backwards compatibility
def get_standard_permutations(n: int) -> list[Permutation]:
    """Get standard permutations for n options. See PermutationSets.for_size()."""
    return PermutationSets.for_size(n)


def permute_options(options: list[str], permutation: Permutation) -> list[str]:
    """
    Permute a list of options according to the given permutation.

    Args:
        options: Original list of options
        permutation: The permutation to apply

    Returns:
        Permuted list of options
    """
    if len(options) != len(permutation):
        raise ValueError(f"Options length {len(options)} != permutation length {len(permutation)}")
    return permutation.apply(options)


def unpermute_predictions(
    predictions: PredictedOptionList,
    original_options: list[str],
    permutation: Permutation
) -> PredictedOptionList:
    """
    Unpermute predictions back to the original option order.

    Since predictions are keyed by option_name, we just need to ensure
    they're in the original order.

    Args:
        predictions: Predictions with option names
        original_options: Original option list (for ordering)
        permutation: The permutation that was applied (for reference)

    Returns:
        PredictedOptionList with options in original order
    """
    # Build a map of option_name -> probability
    prob_map = {p.option_name: p.probability for p in predictions.predicted_options}

    # Reconstruct in original order
    reordered = []
    for opt in original_options:
        prob = prob_map.get(opt, 0.0)  # Default to 0 if missing
        reordered.append(PredictedOption(option_name=opt, probability=prob))

    return PredictedOptionList(predicted_options=reordered)


def average_predictions(
    predictions_list: list[PredictedOptionList],
    original_options: list[str]
) -> PredictedOptionList:
    """
    Average multiple predictions into a single prediction.

    Args:
        predictions_list: List of PredictedOptionList to average
        original_options: Original option names (for ordering)

    Returns:
        Averaged PredictedOptionList
    """
    if not predictions_list:
        raise ValueError("Cannot average empty predictions list")

    n_predictions = len(predictions_list)

    # Sum probabilities for each option
    prob_sums: dict[str, float] = {opt: 0.0 for opt in original_options}

    for pred_list in predictions_list:
        for pred in pred_list.predicted_options:
            if pred.option_name in prob_sums:
                prob_sums[pred.option_name] += pred.probability

    # Average the probabilities
    averaged = []
    for opt in original_options:
        avg_prob = prob_sums[opt] / n_predictions
        averaged.append(PredictedOption(option_name=opt, probability=avg_prob))

    return PredictedOptionList(predicted_options=averaged)


async def run_with_permutations(
    predict_fn: Callable[[list[str]], Awaitable[PredictedOptionList]],
    original_options: list[str],
    permutations: list[Permutation] | None = None,
) -> tuple[PredictedOptionList, list[dict]]:
    """
    Run predictions with multiple option orderings to reduce position bias.

    This orchestrator wraps any prediction function and runs it multiple times
    with different option orderings. The results are unpermuted back to the
    original order and averaged.

    The predict_fn is called once per permutation. The caller is responsible
    for ensuring the same research/context is used across all calls.

    Args:
        predict_fn: Async function that takes a list of options (in permuted order)
                   and returns a PredictedOptionList with predictions for those options.
                   The function should use the option names as given.
        original_options: The original option list (defines the canonical order)
        permutations: List of permutations to use. If None, uses get_standard_permutations().

    Returns:
        Tuple of:
        - Averaged predictions in original option order
        - Log of each permutation run (for analysis/debugging)

    Example:
        ```python
        async def my_predict(options: list[str]) -> PredictedOptionList:
            # Your prediction logic here - called once per permutation
            # options are in permuted order
            result = await llm.predict(question_with_options(options))
            return result

        averaged, log = await run_with_permutations(
            my_predict,
            original_options=["Trump", "Biden", "Other"],
        )
        ```
    """
    n = len(original_options)

    if n < 2:
        # No point permuting with fewer than 2 options
        result = await predict_fn(original_options)
        return result, [{"permutation": list(range(n)), "skipped": True}]

    # Use provided permutations or get standard set
    if permutations is None:
        permutations = get_standard_permutations(n)

    all_predictions: list[PredictedOptionList] = []
    log: list[dict] = []

    for perm in permutations:
        # Permute the options
        permuted_options = permute_options(original_options, perm)

        # Run prediction with permuted options
        prediction = await predict_fn(permuted_options)

        # Unpermute the predictions back to original order
        unpermuted = unpermute_predictions(prediction, original_options, perm)

        all_predictions.append(unpermuted)

        # Log for analysis
        log.append({
            "permutation": perm.forward,
            "permuted_options": permuted_options,
            "raw_predictions": {p.option_name: p.probability for p in prediction.predicted_options},
            "unpermuted_predictions": {p.option_name: p.probability for p in unpermuted.predicted_options},
        })

    # Average all predictions
    averaged = average_predictions(all_predictions, original_options)

    return averaged, log
