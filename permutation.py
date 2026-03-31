"""
Permutation utilities for multiple choice options.

This module provides tools for permuting options in multiple choice questions
to reduce position bias in LLM predictions.
"""

import random
from dataclasses import dataclass
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

    @classmethod
    def identity(cls, n: int) -> "Permutation":
        """Create the identity permutation of size n."""
        indices = list(range(n))
        return cls(forward=indices, inverse=indices.copy())

    @classmethod
    def reversed(cls, n: int) -> "Permutation":
        """Create a reversal permutation of size n."""
        forward = list(range(n - 1, -1, -1))
        return cls.from_forward(forward)

    @classmethod
    def random(cls, n: int, seed: int | None = None) -> "Permutation":
        """Create a random permutation of size n."""
        if seed is not None:
            random.seed(seed)
        forward = list(range(n))
        random.shuffle(forward)
        return cls.from_forward(forward)

    @classmethod
    def rotate(cls, n: int, k: int = 1) -> "Permutation":
        """Create a rotation permutation (shift by k positions)."""
        forward = [(i + k) % n for i in range(n)]
        return cls.from_forward(forward)

    @classmethod
    def swap(cls, n: int, i: int, j: int) -> "Permutation":
        """Create a permutation that swaps positions i and j."""
        forward = list(range(n))
        forward[i], forward[j] = forward[j], forward[i]
        return cls.from_forward(forward)

    def apply(self, items: Sequence[T]) -> list[T]:
        """Apply the permutation to a sequence of items."""
        return [items[self.inverse[i]] for i in range(len(items))]

    def unapply(self, items: Sequence[T]) -> list[T]:
        """Reverse the permutation on a sequence of items."""
        return [items[self.forward[i]] for i in range(len(items))]

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


def generate_diverse_permutations(n: int, count: int, seed: int | None = None) -> list[Permutation]:
    """
    Generate a diverse set of permutations for testing.

    Includes:
    - Identity (for baseline comparison)
    - Reversal
    - Random permutations (not self-inverse)

    Args:
        n: Size of permutation
        count: Number of permutations to generate
        seed: Random seed for reproducibility

    Returns:
        List of diverse permutations
    """
    if seed is not None:
        random.seed(seed)

    permutations = []

    # Always include identity first (for baseline)
    permutations.append(Permutation.identity(n))

    if count >= 2:
        # Include reversal
        permutations.append(Permutation.reversed(n))

    if count >= 3 and n >= 3:
        # Add a rotation (not self-inverse for n > 2)
        permutations.append(Permutation.rotate(n, k=1))

    # Fill remaining with random permutations
    attempts = 0
    max_attempts = count * 10
    while len(permutations) < count and attempts < max_attempts:
        attempts += 1
        perm = Permutation.random(n)

        # Try to get permutations that aren't self-inverse (for diversity)
        # But accept them if we're running low on attempts
        if not perm.is_self_inverse() or attempts > max_attempts // 2:
            # Check it's not a duplicate
            is_dup = any(perm.forward == p.forward for p in permutations)
            if not is_dup:
                permutations.append(perm)

    return permutations[:count]


def detect_ordered_options(options: list[str]) -> bool:
    """
    Detect if options appear to be in a meaningful order (like 1, 2, 3... or A, B, C...).

    This helps decide whether permuting might lose information.

    Args:
        options: List of option strings

    Returns:
        True if options appear to be ordered
    """
    if len(options) < 2:
        return False

    # Check for numeric ordering
    try:
        nums = [float(opt.strip().replace(',', '')) for opt in options]
        if nums == sorted(nums) or nums == sorted(nums, reverse=True):
            return True
    except ValueError:
        pass

    # Check for alphabetic single-letter ordering (A, B, C, ...)
    if all(len(opt.strip()) == 1 and opt.strip().isalpha() for opt in options):
        letters = [opt.strip().upper() for opt in options]
        if letters == sorted(letters):
            return True

    # Check for numeric prefixes (1., 2., 3., ... or 1), 2), 3), ...)
    import re
    prefixes = []
    for opt in options:
        match = re.match(r'^(\d+)[.)]\s*', opt)
        if match:
            prefixes.append(int(match.group(1)))
        else:
            break

    if len(prefixes) == len(options) and prefixes == sorted(prefixes):
        return True

    return False


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
    num_permutations: int = 3,
    seed: int | None = None,
) -> tuple[PredictedOptionList, list[dict]]:
    """
    Run predictions with multiple option orderings to reduce position bias.

    This orchestrator wraps any prediction function and runs it multiple times
    with different option orderings. The results are unpermuted back to the
    original order and averaged.

    The hypothesis is that LLMs may assign different probabilities to options
    based on their position in the list. By permuting options across predictions
    and averaging, we can reduce this position bias.

    Args:
        predict_fn: Async function that takes a list of options (in permuted order)
                   and returns a PredictedOptionList with predictions for those options.
                   The function should use the option names as given.
        original_options: The original option list (defines the canonical order)
        num_permutations: Number of different orderings to use (default: 3)
        seed: Random seed for reproducibility (optional)

    Returns:
        Tuple of:
        - Averaged predictions in original option order
        - Log of each permutation run (for analysis/debugging)

    Example:
        ```python
        async def my_predict(options: list[str]) -> PredictedOptionList:
            # Your prediction logic here
            # options are in permuted order
            result = await llm.predict(options)
            return result

        averaged, log = await run_with_permutations(
            my_predict,
            original_options=["Trump", "Biden", "Other"],
            num_permutations=5
        )
        ```
    """
    if len(original_options) < 2:
        # No point permuting with fewer than 2 options
        result = await predict_fn(original_options)
        return result, [{"permutation": list(range(len(original_options))), "skipped": True}]

    # Check if options appear meaningfully ordered (1, 2, 3 or A, B, C)
    if detect_ordered_options(original_options):
        # Don't permute ordered options - it might confuse the model
        result = await predict_fn(original_options)
        return result, [{"permutation": list(range(len(original_options))), "ordered_options": True}]

    # Generate diverse permutations
    permutations = generate_diverse_permutations(
        len(original_options),
        num_permutations,
        seed=seed
    )

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
