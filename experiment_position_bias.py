"""
Position bias experiment for multiple choice LLM predictions.

Runs predictions on a single MC question across all option orderings,
with multiple repetitions per ordering, to measure how option order
affects LLM predictions.

Usage:
    # Offline dry run (no API calls):
    python experiment_position_bias.py --offline

    # Real experiment on default question:
    python experiment_position_bias.py --reps 5

    # Reuse saved research with a different model:
    python experiment_position_bias.py --research-file experiment_data/research_22427_*.json --model anthropic/claude-sonnet-4-20250514

    # Append more reps to existing results:
    python experiment_position_bias.py --append-to experiment_data/results_22427_*.json --reps 3
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import dotenv

from forecasting_tools import (
    GeneralLlm,
    MetaculusClient,
    MultipleChoiceQuestion,
)

from main import SpringTemplateBot2026, MockSpringTemplateBot
from mock_llm import get_mock_research
from mock_questions import get_mock_multiple_choice_question
from permutation import (
    PermutationSets,
    permute_options,
    unpermute_predictions,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_QUESTION_URL = "https://www.metaculus.com/questions/34484/us-congress-control-after-2026-midterms/"


# --- I/O helpers ---

def find_existing_research(question_id: int, output_dir: str) -> str | None:
    """Find an existing research file for the given question ID."""
    filepath = os.path.join(output_dir, f"research_{question_id}.json")
    if os.path.exists(filepath):
        return filepath
    return None


def save_research(question: MultipleChoiceQuestion, research: str, researcher_config: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"research_{question.id_of_post}.json"
    filepath = os.path.join(output_dir, filename)

    data = {
        "question_url": question.page_url,
        "question_id": question.id_of_post,
        "question_text": question.question_text,
        "options": question.options,
        "background_info": question.background_info or "",
        "resolution_criteria": question.resolution_criteria or "",
        "fine_print": question.fine_print or "",
        "research": research,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_config": {"researcher": researcher_config},
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved research to {filepath}")
    return filepath


def load_research(filepath: str) -> tuple[MultipleChoiceQuestion, str]:
    with open(filepath) as f:
        data = json.load(f)

    question = MultipleChoiceQuestion(
        question_text=data["question_text"],
        id_of_post=data["question_id"],
        id_of_question=data["question_id"],
        page_url=data["question_url"],
        background_info=data.get("background_info", ""),
        resolution_criteria=data.get("resolution_criteria", ""),
        fine_print=data.get("fine_print", ""),
        options=data["options"],
        created_at=datetime.now(timezone.utc),
        scheduled_close_time=datetime.now(timezone.utc),
        scheduled_resolve_time=datetime.now(timezone.utc),
        open_time=datetime.now(timezone.utc),
        my_forecasts=None,
    )

    return question, data["research"]


def save_results(results: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    question_id = results["question_id"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"results_{question_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {filepath}")
    return filepath


def load_results(filepath: str) -> dict:
    with open(filepath) as f:
        return json.load(f)


# --- Summary printing ---

def print_summary(results: dict) -> None:
    options = results["options"]
    predictions = results["predictions"]

    # Group by permutation
    by_perm: dict[str, list[dict]] = {}
    for pred in predictions:
        key = str(pred["permutation"])
        by_perm.setdefault(key, []).append(pred)

    print("\n" + "=" * 70)
    print("POSITION BIAS EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Question: {results['question_text'][:80]}...")
    print(f"Options (canonical order): {options}")
    print(f"Model: {results['model']}")
    print(f"Temperature: {results['temperature']}")
    print(f"Total predictions: {len(predictions)}")
    print()

    # Per-ordering summary
    print("Per-ordering mean probabilities (canonical order):")
    print("-" * 70)

    header = f"{'Permutation':<25}"
    for opt in options:
        header += f" {opt[:12]:>12}"
    header += f" {'N':>5}"
    print(header)
    print("-" * 70)

    for perm_key, preds in sorted(by_perm.items()):
        n = len(preds)
        means = {}
        for opt in options:
            values = [p["predictions_original_order"][opt] for p in preds]
            means[opt] = sum(values) / len(values)

        row = f"{perm_key:<25}"
        for opt in options:
            row += f" {means[opt]:>11.1%}"
        row += f" {n:>5}"
        print(row)

    print("=" * 70)
    print()


# --- Main experiment ---

async def run_experiment(
    question: MultipleChoiceQuestion,
    research: str,
    bot: SpringTemplateBot2026,
    reps: int,
    existing_predictions: list[dict] | None = None,
) -> list[dict]:
    """Run the prediction experiment and return raw prediction records."""
    options = question.options
    permutations = PermutationSets.for_size(len(options))

    predictions = list(existing_predictions) if existing_predictions else []

    # Count existing reps per permutation to continue numbering
    existing_rep_counts: dict[str, int] = {}
    for pred in predictions:
        key = str(pred["permutation"])
        existing_rep_counts[key] = existing_rep_counts.get(key, 0) + 1

    total_calls = len(permutations) * reps
    print(f"Running {len(permutations)} permutations x {reps} reps = {total_calls} LLM calls")

    call_num = 0
    for perm in permutations:
        perm_key = str(perm.forward)
        start_rep = existing_rep_counts.get(perm_key, 0)
        permuted_options = permute_options(options, perm)

        for rep in range(reps):
            call_num += 1
            rep_num = start_rep + rep
            print(f"  [{call_num}/{total_calls}] Permutation {perm.forward}, rep {rep_num}...")

            # Create question copy with permuted options
            permuted_question = question.model_copy(update={"options": permuted_options})

            # Run the prediction
            result = await bot._run_forecast_on_multiple_choice(permuted_question, research)
            predicted_option_list = result.prediction_value

            # Record in presented order
            predictions_presented = {
                p.option_name: p.probability
                for p in predicted_option_list.predicted_options
            }

            # Unpermute back to canonical order
            unpermuted = unpermute_predictions(predicted_option_list, options, perm)
            predictions_original = {
                p.option_name: p.probability
                for p in unpermuted.predicted_options
            }

            predictions.append({
                "permutation": perm.forward,
                "permuted_options": permuted_options,
                "rep": rep_num,
                "predictions_original_order": predictions_original,
                "predictions_presented_order": predictions_presented,
                "reasoning": result.reasoning,
            })

    return predictions


async def main():
    parser = argparse.ArgumentParser(description="Position bias experiment for MC predictions")
    parser.add_argument("--question-url", default=DEFAULT_QUESTION_URL, help="Metaculus question URL")
    parser.add_argument("--research-file", help="Path to saved research JSON (skip research phase)")
    parser.add_argument("--append-to", help="Path to existing results JSON (append more reps)")
    parser.add_argument("--reps", type=int, default=5, help="Repetitions per ordering (default: 5)")
    parser.add_argument("--model", default="openrouter/openai/gpt-4o-mini", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--output-dir", default="experiment_data", help="Output directory")
    parser.add_argument("--offline", action="store_true", help="Use mock question + mock predictions (no API calls)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- Resolve question and research ---
    research_file_path = None
    existing_predictions = None

    if args.append_to:
        # Load existing results and continue
        existing_results = load_results(args.append_to)
        existing_predictions = existing_results["predictions"]
        research_file_path = existing_results.get("research_file")
        if research_file_path and os.path.exists(research_file_path):
            question, research = load_research(research_file_path)
        else:
            raise ValueError(f"Cannot find research file referenced in results: {research_file_path}")
        print(f"Loaded {len(existing_predictions)} existing predictions from {args.append_to}")

    elif args.research_file:
        question, research = load_research(args.research_file)
        research_file_path = args.research_file
        print(f"Loaded research from {args.research_file}")

    elif args.offline:
        question = get_mock_multiple_choice_question()
        # Reuse existing research if available
        existing = find_existing_research(question.id_of_post, args.output_dir)
        if existing:
            _, research = load_research(existing)
            research_file_path = existing
            print(f"Reusing existing research from {existing}")
        else:
            research = get_mock_research()
            research_file_path = save_research(question, research, "mock", args.output_dir)
        print(f"Using mock question: {question.question_text}")

    else:
        # Fetch real question and run research
        print(f"Fetching question from {args.question_url}...")
        client = MetaculusClient()
        question = client.get_question_by_url(args.question_url)
        if not isinstance(question, MultipleChoiceQuestion):
            raise ValueError(f"Expected MultipleChoiceQuestion, got {type(question).__name__}")

        print(f"Question: {question.question_text}")
        print(f"Options: {question.options}")

        # Reuse existing research if available
        existing = find_existing_research(question.id_of_post, args.output_dir)
        if existing:
            _, research = load_research(existing)
            research_file_path = existing
            print(f"Reusing existing research from {existing}")
        else:
            print(f"Running research...")
            # Use the LLM as researcher (no asknews dependency)
            bot_for_research = SpringTemplateBot2026(
                research_reports_per_question=1,
                predictions_per_research_report=1,
                publish_reports_to_metaculus=False,
                llms={
                    "default": GeneralLlm(model=args.model, temperature=args.temperature, timeout=40, allowed_tries=2),
                    "researcher": GeneralLlm(model=args.model, temperature=args.temperature, timeout=40, allowed_tries=2),
                    "parser": GeneralLlm(model=args.model, temperature=args.temperature, timeout=40, allowed_tries=2),
                },
            )
            research = await bot_for_research.run_research(question)
            research_file_path = save_research(question, research, args.model, args.output_dir)
            print(f"Research saved to {research_file_path}")

    # --- Create bot for predictions ---
    if args.offline:
        bot = MockSpringTemplateBot(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=False,
        )
    else:
        bot = SpringTemplateBot2026(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=False,
            llms={
                "default": GeneralLlm(model=args.model, temperature=args.temperature, timeout=40, allowed_tries=2),
                "parser": GeneralLlm(model=args.model, temperature=args.temperature, timeout=40, allowed_tries=2),
                "researcher": "no_research",
            },
        )

    # --- Run experiment ---
    predictions = await run_experiment(
        question=question,
        research=research,
        bot=bot,
        reps=args.reps,
        existing_predictions=existing_predictions,
    )

    # --- Save results ---
    results = {
        "question_id": question.id_of_post,
        "question_text": question.question_text,
        "question_url": question.page_url,
        "options": question.options,
        "research_file": research_file_path,
        "model": args.model if not args.offline else "mock",
        "temperature": args.temperature,
        "predictions": predictions,
    }

    if args.append_to:
        # Overwrite the existing file with merged results
        with open(args.append_to, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Updated results in {args.append_to}")
    else:
        results_path = save_results(results, args.output_dir)
        print(f"Results saved to {results_path}")

    # --- Print summary ---
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
