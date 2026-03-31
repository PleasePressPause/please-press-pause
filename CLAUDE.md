# CLAUDE.md - Project Context for AI Assistants

This file provides context for AI assistants (like Claude) working on this codebase.

## Project Overview

**Please Press Pause** is a forecasting bot for the Metaculus AI Forecasting Tournament. It started as a fork of the [Metaculus bot template](https://github.com/Metaculus/metac-bot-template) but has been customized with additional features.

### Tournament Context
- Metaculus runs AI forecasting tournaments with significant prizes (~$175k annually)
- Bots make probabilistic predictions on real-world questions (binary, numeric, multiple choice, date)
- Performance is measured by prediction accuracy over time

## Key Files

### Core Bot
- `main.py` - Main bot implementation
  - `SpringTemplateBot2026` - Base forecasting bot class
  - `MockSpringTemplateBot` - Mock bot for testing without API costs
  - `AlphabeticalMockBot` - Mock bot that assigns probabilities by alphabetical order (for testing permutation)

### Permutation System (Position Bias Reduction)
- `permutation.py` - Utilities for permuting multiple choice options
  - `Permutation` class - Represents a permutation with forward/inverse mappings
  - `run_with_permutations()` - Orchestrator that wraps any prediction function
  - `permute_options()` / `unpermute_predictions()` - Apply/reverse permutations
  - `generate_diverse_permutations()` - Generate varied permutations including non-self-inverse
  - `detect_ordered_options()` - Skip permutation for ordered options (1,2,3 or A,B,C)

### Testing
- `test_permutation.py` - 33 tests for permutation functionality
- `mock_llm.py` - Mock prediction generators for all question types
- `mock_questions.py` - Mock questions for offline testing

### Configuration
- `.env` - Environment variables (API keys, tokens) - not committed
- `pyproject.toml` - Poetry dependencies and project config

## Architecture Decisions

### Safe-by-Default Testing
The bot uses mock predictions by default to avoid accidental API costs:
- `--real` flag required for actual LLM calls
- `--publish` flag required to submit to Metaculus
- `--mode offline` for fully offline testing with mock questions

### Position Bias Reduction
We hypothesize that LLMs may assign different probabilities to options based on their position. Our approach:
1. The bot itself is unaware of permutation - it just sees options in whatever order
2. An orchestration layer (`run_with_permutations()`) handles permutation
3. Multiple predictions are made with different option orderings
4. Results are unpermuted and averaged

This separation keeps the bot simple and makes the permutation logic reusable.

### GitHub App for CI/CD
We use a GitHub App (`claude-code-assistant-ppp`) for automated operations:
- Can create branches and PRs
- Cannot merge PRs (requires human approval via branch protection)
- Provides audit trail of AI-assisted changes

## Development Workflow

### Running Tests
```bash
poetry run pytest test_permutation.py -v
```

### Running the Bot
```bash
# Offline mode with mock questions (no API calls)
poetry run python main.py --mode offline

# Test with real Metaculus questions but mock predictions
poetry run python main.py --mode test_questions

# Real mode (costs money!)
poetry run python main.py --mode test_questions --real

# Full tournament run with publishing
poetry run python main.py --mode tournament --real --publish
```

### Creating PRs
PRs are created via the GitHub App and require human approval to merge.

## Important Notes

- Never commit `.env` or files with API keys
- The `github_app_auth.py` file contains user-specific paths and shouldn't be committed
- Always run tests before creating PRs
- Mock mode is the default - use `--real` explicitly for actual LLM calls
