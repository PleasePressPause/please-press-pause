# Please Press Pause

A forecasting bot for the [Metaculus AI Forecasting Tournament](https://www.metaculus.com/aib/).

## Overview

This bot makes probabilistic predictions on real-world questions including:
- Binary questions (yes/no outcomes)
- Numeric questions (continuous values)
- Multiple choice questions
- Date questions

## Features

### Safe-by-Default Testing
The bot uses mock predictions by default to avoid accidental API costs:
```bash
# Mock mode (default) - no API calls, no cost
poetry run python main.py --mode offline

# Real mode - requires explicit flag
poetry run python main.py --mode test_questions --real
```

### Position Bias Reduction for Multiple Choice
We implement option permutation to reduce position bias in LLM predictions. The hypothesis is that LLMs may assign different probabilities based on option position. By permuting options and averaging predictions, we can reduce this bias.

```python
from permutation import run_with_permutations

async def my_predict(options: list[str]) -> PredictedOptionList:
    # Your prediction logic here
    return await llm.predict(options)

averaged, log = await run_with_permutations(
    my_predict,
    original_options=["Option A", "Option B", "Option C"],
    num_permutations=5
)
```

## Quick Start

### Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installing-with-pipx)

### Installation
```bash
git clone https://github.com/PleasePressPause/please-press-pause.git
cd please-press-pause
poetry install
```

### Configuration
Copy `.env.template` to `.env` and add your API keys:
- `METACULUS_TOKEN` - Get from https://metaculus.com/aib
- `OPENROUTER_API_KEY` - For LLM access

### Running

```bash
# Offline testing (no API calls)
poetry run python main.py --mode offline

# Test with real Metaculus questions, mock predictions
poetry run python main.py --mode test_questions

# Real predictions (costs money!)
poetry run python main.py --mode test_questions --real

# Tournament mode with publishing
poetry run python main.py --mode tournament --real --publish
```

### Command Line Options
- `--mode` - Run mode: `tournament`, `test_questions`, `offline`
- `--real` - Use real LLM calls (default: mock)
- `--publish` - Submit predictions to Metaculus
- `--full` - Use full prediction count instead of minimal

## Testing

```bash
# Run all tests
poetry run pytest

# Run permutation tests with verbose output
poetry run pytest test_permutation.py -v
```

## Project Structure

```
.
├── main.py                 # Main bot implementation
├── permutation.py          # Option permutation utilities
├── mock_llm.py            # Mock prediction generators
├── mock_questions.py      # Mock questions for offline testing
├── test_permutation.py    # Permutation tests
├── CLAUDE.md              # AI assistant context
└── integrations/          # Third-party integrations
```

## Contributing

This project uses a GitHub App for automated operations. PRs require human approval to merge.

## License

See upstream [Metaculus bot template](https://github.com/Metaculus/metac-bot-template) for license information.
