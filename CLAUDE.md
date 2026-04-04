# CLAUDE.md

## Project

Forecasting bot for the [Metaculus AI Forecasting Tournament](https://www.metaculus.com/aib/). Forked from the Metaculus bot template.

## Commands

```bash
# Run tests
poetry run pytest test_permutation.py -v

# Run bot (mock predictions, no API calls)
poetry run python main.py --mode offline

# Run bot with real Metaculus questions, mock predictions
poetry run python main.py --mode test_questions

# Real predictions (costs money)
poetry run python main.py --mode test_questions --real

# Full tournament run
poetry run python main.py --mode tournament --real --publish
```

## Architecture Decisions

**Safe-by-default**: Mock predictions are used unless `--real` is explicitly passed. Never remove this default.

**Position bias reduction**: The bot is unaware of permutation. An orchestration layer (`run_with_permutations()`) calls the predict function multiple times with different option orderings and averages the results. Keep these concerns separate.

**GitHub App**: The `claude-code-assistant-ppp` app can create branches and PRs but cannot merge them - merging requires human approval via branch protection rules.

## Gotchas

- Never commit `.env` or `github_app_auth.py` (contains user-specific paths)
- GitHub tokens from `github_app_auth.py` expire quickly - run `poetry run python github_app_auth.py` and set the remote URL to authenticate. Do this proactively when you need to push.
- Always run tests before creating PRs
- API credentials (METACULUS_TOKEN, OPENAI_API_KEY, etc.) are only available in GitHub Actions secrets, not locally. If you need to run something that requires credentials, create a GitHub workflow instead of trying to run it locally.
