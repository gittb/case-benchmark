# Contributing to CASE Benchmark

Thank you for your interest in contributing to the CASE Benchmark!

## Ways to Contribute

### 1. Submit Model Results

Evaluated a model? Share your results!

1. Run evaluation using the benchmark
2. Create a PR adding your results to `results/<model_name>/`
3. Include:
   - `results.json` - Evaluation output
   - `model_card.md` - Model description

See [docs/submission.md](docs/submission.md) for detailed instructions.

### 2. Report Issues

Found a bug or have a suggestion?

- Check existing [issues](https://github.com/gittb/case-benchmark/issues)
- Open a new issue with:
  - Clear description
  - Steps to reproduce (for bugs)
  - Expected vs actual behavior

### 3. Improve Documentation

Documentation improvements are always welcome:

- Fix typos or clarify explanations
- Add examples or tutorials
- Translate to other languages

### 4. Add Model Wrappers

Want to add support for a new model?

1. Create a new file in `case_benchmark/models/`
2. Implement the `EmbeddingModel` interface
3. Add tests in `tests/`
4. Update documentation

## Development Setup

```bash
# Clone the repository
git clone https://github.com/gittb/case-benchmark.git
cd case-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check .
ruff format .
```

## Code Style

- Use [ruff](https://github.com/astral-sh/ruff) for formatting and linting
- Follow existing code patterns
- Add docstrings to public functions
- Write tests for new functionality

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with clear messages
6. Open a pull request

## Questions?

Open a [discussion](https://github.com/gittb/case-benchmark/discussions) or reach out via issues.
