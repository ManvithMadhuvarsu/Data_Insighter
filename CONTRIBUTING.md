# Contributing to Data Insighter

Thanks for your interest in improving Data Insighter! This guide covers the
basics for getting a development environment running and submitting changes.

## Development setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate
   # Unix:     source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Create a `.env` file with a `SECRET_KEY` for stable sessions.

4. Run the app:

   ```bash
   python app.py
   ```

## Code style

- Formatting: [black](https://black.readthedocs.io/) (`black .`), line length 100.
- Imports: [isort](https://pycqa.github.io/isort/) with the black profile (`isort .`).
- Linting: [flake8](https://flake8.pycqa.org/) (`flake8 .`). Configuration lives
  in `.flake8` and `pyproject.toml`.

## Tests

Run the suite before opening a pull request:

```bash
pytest -q
```

Please add or update tests for any behavior you change. Visualization changes
should include a rendering test in `tests/test_visualization_generator.py`, and
recommendation changes a case in `tests/test_chart_recommender.py`.

## Pull requests

- Keep changes focused; one logical change per pull request.
- Write a clear description of *what* changed and *why*.
- Make sure `pytest -q` and `flake8 . --select=E9,F63,F7,F82` pass; CI runs both.
