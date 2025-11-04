# TechTestDataiku — SQL → pandas proof-of-concept

**Candidate:** Willie Hernandez
**Email:** williehernandezr@gmail.com

TechTestDataiku is a proof-of-concept tool that translates both **standard SQL queries** and **SAS PROC SQL snippets** into **readable pandas code**.

## Supported (PoC)
- `SELECT` (basic projections and `AS` aliases)
- `FROM`
- `INNER JOIN ... ON a=b` (single/multiple)
- `WHERE` (simple boolean expressions; verify aliases after generation)
- `GROUP BY` with basic aggregates (`SUM`, `AVG`, `COUNT`, `MIN`, `MAX`) → pandas named aggregation
- `HAVING` (aggregate comparisons mapped to generated aliases)
- `ORDER BY`
- `LIMIT`

## Environment setup (uv)

```bash
# 1. Install uv if you don't have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create a virtual environment (Python 3.10+ recommended)
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install runtime dependencies
uv pip install pandas streamlit

# (Optional) Upgrade uv-managed tools and packages
uv pip install --upgrade pip
```

> **Why uv?**  
> - **Single-tool workflow:** One command manages virtual environments and dependency resolution—no juggling `python -m venv`, `pip`, or `conda` commands.  
> - **Speed:** uv uses the Rust-based resolver behind pip’s next-generation tooling, so env creation and installs are typically much faster than CPython’s `venv + pip` or Conda solves.  
> - **Cross-platform consistency:** uv works the same way on macOS, Linux, and Windows.  

## Run the Streamlit app

```bash
# Launch the interactive translator
uv run streamlit run app.py
```

Open the local URL printed in your terminal. Choose one of the editors and click its button:

- **Standard SQL query** → enter ANSI SQL and click `Generate from Standard SQL`.
- **PROC SQL snippet** → paste a SAS `PROC SQL; ... QUIT;` block and click `Generate from PROC SQL`; SQLift extracts the `SELECT` statement automatically (respects `OUTOBS=` limits).

## Validation ideas
- Compare row counts between original SQL and pandas outputs.
- Compare sums or key metrics by group.
- Spot-check a few rows.

## Tests

```bash
uv run python -m unittest discover -s tests -p 'test*.py' -v
```

## Pre-commit hooks

Install hooks locally so linting and formatting run before each commit:

```bash
pip install pre-commit
pre-commit install
```

Run them on-demand with `pre-commit run --all-files`. GitHub Actions also runs the configured hooks in CI.

## Static typing (mypy)

The pre-commit suite runs mypy automatically. To run it manually:

```bash
pip install mypy
mypy app.py sqlift_poc.py
```

## Run with Docker

```bash
# build the image
docker build -t techtest-dataiku .

# launch the Streamlit app on http://localhost:8501
docker run --rm -p 8501:8501 techtest-dataiku
```

## Continuous Integration (GitHub Actions)

The `.github/workflows/ci.yml` workflow runs on every push/PR. It:

- checks out the repo on Ubuntu runners,
- installs Python 3.11 with pandas, streamlit, and pre-commit,
- executes all pre-commit hooks (Black, Flake8, mypy), and
- runs the unittest suite (`python -m unittest discover`).

## Roadmap
- **PySpark backend:** add a second code generator so the same parsed SQL can emit idiomatic PySpark transformations (e.g., using `SparkSession.table`, joins, `groupBy().agg`, window functions).  
- **Broader SQL coverage:** handle more scalar and aggregate functions (e.g., `COALESCE`, `DATE_TRUNC`, `CASE WHEN`, arithmetic/date math) along with vendor-specific expressions.  
- **Dialect-awareness:** allow users to choose hints like PostgreSQL vs. Snowflake vs. SAS PROC SQL to tune quoting rules, boolean operators, and identifier casing.  
- **Test-driven improvements:** `tests/test_sqlift.py` is the best place to add failing cases and lock in fixes—grow the suite as new functions or edge cases get supported.
