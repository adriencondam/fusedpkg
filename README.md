# fusedpkg

`fusedpkg` provides fused/group-penalized generalized linear model tooling.

## Getting Started (With `uv`)

These steps are written for first-time users.

1. Install `uv` (once):

```bash
python -m pip install --upgrade pip
python -m pip install uv
```

2. Clone the repository:

```bash
git clone <repo-url>
cd fusedpkg
```

3. Create and activate a virtual environment:

```bash
uv venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
```

4. Install the project and development tools:

```bash
uv sync --extra dev
```

5. (Optional) verify the package imports:

```bash
uv run python -c "import fusedpkg; print('fusedpkg import OK')"
```

## Quickstart

```python
import pandas as pd
from fusedpkg.mod5 import GridSearch_Group

data = pd.DataFrame(
    {
        "color": ["red", "red", "blue", "blue", "green", "green", "red", "blue"],
        "shape": ["circle", "square", "circle", "square", "circle", "square", "circle", "square"],
        "n_claims": [0, 1, 0, 2, 1, 0, 1, 0],
        "exposure": [1.0] * 8,
    }
)

penalty_types = {"color": "g_fused", "shape": "g_fused"}
input_variables = ["color", "shape"]

model = GridSearch_Group(family="Poisson", lbd_group=0.1, random_state=0)
model.fit(
    data=data,
    penalty_types=penalty_types,
    input_variables=input_variables,
    target="n_claims",
    offset="exposure",
    n_k_fold=2,
)

print(model.lambda_curve[["group_lambda", "Deviance_cv_test"]])
```

## Run Tests

```bash
uv run pytest -q
```

Run a single test module:

```bash
uv run pytest tests/test_mod1.py -q
```

## Development Checks

```bash
uv run ruff check .
```

