# fusedpkg

`fusedpkg` provides fused/group-penalized generalized linear model tooling.

## Installation

```bash
git clone <repo-url>
cd fusedpkg
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -e .[dev]
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

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

