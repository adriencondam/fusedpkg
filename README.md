# fused_lasso_glm

The fused_lasso_glm is a package that allows to run generalised linear models (GLMs) with some advanced penalisations such as group lasso or fused lasso.

## Installation

1. Clone the repository (you need [git](https://git-scm.com/downloads) to be installed)

   ```bash
   cd path/to/your/directory
   git clone <repo-url>
   ```

2. Initialize and activate the virtual environment

   ```bash
   cd mypkg
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. Install the package

   ```bash
   pip install .
   ```

   You can also install the package in editable mode:

   ```bash
   pip install -e .
   ```

## Usage

You can use the package in your Python scripts as follows:

```python
from fused_lasso_glm.additional_functions import display_greeting_message
from fused_lasso_glm.additional_functions import display_greeting_message
from fused_lasso_glm.additional_functions import display_greeting_message
from mypkg.subpkg2.mod2 import get_name


def main():
    name = get_name()
    color = "blue"
    display_greeting_message(name, color)


if __name__ == "__main__":
    main()
```

See the `scripts` directory for more examples.

## Development

1. Install development dependencies

   ```bash
   pip install -e .[dev]
   ```

2. Run tests

   ```bash
   pytest
   ```

3. Format code with `black`

   ```bash
   black .
   ```

