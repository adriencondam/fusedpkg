# Code Review Feedback for fusedpkg

## Scope
This review focuses on repository quality, package usability, testing health, and maintainability.  
Validation commands run during review:
- uv run --with pytest pytest -q (test collection/runtime health)
- uv run --with ruff ruff check . (lint/static quality)
- uv build -q (packaging output sanity)

## What is already good
- The core optimization module includes meaningful docstrings and type hints, especially in src/fusedpkg/mod5.py.
- The project has clear intent and domain focus (fused/group lasso GLM workflows), with realistic example workflows in the scripts/.
- Caching and warm-start ideas in src/fusedpkg/mod5.py are strong performance-minded design decisions.
- The project already has a pyproject.toml and modern packaging layout (src/), which is a good base.
- Documentation exists and is trying to cover install, usage, and development flow, even though it is currently inconsistent.

## Prioritized findings

### P0 (Release blocker): Tests do not run in current state
- Why this matters: CI confidence is effectively zero because test collection fails before any assertion runs.
- Evidence:
  - tests/test_mod1.py:1 imports from mypkg (module does not exist in this repo).
  - tests/subpkg1/test_mod1.py:1 imports from mypkg.
  - uv run --with pytest pytest -q thus fails with ModuleNotFoundError: No module named 'mypkg'.
- Action:
  - Replace stale mypkg imports with fusedpkg paths.

### P0 (Release blocker): Distributed wheel includes legacy old/ code with many lint/runtime issues
- Why this matters: outdated modules are being shipped to users and inflate maintenance risk.
- Evidence:
  - uv build -q wheel contents include fusedpkg/old/*.
  - Ruff reports 103 issues total; biggest source is src/fusedpkg/old/mod21.py (55 findings).
  - Legacy files still reference removed namespace patterns (mypkg) and contain unresolved symbols.
- Action:
  - remove old/ directory from package contents.

### P1 (High): mod5.py is monolithic and contains duplicate class/function definitions
- Why this matters: maintainability and correctness are fragile when later definitions silently override earlier ones.
- Evidence:
  - src/fusedpkg/mod5.py is 2,472 lines.
  - Duplicate class definitions:
    - GridSearch_Group at src/fusedpkg/mod5.py:1053 and src/fusedpkg/mod5.py:1797
    - GridSearch_Fused at src/fusedpkg/mod5.py:1386 and src/fusedpkg/mod5.py:2152
  - Duplicate helper definitions also exist (for example _predict_mean_from_linear_predictor).
- Action:
  - Split file into focused modules (metrics.py, penalties.py, problem_builders.py, grid_search_group.py, grid_search_fused.py, grid_search_generalised.py).
  - Keep one canonical class per concept and remove shadowed duplicates.
  - Add lightweight tests per module.

### P1 (High): Hardcoded machine-specific paths in examples should be abstracted behind a loader
- Why this matters: examples are not runnable outside one author environment and hurt onboarding.
- Evidence:
  - scripts/example.py:15 and scripts/example.py:157 reference local C:\Users\AdrienCondamin\....
  - scripts/example_2.py:12 references local C:\Users\AdrienCondamin\....
- Action:
  - Introduce a data loader abstraction (for example, load_dataset(name: str, data_dir: Path | None = None)).
  - Accept dataset paths through CLI args and/or environment variables.
  - Provide one tiny sample dataset under data/ or document where/how to fetch one reproducibly. This dataset should be used in CI tests, preferably a few megabytes at most in size.

### P1 (High): README is inconsistent with actual package and API
- Why this matters: first-time users will fail immediately following docs.
- Evidence:
  - Title/name mismatch: README.md:1(fused_lasso_glm) vs package name fusedpkg (pyproject.toml:2).
  - Wrong folder in install steps: README.md:17 uses cd mypkg.
  - Usage snippet imports non-existent APIs and duplicates import lines: README.md:39, README.md:40, README.md:41, README.md:42.
- Action:
  - Rewrite README quickstart using import paths that exist today.
  - Add a copy-paste smoke snippet that is executed in CI (doctest or script test).
  - Keep naming aligned across README, package metadata, and examples.

### P1 (High): Current tests include logic defects even beyond import failures
- Why this matters: once imports are fixed, several tests will still not validate behavior correctly.
- Evidence:
  - Duplicate test function name in tests/subpkg1/test_mod1.py:24 and tests/subpkg1/test_mod1.py:31 (one overrides the other).
  - tests/subpkg1/test_mod1.py:43 compares a DataFrame result to a Python list (incorrect assertion shape).
  - tests/test_mod1.py currently only imports and contains no executable test assertions.
- Action:
  - Rebuild tests around actual return types and expected behavior.
  - Add unit tests for helper error paths and integration smoke tests for GridSearch_*.
  - Track coverage for critical public APIs.


### P1 (High): Naming conventions are inconsistent across the codebase
- Why this matters: inconsistent naming increases cognitive load, makes APIs harder to discover, and creates avoidable maintenance errors.
- Evidence:
  - Package and docs naming drift: fusedpkg in project metadata, fused_lasso_glm in README, and mypkg references in tests/docs.
  - Mixed style for public symbols: GridSearch_Generalised (PascalCase with underscore and British spelling) alongside snake_case helpers.
  - Typos and near-duplicates in identifiers:  
    - parcimony_step (likely sparsity/parsimony), 
    - fused_type versus fused_types, 
    - ref_modality_dict versus ref_modalities.
  - Low-signal module names (mod1, mod5) make responsibilities unclear and hinder navigation.
- Action:
  - Define one naming standard and enforce it: 
    - snake_case for functions/variables/modules, 
    - PascalCase for classes, 
    - consistent American spelling (British spelling is rarely used in software engineering).
  - Standardize core domain terms in a small glossary (for example: generalized, sparsity, penalty_types, reference_modalities).
  - Rename ambiguous modules to responsibility-based names (for example: grid_search, penalties, preprocessing, metrics).
  - Add lint enforcement for naming rules in CI so new inconsistencies do not reappear.

### P2 (Medium): Error handling should raise exceptions instead of printing and returning None
- Why this matters: callers cannot reliably detect invalid usage or recover programmatically. These silent failures are hard to track, and logging is missing, leading to lots of wasted efforts on debugging.
- Evidence:
  - src/fusedpkg/additional_functions/mod1.py:104 prints then returns on missing dict for "List" mode.
  - src/fusedpkg/additional_functions/mod1.py:113 prints then returns on unsupported method.
- Action:
  - Raise ValueError with clear messagesm as is done in some other places in the package.
  - Add type hints and explicit return types.
  - Add tests for invalid method and malformed reference mapping.

### P2 (Medium): Numeric parsing in reorder_df_columns is brittle
- Why this matters: non-numeric modality suffixes can break ordering logic unexpectedly.
- Evidence:
  - src/fusedpkg/additional_functions/mod1.py:145 does float(find_first_number(...)) without guarding None.
- Action:
  - Validate parse result before casting.
  - Define deterministic fallback ordering for non-numeric modalities.
  - Add tests for mixed labels (for example, "A", "B10", "B2").

### P2 (Medium): Library-level print() noise should be controlled via logging
- Why this matters: noisy stdout makes library usage hard in notebooks, services, and tests.
- Evidence:
  - Unconditional prints in src/fusedpkg/mod5.py (for example: :73, :756, :820, :1998).
- Action:
  - Replace prints with logging and honor verbosity levels (like info, warning, ...).
  - Keep public APIs quiet by default, no prints in these

### P2 (Medium): Ruff should be part of the standard dev toolchain and CI
- Why this matters: major quality issues are currently unguarded and regressions can land silently.
- Evidence:
  - pyproject.toml:22-25 dev extras include pytest and black, but not ruff.
  - Manual run found 103 lint findings.
- Action:
  - Add Ruff to dev dependencies and CI checks (ruff check ., optionally ruff format or keep black + Ruff linting).
  - Start with fail-on-new-issues policy if full cleanup is too large initially.
  - Add Ruff to pre-commit hooks for local development. Then, Ruff will block commits with new issues.
  - Add Ruff to CI checks.

## Suggested priotities plan

### Phase 1 (P0/P1 stabilization so the package runs for other users)
1. Fix import namespace drift (mypkg -> fusedpkg) across tests/docs/examples.
2. Remove or exclude src/fusedpkg/old from packaged artifacts.
3. Repair README quickstart so a fresh user can run one end-to-end example.
4. Establish CI baseline: pytest + ruff.

### Phase 2 (P1/P2 maintainability)
1. Split mod5.py into cohesive modules and remove duplicate definitions.
2. Introduce a reusable data loader and remove hardcoded local paths from scripts.
3. Convert print-driven library output to structured logging.
4. Strengthen tests around helper edge cases and grid-search smoke paths.
5. Define and enforce naming conventions.


### Phase 3 (polish)
1. Improve doc consistency and API docs.
2. Add contribution guide with local dev commands (uv sync, pytest, ruff).
3. Add minimal benchmark/smoke dataset for reproducible examples.
4. Add pre-commit hooks for Ruff and pytest.
5. Include Ruff in CI checks like GitHub Actions.
6. Improve test coverage for edge cases.
7. Include the general documentation of the package with detailed documentation on how it works, in a separate docs/ folder.
