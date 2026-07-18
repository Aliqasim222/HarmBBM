# Contributing

Contributions that improve numerical reliability, documentation, tests, or
reproducibility are welcome.

1. Create a branch from `main`.
2. Keep optimizer equations and experiment defaults documented.
3. Add or update tests for behavioral changes.
4. Run `pytest` and `python -m scripts.smoke_test` before opening a pull request.
5. Do not commit downloaded datasets, checkpoints, or generated result files.

For changes that alter the optimizer definition, describe the mathematical
change explicitly and provide a reproducible experiment showing its effect.
