# GOLD: Globally Optimal Supervised Discretisation Algorithm

GOLD independently discretises numeric predictors using an exact dynamic programme to minimise the **sum of leave-one-out predictive multiclass Brier losses** plus `(m - 1) * log(N)/2`, where `m` is the number of intervals and `N` is the training sample size. Smoothing is fixed at `alpha = 0.5`, and the minimum bin size is `max(5, floor(N/20))`.

The optimality guarantee is for the **univariate penalised Brier objective**, not downstream classifier AUC.

## Files

- `GOLD.R`: standalone algorithm, with no third-party R dependencies; **safe to source**.
- `GOLD_tests.R`: closed-form loss and exhaustive-optimality checks.
- `GOLD_benchmark.R`: benchmark code and dataset loaders; needs packages `discretization`, `mlbench`, `nnet`, `xgboost`, `pROC`. CACC and Shuttle are excluded to match the completed reported experimental set.

## Usage

```r
source("GOLD.R")
fit <- gold(iris) # outcome in last column; numeric predictors required
head(fit$Disc.data)
fit$cutp
fit$avg_bins

# Apply training-derived boundaries; do not refit on test data
newdata <- gold_transform(iris[1:5,1:4], fit$cutp)
```

The algorithm requires a complete-case training data frame with only numeric predictor columns, a categorical outcome in its last column and at least two observed outcome classes. Handle missing values before fitting. It does not automatically impute or preprocess other feature types.

## Tests

```r
source("GOLD.R")
source("GOLD_tests.R")
stopifnot(gold_self_test())
```

## Benchmark

To regenerate the benchmark, open `GOLD_benchmark.R`, change `GOLD_RUN_BENCHMARK <- FALSE` to `TRUE`, then run from the directory containing these files. Benchmark outputs are written to `GOLD_results/`. The benchmark is computationally intensive. Always record `sessionInfo()` and package versions when publishing experimental results.

**Validation note:** The closed-form Brier expression was independently checked numerically. Full R execution and cross-package tests could not be performed in the environment used to prepare this release; run the supplied test suite in R before publishing.
