# ML741 Assignment 3: Active Learning in Neural Networks

N.H. Chowdhury, 26243881. Stellenbosch University.

A comparison of passive learning by stochastic gradient descent with two active
learning approaches: the sensitivity analysis selective learning algorithm
(SASLA) and incremental learning by uncertainty sampling. Three classification
problems and three function approximation problems, 30 paired runs each.

Everything is written from scratch in Python and NumPy. No machine learning
framework is used for the networks.

## Contents

- `src/` - the network, the three approaches, the data and the statistics
- `scripts/` - the pipeline, listed below in run order
- `data_raw/` - the three datasets that scikit-learn does not ship
- `results/` - every result file the report draws on (`results/cache` holds every finished run)

The report itself is not in this repo. It is uploaded in STEMlearn.

## Datasets

Iris, Breast Cancer and Diabetes come with scikit-learn. Optical Digits
(training and test files pooled), Fuel Consumption (auto-mpg) and Concrete
Strength are in `data_raw/`.

## Reproducing

```powershell
pip install -r requirements.txt

python scripts\phase2_architecture.py        # hidden layer size
python scripts\phase3_lambda.py              # weight decay coefficient
python scripts\phase4_params.py              # learning rate, momentum, beta, uncertainty settings
python scripts\phase4d_active_eta_alpha.py   # test of the shared learning rate and momentum
python scripts\phase5_compare.py             # final comparison, 30 runs
python scripts\shared_params_check.py        # rerun where phase 4d found a better pair
python scripts\analyse.py                    # statistical tests
python scripts\decay_effect.py               # weight decay and SASLA pruning
python scripts\informativeness_spread.py     # why SASLA prunes classification harder
python scripts\verify.py                     # checks of the code
```

Run from the repo root. Final runs use seeds 1 to 30, tuning runs use
seeds from 1000. Finished runs are cached in `results/cache`, so a stopped phase
picks up where it left off. With the cache in place every script only reloads
the saved runs. To run everything from scratch, rename `results` first
(for example to `results_old`) and create an empty `results` folder.

## Headline results

- No approach was significantly more accurate over the six problems
  (Friedman p = 0.084).
- SASLA needed 61.1 to 99.1 percent of the pattern presentations of passive
  learning, with no significant loss in accuracy.
- Uncertainty sampling needed fewer presentations on the classification
  problems only. On function approximation, the committee of three networks
  tripled the cost.
- SASLA pruned the classification problems more strongly, because the
  informativeness spreads far wider there.

## What each file does

### `src/`

- **`network.py`** - the network, SGD with momentum and weight decay, and the
  sensitivity and informativeness that SASLA uses.
- **`selectors.py`** - passive learning, SASLA and uncertainty sampling.
- **`experiment.py`** - one training run: one problem, one approach, one seed.
- **`data.py`** - loads the problems, splits them 60/20/20 and scales them.
- **`metrics.py`** - MSE, error rate, macro F1, RMSE, generalisation factor.
- **`stats.py`** - Wilcoxon with Holm, effect size, Friedman, Iman-Davenport,
  Nemenyi.
- **`sweep.py`** - budgets, seeds, parallel runs and caching.

### `scripts/`

- **`phase2` to `phase5`** - the tuning stages and the final comparison.
- **`shared_params_check.py`** - checks that sharing the learning rate did not
  change the conclusions.
- **`analyse.py`** - the statistical tests.
- **`decay_effect.py`, `informativeness_spread.py`** - the two supplementary
  experiments.
- **`verify.py`** - checks the sensitivity and gradient against finite
  differences and the Friedman test against SciPy.
- **`rerun_problem.py`** - reruns all phases for one problem.

## Licence

MIT
