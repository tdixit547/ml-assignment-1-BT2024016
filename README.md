# ML Assignment 1: Polynomial Regression

Tanmay Dixit | BT2024016

This project contains the improved polynomial models, training and inference
code, model-selection experiments, and measured validation results.
The predictions and report have already been generated from the four personal
datasets. You do not need to rerun training to submit those files.

| Problem | Degree | Previous holdout MSE | New holdout MSE | New R-squared |
| --- | ---: | ---: | ---: | ---: |
| var1 | 5 | 0.617022 | 0.278873 | 0.971026 |
| var2 | 12 | 1.749023 | 0.274358 | 0.995233 |

This uses the same 200-row holdout as the earlier notebook. It was excluded
from new tuning, but its earlier score was already known. Hidden-test results
are unknown. These are the strongest tested choices under the recorded
selection procedure, not a claim of a globally optimal model.

## Selected models

- var1: equal average of degree-5 Lasso (alpha 0.008) and degree-5 ARD fits,
  using all six inputs. Repeated development CV MSE: 0.299307.
- var2: equal average of degree-12 Legendre Ridge (alpha 0.01, degree penalty
  exponent 2) and degree-9 monomial relaxed Lasso (alpha 0.001, relaxation 0.5),
  using all three inputs. Repeated development CV MSE: 0.278119.

Every component is polynomial regression. A fixed average is also a polynomial.
In each term, the sum of exponents is bounded by the reported total degree.
The Legendre basis is a polynomial basis. For degree-penalized Ridge, a
standardized term of total degree k is divided by k^2. Relaxed Lasso blends its
initial coefficients with a small-Ridge refit (alpha 0.01) on selected terms.

Training data are split 800/200 with seed 42. Development-only searches compare
degrees, bases and shrinkage methods. Nine individual finalists and two fixed
averages per problem are compared over three five-fold partitions (seeds 42,
137, 2026). Lowest mean MSE determines the choice. Statistical scaling, sparse
selection and refitting occur within each training fold. These selection scores
can be optimistic after search and are not final test scores.

Settings are frozen before the new holdout comparison. Final models then use
all 1,000 labelled rows per problem. Test targets were not supplied and never
entered selection or fitting.

## Reproduce the final results

Open the self-contained `ML_Assignment_1.ipynb` in Colab. Run its cells in order
and upload all four personal CSVs. Setup checks Colab's scientific stack and
installs only ReportLab if missing. Do not run the old NumPy replacement cell.

For Windows, create a fresh Python 3.11+ environment in this project folder:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe assignment.py train --data-dir data --out outputs_rerun
```

First put these files in `data/`:

```text
BT2024016_train_var1.csv
BT2024016_test_var1.csv
BT2024016_train_var2.csv
BT2024016_test_var2.csv
```

On Linux/macOS use `.venv/bin/python` in a fresh virtual environment. CPU is
sufficient. The default name is Tanmay Dixit and roll is BT2024016. Use a new
output folder for each run. Input hashes prevent scores for the original data
from being reported as scores for changed training data. Small numerical
differences across software versions are possible. `outputs/metrics.json`
records versions used to produce the included files.

## Saved-model inference

JSON models contain powers, coefficients, intercepts and averaging weights.
They do not require loading pickled estimators.

```powershell
.venv\Scripts\python.exe assignment.py predict --model outputs/var1_model.json --test data/BT2024016_test_var1.csv --output recreated_var1.csv
.venv\Scripts\python.exe assignment.py predict --model outputs/var2_model.json --test data/BT2024016_test_var2.csv --output recreated_var2.csv
```

Each CSV contains only header `y`, no index, and one finite value per original
test row. Duplicate inputs are retained. Predictions are not rounded or
clipped. Test rows are never sorted.

## Reproduce model selection

The training command refits frozen settings. To repeat the recorded searches:

```powershell
.venv\Scripts\python.exe experiments/run_search.py --data-dir data --out search_results
.venv\Scripts\python.exe assignment.py train --data-dir data --config search_results/selected_config.json --out search_rerun_outputs
```

`experiments/` records the broad grids, refined grids and fixed finalist
shortlist. The shortlist was chosen after the initial development search,
before repeated CV and the new holdout comparison. This reproduces selection
for these datasets; it is not an automatic general-purpose search for unrelated
data. `--include-expensive` also reruns rejected degree-six ARD fits, which were
slow and did not reach the shortlist.

`evidence/` contains the score grids, original metrics, frozen shortlist,
15-fold comparisons and out-of-fold predictions. Blank sparse-search scores
mean the LARS path reached its iteration cap before that alpha. These were
excluded. Some rejected ARD fits also reached their cap. The final settings
and input hashes are in `selected_config.json`.

## Files and verification

- `assignment.py`: validation, holdout comparison, final fit and inference CLI.
- `polynomial_model.py`: polynomial bases, fitting and portable coefficients.
- `report.py`: three-page PDF generation.
- `ML_Assignment_1.ipynb`: self-contained Colab workflow.
- `selected_config.json`: frozen settings and input hashes.
- `experiments/`: model-selection code using all supplied inputs.
- `evidence/`: measured selection results.
- `outputs/`: fitted models, predictions, holdout audit, plots and final PDF.
- `SUBMISSION_GUIDE.md`: required items and remaining steps.

Run `python -m unittest discover -s tests -v` for known-polynomial, total-degree,
portable-inference, row-order, duplicate-row and invalid-input checks. Final
artifacts were also checked against the supplied personal datasets.

To regenerate the PDF without training again:

```powershell
.venv\Scripts\python.exe assignment.py report --out outputs
```

Upload the extracted project contents to your GitHub repository. Personal
input CSVs are excluded from this package. Ensure your instructor can access
the repository and submit its URL with the required report and predictions.
This archive does not create a GitHub repository.
