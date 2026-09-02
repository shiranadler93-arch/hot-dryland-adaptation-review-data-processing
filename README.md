# solution efficiency scoring

Estimates the effectiveness of individual "solutions" from procured data
on combinations of solutions and their outcomes.

## Structure

```
solution_scoring/
├── __init__.py        			# public API: DataProcessor, LogitModelPipeline
├── data_processing.py 			# DataProcessor: raw Excel -> masked combination matrix
├── logit_model.py      		# LogitModelPipeline: fit / evaluate / extrapolate / save
└── result_reproduction.py      # CLI entry point that runs the full pipeline
```

`data_processing.py` and `logit_model.py` are plain, import-only modules --
no top-level code runs when you import them. All orchestration (loading
data, training, evaluating, saving outputs) lives in `result_reproduction.py`.

## Setup

```bash
git clone <this-repo-url>
cd <this-repo>
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
# or, for an installable CLI command:
pip install -e .
```

## Running the full pipeline

```bash
python -m solution_efficiency_scoring.result_reproduction
```

or, if installed with `pip install -e .`:

```bash
train-logit-model
```

Options:

| Flag                | Default                    | Description                            |
| ------------------- | -------------------------- | -------------------------------------- |
| `input_file`      | (required)                 | Path to the raw Excel data file        |
| `--output-dir`    | `outputs`                | Directory to write all output files to |
| `--output-prefix` | `logreg1_updated_labels` | Prefix used for output filenames       |
| `--n-iter`        | `20000`                  | Number of training iterations          |
| `--alpha`         | `0.1`                    | Class-weight adjustment rate           |
| `--beta`          | `0.05`                   | Sample-weight adjustment rate          |

The raw Excel file is expected to have one row per combination of
solutions, a `config.` column (dropped), and a `classification by result`
column with one of: `significantly effective`, `effective`, `mix`/`MIX`,
`ineffective`, `significantly ineffective`.

## Using it as a library

```python
from solution_efficiency_scoring import DataProcessor, LogitModelPipeline

data = DataProcessor("data.xlsx")
data.process_all()

pipeline = LogitModelPipeline()
pipeline.fit(data.comb_mat_masked, data.Y_masked_labeled, n_iter=5000)
pipeline.evaluate(data.comb_mat_masked, data.Y_masked_labeled, sample_ids=data.remaining_comb)
pipeline.extrapolate(data.remaining_sol, data.inverse_indexing_dict)
pipeline.save_all(output_dir="outputs")
```

## Outputs

Running the pipeline produces, under `--output-dir`:

- `<prefix>_train_data.xlsx` -- processed training data (from `DataProcessor.save_to_excel`)
- `best_logistic_regression_model.pkl` -- the best model found (joblib)
- `training_optimization_history.pkl` -- per-iteration metrics/weights (pickle)
- `<prefix>_optimized_evaluation_results.xlsx` -- summary metrics, classification report, confusion matrix, per-sample evaluation
- `<prefix>_optimized_prediction_results.xlsx` -- extrapolated per-solution predictions
