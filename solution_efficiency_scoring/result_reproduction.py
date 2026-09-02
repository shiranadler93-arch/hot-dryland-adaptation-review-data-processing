"""
Command-line entry point that runs the full solution-effectiveness pipeline:

    1. Load raw combination data from Excel and build the masked combination matrix.
    2. Collapse the 5-way label into effective / neutral / ineffective.
    3. Train a logistic-regression model with iterative class/sample re-weighting.
    4. Evaluate the best model on the training data.
    5. Extrapolate predictions to every unmasked single solution.
    6. Persist the model, training history, and Excel reports.

It's default usage is set up to run a complete reproduction of the results in the paper, but it can also be used to train a new model on a different dataset.

Usage:
    1. for a complete reproduction of the results in the paper, run:
        python -m solution_efficiency_scoring.result_reproduction
    2. for training a new model on a different dataset, run:
        python -m solution_efficiency_scoring.result_reproduction --input_file <path_to_your_data.xlsx> --output-dir <your_output_dir> --output-prefix <your_output_prefix> --n-iter <num_iterations> --alpha <class_weight_adjustment_rate> --beta <sample_weight_adjustment_rate>
"""

from __future__ import annotations

import argparse

from .data_processing import DataProcessor
from .logit_model import LogitModelPipeline


# Collapses the original 5-way label (significantly effective/ineffective, mix)
# down to a 3-way effective / neutral / ineffective label.
DEFAULT_CONVERSION = {
    'score': {-2: -1, 2: 1},
    'label': {
        'significantly effective': 'effective',
        'significantly ineffective': 'ineffective',
    },
    'updated_label_dict': {
        -2: 'ineffective',
        -1: 'ineffective',
        0: 'neutral',
        1: 'effective',
        2: 'effective',
    },
}

# Due to noticable inbalance in the labeled training data, we start with a class-weight that is higher for the neutral class.
DEFAULT_INIT_CLASS_WEIGHT = {'effective': 1, 'neutral': 3, 'ineffective': 1}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_file', default='data.xlsx', help='Path to the raw Excel data file.')
    parser.add_argument('--output-dir', default='outputs', help='Directory to write all output files to.')
    parser.add_argument('--output-prefix', default='logreg', help='Prefix used for output filenames.')
    parser.add_argument('--n-iter', type=int, default=5000, help='Number of training iterations.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Class-weight adjustment rate.')
    parser.add_argument('--beta', type=float, default=0.05, help='Sample-weight adjustment rate.')
    return parser.parse_args(argv)


def run_pipeline(args) -> LogitModelPipeline:
    """Run the full pipeline given parsed CLI args and return the fitted pipeline."""
    # --- Load and pre-process data ---
    data = DataProcessor(args.input_file)
    data.process_all()
    data.update_labels_and_scores(DEFAULT_CONVERSION)
    data.save_to_excel(f'{args.output_dir}/{args.output_prefix}')

    # --- Train ---
    pipeline = LogitModelPipeline()
    pipeline.fit(
        data.comb_mat_masked,
        data.Y_masked_labeled,
        n_iter=args.n_iter,
        alpha=args.alpha,
        beta=args.beta,
        init_class_weight=DEFAULT_INIT_CLASS_WEIGHT,
    )

    # --- Evaluate on training data ---
    pipeline.evaluate(
        data.comb_mat_masked,
        data.Y_masked_labeled,
        sample_ids=data.remaining_comb,
    )

    # --- Extrapolate to unmasked single solutions ---
    pipeline.extrapolate_isolated(data.remaining_sol, data.inverse_indexing_dict)

    # --- Persist everything ---
    pipeline.save_all(output_dir=args.output_dir, output_prefix=args.output_prefix)

    return pipeline


def main(argv=None):
    args = parse_args(argv)
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    run_pipeline(args)
    print(f"Done. Outputs written to: {args.output_dir}")


if __name__ == '__main__':
    main()
