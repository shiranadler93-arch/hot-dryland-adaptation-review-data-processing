"""
Logistic-regression modeling utilities for the solution-effectiveness pipeline.

This module contains:
  - `iterative_logistic_training`: the core training loop that adaptively
    adjusts class_weight and sample_weight across iterations.
  - `LogitModelPipeline`: a class wrapping the full model lifecycle --
    fit, evaluate, extrapolate to unmasked single solutions, and persist
    (pickled model + Excel reports).

It intentionally contains no top-level execution code. Import it and drive
it from a script (see result_reproduction.py) or from your own code / notebook.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder


def iterative_logistic_training(
    X,
    y,
    n_iter: int = 10,
    alpha: float = 0.5,
    beta: float = 1.0,
    init_class_weight: Optional[dict] = None,
):
    """
    Train logistic regression iteratively, adjusting class_weight and sample_weight.

    * `alpha` controls how aggressively class_weight is modified based on per-class F1.
    * `beta` scales the penalty for confident misclassifications in sample_weight.

    Returns:
        best_model: the fitted LogisticRegression with the highest improvement metric.
        history: list of per-iteration dicts (macro_f1, class_weight, sample_weight, ...).
        best_iteration: the 1-indexed iteration number of the best model.
    """
    classes = np.unique(y)
    class_weight = {c: 1.0 for c in classes} if init_class_weight is None else dict(init_class_weight)
    sample_weight = np.ones(len(y))

    best_model = None
    best_score = -np.inf
    best_iteration = None
    history = []

    for iteration in range(1, n_iter + 1):
        reg = LogisticRegression(max_iter=1000, class_weight=class_weight)
        reg.fit(X, y, sample_weight=sample_weight)

        preds = reg.predict(X)
        probs = reg.predict_proba(X)

        report = classification_report(y, preds, output_dict=True)
        macro_f1 = report['macro avg']['f1-score']

        sorted_probs = np.sort(probs, axis=1)
        top_diff = sorted_probs[:, -1] - sorted_probs[:, -2]

        # Diagnostic: mean confidence gap on confidently-wrong predictions (diff > 0.30)
        mis_diag = (preds != y) & (top_diff > 0.30)
        confidently_misclassified_samples_diff_mean = (
            (top_diff * mis_diag.astype(float)).sum() / mis_diag.sum() if mis_diag.sum() > 0 else 0
        )

        history.append({
            'iter': iteration,
            'macro_f1': macro_f1,
            'confidently_misclassified_samples_diff_sum': confidently_misclassified_samples_diff_mean,
            'class_weight': dict(class_weight),
            'sample_weight': sample_weight.copy(),
        })

        # Sample-weight penalty: up-weight confidently-wrong predictions (diff > 1/3)
        mis_penalty = (preds != y) & (top_diff > 1 / 3)
        sample_weight = sample_weight + beta * top_diff * mis_penalty.astype(float)
        sample_weight = np.clip(sample_weight, 1.0, 50.0)

        # Class-weight update: classes with lower F1 get a larger weight next round
        for cls in classes:
            f1 = report[str(cls)]['f1-score']
            class_weight[cls] = class_weight.get(cls, 1.0) * (1 + alpha * (1 - f1))
        avg_weight = np.mean(list(class_weight.values()))
        for cls in class_weight:
            class_weight[cls] /= avg_weight

        # Maximize macro_f1 while minimizing confidence on confidently-wrong samples
        improvement_metric = macro_f1 / np.exp(confidently_misclassified_samples_diff_mean)
        if improvement_metric > best_score:
            best_score = improvement_metric
            best_model = reg
            best_iteration = iteration

    return best_model, history, best_iteration


class LogitModelPipeline:
    """
    End-to-end logistic-regression pipeline: iterative training, evaluation,
    extrapolation to unmasked single solutions, and Excel/pickle persistence.

    Typical usage::

        pipeline = LogitModelPipeline()
        pipeline.fit(data.comb_mat_masked, data.Y_masked_labeled, n_iter=20000, ...)
        pipeline.evaluate(data.comb_mat_masked, data.Y_masked_labeled, sample_ids=data.remaining_comb)
        pipeline.extrapolate(data.remaining_sol, data.inverse_indexing_dict)
        pipeline.save_all(output_dir='outputs', output_prefix='logreg1_updated_labels')
    """

    def __init__(self):
        self.model = None
        self.history = None
        self.best_iteration = None
        self.training_history_df = None
        self.best_class_weight = None
        self.best_sample_weight = None

        # Evaluation artifacts
        self.summary_metrics = None
        self.cls_report_df = None
        self.conf_mat_df = None
        self.comb_eval_df = None

        # Extrapolation artifacts
        self.sol_class_pred_df = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y, n_iter=10, alpha=0.5, beta=1.0, init_class_weight=None):
        """Run iterative training and keep the best model plus its weight history."""
        self.model, self.history, self.best_iteration = iterative_logistic_training(
            X, y, n_iter=n_iter, alpha=alpha, beta=beta, init_class_weight=init_class_weight
        )
        self.training_history_df = pd.DataFrame(self.history)
        best_row = self.training_history_df[self.training_history_df['iter'] == self.best_iteration]
        self.best_class_weight = best_row['class_weight'].values[0]
        self.best_sample_weight = best_row['sample_weight'].values[0]
        return self.model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X, y_true, sample_ids=None, id_col_name='combination index'):
        """
        Compute evaluation metrics for the fitted model on (X, y_true).

        Args:
            X: feature matrix to evaluate on.
            y_true: true labels for X.
            sample_ids: optional per-row identifiers (e.g. combination indices)
                included as the first column of the per-sample evaluation table.
            id_col_name: column name to use for `sample_ids`.
        """
        self._require_fit()

        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)

        cls_report = classification_report(y_true, y_pred, output_dict=True)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        conf_mat = confusion_matrix(y_true, y_pred)

        sorted_probs = np.sort(y_pred_proba, axis=1)
        top_diff = sorted_probs[:, -1] - sorted_probs[:, -2]

        cls_encoder = OneHotEncoder()
        cls_encoder.fit(self.model.classes_.reshape(-1, 1))
        y_true_enc = cls_encoder.transform(np.asarray(y_true).reshape(-1, 1))

        sqr_err = np.square(y_pred_proba - y_true_enc).sum(axis=1)
        brier_score = sqr_err.mean()
        # 2/3 is the Brier score of the null model (uniform 1/3 prediction, 3 classes)
        brier_skill_score = 1 - (brier_score / (2 / 3))

        self.summary_metrics = pd.DataFrame({
            'Metric': ['Balanced Accuracy', 'Accuracy', 'Brier Score', 'Brier Skill Score'],
            'Value': [balanced_acc, acc, brier_score, brier_skill_score],
        })
        self.cls_report_df = pd.DataFrame(cls_report).transpose()

        conf_mat_df = pd.DataFrame(conf_mat, index=self.model.classes_, columns=self.model.classes_)
        conf_mat_df['Total actual label'] = conf_mat_df.sum(axis=1)
        conf_mat_df.loc['Total predicted label'] = conf_mat_df.sum(axis=0)
        self.conf_mat_df = conf_mat_df

        comb_eval_data = {}
        if sample_ids is not None:
            comb_eval_data[id_col_name] = sample_ids
        comb_eval_data['difference between top two predicted probabilities'] = top_diff
        comb_eval_data['TP'] = (np.asarray(y_true) == y_pred).astype(int)
        comb_eval_data['True score'] = y_true

        comb_eval_df = pd.DataFrame(comb_eval_data)
        comb_eval_df['model prob. prediction squared error'] = sqr_err
        comb_eval_df = pd.concat(
            [comb_eval_df, pd.DataFrame(columns=self.model.classes_, data=y_pred_proba)],
            axis=1,
        )
        self.comb_eval_df = comb_eval_df

        return {
            'summary_metrics': self.summary_metrics,
            'classification_report': self.cls_report_df,
            'confusion_matrix': self.conf_mat_df,
            'per_sample_evaluation': self.comb_eval_df,
        }

    # ------------------------------------------------------------------
    # Extrapolation to unmasked single solutions
    # ------------------------------------------------------------------
    def extrapolate_isolated(self, remaining_sol, inverse_indexing_dict):
        """
        Predict class probabilities for every remaining single solution
        (each represented as a one-hot row), i.e. the model's estimate of
        that solution's effect in isolation.
        """
        self._require_fit()

        pool_size = len(remaining_sol)
        probs = self.model.predict_proba(np.eye(pool_size))
        preds = self.model.predict(np.eye(pool_size))

        sorted_probs = np.sort(probs, axis=1)
        top_two = sorted_probs[:, -2:]
        top_diff = top_two[:, 1] - top_two[:, 0]

        sol_df = pd.DataFrame({
            'solution index': remaining_sol,
            'solution name': [inverse_indexing_dict[idx] for idx in remaining_sol],
            'predicted class': preds,
            'difference between top two predicted probabilities': top_diff,
        })
        sol_df = pd.concat(
            [sol_df, pd.DataFrame(columns=self.model.classes_, data=probs)],
            axis=1,
        )
        self.sol_class_pred_df = sol_df
        return sol_df

    # ------------------------------------------------------------------
    # Saving / persistence
    # ------------------------------------------------------------------
    def save_model(self, path='best_logistic_regression_model.pkl'):
        self._require_fit()
        joblib.dump(self.model, path)

    def save_training_history(self, path='training_optimization_history.pkl'):
        if self.history is None:
            raise RuntimeError("No training history to save; call fit() first.")
        with open(path, 'wb') as f:
            pickle.dump(self.history, f)

    def save_evaluation(self, path='evaluation_results.xlsx'):
        if self.summary_metrics is None:
            raise RuntimeError("No evaluation results to save; call evaluate() first.")
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            self.summary_metrics.to_excel(writer, sheet_name='Summary Metrics', index=False)
            self.cls_report_df.to_excel(writer, sheet_name='Classification Report')
            self.conf_mat_df.to_excel(writer, sheet_name='Confusion Matrix')
            self.comb_eval_df.to_excel(writer, sheet_name='Per Combination Evaluation', index=False)

    def save_predictions(self, path='prediction_results.xlsx'):
        if self.sol_class_pred_df is None:
            raise RuntimeError("No predictions to save; call extrapolate() first.")
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            self.sol_class_pred_df.to_excel(writer, sheet_name='solution scores', index=False)

    def save_all(self, output_dir='.', output_prefix='logreg'):
        """Convenience wrapper: save model, history, evaluation and predictions together."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.save_model(out / 'best_logistic_regression_model.pkl')
        self.save_training_history(out / 'training_history.pkl')
        self.save_evaluation(out / f'{output_prefix}_evaluation_results.xlsx')
        self.save_predictions(out / f'{output_prefix}_prediction_results.xlsx')

    def _require_fit(self):
        if self.model is None:
            raise RuntimeError("Model has not been fit yet; call fit() first.")
