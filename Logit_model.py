
import pandas as pd
import numpy as np

from data_processing import DataProcessor  # Use relative import if Data_processing is in the same package
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load and pre-process data
data = DataProcessor(
    'Shiran- raw data for analysis.xlsx'
)
data.process_all()
convertion_dict = {
    'score': {
        -2: -1,
        2: 1
    },
    'label': {
        'significantly effective': 'effective',
        'significantly ineffective': 'ineffective',
    },
    'updated_label_dict': {
        -2: 'ineffective',
        -1: 'ineffective',
        0: 'neutral',
        1: 'effective',
        2: 'effective'
    }
}
data.update_labels_and_scores(convertion_dict)
data.save_to_excel('logreg1_updated_labels')

# ---------------------------------------------------------
# iterative adjustment of class_weight and sample_weight
# ---------------------------------------------------------

def iterative_logistic_training(X, y, n_iter=10, alpha=0.5, beta=1.0, init_class_weight=None):
    """
    Train logistic regression iteratively, adjusting class_weight and sample_weight.

    * `alpha` controls how aggressively class_weight is modified based on per-class F1.
    * `beta` scales the penalty for confident misclassifications in sample_weight.

    Returns the best model found along with the final weights and a history of metrics.
    """
    # initialize
    classes = np.unique(y)
    if init_class_weight is None:
        class_weight = {c: 1.0 for c in classes}
    else:
        class_weight = dict(init_class_weight)

    # uniform starting sample weights
    sample_weight = np.ones(len(y))

    best_model = None
    best_score = -np.inf
    history = []

    for iteration in range(1, n_iter + 1):
        reg = LogisticRegression(max_iter=1000, class_weight=class_weight)
        reg.fit(X, y, sample_weight=sample_weight)

        preds = reg.predict(X)
        probs = reg.predict_proba(X)

        # evaluate F1 per class using sample_weight as importance
        report = classification_report(y, preds, output_dict=True)
        macro_f1 = report['macro avg']['f1-score']

        # compute difference between the top two predicted probabilities
        sorted_probs = np.sort(probs, axis=1)
        top_diff = sorted_probs[:, -1] - sorted_probs[:, -2]
        mis = (preds != y) & (top_diff > 0.30)  # only penalize if the model is confident (diff > 0.30)
        confidently_misclassified_samples_diff_mean = (top_diff*mis.astype(float)).sum()/mis.sum() if mis.sum() > 0 else 0

        history.append({'iter': iteration,
                        'macro_f1': macro_f1,
                        'confidently_misclassified_samples_diff_sum': confidently_misclassified_samples_diff_mean,
                        'class_weight': dict(class_weight),
                        'sample_weight': sample_weight.copy()})

        # compute sample-weight penalty for misclassified, confident examples
        mis = (preds != y) & (top_diff > 1/3)  # only penalize if the model is confident (diff > 1/3)
        sample_weight += beta * top_diff * mis.astype(float)
        # prevent weights from becoming too large
        sample_weight = np.clip(sample_weight, 1.0, 50.0)

        # adjust class_weight based on F1: classes with lower F1 get larger weight
        for cls in classes:
            f1 = report[str(cls)]['f1-score']
            # weight multiplies by factor >1 when f1 is small
            class_weight[cls] = class_weight.get(cls, 1.0) * (1 + alpha * (1 - f1))
        # normalize class_weight to keep the average weight around 1
        avg_weight = np.mean(list(class_weight.values()))
        for cls in class_weight:
            class_weight[cls] /= avg_weight

        # keep best
        improvement_metric = macro_f1/np.exp(confidently_misclassified_samples_diff_mean) # we want to maximize macro_f1 while minimizing the mean difference for confidently misclassified samples
        if improvement_metric > best_score:
            best_score = improvement_metric
            best_model = reg
            best_iteration = iteration

    return best_model, history, best_iteration


# example usage on the full training set
best_reg, training_history, best_iteration = \
    iterative_logistic_training(
        data.comb_mat_masked,
        data.Y_masked_labeled,
        n_iter=20000,
        alpha=0.1, #class weight adjustment
        beta=0.05, #sample weight adjustment
        init_class_weight={
            'effective': 1,
            'neutral': 3,
            'ineffective': 1,
        }
    )

training_history_df = pd.DataFrame(training_history)
best_iteration_row = training_history_df[training_history_df['iter'] == best_iteration]
best_class_weight = best_iteration_row['class_weight'].values[0]
best_sample_weight = best_iteration_row['sample_weight'].values[0]
# save best model and history
import joblib
import pickle
joblib.dump(best_reg, 'best_logistic_regression_model.pkl')
with open('training_optimiztion_history.pkl', 'wb') as f:
    pickle.dump(training_history, f)

# evaluate the best model
Y_pred_best = best_reg.predict(data.comb_mat_masked)
Y_pred_best_proba = best_reg.predict_proba(data.comb_mat_masked)

## Collect evaluation metrics for the best model
cls_report_best = classification_report(data.Y_masked_labeled, Y_pred_best, output_dict=True)
balanced_acc_best = balanced_accuracy_score(data.Y_masked_labeled, Y_pred_best)
acc_best = accuracy_score(data.Y_masked_labeled, Y_pred_best)
conf_mat_best = confusion_matrix(data.Y_masked_labeled, Y_pred_best)

sorted_probs = np.sort(Y_pred_best_proba, axis=1)
top_diff = sorted_probs[:, -1] - sorted_probs[:, -2]
mis = (Y_pred_best != data.Y_masked_labeled) & (top_diff > 0.30)  # only penalize if the model is confident (diff > 0.30)


# brier score and brier skill score
cls_encode = OneHotEncoder()
cls_encode.fit(best_reg.classes_.reshape(-1,1))
data.Y_masked_enc = cls_encode.transform(
    data.Y_masked_labeled.reshape(-1,1)
)
sqr_err = np.square(Y_pred_best_proba-data.Y_masked_enc).sum(axis=1)
brier_score = sqr_err.mean()
BSS = 1-(brier_score/(2/3)) # 2/3 is the brier score of the null model (predicting 1/3 for all categories, for all samples)

# produce table for per combination evaluation
comb_eval_df = pd.DataFrame(
    data={
        'combination index': data.remaining_comb,
        'difference between top two predicted probabilities': top_diff,
        'TP': (data.Y_masked_labeled==Y_pred_best).astype(int),
        'True score': data.Y_masked_labeled
    }
)
comb_eval_df['model prob. prediction squared error'] = sqr_err
comb_eval_df = pd.concat(
    [
        comb_eval_df,
        pd.DataFrame(
            columns=best_reg.classes_,
            data=Y_pred_best_proba
        )
    ],
    axis=1
)

## Format for Excel saving
# Summary metrics
summary_metrics = pd.DataFrame({
    'Metric': ['Balanced Accuracy', 'Accuracy', 'Brier Score', 'Brier Skill Score'],
    'Value': [balanced_acc_best, acc_best, brier_score, BSS]
})

# Classification report as DataFrame
cls_report_df = pd.DataFrame(cls_report_best).transpose()

# Confusion matrix as DataFrame
conf_mat_df = pd.DataFrame(conf_mat_best, 
                          index=best_reg.classes_, 
                          columns=best_reg.classes_)
conf_mat_df['Total actual label'] = conf_mat_df.sum(axis=1)
conf_mat_df.loc['Total predicted label'] = conf_mat_df.sum(axis=0)

# Function to save evaluation to Excel
def save_evaluation_to_excel(filename='evaluation_results.xlsx'):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        summary_metrics.to_excel(writer, sheet_name='Summary Metrics', index=False)
        cls_report_df.to_excel(writer, sheet_name='Classification Report')
        conf_mat_df.to_excel(writer, sheet_name='Confusion Matrix')
        comb_eval_df.to_excel(writer, sheet_name='Per Combination Evaluation', index=False)

# Save the evaluation results
save_evaluation_to_excel('logreg1_updated_labels_optimized_evaluation_results.xlsx')


# ---------------------------------------------------------

## extrapolate to all unmasked single solutions
masked_sol_pool_size = data.comb_mat_masked.shape[1]

sol_class_predictions_prob = best_reg.predict_proba(np.eye(masked_sol_pool_size))
sol_class_predictions = best_reg.predict(np.eye(masked_sol_pool_size))

sol_class_predictions_prob_sorted = np.sort(sol_class_predictions_prob, axis=1)
first_diff = sol_class_predictions_prob_sorted[:,-2:]
first_diff = first_diff[:,1] - first_diff[:,0]

sol_class_pred_df = pd.DataFrame(
    data={
        'solution index': data.remaining_sol,
        'solution name': [data.inverse_indexing_dict[idx] for idx in data.remaining_sol],
        'predicted class': sol_class_predictions,
        'difference between top two predicted probabilities': first_diff
    }
)
sol_class_pred_df = pd.concat(
    [
        sol_class_pred_df,
        pd.DataFrame(
            columns=best_reg.classes_,
            data=sol_class_predictions_prob
        )
    ],
    axis=1
)

# Function to save evaluation to Excel
def save_predictions_to_excel(filename='prediction_results.xlsx'):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        sol_class_pred_df.to_excel(writer, sheet_name='solution scores', index=False)

# Save the evaluation results
save_predictions_to_excel('logreg1_updated_labels_optimized_prediction_results.xlsx')
