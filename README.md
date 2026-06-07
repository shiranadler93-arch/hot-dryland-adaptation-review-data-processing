# hot-dryland-adaptation-review-solutions effectiveness estimation
## Part 1 — Data Processing
## Part 2 — Logit Model Analysis
## Requirements
## How to Run

# Part 1: Data Processing
Data Processing for Solution Combinations - Preparing the data for use in a logit model
"""
This module defines the `DataProcessor` class, which handles the entire data processing pipeline for solution combinations. 
It includes methods for importing raw data from an Excel file, building a combination matrix, applying masking logic to filter out certain combinations and solutions, 
updating labels and scores based on conversion instructions, and saving the processed data back to Excel files.
The `DataProcessor` class is designed to be flexible and modular, allowing for easy adjustments to the data processing steps as needed.

The main steps in the data processing pipeline for data training are as follows:
1. Import and preprocess raw data from an Excel file, including:
   - cleaning, specifically dropping a column named 'config.' 
   - mapping classification labels to numeric scores.
2. Build a combination matrix that represents the presence of solutions in each combination, along with indexing dictionaries for all solutions:
    - Create a data dictionary to store solutions and scores for each combination.
    - Create a solution pool to track all unique solutions across combinations.
    - Build indexing dictionaries to map solutions to indices.
    - Construct the combination matrix where rows represent combinations and columns represent solutions, with binary values indicating the presence of each solution in each combination.
3. Apply an isolation mask to filter out combinations that contain only singular solutions (solutions that appear in only one combination), and track the remaining combinations and solutions after masking:
    - Identify combinations that contain singular solutions and create a mask to filter them out (flag solutions that appear in exactly one combination and combinations that contain only those solutions).
    - Track the list of solutions that are masked out due to being singular.
    - Create masked indexing dictionaries for the remaining solutions and combinations.
    - Construct a masked combination matrix that only includes the remaining combinations and solutions (after applying the isolation mask).
4. Update labels and scores according to specified conversion instructions, without modifying the original raw data: 
    - Update the masked labels and scores based on provided conversion instructions (e.g., mapping old labels to new labels, and old scores to new scores).
5. Save the processed data, including the training data with the combination matrix, raw data, solution indexing, solution mask list, and combination mask tracking, to Excel files for further analysis or modeling.

# Part 2: Logit Model Analysis
Train a logistic regression model to predict the effectiveness of combinations based on the masked combination matrix.
The training process includes an iterative adjustment of class weights and sample weights to optimize the model's performance, particularly focusing on improving 
the F1 score for underperforming classes while also penalizing confidently misclassified samples.
"""
The main steps in the data processing pipeline for this phase are as follows:
1. Load and pre-process the data using the `DataProcessor` class, which includes updating labels and scores based on a provided conversion dictionary.
2. Define the `iterative_logistic_training` function, which trains a logistic regression model iteratively, adjusting class weights and sample weights based on the model's performance in each iteration 
   to optimize the F1 score and reduce confidently misclassified samples. The adjustments are controlled by hyperparameters `alpha` and `beta`, which determine how aggressively the weights are modified 
   based on the F1 score and the confidence of misclassifications, respectively.
3. Train the logistic regression model using the `iterative_logistic_training` function, specifying the number of iterations, adjustment parameters for class weights and sample weights, and initial class weights.
4. Evaluate the best model found during the iterative training process using various metrics, including balanced accuracy, accuracy, classification report, confusion matrix, and Brier score.
5. Save the best model and the training history for further analysis and reference.

# Requirements are as follows:
# Part 1
- The `DataProcessor` class should be initialized with the path to the raw data Excel file.
- The `import_data` method should read the raw data from the specified Excel file, clean it by dropping the 'config.' column, and map the 'classification by result' to numeric scores.
- The `build_combination_matrix` method should create a data dictionary to store solutions and scores for each combination, build a solution pool, create indexing dictionaries, and construct the combination matrix.
- The `apply_isolation_mask` method should identify and mask out combinations that contain only singular solutions, track the remaining combinations and solutions, and create masked indexing dictionaries and a masked combination matrix.
- The `update_labels_and_scores` method should update the masked labels and scores based on provided conversion instructions, without modifying the original raw data.
- The `save_to_excel` method should save the processed data to Excel files, including the training data with the combination matrix, raw data, solution indexing, solution mask list, and combination mask tracking.
# Part 2
- The `iterative_logistic_training` function should accept the feature matrix `X`, target vector `y`, and various hyperparameters.
- The function should return the best model found and a history of metrics across iterations.

# How to run the code:
1. Ensure you have the required libraries installed (pandas, numpy, xlsxwriter).
2. Place the raw data Excel file in the same directory as this script or provide the correct path to the file when initializing the `DataProcessor`.
3. Run the script, and it will execute the complete data processing pipeline and save the processed data to Excel files with the specified prefix (default is 'logreg1').

"""
