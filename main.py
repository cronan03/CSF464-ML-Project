import os
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt

# For reproducibility
import random
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# ----------------------------
# 1. Function Mappings and Complexity Quantification
# ----------------------------

# Define the mapping from operation strings to unique function IDs
operation_map = {
    "Dilation SE1": 0,
    "Dilation SE2": 1,
    "Dilation SE3": 2,
    "Dilation SE4": 3,
    "Dilation SE5": 4,
    "Dilation SE6": 5,
    "Dilation SE7": 6,
    "Dilation SE8": 7,
    "Erosion SE1": 8,
    "Erosion SE2": 9,
    "Erosion SE3": 10,
    "Erosion SE4": 11,
    "Erosion SE5": 12,
    "Erosion SE6": 13,
    "Erosion SE7": 14,
    "Erosion SE8": 15
}

# Reverse mapping for reference
id_to_operation = {v: k for k, v in operation_map.items()}

def get_structuring_element(se_number):
    """
    Get the structuring element based on the SE number.

    Args:
        se_number (int): Structuring element number (1-8).

    Returns:
        np.ndarray: Structuring element array.
    """
    if se_number == 1:
        # Cross-shaped SE
        se = np.array([[0,1,0],
                      [1,1,1],
                      [0,1,0]])
    elif se_number == 2:
        # Square SE
        se = np.ones((3,3))
    elif se_number == 3:
        # Diagonal SE
        se = np.eye(3)
    elif se_number == 4:
        # X-shaped SE
        se = np.array([[1,0,1],
                      [0,1,0],
                      [1,0,1]])
    elif se_number == 5:
        # Larger cross
        se = np.array([[0,0,1,0,0],
                      [0,0,1,0,0],
                      [1,1,1,1,1],
                      [0,0,1,0,0],
                      [0,0,1,0,0]])
    elif se_number == 6:
        # Larger square SE
        se = np.ones((5,5))
    elif se_number == 7:
        # Horizontal line SE
        se = np.array([[1,1,1]])
    elif se_number == 8:
        # Vertical line SE
        se = np.array([[1],
                      [1],
                      [1]])
    else:
        # Default SE
        se = np.ones((3,3))
    return se

def compute_function_complexity(func_id):
    """
    Assign a complexity score to a function based on its characteristics.

    Args:
        func_id (int): Function identifier.

    Returns:
        int: Complexity score.
    """
    # Assign complexity based on structuring element size and type
    if func_id < 8:
        # Dilation functions
        se_number = func_id + 1
        operation_complexity = 1  # Dilation complexity
    else:
        # Erosion functions
        se_number = func_id - 7
        operation_complexity = 1  # Erosion complexity

    se = get_structuring_element(se_number)
    se_size = np.sum(se)  # Sum of ones in the structuring element

    complexity = operation_complexity * se_size
    return complexity

def compute_sequence_complexity(function_sequence):
    """
    Compute the total complexity of a sequence of functions.

    Args:
        function_sequence (list): List of function IDs.

    Returns:
        int: Total complexity score.
    """
    total_complexity = sum([compute_function_complexity(func_id) for func_id in function_sequence])
    return total_complexity

# ----------------------------
# 2. Data Loading
# ----------------------------

def load_tasks(task_folder):
    """
    Load tasks and their function sequences from the specified folder.

    Args:
        task_folder (str): Path to the folder containing task JSON files and solution TXT files.

    Returns:
        pd.DataFrame: DataFrame with columns 'task_id', 'sample_id', 'input_grid', 'output_grid', 'function_sequence', 'sequence_complexity'.
    """
    data = []
    task_files = [f for f in os.listdir(task_folder) if f.endswith('.json')]
    count=0
    for task_file in task_files:
        # if count==3:
        #     break
        # count+=1
        
        task_path = os.path.join(task_folder, task_file)
        task_id = task_file[:-5]  # Remove '.json' extension
        solution_file = f"{task_id}_soln.txt"
        solution_path = os.path.join(task_folder, solution_file)

        if not os.path.exists(solution_path):
            print(f"Warning: No solution file found for {task_id}. Skipping this task.")
            continue

        # Load function sequence from solution file
        with open(solution_path, 'r') as f:
            functions = [line.strip() for line in f.readlines()]
            # Map function names to IDs
            function_sequence = [operation_map[func] for func in functions if func in operation_map]

        # Compute sequence complexity
        sequence_complexity = compute_sequence_complexity(function_sequence)

        # Load task data
        with open(task_path, 'r') as f:
            try:
                task_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for Task {task_id}: {e}. Skipping this task.")
                continue

        # Check if task_data is a list
        if isinstance(task_data, list):
            train_data = task_data
        elif isinstance(task_data, dict):
            # Adjust based on actual structure; assuming 'train' key
            train_data = task_data.get('train', [])
        else:
            print(f"Unexpected JSON structure for Task {task_id}. Skipping this task.")
            continue

        for idx, sample in enumerate(train_data):
            input_grid = np.array(sample['input'])
            output_grid = np.array(sample['output'])
            data.append({
                'task_id': task_id,
                'sample_id': idx,
                'input_grid': input_grid,
                'output_grid': output_grid,
                'function_sequence': function_sequence,
                'sequence_complexity': sequence_complexity
            })
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} input-output pairs from {df['task_id'].nunique()} tasks.")
    return df

# ----------------------------
# 3. Flattening Grids
# ----------------------------

def flatten_grid(grid):
    """
    Flatten the input grid into a 1D array.

    Args:
        grid (np.ndarray): 2D grid.

    Returns:
        np.ndarray: Flattened 1D array.
    """
    return grid.flatten()

# ----------------------------
# 4. Model Training and Evaluation
# ----------------------------

def train_and_evaluate_models(X, y, models):
    """
    Train and evaluate multiple models using Leave-One-Out cross-validation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target matrix.
        models (dict): Dictionary of models to train and evaluate.

    Returns:
        dict: Dictionary containing average accuracies for each model.
    """
    results = {}
    loo = LeaveOneOut()
    for model_name, model in models.items():
        accuracies = []
        print(f"\nTraining and evaluating model: {model_name}")
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = model
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            sample_accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
            accuracies.append(sample_accuracy)

        average_accuracy = np.mean(accuracies)
        results[model_name] = average_accuracy
        print(f"{model_name} - Average Accuracy: {average_accuracy:.4f}")
    return results

# ----------------------------
# 5. Main Execution Flow
# ----------------------------

def main():
    # Define the path to your task folder
    task_folder = 'catasimple'  # Replace with your actual path

    # Check if the path exists
    if not os.path.exists(task_folder):
        print(f"Error: The specified task folder '{task_folder}' does not exist.")
        return

    # Load tasks
    df = load_tasks(task_folder)

    if df.empty:
        print("No data loaded. Please check your task folder and ensure it contains valid task JSON and solution TXT files.")
        return

    # Process each task separately
    task_ids = df['task_id'].unique()
    model_names = ['Decision Tree', 'Neural Network']
    all_results = []

    for task_id in task_ids:
        task_df = df[df['task_id'] == task_id]
        num_samples = len(task_df)
        print(f"\nProcessing Task {task_id} with {num_samples} samples.")
        if num_samples < 2:
            print(f"Not enough samples in Task {task_id} for training and testing. Skipping.")
            continue

        X = []
        y = []
        for idx, row in task_df.iterrows():
            input_grid = row['input_grid']
            output_grid = row['output_grid']
            if input_grid.shape != output_grid.shape:
                print(f"Input and output grids have different shapes in Task {task_id}, Sample {row['sample_id']}. Skipping sample.")
                continue
            X.append(flatten_grid(input_grid))
            y.append(flatten_grid(output_grid))

        X = np.array(X)
        y = np.array(y)

        if len(X) < 2:
            print(f"Not enough valid samples in Task {task_id} after filtering. Skipping.")
            continue

        # Define models, wrapping models that do not support multi-output with MultiOutputClassifier
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=random_seed),
            #'Logistic Regression': MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=random_seed)),
            #'Neural Network': MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_seed)),
              'Neural Network': MultiOutputClassifier(
                MLPClassifier(hidden_layer_sizes=(50,),  # Reduced hidden layers
                             max_iter=200,               # Reduced iterations
                             solver='lbfgs',             # Faster solver
                             early_stopping=True,        # Enable early stopping
                             random_state=random_seed,
                             verbose=False),             # Disable verbose output
                n_jobs=-1
            )
        }

        # Train and evaluate models
        results = train_and_evaluate_models(X, y, models)

        # Collect results
        task_complexity = task_df['sequence_complexity'].iloc[0]
        for model_name in model_names:
            all_results.append({
                'task_id': task_id,
                'model': model_name,
                'accuracy': results.get(model_name, None),
                'complexity': task_complexity
            })

    # Create DataFrame from results
    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("No results to analyze.")
        return

    # Analyze correlation between complexity and accuracy
    print("\nAnalyzing correlation between complexity and model performance...")
    correlation_results = {}
    for model_name in model_names:
        model_results = results_df[results_df['model'] == model_name]
        complexities = model_results['complexity']
        accuracies = model_results['accuracy']
        if len(complexities) < 2:
            print(f"Not enough data points for {model_name} to compute correlation.")
            correlation = np.nan
        else:
            correlation = np.corrcoef(complexities, accuracies)[0, 1]
        correlation_results[model_name] = correlation
        print(f"{model_name} - Correlation between complexity and accuracy: {correlation:.4f}")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    for model_name in model_names:
        model_results = results_df[results_df['model'] == model_name]
        plt.scatter(model_results['complexity'], model_results['accuracy'], label=model_name, alpha=0.7)

    plt.xlabel('Function Sequence Complexity')
    plt.ylabel('Model Accuracy')
    plt.title('Model Accuracy vs Function Sequence Complexity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Boxplot to visualize accuracy distributions across models
    plt.figure(figsize=(8, 6))
    results_df.boxplot(column='accuracy', by='model')
    plt.title('Accuracy Distribution by Model')
    plt.suptitle('')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.show()

    # Summary of Correlations
    print("\n--- Correlation Summary ---")
    for model_name, corr in correlation_results.items():
        if np.isnan(corr):
            print(f"{model_name}: Not enough data to compute correlation.")
        else:
            print(f"{model_name}: Correlation coefficient = {corr:.4f}")

    print("\n--- Experiment Completed ---")
    print("Review the scatter plot and correlation coefficients to analyze the impact of function complexity on model performance.")

if __name__ == "__main__":
    main()
