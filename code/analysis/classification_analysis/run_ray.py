import sys

# Add relevant directories to the system path for importing necessary modules
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/')
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')

# Import necessary functions and libraries
from functions import *
import shap  # SHAP (SHapley Additive exPlanations) for model interpretability
import xgboost as xgb  # XGBoost implementation
from xgboost.sklearn import XGBRegressor
from matplotlib import colors as plt_colors
import time
import numpy as np
import pandas as pd  # Data processing, CSV file I/O (e.g., pd.read_csv)
from tune_sklearn import TuneSearchCV  # Hyperparameter tuning library
from sklearn.model_selection import train_test_split
from sklearn import metrics
import ray  # Distributed computing library

def get_rgb(hex_codes):
    """
    Converts a list of hex color codes into a list of RGB tuples.
    
    Args:
    hex_codes (list of str): List of hex color codes.
    
    Returns:
    list of tuples: List of RGB tuples.
    """
    rgb_values = []
    for hex_code in hex_codes:
        r = int(hex_code[1:3], 16) / 255
        g = int(hex_code[3:5], 16) / 255
        b = int(hex_code[5:7], 16) / 255
        rgb_values.append((r, g, b))
    
    return rgb_values

def readit(path):
    """
    Reads a large CSV file in chunks and concatenates them into a single DataFrame.
    
    Args:
    path (str): File path to the CSV file.
    
    Returns:
    DataFrame: Concatenated DataFrame of all chunks.
    """
    data = pd.concat(pd.read_csv(path, iterator=True, chunksize=100000), ignore_index=True)
    return data

from import_subtypes import *

# Sample data for analysis
data = sample1_rest_combined
holdout = sample2_rest_combined

# Define color scheme for resting-state data classification
colors = ['#f6511d', '#ffb400', '#7fb800', '#0d2c54']
cmap = plt_colors.ListedColormap(np.array(colors))

# Initialize Ray and attach it to the local node Ray instance
ray.init("ray://127.0.0.1:10001", namespace="ray") 

@ray.remote
def tune_search_tuning():
    """
    Performs hyperparameter tuning using the TuneSearchCV library, fits an XGBoost model,
    and evaluates it on a holdout set. SHAP values are computed for model interpretation.
    
    Returns:
    dict: Best hyperparameter combination found during tuning.
    """
    global data, holdout
    
    # Prepare data for model training and evaluation
    X = data.drop(['ID', 'Subtype'], axis=1)
    xcols = X.columns
    y = np.array(data['Subtype'])
    
    X_holdout = holdout.drop(['ID', 'Subtype'], axis=1)
    y_holdout = np.array(holdout['Subtype'])
    
    class_names = list(data.Subtype.unique())
    
    # Compute sample weights to handle class imbalance
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
    
    # Define hyperparameter search space for XGBoost
    params = {'max_depth': [5, 10, 15, 20, 25, 30],
              'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
              'subsample': [0.5 + x / 100 for x in range(10, 50, 10)],
              'kernel': ['linear', 'rbf'],
              'colsample_bytree': [0.5 + x / 100 for x in range(10, 50, 10)],
              'colsample_bylevel': [0.5 + x / 100 for x in range(10, 50, 10)],
              'n_estimators': [100, 500, 1000, 2000],
             }
    
    # Initialize the XGBoost classifier
    xgbclf = xgb.XGBClassifier(objective="multi:softmax", tree_method="hist")

    # Replace RandomizedSearchCV with TuneSearchCV for distributed hyperparameter tuning
    tune_search = TuneSearchCV(estimator=xgbclf,
                               param_distributions=params,
                               scoring='accuracy',
                               n_jobs=24,
                               n_trials=25,
                               verbose=1)

    # Perform hyperparameter tuning
    tune_search.fit(X, y)
    print("cv results: ", tune_search.cv_results_)

    best_combination = tune_search.best_params_
    print("Best parameters:", best_combination)

    # Evaluate the best model on the holdout set
    predictions = tune_search.predict(X_holdout)
    accuracy = metrics.accuracy_score(y_holdout, predictions)
    print("Accuracy: ", accuracy)
    
    # Save the best parameters to a CSV file
    params = tune_search.best_params_
    params_df = pd.DataFrame.from_dict(params, orient='index', columns=['value'])
    params_df.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/rest_include/best_params.csv')
    
    # Re-train the XGBoost model with the best parameters
    cls = xgb.XGBClassifier(**params)
    cls.fit(X, y, sample_weight=sample_weights)
    
    # Compute SHAP values for the holdout set
    explainer = shap.TreeExplainer(cls)
    shap_values = explainer.shap_values(X_holdout, check_additivity=False)
    
    # Generate classification report
    report = classification_report(y_holdout, predictions, output_dict=True)
    
    # Plot SHAP summary plot
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=1)
    plt.subplot(1, 2, 1)
    shap.summary_plot(shap_values, X_holdout.values, plot_type="bar", 
                      class_names=class_names, class_inds='original',
                      feature_names=X.columns, color=cmap, 
                      plot_size=None, show=False)
    plt.savefig(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/rest_include/rest_shap_plot.png', dpi=300, bbox_inches='tight')

    # Save the classification report and top features to CSV files
    output_report = pd.DataFrame(report)
    output_report.to_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/rest_include/rest_class_report.csv')
    
    sorted_idx = cls.feature_importances_.argsort()[::-1]
    sorted_cols = X_holdout.columns[sorted_idx][:15].tolist()
    ft_cols = pd.DataFrame(sorted_cols, columns=['ft_cols'])
    ft_cols.to_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/rest_include/rest_ft_cols.csv')
    
    # Plot individual SHAP plots for each class
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(wspace=0.4)
    for i in range(len(data.Subtype.unique())):
        if len(data.Subtype.unique()) < 5:
            plt.subplot(2, 2, i + 1)
        else: 
            plt.subplot(2, 3, i + 1)
        
        shap.summary_plot(shap_values[i], X_holdout.values, feature_names=X_holdout.columns, plot_size=None, show=False, max_display=10)
        plt.savefig(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/rest_include/rest_ind_shap_plot.png', dpi=300, bbox_inches='tight')
        
    return best_combination

if __name__ == '__main__':
    # Record the start time of the script
    start_time = time.time()

    # Create and execute the remote task for hyperparameter tuning
    remote_clf = tune_search_tuning.remote()
    best_params = ray.get(remote_clf)

    # Record the stop time of the script
    stop_time = time.time()
    print("Stopping at :", stop_time)
    print("Total elapsed time: ", stop_time - start_time)

    # Output the best parameters found
    print("Best params from main function: ", best_params)
