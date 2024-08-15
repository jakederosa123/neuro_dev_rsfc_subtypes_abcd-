#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/')
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')

# Import necessary functions and libraries
from functions import *
import shap  # SHAP (SHapley Additive exPlanations) for model interpretability
import xgboost as xgb  # XGBoost implementation
from xgboost.sklearn import XGBRegressor
from matplotlib import colors as plt_colors
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from hgboost import hgboost

# Set the visual style for plots
sns.set(rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'})
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")  # Enables high-definition plots
plt.rcParams["figure.autolayout"] = True  # Enables tight layout for better visualization of multiplots

# Define class names (for classification)
classes = ['1', '2', '3', '4']

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

def classy(data, holdout, datatype, showfig=None):
    """
    Trains a classifier on the provided dataset, applies SHAP for model interpretation,
    and optionally displays and saves the results.
    
    Args:
    data (DataFrame): Training data containing features and labels.
    holdout (DataFrame): Holdout data for evaluation.
    datatype (str): Type of data (used for labeling outputs).
    showfig (bool): If True, displays and saves figures.
    
    Returns:
    explainer (shap.TreeExplainer): SHAP explainer object.
    shap_values (ndarray): SHAP values for the holdout set.
    X_holdout (DataFrame): Holdout set features.
    sorted_cols (list): Top features sorted by importance.
    report (dict): Classification report.
    """
    # Extract class names from the data
    class_names = list(data.Subtype.unique())
    
    # Select relevant features and labels for training
    X = data.drop(['ID', 'Subtype'], axis=1)
    xcols = X.columns
    y = np.array(data['Subtype'])
    
    # Select relevant features and labels for the holdout set
    X_holdout = holdout.drop(['ID', 'Subtype'], axis=1)
    y_holdout = np.array(holdout['Subtype'])
    
    # Standardize the features
    from scipy import stats
    X = pd.DataFrame(stats.zscore(X), columns=xcols)
    X_holdout = pd.DataFrame(stats.zscore(X_holdout), columns=xcols)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)
    
    # Compute sample weights to handle class imbalance
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
    
    # Define initial parameters for the XGBoost model
    params = {'max_depth': 3, 
              'learning_rate': 0.2, 
              'subsample': 0.8,
              'colsample_bytree': 0.6, 
              'colsample_bylevel': 0.9, 
              'n_estimators': 1000,
              'objective':'multi:softmax',
              'tree_method':"hist"
             }
    
    # Hyperparameter tuning using hgboost
    hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0, top_cv_evals=10, random_state=None, verbose=0)
    params = hgb.xgboost(X, y, pos_label=1)['params']
    
    # Train the classifier
    cls = xgb.XGBClassifier(**params)
    cls.fit(X, y, sample_weight=sample_weights)
    
    # Compute SHAP values for the holdout set
    explainer = shap.TreeExplainer(cls)
    shap_values = explainer.shap_values(X_holdout)
    
    # Make predictions on the holdout set
    predictions = cls.predict(X_holdout)
    
    # Optionally, display and save the results
    if showfig is not None:
        print(accuracy_score(y_holdout, predictions))
        print(confusion_matrix(y_holdout, predictions))
        print(classification_report(y_holdout, predictions))
    
    report = classification_report(y_holdout, predictions, output_dict=True)
    
    # Plot SHAP summary plot
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=1)
    global cmap
    plt.subplot(1, 2, 1)
    shap.summary_plot(shap_values, X.values, plot_type="bar", 
                      class_names=class_names, class_inds='original',
                      feature_names=X.columns, color=cmap, 
                      plot_size=None, show=False)
    plt.savefig(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/{datatype}/{datatype}_shap_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Identify and sort top features by importance
    n_top_features = 15
    sorted_idx = cls.feature_importances_.argsort()[::-1]
    sorted_values = -np.sort(-cls.feature_importances_[sorted_idx][:n_top_features])
    sorted_cols = X_test.columns[sorted_idx][:n_top_features].tolist()
    
    if showfig is not None:
        plt.show()
    else:
        plt.clf()
    
    return explainer, shap_values, X_holdout, X_holdout, sorted_cols, report

def shapit(data, holdout, datatype, showfig=None):
    """
    Wrapper function to run the `classy` function and generate plots and reports.
    
    Args:
    data (DataFrame): Training data containing features and labels.
    holdout (DataFrame): Holdout data for evaluation.
    datatype (str): Type of data (used for labeling outputs).
    showfig (bool): If True, displays and saves figures.
    
    Returns:
    return_sorted_cols (list): Top features sorted by importance.
    report (dict): Classification report.
    """
    explainer1, shap_values1, X1, X1_test, return_sorted_cols, report = classy(data, holdout, datatype, showfig=showfig)
    
    if showfig is not None:
        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(wspace=0.4)

        # Plot individual SHAP plots for each class
        for i in range(len(data.Subtype.unique())):
            if len(data.Subtype.unique()) < 5:
                plt.subplot(2, 2, i + 1)
            else:
                plt.subplot(2, 3, i + 1)
                
            shap.summary_plot(shap_values1[i], X1.values, feature_names=X1.columns,
                              plot_size=None, show=False, max_display=5)
            plt.savefig(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/{datatype}/{datatype}_ind_shap_plot.png', dpi=300, bbox_inches='tight')
    
        plt.show()
        plt.clf()

        # Plot SHAP bar plots for each class
        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(wspace=0.4)

        for i in range(len(data.Subtype.unique())):
            if len(data.Subtype.unique()) < 5:
                plt.subplot(2, 2, i + 1)
            else:
                plt.subplot(2, 3, i + 1)

            shap.summary_plot(shap_values1[i], X1_test, plot_type="bar", 
                              plot_size=None, show=False, max_display=5)

        plt.show()
        plt.clf()

        # Save the classification report and top features
        output_report = pd.DataFrame(report)
        output_report.to_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/{datatype}/{datatype}_class_report.csv')

        ft_cols = pd.DataFrame(return_sorted_cols, columns=['ft_cols'])
        ft_cols.to_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/{datatype}/{datatype}_ft_cols.csv')

    return return_sorted_cols, report

# Import data subtypes from the module
from import_subtypes import *

# Set RGB color mapping for the classes
rest_colors = ['#f6511d', '#ffb400', '#7fb800', '#0d2c54']
cmap = plt_colors.ListedColormap(np.array(rest_colors))

# Perform classification and SHAP analysis on resting-state data
rest_cols, rest_report = shapit(sample1_rest.loc[:100], sample2_rest.loc[:100], 'rest', showfig=True)

# Utility functions for processing lists (not related to the main analysis)
def common_values(list1, list2):
    """
    Finds common values between two lists.
    
    Args:
    list1 (list): First list of values.
    list2 (list): Second list of values.
    
    Returns:
    list: Common values between the two lists.
    """
    set1 = set(list1)
    set2 = set(list2)
    return list(set1.intersection(set2))

def remove_duplicates(input_list):
    """
    Removes duplicate values from a list.
    
    Args:
    input_list (list): List with potential duplicates.
    
    Returns:
    str: String representation of the list with duplicates removed.
    """
    return "[" + ",".join("'" + str(x) + "'" for x in list(set(input_list))) + "]"
