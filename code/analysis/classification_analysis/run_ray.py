import sys
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/')
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')

from functions import *
import shap
import xgboost
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from matplotlib import colors as plt_colors

def get_rgb(hex_codes):
    rgb_values = []
    for hex_code in hex_codes:
        r = int(hex_code[1:3], 16) / 255
        g = int(hex_code[3:5], 16) / 255
        b = int(hex_code[5:7], 16) / 255
        rgb_values.append((r, g, b))
    
    return rgb_values


import shap

import xgboost
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

import time
import numpy as np
import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb

from tune_sklearn import TuneSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import ray


def readit(path):
    #data = reduce_memory_usage(pd.concat(pd.read_csv(path, iterator=True, chunksize=100000), ignore_index=True))
    data = pd.concat(pd.read_csv(path, iterator=True, chunksize=100000), ignore_index=True)
    return data

from import_subtypes import *

#rest_cluster_path = '/pl/active/banich/studies/abcd/data/clustering/rest/sample_1_rest_051922/Output/Results/sample_1_rest_051922_Full_Subtypes.csv'
#df1_rest = readit(rest_cluster_path).drop(['Unnamed: 0', 'Q', 'Key'], axis =1)
#df1_rest['Subtype'] = df1_rest['Subtype']
#print(df1_rest.groupby('Subtype').size())


#df1_rest['Subtype'] = np.where(df1_rest['Subtype'] == 1, 1,
#                            np.where(df1_rest['Subtype'] == 2, 3, 
#                                     np.where(df1_rest['Subtype'] == 3, 2,
#                                              np.where(df1_rest['Subtype'] == 4, 4, False))))

data = sample1_rest_combined
holdout = sample2_rest_combined

colors = ['#f6511d', '#ffb400', '#7fb800', '#0d2c54'] #rest
#colors = ['#f05d5e', '#0f7173', '#fdca40', '#272932', '#240169'] #dti
#colors = ['#7d5fff', '#ff6700', '#21CEA6', '#8d0801']
cmap = plt_colors.ListedColormap(np.array(colors))

#init ray and attach it to local node ray instance
#ray.init(address='auto')
ray.init("ray://127.0.0.1:10001", namespace="ray") 

    
# function to perform the tuning using tune-search library
# add function decorator
@ray.remote

def tune_search_tuning():
    
    global data, holdout
    
    colors = ['#7d5fff', '#ff6700', '#21CEA6', '#8d0801']
    cmap = plt_colors.ListedColormap(np.array(colors))
    
    #test = df1_rest#.iloc[:1000]
    #X = test.iloc[:, 2:93]
    #y = np.array(test['Subtype'])
    #from scipy import stats
    #X = stats.zscore(X)
    #x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=10)
    #print("Shapes - X_train: ", x_train.shape, ", X_val: ", x_val.shape, ", y_train: ", y_train.shape, ", y_val: ", y_val.shape)

    #test = sampling(data, .2)
    test = data
    X = test.drop(['ID', 'Subtype'], axis=1)
    xcols = X.columns
    y = np.array(test['Subtype'])
    
    test_holdout = holdout
    X_holdout = test_holdout.drop(['ID', 'Subtype'], axis=1)
    y_holdout = np.array(test_holdout['Subtype'])
    
    class_names = list(data.Subtype.unique())
    
    from sklearn.utils.class_weight import compute_sample_weight
    #sample_weights = compute_sample_weight(class_weight='balanced',y=y_train)
    sample_weights = compute_sample_weight(class_weight='balanced',y=y)
    
    # numpy arrays are not accepted in params attributes, 
    # so we use python comprehension notation to build lists
    params = {'max_depth': [5,10,15,20,25,30],
              'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4, .5],
              'subsample': [0.5 + x / 100 for x in range(10, 50, 10)],
              #'gamma': [0, 0.25, 0.5, 1.0],
              'kernel': ['linear', 'rbf'],
              'colsample_bytree': [0.5 + x / 100 for x in range(10, 50, 10)],
              'colsample_bylevel': [0.5 + x / 100 for x in range(10, 50, 10)],
              'n_estimators': [100, 500, 1000, 2000],
              #'early_stopping_rounds': 25
              #'num_class': [10]
              }
    
    # define the booster classifier indicating the objective as 
    # multiclass "multi:softmax" and try to speed up execution
    # by setting parameter tree_method = "hist"
    xgbclf = xgb.XGBClassifier(objective="multi:softmax",
                               tree_method="hist")

    # replace RamdomizedSearchCV by TuneSearchCV
    # n_trials sets the number of iterations (different hyperparameter combinations)
    # that will be evaluated
    # verbosity can be set from 0 to 3 (debug level).
    tune_search = TuneSearchCV(estimator=xgbclf,
                               param_distributions=params,
                               scoring='accuracy',
                               n_jobs=24,
                               n_trials=25,
                               verbose=1)

    # perform hyperparameter tuning
    #tune_search.fit(x_train, y_train)
    tune_search.fit(X, y)
    print("cv results: ", tune_search.cv_results_)

    best_combination = tune_search.best_params_
    print("Best parameters:", best_combination)

    # evaluate accuracy based on the test dataset
    predictions = tune_search.predict(X_holdout)

    accuracy = metrics.accuracy_score(y_holdout, predictions)
    print("Accuracy: ", accuracy)
    
    params = tune_search.best_params_
    parmas_df = pd.DataFrame.from_dict(params, orient='index', columns=['value'])
    parmas_df.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/rest_include/best_params.csv')
    
    cls = xgboost.XGBClassifier(**params)
    #cls = xgboost.XGBClassifier(objective='multi:softmax')
    #cls = SVC(kernel='rbf', probability=True, C=1, gamma='scale',random_state=1)
    
    #cls.fit(X_train, y_train, sample_weight=sample_weights)
    cls.fit(X, y, sample_weight=sample_weights)
    # compute SHAP values
    #explainer = shap.KernelExplainer(model=cls.predict_proba, data = X_train, link = "logit")
    explainer = shap.TreeExplainer(cls)
    #shap_values = explainer.shap_values(X)
    shap_values = explainer.shap_values(X_holdout, check_additivity=False)
    
    #predictions = cls.predict(X_test)
    predictions = cls.predict(X_holdout)
    
    
    #print(accuracy_score(y_test, predictions))
     #print(confusion_matrix(y_test, predictions))
     #print(classification_report(y_test, predictions))
    #print(accuracy_score(y_holdout, predictions))
    #print(confusion_matrix(y_holdout, predictions))
    #print(classification_report(y_holdout, predictions))
    #report = classification_report(y_test, predictions, output_dict=True)
    report = classification_report(y_holdout, predictions, output_dict=True)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=1)
    

    plt.subplot(1,2, 1)
    shap.summary_plot(shap_values, X_holdout.values, plot_type="bar", 
                      class_names=class_names, class_inds='original',
                      feature_names = X.columns, color = cmap, 
                      plot_size=None, show=False)
    plt.savefig(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/rest_include/rest_shap_plot.png',dpi=300, bbox_inches='tight')
 #   plt.show()

    from matplotlib import pyplot as plt
    n_top_features = 15

    #plt.subplot(1,2,2)
    
    sorted_idx = cls.feature_importances_.argsort()[::-1]
    sorted_values = -np.sort(-cls.feature_importances_[sorted_idx][:n_top_features])
    sorted_cols = X_holdout.columns[sorted_idx][:n_top_features].tolist()    
    
    output_report = pd.DataFrame(report)
    output_report.to_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/rest_include/rest_class_report.csv')
    
    ft_cols = pd.DataFrame(sorted_cols, columns=['ft_cols'])
    ft_cols.to_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/rest_include/rest_ft_cols.csv')
    
    
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(wspace=0.4)
    
    for i in range(len(data.Subtype.unique())):
        
        if len(data.Subtype.unique()) < 5:
            plt.subplot(2,2,i+1)
        else: 
            plt.subplot(2,3,i+1)
        
        shap.summary_plot(shap_values[i], X_holdout.values, feature_names = X_holdout.columns, plot_size=None, show=False, max_display=10)
        plt.savefig(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/rest_include/rest_ind_shap_plot.png',dpi=300, bbox_inches='tight')
        
    return best_combination

if __name__ == '__main__':

    start_time = time.time()

    # create the task
    remote_clf = tune_search_tuning.remote()

    # get the task result
    best_params = ray.get(remote_clf)

    stop_time = time.time()
    print("Stopping at :", stop_time)
    print("Total elapsed time: ", stop_time - start_time)

    print("Best params from main function: ", best_params)
    
    
# Rest Best Params
#  {'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.7, 'kernel': 'linear', 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9, # 'n_estimators': 1000}
