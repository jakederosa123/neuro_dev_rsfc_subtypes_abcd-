#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/')
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')

from functions import *
import shap
import xgboost
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from matplotlib import colors as plt_colors

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # makes the plots HD in the notebook")
plt.rcParams["figure.autolayout"] = True # enables tigh layout. Better multiplots

# class names
classes = ['1', '2', '3', '4']

def get_rgb(hex_codes):
    rgb_values = []
    for hex_code in hex_codes:
        r = int(hex_code[1:3], 16) / 255
        g = int(hex_code[3:5], 16) / 255
        b = int(hex_code[5:7], 16) / 255
        rgb_values.append((r, g, b))
    
    return rgb_values

def classy(data, holdout, datatype, showfig=None):
    #est = bootit(df1_rest, 1/5)
    
    class_names = list(data.Subtype.unique())
    
    def sampling(data, perc): #1 
        import random
        seed = random.randint(0, 1000000)
        np.random.seed(seed)
        total_n=data.shape[0] # total number of subjects 
        this_index = np.random.choice(total_n, int(total_n * perc), replace=False) # randomly select the subjects' index with replacement for this bootstrapt 
        sampled_data = data.iloc[this_index] # use the selected index to slice the data for this bootstrapt
        return sampled_data   

    #test = sampling(data, .2)
    test = data
    X = test.drop(['ID', 'Subtype'], axis=1)
    xcols = X.columns
    y = np.array(test['Subtype'])
    
    
    test_holdout = holdout
    X_holdout = test_holdout.drop(['ID', 'Subtype'], axis=1)
    y_holdout = np.array(test_holdout['Subtype'])
    
    
    from scipy import stats
    X = pd.DataFrame(stats.zscore(X), columns = xcols)
    X_holdout = pd.DataFrame(stats.zscore(X_holdout), columns = xcols)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=10, 
                                                        stratify=y
                                                       )
    
    from sklearn.utils.class_weight import compute_sample_weight
    #sample_weights = compute_sample_weight(class_weight='balanced',y=y_train)
    sample_weights = compute_sample_weight(class_weight='balanced',y=y)
    

    params = {'max_depth': 3, 
              'learning_rate': 0.2, 
              'subsample': 0.8,
              'colsample_bytree': 0.6, 
              'colsample_bylevel': 0.9, 
              'n_estimators': 1000,
              'objective':'multi:softmax',
              'tree_method':"hist"
             }
    
    from hgboost import hgboost
    hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0, top_cv_evals=10, random_state=None, verbose=0)
    params = hgb.xgboost(X, y, pos_label=1)['params']
        
    cls = xgboost.XGBClassifier(**params)
    #cls = xgboost.XGBClassifier(objective='multi:softmax')
    #cls = SVC(kernel='rbf', probability=True, C=1, gamma='scale',random_state=1)
    
    #cls.fit(X_train, y_train, sample_weight=sample_weights)
    cls.fit(X, y, sample_weight=sample_weights)
    # compute SHAP values
    #explainer = shap.KernelExplainer(model=cls.predict_proba, data = X_train, link = "logit")
    explainer = shap.TreeExplainer(cls)
    #shap_values = explainer.shap_values(X)
    shap_values = explainer.shap_values(X_holdout)
    
    #predictions = cls.predict(X_test)
    predictions = cls.predict(X_holdout)
    
    if showfig is not None:
        #print(accuracy_score(y_test, predictions))
        #print(confusion_matrix(y_test, predictions))
        #print(classification_report(y_test, predictions))
        print(accuracy_score(y_holdout, predictions))
        print(confusion_matrix(y_holdout, predictions))
        print(classification_report(y_holdout, predictions))
    #report = classification_report(y_test, predictions, output_dict=True)
    report = classification_report(y_holdout, predictions, output_dict=True)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=1)
    
    global cmap
    plt.subplot(1,2, 1)
    shap.summary_plot(shap_values, X.values, plot_type="bar", 
                      class_names= class_names, class_inds='original',
                      feature_names = X.columns, color = cmap, 
                      plot_size=None, show=False)
    plt.savefig(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/{datatype}/{datatype}_shap_plot.png',dpi=300, bbox_inches='tight')
    plt.show()

    from matplotlib import pyplot as plt
    n_top_features = 15

    #plt.subplot(1,2,2)
        
    sorted_idx = cls.feature_importances_.argsort()[::-1]
    sorted_values = -np.sort(-cls.feature_importances_[sorted_idx][:n_top_features])
    sorted_cols = X_test.columns[sorted_idx][:n_top_features].tolist()
    #plt.barh(sorted_cols,  sorted_values,  align='center')
    #plt.spines['left'].set_color('black')
    #plt.spines['bottom'].set_color('black')
    #plt.xlabel('Relative Importance')
    #plt.title('Feature Importances')
    #plt.gca().invert_yaxis()
    
    if showfig is not None:
        plt.show()
    else:
        plt.clf()
    
    
    #return explainer, shap_values, X, X_test, sorted_cols, report
    return explainer, shap_values, X_holdout, X_holdout, sorted_cols, report


def shapit(data, holdout, datatype, showfig=None):
    
    explainer1, shap_values1, X1, X1_test, return_sorted_cols, report = classy(data, holdout, datatype, showfig=showfig)
    
    
    if showfig is not None: 
        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(wspace=0.4)

        for i in range(len(data.Subtype.unique())):
            if len(data.Subtype.unique()) < 5:
                plt.subplot(2,2,i+1)
            else: 
                plt.subplot(2,3,i+1)
                
            shap.summary_plot(shap_values1[i], X1.values, feature_names = X1.columns,
                              plot_size=None, show=False, max_display=5)
            plt.savefig(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/{datatype}/{datatype}_ind_shap_plot.png',dpi=300, bbox_inches='tight')
    
        plt.show()
        plt.clf()

        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(wspace=0.4)

        for i in range(len(data.Subtype.unique())):
            if len(data.Subtype.unique()) < 5:
                plt.subplot(2,2,i+1)
            else: 
                plt.subplot(2,3,i+1)

            shap.summary_plot(shap_values1[i], X1_test, plot_type="bar", 
                              plot_size=None, show=False, max_display=5)

        plt.show()
        plt.clf()
        
        output_report = pd.DataFrame(report)
        output_report.to_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/{datatype}/{datatype}_class_report.csv')

        ft_cols = pd.DataFrame(return_sorted_cols, columns=['ft_cols'])
        ft_cols.to_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/classification_analysis/outputs/{datatype}/{datatype}_ft_cols.csv')

    return return_sorted_cols, report


# In[5]:


from import_subtypes import *


# In[6]:


# set RGB tuple per class
rest_colors = ['#f6511d', '#ffb400', '#7fb800', '#0d2c54']
cmap = plt_colors.ListedColormap(np.array(rest_colors))

rest_cols, rest_report = shapit(sample1_rest.loc[:100], sample2_rest.loc[:100], 'rest', showfig=True)
#df2_rest_report = shapit(sample2_rest, showfig=True)    


# In[ ]:


colors = get_rgb(['#f05d5e', '#0f7173', '#fdca40', '#272932', '#240169'])
cmap = plt_colors.ListedColormap(np.array(colors))

dti_cols, dti_report = shapit(sample1_dti.loc[:100], sample2_dti.loc[:100], 'dti', showfig=True)
#df2_dti_report = shapit(sample2_dti, showfig=True)


# In[ ]:


colors = get_rgb(['#7d5fff', '#ff6700', '#21CEA6', '#8d0801'])
cmap = plt_colors.ListedColormap(np.array(colors))

smri_cols, smri_report = shapit(sample1_smri.loc[:120], sample2_smri.loc[:120], 'smri', showfig=True)
#df2_smri_report = shapit(sample2_smri, showfig=True)


# In[ ]:


#df1_smri_report


# In[ ]:


def common_values(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return list(set1.intersection(set2))

def remove_duplicates(input_list):
    return "[" + ",".join("'" + str(x) + "'" for x in list(set(input_list))) + "]"


# In[ ]:


#rest_common = common_values(df1_rest_report, df2_rest_report)
#print('Rest Common')
#len(rest_common)


# In[ ]:


#remove_duplicates(df1_rest_report + df2_rest_report)


# In[ ]:


#dti_common = common_values(df1_dti_report, df2_dti_report)
#print('DTI Common')
#len(dti_common)


# In[ ]:


#remove_duplicates(df1_dti_report + df2_dti_report)


# In[ ]:


#smri_common = common_values(df1_smri_report, df2_smri_report)
#print('SMRI Common')


# In[ ]:


#remove_duplicates(df1_smri_report + df2_smri_report)


