#!/usr/bin/env python
# coding: utf-8


import numpy as np
from numpy import mean

import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve


# In[4]:


from functions import *
from import_data import *


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    #tpr =  TP/(TP + FN) # sensitivity - true positive rate
    #fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    tpr =  TP/(TP + FP) # sensitivity - true positive rate
    fpr = TP/(TP+FN) # 1-specificity - false positive rate
    
    
    
    return tpr, fpr


def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a treshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list

def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    
    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()
    
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    #sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    #plt.xlim(-0.05, 1.05)
    #plt.ylim(-0.05, 1.05)
    plt.xlim(0.05, 1.01)
    plt.ylim(0.05, 1.01)
    #plt.xlabel("False Positive Rate")
    #plt.ylabel("True Positive Rate")
    plt.xlabel("Recall")
    plt.ylabel("Precision")


# In[6]:


def rocit(data):

    df_ova_aux_list = []
    avg_roc_ova_auc_list = []
    roc_ova_auc_score_output_df_list = []

    df_ovo_aux_list = []
    avg_roc_ovo_auc_list = []
    avg_roc_ovo_auc_list_all = []
    roc_ovo_auc_score_output_df_list = []
    
    avg_roc_ova_auc_list_final = []
    avg_roc_ovo_auc_list_all_final = []
    for net in [1,2,3,4]:

        test = data.copy()

        X = test.iloc[:, 2:93]
        y = np.array(test.Subtype)
        
        from scipy import stats
        X = stats.zscore(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=10)
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train)

        #model_multiclass = RandomForestClassifier(n_estimators = 50, criterion = 'gini')
        #model_multiclass = SVC(kernel='rbf', probability=True, C=1, gamma='scale',random_state=1)
        model_multiclass = xgboost.XGBClassifier(objective="multi:softmax")

        model_multiclass.fit(X_train, y_train)
        y_pred = model_multiclass.predict(X_test)
        y_proba = model_multiclass.predict_proba(X_test)
        classes = model_multiclass.classes_

        # Plots the Probability Distributions and the ROC Curves One vs Rest
        sns.set_style("ticks")

        #plt.figure(figsize = (14, 8))
        bins = [i/20 for i in range(20)] + [1]
        roc_auc_ovr = {}
        
        avg_roc_auc_list_final = []
        for i in range(len(classes)):
            # Gets the class
            c = classes[i]

            # Prepares an auxiliar dataframe to help with the plots
            df_aux = X_test.copy()
            df_aux['class'] = [1 if y == c else 0 for y in y_test]
            df_aux['prob'] = y_proba[:, i]
            df_aux['inter'] = c
            df_aux['net'] = net
            df_aux = df_aux.reset_index()

            precision, recall, _ = precision_recall_curve(df_aux['class'], df_aux['prob'])
            avg_roc_auc_list_final.append(auc(recall, precision))
            avg_roc_auc_list_df = pd.DataFrame([['clear', 'maintain', 'replace', 'suppress']] + [avg_roc_auc_list_final]).T
            
            df_ova_aux_list.append(df_aux)
            
        avg_roc_auc_list_df['net'] = net
        avg_roc_ova_auc_list_final.append(avg_roc_auc_list_df)

        
        # Compares with sklearn (average only)
        # "Macro" average = unweighted mean
        roc_auc_score_output = roc_auc_score(y_test, y_proba, labels = classes, multi_class = 'ovr', average = 'macro')
        roc_auc_score_output_df = pd.DataFrame([roc_auc_score_output]).rename({0:'roc'}, axis = 1)
        roc_auc_score_output_df['net'] = net
        roc_ova_auc_score_output_df_list.append(roc_auc_score_output_df)

        
        classes_combinations = []
        class_list = list(classes)
        for i in range(len(class_list)):
            for j in range(i+1, len(class_list)):
                classes_combinations.append([class_list[i], class_list[j]])
                classes_combinations.append([class_list[j], class_list[i]])

        # Plots the Probability Distributions and the ROC Curves One vs ONe
        #plt.figure(figsize = (30, 15))
        bins = [i/20 for i in range(20)] + [1]
        roc_auc_ovo = {}
        
        comp = []
        avg_roc_ovo_auc_list_final = []
        for i in range(len(classes_combinations)):
            # Gets the class
            comb = classes_combinations[i]
            c1 = comb[0]
            c2 = comb[1]
            c1_index = class_list.index(c1)
            #title = c1 + ' vs ' + c2

            # Prepares an auxiliar dataframe to help with the plots
            df_aux2 = X_test.copy()
            df_aux2['class'] = y_test
            df_aux2['prob'] = y_proba[:, c1_index]
            df_aux2['inter'] = str(c1) + '_' + str(c2)
            df_aux2['net'] = net
            # Slices only the subset with both classes
            df_aux2 = df_aux2[(df_aux2['class'] == c1) | (df_aux2['class'] == c2)]
            df_aux2['class'] = [1 if y == c1 else 0 for y in df_aux2['class']]
            df_aux2 = df_aux2.reset_index()
            
            precision, recall, _ = precision_recall_curve(df_aux2['class'], df_aux2['prob'])
            avg_roc_ovo_auc_list_final.append(auc(recall, precision))
            comp.append(str(c1)+"_"+str(c2))
            avg_roc_ovo_auc_list_df = pd.DataFrame([comp] + [avg_roc_ovo_auc_list_final]).T
            df_ovo_aux_list.append(df_aux2)

        avg_roc_ovo_auc_list_df['net'] = net
        avg_roc_ovo_auc_list_all_final.append(avg_roc_ovo_auc_list_df)
        #plt.tight_layout()
  
        
        # Compares with sklearn (average only)
        # "Macro" average = unweighted mean
        roc_ovo_auc_score_output = roc_auc_score(y_test, y_proba, labels = classes, multi_class = 'ovo', average = 'macro')
        roc_ovo_auc_score_output_df = pd.DataFrame([roc_ovo_auc_score_output]).rename({0:'roc'}, axis = 1)
        roc_ovo_auc_score_output_df['net'] = net
        roc_ovo_auc_score_output_df_list.append(roc_ovo_auc_score_output_df)
        
        
    df_ova_aux_list_sub_f = pd.concat(df_ova_aux_list)
    df_ovo_aux_list_sub_f = pd.concat(df_ovo_aux_list)
    
    roc_ova_auc_score_output_df_list_f = pd.concat(roc_ova_auc_score_output_df_list)
    avg_roc_ova_auc_list_f = pd.concat(avg_roc_ova_auc_list_final)
    
    
    roc_ovo_auc_score_output_df_list_f = pd.concat(roc_ovo_auc_score_output_df_list)
    avg_roc_ovo_auc_list_all_f = pd.concat(avg_roc_ovo_auc_list_all_final)
    
    
    return df_ova_aux_list_sub_f, df_ovo_aux_list_sub_f, roc_ova_auc_score_output_df_list_f, avg_roc_ova_auc_list_f, roc_ovo_auc_score_output_df_list_f, avg_roc_ovo_auc_list_all_f


# In[7]:


def bootit(data): #1 
    '''ramdonly select sample data with replacement for a single bootstrap'''
    #np.random.seed(seed)
    total_n=data.shape[0] # total number of subjects 
    this_index = np.random.choice(total_n, int(total_n * 2/3), replace=True) # randomly select the subjects' index with replacement for this bootstrapt 
    sampled_data = data.iloc[this_index] # use the selected index to slice the data for this bootstrapt
    return sampled_data   


# In[8]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import numpy as np
from numpy import mean

import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score


# In[9]:

for rd in df1_rest, df2_rest, df1_rest_2year, df2_rest_2year:
    
    if rd is df1_rest:
        sample = 'sample_1_baseline'
    elif rd is df2_rest:
        sample = 'sample_2_baseline'
    elif rd is df1_rest_2year:
        sample = 'sample_1_year2'
    elif rd is df2_rest_2year:
        sample = 'sample_2_year2'
        
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []

    for i in list(range(1,50)):

        ao,bo,co,do,eo,fo = rocit(bootit(rd))
        #ao,bo,co,do,eo,fo = rocit(bootit(rd))
        ao['boot'] = i; a.append(ao)
        bo['boot'] = i;b.append(bo)
        #co['boot'] = i;c.append(co)
        do['boot'] = i;d.append(do)
        #eo['boot'] = i;e.append(eo)
        fo['boot'] = i;f.append(fo)


    # In[10]:


    mo = pd.concat(a).drop(['index'], axis =1)


    # In[11]:


    m = pd.concat(b).drop(['index'], axis =1)


    # In[12]:


    ova_comps = pd.concat(d)
    ova_comps.columns = ['inter', 'roc', 'net', 'boot']
    ova_comps = ova_comps.groupby(['net', 'inter'], as_index=False).agg({'roc':['mean','std']}).droplevel(level=1, axis=1).round(3)
    otca = ova_comps.iloc[:, 0:3]
    otca = otca.pivot_table(index=['inter'],columns='net',values='roc')


    # In[13]:


    ovo_comps = pd.concat(f)
    ovo_comps.columns = ['inter', 'roc', 'net', 'boot']
    ovo_comps = ovo_comps.groupby(['net', 'inter'], as_index=False).agg({'roc':['mean','std']}).droplevel(level=1, axis=1).round(3)
    otc = ovo_comps.iloc[:, 0:3]
    otc = otc.pivot_table(index=['inter'],columns='net',values='roc')


    # In[49]:


    avg_roc_ovo_auc_list_all_final = []

    mn = m
    roc_auc_ovo = {}
    plt.figure(figsize = (55, 30))
    avg_roc_ovo_auc_list_final=[]
    comp = []
    for i in range(len(uni(mn['inter']))):

        if i == 0:
            ii = "1_2";  ops_cols = ['#F0180A','#0A5AF0']
        elif i == 1:
            ii = "1_3";  ops_cols = ['#F08B0A','#0A5AF0']
        elif i == 2:
            ii = "1_4";  ops_cols = ['#6DAE45','#0A5AF0']
        elif i == 3:
            ii = "2_1";  ops_cols = ['#0A5AF0','#F0180A']
        elif i == 4:
            ii = "2_3";  ops_cols = ['#F08B0A','#F0180A']
        elif i == 5:
            ii = "2_4";  ops_cols = ['#6DAE45','#F0180A']
        elif i == 6:
            ii = "3_1";  ops_cols = ['#0A5AF0','#F08B0A']
        elif i == 7:
            ii = "3_2";  ops_cols = ['#F0180A','#F08B0A']
        elif i == 8:
            ii = "3_4";  ops_cols = ['#6DAE45','#F08B0A']
        elif i == 9:
            ii = "4_1";  ops_cols = ['#0A5AF0','#6DAE45']
        elif i == 10:
            ii = "4_2";  ops_cols = ['#F0180A', '#6DAE45']
        elif i == 11:
            ii = "4_3";  ops_cols = ['#F08B0A','#6DAE45']


        df_aux3 = mn[mn['inter'] == ii].reset_index()

        c1 = ii.split("_",1)[0]
        c2 = ii.split("_",1)[1]
        title = "Subtype: " + c1 + ' vs ' + c2
        # Plots the probability distribution for the class and the rest

        bins = [i/20 for i in range(20)] + [1]

        sns.set(font_scale = 2.8)
        sns.set_style("ticks")
        ax = plt.subplot(4, 6, i+1)
        plt.subplots_adjust(wspace=0.5)
        plt.subplots_adjust(hspace=0.5)

        sns.histplot(x = "prob", data = df_aux3, hue = 'class', palette = ops_cols, bins = bins)
        ax.set_title(title)
        ax.legend([f"{c1}", f"{c2}"])
        ax.set_xlabel(f"P(x = {c1})")
        #ax.legend(loc='upper right')
   
        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(4, 6, i+13)
        plt.subplots_adjust(wspace=0.5)
        plt.subplots_adjust(hspace=0.5)
        
        tpr, fpr = get_all_roc_coordinates(np.array(df_aux3['class']), np.array(df_aux3['prob']))
        plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
        ax_bottom.set_title(f"PR Curve: {c1} vs {c2}")
     # Calculates the ROC AUC OvO
        #roc_auc_ovo[title] = roc_auc_score(df_aux3['class'], df_aux3['prob'])

        precision, recall, _ = precision_recall_curve(df_aux3['class'], df_aux3['prob'])
        avg_roc_ovo_auc_list_final.append(auc(recall, precision))
        comp.append(str(c1)+"_"+str(c2))
        avg_roc_ovo_auc_list_df = pd.DataFrame([comp] + [avg_roc_ovo_auc_list_final]).T

    avg_roc_ovo_auc_list_df['net'] = 1
    avg_roc_ovo_auc_list_all_final.append(avg_roc_ovo_auc_list_df)
    plt.savefig(f'/pl/active/banich/studies/abcd/data/clustering/analysis/classification_output/ovo/{sample}_ovo_roc.png')
    plt.tight_layout()



    ovo_final_output = pd.concat(avg_roc_ovo_auc_list_all_final)
    ops_comp_list = uni(ovo_final_output[0])
    ovo_final_output.columns = ['op', 'roc', 'net']
    ovo_final_output['roc'] = ovo_final_output['roc'].astype(float)
    ovo_final_output = pd.DataFrame(np.array(ovo_final_output.pivot_table(index=['op'],columns='net',values='roc'))).round(3)
    ovo_final_output['operation'] = ops_comp_list 
    ovo_final_output.columns = ['ovo_auc', 'comp']
    ovo_final_output['sample'] = sample
    ovo_final_output.to_csv(f'/pl/active/banich/studies/abcd/data/clustering/analysis/classification_output/ovo/{sample}_ovo_final_output.csv')


    # In[48]:


    roc_ova_auc_score_output_df_list = []
    avg_roc_ova_auc_list_final = []

    mno = mo
    #mno = mo[mo['net'] == n]
    roc_auc_ovr = {}
    plt.figure(figsize = (30, 12))

    avg_roc_auc_list_final = []
    for i in [1,2,3,4]:

        if i == 1:
            ii = 1; ops_cols = ['#B2BEB5','#0A5AF0']
        elif i == 2:
            ii = 2; ops_cols = ['#B2BEB5','#F0180A']
        elif i == 3:
            ii = 3; ops_cols = ['#B2BEB5','#F08B0A']
        elif i == 4:
            ii = 4; ops_cols = ['#B2BEB5','#6DAE45']

        df_aux4 = mno[mno['inter'] == ii].reset_index()
        c = ii

        sns.set(font_scale = 2.3)
        sns.set_style("ticks")

    #for xx in range(len(uni(mno['inter']))):
        title = 'Subtype ' + str(i) #+ ": " + str(c) + " vs rest"
        # Plots the probability distribution for the class and the rest

        ax = plt.subplot(2, 4, i)
        plt.subplots_adjust(wspace=0.5)
        plt.subplots_adjust(hspace=0.5)
        sns.histplot(x = "prob", data = df_aux4, hue = 'class', palette = ops_cols, ax = ax, bins = bins)
        ax.set_title(title)
        ax.legend([f"{c}", "rest"])
        ax.set_xlabel(f"P(x = Subtype {c})")

        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, 4, i+4)
        plt.subplots_adjust(wspace=0.5)
        plt.subplots_adjust(hspace=0.5)
        tpr, fpr = get_all_roc_coordinates(np.array(df_aux4['class']), np.array(df_aux4['prob']))
        plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
        ax_bottom.set_title(f"PR Curve: {c} vs rest")

      # Calculates the ROC AUC OvR
        #roc_auc_ovr[c] = roc_auc_score(df_aux4['class'], df_aux4['prob'])

        precision, recall, _ = precision_recall_curve(df_aux4['class'], df_aux4['prob'])
        avg_roc_auc_list_final.append(auc(recall, precision))
        avg_roc_auc_list_df = pd.DataFrame([['1', '2', '3', '4']] + [avg_roc_auc_list_final]).T
    # avg_roc_auc_list_final.append(roc_auc_ovr)
    '''
    # Displays the ROC AUC for each class
    avg_roc_auc = 0
    i = 0
    for k in roc_auc_ovr:
        avg_roc_auc += roc_auc_ovr[k]
        i += 1
       # print(f"{k} ROC AUC OvR: {roc_auc_ovr[k]:.4f}")
    # print(f"average ROC AUC OvR: {avg_roc_auc/i:.4f}")

    avg_roc_auc_list_final = []
    avg_roc_auc = 0
    i = 0
    for k in roc_auc_ovr:
        avg_roc_auc += roc_auc_ovr[k]
        i += 1
        avg_roc_auc_list_final.append(roc_auc_ovr[k])
    #print(f"average ROC AUC OvR: {avg_roc_auc/i:.4f}")
    avg_roc_auc_list_df = pd.DataFrame([['clear', 'maintain', 'replace', 'suppress']] + [avg_roc_auc_list_final]).T
    avg_roc_auc_list_df['net'] = netc
    avg_roc_ova_auc_list_final.append(avg_roc_auc_list_df)
    '''
    avg_roc_auc_list_df['net'] = 1
    avg_roc_ova_auc_list_final.append(avg_roc_auc_list_df)
    plt.savefig(f'/pl/active/banich/studies/abcd/data/clustering/analysis/classification_output/ova/{sample}_ova_roc.png')
    plt.tight_layout()


    # In[57]:


    ova_final_output = pd.concat(avg_roc_ova_auc_list_final)
    ops_list = uni(ova_final_output[0])
    ova_final_output.columns = ['op', 'roc', 'net']
    ova_final_output = pd.DataFrame(np.array(ova_final_output.pivot_table(index=['op'],columns='net',values='roc'))).round(3)
    ova_final_output.index = ['S1', 'S2', 'S3', 'S4']
    ova_final_output.columns = ['ovr_auc']
    ova_final_output['sample'] = sample
    ova_final_output.to_csv(f'/pl/active/banich/studies/abcd/data/clustering/analysis/classification_output/ova/{sample}_ova_final_output.csv')






