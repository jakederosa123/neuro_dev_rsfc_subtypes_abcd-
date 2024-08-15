"""
This script is designed to perform feature selection using the Boruta algorithm. The Boruta algorithm is a wrapper
built around random forest classification, specifically designed to identify all relevant features in the dataset,
including those that are weakly relevant.

Overview of the Script:
-----------------------
1. Importing Libraries:
   - The necessary Python libraries are imported for data manipulation, model building, and feature selection.
     This includes libraries such as `pandas`, `numpy`, and `BorutaPy` from the scikit-learn ecosystem.

2. Loading and Preparing Data:
   - The script reads in the dataset and performs any necessary preprocessing steps such as handling missing values,
     encoding categorical variables, and splitting the data into features and target variables.

3. Defining and Applying the Boruta Algorithm:
   - The core of the script involves initializing the Boruta algorithm, fitting it to the data, and selecting important features.
     The algorithm runs multiple iterations of the random forest classifier, comparing the importance of real features
     with that of shadow features (randomly shuffled copies of the original features).

4. Evaluating and Saving the Results:
   - The script evaluates the results of the Boruta feature selection process, identifying which features were deemed important.
     These results are then saved to a file or printed for further analysis.

5. Visualizing Feature Importance:
   - The script includes steps to visualize the importance of the selected features using plots or charts.

Purpose:
--------
The primary goal of this script is to identify all relevant features in the dataset using the Boruta algorithm,
ensuring that the final model includes only the features that contribute meaningfully to the prediction task.
"""


#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8
import sys

#from hgboost_function import *

sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/')
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')

import joblib

from import_data import *
from import_subtypes import *


from sklearn.preprocessing import StandardScaler
import pandas as pd

# instantiate StandardScaler
scaler = StandardScaler()

sample1_rest = sample1_rest_include
sample2_rest = sample2_rest_include
full_sample = full_sample_rest_include

rest_cols = sample1_rest.iloc[:, 2:].columns
#dti_cols = sample1_dti.iloc[:, 2:].columns
#smri_cols = sample1_smri.iloc[:, 2:].columns

#sample1_rest[rest_cols] = scaler.fit_transform(sample1_rest[rest_cols])
#sample2_rest[rest_cols] = scaler.fit_transform(sample2_rest[rest_cols])
#full_sample[rest_cols] = scaler.fit_transform(full_sample[rest_cols])

sample1_subtype_dummies = pd.get_dummies(sample1_rest['Subtype'], prefix='Subtype')
sample2_subtype_dummies = pd.get_dummies(sample2_rest['Subtype'], prefix='Subtype')
full_sample_subtype_dummies = pd.get_dummies(full_sample['Subtype'], prefix='Subtype')


# In[5]:


cbcl_vars = cbcl_b_t.iloc[:, 1:].columns.to_list()
#cbcl_r_vars = [cbcl_vars + "_r" for cbcl_vars in cbcl_vars]
cbcl_base_factors_vars = cbcl_base_factors.iloc[:, 1:].columns.to_list()
cog_all_vars = cog_all.iloc[:, 1:].columns.to_list()
upps_vars = upps_factors.iloc[:, 1:].columns.to_list()

all_vars = cbcl_vars + cbcl_base_factors_vars + upps_vars + cog_all_vars + upps_vars + stroop_beh_vars

#Rest Samples
sample1 = sample1_rest#.merge(sample1_dti.drop('Subtype', axis=1), on='ID').merge(sample1_smri.drop('Subtype', axis=1), on='ID')
sample2 = sample2_rest#.merge(sample2_dti.drop('Subtype', axis=1), on='ID').merge(sample2_smri.drop('Subtype', axis=1), on='ID')

#DTI Samples
#sample1 = sample1_dti.merge(sample1_rest.drop('Subtype', axis=1), on='ID').merge(sample1_smri.drop('Subtype', axis=1), on='ID')
#sample2 = sample2_dti.merge(sample2_rest.drop('Subtype', axis=1), on='ID').merge(sample2_smri.drop('Subtype', axis=1), on='ID')

#SMRI Samples
#sample1 = sample1_smri.merge(sample1_dti.drop('Subtype', axis=1), on='ID').merge(sample1_rest.drop('Subtype', axis=1), on='ID')
#sample2 = sample2_smri.merge(sample2_dti.drop('Subtype', axis=1), on='ID').merge(sample2_rest.drop('Subtype', axis=1), on='ID')

sample1_all = (sample1
 .merge(cog_all, on='ID', how='outer')
 .merge(cog_ef_factors, on='ID', how='outer')
 #.merge(demo_one_hot, on='ID', how='outer')
 .merge(cbcl_b_t, on='ID', how='outer')
 .merge(cbcl_base_factors, on='ID', how='outer')
 .merge(upps_factors, on='ID', how='outer')
 .merge(stroop_beh, on='ID', how='outer')
 .drop_duplicates('ID', keep='first')
 .dropna(subset=['Subtype'])
 #.query('abcd_site != "site22"')
)

sample2_all = (sample2
 .merge(cog_all, on='ID', how='outer')
 .merge(cog_ef_factors, on='ID', how='outer')
 #.merge(demo_one_hot, on='ID', how='outer')
 .merge(cbcl_b_t, on='ID', how='outer')
 .merge(cbcl_base_factors, on='ID', how='outer')
 .merge(upps_factors, on='ID', how='outer')
 .merge(stroop_beh, on='ID', how='outer')
 .drop_duplicates('ID', keep='first')
 .dropna(subset=['Subtype'])
 #.query('abcd_site != "site22"')
)


full_sample_all = (full_sample
 .merge(cog_all, on='ID', how='outer')
 .merge(cog_ef_factors, on='ID', how='outer')
 #.merge(demo_one_hot, on='ID', how='outer')
 .merge(cbcl_b_t, on='ID', how='outer')
 .merge(cbcl_base_factors, on='ID', how='outer')
 .merge(upps_factors, on='ID', how='outer')
 .merge(stroop_beh, on='ID', how='outer')
 .drop_duplicates('ID', keep='first')
 .dropna(subset=['Subtype'])
 #.query('abcd_site != "site22"')
)

sample1_all.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/prediction//sample1_all.csv', index=False)
sample2_all.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/prediction//sample2_all.csv', index=False)
full_sample_all.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/prediction//full_sample_all.csv', index=False)

rois = sample1.iloc[:, 2:].columns.to_list()


# In[6]:


#all_vars=['CommonEF', 'UpdatingSpecific', 'Intelligence', 'pc1_new_r', 'pc2_new_r','pc3_new_r',
#          'total_r', 'internalizing_r', 'externalizing_r', 'withdrawn_depressed_r', 
#          'somatic_complaints_r', 'anxious_depressed_r', 'rule_breaking_r', 'agressive_r', 
#          'attention_problems_r', 'thought_problems_r', 'social_problems_r', 'predmeditation', 
#         'sensation_seeking', 'negative_urgency','positive_urgency', 
#          'Stroop_interf_acc_all_r','Happy_Acc_Eq_r','Angry_Acc_Eq_r']


# In[11]:

all_vars=['perserverance', 'LMT_r', 'RAVLT_r']


def borutit(itr, var):
    global full_sample_all
    
        
    import pandas as pd
    from sklearn.preprocessing import PowerTransformer
    
    from BorutaShap import BorutaShap
    from sklearn.model_selection import train_test_split
    fts=[]
    for i in range(itr):

        df = full_sample_all.drop(['ID', 'Subtype'], axis=1)[[var] + rois].dropna()
        y = df[[var]].values
        X = df.drop(var, axis=1)

        X_train_ft, X_test_ft, y_train_ft, y_test_ft = train_test_split(X, y, test_size=0.2)
        # Importing core libraries
        # Classifier/Regressor
        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=0, objective='reg:squarederror')
        # Creates a BorutaShap selector for regression
        selector = BorutaShap(model=model, importance_measure = 'shap', classification = False)
        # Fits the selector
        selector.fit(X = X_train_ft, y = y_train_ft, n_trials = 100, sample = True, verbose = True)

        # n_trials -> number of iterations for Boruta algorithm
        # sample -> samples the data so it goes faster
        #Display features to be removed
        fts.append(selector.accepted)

    def combine_and_deduplicate(arrays):
        result = []
        for array in arrays:
            for element in array:
                if element not in result:
                    result.append(element)
        return result

    len(combine_and_deduplicate(fts))
    new_fts = combine_and_deduplicate(fts)
    features = pd.DataFrame(new_fts, columns=['features'])
    features.to_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/prediction/varimp_boruta_fts/{var}.csv', index=False)
    
    


# In[ ]:


for i in all_vars:
    borutit(10, i)






