"""
This script is designed to handle the import and initial processing of data. The exact operations include reading 
data, cleaning or transforming the data, and preparing it for further analysis or use in other scripts.

Overview of the Script:
-----------------------
1. Importing Libraries:
   - The necessary Python libraries are imported for data manipulation, file handling, and possibly logging or other utilities.

2. Loading Data:
   - The script reads in data from files, databases, or other sources. This could involve specifying file paths, 
     formats, and reading options to correctly load the data.

3. Data Cleaning and Transformation:
   - After loading, the script may perform cleaning operations such as handling missing values, converting data types, 
     or filtering specific rows/columns. Transformation steps might include merging datasets, normalizing data, or 
     deriving new columns.

4. Defining Functions:
   - Functions are defined to encapsulate common operations, making the script more modular and reusable. 
     These functions might include specific data processing tasks or utilities to facilitate data import.

5. Preparing Data for Analysis:
   - The script ensures that the data is in the correct format and structure for further analysis, which may be performed 
     in separate scripts or notebooks.

6. Outputting Data:
   - If applicable, the script might save the processed data to new files, print summaries to the console, 
     or return data structures for use in other parts of the project.

Purpose:
--------
The primary goal of this script is to reliably import and preprocess data, ensuring it is clean and ready for analysis 
or further processing in the workflow.
"""

#!/usr/bin/env python
# coding: utf-8

# In[21]:


from functions import *


bpath = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/behavioral/'
#participants = pd.read_csv("/pl/active/banich/studies/abcd/data/clustering/participants.tsv", sep='\t').rename(columns={'participant_id': 'ID'})
#participants['ID'] = participants['ID'].str.replace('sub-', '')
#demos = participants[['ID', 'sex', 'race_ethnicity', 'age', 'income', 'participant_education', 'parental_education']]

# CBCL -------------------------------------
cbcl_vars = ['anxious_depressed','withdrawn_depressed', 'somatic_complaints', 'social_problems',
              'thought_problems', 'attention_problems', 'rule_breaking', 'agressive',
              'internalizing', 'externalizing', 'total']

cbcl_vars = ['anxious_depressed','withdrawn_depressed', 'somatic_complaints', 'social_problems',
              'thought_problems', 'attention_problems', 'rule_breaking', 'agressive',
              'internalizing', 'externalizing', 'total']

cbcl_r_vars = [cbcl_vars + "_r" for cbcl_vars in cbcl_vars]

cbcl_base_t =pd.read_csv(bpath + 'cbcl_t_baseline.csv')
cbcl_b_t = cbcl_base_t[['ID'] + cbcl_r_vars]

#cbcl_y2_t = pd.read_csv(bpath + 'cbcl_t_y2.csv')

cbcl_base_factors=pd.read_csv(bpath + 'rest_cbcl_cfa_scores.csv').iloc[:, 1:]

print("CBCL: cbcl_base_t, cbcl_base_factors ")
#print("CBCL: cbcl_y2_t")

# Nback -------------------------------------
nback = pd.read_csv(bpath + 'nback_behavior_baseline.csv')
nback_vars = list(nback.iloc[:, 1:].columns)
print("fMRI nback: nback")

# SST  --------------------------------------
sst=pd.read_csv(bpath + 'sst_baseline_r.csv')
print('SST: sst')

# NIH w PC ----------------------------------
pcs=pd.read_csv(bpath + 'nih_baseline.csv')
cog_vars = list(pcs.iloc[:, 1:].columns)
print("NIH: pcs")

# Stroop --------------------------------------
stroop_beh=pd.read_csv(bpath + 'stroop_baseline.csv')
#stroop_beh_vars = list(stroop_beh.iloc[:, 1:].columns)
stroop_beh_vars = ['Stroop_interf_acc_all_r','Happy_Acc_Eq_r','Angry_Acc_Eq_r']
print("Stroop Behavioral: stroop")

# Matrix --------------------------------------
matrix=pd.read_csv(bpath + 'pearson_baseline.csv')
print("Matrix: Reasoning")

# Demos --------------------------------------
demos_baseline = pd.read_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/demos_baseline.csv')
print("All demos: demos_baseline")

# UPPS
upps_factors = pd.read_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/behavioral/rest_upps_cfa_scores.csv').iloc[:, 1:]
print("UPPS Factors: upps_factors")

# Combine all cognitive measures
cog_all = (pcs.merge(matrix, on='ID', how='outer').drop_duplicates('ID', keep='first'))
cog_all_names = cog_all.iloc[:, 1:].columns.to_list()
print("Cognitive: cog_all")

# Combine all behavior measures
beh_all = (cbcl_b_t.merge(cbcl_base_factors, on='ID', how='outer', suffixes=('_left', '_right'))
 .drop_duplicates('ID', keep='first'))
beh_all_names = beh_all.iloc[:, 1:].columns.to_list()

cog_ef_factors = pd.read_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/behavioral/rest_cog_ef_factors.csv')
print("COG EF Factors: cog_ef_factors")


  


