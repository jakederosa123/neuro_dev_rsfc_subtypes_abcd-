"""
This script is designed to import and handle subtype-specific data. The script involves reading subtype data, performing subtype-specific operations, 
and preparing this data for further analysis.

Overview of the Script:
-----------------------
1. Importing Libraries:
   - The necessary Python libraries are imported for data manipulation, file handling, and possibly specific 
     libraries related to subtypes.

2. Loading Subtype Data:
   - The script reads in subtype data from files, which could involve specifying file paths, formats, and reading options 
     to correctly load the data.

3. Subtype-Specific Processing:
   - After loading, the script may perform operations specific to the subtypes, such as filtering, labeling, or 
     deriving new subtype-specific columns.

4. Defining Functions:
   - Functions are defined to encapsulate common subtype-related operations, making the script more modular 
     and reusable. These functions might include specific subtype processing tasks or utilities to facilitate subtype handling.

5. Preparing Subtype Data for Analysis:
   - The script ensures that the subtype data is in the correct format and structure for further analysis, which may be performed 
     in separate scripts or notebooks.

6. Outputting Subtype Data:
   - If applicable, the script might save the processed subtype data to new files, print summaries to the console, 
     or return data structures for use in other parts of the project.

Purpose:
--------
The primary goal of this script is to reliably import and preprocess subtype-specific data, ensuring it is clean and ready 
for analysis or further processing in the workflow.
"""

#!/usr/bin/env python
# coding: utf-8
from functions import *

#--------------------------------------------------------------------------------------------------------------------------------

def read_subtype(path):
    
    data = readit(path).drop(['Unnamed: 0', 'Q', 'Key'], axis =1)
    data['Subtype'] = data['Subtype']
    #print(data.groupby('Subtype').size())
    return data

#--------------------------------------------------------------------------------------------------------------------------------

def rename_subtype(data, x1, x2, x3, x4, x5=None):
    
    if x5 is not None: 
        data['Subtype'] = np.where(data['Subtype'] == 1, x1,
                                   np.where(data['Subtype'] == 2, x2, 
                                            np.where(data['Subtype'] == 3, x3,
                                                     np.where(data['Subtype'] == 4, x4,
                                                              np.where(data['Subtype'] == 5, x5, False)))))
    else: 
        data['Subtype'] = np.where(data['Subtype'] == 1, x1,
                                   np.where(data['Subtype'] == 2, x2, 
                                            np.where(data['Subtype'] == 3, x3,
                                                     np.where(data['Subtype'] == 4, x4, False))))
    
        
        
    return data

#--------------------------------------------------------------------------------------------------------------------------------

path = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/subtype_data/baseline'


#--------------------------------------------------------------------------------------------------------------------------------
#Rest
#--------------------------------------------------------------------------------------------------------------------------------

# -------------- Baseline Include

sample1_rest_include = read_subtype(f'{path}/sample1_rest_baseline_07282023_include/Output/Results/sample1_rest_baseline_07282023_include_Full_Subtypes.csv')
sample1_rest_include = rename_subtype(sample1_rest_include, 3, 1, 2, 4) 
sample1_rest_include_idsub = sample1_rest_include[['ID', 'Subtype']]

sample2_rest_include = read_subtype(f'{path}/sample2_rest_baseline_07282023_include/Output/Results/sample2_rest_baseline_07282023_include_Full_Subtypes.csv')
sample2_rest_include = rename_subtype(sample2_rest_include, 4, 3, 1, 2) 
sample2_rest_include_idsub = sample2_rest_include[['ID', 'Subtype']]

full_sample_rest_include = read_subtype(f'{path}/fullsample_rest_baseline_07282023_include/Output/Results/fullsample_rest_baseline_07282023_include_Full_Subtypes.csv')
full_sample_rest_include = rename_subtype(full_sample_rest_include, 3, 1, 2, 4) 
full_sample_rest_include_idsub = full_sample_rest_include[['ID', 'Subtype']]

print("Resting State Include: sample1_rest_include, sample2_rest_include, full_sample_rest_include /n sample1_rest_include_idsub, sample2_rest_include_idsub, sample2_rest_idsub, full_sample_rest_include_idsub")



# --------------  Baseline Combined

sample1_rest_combined = read_subtype(f'{path}/sample1_rest_baseline_07282023_combined/Output/Results/sample1_rest_baseline_07282023_combined_Full_Subtypes.csv')
sample1_rest_combined = rename_subtype(sample1_rest_combined, 3, 1, 2, 4) 
sample1_rest_combined_idsub = sample1_rest_combined[['ID', 'Subtype']]

sample2_rest_combined = read_subtype(f'{path}/sample2_rest_baseline_07282023_combined/Output/Results/sample2_rest_baseline_07282023_combined_Full_Subtypes.csv')
sample2_rest_combined = rename_subtype(sample2_rest_combined, 2, 4, 3, 1) 
sample2_rest_combined_idsub = sample2_rest_combined[['ID', 'Subtype']]

full_sample_rest_combined = read_subtype(f'{path}/fullsample_rest_baseline_07282023_combined/Output/Results/fullsample_rest_baseline_07282023_combined_Full_Subtypes.csv')
full_sample_rest_combined = rename_subtype(full_sample_rest_combined, 3, 1, 2, 4) 
full_sample_rest_combined_idsub = full_sample_rest_combined[['ID', 'Subtype']]

print("Resting State Combined: sample1_rest_combined, sample2_rest_combined, full_sample_rest_combined /n sample1_rest_combined_idsub, sample2_rest_combined_idsub, full_sample_rest_combined_idsub")

# --------------  Baseline Dont Include

full_sample_rest_dont_include = read_subtype(f'{path}/fullsample_rest_baseline_07282023_dont_include/Output/Results/fullsample_rest_baseline_07282023_dont_include_Full_Subtypes.csv')
full_sample_rest_dont_include = rename_subtype(full_sample_rest_dont_include, 4, 1, 2, 3) 
full_sample_rest_dont_include_idsub = full_sample_rest_dont_include[['ID', 'Subtype']]

print("Resting State Dont Include: full_sample_rest_dont_include /n full_sample_rest_dont_include_idsub")
