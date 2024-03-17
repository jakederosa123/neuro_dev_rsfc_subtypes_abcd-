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

#--------------------------------------------------------------------------------------------------------------------------------
#DTI
#-------------------------------------------------------------------------------------------------------------------------------

#sample1_dti = read_subtype(data_path +'/sample1_dti_baseline_02052023/Output/Results/sample1_dti_baseline_02052023_Full_Subtypes.csv')
#sample2_dti = read_subtype(data_path +'/sample2_dti_baseline_02052023/Output/Results/sample2_dti_baseline_02052023_Full_Subtypes.csv')

#sample1_dti = rename_subtype(sample1_dti, 2, 1, 4, 3, 5) # do not unhash 
#sample2_dti = rename_subtype(sample2_dti, 5, 3, 1, 4, 2)
                        
#sample1_dti = pd.merge(sample1_dti, include, on = 'ID')
#sample2_dti = pd.merge(sample2_dti, include, on = 'ID')

#sample1_dti_idsub = sample1_dti[['ID', 'Subtype']]
#sample2_dti_idsub = sample2_dti[['ID', 'Subtype']]
#print("DTI Baseline: sample1_dti, sample2_dti /n sample1_dti_idsub, sample2_dti_idsub, sample2_dti_idsub")

#--------------------------------------------------------------------------------------------------------------------------------
#SMRI
#--------------------------------------------------------------------------------------------------------------------------------
#sample1_smri = read_subtype(data_path +'/sample1_smri_baseline_02052023/Output/Results/sample1_smri_baseline_02052023_Full_Subtypes.csv')
#sample2_smri = read_subtype(data_path + '/sample2_smri_baseline_02052023/Output/Results/sample2_smri_baseline_02052023_Full_Subtypes.csv')

#sample1_smri = rename_subtype(sample1_smri, 2, 1, 4, 3) 
#sample2_smri = rename_subtype(sample2_smri, 2, 4, 1, 3)

#sample1_smri = pd.merge(sample1_smri, include, on = 'ID')
#sample2_smri = pd.merge(sample2_smri, include, on = 'ID')
                       
#sample1_smri_idsub = sample1_smri[['ID', 'Subtype']]
#sample2_smri_idsub = sample2_smri[['ID', 'Subtype']]

#print("SMRI Baseline: sample1_smri, sample2_smri, sample1_smri_idsub, sample2_smri_idsub")
