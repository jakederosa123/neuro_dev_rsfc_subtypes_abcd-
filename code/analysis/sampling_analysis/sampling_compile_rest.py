#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/')
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')


# In[10]:
import pandas as pd
import numpy as np
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os


def reduce_memory_usage(df, verbose=False):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

def maxcor(data1, data2):
    
    def get_redundant_pairs(df):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_top_abs_correlations(df):
        au_corr = df.unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:df.shape[0]]

    df1_cor = data1.drop(['ID'], axis =1).groupby('Subtype').mean()
    df2_cor = data2.drop(['ID'], axis =1).groupby('Subtype').mean()
    cors = pd.DataFrame(np.corrcoef(df1_cor ,df2_cor))
    cors = cors.iloc[:, 0:len(data1.Subtype.unique())]
    
    del df1_cor, df2_cor
    
    t = pd.DataFrame(get_top_abs_correlations(cors)).reset_index()
    t['pair'] = t['level_0'].astype(str) + "-" + t['level_1'].astype(str) 
    t = t.iloc[:, 2:]
    t= t.loc[:len(data2.Subtype.unique())-1]
    t.columns = ['max_cor', 'match']
    mat = t

    return mat
    
def showmax(data1, data2, cols):
    
    def normalize(df, cols):
        
        result = df.copy()
        
        for feature_name in cols:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

    from scipy import stats

    test1 = data1.copy()
    test1[cols] = stats.zscore(test1[cols])

    test2 = data2.copy()
    test2[cols] = stats.zscore(test2[cols])
    
    mat = maxcor(test1, test2)
    return mat


import itertools

names=[]
for name in itertools.combinations(range(100),2):
    names.append(name)


# In[5]:


#%rm -rf /pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/pair_cor_outputs/sample


# In[6]:


def task(i):
    
    global perc_df, sample_df, sample_name
    perc_df = pd.read_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/perc_df_output/perc_df.csv').iloc[:, 1:]
    sample_df = pd.read_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/perc_df_output/sample_df.csv').iloc[:, 1:]
    
    sample_cor_1 = (perc_df.query('iter == ' + '"' + str(i[0]) + '"')
                .merge(sample_df, on ='ID'))[['ID', 'Subtype'] + list(func_cor_cols)]

    sample_cor_2 = (perc_df.query('iter == ' + '"' + str(i[1]) + '"')
                    .merge(sample_df, on ='ID'))[['ID', 'Subtype'] + list(func_cor_cols)]

    mat = showmax(sample_cor_1, sample_cor_2 , func_cor_cols)

    del sample_cor_1, sample_cor_2

    mat['pair'] = [i] * mat.shape[0]
    mat['sample'] = sample_name
    outfolder = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/pair_cor_outputs/'+str(perc_df.perc.unique()[0])+'/'
    os.system(f'mkdir -p {outfolder}')
    mat.to_csv(outfolder + sample_name + '_' + str(i) + '.csv') 
    #print('finished' + " " + sample_name + " "+ str(i))
    
def perc_max_cors(p):
    
    from random import random
    from time import sleep
    from multiprocessing.pool import Pool
    
    global outdf, names, sample_df, sample_name

    #p = outdf.perc.unique()[0]
    #sample_df = sample_df
    #sample_name = sample_name
    perc_df = outdf.query('perc == ' + '"' + str(p) + '"')
    perc_df.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/perc_df_output/perc_df.csv')
    sample_df.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/perc_df_output/sample_df.csv')
    
   # from timeit import default_timer as timer
    #start = timer()
    # protect the entry point
    if __name__ == '__main__':
        # create and configure the process pool
        with Pool(24) as pool:
            # execute tasks in order
            output = pool.map(task, names)

    #end = timer()
    #print(end - start)
    #print('finished sample -- ' + str(p) + " " + sample_name)
    
    
def pcor_compile(i):
    
    pcor_paths = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/pair_cor_outputs/'+str(i)+'/*'
    pcor_paths_sorted = sorted(glob(pcor_paths, recursive = True))
    
    pcor_list = []
    for path in pcor_paths_sorted:
        pcor_d = pd.read_csv(path, index_col=None, header=0)
        pcor_list.append(pcor_d)

    pcor_df = pd.concat(pcor_list).drop('Unnamed: 0', axis=1)
    
    del pcor_list
    
    pcor_df.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/pair_cor_outputs/'+str(i)+'/final_cor_'+str(i)+'.csv')
    
    #del pcor_list
    pcor_df = pd.read_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/pair_cor_outputs/'+str(i)+'/final_cor_'+str(i)+'.csv')
    
    sns.set_style("white")
    g = sns.displot(data=pcor_df, x='max_cor',hue='sample', kind='kde', palette=ops_cols, linewidth=3)

    (g.set_axis_labels("Sample Perentage Pairwise Max Cors", "Density",  weight='bold')
          .set_titles("{col_name}", weight='bold')
          .despine(left=False))  
    g.tight_layout()

    outfolder = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/'+ str(i) +'/'
    os.system(f'mkdir -p {outfolder}/')
    g.savefig(outfolder + str(i) +'_sample_pair_cor.png')  
    plt.clf()
    
def get_full_max_cors(perc_df, sample_df_subs, sample_name):
     
    global func_cor_cols
    
    perc_cors = []    
    #percs = list(perc_df['iter'])
    for i in range(100):
        sample_cor_1 = (perc_df.query('iter == ' + '"' + str(i) + '"')
                     .merge(sample_df_subs.drop('Subtype', axis=1), on ='ID'))[['ID', 'Subtype'] + list(func_cor_cols)]

        mat = showmax(sample_cor_1, sample_df_subs, func_cor_cols)
        mat['pair'] = [i] * mat.shape[0]
        mat['sample'] = sample_name

        perc_cors.append(mat)

        del mat
        
    cors = reduce_memory_usage(pd.concat(perc_cors))

    return cors

def getARI(data1, data2, sample, perc, itr):
    perc_df = data1.query('perc == ' + '"' + str(perc) + '"')
    perc_df = perc_df.query('iter == ' + '"' + str(itr) + '"')

    perc_df = (perc_df.rename({'Subtype':'Subtype_i'}, axis=1)
     .merge(data2, on = 'ID')
    )

    from sklearn.metrics.cluster import adjusted_rand_score
    ARI = (pd.DataFrame([adjusted_rand_score(perc_df['Subtype_i'], perc_df['Subtype'])])
           .rename({0:'ARI'}, axis=1))

    ARI['sample'] = sample
    ARI['perc'] = perc
    ARI['iter'] = itr

    return ARI


def show_hist(data, var, xlab, sample, save=None):
    sns.set_style("white")
    ops_cols = ["#4645E2", "#2CE26F"]
    fig, ax = plt.subplots()
    g = sns.histplot(data=data, x=var, hue='sample', 
                     palette=ops_cols, 
                     hue_order=["sample1", "sample2"],
                     binwidth=.025,
                     fill = False,
                     kde = True
                    )

    g.set(ylabel = "Count", xlabel = xlab)
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    
    if save is not None:
        outfolder = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/'+ str(sample) +'/'
        os.system(f'mkdir -p {outfolder}/')
        ax.figure.savefig(outfolder + str(sample) +save+'.png') 
    
    else: 
        plt.show()

# In[2]:


# In[7]:


# Whats the purpose of these analyses:
'''
to show the reliability of these subtypes at different subtypes
- show ARI to the downsamples to downsampled full sample is irrelevant we want downsampled to the max of what we present in the paper.
- potentially calculate the magniniude of the differences? -> most likely analyses for followup papers. 
'''


# In[8]:


"""
temp_txtfile = np.arange(0,n_straps,1).astype(str)
np.savetxt(f'{outfolder}/Output/Results/temp_txtfile.txt', temp_txtfile,fmt='%s')
#parallel call 
tstart = time()

pcall = 'bash ' + params['pype_path'] + '/BagPype/scripts/bagpype_bash_parallel.sh %s' % batch
#pcall = 'bash ' + params['pype_path'] + '/BagPype/scripts/bagpype_bash_parallel.sh %s %s' % (batch, resolution_parameter)
subprocess.check_call(pcall, shell=True)
tend = time()
print('Time to run %s bootstraps: ' % n_straps, tend-tstart)
#Bagging 
"""


# In[9]:


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

df1_rest = read_subtype(f'{path}/sample1_rest_baseline_07282023_include/Output/Results/sample1_rest_baseline_07282023_include_Full_Subtypes.csv')
df1_rest = rename_subtype(df1_rest, 3, 1, 2, 4) 

df2_rest = read_subtype(f'{path}/sample2_rest_baseline_07282023_include/Output/Results/sample2_rest_baseline_07282023_include_Full_Subtypes.csv')
df2_rest = rename_subtype(df2_rest, 4, 3, 1, 2) 

full_sample_rest = read_subtype(f'{path}/fullsample_rest_baseline_07282023_include/Output/Results/fullsample_rest_baseline_07282023_include_Full_Subtypes.csv')
full_sample_rest = rename_subtype(full_sample_rest, 3, 1, 2, 4) 


# In[15]:


sample1 = pd.read_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/subtype_data/baseline/sample1_rest_baseline_07282023_include/sample1_rest_baseline_07282023_include.csv')
sample2 = pd.read_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/subtype_data/baseline/sample2_rest_baseline_07282023_include/sample2_rest_baseline_07282023_include.csv')
full_sample =  pd.read_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/subtype_data/baseline/fullsample_rest_baseline_07282023_include.csv')


# In[34]:


func_cor_cols = sample1.iloc[:, 2:].columns

paths = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/rest_include/*/*/Output/Results/*_Full_Subtypes.csv'
paths_sorted = sorted(glob(paths, recursive = True))

samp_nums = [i.split('sampling/', 2)[1] for i in paths_sorted]
samp_nums = [i.split('/', 2)[1] for i in samp_nums]
samp_nums = list(pd.DataFrame(samp_nums)[0].unique())

new_paths_sorted =[]
for i in samp_nums:
    new_paths_sorted.append(list(filter(lambda k: i in k, paths_sorted)))


# In[35]:


filtered_paths = []
for i in new_paths_sorted:
    filtered_paths.append([path for path in i if 'full_sample' not in path])
    
new_paths_sorted = filtered_paths 
del filtered_paths 


# In[38]:


def show_hist(data, var, xlab, sample, save=None):
    sns.set_style("white")
    ops_cols = ["#4645E2", "#2CE26F"]
    fig, ax = plt.subplots()
    g = sns.histplot(data=data, x=var, hue='sample', 
                     palette=ops_cols, 
                     hue_order=["sample1", "sample2"],
                     binwidth=.025,
                     fill = False,
                     kde = True
                    )

    g.set(ylabel = "Count", xlabel = xlab)
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    
    if save is not None:
        outfolder = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/'+ str(sample) +'/'
        os.system(f'mkdir -p {outfolder}/')
        ax.figure.savefig(outfolder + str(sample) +save+'.png') 
    
    else: 
        plt.show()
    


# In[39]:


for r in range(len(new_paths_sorted)):
    
    class boot_outputs:

        def __init__(self, path):
            self.path = path
            self.df = pd.read_csv(self.path)[['ID', 'Subtype', 'Q']]

    #from timeit import default_timer as timer
    #start = timer()
    boot_class_list = *map(boot_outputs, new_paths_sorted[r]),
    #end = timer()
    #print(end - start)
    
    for i in range(len(boot_class_list)):

        boot_class_list[i].name = (boot_class_list[i].path
                                   .split('/Results/')[1]
                                   .split('_Full_Subtypes.csv')[0])

        boot_class_list[i].df['sample'] = boot_class_list[i].name.split('_')[0]
        boot_class_list[i].df['perc'] = boot_class_list[i].name.split('_')[1]
        boot_class_list[i].df['iter'] = boot_class_list[i].name.split('_')[2]
    
    out_list = []
    for i in range(len(boot_class_list)):
        out_list.append(boot_class_list[i].df)

    outdf = reduce_memory_usage(pd.concat(out_list))
    sample_perc = outdf.perc.unique()[0]

    #-----------------------------------------------------------------------------------------------------------------------------------------------
    # Sample Q
    ops_cols = ["#4645E2", "#2CE26F"]
    sns.set_style("white")
    
    q_df = outdf.drop_duplicates(['sample', 'iter'])
    g = sns.displot(data=q_df, x='Q',hue='sample', kind='kde', palette=ops_cols, linewidth=3)
    (g.set_axis_labels("Sample Perentage Q", "Density",  weight='bold')
          .set_titles("{col_name}", weight='bold')
          .despine(left=False))  

    g.tight_layout()
    outfolder = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/'+ str(sample_perc) +'/'
    os.system(f'mkdir -p {outfolder}/')
    g.savefig(outfolder + str(sample_perc) +'_q.png')  
    plt.clf()
    q_df.drop_duplicates(['sample', 'iter']).to_csv(outfolder + str(sample_perc) +'_q_df.csv')
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    # Sample Pair Max Cor
    par_mats = []
    for n in names:
        for i, j in zip([sample1, sample2], ['sample1', 'sample2']):

            sample_cor_1 = (outdf.query('iter == ' + '"' + str(n[0]) + '"')
                        .merge(i, on ='ID'))[['ID', 'Subtype'] + list(func_cor_cols)]

            sample_cor_2 = (outdf.query('iter == ' + '"' + str(n[1]) + '"')
                            .merge(i, on ='ID'))[['ID', 'Subtype'] + list(func_cor_cols)]

            mat = showmax(sample_cor_1, sample_cor_2 , func_cor_cols)

            del sample_cor_1, sample_cor_2

            mat['pair'] = [n] * mat.shape[0]
            mat['sample'] = j

            outfolder = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/pair_cor_outputs/'+str(outdf.perc.unique()[0])+'/'
            os.system(f'mkdir -p {outfolder}')
            mat.to_csv(outfolder + j + '_' + str(n) + '.csv') 
            par_mats.append(mat)
            del mat
            
    pcor_df = pd.concat(par_mats)
      
    sns.set_style("white")
    ops_cols = ["#4645E2", "#2CE26F"]
    fig, ax = plt.subplots()
    g = sns.histplot(data=pcor_df, x='max_cor', hue='sample', 
                     palette=ops_cols, 
                     hue_order=["sample1", "sample2"],
                     binwidth=.025,
                     fill = False,
                     kde = True
                    )

    g.set(ylabel = "Count", xlabel = 'Max Correlation')
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    
    outfolder = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/'+ str(sample_perc) +'/'
    os.system(f'mkdir -p {outfolder}/')
    ax.figure.savefig(outfolder + str(sample_perc)+'_sample_pair_cor.png') 
    
    pcor_df['perc'] = sample_perc
    pcor_df.to_csv(outfolder + str(sample_perc)+'_sample_pair_cor_df.csv')
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    # Full Sample Max Cor
    perc_full_cors = []
    for p in list(outdf.perc.unique()):
        perc_df = outdf.query('perc == ' + '"' + str(p) + '"')

        for i,j in zip([df1_rest, df2_rest], ['sample1', 'sample2']):

            perc_full_max_cor = get_full_max_cors(perc_df, i, j)
            perc_full_max_cor['perc'] = p
            perc_full_cors.append(perc_full_max_cor)

    perc_full_cors_df = reduce_memory_usage(pd.concat(perc_full_cors))

    for i in perc_full_cors_df.perc.unique():

        sns.set_style("white")
        ops_cols = ["#4645E2", "#2CE26F"]
        perc_full_cors_df_sampled = perc_full_cors_df.query('perc == ' + '"' + str(i) + '"')
        g = sns.displot(data=perc_full_cors_df_sampled, x='max_cor',hue='sample', kind='kde', palette=ops_cols, linewidth=3)

        (g.set_axis_labels("Sample Perentage Full Max Cors", "Density",  weight='bold')
              .set_titles("{col_name}", weight='bold')
              .despine(left=False))  
        g.tight_layout()

        os.system(f'mkdir -p /pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/full_cor_outputs/{i}/')
        perc_full_cors_df_sampled.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/full_cor_outputs/'+str(i)+'/final_full_cor_'+str(i)+'.csv')

        outfolder = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/'+ str(i) +'/'
        os.system(f'mkdir -p {outfolder}/')
        g.savefig(outfolder + str(i) +'_full_cor.png')  
        plt.clf()
        
        perc_full_cors_df_sampled.to_csv(outfolder + str(i) +'_full_cor_df.csv')
            
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    #ARI
    perc_full_ari = []
    for p in list(outdf.perc.unique()):
        for r in range(100):
            for i,j in zip([df1_rest, df2_rest], ['sample1', 'sample2']):
                perc_full_ari.append(getARI(outdf, i, j, p, r))

    perc_full_ari_df = reduce_memory_usage(pd.concat(perc_full_ari))
    
    import seaborn as sns
    ops_cols = ["#4645E2", "#2CE26F"]
    for i in perc_full_ari_df.perc.unique():

        sns.set_style("white")
        perc_full_ari_df_sampled = perc_full_ari_df.query('perc == ' + '"' + str(i) + '"')
        g = sns.displot(data=perc_full_ari_df_sampled, x='ARI',hue='sample', kind='kde', palette=ops_cols, linewidth=3)

        (g.set_axis_labels("Sample Perentage ARI", "Density",  weight='bold')
              .set_titles("{col_name}", weight='bold', size=200)
              .despine(left=False))

        g.tight_layout()

        os.system(f'mkdir -p /pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/full_ari_outputs/{i}/')
        perc_full_ari_df_sampled.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/tests/full_ari_outputs/'+str(i)+'/final_full_ari_'+str(i)+'.csv')

        outfolder = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/'+ str(i) +'/'
        os.system(f'mkdir -p {outfolder}/')
        g.savefig(outfolder + str(i) +'_ari.png')  
        plt.clf()
             
        perc_full_ari_df_sampled.to_csv(outfolder + str(i) +'_ari_df.csv')


# In[40]:


def mean_sd(data, columns, group_by, pivot = None, pivot_index=None, pivot_column=None):
    
    data_mean_std = (data[columns]
                            .groupby(group_by)
                            .describe()
                            .T
                            .reset_index(level=0, drop=True)
                            .T
                            .reset_index()
                            .loc[:, list(itertools.chain.from_iterable([group_by, ['mean', 'std']]))]
                            )

    data_mean_std['msd'] = round(data_mean_std['mean'], 3).astype(str) + ' (' + round(data_mean_std['std'], 2).astype(str) + ')'
    
    if pivot is not None:
        
        data_mean_std = (data_mean_std
                        .drop(['mean', 'std'], axis=1)
                        .pivot(index = pivot_index, values= 'msd', columns = pivot_column)
                         .reset_index(drop=True)
                       )
    
    
    return data_mean_std


def sample_violin(data, var, xlab, ylim, save=None):
    ops_cols = ["#4645E2", "#2CE26F"]
    g = sns.catplot(data=data, 
                    x="perc", 
                    y=var, 
                    hue="sample", 
                    #col = 'perc',
                    kind="violin",
                    height=5, aspect=1.2,
                    bw=.25, cut=0, split=True,  
                    palette=ops_cols)

    labs = []
    for i in list(data.perc.unique() * 100):
        labs.append(str(int(i)) + "%")

    g.set_axis_labels("Percentage", xlab)
    g.set_xticklabels(labs)
    g.set_titles("{col_name} {col_var}")
    g.set(ylim=ylim)
    g.despine(left=True)
    
    if save is not None:
        g.savefig(save)
    
from scipy.stats import ttest_ind
from scipy import stats

def sample_kruskal(data, var):
    
    tests = []
    for i in data.perc.unique():
        test = (
            pd.DataFrame(
            stats.kruskal(
                data.query('sample == "sample1" and perc == ' + str(i) +'')[var],
                data.query('sample == "sample2" and perc == ' + str(i) +'')[var]))
            .T
            .rename({0:"t", 1:"p"}, axis=1)
        )

        test['perc'] = i

        tests.append(test.loc[:, ['perc', 't', 'p']])
        
    output = pd.concat(tests).round(3).reset_index(drop=True)
    
    return output

def combine_tests(msd_df, kruskal_df):
    output = (pd.concat([msd_df, kruskal_df], axis=1)
     .loc[:, ["perc", 'sample1', "sample2", 't', 'p']]
    )
    
    return output
    
    
def get_compiled_data(paths):
    paths_list = []
    
    for i in sorted(glob(paths, recursive = True)):
        paths_list.append(pd.read_csv(i).iloc[:, 1:])
    
    df = pd.concat(paths_list)
        
    return df

 


# In[41]:


os.system(f'mkdir -p /pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/all_samples')


# In[86]:


full_cor_df = get_compiled_data('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/*/*_full_cor_df.csv')
full_cor_df.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/output_dfs/full_cor_df.csv')

full_cor_mean_std = mean_sd(full_cor_df, ['perc', 'sample', 'max_cor'], ['perc', 'sample'], 
                            pivot = True, pivot_index='perc', pivot_column='sample')

full_cor_mean_std = combine_tests(full_cor_mean_std, sample_kruskal(full_cor_df, 'max_cor'))
full_cor_mean_std.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/all_samples/full_cor_msd.csv')


# In[87]:


sample_violin(full_cor_df, "max_cor",  "Max Correlations to Full Samples", (0, 1),
             save = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/all_samples/full_cor.png')


# In[88]:


pair_cor_df = get_compiled_data('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/*/*_pair_cor_df.csv')
pair_cor_df.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/output_dfs/pair_cor_df.csv')


# In[89]:


pair_cor_mean_std = mean_sd(pair_cor_df, ['perc', 'sample', 'max_cor'], ['perc', 'sample'], 
                            pivot = True, pivot_index='perc', pivot_column='sample')


pair_cor_mean_std = combine_tests(pair_cor_mean_std, sample_kruskal(pair_cor_df, 'max_cor'))
pair_cor_mean_std.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/all_samples/pair_cor_msd.csv')


# In[90]:


sample_violin(pair_cor_df, "max_cor",  "Max Correlations to Paired Samples", (0, 1),
             save = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/all_samples/pair_cor.png')


# In[95]:


ari_df = get_compiled_data('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/*/*_ari_df.csv')
ari_df.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/output_dfs/ari_df.csv')

ari_df_mean_std = mean_sd(ari_df, ['perc', 'sample', 'ARI'], ['perc', 'sample'], 
                            pivot = True, pivot_index='perc', pivot_column='sample')

ari_df_mean_std = combine_tests(ari_df_mean_std, sample_kruskal(ari_df, 'ARI'))
ari_df_mean_std.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/all_samples/ari_msd.csv')
ari_df_mean_std


# In[96]:


sample_violin(ari_df, "ARI",  "ARI to Full Sample", (0, 1),
              save = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/all_samples/ari.png')


# In[97]:


q_df = get_compiled_data('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/*/*_q_df.csv')
q_df.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/output_dfs/q_df.csv')

q_df_mean_std = mean_sd(q_df, ['perc', 'sample', 'Q'], ['perc', 'sample'], 
                        pivot = True, pivot_index='perc', pivot_column='sample')

q_df_mean_std = combine_tests(q_df_mean_std, sample_kruskal(q_df, 'Q'))
q_df_mean_std.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/all_samples/q_msd.csv')


# In[101]:


sample_violin(q_df, "Q",  "Modularity (Q)", (.3, .6),
              save = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/outputs_rest_include/figures/all_samples/q.png')


# In[103]:




